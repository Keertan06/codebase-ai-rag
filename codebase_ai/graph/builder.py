"""Relationship graph extraction for code chunks and source files."""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

import networkx as nx

from codebase_ai.models import CodeChunk, SourceFile

logger = logging.getLogger(__name__)

JS_TS_IMPORT_PATTERN = re.compile(
    r"^\s*import\s+.+?\s+from\s+[\"'](.+?)[\"']|^\s*const\s+.+?=\s+require\([\"'](.+?)[\"']\)"
)
JS_TS_CALL_PATTERN = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")
HTTP_CALL_PATTERN = re.compile(r"\b(requests|fetch|axios|httpx)\.(get|post|put|delete|patch)\b")


class CodeRelationshipGraphBuilder:
    """Builds a code relationship graph from files and chunks."""

    def build(self, source_files: list[SourceFile], chunks: list[CodeChunk]) -> nx.MultiDiGraph:
        """Build a directed multigraph of code relationships."""

        graph = nx.MultiDiGraph()
        chunks_by_file = self._group_chunks_by_file(chunks)
        module_index = self._build_module_index(source_files)

        for source_file in source_files:
            file_node = self._file_node_id(source_file.path)
            graph.add_node(
                file_node,
                node_type="file",
                file_path=source_file.path,
                language=source_file.language,
            )

        for chunk in chunks:
            chunk_node = self._chunk_node_id(chunk)
            graph.add_node(
                chunk_node,
                node_type="chunk",
                file_path=chunk.file_path,
                symbol_name=chunk.symbol_name,
                chunk_type=chunk.chunk_type,
                language=chunk.language,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
            graph.add_edge(
                self._file_node_id(chunk.file_path),
                chunk_node,
                relationship="contains",
            )

        for source_file in source_files:
            if source_file.language == "python":
                self._extract_python_relationships(
                    graph=graph,
                    source_file=source_file,
                    chunks_by_file=chunks_by_file,
                    module_index=module_index,
                )
            elif source_file.language in {"javascript", "typescript"}:
                self._extract_js_ts_relationships(
                    graph=graph,
                    source_file=source_file,
                    chunks_by_file=chunks_by_file,
                )

        logger.info(
            "Built relationship graph with %s nodes and %s edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    def _extract_python_relationships(
        self,
        graph: nx.MultiDiGraph,
        source_file: SourceFile,
        chunks_by_file: dict[str, list[CodeChunk]],
        module_index: dict[str, str],
    ) -> None:
        try:
            tree = ast.parse(source_file.content)
        except SyntaxError as exc:
            logger.warning("Skipping Python graph extraction for %s: %s", source_file.path, exc)
            return

        chunk_lookup = self._build_python_chunk_lookup(tree, chunks_by_file.get(source_file.path, []))
        current_scope: list[str] = []

        class RelationshipVisitor(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    target_path = module_index.get(alias.name)
                    if target_path:
                        graph.add_edge(
                            self_outer._file_node_id(source_file.path),
                            self_outer._add_file_node(graph, target_path, "python"),
                            relationship="imports",
                            symbol=alias.name,
                        )
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module:
                    target_path = module_index.get(node.module)
                    if target_path:
                        graph.add_edge(
                            self_outer._file_node_id(source_file.path),
                            self_outer._add_file_node(graph, target_path, "python"),
                            relationship="imports",
                            symbol=node.module,
                        )
                self.generic_visit(node)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                chunk_node = chunk_lookup.get(id(node))
                if chunk_node:
                    for base in node.bases:
                        base_name = self_outer._python_name(base)
                        if base_name:
                            graph.add_edge(
                                chunk_node,
                                self_outer._add_symbol_node(graph, source_file.path, base_name),
                                relationship="uses_class",
                            )
                current_scope.append(node.name)
                self.generic_visit(node)
                current_scope.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_callable(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_callable(node)

            def _visit_callable(
                self,
                node: ast.FunctionDef | ast.AsyncFunctionDef,
            ) -> None:
                chunk_node = chunk_lookup.get(id(node))
                current_scope.append(node.name)
                if chunk_node:
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            call_name = self_outer._python_name(child.func)
                            if call_name:
                                graph.add_edge(
                                    chunk_node,
                                    self_outer._add_symbol_node(graph, source_file.path, call_name),
                                    relationship="calls",
                                )
                            api_match = self_outer._python_api_call(child)
                            if api_match:
                                graph.add_edge(
                                    chunk_node,
                                    self_outer._add_external_node(graph, api_match),
                                    relationship="api_call",
                                )
                        elif isinstance(child, ast.Attribute):
                            attribute_name = child.attr
                            graph.add_edge(
                                chunk_node,
                                self_outer._add_symbol_node(graph, source_file.path, attribute_name),
                                relationship="uses_attribute",
                            )
                self.generic_visit(node)
                current_scope.pop()

        self_outer = self
        RelationshipVisitor().visit(tree)

    def _extract_js_ts_relationships(
        self,
        graph: nx.MultiDiGraph,
        source_file: SourceFile,
        chunks_by_file: dict[str, list[CodeChunk]],
    ) -> None:
        file_node = self._file_node_id(source_file.path)
        for line in source_file.content.splitlines():
            import_match = JS_TS_IMPORT_PATTERN.search(line)
            if import_match:
                target = import_match.group(1) or import_match.group(2)
                graph.add_edge(file_node, self._add_external_node(graph, target), relationship="imports")

        for chunk in chunks_by_file.get(source_file.path, []):
            chunk_node = self._chunk_node_id(chunk)
            for called in JS_TS_CALL_PATTERN.findall(chunk.text):
                if called in {"if", "for", "while", "switch", "catch"}:
                    continue
                graph.add_edge(
                    chunk_node,
                    self._add_symbol_node(graph, source_file.path, called),
                    relationship="calls",
                )
            for api_client, method in HTTP_CALL_PATTERN.findall(chunk.text):
                graph.add_edge(
                    chunk_node,
                    self._add_external_node(graph, f"{api_client}.{method}"),
                    relationship="api_call",
                )

    def _build_module_index(self, source_files: list[SourceFile]) -> dict[str, str]:
        module_index: dict[str, str] = {}
        for source_file in source_files:
            if source_file.language != "python":
                continue
            module_name = source_file.path.replace("/", ".")
            if module_name.endswith(".py"):
                module_name = module_name[:-3]
            if module_name.endswith(".__init__"):
                module_name = module_name[: -len(".__init__")]
            module_index[module_name] = source_file.path
        return module_index

    def _group_chunks_by_file(self, chunks: list[CodeChunk]) -> dict[str, list[CodeChunk]]:
        grouped: dict[str, list[CodeChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.file_path, []).append(chunk)
        return grouped

    def _build_python_chunk_lookup(
        self,
        tree: ast.AST,
        chunks: list[CodeChunk],
    ) -> dict[int, str]:
        chunk_lookup: dict[int, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for chunk in chunks:
                if (
                    chunk.symbol_name == getattr(node, "name", None)
                    and chunk.start_line == getattr(node, "lineno", None)
                    and chunk.end_line == getattr(node, "end_lineno", None)
                ):
                    chunk_lookup[id(node)] = self._chunk_node_id(chunk)
                    break
        return chunk_lookup

    def _python_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._python_name(node.value)
            if base:
                return f"{base}.{node.attr}"
            return node.attr
        return None

    def _python_api_call(self, node: ast.Call) -> str | None:
        func_name = self._python_name(node.func)
        if not func_name:
            return None
        if func_name.startswith(("requests.", "httpx.")):
            return func_name
        return None

    def _file_node_id(self, file_path: str) -> str:
        return f"file::{file_path}"

    def _chunk_node_id(self, chunk: CodeChunk) -> str:
        return f"chunk::{chunk.chunk_id}"

    def _symbol_node_id(self, file_path: str, symbol_name: str) -> str:
        normalized = symbol_name.replace(" ", "")
        return f"symbol::{file_path}::{normalized}"

    def _external_node_id(self, name: str) -> str:
        return f"external::{name}"

    def _add_file_node(self, graph: nx.MultiDiGraph, file_path: str, language: str) -> str:
        node_id = self._file_node_id(file_path)
        graph.add_node(node_id, node_type="file", file_path=file_path, language=language)
        return node_id

    def _add_symbol_node(
        self,
        graph: nx.MultiDiGraph,
        file_path: str,
        symbol_name: str,
    ) -> str:
        node_id = self._symbol_node_id(file_path, symbol_name)
        graph.add_node(
            node_id,
            node_type="symbol",
            file_path=file_path,
            symbol_name=symbol_name,
        )
        return node_id

    def _add_external_node(self, graph: nx.MultiDiGraph, name: str) -> str:
        node_id = self._external_node_id(name)
        graph.add_node(node_id, node_type="external", name=name)
        return node_id
