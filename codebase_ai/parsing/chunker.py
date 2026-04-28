"""Code-aware chunking for supported source files."""

from __future__ import annotations

import ast
import logging
import re
from hashlib import sha1

from codebase_ai.config import ChunkingConfig
from codebase_ai.models import CodeChunk, SourceFile

logger = logging.getLogger(__name__)

JS_TS_BLOCK_PATTERN = re.compile(
    r"^\s*(?:export\s+)?(?:(async)\s+)?"
    r"(function|class|const|let|var|interface|type|enum)\s+([A-Za-z_$][\w$]*)",
)


class CodeChunker:
    """Creates meaningful code chunks from source files."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_file(self, source_file: SourceFile) -> list[CodeChunk]:
        """Chunk a source file into logical code units."""

        if source_file.language == "python":
            return self._chunk_python(source_file)
        if source_file.language in {"javascript", "typescript"}:
            return self._chunk_js_ts(source_file)
        return self._fallback_chunk(source_file)

    def chunk_files(self, source_files: list[SourceFile]) -> list[CodeChunk]:
        """Chunk multiple source files."""

        chunks: list[CodeChunk] = []
        for source_file in source_files:
            chunks.extend(self.chunk_file(source_file))
        return chunks

    def _chunk_python(self, source_file: SourceFile) -> list[CodeChunk]:
        try:
            tree = ast.parse(source_file.content)
        except SyntaxError as exc:
            logger.warning(
                "Falling back to line chunking for %s due to syntax error: %s",
                source_file.path,
                exc,
            )
            return self._fallback_chunk(source_file)

        lines = source_file.content.splitlines()
        chunks: list[CodeChunk] = []
        seen_ranges: set[tuple[int, int, str | None]] = set()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
                continue

            symbol_name = getattr(node, "name", None)
            start_line = int(node.lineno)
            end_line = int(node.end_lineno)
            parent_symbol = self._find_python_parent(tree, node)
            chunk_type = self._python_chunk_type(node, parent_symbol)
            range_key = (start_line, end_line, symbol_name)
            if range_key in seen_ranges:
                continue
            seen_ranges.add(range_key)

            text = "\n".join(lines[start_line - 1 : end_line]).strip()
            if not text:
                continue

            chunks.append(
                self._make_chunk(
                    source_file=source_file,
                    chunk_type=chunk_type,
                    symbol_name=symbol_name,
                    start_line=start_line,
                    end_line=end_line,
                    text=text,
                    parent_symbol=parent_symbol,
                    metadata={"ast_type": type(node).__name__},
                )
            )

        if chunks:
            chunks.sort(key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.symbol_name or ""))
            return chunks
        return self._fallback_chunk(source_file)

    def _chunk_js_ts(self, source_file: SourceFile) -> list[CodeChunk]:
        lines = source_file.content.splitlines()
        blocks: list[tuple[int, int, str, str | None]] = []

        line_index = 0
        while line_index < len(lines):
            match = JS_TS_BLOCK_PATTERN.match(lines[line_index])
            if not match:
                line_index += 1
                continue

            declaration_type = match.group(2)
            symbol_name = match.group(3)
            start_line = line_index + 1
            end_line = self._find_js_ts_block_end(lines, line_index)
            blocks.append((start_line, end_line, declaration_type, symbol_name))
            line_index = max(line_index + 1, end_line)

        chunks: list[CodeChunk] = []
        for start_line, end_line, declaration_type, symbol_name in blocks:
            text = "\n".join(lines[start_line - 1 : end_line]).strip()
            if not text:
                continue
            chunks.append(
                self._make_chunk(
                    source_file=source_file,
                    chunk_type=declaration_type,
                    symbol_name=symbol_name,
                    start_line=start_line,
                    end_line=end_line,
                    text=text,
                    metadata={"strategy": "structure_regex"},
                )
            )

        if chunks:
            return chunks
        return self._fallback_chunk(source_file)

    def _fallback_chunk(self, source_file: SourceFile) -> list[CodeChunk]:
        lines = source_file.content.splitlines()
        if not lines:
            return []

        chunk_size = self.config.fallback_chunk_line_count
        overlap = min(self.config.fallback_chunk_overlap, max(chunk_size - 1, 0))
        step = max(chunk_size - overlap, 1)
        chunks: list[CodeChunk] = []

        for start_index in range(0, len(lines), step):
            end_index = min(start_index + chunk_size, len(lines))
            text = "\n".join(lines[start_index:end_index]).strip()
            if not text:
                continue
            chunk_number = len(chunks) + 1
            chunks.append(
                self._make_chunk(
                    source_file=source_file,
                    chunk_type="fallback_block",
                    symbol_name=f"block_{chunk_number}",
                    start_line=start_index + 1,
                    end_line=end_index,
                    text=text,
                    metadata={"strategy": "line_window"},
                )
            )
            if end_index >= len(lines):
                break

        return chunks

    def _find_python_parent(self, tree: ast.AST, target: ast.AST) -> str | None:
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                if child is not target:
                    continue
                if isinstance(parent, ast.ClassDef):
                    return parent.name
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return parent.name
        return None

    def _python_chunk_type(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        parent_symbol: str | None,
    ) -> str:
        if isinstance(node, ast.ClassDef):
            return "class"
        if parent_symbol:
            return "method"
        return "function"

    def _find_js_ts_block_end(self, lines: list[str], start_index: int) -> int:
        brace_balance = 0
        encountered_open_brace = False
        max_end_index = min(
            len(lines),
            start_index + self.config.js_ts_max_chunk_lines,
        )

        for index in range(start_index, max_end_index):
            line = lines[index]
            open_count = line.count("{")
            close_count = line.count("}")
            if open_count:
                encountered_open_brace = True
            brace_balance += open_count
            brace_balance -= close_count

            if encountered_open_brace and brace_balance <= 0:
                return index + 1

            if not encountered_open_brace and line.rstrip().endswith(";"):
                return index + 1

        return max_end_index

    def _make_chunk(
        self,
        source_file: SourceFile,
        chunk_type: str,
        symbol_name: str | None,
        start_line: int,
        end_line: int,
        text: str,
        parent_symbol: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CodeChunk:
        chunk_id = sha1(
            f"{source_file.path}:{chunk_type}:{symbol_name}:{start_line}:{end_line}".encode(
                "utf-8"
            )
        ).hexdigest()
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=source_file.path,
            language=source_file.language,
            chunk_type=chunk_type,
            symbol_name=symbol_name,
            start_line=start_line,
            end_line=end_line,
            text=text,
            parent_symbol=parent_symbol,
            metadata=metadata or {},
        )
