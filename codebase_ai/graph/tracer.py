"""Graph-based flow tracing over the indexed code relationship graph."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import re
from pathlib import Path

import networkx as nx

from codebase_ai.config import AppConfig, TraceConfig

from .store import GraphStore

logger = logging.getLogger(__name__)

QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*")
EDGE_PRIORITY = ("calls", "uses_class", "imports")
NODE_TYPE_PRIORITY = {
    "chunk": 3,
    "symbol": 2,
    "file": 1,
    "external": 0,
}
CHUNK_TYPE_PRIORITY = {
    "function": 3,
    "method": 3,
    "class": 2,
    "const": 1,
    "fallback_block": 0,
}


@dataclass(slots=True)
class TraceStep:
    """A single step in a traced execution flow."""

    symbol_name: str
    file_path: str
    chunk_type: str
    relationship_from_previous: str | None
    depth: int
    node_id: str


@dataclass(slots=True)
class TraceRelationship:
    """A relationship between two adjacent trace steps."""

    source_symbol: str
    source_file_path: str
    relationship: str
    target_symbol: str
    target_file_path: str


@dataclass(slots=True)
class TraceResult:
    """The final result of a graph-based trace."""

    query: str
    entry_step: TraceStep | None
    steps: list[TraceStep]
    relationships: list[TraceRelationship]
    summary: str
    error: str | None = None


@dataclass(slots=True)
class _MatchedNode:
    """Internal entry-point matching candidate."""

    node_id: str
    node_type: str
    score: int
    file_path: str
    symbol_name: str | None
    chunk_type: str | None


@dataclass(slots=True)
class _TraversalCandidate:
    """Internal traversal candidate."""

    target_chunk_id: str
    relationship: str
    priority: int
    resolution_score: int


class FlowTracer:
    """Traces function-level flow using BFS over the relationship graph."""

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        config: TraceConfig | None = None,
    ) -> None:
        app_config = AppConfig()
        # Use the canonical index directory for graph storage to keep
        # behavior consistent across commands.
        self.index_dir = app_config.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.graph_store = GraphStore(self.index_dir)
        self.config = config or app_config.trace

    def trace(self, query: str) -> TraceResult:
        """Trace the most likely graph flow for a user query."""

        if not self.graph_store.exists():
            error_message = (
                f"Graph index not found in {self.index_dir}. "
                "Run the index command with --build-vector-index first."
            )
            return TraceResult(
                query=query,
                entry_step=None,
                steps=[],
                relationships=[],
                summary=error_message,
                error=error_message,
            )

        graph = self.graph_store.load()
        entry_chunk_id = self._detect_entry_point(graph, query)
        if not entry_chunk_id:
            error_message = f"No matching entry point found for query: {query}"
            return TraceResult(
                query=query,
                entry_step=None,
                steps=[],
                relationships=[],
                summary=error_message,
                error=error_message,
            )

        steps, relationships = self._traverse(graph, entry_chunk_id)
        summary = self._build_summary(steps, relationships)
        entry_step = steps[0] if steps else None
        return TraceResult(
            query=query,
            entry_step=entry_step,
            steps=steps,
            relationships=relationships,
            summary=summary,
        )

    def _detect_entry_point(self, graph: nx.MultiDiGraph, query: str) -> str | None:
        query_normalized = self._normalize(query)
        query_tokens = self._query_tokens(query)
        candidates: list[_MatchedNode] = []

        for node_id, attrs in graph.nodes(data=True):
            node_type = str(attrs.get("node_type", ""))
            file_path = str(attrs.get("file_path", ""))
            symbol_name = attrs.get("symbol_name")
            chunk_type = attrs.get("chunk_type")
            score = self._score_node_match(
                query_normalized=query_normalized,
                query_tokens=query_tokens,
                node_type=node_type,
                file_path=file_path,
                symbol_name=str(symbol_name) if symbol_name is not None else None,
                external_name=str(attrs.get("name", "")),
            )
            if score <= 0:
                continue

            candidates.append(
                _MatchedNode(
                    node_id=node_id,
                    node_type=node_type,
                    score=score,
                    file_path=file_path,
                    symbol_name=str(symbol_name) if symbol_name is not None else None,
                    chunk_type=str(chunk_type) if chunk_type is not None else None,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda candidate: (
                candidate.score,
                NODE_TYPE_PRIORITY.get(candidate.node_type, -1),
                CHUNK_TYPE_PRIORITY.get(candidate.chunk_type or "", -1),
                -len(candidate.file_path),
            ),
            reverse=True,
        )
        best_match = candidates[0]

        if best_match.node_type == "chunk":
            return best_match.node_id
        if best_match.node_type == "symbol":
            resolved = self._resolve_symbol_to_chunks(
                graph=graph,
                symbol_name=best_match.symbol_name or "",
                current_file_path=best_match.file_path,
            )
            return resolved[0][0] if resolved else None
        if best_match.node_type == "file":
            return self._entry_chunk_for_file(graph, best_match.file_path)
        return None

    def _traverse(
        self,
        graph: nx.MultiDiGraph,
        entry_chunk_id: str,
    ) -> tuple[list[TraceStep], list[TraceRelationship]]:
        steps: list[TraceStep] = [self._make_step(graph, entry_chunk_id, None, 0)]
        relationships: list[TraceRelationship] = []
        visited_chunks = {entry_chunk_id}
        queue: deque[tuple[str, int]] = deque([(entry_chunk_id, 0)])

        while queue:
            current_chunk_id, depth = queue.popleft()
            if depth >= self.config.max_trace_depth:
                continue

            next_candidates = self._expand_chunk_neighbors(graph, current_chunk_id)
            for candidate in next_candidates[: self.config.max_trace_branching]:
                if candidate.target_chunk_id in visited_chunks:
                    continue

                visited_chunks.add(candidate.target_chunk_id)
                next_step = self._make_step(
                    graph=graph,
                    chunk_node_id=candidate.target_chunk_id,
                    relationship=candidate.relationship,
                    depth=depth + 1,
                )
                steps.append(next_step)
                relationships.append(
                    self._make_relationship(
                        graph=graph,
                        source_chunk_id=current_chunk_id,
                        target_chunk_id=candidate.target_chunk_id,
                        relationship=candidate.relationship,
                    )
                )
                queue.append((candidate.target_chunk_id, depth + 1))

        return steps, relationships

    def _expand_chunk_neighbors(
        self,
        graph: nx.MultiDiGraph,
        chunk_node_id: str,
    ) -> list[_TraversalCandidate]:
        chunk_attrs = graph.nodes[chunk_node_id]
        current_file_path = str(chunk_attrs.get("file_path", ""))
        candidates_by_target: dict[str, _TraversalCandidate] = {}
        outgoing_edges = list(graph.out_edges(chunk_node_id, data=True))

        for priority, relationship_name in enumerate(EDGE_PRIORITY):
            if relationship_name in {"calls", "uses_class"}:
                for _, target_node_id, edge_data in outgoing_edges:
                    if edge_data.get("relationship") != relationship_name:
                        continue

                    target_attrs = graph.nodes[target_node_id]
                    target_type = str(target_attrs.get("node_type", ""))
                    if target_type == "chunk":
                        candidate = _TraversalCandidate(
                            target_chunk_id=target_node_id,
                            relationship=relationship_name,
                            priority=priority,
                            resolution_score=300,
                        )
                        self._store_best_candidate(candidates_by_target, candidate)
                        continue

                    if target_type != "symbol":
                        continue

                    symbol_name = str(target_attrs.get("symbol_name", ""))
                    resolved = self._resolve_symbol_to_chunks(
                        graph=graph,
                        symbol_name=symbol_name,
                        current_file_path=current_file_path,
                    )
                    for target_chunk_id, resolution_score in resolved[: self.config.max_trace_branching]:
                        candidate = _TraversalCandidate(
                            target_chunk_id=target_chunk_id,
                            relationship=relationship_name,
                            priority=priority,
                            resolution_score=resolution_score,
                        )
                        self._store_best_candidate(candidates_by_target, candidate)

            if relationship_name == "imports":
                file_node_id = self._file_node_id(current_file_path)
                if not graph.has_node(file_node_id):
                    continue
                for _, target_node_id, edge_data in graph.out_edges(file_node_id, data=True):
                    if edge_data.get("relationship") != "imports":
                        continue
                    target_attrs = graph.nodes[target_node_id]
                    if target_attrs.get("node_type") != "file":
                        continue
                    target_file_path = str(target_attrs.get("file_path", ""))
                    target_chunk_id = self._entry_chunk_for_file(graph, target_file_path)
                    if not target_chunk_id:
                        continue
                    candidate = _TraversalCandidate(
                        target_chunk_id=target_chunk_id,
                        relationship="imports",
                        priority=priority,
                        resolution_score=200,
                    )
                    self._store_best_candidate(candidates_by_target, candidate)

        candidates = list(candidates_by_target.values())
        candidates.sort(key=lambda item: (item.priority, -item.resolution_score))
        return candidates

    def _resolve_symbol_to_chunks(
        self,
        graph: nx.MultiDiGraph,
        symbol_name: str,
        current_file_path: str | None,
    ) -> list[tuple[str, int]]:
        target_name = self._symbol_leaf(symbol_name)
        if not target_name:
            return []

        matches: list[tuple[str, int]] = []
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") != "chunk":
                continue
            chunk_symbol = self._symbol_leaf(str(attrs.get("symbol_name", "")))
            if not chunk_symbol:
                continue

            score = 0
            if chunk_symbol == target_name:
                score = 300
            elif self._normalize(str(attrs.get("symbol_name", ""))) == self._normalize(symbol_name):
                score = 280
            elif target_name in chunk_symbol or chunk_symbol in target_name:
                score = 180
            if score <= 0:
                continue

            chunk_file_path = str(attrs.get("file_path", ""))
            if current_file_path and chunk_file_path == current_file_path:
                score += 40
            score += CHUNK_TYPE_PRIORITY.get(str(attrs.get("chunk_type", "")), 0) * 5
            matches.append((node_id, score))

        matches.sort(key=lambda item: item[1], reverse=True)
        return matches

    def _entry_chunk_for_file(
        self,
        graph: nx.MultiDiGraph,
        file_path: str,
    ) -> str | None:
        file_chunks: list[tuple[str, int, int]] = []
        file_stem = self._normalize(Path(file_path).stem)

        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") != "chunk":
                continue
            if str(attrs.get("file_path", "")) != file_path:
                continue

            raw_symbol_name = str(attrs.get("symbol_name", ""))
            symbol_name = self._normalize(raw_symbol_name)
            score = CHUNK_TYPE_PRIORITY.get(str(attrs.get("chunk_type", "")), 0) * 10
            if symbol_name and symbol_name == file_stem:
                score += 30
            if symbol_name and file_stem and file_stem in symbol_name:
                score += 25
            if symbol_name == "__init__":
                score -= 15
            if raw_symbol_name.startswith("block_"):
                score -= 25
            start_line = int(attrs.get("start_line", 0))
            file_chunks.append((node_id, score, start_line))

        if not file_chunks:
            return None

        file_chunks.sort(key=lambda item: (-item[1], item[2]))
        return file_chunks[0][0]

    def _build_summary(
        self,
        steps: list[TraceStep],
        relationships: list[TraceRelationship],
    ) -> str:
        if not steps:
            return "No flow could be traced from the current graph."

        start = steps[0]
        start_name = self._display_symbol(start.symbol_name, start.chunk_type)
        if not relationships:
            return (
                f"The flow starts at {start_name} in {start.file_path}, "
                "but no downstream calls, class usages, or imports were found within the trace limits."
            )

        clauses = [f"The flow starts at {start_name} in {start.file_path}"]
        for relationship in relationships[:3]:
            target_name = self._display_symbol(relationship.target_symbol, self._guess_chunk_type(steps, relationship.target_symbol, relationship.target_file_path))
            clauses.append(
                f"{self._relationship_verb(relationship.relationship)} {target_name} in {relationship.target_file_path}"
            )

        return ", then ".join(clauses) + "."

    def _relationship_verb(self, relationship: str) -> str:
        return {
            "calls": "calls",
            "uses_class": "uses",
            "imports": "imports",
        }.get(relationship, relationship)

    def _guess_chunk_type(
        self,
        steps: list[TraceStep],
        symbol_name: str,
        file_path: str,
    ) -> str:
        for step in steps:
            if step.symbol_name == symbol_name and step.file_path == file_path:
                return step.chunk_type
        return "function"

    def _make_step(
        self,
        graph: nx.MultiDiGraph,
        chunk_node_id: str,
        relationship: str | None,
        depth: int,
    ) -> TraceStep:
        attrs = graph.nodes[chunk_node_id]
        return TraceStep(
            symbol_name=str(attrs.get("symbol_name") or "<anonymous>"),
            file_path=str(attrs.get("file_path", "<unknown>")),
            chunk_type=str(attrs.get("chunk_type", "function")),
            relationship_from_previous=relationship,
            depth=depth,
            node_id=chunk_node_id,
        )

    def _make_relationship(
        self,
        graph: nx.MultiDiGraph,
        source_chunk_id: str,
        target_chunk_id: str,
        relationship: str,
    ) -> TraceRelationship:
        source = graph.nodes[source_chunk_id]
        target = graph.nodes[target_chunk_id]
        return TraceRelationship(
            source_symbol=str(source.get("symbol_name") or "<anonymous>"),
            source_file_path=str(source.get("file_path", "<unknown>")),
            relationship=relationship,
            target_symbol=str(target.get("symbol_name") or "<anonymous>"),
            target_file_path=str(target.get("file_path", "<unknown>")),
        )

    def _store_best_candidate(
        self,
        candidates_by_target: dict[str, _TraversalCandidate],
        candidate: _TraversalCandidate,
    ) -> None:
        existing = candidates_by_target.get(candidate.target_chunk_id)
        if existing is None or (candidate.priority, -candidate.resolution_score) < (
            existing.priority,
            -existing.resolution_score,
        ):
            candidates_by_target[candidate.target_chunk_id] = candidate

    def _score_node_match(
        self,
        query_normalized: str,
        query_tokens: set[str],
        node_type: str,
        file_path: str,
        symbol_name: str | None,
        external_name: str,
    ) -> int:
        labels: list[str] = []
        if symbol_name:
            labels.append(symbol_name)
            labels.append(self._symbol_leaf(symbol_name))
        if file_path and node_type == "file":
            labels.append(file_path)
            labels.append(Path(file_path).stem)
        if external_name:
            labels.append(external_name)

        best_score = 0
        for label in labels:
            score = self._score_label(query_normalized, query_tokens, label)
            best_score = max(best_score, score)

        if node_type == "file" and file_path:
            path_segments = {
                segment
                for segment in re.split(r"[/._-]+", self._normalize(file_path))
                if segment
            }
            if query_normalized in path_segments or any(token in path_segments for token in query_tokens):
                best_score = max(best_score, 260)

        if best_score <= 0:
            return 0

        best_score += NODE_TYPE_PRIORITY.get(node_type, 0) * 5
        if node_type == "chunk":
            best_score += 10
            if symbol_name == "__init__" or (symbol_name and symbol_name.startswith("block_")):
                best_score -= 15
        return best_score

    def _score_label(self, query_normalized: str, query_tokens: set[str], label: str) -> int:
        normalized_label = self._normalize(label)
        label_leaf = self._symbol_leaf(normalized_label)
        if not normalized_label:
            return 0

        if normalized_label == query_normalized or label_leaf == query_normalized:
            return 300
        if any(token == normalized_label or token == label_leaf for token in query_tokens):
            return 220
        if query_normalized in normalized_label or query_normalized in label_leaf:
            return 150
        if any(token and (token in normalized_label or token in label_leaf) for token in query_tokens):
            return 100
        return 0

    def _query_tokens(self, query: str) -> set[str]:
        return {self._normalize(token) for token in QUERY_TOKEN_PATTERN.findall(query)}

    def _normalize(self, value: str) -> str:
        return value.strip().lower()

    def _symbol_leaf(self, symbol_name: str) -> str:
        normalized = self._normalize(symbol_name)
        if not normalized:
            return ""
        return normalized.split(".")[-1]

    def _file_node_id(self, file_path: str) -> str:
        return f"file::{file_path}"

    def _display_symbol(self, symbol_name: str, chunk_type: str) -> str:
        if chunk_type in {"function", "method"}:
            return f"{symbol_name}()"
        return symbol_name
