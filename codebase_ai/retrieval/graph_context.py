"""Graph-based retrieval expansion for semantically matched chunks."""

from __future__ import annotations

import logging
from pathlib import Path

from codebase_ai.config import RetrievalConfig
from codebase_ai.graph import GraphStore
from codebase_ai.models import CodeChunk, SearchResult

logger = logging.getLogger(__name__)

RELATIONSHIP_PRIORITY = {
    "calls": 0,
    "imports": 1,
    "api_call": 2,
    "uses_class": 3,
    "uses_attribute": 4,
    "contains": 5,
}


class GraphContextExpander:
    """Expands retrieval results with neighboring graph context."""

    def __init__(
        self,
        storage_dir: str | Path,
        retrieval_config: RetrievalConfig,
    ) -> None:
        self.graph_store = GraphStore(storage_dir)
        self.retrieval_config = retrieval_config

    def expand(self, results: list[SearchResult]) -> list[SearchResult]:
        """Attach graph neighbors and boost graph-connected results."""

        if not results or not self.graph_store.exists():
            return results

        graph = self.graph_store.load()
        expanded: list[SearchResult] = []
        boost_budget = {
            result.chunk.chunk_id
            for result in results[: self.retrieval_config.graph_expand_results]
        }

        for result in results:
            chunk_node = self._chunk_node_id(result.chunk)
            neighbor_descriptions: list[str] = []
            score = result.score

            if graph.has_node(chunk_node):
                neighbors = list(graph.out_edges(chunk_node, data=True)) + list(graph.in_edges(chunk_node, data=True))
                neighbors.sort(
                    key=lambda item: RELATIONSHIP_PRIORITY.get(
                        str(item[2].get("relationship", "related_to")),
                        99,
                    )
                )
                seen_descriptions: set[str] = set()
                for source, target, edge_data in neighbors:
                    neighbor_node = target if source == chunk_node else source
                    relationship = str(edge_data.get("relationship", "related_to"))
                    description = self._format_neighbor(graph.nodes[neighbor_node], relationship)
                    if description in seen_descriptions:
                        continue
                    seen_descriptions.add(description)
                    neighbor_descriptions.append(description)
                    if len(neighbor_descriptions) >= self.retrieval_config.graph_neighbor_limit:
                        break

                if result.chunk.chunk_id in boost_budget and neighbor_descriptions:
                    score += min(
                        len(neighbor_descriptions) * self.retrieval_config.graph_neighbor_boost,
                        self.retrieval_config.graph_neighbor_boost
                        * self.retrieval_config.graph_neighbor_limit,
                    )

            matched_terms = list(result.matched_terms)
            if neighbor_descriptions:
                matched_terms.append("graph")

            expanded.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    vector_score=result.vector_score,
                    rerank_score=score,
                    matched_terms=tuple(matched_terms),
                    graph_neighbors=tuple(neighbor_descriptions),
                )
            )

        expanded.sort(key=lambda item: item.score, reverse=True)
        return expanded

    def _chunk_node_id(self, chunk: CodeChunk) -> str:
        return f"chunk::{chunk.chunk_id}"

    def _format_neighbor(self, node_data: dict, relationship: str) -> str:
        node_type = node_data.get("node_type", "node")
        if node_type == "file":
            return f"{relationship} -> file:{node_data.get('file_path')}"
        if node_type == "chunk":
            symbol = node_data.get("symbol_name") or "<anonymous>"
            return f"{relationship} -> chunk:{node_data.get('file_path')}::{symbol}"
        if node_type == "external":
            return f"{relationship} -> external:{node_data.get('name', 'unknown')}"
        if node_type == "symbol":
            return f"{relationship} -> symbol:{node_data.get('symbol_name', 'unknown')}"
        return f"{relationship} -> {node_type}"
