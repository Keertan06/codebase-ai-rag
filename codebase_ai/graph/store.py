"""Persistence and query helpers for the relationship graph."""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

logger = logging.getLogger(__name__)


class GraphStore:
    """Stores and loads the relationship graph from disk."""

    def __init__(self, storage_dir: str | Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.graph_path = self.storage_dir / "graph.json"

    def save(self, graph: nx.MultiDiGraph) -> None:
        """Persist the graph as node-link JSON."""

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        payload = json_graph.node_link_data(graph, edges="links")
        self.graph_path.write_text(
            __import__("json").dumps(payload, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved graph with %s nodes to %s", graph.number_of_nodes(), self.graph_path)

    def load(self) -> nx.MultiDiGraph:
        """Load the relationship graph from disk."""

        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {self.graph_path}")
        payload = __import__("json").loads(self.graph_path.read_text(encoding="utf-8"))
        return json_graph.node_link_graph(
            payload,
            multigraph=True,
            directed=True,
            edges="links",
        )

    def exists(self) -> bool:
        """Return whether the graph exists on disk."""

        return self.graph_path.exists()
