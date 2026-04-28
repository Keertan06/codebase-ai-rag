"""Services for embedding chunks and building a persistent vector index."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from codebase_ai.config import AppConfig, EmbeddingConfig
from codebase_ai.models import CodeChunk, EmbeddedChunk, SourceFile
from codebase_ai.graph import CodeRelationshipGraphBuilder, GraphStore

from .providers import create_embedding_provider
from .vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


class ChunkEmbeddingIndexer:
    """Embeds code chunks and saves them to a local vector index."""

    def __init__(
        self,
        embedding_config: EmbeddingConfig | None = None,
        storage_dir: str | Path | None = None,
    ) -> None:
        self.app_config = AppConfig()
        self.embedding_config = embedding_config or self.app_config.embedding
        # Use the canonical index directory for storing vector and graph data.
        self.index_dir = self.app_config.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.provider = create_embedding_provider(self.embedding_config)
        self.vector_store = FaissVectorStore(self.index_dir)
        self.graph_store = GraphStore(self.index_dir)
        self.graph_builder = CodeRelationshipGraphBuilder()

    def build_index(self, chunks: list[CodeChunk]) -> Path:
        """Embed chunks and persist the resulting vector index."""

        if not chunks:
            raise ValueError("No chunks were provided for indexing.")

        logger.info("Embedding %s chunks", len(chunks))
        texts = [self._chunk_to_embedding_text(chunk) for chunk in chunks]
        embeddings = self.provider.embed_texts(texts)
        embedded_chunks = [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.vector_store.save(embedded_chunks)
        self._write_manifest(chunk_count=len(chunks))
        return self.index_dir

    def build_graph_index(
        self,
        source_files: list[SourceFile],
        chunks: list[CodeChunk],
    ) -> Path:
        """Build and persist the code relationship graph."""

        graph = self.graph_builder.build(source_files=source_files, chunks=chunks)
        self.graph_store.save(graph)
        return self.index_dir

    def _chunk_to_embedding_text(self, chunk: CodeChunk) -> str:
        symbol = chunk.symbol_name or "<anonymous>"
        parent = f" parent={chunk.parent_symbol}" if chunk.parent_symbol else ""
        return (
            f"path={chunk.file_path}\n"
            f"language={chunk.language}\n"
            f"type={chunk.chunk_type}\n"
            f"symbol={symbol}{parent}\n"
            f"lines={chunk.start_line}-{chunk.end_line}\n"
            f"code:\n{chunk.text}"
        )

    def _write_manifest(self, chunk_count: int) -> None:
        selected_model = (
            self.embedding_config.openai_model
            if self.embedding_config.provider == "openai"
            else self.embedding_config.model_name
        )
        manifest = {
            "provider": self.embedding_config.provider,
            "model_name": self.embedding_config.model_name,
            "openai_model": self.embedding_config.openai_model,
            "selected_model": selected_model,
            "chunk_count": chunk_count,
            "index_directory": str(self.index_dir),
        }
        manifest_path = self.index_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
