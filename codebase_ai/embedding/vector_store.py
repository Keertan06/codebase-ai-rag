"""Local FAISS-backed vector storage."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
import fnmatch

from codebase_ai.models import CodeChunk, EmbeddedChunk, RetrievalFilters, SearchResult

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """Stores chunk embeddings in a local FAISS index."""

    def __init__(self, storage_dir: str | Path) -> None:
        # Default to the canonical index directory when none provided.
        if storage_dir is None or str(storage_dir).strip() == "":
            self.storage_dir = Path(".codebase_ai/index")
        else:
            self.storage_dir = Path(storage_dir)
        # Ensure the storage directory exists immediately so callers
        # can rely on the path being present.
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_dir / "chunks.faiss"
        self.metadata_path = self.storage_dir / "chunks.json"
        self.embedding_matrix_path = self.storage_dir / "embeddings.npy"

    def save(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Persist embeddings and chunk metadata to disk."""

        if not embedded_chunks:
            raise ValueError("Cannot save an empty vector index.")

        try:
            import faiss  # type: ignore
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu and numpy are required for vector storage. "
                "Install dependencies from requirements.txt."
            ) from exc

        # Log where we're saving to help debugging.
        print("Saving index to:", self.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        dimension = len(embedded_chunks[0].embedding)
        matrix = np.array([item.embedding for item in embedded_chunks], dtype="float32")
        index = faiss.IndexFlatIP(dimension)
        index.add(matrix)
        faiss.write_index(index, str(self.storage_dir / "chunks.faiss"))
        np.save(self.embedding_matrix_path, matrix)

        payload = {
            "dimension": dimension,
            "size": len(embedded_chunks),
            "chunks": [self._serialize_chunk(item.chunk) for item in embedded_chunks],
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved FAISS index with %s chunks to %s", len(embedded_chunks), self.storage_dir)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
        candidate_pool_size: int | None = None,
    ) -> list[SearchResult]:
        """Search the local FAISS index for the most similar chunks."""

        try:
            import faiss  # type: ignore
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu and numpy are required for vector search."
            ) from exc

        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Vector index not found in {self.storage_dir}. Run the index command first."
            )

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        query = np.array([query_embedding], dtype="float32")
        chunks = [self._deserialize_chunk(item) for item in metadata["chunks"]]

        active_filters = filters or RetrievalFilters()
        if self._has_active_filters(active_filters):
            if not self.embedding_matrix_path.exists():
                raise FileNotFoundError(
                    f"Embedding matrix not found in {self.storage_dir}. Rebuild the index."
                )
            matrix = np.load(self.embedding_matrix_path)
            matching_ids = [
                index
                for index, chunk in enumerate(chunks)
                if self._matches_filters(chunk, active_filters)
            ]
            if not matching_ids:
                return []

            filtered_matrix = matrix[matching_ids]
            scores = filtered_matrix @ query[0]
            ranked_positions = scores.argsort()[::-1][:top_k]
            return [
                SearchResult(
                    chunk=chunks[matching_ids[position]],
                    score=float(scores[position]),
                    vector_score=float(scores[position]),
                )
                for position in ranked_positions
            ]

        index = faiss.read_index(str(self.index_path))
        search_top_k = candidate_pool_size or top_k
        scores, ids = index.search(query, search_top_k)

        results: list[SearchResult] = []
        for position, score in zip(ids[0], scores[0]):
            if position < 0 or position >= len(chunks):
                continue
            results.append(
                SearchResult(
                    chunk=chunks[position],
                    score=float(score),
                    vector_score=float(score),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def exists(self) -> bool:
        """Return whether the vector index files exist on disk."""

        return (
            self.index_path.exists()
            and self.metadata_path.exists()
            and self.embedding_matrix_path.exists()
        )

    def _serialize_chunk(self, chunk: CodeChunk) -> dict[str, object]:
        return asdict(chunk)

    def _deserialize_chunk(self, payload: dict[str, object]) -> CodeChunk:
        return CodeChunk(**payload)

    def _has_active_filters(self, filters: RetrievalFilters) -> bool:
        return any(
            (
                filters.languages,
                filters.file_globs,
                filters.chunk_types,
                filters.symbol_names,
            )
        )

    def _matches_filters(self, chunk: CodeChunk, filters: RetrievalFilters) -> bool:
        normalized_path = chunk.file_path.replace("\\", "/")
        symbol_filters = {name.lower() for name in filters.symbol_names}
        chunk_type_filters = {chunk_type.lower() for chunk_type in filters.chunk_types}
        language_filters = {language.lower() for language in filters.languages}

        if language_filters and chunk.language.lower() not in language_filters:
            return False

        if filters.file_globs and not any(
            fnmatch.fnmatch(normalized_path, pattern) for pattern in filters.file_globs
        ):
            return False

        if chunk_type_filters and chunk.chunk_type.lower() not in chunk_type_filters:
            return False

        if symbol_filters:
            symbol_name = (chunk.symbol_name or "").lower()
            if symbol_name not in symbol_filters:
                return False

        return True
