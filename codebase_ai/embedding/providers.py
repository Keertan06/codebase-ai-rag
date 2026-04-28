"""Embedding provider implementations."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from codebase_ai.config import EmbeddingConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_PROVIDER = "sentence-transformers"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the embedding size."""


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by sentence-transformers."""

    _model_cache: dict[str, object] = {}

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install dependencies from requirements.txt. Requires: pip install sentence-transformers"
            ) from exc

        self.config = config
        model_name = config.model_name or DEFAULT_SENTENCE_TRANSFORMER_MODEL
        if model_name not in self._model_cache:
            logger.info("Loading sentence-transformers model: %s", model_name)
            self._model_cache[model_name] = SentenceTransformer(model_name)
        self.model = self._model_cache[model_name]
        self._dimension: int | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if self._dimension is None:
            self._dimension = int(vectors.shape[1])
        return vectors.tolist()

    def embedding_dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.model.get_sentence_embedding_dimension())
        return self._dimension


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the OpenAI API."""

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed. Install dependencies from requirements.txt."
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or use a local embedding provider."
            )

        self.config = config
        self.client = OpenAI(api_key=api_key)
        self._dimension: int | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.config.openai_model,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        if embeddings and self._dimension is None:
            self._dimension = len(embeddings[0])
        return embeddings

    def embedding_dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError(
                "Embedding dimension is unknown until at least one OpenAI embedding call succeeds."
            )
        return self._dimension


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Create an embedding provider from configuration."""

    provider_name = _normalize_embedding_provider(config.provider)
    logger.info("Using embedding provider: %s", provider_name)

    if provider_name == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(config)
    if provider_name == "openai":
        return OpenAIEmbeddingProvider(config)

    return SentenceTransformerEmbeddingProvider(config)


def _normalize_embedding_provider(provider: str | None) -> str:
    normalized = (provider or "").strip().lower()
    if not normalized:
        return DEFAULT_EMBEDDING_PROVIDER
    if normalized not in {"sentence-transformers", "openai"}:
        logger.warning(
            "Unsupported embedding provider '%s'; falling back to sentence-transformers.",
            provider,
        )
        return DEFAULT_EMBEDDING_PROVIDER
    return normalized
