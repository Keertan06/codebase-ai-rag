"""Embedding module for vectorization and local vector storage."""

from .indexer import ChunkEmbeddingIndexer
from .providers import create_embedding_provider
from .vector_store import FaissVectorStore

__all__ = ["ChunkEmbeddingIndexer", "FaissVectorStore", "create_embedding_provider"]
