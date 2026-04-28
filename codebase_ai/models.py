"""Shared data models used across the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceFile:
    """Represents a code file discovered during repository scanning."""

    path: str
    language: str
    content: str
    size_bytes: int
    line_count: int


@dataclass(slots=True)
class ScanFilters:
    """Optional filters to constrain repository scanning."""

    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    max_files: int | None = None


@dataclass(slots=True)
class CodeChunk:
    """Represents a meaningful chunk of code with metadata."""

    chunk_id: str
    file_path: str
    language: str
    chunk_type: str
    symbol_name: str | None
    start_line: int
    end_line: int
    text: str
    parent_symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddedChunk:
    """Represents a chunk paired with its embedding vector."""

    chunk: CodeChunk
    embedding: list[float]


@dataclass(slots=True)
class SearchResult:
    """Represents a vector-search match."""

    chunk: CodeChunk
    score: float
    vector_score: float | None = None
    rerank_score: float | None = None
    matched_terms: tuple[str, ...] = ()
    graph_neighbors: tuple[str, ...] = ()


@dataclass(slots=True)
class RetrievalFilters:
    """Metadata filters for retrieval."""

    languages: tuple[str, ...] = ()
    file_globs: tuple[str, ...] = ()
    chunk_types: tuple[str, ...] = ()
    symbol_names: tuple[str, ...] = ()


@dataclass(slots=True)
class LLMAnswer:
    """Structured answer produced by the LLM layer."""

    text: str
    provider: str
    model: str
    prompt: str
