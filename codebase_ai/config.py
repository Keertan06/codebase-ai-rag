"""Application configuration for Codebase Knowledge AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ScannerConfig:
    """Configuration for repository scanning."""

    supported_extensions: dict[str, str] = field(
        default_factory=lambda: {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
        }
    )
    ignored_directories: set[str] = field(
        default_factory=lambda: {
            ".git",
            ".hg",
            ".svn",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "build",
            "coverage",
            "dist",
            "node_modules",
            "venv",
        }
    )
    ignored_file_names: set[str] = field(
        default_factory=lambda: {
            ".DS_Store",
        }
    )
    ignored_path_patterns: tuple[str, ...] = (
        "*.min.js",
        "*.min.css",
        "*.bundle.js",
        "*.d.ts",
    )
    max_file_size_bytes: int = 1_000_000
    default_encoding: str = "utf-8"
    follow_symlinks: bool = False
    max_files: int | None = None


@dataclass(slots=True)
class ChunkingConfig:
    """Configuration for code chunking."""

    fallback_chunk_line_count: int = 80
    fallback_chunk_overlap: int = 10
    js_ts_max_chunk_lines: int = 120


@dataclass(slots=True)
class EmbeddingConfig:
    """Configuration for embedding generation."""

    provider: str = "sentence-transformers"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_model: str = "text-embedding-3-small"
    batch_size: int = 32
    normalize_embeddings: bool = True
    index_dir_name: str = "index"


@dataclass(slots=True)
class RetrievalConfig:
    """Configuration for retrieval and reranking."""

    default_top_k: int = 3
    # Allow up to 4 chunks maximum, but default to 3 for small-model usage
    max_returned_chunks: int = 4
    vector_candidate_multiplier: int = 10
    graph_expand_results: int = 3
    graph_neighbor_limit: int = 4
    semantic_similarity_weight: float = 1.8
    path_match_boost: float = 0.45
    symbol_match_boost: float = 0.3
    chunk_type_match_boost: float = 0.15
    language_match_boost: float = 0.1
    content_term_boost: float = 0.08
    max_content_term_boost: float = 0.35
    graph_neighbor_boost: float = 0.04
    fallback_chunk_penalty: float = 0.3
    min_direct_signal_score: float = 0.18
    # Prioritization for routing-related queries
    route_match_boost: float = 0.6
    # Large-class handling: classes longer than this many lines are deprioritized
    large_class_line_threshold: int = 200
    large_class_penalty: float = 0.7


@dataclass(slots=True)
class LLMConfig:
    """Configuration for LLM-backed answer generation."""

    provider: str = "openai"
    openai_model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 800
    max_context_chunks: int = 3
    max_relationships_per_chunk: int = 4
    max_snippet_lines: int = 50
    max_chunk_tokens: int = 300
    max_chunk_characters: int = 1200
    max_prompt_characters: int = 5600
    approx_chars_per_token: int = 4
    max_prompt_tokens: int = 1400
    ollama_max_context_chunks: int = 3
    ollama_max_snippet_lines: int = 40
    ollama_max_chunk_tokens: int = 300
    ollama_max_chunk_characters: int = 1200
    ollama_max_prompt_characters: int = 4800
    ollama_max_prompt_tokens: int = 1200
    fallback_to_ollama: bool = True


@dataclass(slots=True)
class TraceConfig:
    """Configuration for graph-based flow tracing."""

    max_trace_depth: int = 5
    max_trace_branching: int = 3


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""
    storage_dir: Path = Path(".codebase_ai")
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)

    @property
    def index_dir(self) -> Path:
        """Return the canonical index storage directory.

        This always points to `.codebase_ai/index` per project convention.
        """
        return Path(".codebase_ai/index")
