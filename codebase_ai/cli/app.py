"""CLI application for Codebase Knowledge AI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from codebase_ai.config import AppConfig, EmbeddingConfig, LLMConfig, TraceConfig
from codebase_ai.embedding import ChunkEmbeddingIndexer
from codebase_ai.graph import FlowTracer
from codebase_ai.ingestion import CodebaseScanner
from codebase_ai.llm import AnswerGenerator
from codebase_ai.logging_config import configure_logging
from codebase_ai.models import RetrievalFilters, ScanFilters
from codebase_ai.parsing import CodeChunker
from codebase_ai.retrieval import CodeRetriever

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(
        prog="codebase-ai",
        description="Understand and analyze codebases with retrieval and graph reasoning.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging verbosity.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index",
        help="Scan a repository and print supported source files.",
    )
    index_parser.add_argument(
        "repo_path",
        type=Path,
        help="Path to the repository to scan.",
    )
    index_parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Glob pattern to include. Repeat to add multiple patterns.",
    )
    index_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern to exclude. Repeat to add multiple patterns.",
    )
    index_parser.add_argument(
        "--language",
        action="append",
        choices=["python", "javascript", "typescript"],
        default=[],
        help="Restrict scanning to one or more languages.",
    )
    index_parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Stop scanning after indexing this many supported files.",
    )
    index_parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Also print chunk-level results for Step 2.",
    )
    index_parser.add_argument(
        "--build-vector-index",
        action="store_true",
        help="Generate embeddings for chunks and persist a local FAISS index.",
    )
    index_parser.add_argument(
        "--embedding-provider",
        choices=["sentence-transformers", "openai"],
        default=AppConfig().embedding.provider,
        help="Embedding backend to use when building the vector index.",
    )
    index_parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override the embedding model name for the selected provider.",
    )
    index_parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Directory where the vector index should be stored.",
    )
    index_parser.set_defaults(handler=handle_index)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Retrieve the most relevant code chunks for a question.",
    )
    ask_parser.add_argument(
        "query",
        help="Natural-language question about the codebase.",
    )
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=AppConfig().retrieval.default_top_k,
        help="Number of chunks to return.",
    )
    ask_parser.add_argument(
        "--language",
        action="append",
        choices=["python", "javascript", "typescript"],
        default=[],
        help="Restrict retrieval to one or more languages.",
    )
    ask_parser.add_argument(
        "--file-glob",
        action="append",
        default=[],
        help="Restrict retrieval to files matching one or more glob patterns.",
    )
    ask_parser.add_argument(
        "--chunk-type",
        action="append",
        default=[],
        help="Restrict retrieval to chunk types such as function, class, method, or fallback_block.",
    )
    ask_parser.add_argument(
        "--symbol",
        action="append",
        default=[],
        help="Restrict retrieval to exact symbol names. Repeat for multiple symbols.",
    )
    ask_parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Base storage directory containing the saved vector index.",
    )
    ask_parser.add_argument(
        "--llm-provider",
        choices=["openai", "nvidia", "ollama"],
        default=AppConfig().llm.provider,
        help="LLM provider used to turn retrieved context into a natural-language answer.",
    )
    ask_parser.add_argument(
        "--llm-model",
        default=None,
        help="Override the configured model name for the selected LLM provider.",
    )
    ask_parser.add_argument(
        "--temperature",
        type=float,
        default=AppConfig().llm.temperature,
        help="Sampling temperature for answer generation.",
    )
    ask_parser.add_argument(
        "--max-tokens",
        type=int,
        default=AppConfig().llm.max_tokens,
        help="Maximum number of tokens to generate in the answer.",
    )
    ask_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation and print only the retrieved context.",
    )
    ask_parser.set_defaults(handler=handle_ask)

    trace_parser = subparsers.add_parser(
        "trace",
        help="Trace an ordered code flow from the relationship graph.",
    )
    trace_parser.add_argument(
        "query",
        help="Feature, file, class, or function name to trace.",
    )
    trace_parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Base storage directory containing the saved graph index.",
    )
    trace_parser.add_argument(
        "--max-depth",
        type=int,
        default=AppConfig().trace.max_trace_depth,
        help="Maximum traversal depth for the trace.",
    )
    trace_parser.add_argument(
        "--max-branching",
        type=int,
        default=AppConfig().trace.max_trace_branching,
        help="Maximum number of neighbors to follow from each step.",
    )
    trace_parser.set_defaults(handler=handle_trace)

    return parser


def handle_index(args: argparse.Namespace) -> int:
    """Handle the index command for repository scanning and chunk preview."""

    scanner = CodebaseScanner()
    filters = ScanFilters(
        include_globs=tuple(args.include),
        exclude_globs=tuple(args.exclude),
        languages=tuple(args.language),
        max_files=args.max_files,
    )
    files = scanner.scan(args.repo_path, filters=filters)

    print(f"Indexed {len(files)} supported files from {args.repo_path.resolve()}")
    for source_file in files:
        print(
            f"- [{source_file.language}] {source_file.path} "
            f"({source_file.line_count} lines, {source_file.size_bytes} bytes)"
        )

    # Always chunk the scanned files and build the vector + graph indexes.
    chunker = CodeChunker()
    chunks = chunker.chunk_files(files)
    if args.show_chunks:
        print(f"\nGenerated {len(chunks)} chunks")
        for chunk in chunks:
            symbol_name = chunk.symbol_name or "<anonymous>"
            print(
                f"- [{chunk.language}] {chunk.file_path}:"
                f"{chunk.start_line}-{chunk.end_line} "
                f"{chunk.chunk_type} {symbol_name}"
            )

    print(f"\nGenerated {len(chunks)} chunks for embedding")

    embedding_config = EmbeddingConfig(provider=args.embedding_provider)
    if args.embedding_model:
        if args.embedding_provider == "openai":
            embedding_config.openai_model = args.embedding_model
        else:
            embedding_config.model_name = args.embedding_model

    indexer = ChunkEmbeddingIndexer(
        embedding_config=embedding_config,
        storage_dir=args.index_dir,
    )
    # Build vector index (this calls FaissVectorStore.save internally)
    index_dir = indexer.build_index(chunks)
    # Build and persist relationship graph
    indexer.build_graph_index(source_files=files, chunks=chunks)
    print(
        f"\nVector index saved to {index_dir.resolve()} "
        f"using provider '{args.embedding_provider}'"
    )
    print(f"Relationship graph saved to {index_dir.resolve() / 'graph.json'}")

    logger.info("Index command completed successfully")
    return 0


def handle_ask(args: argparse.Namespace) -> int:
    """Handle metadata-aware retrieval plus LLM answer generation."""

    llm_config = _build_llm_config(args)
    filters = RetrievalFilters(
        languages=tuple(args.language),
        file_globs=tuple(args.file_glob),
        chunk_types=tuple(args.chunk_type),
        symbol_names=tuple(args.symbol),
    )
    retriever = CodeRetriever(storage_dir=args.index_dir)
    effective_top_k = _effective_retrieval_top_k(args.top_k, llm_config)
    results = retriever.retrieve(args.query, top_k=effective_top_k, filters=filters)
    if args.no_llm:
        print("LLM disabled. Showing retrieved context.")
        _print_retrieved_context(args.query, results)
        logger.info("Ask command completed successfully with --no-llm")
        return 0

    answer_generator = AnswerGenerator(config=llm_config)
    answer = answer_generator.generate(args.query, results)

    print(f"Answer ({answer.provider}:{answer.model})")
    print(answer.text)
    _print_retrieved_context(args.query, results)

    logger.info("Ask command completed successfully")
    return 0


def _print_retrieved_context(query: str, results: list) -> None:
    """Print the retrieved chunks for a query."""

    print(f"\nSupporting context for: {query}")
    print(f"Retrieved {len(results)} results")
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        symbol_name = chunk.symbol_name or "<anonymous>"
        matched_terms = ", ".join(result.matched_terms) if result.matched_terms else "vector"
        print(
            f"{index}. [{chunk.language}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
            f"{chunk.chunk_type} {symbol_name} | score={result.score:.4f} | match={matched_terms}"
        )
        print("   ---")
        preview = chunk.text.strip().splitlines()
        for line in preview[:8]:
            print(f"   {line}")
        if len(preview) > 8:
            print("   ...")
        if result.graph_neighbors:
            print("   graph:")
            for neighbor in result.graph_neighbors[:4]:
                print(f"   {neighbor}")


def _build_llm_config(args: argparse.Namespace) -> LLMConfig:
    """Build an LLM config from CLI arguments."""

    base_config = AppConfig().llm
    config = LLMConfig(
        provider=args.llm_provider,
        openai_model=base_config.openai_model,
        base_url=base_config.base_url,
        api_key=base_config.api_key,
        ollama_model=base_config.ollama_model,
        ollama_base_url=base_config.ollama_base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_context_chunks=base_config.max_context_chunks,
        max_relationships_per_chunk=base_config.max_relationships_per_chunk,
        max_snippet_lines=base_config.max_snippet_lines,
        max_chunk_characters=base_config.max_chunk_characters,
        max_prompt_characters=base_config.max_prompt_characters,
        approx_chars_per_token=base_config.approx_chars_per_token,
        max_prompt_tokens=base_config.max_prompt_tokens,
        ollama_max_context_chunks=base_config.ollama_max_context_chunks,
        ollama_max_snippet_lines=base_config.ollama_max_snippet_lines,
        ollama_max_chunk_characters=base_config.ollama_max_chunk_characters,
        ollama_max_prompt_characters=base_config.ollama_max_prompt_characters,
        ollama_max_prompt_tokens=base_config.ollama_max_prompt_tokens,
        fallback_to_ollama=base_config.fallback_to_ollama,
    )
    if args.llm_model:
        if args.llm_provider in {"openai", "nvidia"}:
            config.openai_model = args.llm_model
        else:
            config.ollama_model = args.llm_model
    return config


def _effective_retrieval_top_k(requested_top_k: int, llm_config: LLMConfig) -> int:
    """Clamp retrieval size for providers that need smaller prompts."""

    if llm_config.provider.lower() == "ollama":
        return min(requested_top_k, llm_config.ollama_max_context_chunks)
    return requested_top_k


def handle_trace(args: argparse.Namespace) -> int:
    """Handle graph-based flow tracing without using an LLM."""

    trace_config = TraceConfig(
        max_trace_depth=args.max_depth,
        max_trace_branching=args.max_branching,
    )
    tracer = FlowTracer(storage_dir=args.index_dir, config=trace_config)
    result = tracer.trace(args.query)

    if result.error:
        print("FLOW TRACE:")
        print(result.error)
        logger.warning("Trace command could not resolve a flow for query: %s", args.query)
        return 1

    print("FLOW TRACE:\n")
    for index, step in enumerate(result.steps, start=1):
        symbol_display = _format_trace_symbol(step.symbol_name, step.chunk_type)
        print(f"{index}. {symbol_display}  [{step.file_path}]")

    print("\nRELATIONSHIPS:\n")
    if result.relationships:
        step_type_by_identity = {
            (step.symbol_name, step.file_path): step.chunk_type for step in result.steps
        }
        for relationship in result.relationships:
            source_display = _format_trace_symbol(
                relationship.source_symbol,
                step_type_by_identity.get(
                    (relationship.source_symbol, relationship.source_file_path),
                    "function",
                ),
            )
            target_display = _format_trace_symbol(
                relationship.target_symbol,
                step_type_by_identity.get(
                    (relationship.target_symbol, relationship.target_file_path),
                    "function",
                ),
            )
            print(f"{source_display} -> {relationship.relationship} -> {target_display}")
    else:
        print("No downstream relationships found within the configured trace limits.")

    print("\nSUMMARY:\n")
    print(result.summary)

    logger.info("Trace command completed successfully")
    return 0


def _format_trace_symbol(symbol_name: str, chunk_type: str) -> str:
    """Format a symbol for human-readable trace output."""

    if chunk_type in {"function", "method"}:
        return f"{symbol_name}()"
    return symbol_name


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    configure_logging(level=getattr(logging, args.log_level))
    return args.handler(args)
