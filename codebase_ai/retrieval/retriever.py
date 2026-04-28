"""Hybrid retrieval with metadata filtering and lightweight reranking."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from codebase_ai.config import AppConfig, EmbeddingConfig, RetrievalConfig
from codebase_ai.embedding import FaissVectorStore, create_embedding_provider
from codebase_ai.models import RetrievalFilters, SearchResult
from .graph_context import GraphContextExpander

logger = logging.getLogger(__name__)

QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*")
COMMON_QUERY_TERMS = {
    "a",
    "an",
    "and",
    "code",
    "does",
    "explain",
    "feature",
    "file",
    "flow",
    "for",
    "function",
    "how",
    "implementation",
    "in",
    "is",
    "method",
    "of",
    "show",
    "the",
    "trace",
    "what",
    "where",
    "work",
}


class CodeRetriever:
    """Retrieves relevant code chunks with metadata-aware filtering and reranking."""

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        embedding_config: EmbeddingConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        app_config = AppConfig()
        # Always use the canonical index directory to ensure consistency.
        self.index_dir = app_config.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = FaissVectorStore(self.index_dir)
        self.embedding_config = embedding_config or self._load_embedding_config(app_config)
        self.retrieval_config = retrieval_config or app_config.retrieval
        self.embedding_provider = create_embedding_provider(self.embedding_config)
        self.graph_expander = GraphContextExpander(
            storage_dir=self.index_dir,
            retrieval_config=self.retrieval_config,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[SearchResult]:
        """Retrieve and rerank chunks for a natural-language code query."""
        # Enforce default and an absolute hard cap to keep prompts small.
        requested = top_k or self.retrieval_config.default_top_k
        hard_cap = min(self.retrieval_config.max_returned_chunks, 4)
        active_top_k = min(requested, hard_cap)
        active_filters = filters or RetrievalFilters()
        query_embedding = self.embedding_provider.embed_texts([query])[0]
        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=active_top_k * self.retrieval_config.vector_candidate_multiplier,
            filters=active_filters,
            candidate_pool_size=active_top_k * self.retrieval_config.vector_candidate_multiplier,
        )
        reranked = self._rerank_results(query, candidates, active_filters)
        expanded = self.graph_expander.expand(reranked[:active_top_k])
        return expanded[:active_top_k]

    def _load_embedding_config(self, app_config: AppConfig) -> EmbeddingConfig:
        manifest_path = self.index_dir / "manifest.json"
        if not manifest_path.exists():
            logger.info("No retrieval manifest found at %s; using default embedding config", manifest_path)
            return app_config.embedding

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_provider = str(manifest.get("provider", app_config.embedding.provider)).strip().lower()
        if manifest_provider not in {"sentence-transformers", "openai"}:
            manifest_provider = "sentence-transformers"
        config = EmbeddingConfig(
            provider=manifest_provider,
            model_name=str(manifest.get("model_name", app_config.embedding.model_name)),
            openai_model=str(manifest.get("openai_model", app_config.embedding.openai_model)),
            batch_size=app_config.embedding.batch_size,
            normalize_embeddings=app_config.embedding.normalize_embeddings,
            index_dir_name=app_config.embedding.index_dir_name,
        )
        return config

    def _rerank_results(
        self,
        query: str,
        candidates: list[SearchResult],
        filters: RetrievalFilters,
    ) -> list[SearchResult]:
        query_terms = self._extract_query_terms(query)
        expanded_query_terms = self._expand_query_terms(query_terms)
        query_lower = query.lower()
        symbol_filters = {name.lower() for name in filters.symbol_names}
        chunk_type_filters = {chunk_type.lower() for chunk_type in filters.chunk_types}
        reranked: list[SearchResult] = []
        focused_symbol_terms = self._detect_symbol_focus_terms(query, query_terms, filters)

        # detect routing-related queries to bias results
        route_keywords = {"route", "router", "include_router", "routes", "routing"}
        route_query = any(k in query_lower for k in route_keywords)

        for result in candidates:
            chunk = result.chunk
            base_vector_score = result.vector_score if result.vector_score is not None else result.score
            score = base_vector_score * self.retrieval_config.semantic_similarity_weight
            matched_terms: list[str] = []
            direct_signal_found = False

            normalized_path = chunk.file_path.lower()
            symbol_name = (chunk.symbol_name or "").lower()
            chunk_type = chunk.chunk_type.lower()
            language = chunk.language.lower()
            content_lower = chunk.text.lower()
            path_tokens = self._tokenize_for_overlap(normalized_path)
            content_tokens = self._tokenize_for_overlap(content_lower)

            if focused_symbol_terms:
                symbol_focus_matches = [
                    term for term in focused_symbol_terms if self._symbol_matches_focus(symbol_name, term)
                ]
                if not symbol_focus_matches:
                    continue
                if symbol_name == "__init__":
                    continue
                score += self.retrieval_config.symbol_match_boost * 2
                matched_terms.append("symbol_focus")
                direct_signal_found = True

            path_matches = [
                term for term in expanded_query_terms if len(term) > 2 and term in normalized_path
            ]
            if path_matches:
                score += self.retrieval_config.path_match_boost
                matched_terms.append("path")
                direct_signal_found = True

            if symbol_name and any(term == symbol_name for term in expanded_query_terms):
                score += self.retrieval_config.symbol_match_boost
                matched_terms.append("symbol")
                direct_signal_found = True

            if chunk_type in query_lower:
                score += self.retrieval_config.chunk_type_match_boost
                matched_terms.append("chunk_type")
                direct_signal_found = True

            if language in query_lower:
                score += self.retrieval_config.language_match_boost
                matched_terms.append("language")

            content_matches = sorted(
                {
                    term
                    for term in expanded_query_terms
                    if len(term) > 2 and (term in content_lower or term in content_tokens)
                }
            )
            if content_matches:
                score += min(
                    self._overlap_strength(content_matches, expanded_query_terms, path_tokens, content_tokens)
                    * self.retrieval_config.content_term_boost,
                    self.retrieval_config.max_content_term_boost,
                )
                matched_terms.append("content")
                direct_signal_found = True

            # If this is a routing-related query, boost chunks that contain routing terms
            if route_query:
                route_found = any(
                    term in content_lower or term in normalized_path or term == symbol_name
                    for term in route_keywords
                )
                if route_found:
                    score += self.retrieval_config.route_match_boost
                    matched_terms.append("route_term")
                    direct_signal_found = True

            if chunk_type == "fallback_block":
                score -= self.retrieval_config.fallback_chunk_penalty

            # De-prioritize very large classes unless they directly match query terms
            try:
                chunk_length = max(0, int(chunk.end_line) - int(chunk.start_line) + 1)
            except Exception:
                chunk_length = 0
            if chunk_type == "class" and chunk_length > self.retrieval_config.large_class_line_threshold:
                # If no direct content/path/symbol signals, skip this large class as unrelated
                if not (path_matches or content_matches or symbol_name):
                    continue
                # otherwise apply a penalty to keep prompts small
                score -= self.retrieval_config.large_class_penalty
                matched_terms.append("large_class")

            if not direct_signal_found and base_vector_score < self.retrieval_config.min_direct_signal_score:
                continue

            if symbol_filters and symbol_name in symbol_filters:
                matched_terms.append("symbol_filter")
            if chunk_type_filters and chunk_type in chunk_type_filters:
                matched_terms.append("chunk_type_filter")

            reranked.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    vector_score=base_vector_score,
                    rerank_score=score,
                    matched_terms=tuple(matched_terms),
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked

    def _extract_query_terms(self, query: str) -> set[str]:
        return {term.lower() for term in QUERY_TOKEN_PATTERN.findall(query)}

    def _detect_symbol_focus_terms(
        self,
        query: str,
        query_terms: set[str],
        filters: RetrievalFilters,
    ) -> set[str]:
        if filters.symbol_names:
            return {name.lower() for name in filters.symbol_names if name}

        focused: set[str] = set()
        stripped_query = query.strip().lower()
        normalized_query = stripped_query.removesuffix("()")

        if normalized_query and re.fullmatch(r"[a-z_][a-z0-9_]*", normalized_query):
            focused.add(normalized_query)

        for term in query_terms:
            if term in COMMON_QUERY_TERMS or len(term) <= 2:
                continue
            normalized_term = term.removesuffix("()")
            if "_" in normalized_term or "." in normalized_term:
                focused.add(normalized_term)

        return focused

    def _expand_query_terms(self, query_terms: set[str]) -> set[str]:
        expanded = set(query_terms)
        for term in list(query_terms):
            normalized = term.strip().lower()
            if len(normalized) <= 2:
                continue
            if normalized.endswith("ing") and len(normalized) > 5:
                root = normalized[:-3]
                expanded.add(root)
                if root.endswith("t"):
                    expanded.add(f"{root}e")
            if normalized.endswith("er") and len(normalized) > 4:
                expanded.add(normalized[:-2])
            if normalized.endswith("s") and len(normalized) > 4:
                expanded.add(normalized[:-1])
            if normalized == "routing":
                expanded.update({"route", "router", "routes"})
        return expanded

    def _tokenize_for_overlap(self, text: str) -> set[str]:
        return {token.lower() for token in QUERY_TOKEN_PATTERN.findall(text)}

    def _symbol_matches_focus(self, symbol_name: str, focused_term: str) -> bool:
        if not symbol_name or not focused_term:
            return False
        normalized_symbol = symbol_name.lower().removesuffix("()")
        symbol_leaf = normalized_symbol.split(".")[-1]
        return (
            normalized_symbol == focused_term
            or symbol_leaf == focused_term
            or focused_term in normalized_symbol
            or focused_term in symbol_leaf
        )

    def _overlap_strength(
        self,
        content_matches: list[str],
        expanded_query_terms: set[str],
        path_tokens: set[str],
        content_tokens: set[str],
    ) -> float:
        token_matches = {term for term in expanded_query_terms if term in content_tokens or term in path_tokens}
        return float(len(content_matches) + len(token_matches))
