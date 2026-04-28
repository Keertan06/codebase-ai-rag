"""Prompt construction for codebase question answering."""

from __future__ import annotations

from codebase_ai.config import LLMConfig
from codebase_ai.models import SearchResult
import re

# Limits to keep snippets small for local models.
_SURROUNDING_LINES = 12
_TOP_CHUNK_FALLBACK = 2
_STRICT_PROMPT_TOKEN_LIMIT = 1200
_AGGRESSIVE_SNIPPET_LINE_LIMIT = 30

SYSTEM_PROMPT = """You answer questions about a codebase.
Use only the provided context.
If the context is insufficient, reply exactly: Not enough information found.
Do not invent functions, files, flows, or terms.
Always mention file names in the answer.
Keep the answer short and clear."""

OLLAMA_SYSTEM_PROMPT = """Use only the provided context.
If the context is insufficient, reply exactly: Not enough information found.
Do not invent anything.
Always mention file names.
Keep the answer short."""


class PromptBuilder:
    """Builds prompts from retrieved chunks and graph relationships."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def build(self, question: str, results: list[SearchResult]) -> tuple[str, str]:
        """Build the system and user prompts for answer generation."""
        sorted_results = sorted(results, key=lambda r: getattr(r, "score", 0.0), reverse=True)
        selected_results = sorted_results[: self._max_context_chunks()]
        system_prompt = OLLAMA_SYSTEM_PROMPT if self._is_ollama() else SYSTEM_PROMPT
        user_prompt = self._build_user_prompt(
            question=question,
            results=selected_results,
            compact=self._is_ollama(),
        )

        if self._prompt_too_large(system_prompt, user_prompt) and len(selected_results) > _TOP_CHUNK_FALLBACK:
            selected_results = selected_results[:_TOP_CHUNK_FALLBACK]
            user_prompt = self._build_user_prompt(
                question=question,
                results=selected_results,
                compact=True,
            )

        if self._strict_prompt_limit_exceeded(system_prompt, user_prompt):
            user_prompt = self._build_user_prompt(
                question=question,
                results=selected_results[:_TOP_CHUNK_FALLBACK],
                compact=True,
                snippet_line_limit=_AGGRESSIVE_SNIPPET_LINE_LIMIT,
            )

        user_prompt = self._truncate_prompt(user_prompt, token_limit=_STRICT_PROMPT_TOKEN_LIMIT)
        return system_prompt, user_prompt

    def _build_user_prompt(
        self,
        question: str,
        results: list[SearchResult],
        *,
        compact: bool,
        snippet_line_limit: int | None = None,
    ) -> str:
        context_sections: list[str] = []

        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            source_id = f"S{index}"
            symbol_name = chunk.symbol_name or "<anonymous>"
            snippet = self._extract_relevant_snippet(result, max_lines=snippet_line_limit)
            context_sections.append(
                "\n".join(
                    [
                        f"[{source_id}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line}",
                        f"# {symbol_name}",
                        "```",
                        snippet,
                        "```",
                    ]
                )
            )
        context_text = (
            self._compact_context(context_sections)
            if compact
            else "\n\n".join(context_sections)
        ) or "No relevant code snippets found."

        return "\n\n".join(
            [
                f"CONTEXT:\n{context_text}",
                f"QUESTION:\n{question}",
                (
                    "INSTRUCTIONS:\n"
                    "- Answer using context only\n"
                    "- Keep it short"
                ),
            ]
        )

    def _truncate_snippet(self, text: str, max_lines: int | None = None) -> str:
        lines = text.strip().splitlines()
        line_limit = max_lines or self._max_snippet_lines()
        limited = lines[:line_limit]
        snippet = "\n".join(limited)
        if len(lines) > line_limit:
            snippet += "\n..."
        max_characters = min(
            self._max_chunk_characters(),
            self._max_chunk_tokens() * self.config.approx_chars_per_token,
        )
        if len(snippet) > max_characters:
            snippet = snippet[: max_characters - 4].rstrip() + "\n..."
        if self._estimate_tokens(snippet) > self._max_chunk_tokens():
            max_characters = self._max_chunk_tokens() * self.config.approx_chars_per_token
            snippet = snippet[: max_characters - 4].rstrip() + "\n..."
        return snippet

    def _estimate_tokens(self, text: str) -> int:
        # very rough token estimate by chars / approx_chars_per_token
        if not text:
            return 0
        return max(1, int(len(text) / max(1, self.config.approx_chars_per_token)))

    def _extract_relevant_snippet(
        self,
        result: SearchResult,
        max_lines: int | None = None,
    ) -> str:
        """Return a focused snippet for a SearchResult.

        Strategy:
        - Prefer the lines containing the first matched term (if available).
        - Locate the nearest function/class start above that line.
        - Include a small surrounding window.
        - Enforce line and token caps before returning the snippet.
        """
        chunk = result.chunk
        text = chunk.text or ""
        lines = text.splitlines()
        total = len(lines)

        # Determine line of interest based on matched terms if available
        interest_line = None
        lower_text = text.lower()
        for term in result.matched_terms:
            if not term:
                continue
            idx = lower_text.find(term.lower())
            if idx >= 0:
                # map char index to line number
                interest_line = text[:idx].count("\n")
                break

        if interest_line is None:
            # default to center of chunk
            interest_line = total // 2

        # Try to find function/class start above interest_line
        func_start = None
        func_end = None

        # simple heuristics for common languages
        start_patterns = [
            re.compile(r"^\s*def\s+\w+\s*\(|^\s*class\s+\w+\b"),
            re.compile(r"^\s*(?:async\s+)?function\b|^\s*class\b|^\s*const\s+\w+\s*=\s*\(|^\s*let\s+\w+\s*=\s*\(|^\s*var\s+\w+\s*=\s*\("),
        ]

        for i in range(interest_line, -1, -1):
            line = lines[i]
            if any(p.search(line) for p in start_patterns):
                func_start = i
                break

        # If found a start, search for next top-level start to mark end
        if func_start is not None:
            for j in range(func_start + 1, total):
                if any(p.search(lines[j]) for p in start_patterns):
                    func_end = j - 1
                    break
            if func_end is None:
                func_end = total - 1

        if func_start is None:
            # No function/class found; use a window around interest_line
            start_idx = max(0, interest_line - _SURROUNDING_LINES)
            end_idx = min(total - 1, interest_line + _SURROUNDING_LINES)
        else:
            # include some surrounding context but avoid entire file
            start_idx = max(0, func_start - _SURROUNDING_LINES)
            end_idx = min(total - 1, func_end + _SURROUNDING_LINES)

        selected = lines[start_idx : end_idx + 1]
        snippet = "\n".join(selected).strip()

        if not snippet:
            return self._truncate_snippet(text, max_lines=max_lines)
        return self._truncate_snippet(snippet, max_lines=max_lines)

    def _truncate_prompt(self, prompt: str, token_limit: int | None = None) -> str:
        effective_token_limit = min(
            token_limit or self._max_prompt_tokens(),
            self._max_prompt_tokens(),
            _STRICT_PROMPT_TOKEN_LIMIT,
        )
        max_characters = min(
            self._max_prompt_characters(),
            effective_token_limit * self.config.approx_chars_per_token,
        )
        if len(prompt) <= max_characters:
            return prompt
        return prompt[: max_characters - 16].rstrip() + "\n\n[truncated]"

    def _is_ollama(self) -> bool:
        return self.config.provider.lower() == "ollama"

    def _max_context_chunks(self) -> int:
        if self._is_ollama():
            return self.config.ollama_max_context_chunks
        return self.config.max_context_chunks

    def _max_chunk_characters(self) -> int:
        if self._is_ollama():
            return self.config.ollama_max_chunk_characters
        return self.config.max_chunk_characters

    def _max_chunk_tokens(self) -> int:
        if self._is_ollama():
            return self.config.ollama_max_chunk_tokens
        return self.config.max_chunk_tokens

    def _max_snippet_lines(self) -> int:
        if self._is_ollama():
            return self.config.ollama_max_snippet_lines
        return self.config.max_snippet_lines

    def _max_prompt_characters(self) -> int:
        if self._is_ollama():
            return self.config.ollama_max_prompt_characters
        return self.config.max_prompt_characters

    def _max_prompt_tokens(self) -> int:
        if self._is_ollama():
            return self.config.ollama_max_prompt_tokens
        return self.config.max_prompt_tokens

    def _prompt_too_large(self, system_prompt: str, user_prompt: str) -> bool:
        combined = f"{system_prompt}\n\n{user_prompt}".strip()
        return self._estimate_tokens(combined) > self._max_prompt_tokens()

    def _strict_prompt_limit_exceeded(self, system_prompt: str, user_prompt: str) -> bool:
        combined = f"{system_prompt}\n\n{user_prompt}".strip()
        return self._estimate_tokens(combined) > _STRICT_PROMPT_TOKEN_LIMIT

    def _compact_context(self, sections: list[str]) -> str:
        compact_sections: list[str] = []
        for section in sections:
            lines = section.splitlines()
            if len(lines) < 4:
                compact_sections.append(section)
                continue
            compact_sections.append(
                "\n".join(
                    [
                        lines[0],
                        lines[2],
                        lines[3],
                    ]
                )
            )
        return "\n\n".join(compact_sections) if compact_sections else "No relevant code snippets found."
