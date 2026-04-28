"""LLM answer generation built on retrieval and graph context."""

from __future__ import annotations

from codebase_ai.config import AppConfig, LLMConfig
from codebase_ai.models import LLMAnswer, SearchResult
import logging

from .prompt_builder import PromptBuilder
from .providers import OLLAMA_FAILURE_FALLBACK, create_llm_provider

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates natural-language answers from retrieved code context."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        app_config = AppConfig()
        self.config = config or app_config.llm
        self.prompt_builder = PromptBuilder(self.config)

    def generate(self, question: str, results: list[SearchResult]) -> LLMAnswer:
        """Generate an answer from retrieved search results."""
        # Log USER QUERY
        logger.info("\n=== USER QUERY ===\n%s\n", question)

        # Log RETRIEVED CHUNKS
        if not results:
            logger.info("RETRIEVED CHUNKS: none\n")
        else:
            chunk_lines: list[str] = ["RETRIEVED CHUNKS:"]
            for idx, result in enumerate(results, start=1):
                chunk = result.chunk
                # Use the prompt builder's snippet truncation to keep previews compact
                try:
                    preview = self.prompt_builder._truncate_snippet(chunk.text)
                except Exception:
                    preview = (chunk.text or "").splitlines()[:3]
                    preview = "\n".join(preview)
                    if len(preview) > 200:
                        preview = preview[:197] + "..."

                chunk_lines.append(
                    f"- [{idx}] {chunk.file_path} | score={result.score:.4f} | vector={result.vector_score or 0.0:.4f}\n  preview:\n{preview}\n"
                )
            logger.info("%s\n", "\n".join(chunk_lines))

        # Build final prompts (builder truncates as configured)
        system_prompt, user_prompt = self.prompt_builder.build(question, results)
        # Log FINAL PROMPT (truncated by PromptBuilder)
        truncated_preview = user_prompt if user_prompt else ""
        logger.info("FINAL PROMPT (truncated):\n%s\n", truncated_preview)

        provider = create_llm_provider(self.config)
        try:
            answer_text = provider.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            logger.error("%s failed: %s", provider.provider_name, exc)
            answer_text = OLLAMA_FAILURE_FALLBACK

        # Log FINAL ANSWER
        logger.info("FINAL ANSWER:\n%s\n", answer_text)

        return LLMAnswer(
            text=answer_text,
            provider=provider.provider_name,
            model=provider.model_name,
            prompt=user_prompt,
        )
