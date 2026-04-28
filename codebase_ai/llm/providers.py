"""LLM provider abstractions for answer generation."""

from __future__ import annotations

import json
import logging
import os
import socket
from abc import ABC, abstractmethod
from dataclasses import replace
from urllib import error, request

from codebase_ai.config import LLMConfig

logger = logging.getLogger(__name__)
OLLAMA_FAILURE_FALLBACK = "LLM failed to generate answer. Showing retrieved context instead."


class LLMProvider(ABC):
    """Abstract base class for chat-completion providers."""

    provider_name: str
    model_name: str

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate an answer from prompts."""


class OpenAIProvider(LLMProvider):
    """OpenAI-backed provider implementation."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.provider_name = config.provider.lower()
        self.model_name = config.openai_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        api_key, base_url = self._resolve_credentials()
        if not api_key:
            env_var = "NVIDIA_API_KEY" if self.provider_name == "nvidia" else "OPENAI_API_KEY"
            raise RuntimeError(f"{env_var} is not set for {self.provider_name} answer generation.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed. Install dependencies from requirements.txt."
            ) from exc

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        message = response.choices[0].message if response.choices else None
        output_text = str(getattr(message, "content", "") or "").strip()
        if output_text:
            return output_text

        raise RuntimeError("OpenAI returned an empty response.")

    def _resolve_credentials(self) -> tuple[str | None, str]:
        if self.provider_name == "nvidia":
            return (
                self.config.api_key or os.getenv("NVIDIA_API_KEY"),
                "https://integrate.api.nvidia.com/v1",
            )

        return (
            self.config.api_key or os.getenv("OPENAI_API_KEY"),
            self.config.base_url,
        )


class OllamaProvider(LLMProvider):
    """Ollama-backed provider implementation."""

    REQUEST_TIMEOUT_SECONDS = 60

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.provider_name = "ollama"
        self.model_name = config.ollama_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        endpoint = f"{self.config.ollama_base_url.rstrip('/')}/api/generate"
        http_request = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(2):
            try:
                with request.urlopen(http_request, timeout=self.REQUEST_TIMEOUT_SECONDS) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                content = str(payload.get("response", "")).strip()
                if not content:
                    raise RuntimeError("Ollama returned an empty response.")
                return content
            except (TimeoutError, socket.timeout, error.HTTPError) as exc:
                logger.error("Ollama failed: %s", exc)
                if attempt == 0:
                    continue
                return OLLAMA_FAILURE_FALLBACK
            except error.URLError as exc:
                logger.error("Ollama failed: %s", exc)
                return OLLAMA_FAILURE_FALLBACK
            except Exception as exc:
                logger.error("Ollama failed: %s", exc)
                return OLLAMA_FAILURE_FALLBACK

        return OLLAMA_FAILURE_FALLBACK


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Create an LLM provider from configuration."""

    provider_name = config.provider.lower()
    if provider_name in {"openai", "nvidia"}:
        return OpenAIProvider(config)
    if provider_name == "ollama":
        return OllamaProvider(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def build_ollama_fallback_config(config: LLMConfig) -> LLMConfig:
    """Create an Ollama fallback config from an existing LLM config."""

    return replace(config, provider="ollama")
