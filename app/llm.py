"""Client for any OpenAI-compatible chat API (OpenAI, Yandex/other clouds, Ollama, llama.cpp server)."""

from typing import Protocol

from openai import AsyncOpenAI

from app.config import Settings


class LLM(Protocol):
    async def complete(self, messages: list[dict], max_tokens: int = 512) -> str: ...


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, model: str, api_key: str, temperature: float = 0.0, timeout: float = 120.0):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleLLM":
        return cls(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
        )

    async def complete(self, messages: list[dict], max_tokens: int = 512) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
