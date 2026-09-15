"""Client for any OpenAI-compatible chat API (OpenAI, Yandex/other clouds, Ollama, llama.cpp server)."""

from typing import Protocol

from openai import AsyncOpenAI

from app.config import Settings


class LLM(Protocol):
    async def complete(self, messages: list[dict], max_tokens: int = 512) -> str: ...


def merge_system_into_user(messages: list[dict]) -> list[dict]:
    """For chat templates without a system role (e.g. YandexGPT in Ollama silently drops it):
    prepend system instructions to the first user message."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if not system:
        return rest
    for m in rest:
        if m["role"] == "user":
            m["content"] = f"{system}\n\n{m['content']}"
            return rest
    return [{"role": "user", "content": system}, *rest]


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        timeout: float = 120.0,
        system_role: bool = True,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.system_role = system_role

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleLLM":
        return cls(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
            system_role=settings.llm_system_role,
        )

    async def complete(self, messages: list[dict], max_tokens: int = 512) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages if self.system_role else merge_system_into_user(messages),
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
