"""One small client for every LLM provider the suite supports.

Providers (set CX_LLM_PROVIDER, or leave unset to auto-detect from keys):
  azure   - Azure OpenAI (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, deployments below)
  openai  - OpenAI (OPENAI_API_KEY)
  gemini  - Google Gemini through its OpenAI-compatible endpoint (GEMINI_API_KEY)
  mock    - no network; apps fall back to deterministic logic so demos never break

All three real providers speak the OpenAI wire format, so one SDK covers them.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULTS = {
    "openai": {"chat": "gpt-4.1-mini", "embed": "text-embedding-3-small"},
    "gemini": {"chat": "gemini-2.5-flash", "embed": "gemini-embedding-001"},
    # For Azure these are *deployment names* you create in Azure AI Foundry.
    "azure": {"chat": "gpt-4.1-mini", "embed": "text-embedding-3-small"},
}


def detect_provider() -> str:
    explicit = os.getenv("CX_LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return "mock"


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


@dataclass
class LLM:
    provider: str = field(default_factory=detect_provider)
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        self._client = None
        if self.provider == "mock":
            return
        from openai import AzureOpenAI, OpenAI  # imported lazily so mock mode needs no SDK

        defaults = DEFAULTS.get(self.provider, DEFAULTS["openai"])
        self.chat_model = os.getenv("CX_CHAT_MODEL", defaults["chat"])
        self.embed_model = os.getenv("CX_EMBED_MODEL", defaults["embed"])
        if self.provider == "azure":
            self._client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                timeout=30, max_retries=1,
            )
        elif self.provider == "gemini":
            self._client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_BASE_URL, timeout=30, max_retries=1)
        elif self.provider == "openai":
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=30, max_retries=1)
        else:
            raise ValueError(f"Unknown CX_LLM_PROVIDER: {self.provider}")

    @property
    def live(self) -> bool:
        return self._client is not None

    @property
    def label(self) -> str:
        return "demo mode (no LLM)" if not self.live else f"{self.provider}:{self.chat_model}"

    def chat(self, system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
        if not self.live:
            raise RuntimeError("chat() called in mock mode; callers should use their fallback")
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.chat_model,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        self.usage.calls += 1
        self.usage.seconds += time.perf_counter() - start
        if resp.usage:
            self.usage.prompt_tokens += resp.usage.prompt_tokens or 0
            self.usage.completion_tokens += resp.usage.completion_tokens or 0
        return resp.choices[0].message.content or ""

    def chat_json(self, system: str, user: str) -> dict:
        return parse_json(self.chat(system, user, json_mode=True))

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.live:
            raise RuntimeError("embed() called in mock mode")
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            resp = self._client.embeddings.create(model=self.embed_model, input=texts[i : i + 64])
            out.extend(d.embedding for d in resp.data)
        return out


def parse_json(text: str) -> dict:
    """Parse a JSON object even if the model wrapped it in a code fence."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model output: {text[:200]}")
    return json.loads(text[start : end + 1])
