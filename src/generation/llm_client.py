"""LLM client: local Ollama first, Groq (OpenAI-compatible API) as fallback."""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from src.common import config

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str          # "ollama:<name>" or "groq:<name>"
    latency_s: float


class LLMClient:
    def __init__(
        self,
        model: str = config.LLM_MODEL,
        ollama_host: str = config.OLLAMA_HOST,
        groq_api_key: str = config.GROQ_API_KEY,
        groq_model: str = GROQ_DEFAULT_MODEL,
        timeout_s: float = 120.0,
        use_fallback: bool = True,
        http: httpx.Client | None = None,
    ):
        self.model = model
        self.ollama_host = ollama_host.rstrip("/")
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.use_fallback = use_fallback
        self.http = http or httpx.Client(timeout=timeout_s)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            text = self._ollama(messages, temperature, max_tokens, json_mode)
            model = f"ollama:{self.model}"
        except (httpx.HTTPError, LLMError) as e:
            if not (self.use_fallback and self.groq_api_key):
                raise LLMError(f"Ollama failed and no Groq fallback: {e}") from e
            text = self._groq(messages, temperature, max_tokens, json_mode)
            model = f"groq:{self.groq_model}"
        return LLMResponse(text=text.strip(), model=model,
                           latency_s=time.perf_counter() - t0)

    def generate(self, system: str, user: str, **kw) -> LLMResponse:
        return self.chat([{"role": "system", "content": system},
                          {"role": "user", "content": user}], **kw)

    # --- backends ---------------------------------------------------------
    def _ollama(self, messages, temperature, max_tokens, json_mode) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "seed": config.SEED},
        }
        if json_mode:
            payload["format"] = "json"
        r = self.http.post(f"{self.ollama_host}/api/chat", json=payload)
        r.raise_for_status()
        try:
            return r.json()["message"]["content"]
        except (KeyError, ValueError) as e:
            raise LLMError(f"Bad Ollama response: {r.text[:200]}") from e

    def _groq(self, messages, temperature, max_tokens, json_mode) -> str:
        payload = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": config.SEED,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = self.http.post(GROQ_URL, json=payload,
                           headers={"Authorization": f"Bearer {self.groq_api_key}"})
        r.raise_for_status()
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"Bad Groq response: {r.text[:200]}") from e
