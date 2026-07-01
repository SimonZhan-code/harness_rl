"""Policy client — the OpenAI-compatible seam.

The harness talks to the policy ONLY through this client, which wraps an
OpenAI-compatible `chat/completions` endpoint. Swapping `base_url` is the ONLY change
between Step 1 (vLLM / vendor API) and Step 2 (SkyRL-served policy). No GPU deps here.

`StubModel` is an in-process fake used for local (no-GPU) seam/unit tests: it returns
canned actions so the whole harness → gamma → logging path can be exercised offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from harness_rl.types import Message


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelClient:
    """Thin wrapper over LiteLLM (uniform OpenAI-compatible layer).

    LiteLLM is imported lazily so importing this module needs no network/model deps.
    For open models point `base_url` at a vLLM/SGLang server; for closed models use the
    vendor model id and set the api key via env.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra = extra or {}

    def chat(self, messages: list[Message]) -> ChatResult:
        import litellm  # lazy

        resp = litellm.completion(
            model=self.model,
            messages=[m.to_openai() for m in messages],
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.extra,
        )
        choice = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {}) or {}
        return ChatResult(
            text=choice,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            raw=dict(resp),
        )


class StubModel(ModelClient):
    """No-network fake policy. `responder(messages) -> str` returns the raw model text.

    Used for the local seam test and eval dry-runs (no GPU, no API).
    """

    def __init__(self, responder: Callable[[list[Message]], str], model: str = "stub") -> None:
        super().__init__(model=model)
        self._responder = responder

    def chat(self, messages: list[Message]) -> ChatResult:
        text = self._responder(messages)
        # rough token accounting so cost fields populate in traces
        pt = sum(max(1, len(m.content or "") // 4) for m in messages)
        return ChatResult(text=text, prompt_tokens=pt, completion_tokens=max(1, len(text) // 4))
