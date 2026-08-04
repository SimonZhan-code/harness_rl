"""Policy client — the OpenAI-compatible seam.

The harness talks to the policy ONLY through this client, which wraps an
OpenAI-compatible `chat/completions` endpoint. Swapping `base_url` is the ONLY change
between Step 1 and Step 2. **Local (open-model) inference uses SGLang primary, vLLM backup**
(both serve OpenAI-compatible endpoints; Step 2 slime rollouts serve via SGLang). **Closed
models** go through the vendor API as-is (whatever inference the provider uses). No GPU deps here.

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
    For open models, serve locally with **SGLang** (primary) or **vLLM** (backup) and use
    `ModelClient.for_sglang(...)` / `.for_vllm(...)`; for closed models pass the vendor model
    id (e.g. `gpt-5.5`, `gemini/gemini-3.5-flash`) with the api key set via env.
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

    @classmethod
    def _for_openai_server(cls, served_model: str, base_url: str, api_key: str,
                           **kwargs: Any) -> "ModelClient":
        """Client for any self-hosted OpenAI-compatible server (SGLang / vLLM).

        LiteLLM routes to a local server when the model is prefixed `openai/` and `base_url`
        points at it. `api_key` is a placeholder unless the server was launched with one.
        """
        model = served_model if served_model.startswith("openai/") else f"openai/{served_model}"
        return cls(model=model, base_url=base_url, api_key=api_key, **kwargs)

    @classmethod
    def for_sglang(cls, served_model: str, base_url: str = "http://localhost:30000/v1",
                   api_key: str = "local", **kwargs: Any) -> "ModelClient":
        """PRIMARY local backend. SGLang default OpenAI-compatible port is 30000."""
        return cls._for_openai_server(served_model, base_url, api_key, **kwargs)

    @classmethod
    def for_vllm(cls, served_model: str, base_url: str = "http://localhost:8000/v1",
                 api_key: str = "local", **kwargs: Any) -> "ModelClient":
        """BACKUP local backend. vLLM default OpenAI-compatible port is 8000."""
        return cls._for_openai_server(served_model, base_url, api_key, **kwargs)

    def chat(self, messages: list[Message], temperature: float | None = None) -> ChatResult:
        import litellm  # lazy

        resp = litellm.completion(
            model=self.model,
            messages=[m.to_openai() for m in messages],
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature if temperature is None else temperature,
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

    def chat_many(self, messages: list[Message], n: int,
                  temperature: float | None = None) -> list[ChatResult]:
        """Draw `n` independent completions for the SAME context — used to branch B summaries at a
        compaction point (tree rollout). Prefers one `n=n` request (all `choices` surfaced); falls
        back to `n` sequential `chat` calls if the server/route doesn't honor `n`. A higher
        `temperature` gives the sibling summaries diversity."""
        if n <= 1:
            return [self.chat(messages, temperature=temperature)]
        import litellm  # lazy

        try:
            resp = litellm.completion(
                model=self.model,
                messages=[m.to_openai() for m in messages],
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens,
                n=n,
                **self.extra,
            )
            choices = resp.get("choices", []) or []
            usage = resp.get("usage", {}) or {}
            comp = usage.get("completion_tokens", 0) // max(1, len(choices))
            out = [ChatResult(text=c["message"]["content"],
                              prompt_tokens=usage.get("prompt_tokens", 0),
                              completion_tokens=comp, raw={})
                   for c in choices]
            if len(out) >= n:
                return out[:n]
        except Exception:
            pass
        return [self.chat(messages, temperature=temperature) for _ in range(n)]


class StubModel(ModelClient):
    """No-network fake policy. `responder(messages) -> str` returns the raw model text.

    Used for the local seam test and eval dry-runs (no GPU, no API).
    """

    def __init__(self, responder: Callable[[list[Message]], str], model: str = "stub") -> None:
        super().__init__(model=model)
        self._responder = responder

    def chat(self, messages: list[Message], temperature: float | None = None) -> ChatResult:
        text = self._responder(messages)
        # rough token accounting so cost fields populate in traces
        pt = sum(max(1, len(m.content or "") // 4) for m in messages)
        return ChatResult(text=text, prompt_tokens=pt, completion_tokens=max(1, len(text) // 4))

    def chat_many(self, messages: list[Message], n: int,
                  temperature: float | None = None) -> list[ChatResult]:
        """Call the responder `n` times; a stateful responder can vary output per call to emulate
        diverse summary samples for the tree-rollout stub."""
        return [self.chat(messages, temperature=temperature) for _ in range(n)]
