from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from openai import OpenAI


class RealAgentUnavailable(RuntimeError):
    """Raised when the configured real-victim LLM endpoint is not reachable."""


@dataclass(frozen=True, slots=True)
class QwenVLLMConfig:
    base_url: str = "http://127.0.0.1:8010/v1"
    model: str = "Qwen/Qwen3-14B"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = 20260615
    max_tokens: int = 768
    timeout_s: float = 120.0
    enable_thinking: bool = False
    require_live_endpoint: bool = True
    strict_model_check: bool = True

    @classmethod
    def from_env(cls) -> "QwenVLLMConfig":
        return cls(
            base_url=str(os.environ.get("UMPEEK_REAL_AGENT_VLLM_BASE_URL", "http://127.0.0.1:8010/v1")).rstrip("/"),
            model=str(os.environ.get("UMPEEK_REAL_AGENT_MODEL", "Qwen/Qwen3-14B")),
            api_key=str(os.environ.get("UMPEEK_REAL_AGENT_API_KEY", "EMPTY")),
            temperature=float(os.environ.get("UMPEEK_REAL_AGENT_TEMPERATURE", "0.0")),
            top_p=float(os.environ.get("UMPEEK_REAL_AGENT_TOP_P", "1.0")),
            seed=_optional_int(os.environ.get("UMPEEK_REAL_AGENT_SEED", "20260615")),
            max_tokens=int(os.environ.get("UMPEEK_REAL_AGENT_MAX_TOKENS", "768")),
            timeout_s=float(os.environ.get("UMPEEK_REAL_AGENT_TIMEOUT", "120.0")),
            enable_thinking=_truthy(os.environ.get("UMPEEK_REAL_AGENT_ENABLE_THINKING"), default=False),
            require_live_endpoint=_truthy(os.environ.get("UMPEEK_REAL_AGENT_REQUIRE_LIVE_ENDPOINT"), default=True),
            strict_model_check=_truthy(os.environ.get("UMPEEK_REAL_AGENT_STRICT_MODEL_CHECK"), default=True),
        )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    raw_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _truthy(raw: Any, *, default: bool = False) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _optional_int(raw: Any) -> int | None:
    text = str(raw or "").strip()
    if not text:
        return None
    return int(text)


def strip_qwen_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


class QwenVLLMClient:
    """OpenAI-compatible client for Qwen3 served by vLLM in non-thinking mode."""

    def __init__(self, config: QwenVLLMConfig | None = None) -> None:
        self.config = config or QwenVLLMConfig.from_env()
        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout_s,
        )

    def healthcheck(self) -> None:
        if not self.config.require_live_endpoint:
            return
        root = self.config.base_url.rsplit("/v1", 1)[0].rstrip("/")
        endpoint = root + "/health"
        request = urllib.request.Request(endpoint, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=min(10.0, self.config.timeout_s)) as response:
                if response.status >= 400:
                    raise RealAgentUnavailable(f"vLLM healthcheck returned HTTP {response.status}.")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RealAgentUnavailable(
                f"vLLM endpoint is not reachable at {self.config.base_url}. "
                "Start Qwen3-14B with scripts/launch_qwen3_14b_vllm.py first."
            ) from exc
        if self.config.strict_model_check:
            self._validate_served_model(root)

    def _validate_served_model(self, root: str) -> None:
        endpoint = root + "/v1/models"
        try:
            with urllib.request.urlopen(endpoint, timeout=min(10.0, self.config.timeout_s)) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RealAgentUnavailable(f"Could not verify served vLLM model at {endpoint}: {exc}") from exc
        served_ids = [
            str(item.get("id") or "")
            for item in parsed.get("data", [])
            if isinstance(item, Mapping)
        ]
        wanted = str(self.config.model)
        aliases = {wanted, wanted.split("/")[-1], os.environ.get("UMPEEK_REAL_AGENT_SERVED_MODEL", "")}
        aliases = {alias for alias in aliases if alias}
        if not any(served_id in aliases or wanted in served_id for served_id in served_ids):
            raise RealAgentUnavailable(
                "Configured endpoint is not serving Qwen3-14B via vLLM. "
                f"wanted={wanted!r}, served_models={served_ids!r}, endpoint={self.config.base_url!r}."
            )

    def chat(self, messages: Sequence[Mapping[str, str]], *, response_format_json: bool = False) -> LLMResponse:
        started = time.perf_counter()
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": bool(self.config.enable_thinking)}
        }
        if self.config.seed is not None:
            extra_body["seed"] = int(self.config.seed)
        kwargs: dict[str, Any] = {}
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            completion = self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": str(m["role"]), "content": str(m["content"])} for m in messages],
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                extra_body=extra_body,
                **kwargs,
            )
        except Exception as exc:
            raise RealAgentUnavailable(f"Qwen/vLLM chat completion failed: {exc}") from exc
        latency = time.perf_counter() - started
        choice = completion.choices[0] if completion.choices else None
        raw_text = str(choice.message.content if choice and choice.message else "")
        usage = completion.usage
        return LLMResponse(
            text=strip_qwen_thinking(raw_text),
            raw_text=raw_text,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_s=round(latency, 6),
            model=str(completion.model or self.config.model),
            metadata={
                "provider": "vllm_openai_compatible",
                "non_thinking_mode": not self.config.enable_thinking,
                "chat_template_kwargs": {"enable_thinking": bool(self.config.enable_thinking)},
                "response_format_json": bool(response_format_json),
                "temperature": float(self.config.temperature),
                "top_p": float(self.config.top_p),
                "seed": self.config.seed,
            },
        )


def parse_json_object(text: str) -> dict[str, Any]:
    raw = strip_qwen_thinking(text)
    try:
        parsed = json.loads(raw)
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}
