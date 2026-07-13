"""Small LiteLLM adapter that returns validated structured data."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import AgentConfig, provider_supports_cache_control
from ..observability.turn_metrics import TurnMetricsAccumulator


OutputT = TypeVar("OutputT", bound=BaseModel)


class LLMFailure(RuntimeError):
    """Redacted provider or structured-output failure."""


@dataclass(frozen=True, slots=True)
class StructuredLLMResponse(Generic[OutputT]):
    value: OutputT
    raw_model: str
    finish_reason: str | None


def _content(message: Any) -> str:
    if isinstance(message, dict):
        value = message.get("content")
    else:
        value = getattr(message, "content", None)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks)
    return ""


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _json_object_instruction(response_model: type[BaseModel]) -> dict[str, str]:
    schema = json.dumps(
        response_model.model_json_schema(),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return {
        "role": "system",
        "content": (
            "Return exactly one JSON object and no surrounding prose. The object "
            f"must validate against this JSON Schema: {schema}"
        ),
    }


def _provider_status_code(exc: Exception) -> int | None:
    status: Any = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, str) and status.isdigit():
        status = int(status)
    return status if isinstance(status, int) else None


def _retryable_provider_error(exc: Exception) -> bool:
    status = _provider_status_code(exc)
    if status is None:
        return True
    return status in {408, 409, 429} or status >= 500


class LiteLLMStructuredClient:
    def __init__(
        self,
        config: AgentConfig,
        *,
        metrics: TurnMetricsAccumulator | None = None,
        completion_fn: Any | None = None,
    ) -> None:
        self.config = config
        self.metrics = metrics
        self._completion_fn = completion_fn

    def _completion(self, **kwargs: Any) -> Any:
        if self._completion_fn is not None:
            return self._completion_fn(**kwargs)
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import litellm

        litellm.suppress_debug_info = True
        litellm.turn_off_message_logging = True
        litellm.redact_messages_in_exceptions = True
        litellm.redact_user_api_key_info = True
        litellm.set_verbose = False
        return litellm.completion(**kwargs)

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[OutputT],
        critic: bool = False,
    ) -> StructuredLLMResponse[OutputT]:
        options = self.config.completion_options(critic=critic)
        options["num_retries"] = 0
        model = str(options["model"])
        provider = options.get("custom_llm_provider")
        provider_name = str(provider) if provider is not None else None
        request_messages = [dict(message) for message in messages]
        if provider_supports_cache_control(model, provider_name):
            request_messages[0] = dict(request_messages[0])
            request_messages[0]["cache_control"] = {"type": "ephemeral"}
        output_mode = self.config.structured_output_mode
        if output_mode == "auto":
            output_mode = (
                "json_object"
                if provider_name == "openai" and options.get("api_base")
                else "json_schema"
            )
        if output_mode == "json_object":
            insertion_index = (
                1
                if request_messages and request_messages[0].get("role") == "system"
                else 0
            )
            request_messages.insert(
                insertion_index, _json_object_instruction(response_model)
            )
        options["messages"] = request_messages
        options["response_format"] = (
            {"type": "json_object"} if output_mode == "json_object" else response_model
        )

        last_error: Exception | None = None
        backoff_spent = 0.0
        for attempt in range(self.config.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._completion(**options)
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries or not _retryable_provider_error(
                    exc
                ):
                    break
                delay = min(float(2**attempt), 20.0 - backoff_spent)
                if delay <= 0:
                    break
                time.sleep(delay)
                backoff_spent += delay
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            try:
                choice = response.choices[0]
                parsed = getattr(choice.message, "parsed", None)
                if isinstance(parsed, response_model):
                    value = parsed
                elif parsed is not None:
                    value = response_model.model_validate(parsed)
                else:
                    payload = json.loads(_strip_fence(_content(choice.message)))
                    value = response_model.model_validate(payload)
            except (
                AttributeError,
                IndexError,
                json.JSONDecodeError,
                ValidationError,
            ) as exc:
                last_error = exc
                self._record_metrics(response, elapsed_ms, model)
                if attempt >= self.config.max_retries:
                    raise LLMFailure(
                        f"invalid structured response: {type(exc).__name__}"
                    ) from None
                delay = min(float(2**attempt), 20.0 - backoff_spent)
                if delay <= 0:
                    raise LLMFailure(
                        f"invalid structured response: {type(exc).__name__}"
                    ) from None
                time.sleep(delay)
                backoff_spent += delay
                continue

            self._record_metrics(response, elapsed_ms, model)
            return StructuredLLMResponse(
                value=value,
                raw_model=str(getattr(response, "model", model)),
                finish_reason=getattr(choice, "finish_reason", None),
            )

        error_name = type(last_error).__name__ if last_error else "UnknownError"
        raise LLMFailure(f"provider request failed: {error_name}") from None

    def _record_metrics(self, response: Any, elapsed_ms: float, model: str) -> None:
        if self.metrics is None:
            return
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        hidden = getattr(response, "_hidden_params", {}) or {}
        self.metrics.add(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            thinking_tokens=int(getattr(details, "reasoning_tokens", 0) or 0),
            cost=float(hidden.get("response_cost", 0.0) or 0.0),
            elapsed_ms=elapsed_ms,
            model=str(getattr(response, "model", model)),
        )
