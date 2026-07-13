import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.llm.client import (
    LLMFailure,
    LiteLLMStructuredClient,
)
from track_1_agent_under_test.car_guard.observability.turn_metrics import (
    TurnMetricsAccumulator,
)
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft


def test_default_litellm_import_uses_bundled_cost_map(monkeypatch) -> None:
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    module = ModuleType("litellm")
    module.completion = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", module)
    client = LiteLLMStructuredClient(AgentConfig(llm="test/model"))

    result = client._completion(marker="value")

    assert result == {"marker": "value"}
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    assert module.suppress_debug_info is True
    assert module.turn_off_message_logging is True
    assert module.redact_messages_in_exceptions is True
    assert module.redact_user_api_key_info is True
    assert module.set_verbose is False


def valid_response():
    message = SimpleNamespace(
        content=(
            '{"language":"en","intent_kind":"conversation",'
            '"call_for_action":false,"goals":[],"explicit_slots":{},'
            '"explicit_constraints":{},"ambiguities":[]}'
        )
    )
    return SimpleNamespace(
        model="test",
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
        _hidden_params={},
    )


def test_client_retries_with_sdk_retries_disabled(monkeypatch) -> None:
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return valid_response()

    monkeypatch.setattr("time.sleep", lambda _: None)
    client = LiteLLMStructuredClient(
        AgentConfig(llm="test", max_retries=2), completion_fn=completion
    )

    response = client.generate(
        messages=[{"role": "user", "content": "hello"}],
        response_model=IntentDraft,
    )

    assert response.value.intent_kind.value == "conversation"
    assert len(calls) == 3
    assert all(call["num_retries"] == 0 for call in calls)


def test_client_retries_malformed_structured_response_with_same_schema(
    monkeypatch,
) -> None:
    calls = []
    metrics = TurnMetricsAccumulator()

    def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            response = valid_response()
            response.choices[0].message.content = "not json"
            return response
        return valid_response()

    monkeypatch.setattr("time.sleep", lambda _: None)
    client = LiteLLMStructuredClient(
        AgentConfig(llm="test", max_retries=1),
        completion_fn=completion,
        metrics=metrics,
    )

    response = client.generate(
        messages=[{"role": "user", "content": "hello"}],
        response_model=IntentDraft,
    )

    assert response.value.intent_kind.value == "conversation"
    assert len(calls) == 2
    assert calls[0]["response_format"] is calls[1]["response_format"]
    assert metrics.llm_calls == 2


def test_client_does_not_retry_non_retryable_provider_error(monkeypatch) -> None:
    calls = []

    class AuthenticationFailure(RuntimeError):
        status_code = 401

    def completion(**kwargs):
        calls.append(kwargs)
        raise AuthenticationFailure("synthetic")

    monkeypatch.setattr("time.sleep", lambda _: None)
    client = LiteLLMStructuredClient(
        AgentConfig(llm="test", max_retries=2), completion_fn=completion
    )

    with pytest.raises(LLMFailure, match="AuthenticationFailure"):
        client.generate(
            messages=[{"role": "user", "content": "hello"}],
            response_model=IntentDraft,
        )

    assert len(calls) == 1


@pytest.mark.parametrize("status_location", ["string", "response"])
def test_client_normalizes_non_retryable_provider_status(
    monkeypatch, status_location: str
) -> None:
    calls = []

    class ProviderFailure(RuntimeError):
        pass

    failure = ProviderFailure("synthetic")
    if status_location == "string":
        failure.status_code = "400"  # type: ignore[attr-defined]
    else:
        failure.response = SimpleNamespace(status_code=400)  # type: ignore[attr-defined]

    def completion(**kwargs):
        calls.append(kwargs)
        raise failure

    monkeypatch.setattr("time.sleep", lambda _: None)
    client = LiteLLMStructuredClient(
        AgentConfig(llm="test", max_retries=2), completion_fn=completion
    )

    with pytest.raises(LLMFailure, match="ProviderFailure"):
        client.generate(
            messages=[{"role": "user", "content": "hello"}],
            response_model=IntentDraft,
        )

    assert len(calls) == 1


def test_auto_custom_openai_route_uses_json_object_with_schema_instruction() -> None:
    calls = []
    original_messages = [{"role": "user", "content": "Return intent JSON."}]

    def completion(**kwargs):
        calls.append(kwargs)
        return valid_response()

    client = LiteLLMStructuredClient(
        AgentConfig(
            llm="openai/profile-model",
            provider="openai",
            api_base="https://provider.example/v1",
            structured_output_mode="auto",
        ),
        completion_fn=completion,
    )

    client.generate(messages=original_messages, response_model=IntentDraft)

    assert original_messages == [{"role": "user", "content": "Return intent JSON."}]
    assert calls[0]["response_format"] == {"type": "json_object"}
    instruction = calls[0]["messages"][0]
    assert instruction["role"] == "system"
    assert "exactly one JSON object" in instruction["content"]
    assert '"intent_kind"' in instruction["content"]
    assert '"desired_outcome"' in instruction["content"]


def test_forced_json_schema_keeps_pydantic_response_format() -> None:
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return valid_response()

    client = LiteLLMStructuredClient(
        AgentConfig(
            llm="openai/profile-model",
            provider="openai",
            api_base="https://provider.example/v1",
            structured_output_mode="json_schema",
        ),
        completion_fn=completion,
    )

    client.generate(
        messages=[{"role": "system", "content": "Extract intent."}],
        response_model=IntentDraft,
    )

    assert calls[0]["response_format"] is IntentDraft
    assert len(calls[0]["messages"]) == 1
