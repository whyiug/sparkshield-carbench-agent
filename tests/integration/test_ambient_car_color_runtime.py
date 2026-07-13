from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import DecisionProposal
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
    _ambient_car_color_intent,
)


def tool(
    name: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


CAR_COLOR_TOOL = tool("get_car_color")
AMBIENT_TOOL = tool(
    "set_ambient_lights",
    {
        "on": {"type": "boolean"},
        "lightcolor": {
            "type": "string",
            "enum": [
                "RED",
                "GREEN",
                "BLUE",
                "YELLOW",
                "WHITE",
                "PINK",
                "ORANGE",
                "PURPLE",
                "CYAN",
                "NONE",
            ],
        },
    },
    ["on", "lightcolor"],
)


class NoModelClient:
    def generate(self, **kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("the deterministic color workflow must not call a model")


class ExplicitColorClient:
    def __init__(self) -> None:
        self.intent = IntentDraft.model_validate(
            {
                "language": "en",
                "intent_kind": "action",
                "call_for_action": True,
                "goals": [
                    {
                        "semantic_operation": "set_ambient_lights",
                        "desired_outcome": {"enabled": True, "color": "PURPLE"},
                    }
                ],
                "explicit_slots": {"enabled": True, "color": "PURPLE"},
            }
        )

    def generate(self, *, messages, response_model, critic=False):
        del critic
        if response_model is IntentDraft:
            return SimpleNamespace(value=self.intent)
        if response_model is DecisionProposal:
            payload = json.loads(messages[-1]["content"])
            goal = payload["semantic_goals"]["goals"][0]
            return SimpleNamespace(
                value=DecisionProposal.model_validate(
                    {
                        "kind": "tool_set",
                        "goal_ids": [goal["goal_id"]],
                        "tool_calls": [
                            {
                                "tool_name": "set_ambient_lights",
                                "arguments": {"on": True, "lightcolor": "PURPLE"},
                                "argument_sources": {
                                    "on": "model-output",
                                    "lightcolor": "model-output",
                                },
                                "goal_id": goal["goal_id"],
                            }
                        ],
                    }
                )
            )
        raise AssertionError(f"unexpected response model: {response_model}")


class ConversationClient:
    def __init__(self) -> None:
        self.intent = IntentDraft.model_validate(
            {
                "language": "en",
                "intent_kind": "conversation",
                "call_for_action": False,
                "goals": [],
                "explicit_slots": {},
            }
        )

    def generate(self, *, messages, response_model, critic=False):
        del messages, critic
        if response_model is IntentDraft:
            return SimpleNamespace(value=self.intent)
        raise AssertionError("a reporting utterance must not reach action planning")


def runtime(client: Any | None = None) -> CARGuardOrchestrator:
    selected = client or NoModelClient()
    return CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False),
        client_factory=lambda session: selected,
    )


def user_event(
    context_id: str,
    text: str,
    *,
    tools: tuple[dict[str, Any], ...] = (CAR_COLOR_TOOL, AMBIENT_TOOL),
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=f"{context_id}-user",
        system_policy="Follow the current safety policy.",
        user_text=text,
        live_tools=tools,
    )


def result_event(
    context_id: str, tool_name: str, content: Any, *, suffix: str
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=f"{context_id}-{suffix}",
        tool_results=({"toolName": tool_name, "content": content},),
    )


@pytest.mark.parametrize("color", ["PURPLE", "WHITE"])
def test_verified_exterior_color_drives_exact_ambient_set(color: str) -> None:
    context_id = f"ambient-{color.casefold()}"
    agent = runtime()

    first = agent.handle_event(
        user_event(
            context_id,
            "Turn on the ambient lighting and match it to my car's exterior color.",
        )
    )
    assert first.tool_calls == ({"tool_name": "get_car_color", "arguments": {}},)

    action = agent.handle_event(
        result_event(
            context_id,
            "get_car_color",
            {"status": "SUCCESS", "result": {"car_color": color}},
            suffix="color",
        )
    )
    assert action.tool_calls == (
        {
            "tool_name": "set_ambient_lights",
            "arguments": {"on": True, "lightcolor": color},
        },
    )

    completed = agent.handle_event(
        result_event(
            context_id,
            "set_ambient_lights",
            {"status": "SUCCESS", "result": {"on": True, "lightcolor": color}},
            suffix="set",
        )
    )
    assert completed.tool_calls == ()
    assert completed.text is not None
    assert "turned on" in completed.text.casefold()


@pytest.mark.parametrize(
    "result",
    [
        {"status": "SUCCESS", "result": {}},
        {"status": "SUCCESS", "result": {"car_color": "BLACK"}},
        {"status": "SUCCESS", "result": {"car_color": ["PURPLE"]}},
    ],
)
def test_missing_unknown_or_non_scalar_car_color_never_reaches_set(
    result: dict[str, Any],
) -> None:
    context_id = f"bad-color-{len(json.dumps(result))}"
    agent = runtime()
    first = agent.handle_event(
        user_event(
            context_id,
            "Set the cabin lights to match the color of my vehicle.",
        )
    )
    assert first.tool_calls[0]["tool_name"] == "get_car_color"

    blocked = agent.handle_event(
        result_event(context_id, "get_car_color", result, suffix="color")
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None


def test_missing_live_lightcolor_parameter_blocks_before_color_read() -> None:
    malformed_set = tool(
        "set_ambient_lights",
        {"on": {"type": "boolean"}},
        ["on"],
    )
    agent = runtime()
    blocked = agent.handle_event(
        user_event(
            "missing-lightcolor",
            "Match the interior lighting to my car color.",
            tools=(CAR_COLOR_TOOL, malformed_set),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None


def test_explicit_ambient_color_does_not_trigger_car_color_read() -> None:
    agent = runtime(ExplicitColorClient())
    action = agent.handle_event(
        user_event(
            "explicit-purple",
            "Turn on the ambient lights in purple.",
        )
    )
    assert action.tool_calls == (
        {
            "tool_name": "set_ambient_lights",
            "arguments": {"on": True, "lightcolor": "PURPLE"},
        },
    )


@pytest.mark.parametrize(
    "text",
    [
        "Match the ambient lighting to my car color and set the fan to level three.",
        "Set the fan to level three, then match the ambient lights to my car color.",
        "Please match the cabin lighting to my vehicle color. Turn on the air conditioning.",
    ],
)
def test_compound_control_requests_are_not_collapsed_to_ambient_only(
    text: str,
) -> None:
    assert _ambient_car_color_intent(text, turn_id="compound") is None


@pytest.mark.parametrize(
    "text",
    [
        "The ambient lighting matches my car's exterior color.",
        "Does the ambient lighting match my car's exterior color?",
        "I noticed the cabin lighting matches the color of my vehicle.",
        "Match the ambient lights to my car's exterior color, but do not change it.",
    ],
)
def test_reports_and_questions_never_authorize_car_color_actions(text: str) -> None:
    assert _ambient_car_color_intent(text, turn_id="report") is None
    agent = runtime(ConversationClient())
    outbound = agent.handle_event(user_event(f"report-{len(text)}", text))
    assert outbound.tool_calls == ()
    session = agent.sessions.get(f"report-{len(text)}")
    assert session is not None
    assert session.authorized_action_goal_ids == set()
