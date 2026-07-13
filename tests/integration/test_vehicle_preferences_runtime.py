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


PREFERENCE_TOOL = tool(
    "get_user_preferences",
    {
        "preference_categories": {
            "type": "object",
            "properties": {
                "vehicle_settings": {
                    "type": "object",
                    "properties": {
                        "climate_control": {"type": "boolean"},
                        "vehicle_settings": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        }
    },
    ["preference_categories"],
)
FAN_TOOL = tool(
    "set_fan_speed",
    {"level": {"type": "integer", "minimum": 0, "maximum": 5}},
    ["level"],
)
CIRCULATION_TOOL = tool(
    "set_air_circulation",
    {
        "mode": {
            "type": "string",
            "enum": ["FRESH_AIR", "RECIRCULATION", "AUTO"],
        }
    },
    ["mode"],
)
STEERING_TOOL = tool(
    "set_steering_wheel_heating",
    {"level": {"type": "integer", "minimum": 0, "maximum": 3}},
    ["level"],
)
TEMPERATURE_TOOL = tool(
    "set_climate_temperature",
    {
        "temperature": {
            "type": "number",
            "minimum": 16,
            "maximum": 28,
            "multipleOf": 0.5,
        },
        "seat_zone": {
            "type": "string",
            "enum": ["ALL_ZONES", "DRIVER", "PASSENGER"],
        },
    },
    ["temperature", "seat_zone"],
)


class PreferenceClient:
    def __init__(
        self,
        *,
        operation: str,
        desired_outcome: dict[str, Any],
        tool_name: str,
        expected_arguments: dict[str, Any],
    ) -> None:
        self.operation = operation
        self.tool_name = tool_name
        self.expected_arguments = expected_arguments
        self.action_calls = 0
        self.intent = IntentDraft.model_validate(
            {
                "language": "en",
                "intent_kind": "action",
                "call_for_action": True,
                "goals": [
                    {
                        "semantic_operation": operation,
                        "desired_outcome": desired_outcome,
                    }
                ],
                "explicit_slots": desired_outcome,
            }
        )

    def generate(self, *, messages, response_model, critic=False):
        del critic
        if response_model is IntentDraft:
            return SimpleNamespace(value=self.intent)
        if response_model is DecisionProposal:
            self.action_calls += 1
            payload = json.loads(messages[-1]["content"])
            goal = next(
                item
                for item in payload["semantic_goals"]["goals"]
                if item["semantic_operation"] == self.operation
            )
            return SimpleNamespace(
                value=DecisionProposal.model_validate(
                    {
                        "kind": "tool_set",
                        "goal_ids": [goal["goal_id"]],
                        "tool_calls": [
                            {
                                "tool_name": self.tool_name,
                                "arguments": self.expected_arguments,
                                "argument_sources": {
                                    name: "model-output"
                                    for name in self.expected_arguments
                                },
                                "goal_id": goal["goal_id"],
                            }
                        ],
                    }
                )
            )
        raise AssertionError(f"unexpected response model: {response_model}")


def runtime(client: PreferenceClient) -> CARGuardOrchestrator:
    return CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False),
        client_factory=lambda session: client,
    )


def user_event(
    context_id: str,
    text: str,
    *,
    controls: tuple[dict[str, Any], ...],
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=f"{context_id}-user",
        system_policy="Follow the current safety policy.",
        user_text=text,
        live_tools=(PREFERENCE_TOOL, *controls),
    )


def result_event(context_id: str, content: Any) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=f"{context_id}-preference",
        tool_results=({"toolName": "get_user_preferences", "content": content},),
    )


@pytest.mark.parametrize(
    (
        "case",
        "text",
        "operation",
        "desired",
        "control",
        "category",
        "preference_result",
        "expected_arguments",
    ),
    [
        (
            "circulation",
            "Change the air circulation to my preferred mode.",
            "set_air_circulation",
            {},
            CIRCULATION_TOOL,
            {"vehicle_settings": {"climate_control": True}},
            ["Has preference for fresh air mode to improve air quality"],
            {"mode": "FRESH_AIR"},
        ),
        (
            "fan",
            "Turn on the fan at my preferred speed.",
            "set_fan_speed",
            {},
            FAN_TOOL,
            {"vehicle_settings": {"climate_control": True}},
            ["user prefers fan speed level 3 as the default"],
            {"level": 3},
        ),
        (
            "steering",
            "Turn on steering wheel heating at my preferred level.",
            "set_steering_wheel_heating",
            {},
            STEERING_TOOL,
            {"vehicle_settings": {"vehicle_settings": True}},
            ["user prefers steering wheel heating level 2"],
            {"level": 2},
        ),
        (
            "temperature",
            "Set my default comfortable temperature for all zones.",
            "set_climate_temperature",
            {"seat_zone": "ALL_ZONES"},
            TEMPERATURE_TOOL,
            {"vehicle_settings": {"climate_control": True}},
            ["default comfortable temperature is 22 degrees"],
            {"temperature": 22, "seat_zone": "ALL_ZONES"},
        ),
    ],
)
def test_direct_vehicle_preferences_are_read_before_clarification_and_bound(
    case: str,
    text: str,
    operation: str,
    desired: dict[str, Any],
    control: dict[str, Any],
    category: dict[str, Any],
    preference_result: list[str],
    expected_arguments: dict[str, Any],
) -> None:
    client = PreferenceClient(
        operation=operation,
        desired_outcome=desired,
        tool_name=control["function"]["name"],
        expected_arguments=expected_arguments,
    )
    agent = runtime(client)
    first = agent.handle_event(user_event(case, text, controls=(control,)))
    assert first.text is None
    assert first.tool_calls == (
        {
            "tool_name": "get_user_preferences",
            "arguments": {"preference_categories": category},
        },
    )
    assert client.action_calls == 0

    preference_key = next(iter(category["vehicle_settings"]))
    action = agent.handle_event(
        result_event(
            case,
            {
                "status": "SUCCESS",
                "result": {
                    "vehicle_settings": {
                        preference_key: preference_result,
                    }
                },
            },
        )
    )
    assert action.tool_calls == (
        {
            "tool_name": control["function"]["name"],
            "arguments": expected_arguments,
        },
    )


@pytest.mark.parametrize(
    "preference_result",
    [
        ["user prefers fan speed level 2", "user prefers fan speed level 3"],
        ["user never prefers fan speed level 3"],
        ["user prefers fan speed level 9"],
    ],
)
def test_conflicting_negated_or_out_of_range_preference_never_sets(
    preference_result: list[str],
) -> None:
    client = PreferenceClient(
        operation="set_fan_speed",
        desired_outcome={},
        tool_name="set_fan_speed",
        expected_arguments={"level": 3},
    )
    agent = runtime(client)
    first = agent.handle_event(
        user_event(
            "bad-fan-pref",
            "Turn on the fan at my preferred speed.",
            controls=(FAN_TOOL,),
        )
    )
    assert first.tool_calls[0]["tool_name"] == "get_user_preferences"
    blocked = agent.handle_event(
        result_event(
            "bad-fan-pref",
            {
                "status": "SUCCESS",
                "result": {
                    "vehicle_settings": {
                        "climate_control": preference_result,
                    }
                },
            },
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert client.action_calls == 0


def test_missing_preference_field_never_sets() -> None:
    client = PreferenceClient(
        operation="set_air_circulation",
        desired_outcome={},
        tool_name="set_air_circulation",
        expected_arguments={"mode": "FRESH_AIR"},
    )
    agent = runtime(client)
    first = agent.handle_event(
        user_event(
            "missing-pref-field",
            "Change the air circulation to my preferred mode.",
            controls=(CIRCULATION_TOOL,),
        )
    )
    assert first.tool_calls
    clarification = agent.handle_event(
        result_event(
            "missing-pref-field",
            {"status": "SUCCESS", "result": {"vehicle_settings": {}}},
        )
    )
    assert clarification.tool_calls == ()
    assert clarification.text is not None
    assert client.action_calls == 0

    action = agent.handle_event(
        InboundEvent(
            context_id="missing-pref-field",
            message_id="missing-pref-field-choice",
            user_text="Fresh air, please.",
        )
    )
    assert action.tool_calls == (
        {
            "tool_name": "set_air_circulation",
            "arguments": {"mode": "FRESH_AIR"},
        },
    )


def test_failed_preference_lookup_falls_back_to_clarification() -> None:
    client = PreferenceClient(
        operation="set_fan_speed",
        desired_outcome={},
        tool_name="set_fan_speed",
        expected_arguments={"level": 3},
    )
    agent = runtime(client)
    first = agent.handle_event(
        user_event(
            "failed-fan-pref",
            "Turn on the fan at my preferred speed.",
            controls=(FAN_TOOL,),
        )
    )
    assert first.tool_calls[0]["tool_name"] == "get_user_preferences"

    clarification = agent.handle_event(
        result_event(
            "failed-fan-pref",
            {"status": "ERROR", "error": "preferences unavailable"},
        )
    )
    assert clarification.tool_calls == ()
    assert clarification.text is not None
    assert client.action_calls == 0


def test_explicit_value_overrides_preference_lookup() -> None:
    client = PreferenceClient(
        operation="set_fan_speed",
        desired_outcome={"level": 4},
        tool_name="set_fan_speed",
        expected_arguments={"level": 4},
    )
    agent = runtime(client)
    action = agent.handle_event(
        user_event(
            "explicit-fan",
            "Set the fan speed to level 4.",
            controls=(FAN_TOOL,),
        )
    )
    assert action.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 4}},
    )


def test_warming_temperature_preference_precedes_other_workflow_reads() -> None:
    client = PreferenceClient(
        operation="set_climate_temperature",
        desired_outcome={"seat_zone": "ALL_ZONES"},
        tool_name="set_climate_temperature",
        expected_arguments={"temperature": 22, "seat_zone": "ALL_ZONES"},
    )
    agent = runtime(client)
    first = agent.handle_event(
        user_event(
            "warming-preference",
            "Set my default comfortable temperature for all zones as part of warming the car.",
            controls=(TEMPERATURE_TOOL,),
        )
    )
    assert first.tool_calls == (
        {
            "tool_name": "get_user_preferences",
            "arguments": {
                "preference_categories": {
                    "vehicle_settings": {"climate_control": True}
                }
            },
        },
    )
