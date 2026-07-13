from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import (
    DecisionProposal,
    EvidenceSourceKind,
)
from track_1_agent_under_test.car_guard.planning.intent_grounding import (
    _climate_temperature_zone_sync_zones,
)
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
    _fan_speed_preference_intent,
    _steering_wheel_heating_preference_intent,
)


def _tool(
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


PREFERENCE_TOOL = _tool(
    "get_user_preferences",
    {
        "preference_categories": {
            "type": "object",
            "properties": {
                "vehicle_settings": {
                    "type": "object",
                    "properties": {
                        "vehicle_settings": {"type": "boolean"},
                        "climate_control": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        }
    },
    ["preference_categories"],
)
STEERING_SET_TOOL = _tool(
    "set_steering_wheel_heating",
    {
        "level": {
            "type": "integer",
            "minimum": 0,
            "maximum": 3,
        }
    },
    ["level"],
)
FAN_SET_TOOL = _tool(
    "set_fan_speed",
    {
        "level": {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
        }
    },
    ["level"],
)
TEMPERATURE_READ_TOOL = _tool("get_temperature_inside_car")
TEMPERATURE_SET_TOOL = _tool(
    "set_climate_temperature",
    {
        "temperature": {
            "type": "number",
            "multipleOf": 0.5,
            "minimum": 16,
            "maximum": 28,
        },
        "seat_zone": {
            "type": "string",
            "enum": ["ALL_ZONES", "DRIVER", "PASSENGER"],
        },
    },
    ["temperature", "seat_zone"],
)


class CapabilityClient:
    def __init__(self, intent: dict[str, Any]) -> None:
        self.intent = IntentDraft.model_validate(intent)
        self.action_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        del critic
        if response_model is IntentDraft:
            return SimpleNamespace(value=self.intent)
        if response_model is DecisionProposal:
            self.action_calls += 1
            payload = json.loads(messages[-1]["content"])
            goal = payload["semantic_goals"]["goals"][0]
            operation = goal["semantic_operation"]
            tool_name = {
                "set_climate_temperature": "set_climate_temperature",
                "set_fan_speed": "set_fan_speed",
                "set_steering_wheel_heating": "set_steering_wheel_heating",
            }[operation]
            arguments = dict(goal["desired_outcome"])
            return SimpleNamespace(
                value=DecisionProposal.model_validate(
                    {
                        "kind": "tool_set",
                        "goal_ids": [goal["goal_id"]],
                        "tool_calls": [
                            {
                                "tool_name": tool_name,
                                "arguments": arguments,
                                "argument_sources": {
                                    name: "model-output" for name in arguments
                                },
                                "goal_id": goal["goal_id"],
                            }
                        ],
                    }
                )
            )
        raise AssertionError(f"unexpected response model: {response_model}")


def _runtime(client: CapabilityClient) -> CARGuardOrchestrator:
    return CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=24),
        client_factory=lambda session: client,
    )


def _user_event(
    *,
    context_id: str,
    text: str,
    tools: tuple[dict[str, Any], ...],
    suffix: str = "user",
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=f"{context_id}-{suffix}",
        system_policy="Use only current verified vehicle state and available controls.",
        user_text=text,
        live_tools=tools,
    )


def _result_event(
    *,
    context_id: str,
    tool_name: str,
    result: dict[str, Any],
    suffix: str = "result",
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=f"{context_id}-{tool_name}-{suffix}",
        tool_results=(
            {
                "toolName": tool_name,
                "content": {"status": "SUCCESS", "result": result},
            },
        ),
    )


def test_missing_steering_level_uses_bound_stored_preference() -> None:
    client = CapabilityClient(
        {
            "language": "en",
            "intent_kind": "action",
            "call_for_action": True,
            "goals": [
                {
                    "semantic_operation": "set_steering_wheel_heating",
                    "desired_outcome": {"level": 99},
                }
            ],
            "explicit_slots": {"level": 99},
        }
    )
    agent = _runtime(client)
    context_id = "steering-preference"

    read = agent.handle_event(
        _user_event(
            context_id=context_id,
            text="Turn on the steering wheel heating.",
            tools=(PREFERENCE_TOOL, STEERING_SET_TOOL),
        )
    )
    assert read.tool_calls == (
        {
            "tool_name": "get_user_preferences",
            "arguments": {
                "preference_categories": {
                    "vehicle_settings": {"vehicle_settings": True}
                }
            },
        },
    )

    action = agent.handle_event(
        _result_event(
            context_id=context_id,
            tool_name="get_user_preferences",
            result={
                "vehicle_settings": {
                    "vehicle_settings": [
                        "My preferred steering wheel heating is level 3."
                    ]
                }
            },
        )
    )
    assert action.tool_calls == (
        {
            "tool_name": "set_steering_wheel_heating",
            "arguments": {"level": 3},
        },
    )
    session = agent.sessions.get(context_id)
    assert session is not None and session.intent is not None
    goal = session.intent.goals[0]
    derived = next(
        evidence
        for evidence in session.evidence.evidence.values()
        if evidence.proposition == f"selected_argument:{goal.goal_id}:level"
    )
    assert derived.source_kind is EvidenceSourceKind.DERIVED
    assert derived.value == 3
    assert len(derived.derived_from) == 1
    parent = session.evidence.evidence[derived.derived_from[0]]
    assert parent.proposition == "preferred_steering_wheel_heating_level"
    assert parent.source_kind is EvidenceSourceKind.TOOL


def test_level_free_fan_command_uses_bound_stored_preference() -> None:
    client = CapabilityClient(
        {
            "language": "en",
            "intent_kind": "action",
            "call_for_action": True,
            "goals": [
                {
                    "semantic_operation": "set_fan_speed",
                    "desired_outcome": {"level": 99},
                }
            ],
            "explicit_slots": {"level": 99},
        }
    )
    agent = _runtime(client)
    context_id = "fan-preference"

    read = agent.handle_event(
        _user_event(
            context_id=context_id,
            text=(
                "Hello! The cabin air feels a bit stale. Could you please turn "
                "on the fan for me?"
            ),
            tools=(PREFERENCE_TOOL, FAN_SET_TOOL),
        )
    )
    assert read.tool_calls == (
        {
            "tool_name": "get_user_preferences",
            "arguments": {
                "preference_categories": {
                    "vehicle_settings": {"climate_control": True}
                }
            },
        },
    )

    action = agent.handle_event(
        _result_event(
            context_id=context_id,
            tool_name="get_user_preferences",
            result={
                "vehicle_settings": {
                    "climate_control": ["The preferred fan speed is level 4."]
                }
            },
        )
    )
    assert action.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 4}},
    )
    session = agent.sessions.get(context_id)
    assert session is not None and session.intent is not None
    goal = session.intent.goals[0]
    selected = next(
        evidence
        for evidence in session.evidence.evidence.values()
        if evidence.proposition == f"selected_argument:{goal.goal_id}:level"
    )
    assert selected.source_kind is EvidenceSourceKind.DERIVED
    assert selected.value == 4
    assert len(selected.derived_from) == 1
    parent = session.evidence.evidence[selected.derived_from[0]]
    assert parent.proposition == "preferred_fan_speed_level"
    assert parent.source_kind is EvidenceSourceKind.TOOL


def test_level_free_fan_without_set_control_returns_clear_limitation() -> None:
    client = CapabilityClient(
        {
            "language": "en",
            "intent_kind": "action",
            "call_for_action": True,
            "goals": [],
            "explicit_slots": {},
        }
    )
    agent = _runtime(client)
    blocked = agent.handle_event(
        _user_event(
            context_id="fan-control-missing",
            text="Please enable the cabin fan.",
            tools=(PREFERENCE_TOOL,),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    lowered = blocked.text.casefold()
    assert "fan speed control" in lowered
    assert "pending action" not in lowered
    assert any(
        phrase in lowered
        for phrase in ("can't", "cannot", "could not", "unavailable", "not available")
    )
    session = agent.sessions.get("fan-control-missing")
    assert session is not None and session.intent is not None
    assert session.intent.goals[0].semantic_operation == "set_fan_speed"
    assert session.budget.attempted_sets == set()


def test_explicit_fan_level_keeps_absolute_value_and_skips_preference() -> None:
    client = CapabilityClient(
        {
            "language": "en",
            "intent_kind": "action",
            "call_for_action": True,
            "goals": [],
            "explicit_slots": {},
        }
    )
    agent = _runtime(client)
    action = agent.handle_event(
        _user_event(
            context_id="explicit-fan-level",
            text="Set the fan to level 2.",
            tools=(PREFERENCE_TOOL, FAN_SET_TOOL),
        )
    )
    assert action.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 2}},
    )


@pytest.mark.parametrize(
    "text",
    (
        "Turn on the fan.",
        "Could you please switch the HVAC fan on?",
        "Please activate the cabin fan.",
        "The air feels stagnant. Enable the fan, please.",
        "Hi there! It's getting a bit stuffy in here. Can you start the fan?",
        "I need the fan to be turned on.",
    ),
)
def test_level_free_fan_command_recovers_preference_goal(text: str) -> None:
    intent = _fan_speed_preference_intent(text, turn_id="synthetic")
    assert intent is not None
    assert intent.call_for_action
    assert len(intent.goals) == 1
    assert intent.goals[0].semantic_operation == "set_fan_speed"
    assert intent.goals[0].desired_outcome == {}


@pytest.mark.parametrize(
    "text",
    (
        "Don't turn on the fan.",
        "If the cabin gets warm, turn on the fan.",
        'She said, "Turn on the fan."',
        "Turn on the fan at level 3.",
        "Turn on the fan and open the passenger window.",
        "How would you turn on the fan?",
        "Maybe activate the HVAC fan.",
    ),
)
def test_level_free_fan_recovery_rejects_unsafe_scope(text: str) -> None:
    assert _fan_speed_preference_intent(text, turn_id="synthetic") is None


@pytest.mark.parametrize(
    "text",
    (
        "Turn on the steering wheel heating.",
        "Please enable steering wheel heating.",
        "Could you please turn on the steering heating?",
    ),
)
def test_level_free_steering_command_recovers_preference_goal(text: str) -> None:
    intent = _steering_wheel_heating_preference_intent(text, turn_id="synthetic")
    assert intent is not None
    assert intent.call_for_action
    assert len(intent.goals) == 1
    assert intent.goals[0].semantic_operation == "set_steering_wheel_heating"
    assert intent.goals[0].desired_outcome == {}


@pytest.mark.parametrize(
    "text",
    (
        "Don't turn on the steering wheel heating.",
        "If it gets cold, turn on the steering wheel heating.",
        'She said, "Turn on the steering wheel heating."',
        "Turn on the steering wheel heating to level 2.",
        "Turn on the steering wheel heating and the passenger seat heating.",
        "How would you turn on the steering wheel heating?",
    ),
)
def test_level_free_steering_recovery_rejects_unsafe_scope(text: str) -> None:
    assert (
        _steering_wheel_heating_preference_intent(text, turn_id="synthetic") is None
    )


def test_passenger_temperature_matches_fresh_driver_temperature() -> None:
    client = CapabilityClient(
        {
            "language": "en",
            "intent_kind": "action",
            "call_for_action": True,
            "goals": [
                {
                    "semantic_operation": "set_climate_temperature",
                    "desired_outcome": {
                        "temperature": 19,
                        "seat_zone": "PASSENGER",
                    },
                }
            ],
            "explicit_slots": {"temperature": 19, "seat_zone": "PASSENGER"},
        }
    )
    agent = _runtime(client)
    context_id = "passenger-temperature-sync"

    read = agent.handle_event(
        _user_event(
            context_id=context_id,
            text=(
                "Set the passenger temperature to the same as the driver "
                "temperature."
            ),
            tools=(TEMPERATURE_READ_TOOL, TEMPERATURE_SET_TOOL),
        )
    )
    assert read.tool_calls == (
        {"tool_name": "get_temperature_inside_car", "arguments": {}},
    )

    action = agent.handle_event(
        _result_event(
            context_id=context_id,
            tool_name="get_temperature_inside_car",
            result={
                "climate_temperature_driver": 22.5,
                "climate_temperature_passenger": 18,
                "temperature_unit": "Celsius",
            },
        )
    )
    assert action.tool_calls == (
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"temperature": 22.5, "seat_zone": "PASSENGER"},
        },
    )
    session = agent.sessions.get(context_id)
    assert session is not None and session.intent is not None
    goal = session.intent.goals[0]
    assert goal.desired_outcome == {
        "temperature": 22.5,
        "seat_zone": "PASSENGER",
    }
    derived_id = session.derived_value_evidence_by_goal[goal.goal_id]["temperature"]
    derived = session.evidence.evidence[derived_id]
    assert derived.source_kind is EvidenceSourceKind.DERIVED
    assert derived.derivation == "climate_temperature_zone_sync_from_source_v1"
    assert len(derived.derived_from) == 1
    parent = session.evidence.evidence[derived.derived_from[0]]
    assert parent.source_kind is EvidenceSourceKind.TOOL
    assert parent.value["climate_temperature_driver"] == 22.5


def test_temperature_zone_sync_rejects_malformed_source_snapshot() -> None:
    client = CapabilityClient(
        {
            "language": "en",
            "intent_kind": "action",
            "call_for_action": True,
            "goals": [
                {
                    "semantic_operation": "set_climate_temperature",
                    "desired_outcome": {"seat_zone": "PASSENGER"},
                }
            ],
            "explicit_slots": {"seat_zone": "PASSENGER"},
        }
    )
    agent = _runtime(client)
    context_id = "malformed-temperature-sync"
    read = agent.handle_event(
        _user_event(
            context_id=context_id,
            text=(
                "Set the passenger temperature to the same as the driver "
                "temperature."
            ),
            tools=(TEMPERATURE_READ_TOOL, TEMPERATURE_SET_TOOL),
        )
    )
    assert read.tool_calls

    blocked = agent.handle_event(
        _result_event(
            context_id=context_id,
            tool_name="get_temperature_inside_car",
            result={
                "climate_temperature_driver": "22.5",
                "climate_temperature_passenger": 18,
                "temperature_unit": "Celsius",
            },
        )
    )
    assert blocked.tool_calls == ()
    session = agent.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()


def test_repeated_temperature_sync_requires_a_fresh_snapshot() -> None:
    client = CapabilityClient(
        {
            "language": "en",
            "intent_kind": "action",
            "call_for_action": True,
            "goals": [
                {
                    "semantic_operation": "set_climate_temperature",
                    "desired_outcome": {"seat_zone": "PASSENGER"},
                }
            ],
            "explicit_slots": {"seat_zone": "PASSENGER"},
        }
    )
    agent = _runtime(client)
    context_id = "fresh-temperature-sync"
    request = "Set the passenger temperature to the same as the driver temperature."

    first_read = agent.handle_event(
        _user_event(
            context_id=context_id,
            text=request,
            tools=(TEMPERATURE_READ_TOOL, TEMPERATURE_SET_TOOL),
            suffix="first-user",
        )
    )
    assert first_read.tool_calls[0]["tool_name"] == "get_temperature_inside_car"
    first_action = agent.handle_event(
        _result_event(
            context_id=context_id,
            tool_name="get_temperature_inside_car",
            result={
                "climate_temperature_driver": 21,
                "climate_temperature_passenger": 18,
                "temperature_unit": "Celsius",
            },
            suffix="first-result",
        )
    )
    assert first_action.tool_calls[0]["arguments"]["temperature"] == 21
    agent.handle_event(
        _result_event(
            context_id=context_id,
            tool_name="set_climate_temperature",
            result={"temperature": 21, "seat_zone": "PASSENGER"},
            suffix="first-set",
        )
    )

    second_read = agent.handle_event(
        _user_event(
            context_id=context_id,
            text=request,
            tools=(TEMPERATURE_READ_TOOL, TEMPERATURE_SET_TOOL),
            suffix="second-user",
        )
    )
    assert second_read.tool_calls == (
        {"tool_name": "get_temperature_inside_car", "arguments": {}},
    )
    second_action = agent.handle_event(
        _result_event(
            context_id=context_id,
            tool_name="get_temperature_inside_car",
            result={
                "climate_temperature_driver": 23.5,
                "climate_temperature_passenger": 21,
                "temperature_unit": "Celsius",
            },
            suffix="second-result",
        )
    )
    assert second_action.tool_calls == (
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"temperature": 23.5, "seat_zone": "PASSENGER"},
        },
    )


@pytest.mark.parametrize(
    "text",
    (
        "Don't set the passenger temperature to the same as the driver temperature.",
        (
            "If I asked you to set the passenger temperature to the same as the "
            "driver temperature, what would happen?"
        ),
        (
            'She said, "Set the passenger temperature to the same as the driver '
            'temperature."'
        ),
        "Set the passenger temperature to the same as the passenger temperature.",
    ),
)
def test_temperature_zone_sync_grammar_rejects_non_authorizing_text(text: str) -> None:
    assert _climate_temperature_zone_sync_zones(text) is None


def test_negated_temperature_sync_never_reaches_a_set() -> None:
    client = CapabilityClient(
        {
            "language": "en",
            "intent_kind": "action",
            "call_for_action": True,
            "goals": [
                {
                    "semantic_operation": "set_climate_temperature",
                    "desired_outcome": {
                        "temperature": 22,
                        "seat_zone": "PASSENGER",
                    },
                }
            ],
            "explicit_slots": {"temperature": 22, "seat_zone": "PASSENGER"},
        }
    )
    agent = _runtime(client)
    blocked = agent.handle_event(
        _user_event(
            context_id="negated-temperature-sync",
            text=(
                "Don't set the passenger temperature to the same as the driver "
                "temperature."
            ),
            tools=(TEMPERATURE_READ_TOOL, TEMPERATURE_SET_TOOL),
        )
    )
    assert blocked.tool_calls == ()
    session = agent.sessions.get("negated-temperature-sync")
    assert session is not None
    assert session.budget.attempted_sets == set()
