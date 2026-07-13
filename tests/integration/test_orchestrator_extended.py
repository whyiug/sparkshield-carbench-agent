from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.a2a.session_store import SuccessfulReadResult
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import DecisionProposal
from track_1_agent_under_test.car_guard.domain import (
    CommitDecision,
    EvidenceSourceKind,
    GateOutcome,
)
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.llm import LLMFailure
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
    _active_route_charging_request,
    _trip_range_information_intent,
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


def action_intent(
    goals: list[dict[str, Any]], *, explicit_slots: dict[str, Any]
) -> dict[str, Any]:
    return {
        "language": "en",
        "intent_kind": "action",
        "call_for_action": True,
        "goals": goals,
        "explicit_slots": explicit_slots,
    }


ActionFactory = Callable[[dict[str, Any]], dict[str, Any]]


class FakeStructuredClient:
    def __init__(self, intent: dict[str, Any], action_factory: ActionFactory) -> None:
        self.intent = IntentDraft.model_validate(intent)
        self.action_factory = action_factory
        self.intent_calls = 0
        self.action_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        del critic
        if response_model is IntentDraft:
            self.intent_calls += 1
            return SimpleNamespace(value=self.intent)
        if response_model is DecisionProposal:
            self.action_calls += 1
            payload = json.loads(messages[-1]["content"])
            return SimpleNamespace(
                value=DecisionProposal.model_validate(self.action_factory(payload))
            )
        raise AssertionError(f"unexpected response model: {response_model}")


class FailingActionClient(FakeStructuredClient):
    def generate(self, *, messages, response_model, critic=False):
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise LLMFailure("invalid structured response: ValidationError")
        return super().generate(
            messages=messages,
            response_model=response_model,
            critic=critic,
        )


class SequentialIntentClient:
    def __init__(self, intents: list[dict[str, Any]]) -> None:
        self.intents = [IntentDraft.model_validate(intent) for intent in intents]
        self.intent_calls = 0
        self.action_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        del messages, critic
        if response_model is IntentDraft:
            if self.intent_calls >= len(self.intents):
                raise AssertionError("unexpected additional intent extraction")
            intent = self.intents[self.intent_calls]
            self.intent_calls += 1
            return SimpleNamespace(value=intent)
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError("the deterministic base 48 workflow must not plan")
        raise AssertionError(f"unexpected response model: {response_model}")


class FailingCriticClient(FakeStructuredClient):
    def __init__(self, intent: dict[str, Any], action_factory: ActionFactory) -> None:
        super().__init__(intent, action_factory)
        self.critic_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        if critic:
            self.critic_calls += 1
            raise LLMFailure("invalid structured response: ValidationError")
        return super().generate(
            messages=messages,
            response_model=response_model,
            critic=critic,
        )


class NonAcceptingCriticClient(FakeStructuredClient):
    def __init__(
        self,
        intent: dict[str, Any],
        action_factory: ActionFactory,
        *,
        decision: str,
        reason_code: str,
    ) -> None:
        super().__init__(intent, action_factory)
        self.decision = decision
        self.reason_code = reason_code
        self.critic_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        if critic:
            self.critic_calls += 1
            return SimpleNamespace(
                value={
                    "decision": self.decision,
                    "reason_code": self.reason_code,
                    "reason": "The critic rejected the deterministic candidate.",
                }
            )
        return super().generate(
            messages=messages,
            response_model=response_model,
            critic=critic,
        )


def goal_from_payload(
    payload: dict[str, Any], semantic_operation: str
) -> dict[str, Any]:
    return next(
        goal
        for goal in payload["semantic_goals"]["goals"]
        if goal["semantic_operation"] == semantic_operation
    )


def set_proposal(
    payload: dict[str, Any],
    *,
    semantic_operation: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    goal = goal_from_payload(payload, semantic_operation)
    return {
        "kind": "tool_set",
        "goal_ids": [goal["goal_id"]],
        "tool_calls": [
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "argument_sources": {name: "model-output" for name in arguments},
                "goal_id": goal["goal_id"],
            }
        ],
    }


def runtime_for(
    client: FakeStructuredClient,
    *,
    enable_critic: bool = False,
    soft_max_steps: int = 36,
) -> CARGuardOrchestrator:
    config = AgentConfig(
        llm="test/model",
        enable_critic=enable_critic,
        soft_max_steps=soft_max_steps,
    )
    return CARGuardOrchestrator(config, client_factory=lambda session: client)


def user_event(
    *,
    context_id: str,
    message_id: str,
    text: str,
    tools: tuple[dict[str, Any], ...],
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        system_policy="Follow the current safety policy.",
        user_text=text,
        live_tools=tools,
    )


def result_event(
    *,
    context_id: str,
    message_id: str,
    tool_name: str,
    content: Any,
    tool_call_id: str | None = None,
) -> InboundEvent:
    result = {"toolName": tool_name, "content": content}
    if tool_call_id is not None:
        result["toolCallId"] = tool_call_id
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        tool_results=(result,),
    )


FAN_TOOL = tool(
    "set_fan_speed",
    {"level": {"type": "integer", "minimum": 0, "maximum": 5}},
    ["level"],
)

AMBIENT_LIGHT_TOOL = tool(
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

SEAT_HEATING_TOOL = tool(
    "set_seat_heating",
    {
        "level": {
            "type": "number",
            "multipleOf": 1,
            "minimum": 0,
            "maximum": 3,
        },
        "seat_zone": {
            "type": "string",
            "enum": ["ALL_ZONES", "DRIVER", "PASSENGER"],
        },
    },
    ["level", "seat_zone"],
)

SEAT_OCCUPANCY_TOOL = tool("get_seats_occupancy")
SEAT_HEATING_LEVEL_TOOL = tool("get_seat_heating_level")

STEERING_WHEEL_HEATING_TOOL = tool(
    "set_steering_wheel_heating",
    {
        "level": {
            "type": "number",
            "multipleOf": 1,
            "minimum": 0,
            "maximum": 3,
        }
    },
    ["level"],
)

AIRFLOW_DIRECTION_TOOL = tool(
    "set_fan_airflow_direction",
    {
        "direction": {
            "type": "string",
            "enum": [
                "FEET",
                "HEAD",
                "HEAD_FEET",
                "WINDSHIELD",
                "WINDSHIELD_FEET",
                "WINDSHIELD_HEAD",
                "WINDSHIELD_HEAD_FEET",
            ],
        }
    },
    ["direction"],
)

DISTANCE_BY_SOC_TOOL = tool(
    "get_distance_by_soc",
    {
        "initial_state_of_charge": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "final_state_of_charge": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "default": 0,
        },
    },
    ["initial_state_of_charge"],
)

CALENDAR_TOOL = tool(
    "get_entries_from_calendar",
    {
        "month": {"type": "integer", "minimum": 1, "maximum": 12},
        "day": {"type": "integer", "minimum": 1, "maximum": 31},
    },
    ["month", "day"],
)

NAVIGATION_STATE_TOOL = tool(
    "get_current_navigation_state",
    {
        "detailed_information": {"type": "boolean", "default": False},
    },
)

PREFERENCE_TOOL = tool(
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


AC_TOOLS = (
    tool("get_climate_settings"),
    tool("get_vehicle_window_positions"),
    tool("set_air_conditioning", {"on": {"type": "boolean"}}, ["on"]),
    FAN_TOOL,
    tool(
        "open_close_window",
        {
            "window": {
                "type": "string",
                "enum": [
                    "ALL",
                    "DRIVER",
                    "PASSENGER",
                    "DRIVER_REAR",
                    "PASSENGER_REAR",
                    "RIGHT_REAR",
                    "LEFT_REAR",
                ],
            },
            "percentage": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        ["window", "percentage"],
    ),
)


CLIMATE_TEMPERATURE_TOOL = tool(
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


BASE_40_TOOLS = (
    tool("get_temperature_inside_car"),
    *AC_TOOLS,
    CLIMATE_TEMPERATURE_TOOL,
)


DEFROST_POLICY = """
AUT-POL:010: Front defrost requires effective climate settings.
AUT-POL:011: Air conditioning requires effective fan and window settings.
"""


DEFROST_TOOLS = (
    tool("get_climate_settings"),
    tool("get_vehicle_window_positions"),
    tool(
        "set_fan_speed",
        {"level": {"type": "number", "minimum": 0, "maximum": 5}},
        ["level"],
    ),
    tool(
        "set_fan_airflow_direction",
        {"direction": {"type": "string", "enum": ["FEET", "WINDSHIELD"]}},
        ["direction"],
    ),
    tool("set_air_conditioning", {"on": {"type": "boolean"}}, ["on"]),
    tool(
        "set_window_defrost",
        {
            "on": {"type": "boolean"},
            "defrost_window": {"type": "string", "enum": ["FRONT", "REAR", "ALL"]},
        },
        ["on", "defrost_window"],
    ),
    tool(
        "open_close_window",
        {
            "window": {
                "type": "string",
                "enum": ["DRIVER", "PASSENGER", "DRIVER_REAR", "PASSENGER_REAR"],
            },
            "percentage": {"type": "number", "minimum": 0, "maximum": 100},
        },
        ["window", "percentage"],
    ),
)

FULL_DEFROST_TOOLS = (
    *DEFROST_TOOLS[:-1],
    tool(
        "open_close_window",
        {
            "window": {
                "type": "string",
                "enum": [
                    "ALL",
                    "DRIVER",
                    "PASSENGER",
                    "DRIVER_REAR",
                    "PASSENGER_REAR",
                ],
            },
            "percentage": {"type": "number", "minimum": 0, "maximum": 100},
        },
        ["window", "percentage"],
    ),
)


SUNROOF_POLICY = """
AUT-POL:005:The sunroof can only be opened if the sunshade is fully opened.
LLM-POL:008:AUT-POL:009: Check current weather before opening the sunroof.
CURRENT_LOCATION = {"id":"loc-home","name":"Home"}
DATETIME = {"year":2026,"month":7,"day":11,"hour":16,"minute":15}
"""


SUNROOF_TOOLS = (
    tool("get_sunroof_and_sunshade_position"),
    tool(
        "get_weather",
        {
            "location_or_poi_id": {"type": "string"},
            "month": {"type": "number"},
            "day": {"type": "number"},
            "time_hour_24hformat": {"type": "number"},
            "time_minutes": {"type": "number", "default": 0},
        },
        ["location_or_poi_id", "month", "day", "time_hour_24hformat"],
    ),
    tool(
        "open_close_sunroof",
        {"percentage": {"type": "number", "minimum": 0, "maximum": 100}},
        ["percentage"],
    ),
    tool(
        "open_close_sunshade",
        {"percentage": {"type": "number", "minimum": 0, "maximum": 100}},
        ["percentage"],
    ),
)


def ac_client() -> FakeStructuredClient:
    return FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_air_conditioning",
                    "desired_outcome": {"enabled": True},
                }
            ],
            explicit_slots={"enabled": True},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_air_conditioning",
            tool_name="set_air_conditioning",
            arguments={"on": True},
        ),
    )


def sunroof_client() -> FakeStructuredClient:
    return FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_sunroof_position",
                    "desired_outcome": {"percentage": 50},
                }
            ],
            explicit_slots={"percentage": 50},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_sunroof_position",
            tool_name="open_close_sunroof",
            arguments={"percentage": 50},
        ),
    )


def defrost_client() -> FakeStructuredClient:
    return FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_window_defrost",
                    "desired_outcome": {"enabled": True, "window": "FRONT"},
                }
            ],
            explicit_slots={"enabled": True, "window": "FRONT"},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_window_defrost",
            tool_name="set_window_defrost",
            arguments={"on": True, "defrost_window": "FRONT"},
        ),
    )


def ambiguous_defrost_client() -> FakeStructuredClient:
    return FakeStructuredClient(
        {
            **action_intent(
                [
                    {
                        "semantic_operation": "set_window_defrost",
                        "desired_outcome": {"enabled": True},
                    }
                ],
                explicit_slots={"enabled": True},
            ),
            "ambiguities": [
                {
                    "name": "window",
                    "candidate_values": ["ALL", "FRONT", "REAR"],
                    "goal_index": 0,
                }
            ],
        },
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_window_defrost",
            tool_name="set_window_defrost",
            arguments={"on": True, "defrost_window": "FRONT"},
        ),
    )


def test_climate_information_then_ac_keeps_existing_prerequisite_chain() -> None:
    client = ac_client()
    runtime = runtime_for(client)
    context_id = "base90-climate-then-ac"
    base90_tools = (
        *AC_TOOLS,
        tool(
            "set_air_circulation",
            {"mode": {"type": "string", "enum": ["FRESH_AIR", "RECIRCULATION"]}},
            ["mode"],
        ),
    )

    climate = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="base90-climate-info",
            text="What are the current climate settings?",
            tools=base90_tools,
        )
    )
    assert climate.tool_calls == (
        {"tool_name": "get_climate_settings", "arguments": {}},
    )
    answer = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base90-climate-info-result",
            tool_name="get_climate_settings",
            content={
                "status": "SUCCESS",
                "result": {
                    "fan_speed": 0,
                    "fan_airflow_direction": "WINDSHIELD",
                    "air_conditioning": False,
                    "air_circulation": "RECIRCULATION",
                    "window_front_defrost": False,
                    "window_rear_defrost": False,
                },
            },
        )
    )
    assert answer.tool_calls == ()
    assert answer.text is not None and "air conditioning is off" in answer.text
    assert client.intent_calls == 0
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.authorized_action_goal_ids == set()
    assert session.budget.attempted_sets == set()

    current = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="base90-ac-user",
            text="Okay, so the AC is off. Can you turn on the air conditioning?",
            tools=base90_tools,
        )
    )
    assert current.tool_calls == (
        {"tool_name": "get_vehicle_window_positions", "arguments": {}},
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base90-ac-windows",
            tool_name="get_vehicle_window_positions",
            content={
                "status": "SUCCESS",
                "result": {
                    "window_driver_position": 0,
                    "window_passenger_position": 0,
                    "window_driver_rear_position": 0,
                    "window_passenger_rear_position": 0,
                },
            },
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 1}},
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base90-ac-fan",
            tool_name="set_fan_speed",
            content={"status": "SUCCESS", "result": {"level": 1}},
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "set_air_conditioning", "arguments": {"on": True}},
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base90-ac-set",
            tool_name="set_air_conditioning",
            content={"status": "SUCCESS", "result": {"on": True}},
        )
    )
    assert outbound.tool_calls == ()
    assert outbound.text is not None and outbound.text.startswith("Done")
    assert len(session.budget.successful_sets) == 2

    client.intent = IntentDraft.model_validate(
        action_intent(
            [
                {
                    "semantic_operation": "set_air_circulation",
                    "desired_outcome": {"mode": "FRESH_AIR"},
                }
            ],
            explicit_slots={"mode": "FRESH_AIR"},
        )
    )
    client.action_factory = lambda payload: set_proposal(
        payload,
        semantic_operation="set_air_circulation",
        tool_name="set_air_circulation",
        arguments={"mode": "FRESH_AIR"},
    )
    outbound = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="base90-circulation-user",
            text=(
                "The AC is on now. Can you set the air circulation to fresh air mode?"
            ),
            tools=base90_tools,
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "set_air_circulation", "arguments": {"mode": "FRESH_AIR"}},
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base90-circulation-set",
            tool_name="set_air_circulation",
            content={"status": "SUCCESS", "result": {"mode": "FRESH_AIR"}},
        )
    )
    assert outbound.tool_calls == ()
    assert outbound.text is not None and outbound.text.startswith("Done")
    assert len(session.budget.successful_sets) == 3


def test_climate_information_with_extra_ac_command_falls_through_to_action() -> None:
    client = ac_client()
    runtime = runtime_for(client)
    context_id = "climate-info-extra-ac-command"

    outbound = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="climate-info-extra-ac-user",
            text="What are the current climate settings, and turn on the AC?",
            tools=AC_TOOLS,
        )
    )

    assert outbound.tool_calls == (
        {"tool_name": "get_climate_settings", "arguments": {}},
    )
    assert client.intent_calls == 1
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert len(session.authorized_action_goal_ids) == 1
    assert session.budget.attempted_sets == set()


def test_base90_hall88_missing_fan_level_blocks_before_any_set() -> None:
    client = ac_client()
    runtime = runtime_for(client)
    context_id = "base90-hall88-missing-fan-level"
    hall_tools = tuple(
        spec for spec in AC_TOOLS if spec["function"]["name"] != "set_fan_speed"
    ) + (tool("set_fan_speed"),)

    runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="base90-hall88-info",
            text="What are the current climate settings?",
            tools=hall_tools,
        )
    )
    answer = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base90-hall88-info-result",
            tool_name="get_climate_settings",
            content={
                "status": "SUCCESS",
                "result": {
                    "fan_speed": 0,
                    "fan_airflow_direction": "WINDSHIELD",
                    "air_conditioning": False,
                    "air_circulation": "RECIRCULATION",
                    "window_front_defrost": False,
                    "window_rear_defrost": False,
                },
            },
        )
    )
    assert answer.text is not None and "air conditioning is off" in answer.text

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="base90-hall88-ac",
            text="Please turn on the air conditioning.",
            tools=hall_tools,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "cannot express" in blocked.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.pending_calls == []
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_ac_reads_both_goal_conditioned_facts_before_set() -> None:
    client = ac_client()
    runtime = runtime_for(client)

    first = runtime.handle_event(
        user_event(
            context_id="ac",
            message_id="user-ac",
            text="Turn on the air conditioning.",
            tools=AC_TOOLS,
        )
    )
    assert first.text is None
    assert first.tool_calls == ({"tool_name": "get_climate_settings", "arguments": {}},)
    assert client.action_calls == 0

    second = runtime.handle_event(
        result_event(
            context_id="ac",
            message_id="climate-result",
            tool_name="get_climate_settings",
            content={"status": "SUCCESS", "result": {"fan_speed": 1}},
        )
    )
    assert second.text is None
    assert second.tool_calls == (
        {"tool_name": "get_vehicle_window_positions", "arguments": {}},
    )
    assert client.action_calls == 0

    third = runtime.handle_event(
        result_event(
            context_id="ac",
            message_id="window-result",
            tool_name="get_vehicle_window_positions",
            content={
                "status": "SUCCESS",
                "result": {
                    "window_driver_position": 0,
                    "window_passenger_position": 0,
                    "window_driver_rear_position": 0,
                    "window_passenger_rear_position": 0,
                },
            },
        )
    )
    assert third.text is None
    assert third.tool_calls == (
        {"tool_name": "set_air_conditioning", "arguments": {"on": True}},
    )
    assert client.action_calls == 1

    final = runtime.handle_event(
        result_event(
            context_id="ac",
            message_id="ac-result",
            tool_name="set_air_conditioning",
            content={"status": "SUCCESS", "result": {"on": True}},
        )
    )
    assert final.tool_calls == ()
    assert final.text is not None and final.text.startswith("Done")
    session = runtime.sessions.get("ac")
    assert session is not None
    assert session.goal_dag.goals[0].status.value == "done"


def test_frozen_ac_bundle_keeps_evidence_snapshot_until_final_set() -> None:
    client = ac_client()
    runtime = runtime_for(client)
    context_id = "ac-prerequisites"

    runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="ac-prerequisite-user",
            text="Turn on the air conditioning.",
            tools=AC_TOOLS,
        )
    )
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="ac-prerequisite-climate",
            tool_name="get_climate_settings",
            content={"status": "SUCCESS", "result": {"fan_speed": 0}},
        )
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="ac-prerequisite-windows",
            tool_name="get_vehicle_window_positions",
            content={
                "status": "SUCCESS",
                "result": {
                    "window_driver_position": 30,
                    "window_passenger_position": 0,
                    "window_driver_rear_position": 0,
                    "window_passenger_rear_position": 0,
                },
            },
        )
    )

    steps = (
        (
            "open_close_window",
            {"window": "DRIVER", "percentage": 0},
            {"window": "DRIVER", "percentage": 0},
        ),
        ("set_fan_speed", {"level": 1}, {"level": 1}),
        ("set_air_conditioning", {"on": True}, {"on": True}),
    )
    for index, (tool_name, arguments, result) in enumerate(steps):
        assert outbound.tool_calls == (
            {"tool_name": tool_name, "arguments": arguments},
        )
        session = runtime.sessions.get(context_id)
        assert session is not None
        assert session.evidence.current_state_version == 0
        outbound = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"ac-prerequisite-set-{index}",
                tool_name=tool_name,
                content={"status": "SUCCESS", "result": result},
            )
        )

    assert outbound.tool_calls == ()
    assert outbound.text is not None and outbound.text.startswith("Done")
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.evidence.current_state_version == 1
    assert client.action_calls == 1


def test_base94_closes_only_windows_strictly_above_twenty_percent() -> None:
    client = ac_client()
    runtime = runtime_for(client)
    context_id = "base94-window-threshold"

    outbound = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-user",
            system_policy=(
                "AUT-POL:011: Air conditioning requires effective fan and window "
                "settings."
            ),
            user_text="Could you turn on the air conditioning?",
            live_tools=AC_TOOLS,
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_climate_settings", "arguments": {}},
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-climate",
            tool_name="get_climate_settings",
            content={
                "status": "SUCCESS",
                "result": {"fan_speed": 0, "air_conditioning": False},
            },
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_vehicle_window_positions", "arguments": {}},
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-windows",
            tool_name="get_vehicle_window_positions",
            content={
                "status": "SUCCESS",
                "result": {
                    "window_driver_position": 25,
                    "window_passenger_position": 20,
                    "window_driver_rear_position": 20,
                    "window_passenger_rear_position": 30,
                },
            },
        )
    )
    expected_steps = (
        ("open_close_window", {"window": "DRIVER", "percentage": 0}),
        ("open_close_window", {"window": "PASSENGER_REAR", "percentage": 0}),
        ("set_fan_speed", {"level": 1}),
        ("set_air_conditioning", {"on": True}),
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    assert [call.arguments for call in session.execution_bundle.calls] == [
        arguments for _, arguments in expected_steps
    ]
    for index, (tool_name, arguments) in enumerate(expected_steps):
        assert outbound.tool_calls == (
            {"tool_name": tool_name, "arguments": arguments},
        )
        outbound = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"{context_id}-set-{index}",
                tool_name=tool_name,
                content={"status": "SUCCESS", "result": arguments},
            )
        )
    assert outbound.tool_calls == ()
    assert outbound.text is not None and outbound.text.startswith("Done")
    assert client.action_calls == 1


def test_base94_missing_required_window_enum_blocks_before_any_set() -> None:
    client = ac_client()
    runtime = runtime_for(client)
    context_id = "base94-missing-passenger-rear-enum"
    missing_enum_tools = tuple(
        spec for spec in AC_TOOLS if spec["function"]["name"] != "open_close_window"
    ) + (
        tool(
            "open_close_window",
            {
                "window": {
                    "type": "string",
                    "enum": ["ALL", "DRIVER", "PASSENGER", "DRIVER_REAR"],
                },
                "percentage": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            ["window", "percentage"],
        ),
    )
    outbound = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="Turn on the air conditioning.",
            tools=missing_enum_tools,
        )
    )
    assert outbound.tool_calls
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-climate",
            tool_name="get_climate_settings",
            content={"status": "SUCCESS", "result": {"fan_speed": 0}},
        )
    )
    assert outbound.tool_calls
    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-windows",
            tool_name="get_vehicle_window_positions",
            content={
                "status": "SUCCESS",
                "result": {
                    "window_driver_position": 25,
                    "window_passenger_position": 20,
                    "window_driver_rear_position": 20,
                    "window_passenger_rear_position": 30,
                },
            },
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_defrost_policy_preview_closes_all_required_reads_before_frozen_sets() -> None:
    client = defrost_client()
    runtime = runtime_for(client)
    context_id = "defrost-policy-preview"

    climate = runtime.handle_event(
        InboundEvent(
            message_id="defrost-user",
            context_id=context_id,
            system_policy=DEFROST_POLICY,
            user_text="Turn on the front window defrost.",
            live_tools=DEFROST_TOOLS,
        )
    )
    assert climate.tool_calls == (
        {"tool_name": "get_climate_settings", "arguments": {}},
    )
    assert client.action_calls == 0
    windows = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="defrost-climate-result",
            tool_name="get_climate_settings",
            content={
                "status": "SUCCESS",
                "result": {
                    "fan_speed": 0,
                    "fan_airflow_direction": "FEET",
                    "air_conditioning": False,
                },
            },
        )
    )
    assert windows.tool_calls == (
        {"tool_name": "get_vehicle_window_positions", "arguments": {}},
    )
    assert client.action_calls == 0
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="defrost-window-result",
            tool_name="get_vehicle_window_positions",
            content={
                "status": "SUCCESS",
                "result": {
                    "window_driver_position": 30,
                    "window_passenger_position": 40,
                    "window_driver_rear_position": 0,
                    "window_passenger_rear_position": 0,
                },
            },
        )
    )

    steps = (
        (
            "open_close_window",
            {"window": "DRIVER", "percentage": 0},
            {"window": "DRIVER", "percentage": 0},
        ),
        (
            "open_close_window",
            {"window": "PASSENGER", "percentage": 0},
            {"window": "PASSENGER", "percentage": 0},
        ),
        ("set_fan_speed", {"level": 2}, {"level": 2}),
        (
            "set_fan_airflow_direction",
            {"direction": "WINDSHIELD"},
            {"direction": "WINDSHIELD"},
        ),
        ("set_air_conditioning", {"on": True}, {"on": True}),
        (
            "set_window_defrost",
            {"on": True, "defrost_window": "FRONT"},
            {"on": True, "defrost_window": "FRONT"},
        ),
    )
    for index, (tool_name, arguments, result) in enumerate(steps):
        assert outbound.tool_calls == (
            {"tool_name": tool_name, "arguments": arguments},
        )
        session = runtime.sessions.get(context_id)
        assert session is not None
        assert session.evidence.current_state_version == 0
        outbound = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"defrost-set-{index}",
                tool_name=tool_name,
                content={"status": "SUCCESS", "result": result},
            )
        )

    assert outbound.tool_calls == ()
    assert outbound.text is not None and outbound.text.startswith("Done")
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.evidence.current_state_version == 1
    assert sum(session.budget.get_counts.values()) == 2
    assert client.action_calls == 1


def test_all_windows_then_unspecified_defrost_preserves_both_goals() -> None:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_window_position",
                    "desired_outcome": {"window": "all", "percentage": 0},
                },
                {
                    "semantic_operation": "set_window_defrost",
                    "desired_outcome": {"enabled": True, "window": "all"},
                    "depends_on_indices": [0],
                },
            ],
            explicit_slots={"window": "all", "percentage": 0, "enabled": True},
        ),
        lambda payload: pytest.fail(f"planner must not return a proposal: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "all-windows-then-defrost"

    outbound = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id="windows-defrost-user",
            system_policy=DEFROST_POLICY,
            user_text="Close all windows. Then turn on the defrost.",
            live_tools=FULL_DEFROST_TOOLS,
        )
    )
    assert outbound.tool_calls == (
        {
            "tool_name": "open_close_window",
            "arguments": {"window": "ALL", "percentage": 0},
        },
    )

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="all-windows-result",
            tool_name="open_close_window",
            content={
                "status": "SUCCESS",
                "result": {"window": "ALL", "percentage": 0},
            },
        )
    )
    assert outbound.tool_calls == ()
    assert outbound.text is not None and "front" in outbound.text

    outbound = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id="front-selection",
            user_text="Defrost the front window.",
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_climate_settings", "arguments": {}},
    )

    results = {
        "get_climate_settings": {
            "fan_speed": 0,
            "fan_airflow_direction": "WINDSHIELD_HEAD_FEET",
            "air_conditioning": False,
        },
        "get_vehicle_window_positions": {
            "window_driver_position": 0,
            "window_passenger_position": 0,
            "window_driver_rear_position": 0,
            "window_passenger_rear_position": 0,
        },
        "set_fan_speed": {"level": 2},
        "set_air_conditioning": {"on": True},
        "set_window_defrost": {"on": True, "defrost_window": "FRONT"},
    }
    observed: list[str] = []
    while outbound.tool_calls:
        tool_name = outbound.tool_calls[0]["tool_name"]
        observed.append(tool_name)
        outbound = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"defrost-result-{len(observed)}",
                tool_name=tool_name,
                content={"status": "SUCCESS", "result": results[tool_name]},
            )
        )

    assert observed == [
        "get_climate_settings",
        "get_vehicle_window_positions",
        "set_fan_speed",
        "set_air_conditioning",
        "set_window_defrost",
    ]
    assert outbound.text is not None and outbound.text.startswith("Done")
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert [goal.status.value for goal in session.goal_dag.goals] == ["done", "done"]


def test_planner_failure_serializes_independent_overlapping_goals() -> None:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_window_position",
                    "desired_outcome": {"window": "all", "percentage": 0},
                },
                {
                    "semantic_operation": "set_window_defrost",
                    "desired_outcome": {"enabled": True, "window": "all"},
                },
            ],
            explicit_slots={
                "window": "all",
                "percentage": 0,
                "enabled": True,
            },
        ),
        lambda payload: pytest.fail(f"planner must not return a proposal: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "serial-overlapping-goals"
    outbound = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id="serial-overlapping-user",
            system_policy=DEFROST_POLICY,
            user_text="Close all windows and turn on the defrost.",
            live_tools=FULL_DEFROST_TOOLS,
        )
    )
    assert outbound.tool_calls == ()
    assert outbound.text is not None and "front" in outbound.text.casefold()
    outbound = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id="serial-overlapping-front",
            user_text="Defrost the front window.",
        )
    )

    observed: list[tuple[str, dict[str, Any]]] = []
    window_reads = 0
    for index in range(12):
        if not outbound.tool_calls:
            break
        call = outbound.tool_calls[0]
        tool_name = call["tool_name"]
        arguments = call["arguments"]
        observed.append((tool_name, arguments))
        if tool_name == "get_climate_settings":
            result = {
                "fan_speed": 0,
                "fan_airflow_direction": "WINDSHIELD",
                "air_conditioning": False,
            }
        elif tool_name == "get_vehicle_window_positions":
            window_reads += 1
            result = (
                {
                    "window_driver_position": 100,
                    "window_passenger_position": 25,
                    "window_driver_rear_position": 0,
                    "window_passenger_rear_position": 25,
                }
                if window_reads == 1
                else {
                    "window_driver_position": 0,
                    "window_passenger_position": 0,
                    "window_driver_rear_position": 0,
                    "window_passenger_rear_position": 0,
                }
            )
        else:
            result = arguments
        outbound = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"serial-overlapping-result-{index}",
                tool_name=tool_name,
                content={"status": "SUCCESS", "result": result},
            )
        )

    assert observed == [
        ("get_climate_settings", {}),
        ("get_vehicle_window_positions", {}),
        ("open_close_window", {"window": "ALL", "percentage": 0}),
        ("get_climate_settings", {}),
        ("get_vehicle_window_positions", {}),
        ("set_fan_speed", {"level": 2}),
        ("set_air_conditioning", {"on": True}),
        ("set_window_defrost", {"on": True, "defrost_window": "FRONT"}),
    ]
    assert outbound.text is not None and outbound.text.startswith("Done")
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert [goal.status.value for goal in session.goal_dag.goals] == ["done", "done"]
    assert client.action_calls == 2


def test_unique_bound_action_survives_structured_planner_failure() -> None:
    healthy = defrost_client()
    client = FailingActionClient(
        healthy.intent.model_dump(mode="python"), healthy.action_factory
    )
    runtime = runtime_for(client)
    context_id = "defrost-planner-fallback"

    climate = runtime.handle_event(
        InboundEvent(
            message_id="defrost-fallback-user",
            context_id=context_id,
            system_policy=DEFROST_POLICY,
            user_text="Turn on the front window defrost.",
            live_tools=DEFROST_TOOLS,
        )
    )
    assert climate.tool_calls == (
        {"tool_name": "get_climate_settings", "arguments": {}},
    )
    windows = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="defrost-fallback-climate",
            tool_name="get_climate_settings",
            content={
                "status": "SUCCESS",
                "result": {
                    "fan_speed": 0,
                    "fan_airflow_direction": "WINDSHIELD",
                    "air_conditioning": False,
                },
            },
        )
    )
    assert windows.tool_calls == (
        {"tool_name": "get_vehicle_window_positions", "arguments": {}},
    )

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="defrost-fallback-windows",
            tool_name="get_vehicle_window_positions",
            content={
                "status": "SUCCESS",
                "result": {
                    "window_driver_position": 50,
                    "window_passenger_position": 50,
                    "window_driver_rear_position": 25,
                    "window_passenger_rear_position": 100,
                },
            },
        )
    )

    assert outbound.text is None
    assert outbound.tool_calls
    assert outbound.tool_calls[0]["tool_name"] in {
        "open_close_window",
        "set_fan_speed",
        "set_air_conditioning",
        "set_window_defrost",
    }
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    assert session.execution_bundle.calls[-1].tool_name == "set_window_defrost"
    assert client.action_calls == 1


def test_planner_failure_freezes_all_uniquely_bound_goals_serially() -> None:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_speed",
                    "desired_outcome": {"level": 2},
                },
                {
                    "semantic_operation": "set_ambient_lights",
                    "desired_outcome": {"color": "PURPLE", "enabled": True},
                },
            ],
            explicit_slots={"level": 2, "color": "PURPLE", "enabled": True},
        ),
        lambda payload: pytest.fail("the failing planner must not return a proposal"),
    )
    runtime = runtime_for(client)

    outbound = runtime.handle_event(
        user_event(
            context_id="multi-goal-planner-failure",
            message_id="multi-goal-planner-failure-user",
            text="Set the fan speed to level 2 and change the ambient lights to purple.",
            tools=(FAN_TOOL, AMBIENT_LIGHT_TOOL),
        )
    )

    assert outbound.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 2}},
    )
    session = runtime.sessions.get("multi-goal-planner-failure")
    assert session is not None
    assert len(session.goal_dag.ready_goals) == 2
    assert session.execution_bundle is not None
    assert [call.tool_name for call in session.execution_bundle.calls] == [
        "set_fan_speed",
        "set_ambient_lights",
    ]
    assert len(set(session.execution_bundle.call_goal_ids)) == 2
    assert client.action_calls == 1

    second = runtime.handle_event(
        result_event(
            context_id="multi-goal-planner-failure",
            message_id="multi-goal-fan-result",
            tool_name="set_fan_speed",
            content={"status": "SUCCESS", "result": {"level": 2}},
        )
    )
    assert second.tool_calls == (
        {
            "tool_name": "set_ambient_lights",
            "arguments": {"on": True, "lightcolor": "PURPLE"},
        },
    )

    final = runtime.handle_event(
        result_event(
            context_id="multi-goal-planner-failure",
            message_id="multi-goal-ambient-result",
            tool_name="set_ambient_lights",
            content={
                "status": "SUCCESS",
                "result": {"on": True, "lightcolor": "PURPLE"},
            },
        )
    )
    assert final.tool_calls == ()
    assert final.text is not None and final.text.startswith("Done")
    assert [goal.status.value for goal in session.goal_dag.goals] == ["done", "done"]


def test_multi_goal_ac_fallback_preserves_policy_aware_user_order() -> None:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_air_conditioning",
                    "desired_outcome": {"enabled": True},
                },
                {
                    "semantic_operation": "set_window_position",
                    "desired_outcome": {"window": "ALL", "percentage": 0},
                },
                {
                    "semantic_operation": "set_fan_speed",
                    "desired_outcome": {"level": 3},
                },
            ],
            explicit_slots={
                "enabled": True,
                "window": "ALL",
                "percentage": 0,
                "level": 3,
            },
        ),
        lambda payload: pytest.fail("the failing planner must not return a proposal"),
    )
    runtime = runtime_for(client)
    context_id = "multi-goal-ac-fallback"
    outbound = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id="multi-goal-ac-user",
            system_policy=DEFROST_POLICY,
            user_text=(
                "Close all windows completely, turn on the air conditioning, and "
                "set the fan speed to level 3."
            ),
            live_tools=AC_TOOLS,
        )
    )

    read_results = {
        "get_climate_settings": {
            "status": "SUCCESS",
            "result": {"fan_speed": 0, "air_conditioning": False},
        },
        "get_vehicle_window_positions": {
            "status": "SUCCESS",
            "result": {
                "window_driver_position": 25,
                "window_passenger_position": 0,
                "window_driver_rear_position": 25,
                "window_passenger_rear_position": 0,
            },
        },
    }
    read_count = 0
    while outbound.tool_calls and outbound.tool_calls[0]["tool_name"].startswith(
        "get_"
    ):
        read_count += 1
        assert read_count <= 4
        tool_name = outbound.tool_calls[0]["tool_name"]
        outbound = runtime.handle_event(
            InboundEvent(
                context_id=context_id,
                message_id=f"multi-goal-ac-read-{read_count}",
                tool_results=(
                    {
                        "toolCallId": "reused-evaluator-id",
                        "toolName": tool_name,
                        "content": read_results[tool_name],
                    },
                ),
            )
        )

    assert outbound.tool_calls == (
        {
            "tool_name": "open_close_window",
            "arguments": {"window": "ALL", "percentage": 0},
        },
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    assert [call.tool_name for call in session.execution_bundle.calls] == [
        "open_close_window",
        "set_air_conditioning",
        "set_fan_speed",
    ]
    assert [call.arguments for call in session.execution_bundle.calls] == [
        {"window": "ALL", "percentage": 0},
        {"on": True},
        {"level": 3},
    ]

    set_results = {
        "open_close_window": {"window": "ALL", "percentage": 0},
        "set_air_conditioning": {"on": True},
        "set_fan_speed": {"level": 3},
    }
    set_count = 0
    while outbound.tool_calls:
        set_count += 1
        assert set_count <= 3
        tool_name = outbound.tool_calls[0]["tool_name"]
        outbound = runtime.handle_event(
            InboundEvent(
                context_id=context_id,
                message_id=f"multi-goal-ac-set-{set_count}",
                tool_results=(
                    {
                        "toolCallId": "reused-evaluator-id",
                        "toolName": tool_name,
                        "content": {
                            "status": "SUCCESS",
                            "result": set_results[tool_name],
                        },
                    },
                ),
            )
        )

    assert set_count == 3
    assert outbound.text is not None and outbound.text.startswith("Done")
    assert [goal.status.value for goal in session.goal_dag.goals] == [
        "done",
        "done",
        "done",
    ]
    assert client.action_calls == 1


def test_relative_fan_delta_uses_current_evidence_and_freezes_all_user_goals() -> None:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_airflow_direction",
                    "desired_outcome": {"direction": "FEET"},
                }
            ],
            explicit_slots={"direction": "FEET"},
        ),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    controls = (
        tool("get_climate_settings"),
        tool(
            "set_fan_airflow_direction",
            {"direction": {"type": "string", "enum": ["HEAD", "FEET"]}},
            ["direction"],
        ),
        FAN_TOOL,
    )
    context_id = "relative-fan-base-28"

    outbound = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="relative-fan-user",
            text=(
                "Could you change the fan so the air blows towards my feet? "
                "And could you bump up the fan speed a little bit, just one level?"
            ),
            tools=controls,
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_climate_settings", "arguments": {}},
    )

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="relative-fan-read",
            tool_name="get_climate_settings",
            content={
                "status": "SUCCESS",
                "result": {"fan_speed": 0, "fan_airflow_direction": "HEAD"},
            },
        )
    )
    assert outbound.tool_calls == (
        {
            "tool_name": "set_fan_airflow_direction",
            "arguments": {"direction": "FEET"},
        },
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    assert [call.tool_name for call in session.execution_bundle.calls] == [
        "set_fan_airflow_direction",
        "set_fan_speed",
    ]
    assert session.execution_bundle.calls[1].arguments == {"level": 1}

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="relative-fan-airflow",
            tool_name="set_fan_airflow_direction",
            content={"status": "SUCCESS", "result": {"direction": "FEET"}},
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 1}},
    )

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="relative-fan-speed",
            tool_name="set_fan_speed",
            content={"status": "SUCCESS", "result": {"level": 1}},
        )
    )
    assert outbound.tool_calls == ()
    assert outbound.text is not None and outbound.text.startswith("Done")
    assert [goal.status.value for goal in session.goal_dag.goals] == ["done", "done"]
    assert client.action_calls == 0


def test_relative_whole_car_temperature_reads_before_one_frozen_ac_bundle() -> None:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_air_conditioning",
                    "desired_outcome": {"enabled": True},
                },
                {
                    "semantic_operation": "set_climate_temperature",
                    "desired_outcome": {},
                },
            ],
            explicit_slots={"enabled": True},
        ),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "relative-temperature-base-40"

    outbound = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="relative-temperature-user",
            text=(
                "Can you turn on the AC for me and maybe drop the temperature by "
                "like, 4 degrees for the whole car?"
            ),
            tools=BASE_40_TOOLS,
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_temperature_inside_car", "arguments": {}},
    )

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="relative-temperature-read",
            tool_name="get_temperature_inside_car",
            content={
                "status": "SUCCESS",
                "result": {
                    "climate_temperature_driver": 26.0,
                    "climate_temperature_passenger": 26.0,
                    "temperature_unit": "Celsius",
                },
            },
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_climate_settings", "arguments": {}},
    )

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="relative-temperature-climate",
            tool_name="get_climate_settings",
            content={
                "status": "SUCCESS",
                "result": {"fan_speed": 0, "air_conditioning": False},
            },
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_vehicle_window_positions", "arguments": {}},
    )

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="relative-temperature-windows",
            tool_name="get_vehicle_window_positions",
            content={
                "status": "SUCCESS",
                "result": {
                    "window_driver_position": 100,
                    "window_passenger_position": 100,
                    "window_driver_rear_position": 0,
                    "window_passenger_rear_position": 25,
                },
            },
        )
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    assert [call.tool_name for call in session.execution_bundle.calls] == [
        "open_close_window",
        "open_close_window",
        "open_close_window",
        "set_fan_speed",
        "set_air_conditioning",
        "set_climate_temperature",
    ]
    assert session.execution_bundle.calls[-1].arguments == {
        "temperature": 22,
        "seat_zone": "ALL_ZONES",
    }
    assert outbound.tool_calls == (
        {
            "tool_name": "open_close_window",
            "arguments": {"window": "DRIVER", "percentage": 0},
        },
    )

    seen: list[str] = []
    while outbound.tool_calls:
        tool_name = outbound.tool_calls[0]["tool_name"]
        arguments = outbound.tool_calls[0]["arguments"]
        seen.append(tool_name)
        outbound = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"relative-temperature-set-{len(seen)}",
                tool_name=tool_name,
                content={"status": "SUCCESS", "result": arguments},
            )
        )
    assert seen == [
        "open_close_window",
        "open_close_window",
        "open_close_window",
        "set_fan_speed",
        "set_air_conditioning",
        "set_climate_temperature",
    ]
    assert outbound.text is not None and "22 degrees" in outbound.text
    assert [goal.status.value for goal in session.goal_dag.goals] == ["done", "done"]
    assert session.evidence.current_state_version == 1
    assert client.action_calls == 0


def _single_zone_temperature_runtime(
    target_temperature: int,
) -> CARGuardOrchestrator:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_climate_temperature",
                    "desired_outcome": {
                        "temperature": target_temperature,
                        "seat_zone": "DRIVER",
                    },
                }
            ],
            explicit_slots={
                "temperature": target_temperature,
                "seat_zone": "DRIVER",
            },
        ),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    return runtime_for(client)


def _single_zone_temperature_user_event(
    *, context_id: str, target_temperature: int
) -> InboundEvent:
    return InboundEvent(
        message_id=f"{context_id}-user",
        context_id=context_id,
        system_policy=(
            "AUT-POL:012: Inform the user when a single seat zone temperature "
            "will differ from the other zone by more than 3 degrees Celsius."
        ),
        user_text=(
            f"Set the climate temperature to {target_temperature} degrees for "
            "the driver zone."
        ),
        live_tools=(tool("get_temperature_inside_car"), CLIMATE_TEMPERATURE_TOOL),
    )


@pytest.mark.parametrize(
    ("target_temperature", "expected_difference"),
    [(22, "5"), (24, "7")],
)
def test_base52_success_reports_single_zone_temperature_difference(
    target_temperature: int, expected_difference: str
) -> None:
    context_id = f"base52-pol012-{target_temperature}"
    runtime = _single_zone_temperature_runtime(target_temperature)

    outbound = runtime.handle_event(
        _single_zone_temperature_user_event(
            context_id=context_id, target_temperature=target_temperature
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_temperature_inside_car", "arguments": {}},
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-read",
            tool_name="get_temperature_inside_car",
            content={
                "status": "SUCCESS",
                "result": {
                    "climate_temperature_driver": 25.0,
                    "climate_temperature_passenger": 17.0,
                    "temperature_unit": "Celsius",
                },
            },
        )
    )
    assert outbound.tool_calls == (
        {
            "tool_name": "set_climate_temperature",
            "arguments": {
                "temperature": target_temperature,
                "seat_zone": "DRIVER",
            },
        },
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-set",
            tool_name="set_climate_temperature",
            content={
                "status": "SUCCESS",
                "result": {
                    "temperature": target_temperature,
                    "seat_zone": "DRIVER",
                },
            },
        )
    )

    assert outbound.tool_calls == ()
    assert outbound.text is not None
    assert (
        f"temperature difference is {expected_difference} degrees Celsius"
        in outbound.text
    )
    assert "passenger zone remains at 17 degrees Celsius" in outbound.text


def test_single_zone_temperature_difference_at_threshold_is_not_reported() -> None:
    context_id = "base52-pol012-threshold"
    runtime = _single_zone_temperature_runtime(20)
    outbound = runtime.handle_event(
        _single_zone_temperature_user_event(
            context_id=context_id, target_temperature=20
        )
    )
    assert outbound.tool_calls[0]["tool_name"] == "get_temperature_inside_car"
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-read",
            tool_name="get_temperature_inside_car",
            content={
                "status": "SUCCESS",
                "result": {
                    "climate_temperature_driver": 25.0,
                    "climate_temperature_passenger": 17.0,
                    "temperature_unit": "Celsius",
                },
            },
        )
    )
    assert outbound.tool_calls[0]["tool_name"] == "set_climate_temperature"
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-set",
            tool_name="set_climate_temperature",
            content={
                "status": "SUCCESS",
                "result": {"temperature": 20, "seat_zone": "DRIVER"},
            },
        )
    )

    assert outbound.text is not None and "Done." in outbound.text
    assert "temperature difference" not in outbound.text
    assert "passenger zone" not in outbound.text


def test_failed_single_zone_temperature_set_does_not_report_difference() -> None:
    context_id = "base52-pol012-set-failure"
    runtime = _single_zone_temperature_runtime(22)
    outbound = runtime.handle_event(
        _single_zone_temperature_user_event(
            context_id=context_id, target_temperature=22
        )
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-read",
            tool_name="get_temperature_inside_car",
            content={
                "status": "SUCCESS",
                "result": {
                    "climate_temperature_driver": 25.0,
                    "climate_temperature_passenger": 17.0,
                    "temperature_unit": "Celsius",
                },
            },
        )
    )
    assert outbound.tool_calls[0]["tool_name"] == "set_climate_temperature"
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-set",
            tool_name="set_climate_temperature",
            content={"status": "FAILURE", "errors": {"SET_001": "rejected"}},
        )
    )

    assert outbound.text is not None and "haven't performed" in outbound.text
    assert "temperature difference" not in outbound.text
    assert "passenger zone" not in outbound.text


def test_missing_other_zone_temperature_never_fabricates_difference() -> None:
    context_id = "base52-pol012-missing-passenger"
    runtime = _single_zone_temperature_runtime(22)
    outbound = runtime.handle_event(
        _single_zone_temperature_user_event(
            context_id=context_id, target_temperature=22
        )
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-read",
            tool_name="get_temperature_inside_car",
            content={
                "status": "SUCCESS",
                "result": {
                    "climate_temperature_driver": 25.0,
                    "temperature_unit": "Celsius",
                },
            },
        )
    )

    assert not any(
        call["tool_name"] == "set_climate_temperature" for call in outbound.tool_calls
    )
    assert outbound.text is None or "temperature difference" not in outbound.text
    assert outbound.text is None or "passenger zone" not in outbound.text


@pytest.mark.parametrize(
    "temperature_result",
    [
        {
            "climate_temperature_driver": 26.0,
            "climate_temperature_passenger": 24.0,
            "temperature_unit": "Celsius",
        },
        {
            "climate_temperature_driver": 26.0,
            "climate_temperature_passenger": 26.0,
        },
        {
            "climate_temperature_driver": 26.0,
            "climate_temperature_passenger": 26.0,
            "temperature_unit": "Fahrenheit",
        },
    ],
    ids=["different-zone-settings", "missing-unit", "wrong-unit"],
)
def test_relative_whole_car_temperature_fails_closed_on_invalid_snapshot(
    temperature_result: dict[str, Any],
) -> None:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_climate_temperature",
                    "desired_outcome": {"seat_zone": "whole_car"},
                }
            ],
            explicit_slots={"seat_zone": "whole_car"},
        ),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "relative-temperature-invalid-snapshot"
    outbound = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="relative-temperature-invalid-user",
            text="Lower the temperature by 4 degrees for the whole car.",
            tools=(tool("get_temperature_inside_car"), CLIMATE_TEMPERATURE_TOOL),
        )
    )
    assert outbound.tool_calls[0]["tool_name"] == "get_temperature_inside_car"

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="relative-temperature-invalid-result",
            tool_name="get_temperature_inside_car",
            content={"status": "SUCCESS", "result": temperature_result},
        )
    )
    assert outbound.tool_calls == ()
    assert outbound.text is not None and "haven't performed" in outbound.text
    assert client.action_calls == 0


def test_relative_temperature_compound_request_fails_closed_without_window_control() -> (
    None
):
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_air_conditioning",
                    "desired_outcome": {"enabled": True},
                },
                {
                    "semantic_operation": "set_climate_temperature",
                    "desired_outcome": {},
                },
            ],
            explicit_slots={"enabled": True},
        ),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "relative-temperature-hallucination-38"
    controls = tuple(
        definition
        for definition in BASE_40_TOOLS
        if definition["function"]["name"] != "open_close_window"
    )
    outbound = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="relative-temperature-missing-window-user",
            text=(
                "Can you turn on the AC for me and maybe drop the temperature by "
                "like, 4 degrees for the whole car?"
            ),
            tools=controls,
        )
    )
    reads = {
        "get_temperature_inside_car": {
            "climate_temperature_driver": 26.0,
            "climate_temperature_passenger": 26.0,
            "temperature_unit": "Celsius",
        },
        "get_climate_settings": {
            "fan_speed": 0,
            "air_conditioning": False,
        },
        "get_vehicle_window_positions": {
            "window_driver_position": 100,
            "window_passenger_position": 100,
            "window_driver_rear_position": 0,
            "window_passenger_rear_position": 25,
        },
    }
    seen: list[str] = []
    while outbound.tool_calls:
        tool_name = outbound.tool_calls[0]["tool_name"]
        assert tool_name in reads
        seen.append(tool_name)
        outbound = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"relative-temperature-missing-window-read-{len(seen)}",
                tool_name=tool_name,
                content={"status": "SUCCESS", "result": reads[tool_name]},
            )
        )

    assert seen == [
        "get_temperature_inside_car",
        "get_climate_settings",
        "get_vehicle_window_positions",
    ]
    assert outbound.text is not None and "did not change" in outbound.text
    assert "open_close_window" not in outbound.text
    assert "26 to 22 degrees" in outbound.text
    assert "close the windows manually" in outbound.text
    assert client.action_calls == 0


@pytest.mark.parametrize(
    ("utterance", "current", "expected"),
    [
        ("Please increase the fan speed by one level.", 2, 3),
        ("Please increase the fan speed level by two.", 0, 2),
    ],
)
def test_relative_fan_delta_is_not_treated_as_an_absolute_level(
    utterance: str, current: int, expected: int
) -> None:
    client = FailingActionClient(
        action_intent([], explicit_slots={}),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = f"relative-fan-{current}-{expected}"
    outbound = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="relative-user",
            text=utterance,
            tools=(tool("get_climate_settings"), FAN_TOOL),
        )
    )
    assert outbound.tool_calls[0]["tool_name"] == "get_climate_settings"

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="relative-read",
            tool_name="get_climate_settings",
            content={"status": "SUCCESS", "result": {"fan_speed": current}},
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": expected}},
    )


def test_relative_fan_delta_does_not_clamp_past_the_live_limit() -> None:
    client = FailingActionClient(
        action_intent([], explicit_slots={}),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "relative-fan-out-of-range"
    outbound = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="relative-limit-user",
            text="Please increase the fan speed by one level.",
            tools=(tool("get_climate_settings"), FAN_TOOL),
        )
    )
    assert outbound.tool_calls[0]["tool_name"] == "get_climate_settings"

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="relative-limit-read",
            tool_name="get_climate_settings",
            content={"status": "SUCCESS", "result": {"fan_speed": 5}},
        )
    )
    assert outbound.tool_calls == ()
    assert outbound.text is not None and "haven't performed" in outbound.text


def test_completion_followup_does_not_replay_a_different_fan_operation() -> None:
    def airflow_action(payload: dict[str, Any]) -> dict[str, Any]:
        return set_proposal(
            payload,
            semantic_operation="set_fan_airflow_direction",
            tool_name="set_fan_airflow_direction",
            arguments={"direction": "FEET"},
        )

    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_airflow_direction",
                    "desired_outcome": {"direction": "FEET"},
                }
            ],
            explicit_slots={"direction": "FEET"},
        ),
        airflow_action,
    )
    runtime = runtime_for(client)
    context_id = "fan-operation-followup"
    first = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="airflow-user",
            text="Set the fan airflow direction to feet.",
            tools=(
                tool(
                    "set_fan_airflow_direction",
                    {"direction": {"type": "string", "enum": ["HEAD", "FEET"]}},
                    ["direction"],
                ),
            ),
        )
    )
    assert first.tool_calls[0]["tool_name"] == "set_fan_airflow_direction"
    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="airflow-result",
            tool_name="set_fan_airflow_direction",
            content={"status": "SUCCESS", "result": {"direction": "FEET"}},
        )
    )
    assert completed.text is not None and completed.text.startswith("Done")

    followup = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id="fan-speed-followup",
            user_text="Did you increase the fan speed?",
        )
    )

    assert followup.tool_calls == ()
    assert followup.text != completed.text
    assert client.intent_calls == 2


def test_policy_validated_bound_action_survives_structured_critic_failure() -> None:
    healthy = defrost_client()
    client = FailingCriticClient(
        healthy.intent.model_dump(mode="python"), healthy.action_factory
    )
    runtime = runtime_for(client, enable_critic=True)
    context_id = "defrost-critic-fallback"

    climate = runtime.handle_event(
        InboundEvent(
            message_id="defrost-critic-user",
            context_id=context_id,
            system_policy=DEFROST_POLICY,
            user_text="Turn on the front window defrost.",
            live_tools=DEFROST_TOOLS,
        )
    )
    assert climate.tool_calls == (
        {"tool_name": "get_climate_settings", "arguments": {}},
    )
    windows = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="defrost-critic-climate",
            tool_name="get_climate_settings",
            content={
                "status": "SUCCESS",
                "result": {
                    "fan_speed": 0,
                    "fan_airflow_direction": "WINDSHIELD",
                    "air_conditioning": False,
                },
            },
        )
    )
    assert windows.tool_calls == (
        {"tool_name": "get_vehicle_window_positions", "arguments": {}},
    )

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="defrost-critic-windows",
            tool_name="get_vehicle_window_positions",
            content={
                "status": "SUCCESS",
                "result": {
                    "window_driver_position": 50,
                    "window_passenger_position": 50,
                    "window_driver_rear_position": 25,
                    "window_passenger_rear_position": 100,
                },
            },
        )
    )

    assert outbound.text is None
    assert outbound.tool_calls
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    assert session.execution_bundle.calls[-1].tool_name == "set_window_defrost"
    assert client.action_calls == 1
    assert client.critic_calls == 1


@pytest.mark.parametrize(
    ("decision", "reason_code"),
    [
        ("ask", "unresolved_ambiguity"),
        ("decline", "policy_risk"),
        ("replan_once", "plan_structure"),
    ],
)
def test_policy_validated_bound_action_survives_false_critic_veto(
    decision: str, reason_code: str
) -> None:
    client = NonAcceptingCriticClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_speed",
                    "desired_outcome": {"level": 2},
                }
            ],
            explicit_slots={"level": 2},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_fan_speed",
            tool_name="set_fan_speed",
            arguments={"level": 2},
        ),
        decision=decision,
        reason_code=reason_code,
    )
    runtime = runtime_for(client, enable_critic=True)

    outbound = runtime.handle_event(
        user_event(
            context_id="false-critic-decline",
            message_id="false-critic-decline-user",
            text="Set the fan to level two.",
            tools=(FAN_TOOL,),
        )
    )

    assert outbound.text is None
    assert outbound.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 2}},
    )
    session = runtime.sessions.get("false-critic-decline")
    assert session is not None
    assert len(session.pending_calls) == 1
    assert client.critic_calls == 1


def test_failed_bundle_step_invalidates_snapshot_after_prior_success() -> None:
    client = ac_client()
    runtime = runtime_for(client)
    context_id = "ac-prerequisite-failure"

    runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="ac-failure-user",
            text="Turn on the air conditioning.",
            tools=AC_TOOLS,
        )
    )
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="ac-failure-climate",
            tool_name="get_climate_settings",
            content={"status": "SUCCESS", "result": {"fan_speed": 0}},
        )
    )
    close_windows = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="ac-failure-windows",
            tool_name="get_vehicle_window_positions",
            content={
                "status": "SUCCESS",
                "result": {
                    "window_driver_position": 30,
                    "window_passenger_position": 0,
                    "window_driver_rear_position": 0,
                    "window_passenger_rear_position": 0,
                },
            },
        )
    )
    assert close_windows.tool_calls[0]["tool_name"] == "open_close_window"
    fan = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="ac-failure-close-result",
            tool_name="open_close_window",
            content={
                "status": "SUCCESS",
                "result": {"window": "DRIVER", "percentage": 0},
            },
        )
    )
    assert fan.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 1}},
    )

    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="ac-failure-fan-result",
            tool_name="set_fan_speed",
            content={"status": "FAILURE", "errors": {"reason": "unavailable"}},
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None and not blocked.text.startswith("Done")
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.execution_bundle is None
    assert session.evidence.current_state_version == 1


def test_runtime_exception_after_intermediate_set_aborts_frozen_snapshot() -> None:
    client = ac_client()
    runtime = runtime_for(client)
    context_id = "ac-prerequisite-exception"

    runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="ac-exception-user",
            text="Turn on the air conditioning.",
            tools=AC_TOOLS,
        )
    )
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="ac-exception-climate",
            tool_name="get_climate_settings",
            content={"status": "SUCCESS", "result": {"fan_speed": 0}},
        )
    )
    close_windows = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="ac-exception-windows",
            tool_name="get_vehicle_window_positions",
            content={
                "status": "SUCCESS",
                "result": {
                    "window_driver_position": 30,
                    "window_passenger_position": 0,
                    "window_driver_rear_position": 0,
                    "window_passenger_rear_position": 0,
                },
            },
        )
    )
    assert close_windows.tool_calls[0]["tool_name"] == "open_close_window"

    def fail_after_progress(session, bundle):
        del session, bundle
        raise RuntimeError("synthetic bundle failure")

    runtime._validate_bundle_step = fail_after_progress  # type: ignore[method-assign]
    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="ac-exception-close-result",
            tool_name="open_close_window",
            content={
                "status": "SUCCESS",
                "result": {"window": "ALL", "percentage": 0},
            },
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.execution_bundle is None
    assert session.pending_calls == []
    assert session.evidence.current_state_version == 1
    assert session.goal_dag.goals[0].status.value == "failed"


@pytest.mark.parametrize("result_kind", ["malformed", "unmatched"])
def test_unknown_set_result_aborts_pending_bundle(result_kind: str) -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_speed",
                    "desired_outcome": {"level": 2},
                }
            ],
            explicit_slots={"level": 2},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_fan_speed",
            tool_name="set_fan_speed",
            arguments={"level": 2},
        ),
    )
    runtime = runtime_for(client)
    context_id = f"unknown-set-{result_kind}"
    action = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{result_kind}-user",
            text="Set the fan to level two.",
            tools=(FAN_TOOL,),
        )
    )
    assert action.tool_calls[0]["tool_name"] == "set_fan_speed"
    event = (
        InboundEvent(
            context_id=context_id,
            message_id="malformed-set-result",
            malformed_parts=("part[0]: malformed tool result",),
        )
        if result_kind == "malformed"
        else result_event(
            context_id=context_id,
            message_id="unmatched-set-result",
            tool_name="set_air_conditioning",
            content={"status": "SUCCESS", "result": {"on": True}},
        )
    )

    blocked = runtime.handle_event(event)

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.execution_bundle is None
    assert session.pending_calls == []
    assert session.evidence.current_state_version == 1
    assert session.goal_dag.goals[0].status.value == "failed"


def test_policy_weather_read_is_deterministic_before_action_planning() -> None:
    client = sunroof_client()
    runtime = runtime_for(client)

    roof_state = runtime.handle_event(
        InboundEvent(
            message_id="sunroof-user",
            context_id="sunroof",
            system_policy=SUNROOF_POLICY,
            user_text="Set the sunroof to 50 percent.",
            live_tools=SUNROOF_TOOLS,
        )
    )
    assert roof_state.tool_calls == (
        {"tool_name": "get_sunroof_and_sunshade_position", "arguments": {}},
    )
    assert client.action_calls == 0

    weather = runtime.handle_event(
        result_event(
            context_id="sunroof",
            message_id="roof-state-result",
            tool_name="get_sunroof_and_sunshade_position",
            content={
                "status": "SUCCESS",
                "result": {"sunroof_position": 0, "sunshade_position": 100},
            },
        )
    )
    assert weather.tool_calls == (
        {
            "tool_name": "get_weather",
            "arguments": {
                "location_or_poi_id": "loc-home",
                "month": 7,
                "day": 11,
                "time_hour_24hformat": 16,
                "time_minutes": 15,
            },
        },
    )
    assert client.action_calls == 0

    action = runtime.handle_event(
        result_event(
            context_id="sunroof",
            message_id="weather-result",
            tool_name="get_weather",
            content={
                "status": "SUCCESS",
                "result": {"current_slot": {"condition": "sunny"}},
            },
        )
    )
    assert action.tool_calls == (
        {"tool_name": "open_close_sunroof", "arguments": {"percentage": 50}},
    )
    assert client.action_calls == 1

    final = runtime.handle_event(
        result_event(
            context_id="sunroof",
            message_id="sunroof-result",
            tool_name="open_close_sunroof",
            content={"status": "SUCCESS", "result": {"percentage": 50}},
        )
    )
    assert final.tool_calls == ()
    assert final.text is not None and final.text.startswith("Done")


def test_sunshade_match_uses_observed_sunroof_position() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_sunshade_position",
                    "desired_outcome": {},
                }
            ],
            explicit_slots={},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_sunshade_position",
            tool_name="open_close_sunshade",
            arguments={"percentage": 60},
        ),
    )
    runtime = runtime_for(client)
    context_id = "sunshade-match-sunroof"

    roof_state = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="sunshade-match-user",
            text="Adjust the sunshade to match the sunroof.",
            tools=(SUNROOF_TOOLS[0], SUNROOF_TOOLS[3]),
        )
    )

    assert roof_state.tool_calls == (
        {"tool_name": "get_sunroof_and_sunshade_position", "arguments": {}},
    )
    assert client.intent_calls == 0
    assert client.action_calls == 0

    action = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="sunshade-match-state",
            tool_name="get_sunroof_and_sunshade_position",
            content={
                "status": "SUCCESS",
                "result": {"sunroof_position": 60, "sunshade_position": 100},
            },
        )
    )

    assert action.tool_calls == (
        {"tool_name": "open_close_sunshade", "arguments": {"percentage": 60}},
    )
    assert client.intent_calls == 0
    assert client.action_calls == 0
    session = runtime.sessions.get(context_id)
    assert session is not None
    goal_id = session.goal_dag.goals[0].goal_id
    assert session.grounded_desired_values_by_goal[goal_id] == {}
    selected = [
        evidence
        for evidence in session.evidence.evidence.values()
        if evidence.proposition == f"selected_argument:{goal_id}:percentage"
    ]
    assert len(selected) == 1
    assert selected[0].value == 60
    assert selected[0].source_kind is EvidenceSourceKind.DERIVED
    assert selected[0].derived_from


def test_policy_confirmation_freezes_action_before_planner() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_sunroof_position",
                    "desired_outcome": {"percentage": 50},
                }
            ],
            explicit_slots={"percentage": 50},
        ),
        lambda payload: {
            "kind": "ask_user",
            "goal_ids": [goal_from_payload(payload, "set_sunroof_position")["goal_id"]],
            "user_text": "Model-owned confirmation wording must not be used.",
            "policy_rules_used": ["POL-008"],
        },
    )
    runtime = runtime_for(client)
    context_id = "planner-confirmation"

    runtime.handle_event(
        InboundEvent(
            message_id="planner-confirmation-user",
            context_id=context_id,
            system_policy=SUNROOF_POLICY,
            user_text="Set the sunroof to 50 percent.",
            live_tools=SUNROOF_TOOLS,
        )
    )
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="planner-confirmation-roof",
            tool_name="get_sunroof_and_sunshade_position",
            content={
                "status": "SUCCESS",
                "result": {"sunroof_position": 0, "sunshade_position": 100},
            },
        )
    )
    confirmation = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="planner-confirmation-weather",
            tool_name="get_weather",
            content={
                "status": "SUCCESS",
                "result": {"current_slot": {"condition": "rainy"}},
            },
        )
    )

    assert confirmation.tool_calls == ()
    assert confirmation.text is not None and "Shall I go ahead" in confirmation.text
    assert "Model-owned" not in confirmation.text
    assert client.action_calls == 0
    action = runtime.handle_event(
        InboundEvent(
            message_id="planner-confirmation-yes",
            context_id=context_id,
            user_text="Yes.",
        )
    )
    assert action.tool_calls == (
        {"tool_name": "open_close_sunroof", "arguments": {"percentage": 50}},
    )


def test_missing_sunroof_percentage_is_resolved_from_bound_preference_evidence() -> (
    None
):
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_sunroof_position",
                    "desired_outcome": {"percentage": 100},
                }
            ],
            explicit_slots={"percentage": 100},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_sunroof_position",
            tool_name="open_close_sunroof",
            arguments={"percentage": 50},
        ),
    )
    runtime = runtime_for(client)
    controls = (PREFERENCE_TOOL, *SUNROOF_TOOLS)

    preference = runtime.handle_event(
        InboundEvent(
            context_id="preference-sunroof",
            message_id="preference-sunroof-user",
            system_policy=SUNROOF_POLICY,
            user_text="Open the sunroof.",
            live_tools=controls,
        )
    )
    assert preference.tool_calls == (
        {
            "tool_name": "get_user_preferences",
            "arguments": {
                "preference_categories": {
                    "vehicle_settings": {"vehicle_settings": True}
                }
            },
        },
    )
    assert client.action_calls == 0

    roof_read = runtime.handle_event(
        result_event(
            context_id="preference-sunroof",
            message_id="preference-sunroof-result",
            tool_name="get_user_preferences",
            content={
                "status": "SUCCESS",
                "result": {
                    "vehicle_settings": {
                        "vehicle_settings": [
                            "Default value to open the sunroof is 50%, never "
                            "wants to open the sunroof fully"
                        ]
                    }
                },
            },
        )
    )

    assert roof_read.tool_calls == (
        {"tool_name": "get_sunroof_and_sunshade_position", "arguments": {}},
    )
    assert client.action_calls == 0

    weather_read = runtime.handle_event(
        result_event(
            context_id="preference-sunroof",
            message_id="preference-sunroof-state",
            tool_name="get_sunroof_and_sunshade_position",
            content={
                "status": "SUCCESS",
                "result": {"sunroof_position": 0, "sunshade_position": 100},
            },
        )
    )
    assert weather_read.tool_calls[0]["tool_name"] == "get_weather"
    assert client.action_calls == 0

    action = runtime.handle_event(
        result_event(
            context_id="preference-sunroof",
            message_id="preference-sunroof-weather",
            tool_name="get_weather",
            content={
                "status": "SUCCESS",
                "result": {"current_slot": {"condition": "sunny"}},
            },
        )
    )
    assert action.tool_calls == (
        {"tool_name": "open_close_sunroof", "arguments": {"percentage": 50}},
    )
    assert client.action_calls == 1
    session = runtime.sessions.get("preference-sunroof")
    assert session is not None
    goal_id = session.goal_dag.goals[0].goal_id
    assert session.grounded_desired_values_by_goal[goal_id] == {}
    assert session.grounded_value_sources_by_goal[goal_id] == {}
    preference_evidence = next(
        evidence
        for evidence in session.evidence.evidence.values()
        if evidence.proposition == "preferred_sunroof_percentage"
    )
    selected = next(
        evidence
        for evidence in session.evidence.evidence.values()
        if evidence.proposition.endswith(":percentage")
    )
    assert preference_evidence.evidence_id is not None
    assert selected.derived_from == [preference_evidence.evidence_id]


def test_missing_policy_read_capability_fails_before_action_planning() -> None:
    client = sunroof_client()
    runtime = runtime_for(client)
    controls = tuple(
        control
        for control in SUNROOF_TOOLS
        if control["function"]["name"] != "get_weather"
    )

    runtime.handle_event(
        InboundEvent(
            message_id="missing-weather-user",
            context_id="missing-weather",
            system_policy=SUNROOF_POLICY,
            user_text="Set the sunroof to 50 percent.",
            live_tools=controls,
        )
    )
    blocked = runtime.handle_event(
        result_event(
            context_id="missing-weather",
            message_id="missing-weather-roof-result",
            tool_name="get_sunroof_and_sunshade_position",
            content={
                "status": "SUCCESS",
                "result": {"sunroof_position": 0, "sunshade_position": 100},
            },
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "current weather condition information" in blocked.text.casefold()
    assert "get_weather" not in blocked.text
    assert client.action_calls == 0


def test_missing_triggered_prerequisite_fails_before_policy_read() -> None:
    client = sunroof_client()
    runtime = runtime_for(client)
    controls = tuple(
        control
        for control in SUNROOF_TOOLS
        if control["function"]["name"] != "open_close_sunshade"
    )

    first = runtime.handle_event(
        InboundEvent(
            message_id="missing-shade-user",
            context_id="missing-shade",
            system_policy=SUNROOF_POLICY,
            user_text="Set the sunroof to 50 percent.",
            live_tools=controls,
        )
    )
    assert first.tool_calls[0]["tool_name"] == "get_sunroof_and_sunshade_position"
    blocked = runtime.handle_event(
        result_event(
            context_id="missing-shade",
            message_id="missing-shade-roof-result",
            tool_name="get_sunroof_and_sunshade_position",
            content={
                "status": "SUCCESS",
                "result": {"sunroof_position": 0, "sunshade_position": 0},
            },
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "can't open the sunshade" in blocked.text
    assert "can't safely move the sunroof" in blocked.text
    assert "open_close_sunshade" not in blocked.text
    assert "open_sunshade_for_sunroof" not in blocked.text
    assert "POL-005" not in blocked.text
    assert "haven't performed either action" in blocked.text
    assert client.action_calls == 0
    session = runtime.sessions.get("missing-shade")
    assert session is not None
    assert sum(session.budget.get_counts.values()) == 1


def test_rainy_sunroof_bundle_reuses_frozen_evidence_after_prerequisite() -> None:
    client = sunroof_client()
    runtime = runtime_for(client)
    context_id = "rainy-sunroof"

    outbound = runtime.handle_event(
        InboundEvent(
            message_id="rainy-user",
            context_id=context_id,
            system_policy=SUNROOF_POLICY,
            user_text="Set the sunroof to 50 percent.",
            live_tools=SUNROOF_TOOLS,
        )
    )
    assert outbound.tool_calls[0]["tool_name"] == "get_sunroof_and_sunshade_position"
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="rainy-roof-1",
            tool_name="get_sunroof_and_sunshade_position",
            content={
                "status": "SUCCESS",
                "result": {"sunroof_position": 0, "sunshade_position": 0},
            },
        )
    )
    assert outbound.tool_calls[0]["tool_name"] == "get_weather"
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="rainy-weather-1",
            tool_name="get_weather",
            content={
                "status": "SUCCESS",
                "result": {"current_slot": {"condition": "rainy"}},
            },
        )
    )
    assert outbound.tool_calls == ()
    assert outbound.text is not None and "Shall I go ahead" in outbound.text

    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    first_confirmation_id = session.execution_bundle.confirmation_id
    outbound = runtime.handle_event(
        InboundEvent(
            message_id="rainy-confirmation-unclear",
            context_id=context_id,
            user_text="Maybe, I am still deciding.",
        )
    )
    assert outbound.tool_calls == ()
    assert outbound.text is not None and "Shall I go ahead" in outbound.text
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    assert session.execution_bundle.confirmation_id != first_confirmation_id

    outbound = runtime.handle_event(
        InboundEvent(
            message_id="rainy-confirmation",
            context_id=context_id,
            user_text=(
                "Yeah, go ahead and open the sunroof to 50%. And if the "
                "sunshade needs to open first, then open it all the way, 100%."
            ),
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "open_close_sunshade", "arguments": {"percentage": 100}},
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="rainy-shade-result",
            tool_name="open_close_sunshade",
            content={"status": "SUCCESS", "result": {"percentage": 100.0}},
        )
    )

    refresh_count = 0
    while outbound.tool_calls and outbound.tool_calls[0]["tool_name"].startswith(
        "get_"
    ):
        refresh_count += 1
        assert refresh_count <= 2
        tool_name = outbound.tool_calls[0]["tool_name"]
        content = (
            {
                "status": "SUCCESS",
                "result": {"sunroof_position": 0, "sunshade_position": 100},
            }
            if tool_name == "get_sunroof_and_sunshade_position"
            else {
                "status": "SUCCESS",
                "result": {"current_slot": {"condition": "rainy"}},
            }
        )
        outbound = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"rainy-refresh-{refresh_count}",
                tool_name=tool_name,
                content=content,
            )
        )

    assert refresh_count == 0
    assert outbound.tool_calls == (
        {"tool_name": "open_close_sunroof", "arguments": {"percentage": 50}},
    )
    assert client.action_calls == 0
    final = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="rainy-sunroof-result",
            tool_name="open_close_sunroof",
            content={"status": "SUCCESS", "result": {"percentage": 50.0}},
        )
    )
    assert final.tool_calls == ()
    assert final.text == (
        "Done. I opened the sunshade to 100 percent and opened the sunroof "
        "to 50 percent."
    )
    assert session.confirmation_latch.active == []

    followup = runtime.handle_event(
        InboundEvent(
            message_id="rainy-status-followup",
            context_id=context_id,
            user_text=(
                "So, was the sunroof opened then? I'm not sure if it actually opened."
            ),
        )
    )
    assert followup.tool_calls == ()
    assert followup.text == final.text
    assert client.intent_calls == 1


def test_multiple_set_goals_are_serial_and_complete_only_after_all_results() -> None:
    intent = action_intent(
        [
            {
                "semantic_operation": "set_fan_speed",
                "desired_outcome": {"level": 2},
            },
            {
                "semantic_operation": "set_air_circulation",
                "desired_outcome": {"mode": "RECIRCULATE"},
            },
        ],
        explicit_slots={"level": 2, "mode": "RECIRCULATE"},
    )

    def next_ready_goal(payload: dict[str, Any]) -> dict[str, Any]:
        goals = payload["semantic_goals"]["goals"]
        target = next(goal for goal in goals if goal["status"] != "done")
        if target["semantic_operation"] == "set_fan_speed":
            return set_proposal(
                payload,
                semantic_operation="set_fan_speed",
                tool_name="set_fan_speed",
                arguments={"level": 2},
            )
        return set_proposal(
            payload,
            semantic_operation="set_air_circulation",
            tool_name="set_air_circulation",
            arguments={"mode": "RECIRCULATE"},
        )

    client = FakeStructuredClient(intent, next_ready_goal)
    runtime = runtime_for(client)
    controls = (
        FAN_TOOL,
        tool(
            "set_air_circulation",
            {"mode": {"type": "string", "enum": ["RECIRCULATE"]}},
            ["mode"],
        ),
    )

    first = runtime.handle_event(
        user_event(
            context_id="multi",
            message_id="multi-user",
            text="Set fan level two and set air circulation to recirculate.",
            tools=controls,
        )
    )
    assert first.text is None
    assert len(first.tool_calls) == 1
    assert first.tool_calls[0]["tool_name"] == "set_fan_speed"

    second = runtime.handle_event(
        result_event(
            context_id="multi",
            message_id="fan-result",
            tool_name="set_fan_speed",
            content={"status": "SUCCESS", "result": {"level": 2}},
        )
    )
    assert second.text is None
    assert len(second.tool_calls) == 1
    assert second.tool_calls[0]["tool_name"] == "set_air_circulation"
    session = runtime.sessions.get("multi")
    assert session is not None
    assert [goal.status.value for goal in session.goal_dag.goals] == [
        "done",
        "pending",
    ]

    final = runtime.handle_event(
        result_event(
            context_id="multi",
            message_id="circulation-result",
            tool_name="set_air_circulation",
            content={
                "status": "SUCCESS",
                "result": {"mode": "RECIRCULATE"},
            },
        )
    )
    assert final.tool_calls == ()
    assert final.text is not None and final.text.startswith("Done")
    session = runtime.sessions.get("multi")
    assert session is not None
    assert [goal.status.value for goal in session.goal_dag.goals] == ["done", "done"]
    assert client.action_calls == 2


def test_spelling_only_enum_variants_use_the_exact_live_binding() -> None:
    intent = action_intent(
        [
            {
                "semantic_operation": "set_air_circulation",
                "desired_outcome": {"mode": "fresh_air"},
            }
        ],
        explicit_slots={"mode": "fresh_air"},
    )

    def lowercase_proposal(payload: dict[str, Any]) -> dict[str, Any]:
        return set_proposal(
            payload,
            semantic_operation="set_air_circulation",
            tool_name="set_air_circulation",
            arguments={"mode": "fresh_air"},
        )

    client = FakeStructuredClient(intent, lowercase_proposal)
    runtime = runtime_for(client)
    outbound = runtime.handle_event(
        user_event(
            context_id="fresh-air-enum",
            message_id="fresh-air-user",
            text="Change the air circulation to fresh air mode.",
            tools=(
                tool(
                    "set_air_circulation",
                    {
                        "mode": {
                            "type": "string",
                            "enum": ["FRESH_AIR", "RECIRCULATION", "AUTO"],
                        }
                    },
                    ["mode"],
                ),
            ),
        )
    )

    assert outbound.text is None
    assert outbound.tool_calls == (
        {"tool_name": "set_air_circulation", "arguments": {"mode": "FRESH_AIR"}},
    )


def test_planner_cannot_change_a_grounded_enum_value() -> None:
    intent = action_intent(
        [
            {
                "semantic_operation": "set_trunk_position",
                "desired_outcome": {"action": "open"},
            }
        ],
        explicit_slots={"action": "open"},
    )

    def changed_proposal(payload: dict[str, Any]) -> dict[str, Any]:
        return set_proposal(
            payload,
            semantic_operation="set_trunk_position",
            tool_name="open_close_trunk_door",
            arguments={"action": "close"},
        )

    client = FakeStructuredClient(intent, changed_proposal)
    runtime = runtime_for(client)
    outbound = runtime.handle_event(
        user_event(
            context_id="trunk-enum-drift",
            message_id="trunk-user",
            text="Open the trunk door.",
            tools=(
                tool(
                    "open_close_trunk_door",
                    {
                        "action": {
                            "type": "string",
                            "enum": ["OPEN", "CLOSE"],
                        }
                    },
                    ["action"],
                ),
            ),
        )
    )

    assert outbound.text is None
    assert outbound.tool_calls == (
        {
            "tool_name": "open_close_trunk_door",
            "arguments": {"action": "OPEN"},
        },
    )
    session = runtime.sessions.get("trunk-enum-drift")
    assert session is not None
    assert session.goal_dag.goals[0].status.value != "done"
    assert len(session.pending_calls) == 1


@pytest.mark.parametrize("kind", ["respond", "complete"])
def test_model_cannot_respond_or_complete_an_unfinished_action(kind: str) -> None:
    def premature(payload: dict[str, Any]) -> dict[str, Any]:
        goal = goal_from_payload(payload, "set_fan_speed")
        return {
            "kind": kind,
            "goal_ids": [goal["goal_id"]],
            "user_text": "I completed it.",
        }

    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_speed",
                    "desired_outcome": {"level": 2},
                }
            ],
            explicit_slots={"level": 2},
        ),
        premature,
    )
    runtime = runtime_for(client)

    outbound = runtime.handle_event(
        user_event(
            context_id=f"premature-{kind}",
            message_id="premature-user",
            text="Set the fan to level two.",
            tools=(FAN_TOOL,),
        )
    )
    assert outbound.text is None
    assert outbound.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 2.0}},
    )
    session = runtime.sessions.get(f"premature-{kind}")
    assert session is not None
    assert session.goal_dag.goals[0].status.value != "done"
    assert len(session.pending_calls) == 1

    completed = runtime.handle_event(
        result_event(
            context_id=f"premature-{kind}",
            message_id=f"premature-{kind}-result",
            tool_name="set_fan_speed",
            content={"status": "SUCCESS", "result": {"level": 2}},
        )
    )
    assert completed.tool_calls == ()
    assert completed.text == "Done. I set the fan speed to level 2."


def test_model_cannot_omit_goal_scope_to_claim_early_completion() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_speed",
                    "desired_outcome": {"level": 2},
                }
            ],
            explicit_slots={"level": 2},
        ),
        lambda payload: {
            "kind": "respond",
            "goal_ids": [],
            "user_text": "Done, I changed the fan.",
        },
    )
    runtime = runtime_for(client)
    outbound = runtime.handle_event(
        user_event(
            context_id="empty-goal-scope",
            message_id="empty-goal-user",
            text="Set the fan to level two.",
            tools=(FAN_TOOL,),
        )
    )

    assert outbound.text is None
    assert outbound.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 2}},
    )
    session = runtime.sessions.get("empty-goal-scope")
    assert session is not None
    assert session.goal_dag.goals[0].status.value != "done"
    assert len(session.pending_calls) == 1


@pytest.mark.parametrize("user_text", ["What time is it?", "What is the fan speed?"])
def test_model_hallucinated_write_intent_is_not_user_authorization(
    user_text: str,
) -> None:
    def action_must_not_run(payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(f"planner received ungrounded goals: {payload}")

    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_speed",
                    "desired_outcome": {"level": 5},
                }
            ],
            explicit_slots={"level": 5},
        ),
        action_must_not_run,
    )
    runtime = runtime_for(client)
    outbound = runtime.handle_event(
        user_event(
            context_id=f"ungrounded-{len(user_text)}",
            message_id="ungrounded-user",
            text=user_text,
            tools=(FAN_TOOL,),
        )
    )

    assert outbound.tool_calls == ()
    session = runtime.sessions.get(f"ungrounded-{len(user_text)}")
    assert session is not None
    assert session.goal_dag.goals == []
    assert session.authorized_action_goal_ids == set()
    assert session.grounded_desired_values_by_goal == {}
    assert client.action_calls == 0


def test_model_cannot_mislabel_a_write_tool_as_a_read() -> None:
    def disguised_write(payload: dict[str, Any]) -> dict[str, Any]:
        goal = goal_from_payload(payload, "set_fan_speed")
        return {
            "kind": "tool_get",
            "goal_ids": [goal["goal_id"]],
            "tool_calls": [
                {
                    "tool_name": "set_fan_speed",
                    "arguments": {"level": 4},
                    "goal_id": goal["goal_id"],
                }
            ],
        }

    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_speed",
                    "desired_outcome": {"level": 2},
                }
            ],
            explicit_slots={"level": 2},
        ),
        disguised_write,
    )
    runtime = runtime_for(client)
    outbound = runtime.handle_event(
        user_event(
            context_id="disguised-write",
            message_id="disguised-write-user",
            text="Set the fan to level two.",
            tools=(FAN_TOOL,),
        )
    )

    assert outbound.text is None
    assert outbound.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 2}},
    )
    session = runtime.sessions.get("disguised-write")
    assert session is not None
    assert len(session.pending_calls) == 1
    assert session.pending_calls[0].state_changing is True
    assert session.pending_calls[0].arguments == {"level": 2}
    assert session.budget.attempted_sets


def test_model_cannot_expand_a_recipe_read_to_unrequested_parameters() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "cross_domain_request",
                    "desired_outcome": {"preference_categories": ["climate"]},
                }
            ],
            explicit_slots={"preference_categories": ["climate"]},
        ),
        lambda payload: {
            "kind": "tool_get",
            "goal_ids": [payload["semantic_goals"]["goals"][0]["goal_id"]],
            "tool_calls": [
                {
                    "tool_name": "get_user_preferences",
                    "arguments": {
                        "preference_categories": ["unrequested-private-category"]
                    },
                    "goal_id": payload["semantic_goals"]["goals"][0]["goal_id"],
                }
            ],
        },
    )
    runtime = runtime_for(client)
    outbound = runtime.handle_event(
        user_event(
            context_id="read-scope",
            message_id="read-scope-user",
            text="Show my climate preferences.",
            tools=(
                tool(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    ["preference_categories"],
                ),
            ),
        )
    )

    assert outbound.tool_calls == ()
    assert outbound.text is not None
    session = runtime.sessions.get("read-scope")
    assert session is not None
    assert session.pending_calls == []


@pytest.mark.parametrize("kind", ["respond", "complete"])
def test_read_goal_cannot_answer_before_its_own_tool_result(kind: str) -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "find_contact",
                    "desired_outcome": {"first_name": "Alex"},
                }
            ],
            explicit_slots={"first_name": "Alex"},
        ),
        lambda payload: {
            "kind": kind,
            "goal_ids": [payload["semantic_goals"]["goals"][0]["goal_id"]],
            "user_text": "Alex's contact is ready.",
        },
    )
    runtime = runtime_for(client)
    outbound = runtime.handle_event(
        user_event(
            context_id=f"early-read-{kind}",
            message_id=f"early-read-{kind}-user",
            text="Find Alex's contact.",
            tools=(
                tool(
                    "get_contact_id_by_contact_name",
                    {
                        "contact_first_name": {"type": "string"},
                        "contact_last_name": {"type": "string"},
                    },
                ),
            ),
        )
    )

    assert outbound.tool_calls == ()
    assert outbound.text is not None
    assert "contact is ready" not in outbound.text
    session = runtime.sessions.get(f"early-read-{kind}")
    assert session is not None
    assert session.goal_dag.goals[0].status.value != "done"


def test_read_goal_can_complete_only_after_its_scoped_success_result() -> None:
    proposals = iter(("read", "complete"))

    def next_step(payload: dict[str, Any]) -> dict[str, Any]:
        goal_id = payload["semantic_goals"]["goals"][0]["goal_id"]
        if next(proposals) == "read":
            return {
                "kind": "tool_get",
                "goal_ids": [goal_id],
                "tool_calls": [
                    {
                        "tool_name": "get_contact_id_by_contact_name",
                        "arguments": {"contact_first_name": "Alex"},
                        "goal_id": goal_id,
                    }
                ],
            }
        return {
            "kind": "complete",
            "goal_ids": [goal_id],
            "user_text": "I found Alex's contact.",
        }

    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "find_contact",
                    "desired_outcome": {"first_name": "Alex"},
                }
            ],
            explicit_slots={"first_name": "Alex"},
        ),
        next_step,
    )
    runtime = runtime_for(client)
    controls = (
        tool(
            "get_contact_id_by_contact_name",
            {
                "contact_first_name": {"type": "string"},
                "contact_last_name": {"type": "string"},
            },
        ),
    )
    first = runtime.handle_event(
        user_event(
            context_id="read-complete",
            message_id="read-complete-user",
            text="Find Alex's contact.",
            tools=controls,
        )
    )
    assert first.tool_calls == (
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {"contact_first_name": "Alex"},
        },
    )

    final = runtime.handle_event(
        result_event(
            context_id="read-complete",
            message_id="read-complete-result",
            tool_name="get_contact_id_by_contact_name",
            content={
                "status": "SUCCESS",
                "result": {"id": "contact-1", "name": "Alex Example"},
            },
        )
    )
    assert final.tool_calls == ()
    assert final.text == "I found Alex's contact."
    session = runtime.sessions.get("read-complete")
    assert session is not None
    assert session.goal_dag.goals[0].status.value == "done"


def test_current_day_calendar_question_recovers_and_reads_system_date() -> None:
    client = FakeStructuredClient(
        {
            "language": "en",
            "intent_kind": "information",
            "call_for_action": False,
            "goals": [
                {
                    "semantic_operation": "get_entries_from_calendar",
                    "desired_outcome": {"month": 6, "day": 6},
                }
            ],
            "explicit_slots": {},
        },
        lambda payload: {
            "kind": "complete",
            "goal_ids": [payload["semantic_goals"]["goals"][0]["goal_id"]],
            "user_text": (
                "I found one remaining meeting today at 18:00 called Leadership "
                "Development."
            ),
        },
    )
    runtime = runtime_for(client)
    context_id = "calendar-today-recovery"

    outbound = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id="calendar-today-user",
            system_policy=(
                'DATETIME = {"year":2025,"month":6,"day":6,"hour":16,'
                '"minute":15}. AUT-POL:023: Calendar entries can only be '
                "requested for the current day."
            ),
            user_text="What's on my calendar today?",
            live_tools=(CALENDAR_TOOL,),
        )
    )

    assert outbound.tool_calls == (
        {
            "tool_name": "get_entries_from_calendar",
            "arguments": {"month": 6, "day": 6},
        },
    )
    assert client.action_calls == 0

    final = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="calendar-today-result",
            tool_name="get_entries_from_calendar",
            content={
                "status": "SUCCESS",
                "result": {
                    "date": {"year": 2025, "month": 6, "day": 6},
                    "meetings": [
                        {
                            "start": {"hour": "18", "minute": "00"},
                            "duration": "60min",
                            "topic": "Leadership Development",
                        }
                    ],
                },
            },
        )
    )

    assert final.tool_calls == ()
    assert final.text is not None and "18:00" in final.text
    assert client.action_calls == 0
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.goal_dag.goals[0].status.value == "done"


def test_current_navigation_question_reads_and_summarizes_details() -> None:
    client = FakeStructuredClient(
        {
            "language": "en",
            "intent_kind": "information",
            "call_for_action": False,
            "goals": [
                {
                    "semantic_operation": "read_current_navigation",
                    "desired_outcome": {},
                }
            ],
            "explicit_slots": {},
        },
        lambda payload: pytest.fail(
            "the deterministic navigation path called the planner"
        ),
    )
    runtime = runtime_for(client)
    context_id = "current-navigation-details"

    outbound = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="current-navigation-user",
            text="Can you tell me the current navigation status?",
            tools=(NAVIGATION_STATE_TOOL,),
        )
    )

    assert outbound.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    assert client.action_calls == 0

    final = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="current-navigation-result",
            tool_name="get_current_navigation_state",
            content={
                "status": "SUCCESS",
                "result": {
                    "navigation_active": True,
                    "waypoints_id": ["loc-dortmund", "loc-cologne"],
                    "routes_to_final_destination_id": ["route-one"],
                    "details": {
                        "waypoints": [
                            {"id": "loc-dortmund", "name": "Dortmund"},
                            {"id": "loc-cologne", "name": "Cologne"},
                        ],
                        "routes": [
                            {
                                "route_id": "route-one",
                                "start_id": "loc-dortmund",
                                "destination_id": "loc-cologne",
                                "name_via": "B572",
                                "distance_km": 88.75,
                                "duration_hours": 1,
                                "duration_minutes": 7,
                                "road_types": ["country road"],
                                "includes_toll": False,
                                "alias": ["fastest", "first", "shortest"],
                            }
                        ],
                    },
                },
            },
        )
    )

    assert final.tool_calls == ()
    assert final.text is not None
    assert "Dortmund" in final.text and "Cologne" in final.text
    assert "88.75" in final.text and "1 hour and 7 minutes" in final.text
    assert "no toll roads" in final.text and "fastest" in final.text
    assert client.action_calls == 0
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.goal_dag.goals[0].status.value == "done"


def test_explicit_soc_range_uses_one_grounded_distance_read() -> None:
    intent = {
        "language": "en",
        "intent_kind": "information",
        "call_for_action": False,
        "goals": [
            {
                "semantic_operation": "read_charging_status",
                "desired_outcome": {},
            },
            {
                "semantic_operation": "get_distance_by_soc",
                "desired_outcome": {"initial_soc": 50},
                "depends_on_indices": [0],
            },
        ],
        "explicit_slots": {"initial_soc": 50},
        "explicit_constraints": {"target_soc": 10},
    }

    def complete_range(payload: dict[str, Any]) -> dict[str, Any]:
        goal = goal_from_payload(payload, "get_distance_by_soc")
        return {
            "kind": "complete",
            "goal_ids": [goal["goal_id"]],
            "user_text": (
                "I estimate you can drive 127 kilometers before reaching 10 percent."
            ),
        }

    client = FakeStructuredClient(intent, complete_range)
    runtime = runtime_for(client)
    context_id = "explicit-soc-range"

    outbound = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="explicit-soc-range-user",
            text="What is my driving range from 50% charge down to 10% charge?",
            tools=(DISTANCE_BY_SOC_TOOL,),
        )
    )

    assert outbound.tool_calls == (
        {
            "tool_name": "get_distance_by_soc",
            "arguments": {
                "initial_state_of_charge": 50,
                "final_state_of_charge": 10,
            },
        },
    )
    assert client.action_calls == 0

    final = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="explicit-soc-range-result",
            tool_name="get_distance_by_soc",
            content={
                "status": "SUCCESS",
                "result": {"distance_km_for_50_until_10_percent_soc": "127km"},
            },
        )
    )

    assert final.tool_calls == ()
    assert final.text is not None and "127" in final.text
    assert client.action_calls == 0
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert [goal.semantic_operation for goal in session.goal_dag.goals] == [
        "get_distance_by_soc"
    ]
    assert session.goal_dag.goals[0].status.value == "done"


def test_reverse_soc_range_is_not_dispatched_deterministically() -> None:
    client = FakeStructuredClient(
        {
            "language": "en",
            "intent_kind": "information",
            "call_for_action": False,
            "goals": [
                {
                    "semantic_operation": "get_distance_by_soc",
                    "desired_outcome": {"initial_soc": 10, "final_soc": 50},
                }
            ],
            "explicit_slots": {"initial_soc": 10, "final_soc": 50},
        },
        lambda payload: {
            "kind": "respond",
            "goal_ids": [payload["semantic_goals"]["goals"][0]["goal_id"]],
            "user_text": "I need a valid decreasing state-of-charge range.",
        },
    )
    runtime = runtime_for(client)

    outbound = runtime.handle_event(
        user_event(
            context_id="reverse-soc-range",
            message_id="reverse-soc-range-user",
            text="Calculate my driving range from 10% to 50%.",
            tools=(DISTANCE_BY_SOC_TOOL,),
        )
    )

    assert outbound.tool_calls == ()
    assert outbound.text is not None
    assert client.action_calls == 1


def test_text_fallback_is_revalidated_instead_of_forged_allow() -> None:
    runtime = runtime_for(ac_client())
    delegate = runtime.precommit

    class RejectOnce:
        def __init__(self) -> None:
            self.calls = 0

        def validate(self, proposal, context):
            self.calls += 1
            if self.calls == 1:
                return CommitDecision(outcome=GateOutcome.POLICY_CONFLICT)
            return delegate.validate(proposal, context)

    gate = RejectOnce()
    runtime.precommit = gate  # type: ignore[assignment]
    outbound = runtime.handle_event(
        InboundEvent(
            context_id="fallback-gate",
            message_id="fallback-gate-message",
            system_policy="Follow the current safety policy.",
            live_tools=AC_TOOLS,
            malformed_parts=("part[0]:malformed-data",),
        )
    )

    assert gate.calls == 2
    assert outbound.tool_calls == ()
    assert outbound.text is not None and "did not perform" in outbound.text


@pytest.mark.parametrize(
    "content",
    [
        "{",
        {"status": "SUCCESS", "result": {"level": "unknown"}},
    ],
    ids=["malformed", "unknown-value"],
)
def test_malformed_or_unknown_set_result_never_marks_done(content: Any) -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_speed",
                    "desired_outcome": {"level": 2},
                }
            ],
            explicit_slots={"level": 2},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_fan_speed",
            tool_name="set_fan_speed",
            arguments={"level": 2},
        ),
    )
    runtime = runtime_for(client)
    runtime.handle_event(
        user_event(
            context_id="bad-set",
            message_id="bad-set-user",
            text="Set the fan to level two.",
            tools=(FAN_TOOL,),
        )
    )

    outbound = runtime.handle_event(
        result_event(
            context_id="bad-set",
            message_id="bad-set-result",
            tool_name="set_fan_speed",
            content=content,
        )
    )
    assert outbound.tool_calls == ()
    assert outbound.text is not None
    assert not outbound.text.startswith("Done")
    session = runtime.sessions.get("bad-set")
    assert session is not None
    assert session.goal_dag.goals[0].status.value == "failed"
    assert session.execution_bundle is None


def test_soft_budget_rejects_first_set_when_full_ac_bundle_cannot_finish() -> None:
    runtime = runtime_for(ac_client(), soft_max_steps=5)
    runtime.handle_event(
        user_event(
            context_id="budget",
            message_id="budget-user",
            text="Turn on the air conditioning.",
            tools=AC_TOOLS,
        )
    )
    runtime.handle_event(
        result_event(
            context_id="budget",
            message_id="budget-climate",
            tool_name="get_climate_settings",
            content={"status": "SUCCESS", "result": {"fan_speed": 0}},
        )
    )
    outbound = runtime.handle_event(
        result_event(
            context_id="budget",
            message_id="budget-windows",
            tool_name="get_vehicle_window_positions",
            content={
                "status": "SUCCESS",
                "result": {
                    "window_driver_position": 30,
                    "window_passenger_position": 0,
                    "window_driver_rear_position": 0,
                    "window_passenger_rear_position": 0,
                },
            },
        )
    )

    assert outbound.tool_calls == ()
    assert outbound.text is not None
    assert "haven't performed" in outbound.text
    session = runtime.sessions.get("budget")
    assert session is not None
    assert session.budget.steps == 3
    assert session.budget.attempted_sets == set()
    assert session.goal_dag.goals[0].status.value != "done"


def test_exhausted_budget_returns_cached_safe_text_instead_of_raising() -> None:
    runtime = runtime_for(ac_client(), soft_max_steps=1)
    first = runtime.handle_event(
        user_event(
            context_id="exhausted",
            message_id="exhausted-user",
            text="Turn on the air conditioning.",
            tools=AC_TOOLS,
        )
    )
    assert first.tool_calls == ()
    session = runtime.sessions.get("exhausted")
    assert session is not None and session.budget.steps == 1

    event = InboundEvent(
        context_id="exhausted",
        message_id="exhausted-malformed",
        malformed_parts=("part[0]:malformed-data",),
    )
    safe = runtime.handle_event(event)
    replay = runtime.handle_event(event)

    assert safe.tool_calls == ()
    assert safe.text is not None and "safe interaction limit" in safe.text
    assert replay is safe
    assert session.budget.steps == 1


def test_exhausted_budget_fallback_is_localized_and_revalidated() -> None:
    client = FakeStructuredClient(
        {
            **action_intent(
                [
                    {
                        "semantic_operation": "set_fan_speed",
                        "desired_outcome": {"level": 2},
                    }
                ],
                explicit_slots={"level": 2},
            ),
            "language": "zh",
        },
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_fan_speed",
            tool_name="set_fan_speed",
            arguments={"level": 2},
        ),
    )
    runtime = runtime_for(client, soft_max_steps=1)
    runtime.handle_event(
        user_event(
            context_id="exhausted-zh",
            message_id="exhausted-zh-user",
            text="Set the fan to level two.",
            tools=(FAN_TOOL,),
        )
    )
    safe = runtime.handle_event(
        InboundEvent(
            context_id="exhausted-zh",
            message_id="exhausted-zh-malformed",
            malformed_parts=("part[0]:malformed-data",),
        )
    )

    session = runtime.sessions.get("exhausted-zh")
    assert session is not None
    assert safe.text is not None and any(
        "\u4e00" <= char <= "\u9fff" for char in safe.text
    )
    decision = runtime.precommit.validate(
        DecisionProposal(kind="respond", user_text=safe.text),
        runtime._basic_gate_context(session),
    )
    assert decision.outcome is GateOutcome.ALLOW


def test_get_cannot_omit_a_grounded_optional_filter() -> None:
    intent = {
        "language": "en",
        "intent_kind": "information",
        "call_for_action": False,
        "goals": [
            {
                "semantic_operation": "search_poi_at_location",
                "desired_outcome": {
                    "location_id": "Airport",
                    "category": "restaurants",
                    "filters": ["vegan"],
                },
            }
        ],
        "explicit_slots": {
            "location_id": "Airport",
            "category": "restaurants",
            "filters": ["vegan"],
        },
    }

    def omit_filter(payload: dict[str, Any]) -> dict[str, Any]:
        goal = goal_from_payload(payload, "search_poi_at_location")
        return {
            "kind": "tool_get",
            "goal_ids": [goal["goal_id"]],
            "tool_calls": [
                {
                    "tool_name": "search_poi_at_location",
                    "arguments": {
                        "location_id": "Airport",
                        "category_poi": "restaurants",
                    },
                    "goal_id": goal["goal_id"],
                }
            ],
        }

    runtime = runtime_for(FakeStructuredClient(intent, omit_filter))
    outbound = runtime.handle_event(
        user_event(
            context_id="poi-filter",
            message_id="poi-filter-user",
            text="Search POI at Airport for restaurants with vegan filters.",
            tools=(
                tool(
                    "search_poi_at_location",
                    {
                        "location_id": {"type": "string"},
                        "category_poi": {"type": "string"},
                        "filters": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    ["location_id", "category_poi"],
                ),
            ),
        )
    )

    assert outbound.tool_calls == ()


def test_wrong_task_result_id_cannot_poison_the_correct_task() -> None:
    runtime = runtime_for(
        FakeStructuredClient(
            action_intent(
                [
                    {
                        "semantic_operation": "set_fan_speed",
                        "desired_outcome": {"level": 2},
                    }
                ],
                explicit_slots={"level": 2},
            ),
            lambda payload: set_proposal(
                payload,
                semantic_operation="set_fan_speed",
                tool_name="set_fan_speed",
                arguments={"level": 2},
            ),
        )
    )
    first = runtime.handle_event(
        InboundEvent(
            context_id="task-dedup",
            task_id="task-good",
            message_id="task-dedup-user",
            system_policy="Follow the current safety policy.",
            user_text="Set the fan to level two.",
            live_tools=(FAN_TOOL,),
        )
    )
    assert first.tool_calls

    result = {
        "toolCallId": "opaque-shared",
        "toolName": "set_fan_speed",
        "content": {"status": "SUCCESS", "result": {"level": 2}},
    }
    wrong = runtime.handle_event(
        InboundEvent(
            context_id="task-dedup",
            task_id="task-wrong",
            message_id="task-dedup-wrong",
            tool_results=(result,),
        )
    )
    assert wrong.tool_calls == ()
    session = runtime.sessions.get("task-dedup")
    assert session is not None and len(session.pending_calls) == 1

    correct = runtime.handle_event(
        InboundEvent(
            context_id="task-dedup",
            task_id="task-good",
            message_id="task-dedup-correct",
            tool_results=(result,),
        )
    )
    assert correct.text is not None and correct.text.startswith("Done")
    assert session.goal_dag.goals[0].status.value == "done"


def test_clarification_label_cannot_launder_a_different_value() -> None:
    client = FakeStructuredClient(
        {
            **action_intent(
                [
                    {
                        "semantic_operation": "set_fan_speed",
                        "desired_outcome": {"level": "low"},
                    }
                ],
                explicit_slots={"level": "low"},
            ),
            "ambiguities": [
                {
                    "name": "level",
                    "candidate_values": [5, 1],
                    "candidate_labels": ["low", "high"],
                    "goal_index": 0,
                }
            ],
        },
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_fan_speed",
            tool_name="set_fan_speed",
            arguments={"level": 5},
        ),
    )
    runtime = runtime_for(client)
    permissive_fan = tool("set_fan_speed", {"level": {}}, ["level"])
    question = runtime.handle_event(
        user_event(
            context_id="label-laundering",
            message_id="label-laundering-user",
            text="Set the fan to low.",
            tools=(permissive_fan,),
        )
    )
    assert question.text is not None
    assert "5" in question.text and "1" in question.text
    assert "low" not in question.text.casefold()

    answer = runtime.handle_event(
        InboundEvent(
            context_id="label-laundering",
            message_id="label-laundering-answer",
            user_text="low",
        )
    )
    assert answer.tool_calls == ()
    session = runtime.sessions.get("label-laundering")
    assert session is not None
    grounded = next(iter(session.grounded_desired_values_by_goal.values()))
    assert grounded.get("level") != 5


def test_unsupported_ambient_color_asks_live_enum_then_uses_explicit_choice() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_ambient_lights",
                    "desired_outcome": {"color": "brown"},
                }
            ],
            explicit_slots={"color": "brown"},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_ambient_lights",
            tool_name="set_ambient_lights",
            arguments={"on": True, "lightcolor": "purple"},
        ),
    )
    runtime = runtime_for(client)
    question = runtime.handle_event(
        InboundEvent(
            context_id="ambient-live-enum",
            message_id="ambient-live-enum-user",
            system_policy=(
                "Make sure answers sound natural in first person and in the user "
                "language. Responses are forwarded to text-to-speech, so do not "
                "include visual formatting. POL-002: Use kilometers and meters for "
                "distance, degrees Celsius for temperature, and 24h datetime format."
            ),
            user_text="Change the ambient lights to brown.",
            live_tools=(AMBIENT_LIGHT_TOOL,),
        )
    )

    assert question.tool_calls == ()
    assert question.text is not None and "purple" in question.text.casefold()
    assert client.action_calls == 0

    for index, unsafe_answer in enumerate(
        (
            "Not purple.",
            "Is purple supported?",
            "Maybe purple?",
            "Purple?",
            "Use anything except purple.",
            "Set it without purple.",
            "Change them to purple or cyan.",
            "Set the temperature to 20 and explain the purple option.",
        )
    ):
        repeated = runtime.handle_event(
            InboundEvent(
                context_id="ambient-live-enum",
                message_id=f"ambient-live-enum-unsafe-{index}",
                user_text=unsafe_answer,
            )
        )
        assert repeated.tool_calls == ()
        assert repeated.text is not None and "purple" in repeated.text.casefold()
        assert client.action_calls == 0

    action = runtime.handle_event(
        InboundEvent(
            context_id="ambient-live-enum",
            task_id="ambient-live-enum",
            message_id="ambient-live-enum-choice",
            user_text="Change them to purple.",
        )
    )

    assert action.tool_calls == (
        {
            "tool_name": "set_ambient_lights",
            "arguments": {"on": True, "lightcolor": "PURPLE"},
        },
    )
    assert client.action_calls == 1
    session = runtime.sessions.get("ambient-live-enum")
    assert session is not None
    goal_id = session.goal_dag.goals[0].goal_id
    assert "enabled" not in session.grounded_value_sources_by_goal[goal_id]
    evidence_id = session.derived_value_evidence_by_goal[goal_id]["enabled"]
    derived = session.evidence.evidence[evidence_id]
    assert derived.source_kind is EvidenceSourceKind.DERIVED
    assert derived.derivation == "ambient_color_selection_enables_lights_v1"
    assert len(derived.derived_from) == 1
    parent = session.evidence.evidence[derived.derived_from[0]]
    assert parent.source_kind is EvidenceSourceKind.USER
    assert parent.value == "PURPLE"
    assert parent.source_turn_id == "ambient-live-enum-choice"

    final = runtime.handle_event(
        InboundEvent(
            context_id="ambient-live-enum",
            task_id="ambient-live-enum",
            message_id="ambient-live-enum-result",
            tool_results=(
                {
                    "toolCallId": "opaque-ambient-call",
                    "toolName": "set_ambient_lights",
                    "content": json.dumps(
                        {
                            "status": "SUCCESS",
                            "result": {"on": True, "lightcolor": "PURPLE"},
                        }
                    ),
                },
            ),
        )
    )
    assert final.tool_calls == ()
    assert final.text is not None and final.text.startswith("Done")


def test_compound_slot_name_is_humanized_in_live_clarification() -> None:
    client = FakeStructuredClient(
        {
            **action_intent(
                [
                    {
                        "semantic_operation": "set_seat_heating",
                        "desired_outcome": {"level": 1},
                    }
                ],
                explicit_slots={"level": 1},
            ),
            "ambiguities": [
                {
                    "name": "seat_zone",
                    "candidate_values": ["ALL_ZONES", "DRIVER", "PASSENGER"],
                    "goal_index": 0,
                }
            ],
        },
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_seat_heating",
            tool_name="set_seat_heating",
            arguments={"seat_zone": "ALL_ZONES", "level": 1},
        ),
    )
    runtime = runtime_for(client)

    question = runtime.handle_event(
        user_event(
            context_id="seat-zone-live-enum",
            message_id="seat-zone-live-enum-user",
            text="Set the seat heating to level 1.",
            tools=(SEAT_HEATING_TOOL,),
        )
    )

    assert question.tool_calls == ()
    assert question.text is not None
    assert "seat zone" in question.text.casefold()
    assert "seat_zone" not in question.text
    assert "all zones" in question.text.casefold()
    assert client.action_calls == 0

    runtime.handle_event(
        InboundEvent(
            context_id="seat-zone-live-enum",
            message_id="seat-zone-live-enum-choice",
            user_text="All zones, please.",
        )
    )
    session = runtime.sessions.get("seat-zone-live-enum")
    assert session is not None and session.intent is not None
    assert session.pending_clarifications == {}
    assert session.intent.goals[0].desired_outcome == {
        "level": 1,
        "seat_zone": "ALL_ZONES",
    }


@pytest.mark.parametrize(
    "answer",
    [
        "Defrost the front window.",
        "Just the front, please.",
        "Yeah, just the front window defrost, please.",
        "I just need the front window defrost turned on. That's all.",
    ],
)
def test_unique_assertive_defrost_clarification_is_accepted(answer: str) -> None:
    client = ambiguous_defrost_client()
    runtime = runtime_for(client)
    context_id = f"defrost-choice-{abs(hash(answer))}"

    question = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="Turn on the window defrost.",
            tools=DEFROST_TOOLS,
        )
    )
    assert question.text is not None
    assert "front" in question.text.casefold()

    outbound = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-answer",
            user_text=answer,
        )
    )

    assert outbound.tool_calls == (
        {
            "tool_name": "set_window_defrost",
            "arguments": {"on": True, "defrost_window": "FRONT"},
        },
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert session.pending_clarifications == {}
    assert session.intent.goals[0].desired_outcome == {
        "enabled": True,
        "window": "FRONT",
    }


def test_explicit_front_windshield_request_skips_model_ambiguity() -> None:
    client = ambiguous_defrost_client()
    runtime = runtime_for(client)

    outbound = runtime.handle_event(
        user_event(
            context_id="explicit-front-defrost",
            message_id="explicit-front-defrost-user",
            text=(
                "Hey there! My front windshield is getting all foggy, can you "
                "turn on the defrost for me?"
            ),
            tools=DEFROST_TOOLS,
        )
    )

    assert outbound.text is None
    assert outbound.tool_calls == (
        {
            "tool_name": "set_window_defrost",
            "arguments": {"on": True, "defrost_window": "FRONT"},
        },
    )
    session = runtime.sessions.get("explicit-front-defrost")
    assert session is not None and session.intent is not None
    assert session.pending_clarifications == {}
    assert session.intent.goals[0].desired_outcome == {
        "enabled": True,
        "window": "FRONT",
    }


@pytest.mark.parametrize(
    "answer",
    [
        "Do not defrost the front window.",
        "Defrost the front or rear window.",
        "If needed, defrost the front window.",
        "Not the front.",
        "Would front work?",
        "Front or rear.",
        "Choose front and rear.",
        "If possible, front.",
    ],
)
def test_unsafe_or_non_unique_defrost_clarification_is_reasked(
    answer: str,
) -> None:
    client = ambiguous_defrost_client()
    runtime = runtime_for(client)
    context_id = f"defrost-reask-{abs(hash(answer))}"
    runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="Turn on the window defrost.",
            tools=DEFROST_TOOLS,
        )
    )

    outbound = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-answer",
            user_text=answer,
        )
    )

    assert outbound.tool_calls == ()
    assert outbound.text is not None
    assert "front" in outbound.text.casefold()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.pending_clarifications
    assert client.action_calls == 0


def test_small_bounded_numeric_domain_uses_live_clarification_options() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_seat_heating",
                    "desired_outcome": {"seat_zone": "DRIVER"},
                }
            ],
            explicit_slots={"seat_zone": "DRIVER"},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_seat_heating",
            tool_name="set_seat_heating",
            arguments={"seat_zone": "DRIVER", "level": 1},
        ),
    )
    runtime = runtime_for(client)

    question = runtime.handle_event(
        user_event(
            context_id="seat-level-live-domain",
            message_id="seat-level-live-domain-user",
            text="Turn down the driver seat heating.",
            tools=(SEAT_HEATING_TOOL,),
        )
    )

    assert question.tool_calls == ()
    assert question.text is not None
    assert "level" in question.text.casefold()
    assert all(str(value) in question.text for value in range(4))
    assert client.action_calls == 0

    action = runtime.handle_event(
        InboundEvent(
            context_id="seat-level-live-domain",
            message_id="seat-level-live-domain-choice",
            user_text="Let's go with level 1 for both, please.",
        )
    )
    assert action.tool_calls == ()
    assert client.action_calls == 1
    session = runtime.sessions.get("seat-level-live-domain")
    assert session is not None and session.intent is not None
    assert session.pending_clarifications == {}
    assert session.intent.goals[0].desired_outcome == {
        "seat_zone": "DRIVER",
        "level": 1,
    }


def test_split_seat_goals_collapse_before_polite_level_clarification() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_seat_heating",
                    "desired_outcome": {
                        "level": "lowered",
                        "seat_zone": "driver",
                    },
                },
                {
                    "semantic_operation": "set_seat_heating",
                    "desired_outcome": {
                        "level": "lowered",
                        "seat_zone": "passenger",
                    },
                },
            ],
            explicit_slots={
                "seat_zone": ["driver", "passenger"],
                "level": ["lowered"],
            },
        ),
        lambda payload: pytest.fail("the occupancy read must precede planning"),
    )
    runtime = runtime_for(client)

    question = runtime.handle_event(
        user_event(
            context_id="split-seat-goals",
            message_id="split-seat-goals-user",
            text=(
                "Hey there, could you turn down the seat heating for both me and "
                "my passenger? It's getting a bit too warm in here."
            ),
            tools=(SEAT_HEATING_TOOL, SEAT_OCCUPANCY_TOOL),
        )
    )
    assert question.text is not None and "level" in question.text.casefold()
    session = runtime.sessions.get("split-seat-goals")
    assert session is not None and session.intent is not None
    assert [goal.desired_outcome for goal in session.intent.goals] == [
        {"seat_zone": "ALL_ZONES"}
    ]

    read = runtime.handle_event(
        InboundEvent(
            context_id="split-seat-goals",
            message_id="split-seat-goals-choice",
            user_text="Let's go with level 1 for both, please.",
        )
    )
    assert read.tool_calls == ({"tool_name": "get_seats_occupancy", "arguments": {}},)
    assert client.action_calls == 0


def test_ambient_color_change_preserves_explicit_off_state() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_ambient_lights",
                    "desired_outcome": {"color": "purple", "enabled": True},
                }
            ],
            explicit_slots={"color": "purple", "enabled": True},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_ambient_lights",
            tool_name="set_ambient_lights",
            arguments={"on": False, "lightcolor": "purple"},
        ),
    )
    runtime = runtime_for(client)

    action = runtime.handle_event(
        user_event(
            context_id="ambient-explicit-off",
            message_id="ambient-explicit-off-user",
            text=(
                "Set the ambient lights to purple without first needing to turn "
                "them on."
            ),
            tools=(AMBIENT_LIGHT_TOOL,),
        )
    )

    assert action.tool_calls == (
        {
            "tool_name": "set_ambient_lights",
            "arguments": {"on": False, "lightcolor": "PURPLE"},
        },
    )
    session = runtime.sessions.get("ambient-explicit-off")
    assert session is not None
    goal_id = session.goal_dag.goals[0].goal_id
    assert session.grounded_value_sources_by_goal[goal_id]["enabled"] == (
        "ambient-explicit-off-user"
    )
    assert goal_id not in session.derived_value_evidence_by_goal


def test_missing_airflow_direction_asks_before_planning_and_binds_followup() -> None:
    client = FakeStructuredClient(
        {
            **action_intent(
                [
                    {
                        "semantic_operation": "set_fan_airflow_direction",
                        "desired_outcome": {},
                    }
                ],
                explicit_slots={},
            ),
            "ambiguities": [
                {
                    "name": "direction",
                    "candidate_values": [
                        "UP",
                        "DOWN",
                        "LEFT",
                        "RIGHT",
                        "FACE",
                        "FEET",
                    ],
                    "candidate_labels": [
                        "up",
                        "down",
                        "left",
                        "right",
                        "face",
                        "feet",
                    ],
                    "goal_index": 0,
                }
            ],
        },
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_fan_airflow_direction",
            tool_name="set_fan_airflow_direction",
            arguments={"direction": "windshield"},
        ),
    )
    runtime = runtime_for(client)
    question = runtime.handle_event(
        user_event(
            context_id="airflow-live-enum",
            message_id="airflow-live-enum-user",
            text="Change the fan airflow direction.",
            tools=(AIRFLOW_DIRECTION_TOOL,),
        )
    )

    assert question.tool_calls == ()
    assert question.text is not None and "windshield" in question.text.casefold()
    assert " up" not in question.text.casefold()
    assert client.action_calls == 0

    ambiguous = runtime.handle_event(
        InboundEvent(
            context_id="airflow-live-enum",
            message_id="airflow-live-enum-ambiguous",
            user_text="Direct the air to the windshield and feet.",
        )
    )
    assert ambiguous.tool_calls == ()
    assert ambiguous.text is not None and "windshield" in ambiguous.text.casefold()
    assert client.action_calls == 0

    action = runtime.handle_event(
        InboundEvent(
            context_id="airflow-live-enum",
            message_id="airflow-live-enum-choice",
            user_text="Direct the air to the windshield.",
        )
    )

    assert action.tool_calls == (
        {
            "tool_name": "set_fan_airflow_direction",
            "arguments": {"direction": "WINDSHIELD"},
        },
    )
    assert client.action_calls == 1


def test_grounded_live_enum_value_removes_conflicting_model_ambiguity() -> None:
    client = FakeStructuredClient(
        {
            **action_intent(
                [
                    {
                        "semantic_operation": "set_fan_airflow_direction",
                        "desired_outcome": {"airflow_direction": "windshield"},
                    }
                ],
                explicit_slots={"airflow_direction": "windshield"},
            ),
            "ambiguities": [
                {
                    "name": "direction",
                    "candidate_values": ["UP", "DOWN"],
                    "candidate_labels": ["up", "down"],
                    "goal_index": 0,
                }
            ],
        },
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_fan_airflow_direction",
            tool_name="set_fan_airflow_direction",
            arguments={"direction": "windshield"},
        ),
    )
    runtime = runtime_for(client)

    action = runtime.handle_event(
        user_event(
            context_id="airflow-grounded-over-model-ambiguity",
            message_id="airflow-grounded-user",
            text="Direct the air to the windshield.",
            tools=(AIRFLOW_DIRECTION_TOOL,),
        )
    )

    assert action.tool_calls == (
        {
            "tool_name": "set_fan_airflow_direction",
            "arguments": {"direction": "WINDSHIELD"},
        },
    )
    session = runtime.sessions.get("airflow-grounded-over-model-ambiguity")
    assert session is not None and session.intent is not None
    assert session.intent.unresolved_slots == []


def test_exact_compound_enum_choice_wins_before_phrase_overlap() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_airflow_direction",
                    "desired_outcome": {},
                }
            ],
            explicit_slots={},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_fan_airflow_direction",
            tool_name="set_fan_airflow_direction",
            arguments={"direction": "WINDSHIELD_HEAD"},
        ),
    )
    runtime = runtime_for(client)
    runtime.handle_event(
        user_event(
            context_id="airflow-compound-choice",
            message_id="airflow-compound-user",
            text="Change the fan airflow direction.",
            tools=(AIRFLOW_DIRECTION_TOOL,),
        )
    )

    action = runtime.handle_event(
        InboundEvent(
            context_id="airflow-compound-choice",
            message_id="airflow-compound-choice",
            user_text="windshield head",
        )
    )

    assert action.tool_calls == (
        {
            "tool_name": "set_fan_airflow_direction",
            "arguments": {"direction": "WINDSHIELD_HEAD"},
        },
    )


def test_same_named_live_slots_are_resolved_by_goal_id() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_airflow_direction",
                    "desired_outcome": {},
                },
                {
                    "semantic_operation": "set_fan_airflow_direction",
                    "desired_outcome": {},
                },
            ],
            explicit_slots={},
        ),
        lambda payload: pytest.fail("both goal-scoped slots must resolve first"),
    )
    runtime = runtime_for(client)
    first_question = runtime.handle_event(
        user_event(
            context_id="airflow-two-goals",
            message_id="airflow-two-goals-user",
            text=(
                "Change the front fan airflow direction and change the rear fan "
                "airflow direction."
            ),
            tools=(AIRFLOW_DIRECTION_TOOL,),
        )
    )
    assert first_question.tool_calls == ()

    session = runtime.sessions.get("airflow-two-goals")
    assert session is not None and session.intent is not None
    slots = session.intent.unresolved_slots
    assert len(slots) == 2
    assert [slot.name for slot in slots] == ["direction", "direction"]
    assert slots[0].goal_id != slots[1].goal_id
    first_goal_id = slots[0].goal_id
    second_goal_id = slots[1].goal_id
    assert first_goal_id is not None and second_goal_id is not None

    second_question = runtime.handle_event(
        InboundEvent(
            context_id="airflow-two-goals",
            message_id="airflow-two-goals-first-choice",
            user_text="feet",
        )
    )

    assert second_question.tool_calls == ()
    assert second_question.text is not None
    goals = {goal.goal_id: goal for goal in session.intent.goals}
    assert goals[first_goal_id].desired_outcome == {"direction": "FEET"}
    assert goals[second_goal_id].desired_outcome == {}
    assert session.grounded_desired_values_by_goal[first_goal_id] == {
        "direction": "FEET"
    }
    assert session.grounded_desired_values_by_goal[second_goal_id] == {}
    assert session.grounded_value_sources_by_goal[first_goal_id] == {
        "direction": "airflow-two-goals-first-choice"
    }
    assert session.grounded_value_sources_by_goal[second_goal_id] == {}
    pending = next(iter(session.pending_clarifications.values()))
    assert pending.goal_id == second_goal_id
    assert client.action_calls == 0


def test_stale_goal_bound_clarification_cannot_mutate_or_execute() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_airflow_direction",
                    "desired_outcome": {},
                }
            ],
            explicit_slots={},
        ),
        lambda payload: pytest.fail("a stale clarification must not reach planning"),
    )
    runtime = runtime_for(client)
    runtime.handle_event(
        user_event(
            context_id="airflow-stale-clarification",
            message_id="airflow-stale-user",
            text="Change the fan airflow direction.",
            tools=(AIRFLOW_DIRECTION_TOOL,),
        )
    )
    session = runtime.sessions.get("airflow-stale-clarification")
    assert session is not None and session.intent is not None
    before = session.intent.model_dump(mode="json")
    request_id, request = next(iter(session.pending_clarifications.items()))
    session.pending_clarifications[request_id] = request.model_copy(
        update={"goal_id": "stale-goal-id"}, deep=True
    )

    rejected = runtime.handle_event(
        InboundEvent(
            context_id="airflow-stale-clarification",
            message_id="airflow-stale-choice",
            user_text="feet",
        )
    )

    assert rejected.tool_calls == ()
    assert rejected.text is not None
    assert session.intent.model_dump(mode="json") == before
    assert session.pending_clarifications == {}
    assert client.action_calls == 0


def test_missing_target_tool_does_not_invent_live_enum_choices() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_airflow_direction",
                    "desired_outcome": {},
                }
            ],
            explicit_slots={},
        ),
        lambda payload: pytest.fail("the planner must not run without the target tool"),
    )
    runtime = runtime_for(client)

    response = runtime.handle_event(
        user_event(
            context_id="airflow-missing-target",
            message_id="airflow-missing-target-user",
            text="Change the fan airflow direction.",
            tools=(FAN_TOOL,),
        )
    )

    assert response.tool_calls == ()
    assert response.text is not None
    assert "WINDSHIELD" not in response.text
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "direction_schema",
    [
        {"type": "string"},
        {"type": "string", "enum": ["FEET"]},
        {"type": "string", "enum": ["FEET", 1]},
    ],
)
def test_non_enumerable_missing_string_never_reaches_planner(
    direction_schema: dict[str, Any],
) -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_airflow_direction",
                    "desired_outcome": {},
                }
            ],
            explicit_slots={},
        ),
        lambda payload: pytest.fail("the planner must not invent a direction"),
    )
    runtime = runtime_for(client)
    direction_tool = tool(
        "set_fan_airflow_direction",
        {"direction": direction_schema},
        ["direction"],
    )

    response = runtime.handle_event(
        user_event(
            context_id=f"airflow-no-enum-{len(direction_schema)}",
            message_id=f"airflow-no-enum-user-{len(direction_schema)}",
            text="Change the fan airflow direction.",
            tools=(direction_tool,),
        )
    )

    assert response.tool_calls == ()
    assert response.text is not None
    assert client.action_calls == 0


def test_non_ascii_enum_choice_requires_an_exact_unicode_spelling() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fan_airflow_direction",
                    "desired_outcome": {},
                }
            ],
            explicit_slots={},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="set_fan_airflow_direction",
            tool_name="set_fan_airflow_direction",
            arguments={"direction": "BLÅ"},
        ),
    )
    runtime = runtime_for(client)
    unicode_tool = tool(
        "set_fan_airflow_direction",
        {"direction": {"type": "string", "enum": ["RED", "BLÅ"]}},
        ["direction"],
    )
    runtime.handle_event(
        user_event(
            context_id="airflow-unicode-enum",
            message_id="airflow-unicode-user",
            text="Change the fan airflow direction.",
            tools=(unicode_tool,),
        )
    )

    rejected = runtime.handle_event(
        InboundEvent(
            context_id="airflow-unicode-enum",
            message_id="airflow-unicode-wrong",
            user_text="blé",
        )
    )
    assert rejected.tool_calls == ()
    assert rejected.text is not None
    assert client.action_calls == 0

    accepted = runtime.handle_event(
        InboundEvent(
            context_id="airflow-unicode-enum",
            message_id="airflow-unicode-exact",
            user_text="blå",
        )
    )
    assert accepted.tool_calls == (
        {
            "tool_name": "set_fan_airflow_direction",
            "arguments": {"direction": "BLÅ"},
        },
    )


def test_contexts_keep_goals_pending_calls_and_results_isolated() -> None:
    def fan_client(level: int) -> FakeStructuredClient:
        return FakeStructuredClient(
            action_intent(
                [
                    {
                        "semantic_operation": "set_fan_speed",
                        "desired_outcome": {"level": level},
                    }
                ],
                explicit_slots={"level": level},
            ),
            lambda payload: set_proposal(
                payload,
                semantic_operation="set_fan_speed",
                tool_name="set_fan_speed",
                arguments={"level": level},
            ),
        )

    clients = {"ctx-a": fan_client(1), "ctx-b": fan_client(3)}
    runtime = CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False),
        client_factory=lambda session: clients[session.context_id],
    )
    first_a = runtime.handle_event(
        user_event(
            context_id="ctx-a",
            message_id="same-message-id",
            text="Set my fan to level one.",
            tools=(FAN_TOOL,),
        )
    )
    first_b = runtime.handle_event(
        user_event(
            context_id="ctx-b",
            message_id="same-message-id",
            text="Set my fan to level three.",
            tools=(FAN_TOOL,),
        )
    )
    assert first_a.tool_calls[0]["arguments"] == {"level": 1}
    assert first_b.tool_calls[0]["arguments"] == {"level": 3}

    final_a = runtime.handle_event(
        result_event(
            context_id="ctx-a",
            message_id="same-result-id",
            tool_name="set_fan_speed",
            content={"status": "SUCCESS", "result": {"level": 1}},
        )
    )
    assert final_a.text is not None and final_a.text.startswith("Done")
    session_a = runtime.sessions.get("ctx-a")
    session_b = runtime.sessions.get("ctx-b")
    assert session_a is not None and session_b is not None
    assert session_a.goal_dag.goals[0].status.value == "done"
    assert session_b.goal_dag.goals[0].status.value != "done"
    assert len(session_b.pending_calls) == 1

    final_b = runtime.handle_event(
        result_event(
            context_id="ctx-b",
            message_id="same-result-id",
            tool_name="set_fan_speed",
            content={"status": "SUCCESS", "result": {"level": 3}},
        )
    )
    assert final_b.text is not None and final_b.text.startswith("Done")
    assert clients["ctx-a"].action_calls == 1
    assert clients["ctx-b"].action_calls == 1


def test_terminal_tool_call_tombstone_never_replays_the_tool() -> None:
    phone_number = "+12025550123"
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "call_phone_by_number",
                    "desired_outcome": {
                        "contact_ids": ["contact-1"],
                        "phone_number": phone_number,
                    },
                }
            ],
            explicit_slots={
                "contact_ids": ["contact-1"],
                "phone_number": phone_number,
            },
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="call_phone_by_number",
            tool_name="call_phone_by_number",
            arguments={"phone_number": phone_number},
        ),
    )
    runtime = runtime_for(client)
    phone_tools = (
        tool(
            "get_contact_information",
            {"contact_ids": {"type": "array", "items": {"type": "string"}}},
            ["contact_ids"],
        ),
        tool(
            "call_phone_by_number",
            {"phone_number": {"type": "string"}},
            ["phone_number"],
        ),
    )
    initial = runtime.handle_event(
        user_event(
            context_id="phone",
            message_id="phone-user",
            text=("Call contact-1 at phone number +12025550123."),
            tools=phone_tools,
        )
    )
    assert initial.tool_calls[0]["tool_name"] == "get_contact_information"

    contact_result = result_event(
        context_id="phone",
        message_id="phone-contact-result",
        tool_name="get_contact_information",
        content={
            "status": "SUCCESS",
            "result": {
                "contact_ids": ["contact-1"],
                "phone_numbers": [phone_number],
            },
        },
    )
    terminal_call = runtime.handle_event(contact_result)
    assert terminal_call.text is None
    assert terminal_call.tool_calls == (
        {
            "tool_name": "call_phone_by_number",
            "arguments": {"phone_number": phone_number},
        },
    )
    assert terminal_call.terminal
    assert runtime.sessions.get("phone") is None
    tombstone = runtime.sessions.get_tombstone("phone")
    assert tombstone is not None and tombstone.terminal
    assert tombstone.replay_for("phone-contact-result") is None

    replay = runtime.handle_event(contact_result)
    assert replay.tool_calls == ()
    assert replay.text is not None
    assert replay.terminal
    assert replay is not terminal_call
    assert client.action_calls == 1


@pytest.mark.parametrize(
    ("utterance", "reported_operation"),
    [
        ("Turn on the high beams.", "set_low_beams"),
        ("Turn on the low beams.", "set_high_beams"),
        ("Turn on the beams.", "set_high_beams"),
        ("Turn on the beams.", "set_low_beams"),
        ("What are high beams? Turn on the low beams.", "set_high_beams"),
        ("Turn on the low beams but not the high beams.", "set_high_beams"),
        ("Turn on the low beams rather than the high beams.", "set_high_beams"),
    ],
)
def test_model_reported_beam_sibling_is_rejected_before_planning(
    utterance: str,
    reported_operation: str,
) -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": reported_operation,
                    "desired_outcome": {"enabled": True},
                }
            ],
            explicit_slots={"enabled": True},
        ),
        lambda payload: pytest.fail(
            f"the planner must not receive the mismatched goal: {payload}"
        ),
    )
    runtime = runtime_for(client)
    exterior_light_tools = (
        tool("get_exterior_lights_status"),
        tool(
            "set_head_lights_low_beams",
            {"on": {"type": "boolean"}},
            ["on"],
        ),
        tool(
            "set_head_lights_high_beams",
            {"on": {"type": "boolean"}},
            ["on"],
        ),
    )

    response = runtime.handle_event(
        user_event(
            context_id=f"beam-sibling-{reported_operation}",
            message_id=f"beam-sibling-user-{reported_operation}",
            text=utterance,
            tools=exterior_light_tools,
        )
    )

    assert response.tool_calls == ()
    assert response.text is not None
    assert client.action_calls == 0


NAVIGATION_FAMILY_TOOLS = (
    tool(
        "get_current_navigation_state",
        {"detailed_information": {"type": "boolean"}},
    ),
    tool(
        "navigation_delete_destination",
        {"destination_id_to_delete": {"type": "string"}},
        ["destination_id_to_delete"],
    ),
    tool(
        "navigation_delete_waypoint",
        {
            "waypoint_id_to_delete": {"type": "string"},
            "route_id_without_waypoint": {"type": "string"},
        },
        ["waypoint_id_to_delete", "route_id_without_waypoint"],
    ),
    tool("delete_current_navigation"),
    tool(
        "navigation_replace_final_destination",
        {
            "new_destination_id": {"type": "string"},
            "route_id_leading_to_new_destination": {"type": "string"},
        },
        ["new_destination_id", "route_id_leading_to_new_destination"],
    ),
    tool(
        "navigation_replace_one_waypoint",
        {
            "waypoint_id_to_replace": {"type": "string"},
            "new_waypoint_id": {"type": "string"},
            "route_id_leading_to_new_waypoint": {"type": "string"},
            "route_id_leading_away_from_new_waypoint": {"type": "string"},
        },
        [
            "waypoint_id_to_replace",
            "new_waypoint_id",
            "route_id_leading_to_new_waypoint",
            "route_id_leading_away_from_new_waypoint",
        ],
    ),
)

NAVIGATION_REPLACEMENT_TOOLS = (
    *NAVIGATION_FAMILY_TOOLS,
    tool(
        "get_location_id_by_location_name",
        {"location": {"type": "string"}},
        ["location"],
    ),
    tool(
        "get_routes_from_start_to_destination",
        {
            "start_id": {"type": "string"},
            "destination_id": {"type": "string"},
        },
        ["start_id", "destination_id"],
    ),
)

TRIP_RANGE_TOOLS = (
    tool("get_charging_specs_and_status"),
    tool(
        "get_location_id_by_location_name",
        {"location": {"type": "string"}},
        ["location"],
    ),
    tool(
        "get_routes_from_start_to_destination",
        {
            "start_id": {"type": "string"},
            "destination_id": {"type": "string"},
        },
        ["start_id", "destination_id"],
    ),
)


@pytest.mark.parametrize(
    "request_text",
    (
        "Find navigation to Northbridge and check my battery.",
        "Search for routes to Northbridge. Also check my battery range.",
        "Show me routes to Northbridge and tell me if I have enough charge.",
        "Navigate to Northbridge and check my battery range.",
        (
            "Can you please find me directions to Northbridge and also check "
            "my car's battery range for that trip?"
        ),
        (
            "Can you please find me directions to Northbridge? And also, can "
            "you check if my car has enough battery for that trip?"
        ),
        (
            "I need to go to Northbridge. Can you please find me directions "
            "and also tell me if my car has enough battery for the trip?"
        ),
        (
            "I need to go to Northbridge. Can you show me routes there? And "
            "also, can you check if I have enough battery for the journey?"
        ),
        (
            "Can you show me routes to Northbridge? And also, can you check "
            "my battery range for that trip?"
        ),
        (
            "Can you find me a route to Northbridge and also check if my car "
            "has enough battery for that trip?"
        ),
        (
            "Can you find me a route to Northbridge, and also check if my car "
            "has enough battery for that trip?"
        ),
        "Find me a route to Northbridge, then check my battery range.",
        "Show me route options to Northbridge and check my battery range.",
        (
            "I need to go to Northbridge. Can you find me a route there and "
            "also check if my car has enough battery for that trip?"
        ),
    ),
)
def test_route_preview_and_range_requests_enter_read_only_workflow_without_model(
    request_text: str,
) -> None:
    client = FailingActionClient(
        {
            "language": "en",
            "intent_kind": "conversation",
            "call_for_action": False,
            "goals": [],
            "explicit_slots": {},
        },
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    initial = runtime.handle_event(
        InboundEvent(
            message_id=f"route-range-{len(request_text)}",
            context_id=f"route-range-{len(request_text)}",
            system_policy=(
                "Follow the current safety policy. "
                'CURRENT_LOCATION={"id":"loc_origin","name":"Origin"}'
            ),
            user_text=request_text,
            live_tools=TRIP_RANGE_TOOLS,
        )
    )

    assert initial.tool_calls
    assert {call["tool_name"] for call in initial.tool_calls}.issubset(
        {
            "get_charging_specs_and_status",
            "get_location_id_by_location_name",
            "get_routes_from_start_to_destination",
        }
    )
    assert client.intent_calls == 0
    assert client.action_calls == 0
    session = runtime.sessions.get(f"route-range-{len(request_text)}")
    assert session is not None and session.intent is not None
    assert not session.intent.call_for_action
    assert session.intent.intent_kind.value == "information"
    assert [goal.semantic_operation for goal in session.intent.goals] == [
        "assess_battery_charge_trip_range"
    ]
    assert session.budget.attempted_sets == set()


@pytest.mark.parametrize(
    "request_text",
    (
        "Navigate to Northbridge.",
        "Do not find routes to Northbridge and check my battery.",
        "If needed, find routes to Northbridge and check my battery.",
        'The driver said "find routes to Northbridge and check my battery".',
        "Find routes to Northbridge and check my battery, then call Alex.",
        "Find routes to Paris and Berlin and check my battery.",
        "Find routes to Paris or Berlin and check my battery.",
        ("Find routes to Paris and turn on the headlights and check my battery range."),
        "Find routes to Paris and open the windows and check my battery range.",
        "Find routes to Paris then Berlin then check my battery.",
        "Find routes to Paris then adjust the fan then check my battery.",
        "Check my battery range.",
    ),
)
def test_route_preview_and_range_parser_rejects_nonexact_or_unsafe_scope(
    request_text: str,
) -> None:
    assert (
        _trip_range_information_intent(
            request_text,
            turn_id="synthetic-route-range",
            language="en",
        )
        is None
    )


def _base34_navigation_state() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [
                "loc_bel_437063",
                "loc_buc_567170",
                "loc_rom_294918",
            ],
            "routes_to_final_destination_id": [
                "rll_bel_buc_632742",
                "rll_buc_rom_989928",
            ],
            "details": {
                "waypoints": [
                    {"id": "loc_bel_437063", "name": "Belgrade"},
                    {"id": "loc_buc_567170", "name": "Bucharest"},
                    {"id": "loc_rom_294918", "name": "Rome"},
                ]
            },
        },
    }


def _route_result(
    *, start_id: str, destination_id: str, fastest_id: str, other_id: str
) -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": other_id,
                    "start_id": start_id,
                    "destination_id": destination_id,
                    "name_via": "A10",
                    "distance_km": 110.0,
                    "duration_hours": 2,
                    "duration_minutes": 0,
                    "road_types": ["highway"],
                    "alias": ["first"],
                    "includes_toll": False,
                },
                {
                    "route_id": fastest_id,
                    "start_id": start_id,
                    "destination_id": destination_id,
                    "name_via": "B20",
                    "distance_km": 100.0,
                    "duration_hours": 1,
                    "duration_minutes": 0,
                    "road_types": ["country road"],
                    "alias": ["second", "fastest", "shortest"],
                    "includes_toll": False,
                },
            ]
        },
    }


def _trip_route_result(
    *,
    start_id: str = "loc_mil_253463",
    destination_id: str = "loc_pra_198238",
    distance_km: float = 768.49,
) -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": "rll_mil_pra_111111",
                    "start_id": start_id,
                    "destination_id": destination_id,
                    "name_via": "B2",
                    "distance_km": distance_km + 10,
                    "duration_hours": 8,
                    "duration_minutes": 10,
                    "road_types": ["highway"],
                    "includes_toll": False,
                    "alias": ["second"],
                },
                {
                    "route_id": "rll_mil_pra_222222",
                    "start_id": start_id,
                    "destination_id": destination_id,
                    "name_via": "A1",
                    "distance_km": distance_km,
                    "duration_hours": 8,
                    "duration_minutes": 0,
                    "road_types": ["highway"],
                    "includes_toll": False,
                    "alias": ["fastest", "first", "shortest"],
                },
            ]
        },
    }


def _synthetic_trip_route_result(
    *,
    start_id: str,
    destination_id: str,
    alternative_route_id: str,
    preferred_route_id: str,
    distance_km: float = 200,
    alternative_via: str = "B2",
    preferred_via: str = "A1",
) -> dict[str, Any]:
    result = _trip_route_result(
        start_id=start_id,
        destination_id=destination_id,
        distance_km=distance_km,
    )
    routes = result["result"]["routes"]
    routes[0].update(
        {"route_id": alternative_route_id, "name_via": alternative_via}
    )
    routes[1].update({"route_id": preferred_route_id, "name_via": preferred_via})
    return result


def test_trip_range_feasibility_uses_goal_bound_status_location_and_route() -> None:
    client = FailingActionClient(
        {
            "language": "en",
            "intent_kind": "information",
            "call_for_action": False,
            "goals": [
                {
                    "semantic_operation": "resolve_location",
                    "desired_outcome": {"location": "Prague"},
                },
                {
                    "semantic_operation": "get_ev_range",
                    "desired_outcome": {},
                },
            ],
            "explicit_slots": {
                "current_location": "Milan",
                "destination": "Prague",
            },
            "explicit_constraints": {"no_stops": True},
        },
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "trip-range-base38-derived"
    initial = runtime.handle_event(
        InboundEvent(
            message_id="trip-range-user",
            context_id=context_id,
            system_policy=(
                "Follow the current safety policy. "
                'CURRENT_LOCATION={"id":"loc_mil_253463","name":"Milan"}'
            ),
            user_text=(
                "Hey there! I'm in Milan right now and I'm thinking of heading to "
                "Prague. Can you tell me if I can make it all the way there without "
                "having to stop and charge up?"
            ),
            live_tools=TRIP_RANGE_TOOLS,
        )
    )
    assert initial.tool_calls == (
        {"tool_name": "get_charging_specs_and_status", "arguments": {}},
    )

    location = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="trip-range-charging",
            tool_name="get_charging_specs_and_status",
            tool_call_id="evaluator-trip-charging",
            content={
                "status": "SUCCESS",
                "result": {
                    "battery_capacity_kwh": 60,
                    "max_charging_power_ac": 22,
                    "max_charging_power_dc": 250,
                    "state_of_charge": 50.0,
                    "remaining_range": "178.0km",
                },
            },
        )
    )
    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Prague"},
        },
    )

    route = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="trip-range-location",
            tool_name="get_location_id_by_location_name",
            tool_call_id="evaluator-trip-location",
            content={
                "status": "SUCCESS",
                "result": {"id": "loc_pra_198238"},
            },
        )
    )
    assert route.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_mil_253463",
                "destination_id": "loc_pra_198238",
            },
        },
    )

    answer = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="trip-range-route",
            tool_name="get_routes_from_start_to_destination",
            tool_call_id="evaluator-trip-route",
            content=_trip_route_result(),
        )
    )
    assert answer.tool_calls == ()
    assert answer.text is not None
    assert "50 percent" in answer.text
    assert "178" in answer.text
    assert "Prague" in answer.text and "768.49" in answer.text
    assert "cannot reach" in answer.text
    assert "need to charge" in answer.text
    assert client.intent_calls == 1
    assert client.action_calls == 0
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()


def test_route_range_preview_binds_second_route_before_assessing_range() -> None:
    client = FailingActionClient(
        {
            "language": "en",
            "intent_kind": "conversation",
            "call_for_action": False,
            "goals": [],
            "explicit_slots": {},
        },
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "route-range-select-second"
    start_id = "loc_ori_123456"
    destination_id = "loc_nor_654321"
    alternative_route_id = "rll_ori_nor_111111"
    initial = runtime.handle_event(
        InboundEvent(
            message_id="route-range-user",
            context_id=context_id,
            system_policy=(
                "Follow the current safety policy. "
                f'CURRENT_LOCATION={{"id":"{start_id}","name":"Origin"}}'
            ),
            user_text=(
                "Can you find me a route to Northbridge and also check if my car "
                "has enough battery for that trip?"
            ),
            live_tools=TRIP_RANGE_TOOLS,
        )
    )
    calls = {call["tool_name"]: call for call in initial.tool_calls}
    assert set(calls) == {
        "get_charging_specs_and_status",
        "get_location_id_by_location_name",
    }

    route = runtime.handle_event(
        InboundEvent(
            message_id="route-range-initial-results",
            context_id=context_id,
            tool_results=(
                {
                    "toolName": "get_location_id_by_location_name",
                    "content": {
                        "status": "SUCCESS",
                        "result": {"id": destination_id},
                    },
                },
                {
                    "toolName": "get_charging_specs_and_status",
                    "content": {
                        "status": "SUCCESS",
                        "result": {
                            "battery_capacity_kwh": 60,
                            "max_charging_power_ac": 22,
                            "max_charging_power_dc": 250,
                            "state_of_charge": 50.0,
                            "remaining_range": "178.0km",
                        },
                    },
                },
            ),
        )
    )
    assert route.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": start_id,
                "destination_id": destination_id,
            },
        },
    )

    presented = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="route-range-routes",
            tool_name="get_routes_from_start_to_destination",
            content=_synthetic_trip_route_result(
                start_id=start_id,
                destination_id=destination_id,
                alternative_route_id=alternative_route_id,
                preferred_route_id="rll_ori_nor_222222",
                alternative_via="B2",
                preferred_via="A1",
            ),
        )
    )
    assert presented.tool_calls == ()
    assert presented.text is not None
    assert "fastest and shortest" in presented.text
    assert "further route alternative" in presented.text

    assessed = runtime.handle_event(
        InboundEvent(
            message_id="route-range-selection",
            context_id=context_id,
            user_text="Use the second route via B2.",
        )
    )
    assert assessed.tool_calls == ()
    assert assessed.text is not None
    assert "50 percent" in assessed.text
    assert "210" in assessed.text
    assert "cannot reach" in assessed.text
    assert client.intent_calls == 0
    assert client.action_calls == 0
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    summary = runtime._validated_trip_route_summary(  # noqa: SLF001
        session, require_range_shortfall=True
    )
    assert summary is not None
    assert summary[1]["route_id"] == alternative_route_id


@pytest.mark.parametrize(
    "route_reply",
    (
        "Use the second route via B2. Start navigation.",
        "Can you start navigation on the second route via B2?",
        "Set my navigation to the second route via B2.",
        "Set navigation to the second route via B2.",
    ),
)
def test_route_range_choice_rejects_navigation_action_upgrade(
    route_reply: str,
) -> None:
    client = FailingActionClient(
        {
            "language": "en",
            "intent_kind": "conversation",
            "call_for_action": False,
            "goals": [],
            "explicit_slots": {},
        },
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = f"route-range-action-upgrade-{len(route_reply)}"
    start_id = "loc_ori_123456"
    destination_id = "loc_nor_654321"
    runtime.handle_event(
        InboundEvent(
            message_id="route-range-user",
            context_id=context_id,
            system_policy=(
                "Follow the current safety policy. "
                f'CURRENT_LOCATION={{"id":"{start_id}","name":"Origin"}}'
            ),
            user_text=(
                "Can you find me a route to Northbridge and also check if my car "
                "has enough battery for that trip?"
            ),
            live_tools=TRIP_RANGE_TOOLS,
        )
    )
    runtime.handle_event(
        InboundEvent(
            message_id="route-range-initial-results",
            context_id=context_id,
            tool_results=(
                {
                    "toolName": "get_location_id_by_location_name",
                    "content": {
                        "status": "SUCCESS",
                        "result": {"id": destination_id},
                    },
                },
                {
                    "toolName": "get_charging_specs_and_status",
                    "content": {
                        "status": "SUCCESS",
                        "result": {
                            "battery_capacity_kwh": 60,
                            "max_charging_power_ac": 22,
                            "max_charging_power_dc": 250,
                            "state_of_charge": 50.0,
                            "remaining_range": "178.0km",
                        },
                    },
                },
            ),
        )
    )
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="route-range-routes",
            tool_name="get_routes_from_start_to_destination",
            content=_synthetic_trip_route_result(
                start_id=start_id,
                destination_id=destination_id,
                alternative_route_id="rll_ori_nor_111111",
                preferred_route_id="rll_ori_nor_222222",
            ),
        )
    )

    rejected = runtime.handle_event(
        InboundEvent(
            message_id="route-range-action-upgrade",
            context_id=context_id,
            user_text=route_reply,
        )
    )

    assert rejected.tool_calls == ()
    assert rejected.text is not None
    assert "route choice alias" in rejected.text.casefold()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert (
        runtime._validated_trip_route_summary(  # noqa: SLF001
            session, require_range_shortfall=True
        )
        is None
    )
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_trip_range_feasibility_rejects_route_for_other_endpoints() -> None:
    client = FailingActionClient(
        {
            "language": "en",
            "intent_kind": "information",
            "call_for_action": False,
            "goals": [
                {
                    "semantic_operation": "assess_battery_charge_trip_range",
                    "desired_outcome": {"destination_name": "Prague"},
                }
            ],
            "explicit_slots": {"destination_name": "Prague"},
        },
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "trip-range-wrong-route"
    runtime.handle_event(
        InboundEvent(
            message_id="trip-range-wrong-user",
            context_id=context_id,
            system_policy=(
                "Follow the current safety policy. "
                'CURRENT_LOCATION={"id":"loc_mil_253463","name":"Milan"}'
            ),
            user_text="Can my battery range get me to Prague without charging?",
            live_tools=TRIP_RANGE_TOOLS,
        )
    )
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="trip-range-wrong-charging",
            tool_name="get_charging_specs_and_status",
            content={
                "status": "SUCCESS",
                "result": {
                    "state_of_charge": 50,
                    "remaining_range": "178km",
                },
            },
        )
    )
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="trip-range-wrong-location",
            tool_name="get_location_id_by_location_name",
            content={
                "status": "SUCCESS",
                "result": {"id": "loc_pra_198238"},
            },
        )
    )
    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="trip-range-wrong-route-result",
            tool_name="get_routes_from_start_to_destination",
            content=_trip_route_result(start_id="loc_wrong_123456"),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "couldn't verify" in blocked.text
    assert client.action_calls == 0


def test_trip_range_feasibility_does_not_infer_missing_route_result() -> None:
    client = FailingActionClient(
        {
            "language": "en",
            "intent_kind": "information",
            "call_for_action": False,
            "goals": [
                {
                    "semantic_operation": "resolve_location",
                    "desired_outcome": {"location": "Prague"},
                },
                {"semantic_operation": "get_ev_range", "desired_outcome": {}},
            ],
            "explicit_slots": {"destination": "Prague"},
        },
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "trip-range-missing-route-result"
    runtime.handle_event(
        InboundEvent(
            message_id="trip-range-missing-user",
            context_id=context_id,
            system_policy=(
                "Follow the current safety policy. "
                'CURRENT_LOCATION={"id":"loc_mil_253463","name":"Milan"}'
            ),
            user_text="Can you tell me if I can make it to Prague without charging?",
            live_tools=TRIP_RANGE_TOOLS,
        )
    )
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="trip-range-missing-charging",
            tool_name="get_charging_specs_and_status",
            content={
                "status": "SUCCESS",
                "result": {
                    "state_of_charge": 50.0,
                    "remaining_range": "178km",
                },
            },
        )
    )
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="trip-range-missing-location",
            tool_name="get_location_id_by_location_name",
            content={
                "status": "SUCCESS",
                "result": {"id": "loc_pra_198238"},
            },
        )
    )
    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="trip-range-missing-route",
            tool_name="get_routes_from_start_to_destination",
            content={"status": "SUCCESS", "result": {}},
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "couldn't verify" in blocked.text
    assert "768" not in blocked.text and "178" not in blocked.text
    assert client.action_calls == 0


def test_navigation_waypoint_replacement_is_derived_from_exact_read_chain() -> None:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "navigation_replace_one_waypoint",
                    "desired_outcome": {
                        "waypoint_id_to_replace": "intermediate_stop",
                        "new_waypoint_id": "Frankfurt",
                    },
                }
            ],
            explicit_slots={"new_waypoint": "Frankfurt"},
        ),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "navigation-replace-derived-chain"

    current = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="navigation-replace-user",
            text="Hey, can you replace my current intermediate stop with Frankfurt?",
            tools=NAVIGATION_REPLACEMENT_TOOLS,
        )
    )
    assert current.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )

    location = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="navigation-replace-current",
            tool_name="get_current_navigation_state",
            content=_base34_navigation_state(),
            tool_call_id="evaluator-navigation-state",
        )
    )
    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Frankfurt"},
        },
    )

    route_to = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="navigation-replace-location",
            tool_name="get_location_id_by_location_name",
            content={
                "status": "SUCCESS",
                "result": {"id": "loc_fra_178468"},
            },
            tool_call_id="evaluator-navigation-location",
        )
    )
    assert route_to.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_bel_437063",
                "destination_id": "loc_fra_178468",
            },
        },
    )

    route_from = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="navigation-replace-route-to",
            tool_name="get_routes_from_start_to_destination",
            content=_route_result(
                start_id="loc_bel_437063",
                destination_id="loc_fra_178468",
                fastest_id="rll_bel_fra_835188",
                other_id="rll_bel_fra_111111",
            ),
            tool_call_id="evaluator-navigation-route-to",
        )
    )
    assert route_from.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_fra_178468",
                "destination_id": "loc_rom_294918",
            },
        },
    )

    replace = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="navigation-replace-route-from",
            tool_name="get_routes_from_start_to_destination",
            content=_route_result(
                start_id="loc_fra_178468",
                destination_id="loc_rom_294918",
                fastest_id="rll_fra_rom_609098",
                other_id="rll_fra_rom_111111",
            ),
            tool_call_id="evaluator-navigation-route-from",
        )
    )
    expected_arguments = {
        "waypoint_id_to_replace": "loc_buc_567170",
        "new_waypoint_id": "loc_fra_178468",
        "route_id_leading_to_new_waypoint": "rll_bel_fra_835188",
        "route_id_leading_away_from_new_waypoint": "rll_fra_rom_609098",
    }
    assert replace.tool_calls == (
        {
            "tool_name": "navigation_replace_one_waypoint",
            "arguments": expected_arguments,
        },
    )
    assert client.action_calls == 0

    session = runtime.sessions.get(context_id)
    assert session is not None
    goal_id = session.goal_dag.goals[0].goal_id
    derived = session.derived_value_evidence_by_goal[goal_id]
    assert set(expected_arguments).issubset(derived)
    assert all(
        session.evidence.evidence[derived[name]].source_kind
        is EvidenceSourceKind.DERIVED
        for name in expected_arguments
    )

    done = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="navigation-replace-done",
            tool_name="navigation_replace_one_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_replaced": True,
                    "new_waypoints": [
                        "loc_bel_437063",
                        "loc_fra_178468",
                        "loc_rom_294918",
                    ],
                    "new_routes": [
                        "rll_bel_fra_835188",
                        "rll_fra_rom_609098",
                    ],
                },
            },
        )
    )
    assert done.text is not None and done.text.startswith("Done")
    assert "fastest route for each segment" in done.text
    assert "alternative routes?" in done.text


def test_navigation_final_destination_delete_uses_current_navigation_id() -> None:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "navigation_delete_final_destination",
                    "desired_outcome": {"destination_id": "Cologne"},
                }
            ],
            explicit_slots={"destination": "Cologne"},
        ),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "navigation-delete-final-derived"

    current = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="navigation-delete-final-user",
            text=(
                "Hey there! Could you please remove the final destination from my "
                "current route? I've changed my mind about going to Cologne."
            ),
            tools=NAVIGATION_FAMILY_TOOLS,
        )
    )
    assert current.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )

    navigation_state = {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [
                "loc_dor_399984",
                "loc_ess_699309",
                "loc_dui_607981",
                "loc_col_464166",
            ],
            "routes_to_final_destination_id": [
                "rll_dor_ess_111111",
                "rll_ess_dui_222222",
                "rll_dui_col_333333",
            ],
            "details": {
                "waypoints": [
                    {"id": "loc_dor_399984", "name": "Dortmund"},
                    {"id": "loc_ess_699309", "name": "Essen"},
                    {"id": "loc_dui_607981", "name": "Duisburg"},
                    {"id": "loc_col_464166", "name": "Cologne"},
                ]
            },
        },
    }
    fresh = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="navigation-delete-final-current",
            tool_name="get_current_navigation_state",
            content=navigation_state,
        )
    )
    assert fresh.tool_calls == (
        {"tool_name": "get_current_navigation_state", "arguments": {}},
    )

    delete = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="navigation-delete-final-fresh",
            tool_name="get_current_navigation_state",
            content=navigation_state,
        )
    )
    assert delete.tool_calls == (
        {
            "tool_name": "navigation_delete_destination",
            "arguments": {"destination_id_to_delete": "loc_col_464166"},
        },
    )
    assert client.action_calls == 0

    session = runtime.sessions.get(context_id)
    assert session is not None
    goal_id = session.goal_dag.goals[0].goal_id
    evidence_id = session.derived_value_evidence_by_goal[goal_id]["destination_id"]
    assert (
        session.evidence.evidence[evidence_id].source_kind is EvidenceSourceKind.DERIVED
    )


def test_navigation_replacement_rejects_route_result_for_other_endpoints() -> None:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "navigation_replace_one_waypoint",
                    "desired_outcome": {"new_waypoint_id": "Frankfurt"},
                }
            ],
            explicit_slots={"new_waypoint": "Frankfurt"},
        ),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "navigation-replace-wrong-route-endpoints"
    runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="wrong-route-user",
            text="Replace the intermediate stop with Frankfurt.",
            tools=NAVIGATION_REPLACEMENT_TOOLS,
        )
    )
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="wrong-route-current",
            tool_name="get_current_navigation_state",
            content=_base34_navigation_state(),
        )
    )
    route_to = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="wrong-route-location",
            tool_name="get_location_id_by_location_name",
            content={
                "status": "SUCCESS",
                "result": {"id": "loc_fra_178468"},
            },
        )
    )
    assert route_to.tool_calls

    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="wrong-route-result",
            tool_name="get_routes_from_start_to_destination",
            content=_route_result(
                start_id="loc_wrong_123456",
                destination_id="loc_fra_178468",
                fastest_id="rll_wrong_fra_123456",
                other_id="rll_wrong_fra_654321",
            ),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_navigation_replacement_disambiguates_intermediate_names_not_ids() -> None:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "navigation_replace_one_waypoint",
                    "desired_outcome": {"new_waypoint_id": "Frankfurt"},
                }
            ],
            explicit_slots={"new_waypoint": "Frankfurt"},
        ),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "navigation-replace-name-clarification"
    runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="navigation-clarify-user",
            text="Replace an intermediate stop with Frankfurt.",
            tools=NAVIGATION_REPLACEMENT_TOOLS,
        )
    )
    clarify = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="navigation-clarify-current",
            tool_name="get_current_navigation_state",
            content={
                "status": "SUCCESS",
                "result": {
                    "navigation_active": True,
                    "waypoints_id": [
                        "loc_bel_437063",
                        "loc_buc_567170",
                        "loc_vie_151617",
                        "loc_rom_294918",
                    ],
                    "routes_to_final_destination_id": [
                        "rll_bel_buc_632742",
                        "rll_buc_vie_111111",
                        "rll_vie_rom_222222",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_bel_437063", "name": "Belgrade"},
                            {"id": "loc_buc_567170", "name": "Bucharest"},
                            {"id": "loc_vie_151617", "name": "Vienna"},
                            {"id": "loc_rom_294918", "name": "Rome"},
                        ]
                    },
                },
            },
        )
    )
    assert clarify.tool_calls == ()
    assert clarify.text is not None
    assert "Bucharest" in clarify.text and "Vienna" in clarify.text
    assert "loc_" not in clarify.text

    location = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="navigation-clarify-choice",
            text="Bucharest.",
            tools=NAVIGATION_REPLACEMENT_TOOLS,
        )
    )
    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Frankfurt"},
        },
    )
    assert client.intent_calls == 1


def test_navigation_delete_rejects_named_destination_state_mismatch() -> None:
    client = FailingActionClient(
        action_intent(
            [
                {
                    "semantic_operation": "navigation_delete_final_destination",
                    "desired_outcome": {"destination_id": "Cologne"},
                }
            ],
            explicit_slots={"destination": "Cologne"},
        ),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "navigation-delete-destination-mismatch"
    current = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="destination-mismatch-user",
            text="Remove the final destination Cologne from my route.",
            tools=NAVIGATION_FAMILY_TOOLS,
        )
    )
    assert current.tool_calls

    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="destination-mismatch-current",
            tool_name="get_current_navigation_state",
            content={
                "status": "SUCCESS",
                "result": {
                    "navigation_active": True,
                    "waypoints_id": [
                        "loc_dor_399984",
                        "loc_ess_699309",
                        "loc_ber_100001",
                    ],
                    "routes_to_final_destination_id": [
                        "rll_dor_ess_111111",
                        "rll_ess_ber_222222",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_dor_399984", "name": "Dortmund"},
                            {"id": "loc_ess_699309", "name": "Essen"},
                            {"id": "loc_ber_100001", "name": "Berlin"},
                        ]
                    },
                },
            },
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()


@pytest.mark.parametrize(
    ("utterance", "reported_operation", "desired_outcome", "explicit_slots"),
    [
        (
            "Delete navigation waypoint old.",
            "navigation_delete_final_destination",
            {"destination_id": "old"},
            {"destination_id": "old"},
        ),
        (
            "Delete navigation waypoint old.",
            "delete_current_navigation",
            {},
            {},
        ),
        (
            "Start navigation to airport.",
            "delete_current_navigation",
            {},
            {},
        ),
        (
            "Replace navigation waypoint old with new.",
            "navigation_replace_final_destination",
            {"new_destination_id": "new"},
            {"new_destination_id": "new"},
        ),
        (
            "Replace final navigation destination old with new.",
            "navigation_replace_one_waypoint",
            {
                "waypoint_id_to_replace": "old",
                "new_waypoint_id": "new",
            },
            {
                "waypoint_id_to_replace": "old",
                "new_waypoint_id": "new",
            },
        ),
    ],
)
def test_navigation_family_substitution_is_rejected_before_read_or_planning(
    utterance: str,
    reported_operation: str,
    desired_outcome: dict[str, Any],
    explicit_slots: dict[str, Any],
) -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": reported_operation,
                    "desired_outcome": desired_outcome,
                }
            ],
            explicit_slots=explicit_slots,
        ),
        lambda payload: pytest.fail(
            f"the planner must not receive a substituted navigation goal: {payload}"
        ),
    )
    runtime = runtime_for(client)

    response = runtime.handle_event(
        user_event(
            context_id=f"navigation-family-{reported_operation}-{len(utterance)}",
            message_id=(
                f"navigation-family-user-{reported_operation}-{len(utterance)}"
            ),
            text=utterance,
            tools=NAVIGATION_FAMILY_TOOLS,
        )
    )

    assert response.tool_calls == ()
    assert response.text is not None
    assert client.action_calls == 0


def test_matching_whole_navigation_delete_still_reads_before_write() -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "delete_current_navigation",
                    "desired_outcome": {},
                }
            ],
            explicit_slots={},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation="delete_current_navigation",
            tool_name="delete_current_navigation",
            arguments={},
        ),
    )
    runtime = runtime_for(client)

    read = runtime.handle_event(
        user_event(
            context_id="matching-whole-navigation-delete",
            message_id="matching-whole-navigation-delete-user",
            text="Delete the current navigation.",
            tools=NAVIGATION_FAMILY_TOOLS,
        )
    )

    assert read.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {},
        },
    )
    assert client.action_calls == 0

    write = runtime.handle_event(
        result_event(
            context_id="matching-whole-navigation-delete",
            message_id="matching-whole-navigation-delete-state",
            tool_name="get_current_navigation_state",
            content={
                "status": "SUCCESS",
                "result": {
                    "navigation_active": True,
                    "waypoints_id": ["old", "destination"],
                },
            },
        )
    )

    assert write.tool_calls == (
        {"tool_name": "delete_current_navigation", "arguments": {}},
    )
    assert client.action_calls == 1


@pytest.mark.parametrize(
    ("utterance", "semantic_operation", "tool_name"),
    [
        (
            "Turn off the high beams.",
            "set_high_beams",
            "set_head_lights_high_beams",
        ),
        (
            "Turn off the low beams.",
            "set_low_beams",
            "set_head_lights_low_beams",
        ),
    ],
)
def test_matching_beam_target_reaches_its_own_live_tool(
    utterance: str,
    semantic_operation: str,
    tool_name: str,
) -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": semantic_operation,
                    "desired_outcome": {"enabled": False},
                }
            ],
            explicit_slots={"enabled": False},
        ),
        lambda payload: set_proposal(
            payload,
            semantic_operation=semantic_operation,
            tool_name=tool_name,
            arguments={"on": False},
        ),
    )
    runtime = runtime_for(client)
    exterior_light_tools = (
        tool(
            "set_head_lights_low_beams",
            {"on": {"type": "boolean"}},
            ["on"],
        ),
        tool(
            "set_head_lights_high_beams",
            {"on": {"type": "boolean"}},
            ["on"],
        ),
    )

    response = runtime.handle_event(
        user_event(
            context_id=f"matching-beam-{semantic_operation}",
            message_id=f"matching-beam-user-{semantic_operation}",
            text=utterance,
            tools=exterior_light_tools,
        )
    )

    assert response.tool_calls == (
        {"tool_name": tool_name, "arguments": {"on": False}},
    )
    assert client.action_calls == 1


@pytest.mark.parametrize(
    "utterance",
    [
        "Set the fan speed to auto.",
        "Set the fan airflow direction to auto.",
    ],
)
def test_model_selected_circulation_cannot_replace_explicit_fan_sibling(
    utterance: str,
) -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_air_circulation",
                    "desired_outcome": {"mode": "AUTO"},
                }
            ],
            explicit_slots={"mode": "AUTO"},
        ),
        lambda payload: pytest.fail(
            f"the planner must not receive a sibling-substituted goal: {payload}"
        ),
    )
    runtime = runtime_for(client)
    circulation_tool = tool(
        "set_air_circulation",
        {
            "mode": {
                "type": "string",
                "enum": ["FRESH_AIR", "RECIRCULATION", "AUTO"],
            }
        },
        ["mode"],
    )

    response = runtime.handle_event(
        user_event(
            context_id=f"fan-sibling-circulation-{len(utterance)}",
            message_id=f"fan-sibling-circulation-user-{len(utterance)}",
            text=utterance,
            tools=(circulation_tool,),
        )
    )

    assert response.tool_calls == ()
    assert response.text is not None
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "utterance",
    ["Turn on the reading light.", "Turn on the ambient lights."],
)
def test_model_selected_fog_light_cannot_replace_interior_light(
    utterance: str,
) -> None:
    client = FakeStructuredClient(
        action_intent(
            [
                {
                    "semantic_operation": "set_fog_lights",
                    "desired_outcome": {"enabled": True},
                }
            ],
            explicit_slots={"enabled": True},
        ),
        lambda payload: pytest.fail(
            f"the planner must not receive a cross-recipe light goal: {payload}"
        ),
    )
    runtime = runtime_for(client)
    fog_tool = tool("set_fog_lights", {"on": {"type": "boolean"}}, ["on"])

    response = runtime.handle_event(
        user_event(
            context_id=f"cross-recipe-fog-{len(utterance)}",
            message_id=f"cross-recipe-fog-user-{len(utterance)}",
            text=utterance,
            tools=(fog_tool,),
        )
    )

    assert response.tool_calls == ()
    assert response.text is not None
    assert client.action_calls == 0


BASE_48_POLICY = """
AUT-POL:017: Tools to delete, replace, or add a waypoint or a destination can
only be used when the navigation system is already active and a route is set.
AUT-POL:018: If navigation is active, replace a destination with the
corresponding edit tool and do not reload the whole navigation system.
CURRENT_LOCATION = {"id":"loc_dor_399984","name":"Dortmund"}
"""

BASE_48_TOOLS = (
    *NAVIGATION_REPLACEMENT_TOOLS,
    tool(
        "search_poi_at_location",
        {
            "location_id": {"type": "string"},
            "category_poi": {
                "type": "string",
                "enum": [
                    "airports",
                    "bakery",
                    "fast_food",
                    "parking",
                    "public_toilets",
                    "restaurants",
                    "supermarkets",
                    "charging_stations",
                ],
            },
            "filters": {"type": "array", "items": {"type": "string"}},
        },
        ["location_id", "category_poi"],
    ),
)


def _base48_client(
    *, replacement_intents: list[dict[str, Any]] | None = None
) -> SequentialIntentClient:
    poi_intent = {
        "language": "en",
        "intent_kind": "information",
        "call_for_action": False,
        "goals": [
            {
                "semantic_operation": "search_poi_at_location",
                "desired_outcome": {
                    "location_id": "Munich",
                    "category": "restaurants",
                },
            }
        ],
        "explicit_slots": {
            "location_id": "Munich",
            "category": "restaurants",
        },
    }
    if replacement_intents is None:
        replacement_intents = [
            action_intent(
                [
                    {
                        "semantic_operation": "navigation_replace_final_destination",
                        "desired_outcome": {"new_destination_id": "Munich"},
                    }
                ],
                explicit_slots={"new_destination_id": "Munich"},
            )
        ]
    return SequentialIntentClient([poi_intent, *replacement_intents])


def _base48_navigation_state() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": ["loc_dor_399984", "loc_düs_892560"],
            "routes_to_final_destination_id": ["rll_dor_düs_836702"],
            "details": {
                "waypoints": [
                    {"id": "loc_dor_399984", "name": "Dortmund"},
                    {"id": "loc_düs_892560", "name": "Düsseldorf"},
                ]
            },
        },
    }


def _base48_poi_result(
    *, corresponding_location_id: str = "loc_mun_9995"
) -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "pois_found": [
                {
                    "id": "poi_res_799429",
                    "name": "Brauhaus Germania",
                    "category": "restaurants",
                    "phone_number": "+49 873 7418665",
                    "corresponding_location_id": corresponding_location_id,
                },
                {
                    "id": "poi_res_263439",
                    "name": "Gasthaus Zum Adler",
                    "category": "restaurants",
                    "phone_number": "+49 480 5581510",
                    "corresponding_location_id": corresponding_location_id,
                },
            ]
        },
    }


def _base48_route_result(
    *,
    start_id: str = "loc_dor_399984",
    destination_id: str = "loc_mun_9995",
    third_alias: str = "third",
    first_distance: Any = 562.25,
    second_via: str = "K816, A46",
) -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": "rll_dor_mun_199807",
                    "start_id": start_id,
                    "destination_id": destination_id,
                    "name_via": "L372, B813",
                    "distance_km": first_distance,
                    "duration_hours": 6,
                    "duration_minutes": 54,
                    "road_types": ["country road", "highway", "urban"],
                    "includes_toll": False,
                    "alias": ["fastest", "first", "shortest"],
                },
                {
                    "route_id": "rll_dor_mun_475855",
                    "start_id": start_id,
                    "destination_id": destination_id,
                    "name_via": second_via,
                    "distance_km": 572.11,
                    "duration_hours": 7,
                    "duration_minutes": 14,
                    "road_types": ["country road", "highway", "urban"],
                    "includes_toll": False,
                    "alias": ["second"],
                },
                {
                    "route_id": "rll_dor_mun_750110",
                    "start_id": start_id,
                    "destination_id": destination_id,
                    "name_via": "A14",
                    "distance_km": 594.41,
                    "duration_hours": 7,
                    "duration_minutes": 33,
                    "road_types": ["highway"],
                    "includes_toll": False,
                    "alias": [third_alias],
                },
            ]
        },
    }


def _run_base48_to_route_prompt(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    route_result: dict[str, Any] | None = None,
    replacement_user_text: str = "Okay, can you change my destination to Munich?",
) -> Any:
    first = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-poi-user",
            context_id=context_id,
            system_policy=BASE_48_POLICY,
            user_text="Can you show me some restaurants in Munich?",
            live_tools=BASE_48_TOOLS,
        )
    )
    assert first.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Munich"},
        },
    )
    search = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-location-result",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": "loc_mun_9995"}},
        )
    )
    assert search.tool_calls == (
        {
            "tool_name": "search_poi_at_location",
            "arguments": {
                "location_id": "loc_mun_9995",
                "category_poi": "restaurants",
            },
        },
    )
    pois = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-poi-result",
            tool_name="search_poi_at_location",
            content=_base48_poi_result(),
        )
    )
    assert pois.text is not None
    assert "Brauhaus Germania" in pois.text
    assert "Gasthaus Zum Adler" in pois.text

    state_read = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-replace-user",
            context_id=context_id,
            user_text=replacement_user_text,
        )
    )
    assert state_read.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state-result",
            tool_name="get_current_navigation_state",
            content=_base48_navigation_state(),
        )
    )
    assert routes.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_dor_399984",
                "destination_id": "loc_mun_9995",
            },
        },
    )
    return runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-route-result",
            tool_name="get_routes_from_start_to_destination",
            content=route_result or _base48_route_result(),
        )
    )


def _empty_base48_replacement_intent() -> dict[str, Any]:
    return {
        "language": "en",
        "intent_kind": "information",
        "call_for_action": False,
        "goals": [],
        "explicit_slots": {},
    }


def _qwen_start_navigation_intent() -> dict[str, Any]:
    return action_intent(
        [
            {
                "semantic_operation": "start_navigation",
                "desired_outcome": {"destination": "Munich"},
            }
        ],
        explicit_slots={"destination": "Munich"},
    )


@pytest.mark.parametrize(
    ("replacement_intents", "expected_intent_calls"),
    [
        (
            [
                _empty_base48_replacement_intent(),
                _empty_base48_replacement_intent(),
            ],
            1,
        ),
        ([_qwen_start_navigation_intent()], 1),
    ],
    ids=["empty-focused-and-fallback", "qwen-start-navigation-sibling"],
)
def test_base48_exact_official_destination_replacement_transcript(
    replacement_intents: list[dict[str, Any]], expected_intent_calls: int
) -> None:
    client = _base48_client(replacement_intents=replacement_intents)
    runtime = runtime_for(client)
    context_id = f"base48-exact-{expected_intent_calls}"
    prompt = _run_base48_to_route_prompt(
        runtime,
        context_id=context_id,
        replacement_user_text=(
            "Actually, I want to change my destination to Munich. "
            "Can you navigate me there?"
        ),
    )

    assert prompt.text is not None
    assert "fastest and shortest" in prompt.text
    assert "2 further route alternatives" in prompt.text
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert [goal.semantic_operation for goal in session.intent.goals] == [
        "navigation_replace_final_destination"
    ]
    assert session.intent.goals[0].desired_outcome["new_destination_name"] == ("Munich")

    detail = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-detail-user",
            context_id=context_id,
            user_text="Can you tell me more about the alternative routes?",
        )
    )
    assert detail.tool_calls == ()
    assert detail.text is not None
    assert "K816, A46" in detail.text and "A14" in detail.text

    selected = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-selection-user",
            context_id=context_id,
            user_text=("I would like to take the second route, via K816 and A46."),
        )
    )
    if selected.tool_calls and selected.tool_calls[0]["tool_name"] == (
        "get_current_navigation_state"
    ):
        selected = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"{context_id}-policy-state-result",
                tool_name="get_current_navigation_state",
                content=_base48_navigation_state(),
            )
        )
    assert selected.tool_calls == (
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": "loc_mun_9995",
                "route_id_leading_to_new_destination": "rll_dor_mun_475855",
            },
        },
    )
    assert client.intent_calls == expected_intent_calls
    assert client.action_calls == 0

    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-replacement-result",
            tool_name="navigation_replace_final_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "destination_replaced": True,
                    "new_waypoints": ["loc_dor_399984", "loc_mun_9995"],
                    "new_routes": ["rll_dor_mun_475855"],
                },
            },
        )
    )
    assert completed.tool_calls == ()
    assert completed.text is not None and "replaced the destination" in completed.text


def test_bare_named_navigation_does_not_become_destination_replacement() -> None:
    client = SequentialIntentClient([_qwen_start_navigation_intent()])
    runtime = runtime_for(client)
    context_id = "base48-bare-start-navigation"

    response = runtime.handle_event(
        InboundEvent(
            message_id="base48-bare-start-navigation-user",
            context_id=context_id,
            system_policy=BASE_48_POLICY,
            user_text="Can you navigate me to Munich?",
            live_tools=BASE_48_TOOLS,
        )
    )

    assert all(
        call["tool_name"] != "navigation_replace_final_destination"
        for call in response.tool_calls
    )
    assert response.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert [goal.semantic_operation for goal in session.intent.goals] == [
        "create_navigation"
    ]
    assert client.intent_calls == 0
    assert client.action_calls == 0


def test_base48_named_poi_then_user_selected_destination_route() -> None:
    client = _base48_client()
    runtime = runtime_for(client)
    context_id = "base48-full"
    prompt = _run_base48_to_route_prompt(runtime, context_id=context_id)

    assert prompt.text is not None
    assert "fastest and shortest" in prompt.text
    assert "2 further route alternatives" in prompt.text

    detail = runtime.handle_event(
        InboundEvent(
            message_id="base48-detail-user",
            context_id=context_id,
            user_text="Can you tell me more about the alternative routes?",
        )
    )
    assert detail.tool_calls == ()
    assert detail.text is not None
    assert "K816, A46" in detail.text and "A14" in detail.text

    conflict = runtime.handle_event(
        InboundEvent(
            message_id="base48-conflict-user",
            context_id=context_id,
            user_text="Can we take the second route via A14?",
        )
    )
    assert conflict.tool_calls == ()
    assert conflict.text is not None and "Which option" in conflict.text

    conflicting_ordinal = runtime.handle_event(
        InboundEvent(
            message_id="base48-conflicting-ordinal-user",
            context_id=context_id,
            user_text="Can we take the first route via K816, A46?",
        )
    )
    assert conflicting_ordinal.tool_calls == ()
    assert conflicting_ordinal.text is not None and "Which option" in (
        conflicting_ordinal.text
    )

    selected = runtime.handle_event(
        InboundEvent(
            message_id="base48-selection-user",
            context_id=context_id,
            user_text="Can we take the second route via K816, A46?",
        )
    )
    if selected.tool_calls and selected.tool_calls[0]["tool_name"] == (
        "get_current_navigation_state"
    ):
        selected = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id="base48-policy-state-result",
                tool_name="get_current_navigation_state",
                content=_base48_navigation_state(),
            )
        )
    assert selected.tool_calls == (
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": "loc_mun_9995",
                "route_id_leading_to_new_destination": "rll_dor_mun_475855",
            },
        },
    )
    session = runtime.sessions.get(context_id)
    assert session is not None
    binding = session.resolved_location_binding(
        "Munich", state_version=session.evidence.current_state_version
    )
    assert (
        binding is not None and binding.source_evidence_id in session.evidence.evidence
    )
    goal = next(goal for goal in session.intent.goals if goal.status.value == "pending")
    route_evidence_id = session.derived_value_evidence_by_goal[goal.goal_id]["route_id"]
    route_evidence = session.evidence.evidence[route_evidence_id]
    assert route_evidence.source_kind is EvidenceSourceKind.DERIVED
    assert route_evidence.derivation == "user_selected_route_alias_v1"
    assert route_evidence.derived_from
    assert client.intent_calls == 1
    assert client.action_calls == 0
    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base48-replacement-result",
            tool_name="navigation_replace_final_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "destination_replaced": True,
                    "new_waypoints": ["loc_dor_399984", "loc_mun_9995"],
                    "new_routes": ["rll_dor_mun_475855"],
                },
            },
        )
    )
    assert completed.text is not None and "replaced the destination" in completed.text
    assert completed.tool_calls == ()


def test_base48_rejects_poi_result_bound_to_another_location() -> None:
    client = _base48_client()
    runtime = runtime_for(client)
    context_id = "base48-wrong-poi-location"
    first = runtime.handle_event(
        InboundEvent(
            message_id="base48-wrong-poi-user",
            context_id=context_id,
            system_policy=BASE_48_POLICY,
            user_text="Can you show me some restaurants in Munich?",
            live_tools=BASE_48_TOOLS,
        )
    )
    assert first.tool_calls
    search = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base48-wrong-poi-location-result",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": "loc_mun_9995"}},
        )
    )
    assert search.tool_calls
    rejected = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base48-wrong-poi-result",
            tool_name="search_poi_at_location",
            content=_base48_poi_result(corresponding_location_id="loc_ber_217736"),
        )
    )
    assert rejected.tool_calls == ()
    assert rejected.text is not None and "couldn't verify" in rejected.text


def test_hallucination46_missing_poi_category_stops_after_location_read() -> None:
    client = _base48_client()
    runtime = runtime_for(client)
    context_id = "hallucination46-missing-category"
    tools_without_category = tuple(
        spec
        for spec in BASE_48_TOOLS
        if spec["function"]["name"] != "search_poi_at_location"
    ) + (
        tool(
            "search_poi_at_location",
            {"location_id": {"type": "string"}},
            ["location_id"],
        ),
    )
    first = runtime.handle_event(
        InboundEvent(
            message_id="hallucination46-user",
            context_id=context_id,
            system_policy=BASE_48_POLICY,
            user_text="Can you show me some restaurants in Munich?",
            live_tools=tools_without_category,
        )
    )
    assert first.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Munich"},
        },
    )
    stopped = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="hallucination46-location-result",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": "loc_mun_9995"}},
        )
    )

    assert stopped.tool_calls == ()
    assert stopped.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert [result.tool_name for result in session.successful_read_results] == [
        "get_location_id_by_location_name"
    ]
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "route_result",
    [
        _base48_route_result(destination_id="loc_ber_217736"),
        _base48_route_result(start_id="loc_ber_217736"),
        _base48_route_result(third_alias="second"),
        _base48_route_result(first_distance="NaN"),
        _base48_route_result(second_via="K816\nA46"),
    ],
)
def test_base48_rejects_mismatched_or_ambiguous_route_results(
    route_result: dict[str, Any],
) -> None:
    runtime = runtime_for(_base48_client())
    rejected = _run_base48_to_route_prompt(
        runtime,
        context_id=f"base48-invalid-route-{hash(json.dumps(route_result))}",
        route_result=route_result,
    )

    assert rejected.tool_calls == ()
    assert rejected.text is not None and "couldn't verify" in rejected.text


def test_base48_rejects_conflicting_cross_goal_location_resolution() -> None:
    runtime = runtime_for(_base48_client())
    context_id = "base48-cross-goal-location-conflict"
    first = runtime.handle_event(
        InboundEvent(
            message_id="base48-cache-poi-user",
            context_id=context_id,
            system_policy=BASE_48_POLICY,
            user_text="Can you show me some restaurants in Munich?",
            live_tools=BASE_48_TOOLS,
        )
    )
    assert first.tool_calls
    search = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base48-cache-location-result",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": "loc_mun_9995"}},
        )
    )
    assert search.tool_calls
    pois = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base48-cache-poi-result",
            tool_name="search_poi_at_location",
            content=_base48_poi_result(),
        )
    )
    assert pois.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    session.successful_read_results.append(
        SuccessfulReadResult(
            goal_ids=("other-goal",),
            tool_name="get_location_id_by_location_name",
            arguments={"location": "Munich"},
            semantic_operation="resolve_trip_destination",
            call_id="other-location-call",
            value={"id": "loc_ber_217736"},
            source_turn_id="other-location-result",
            state_version=session.evidence.current_state_version,
        )
    )

    state = runtime.handle_event(
        InboundEvent(
            message_id="base48-cache-replace-user",
            context_id=context_id,
            user_text="Can you change my destination to Munich?",
        )
    )
    assert state.tool_calls and state.tool_calls[0]["tool_name"] == (
        "get_current_navigation_state"
    )
    rejected = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base48-cache-state-result",
            tool_name="get_current_navigation_state",
            content=_base48_navigation_state(),
        )
    )
    assert rejected.tool_calls == ()
    assert rejected.text is not None and "couldn't verify" in rejected.text


def test_base48_presents_two_distinct_fastest_and_shortest_routes() -> None:
    result = _base48_route_result()
    routes = result["result"]["routes"][:2]
    routes[0]["alias"] = ["fastest", "first"]
    routes[1]["alias"] = ["second", "shortest"]
    routes[1]["distance_km"] = 552.11
    runtime = runtime_for(_base48_client())

    prompt = _run_base48_to_route_prompt(
        runtime,
        context_id="base48-two-routes",
        route_result={"status": "SUCCESS", "result": {"routes": routes}},
    )

    assert prompt.text is not None
    assert "fastest" in prompt.text and "shortest" in prompt.text
    assert "Which of the routes described above" in prompt.text


@pytest.mark.parametrize(
    "question",
    [
        "Would the second route via K816 and A46 take longer?",
        "How long would the second route via K816 and A46 take?",
        "Do you recommend I take the second route via K816 and A46?",
        "Should I take the second route via K816 and A46?",
        "Should I take the second route via K816 and A46",
        "Why would you take the second route via K816 and A46",
        "Okay should I take the second route via K816 and A46",
        "I wonder should I take the second route via K816 and A46",
        "Which route would you take, the second route via K816 and A46?",
        "Why would you take the second route via K816 and A46?",
        "When would we take the second route via K816 and A46?",
        "How would you take the second route via K816 and A46?",
        "I want information about the second route via K816 and A46",
        "I need details about the second route via K816 and A46",
        "Okay can you tell me how long the second route would take",
        "Can we take a closer look at the second route via K816 and A46?",
        "Would you take a minute to explain the second route via K816 and A46?",
        "Second route via K816 and A46 details?",
        "Second route via K816 and A46 duration",
        "Second route via K816 and A46 cost",
        "Second route via K816 and A46 traffic",
    ],
)
def test_base48_route_information_questions_do_not_authorize_selection(
    question: str,
) -> None:
    client = _base48_client()
    runtime = runtime_for(client)
    context_id = f"base48-route-information-{abs(hash(question))}"
    prompt = _run_base48_to_route_prompt(runtime, context_id=context_id)
    assert prompt.text is not None

    response = runtime.handle_event(
        InboundEvent(
            message_id="base48-route-information-question",
            context_id=context_id,
            user_text=question,
        )
    )
    assert response.tool_calls == ()
    assert response.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert len(session.pending_clarifications) == 1
    assert session.completed_action_calls_by_goal == {}
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "selection",
    [
        "I would like to take the second route via K816 and A46",
        "Can we take the second route via K816 and A46?",
        "Please take the second route via K816 and A46",
        "the second route please",
        "second please",
        "via K816 and A46 please",
    ],
)
def test_base48_explicit_route_choice_is_authorized(selection: str) -> None:
    client = _base48_client()
    runtime = runtime_for(client)
    context_id = f"base48-explicit-route-choice-{abs(hash(selection))}"
    prompt = _run_base48_to_route_prompt(runtime, context_id=context_id)
    assert prompt.text is not None
    selected = runtime.handle_event(
        InboundEvent(
            message_id="base48-route-explicit-selection",
            context_id=context_id,
            user_text=selection,
        )
    )
    if selected.tool_calls and selected.tool_calls[0]["tool_name"] == (
        "get_current_navigation_state"
    ):
        selected = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id="base48-route-explicit-selection-state",
                tool_name="get_current_navigation_state",
                content=_base48_navigation_state(),
            )
        )
    assert selected.tool_calls == (
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": "loc_mun_9995",
                "route_id_leading_to_new_destination": "rll_dor_mun_475855",
            },
        },
    )
    assert client.action_calls == 0


def test_base48_rejects_changed_navigation_state_after_route_selection() -> None:
    client = _base48_client()
    runtime = runtime_for(client)
    context_id = "base48-navigation-state-toctou"
    prompt = _run_base48_to_route_prompt(runtime, context_id=context_id)
    assert prompt.text is not None
    policy_read = runtime.handle_event(
        InboundEvent(
            message_id="base48-toctou-selection",
            context_id=context_id,
            user_text="Can we take the second route via K816 and A46?",
        )
    )
    assert policy_read.tool_calls == (
        {"tool_name": "get_current_navigation_state", "arguments": {}},
    )

    rejected = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id="base48-toctou-changed-state",
            tool_name="get_current_navigation_state",
            content={
                "status": "SUCCESS",
                "result": {
                    "navigation_active": True,
                    "waypoints_id": ["loc_ber_217736", "loc_col_464166"],
                    "routes_to_final_destination_id": ["rll_ber_col_123456"],
                },
            },
        )
    )

    assert rejected.tool_calls == ()
    assert rejected.text is not None and "couldn't verify" in rejected.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.completed_action_calls_by_goal == {}
    assert client.action_calls == 0


def test_base48_rejects_fastest_and_shortest_aliases_on_worst_route() -> None:
    client = _base48_client()
    runtime = runtime_for(client)
    result = _base48_route_result()
    routes = result["result"]["routes"]
    routes[0]["alias"] = ["first"]
    routes[2]["alias"] = ["third", "fastest", "shortest"]

    rejected = _run_base48_to_route_prompt(
        runtime,
        context_id="base48-false-fastest-shortest-aliases",
        route_result=result,
    )

    assert rejected.tool_calls == ()
    assert rejected.text is not None and "couldn't verify" in rejected.text
    session = runtime.sessions.get("base48-false-fastest-shortest-aliases")
    assert session is not None
    assert session.pending_clarifications == {}
    assert session.completed_action_calls_by_goal == {}
    assert client.action_calls == 0


BASE_50_POLICY = """
AUT-POL:017: Tools to delete, replace, or add a waypoint or a destination can
only be used when the navigation system is already active and a route is set.
AUT-POL:018: If navigation is active, edit the active route with the
corresponding tool and do not reload the navigation system.
AUT-POL:019: A route must retain at least a start and a destination. A final
destination cannot be deleted when there is no intermediate stop.
CURRENT_LOCATION = {"id":"loc_bre_831815","name":"Bremen"}
"""

BASE_50_OFFICIAL_TEXT = (
    "Hey there! I've changed my mind about going to Essen after Dortmund. "
    "Can you remove Essen from my route so Dortmund is the final stop?"
)


def _base50_navigation_state() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [
                "loc_bre_831815",
                "loc_dor_399984",
                "loc_ess_699309",
            ],
            "routes_to_final_destination_id": [
                "rll_bre_dor_359171",
                "rll_dor_ess_476903",
            ],
            "details": {
                "waypoints": [
                    {"id": "loc_bre_831815", "name": "Bremen"},
                    {"id": "loc_dor_399984", "name": "Dortmund"},
                    {"id": "loc_ess_699309", "name": "Essen"},
                ]
            },
        },
    }


def _empty_base50_intent() -> dict[str, Any]:
    return {
        "language": "en",
        "intent_kind": "information",
        "call_for_action": False,
        "goals": [],
        "explicit_slots": {},
    }


def _misclassified_base50_replacement_intent() -> dict[str, Any]:
    return action_intent(
        [
            {
                "semantic_operation": "navigation_replace_final_destination",
                "desired_outcome": {"new_destination_id": "Dortmund"},
            }
        ],
        explicit_slots={"new_destination_id": "Dortmund"},
    )


def _misclassified_base50_waypoint_delete_intent() -> dict[str, Any]:
    return action_intent(
        [
            {
                "semantic_operation": "navigation_delete_one_waypoint",
                "desired_outcome": {"waypoint_id": "Essen"},
            }
        ],
        explicit_slots={"waypoint_id": "Essen"},
    )


@pytest.mark.parametrize(
    ("intents", "expected_intent_calls"),
    [
        ([_empty_base50_intent(), _empty_base50_intent()], 2),
        ([_misclassified_base50_replacement_intent()], 1),
        ([_misclassified_base50_waypoint_delete_intent()], 1),
    ],
    ids=[
        "empty-focused-and-fallback",
        "misclassified-final-replacement",
        "misclassified-waypoint-delete",
    ],
)
def test_base50_exact_official_final_destination_delete_transcript(
    intents: list[dict[str, Any]], expected_intent_calls: int
) -> None:
    client = SequentialIntentClient(intents)
    runtime = runtime_for(client)
    context_id = f"base50-exact-{expected_intent_calls}"

    detailed = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=BASE_50_POLICY,
            user_text=BASE_50_OFFICIAL_TEXT,
            live_tools=NAVIGATION_FAMILY_TOOLS,
        )
    )
    assert detailed.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )

    fresh = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-detailed-state",
            tool_name="get_current_navigation_state",
            content=_base50_navigation_state(),
        )
    )
    assert fresh.tool_calls == (
        {"tool_name": "get_current_navigation_state", "arguments": {}},
    )

    delete = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-fresh-state",
            tool_name="get_current_navigation_state",
            content=_base50_navigation_state(),
        )
    )
    assert delete.tool_calls == (
        {
            "tool_name": "navigation_delete_destination",
            "arguments": {"destination_id_to_delete": "loc_ess_699309"},
        },
    )

    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert [goal.semantic_operation for goal in session.intent.goals] == [
        "navigation_delete_final_destination"
    ]
    goal = session.intent.goals[0]
    assert goal.desired_outcome["destination_name_to_delete"] == "Essen"
    assert goal.desired_outcome["remaining_destination_name"] == "Dortmund"
    assert goal.desired_outcome["destination_id"] == "loc_ess_699309"
    assert client.intent_calls == expected_intent_calls
    assert client.action_calls == 0

    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-delete-result",
            tool_name="navigation_delete_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "destination_deleted": True,
                    "new_waypoints": [
                        "loc_bre_831815",
                        "loc_dor_399984",
                    ],
                    "new_routes": ["rll_bre_dor_359171"],
                },
            },
        )
    )
    assert completed.tool_calls == ()
    assert completed.text is not None and "removed the destination" in completed.text


def test_base50_rejects_changed_navigation_state_before_delete() -> None:
    client = SequentialIntentClient([_misclassified_base50_replacement_intent()])
    runtime = runtime_for(client)
    context_id = "base50-navigation-state-toctou"
    detailed = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=BASE_50_POLICY,
            user_text=BASE_50_OFFICIAL_TEXT,
            live_tools=NAVIGATION_FAMILY_TOOLS,
        )
    )
    assert detailed.tool_calls
    fresh = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-detailed-state",
            tool_name="get_current_navigation_state",
            content=_base50_navigation_state(),
        )
    )
    assert fresh.tool_calls == (
        {"tool_name": "get_current_navigation_state", "arguments": {}},
    )

    changed = _base50_navigation_state()
    changed["result"]["waypoints_id"][-1] = "loc_col_464166"
    changed["result"]["routes_to_final_destination_id"][-1] = "rll_dor_col_333333"
    changed["result"]["details"]["waypoints"][-1] = {
        "id": "loc_col_464166",
        "name": "Cologne",
    }
    rejected = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-changed-state",
            tool_name="get_current_navigation_state",
            content=changed,
        )
    )

    assert rejected.tool_calls == ()
    assert rejected.text is not None and "couldn't verify" in rejected.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert session.completed_action_calls_by_goal == {}
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state["result"]["details"]["waypoints"].__setitem__(
            1, {"id": "loc_dor_399984", "name": "Cologne"}
        ),
        lambda state: (
            state["result"]["details"]["waypoints"].__setitem__(
                0, {"id": "loc_bre_831815", "name": "Essen"}
            ),
            state["result"]["details"]["waypoints"].__setitem__(
                2, {"id": "loc_ess_699309", "name": "Cologne"}
            ),
        ),
        lambda state: state["result"]["details"]["waypoints"].__setitem__(
            0, {"id": "loc_bre_831815", "name": "Essen"}
        ),
        lambda state: state["result"].pop("details"),
        lambda state: state["result"]["routes_to_final_destination_id"].__setitem__(
            1, "rll_bre_dor_359171"
        ),
    ],
    ids=[
        "remaining-not-penultimate",
        "target-not-final",
        "ambiguous-target-name",
        "missing-details",
        "duplicate-route",
    ],
)
def test_base50_named_delete_rejects_unverified_detailed_state(
    mutate: Callable[[dict[str, Any]], Any],
) -> None:
    intent = action_intent(
        [
            {
                "semantic_operation": "navigation_delete_final_destination",
                "desired_outcome": {
                    "destination_name_to_delete": "Essen",
                    "remaining_destination_name": "Dortmund",
                },
            }
        ],
        explicit_slots={
            "destination_name_to_delete": "Essen",
            "remaining_destination_name": "Dortmund",
        },
    )
    client = FailingActionClient(
        intent, lambda payload: pytest.fail(f"planner must not run: {payload}")
    )
    runtime = runtime_for(client)
    context_id = f"base50-invalid-state-{abs(hash(str(mutate)))}"
    current = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=BASE_50_POLICY,
            user_text=BASE_50_OFFICIAL_TEXT,
            live_tools=NAVIGATION_FAMILY_TOOLS,
        )
    )
    assert current.tool_calls
    state = _base50_navigation_state()
    mutate(state)
    rejected = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=state,
        )
    )

    assert rejected.tool_calls == ()
    assert rejected.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_hallucination48_missing_delete_tool_stops_before_navigation_read() -> None:
    client = SequentialIntentClient([_misclassified_base50_replacement_intent()])
    runtime = runtime_for(client)
    tools_without_delete = tuple(
        spec
        for spec in NAVIGATION_FAMILY_TOOLS
        if spec["function"]["name"] != "navigation_delete_destination"
    )

    blocked = runtime.handle_event(
        InboundEvent(
            message_id="hallucination48-user",
            context_id="hallucination48-missing-delete-tool",
            system_policy=BASE_50_POLICY,
            user_text=BASE_50_OFFICIAL_TEXT,
            live_tools=tools_without_delete,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None and "required control" in blocked.text
    session = runtime.sessions.get("hallucination48-missing-delete-tool")
    assert session is not None
    assert session.successful_read_results == []
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


BASE_80_POLICY = """
AUT-POL:017: Tools to delete, replace, or add a waypoint or a destination can
only be used when the navigation system is already active and a route is set.
AUT-POL:018: If navigation is active, edit the active route with the
corresponding tool and do not reload or delete the whole navigation system.
AUT-POL:019: A route must retain at least a start and a destination.
CURRENT_LOCATION = {"id":"loc_min_459749","name":"Minsk"}
"""


def _base80_navigation_state(*, include_rome: bool) -> dict[str, Any]:
    waypoint_ids = ["loc_min_459749", "loc_bel_437063", "loc_mun_9995"]
    route_ids = ["rll_min_bel_141719", "rll_bel_mun_791850"]
    waypoint_names = ["Minsk", "Belgrade", "Munich"]
    if include_rome:
        waypoint_ids.append("loc_rom_294918")
        route_ids.append("rll_mun_rom_149655")
        waypoint_names.append("Rome")
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": waypoint_ids,
            "routes_to_final_destination_id": route_ids,
            "details": {
                "waypoints": [
                    {"id": waypoint_id, "name": name}
                    for waypoint_id, name in zip(
                        waypoint_ids, waypoint_names, strict=True
                    )
                ]
            },
        },
    }


def _base80_direct_routes() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": "rll_min_mun_745409",
                    "start_id": "loc_min_459749",
                    "destination_id": "loc_mun_9995",
                    "name_via": "A46, A16, A12",
                    "distance_km": 1498.03,
                    "duration_hours": 18,
                    "duration_minutes": 32,
                    "road_types": ["highway"],
                    "includes_toll": False,
                    "alias": ["fastest", "first"],
                },
                {
                    "route_id": "rll_min_mun_803896",
                    "start_id": "loc_min_459749",
                    "destination_id": "loc_mun_9995",
                    "name_via": "A97, A58, A11",
                    "distance_km": 1497.36,
                    "duration_hours": 19,
                    "duration_minutes": 4,
                    "road_types": ["highway"],
                    "includes_toll": False,
                    "alias": ["second", "shortest"],
                },
                {
                    "route_id": "rll_min_mun_458573",
                    "start_id": "loc_min_459749",
                    "destination_id": "loc_mun_9995",
                    "name_via": "A88, A84",
                    "distance_km": 1547.45,
                    "duration_hours": 19,
                    "duration_minutes": 40,
                    "road_types": ["country road", "highway", "urban"],
                    "includes_toll": False,
                    "alias": ["third"],
                },
            ]
        },
    }


def test_base80_generic_named_deletes_never_delete_whole_navigation() -> None:
    client = SequentialIntentClient([])
    runtime = runtime_for(client)
    context_id = "base80-generic-named-deletes"

    detailed = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-rome-user",
            context_id=context_id,
            system_policy=BASE_80_POLICY,
            user_text="Remove Rome from my trip.",
            live_tools=NAVIGATION_REPLACEMENT_TOOLS,
        )
    )
    assert detailed.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    fresh = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-rome-detailed",
            tool_name="get_current_navigation_state",
            content=_base80_navigation_state(include_rome=True),
        )
    )
    assert fresh.tool_calls == (
        {"tool_name": "get_current_navigation_state", "arguments": {}},
    )
    delete_rome = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-rome-fresh",
            tool_name="get_current_navigation_state",
            content=_base80_navigation_state(include_rome=True),
        )
    )
    assert delete_rome.tool_calls == (
        {
            "tool_name": "navigation_delete_destination",
            "arguments": {"destination_id_to_delete": "loc_rom_294918"},
        },
    )
    removed_rome = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-rome-delete",
            tool_name="navigation_delete_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "destination_deleted": True,
                    "new_waypoints": [
                        "loc_min_459749",
                        "loc_bel_437063",
                        "loc_mun_9995",
                    ],
                    "new_routes": [
                        "rll_min_bel_141719",
                        "rll_bel_mun_791850",
                    ],
                },
            },
        )
    )
    assert removed_rome.tool_calls == ()
    assert removed_rome.text is not None

    detailed = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-belgrade-user",
            context_id=context_id,
            user_text="Remove Belgrade from my trip.",
            live_tools=NAVIGATION_REPLACEMENT_TOOLS,
        )
    )
    assert detailed.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-belgrade-state",
            tool_name="get_current_navigation_state",
            content=_base80_navigation_state(include_rome=False),
        )
    )
    assert routes.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_min_459749",
                "destination_id": "loc_mun_9995",
            },
        },
    )
    clarification = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=_base80_direct_routes(),
        )
    )
    assert clarification.tool_calls == ()
    assert clarification.text is not None and "fastest" in clarification.text
    for index, unsafe_choice in enumerate(
        (
            "Do not choose the fastest route.",
            'She said, "Choose the fastest route."',
            "Choose the fastest route. Delete the current navigation.",
        )
    ):
        rejected = runtime.handle_event(
            InboundEvent(
                context_id=context_id,
                message_id=f"{context_id}-unsafe-choice-{index}",
                user_text=unsafe_choice,
            )
        )
        assert rejected.tool_calls == ()
        assert rejected.text is not None and "Which option" in rejected.text
    delete_belgrade = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-choice",
            user_text="Choose the fastest route.",
        )
    )
    assert delete_belgrade.tool_calls == (
        {
            "tool_name": "navigation_delete_waypoint",
            "arguments": {
                "waypoint_id_to_delete": "loc_bel_437063",
                "route_id_without_waypoint": "rll_min_mun_745409",
            },
        },
    ), delete_belgrade.text
    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-belgrade-delete",
            tool_name="navigation_delete_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_deleted": True,
                    "new_waypoints": ["loc_min_459749", "loc_mun_9995"],
                    "new_routes": ["rll_min_mun_745409"],
                },
            },
        )
    )
    assert completed.tool_calls == ()
    assert completed.text is not None and "removed Belgrade" in completed.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    completed_tool_names = {
        call.tool_name
        for calls in session.completed_action_calls_by_goal.values()
        for call in calls
    }
    assert "delete_current_navigation" not in completed_tool_names
    assert {
        "navigation_delete_destination",
        "navigation_delete_waypoint",
    }.issubset(completed_tool_names)
    assert client.intent_calls == 0
    assert client.action_calls == 0


def test_base80_fastest_route_selection_uses_verified_alias_not_first_position() -> (
    None
):
    client = SequentialIntentClient([])
    runtime = runtime_for(client)
    context_id = "base80-fastest-is-second"

    detailed = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=BASE_80_POLICY,
            user_text="Remove Belgrade from my navigation.",
            live_tools=NAVIGATION_REPLACEMENT_TOOLS,
        )
    )
    assert detailed.tool_calls
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=_base80_navigation_state(include_rome=False),
        )
    )
    assert routes.tool_calls
    route_result = _base80_direct_routes()
    first, second = route_result["result"]["routes"][:2]
    first.update(
        {
            "alias": ["first", "shortest"],
            "distance_km": 1497.36,
            "duration_hours": 19,
            "duration_minutes": 4,
        }
    )
    second.update(
        {
            "alias": ["fastest", "second"],
            "distance_km": 1498.03,
            "duration_hours": 18,
            "duration_minutes": 32,
        }
    )
    clarification = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=route_result,
        )
    )
    assert clarification.tool_calls == ()

    selected = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-choice",
            user_text="Choose the fastest route.",
        )
    )
    assert selected.tool_calls == (
        {
            "tool_name": "navigation_delete_waypoint",
            "arguments": {
                "waypoint_id_to_delete": "loc_bel_437063",
                "route_id_without_waypoint": "rll_min_mun_803896",
            },
        },
    )
    assert client.intent_calls == 0
    assert client.action_calls == 0


@pytest.mark.parametrize("corruption", ["duplicate-fastest", "wrong-endpoint"])
def test_base80_invalid_route_evidence_blocks_waypoint_delete(corruption: str) -> None:
    client = SequentialIntentClient([])
    runtime = runtime_for(client)
    context_id = f"base80-invalid-route-{corruption}"
    detailed = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=BASE_80_POLICY,
            user_text="Remove Belgrade from my trip.",
            live_tools=NAVIGATION_REPLACEMENT_TOOLS,
        )
    )
    assert detailed.tool_calls
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=_base80_navigation_state(include_rome=False),
        )
    )
    assert routes.tool_calls
    route_result = _base80_direct_routes()
    if corruption == "duplicate-fastest":
        route_result["result"]["routes"][1]["alias"].append("fastest")
    else:
        route_result["result"]["routes"][0]["destination_id"] = "loc_wrong_1"
    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=route_result,
        )
    )
    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.intent_calls == 0
    assert client.action_calls == 0


def test_base80_missing_waypoint_delete_tool_blocks_before_route_lookup() -> None:
    client = SequentialIntentClient([])
    runtime = runtime_for(client)
    context_id = "base80-missing-waypoint-delete-tool"
    tools_without_waypoint_delete = tuple(
        candidate
        for candidate in NAVIGATION_REPLACEMENT_TOOLS
        if candidate["function"]["name"] != "navigation_delete_waypoint"
    )
    detailed = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=BASE_80_POLICY,
            user_text="Remove Belgrade from my trip.",
            live_tools=tools_without_waypoint_delete,
        )
    )
    assert detailed.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=_base80_navigation_state(include_rome=False),
        )
    )
    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert not any(
        result.tool_name == "get_routes_from_start_to_destination"
        for result in session.successful_read_results
    )


BASE_88_POLICY = """
AUT-POL:017: Tools to delete a waypoint can only be used when navigation is active.
AUT-POL:018: Edit the active route with the corresponding tool.
CURRENT_LOCATION = {"id":"loc_bru_597661","name":"Brussels"}
"""
BASE_88_TOOLS = (
    *NAVIGATION_REPLACEMENT_TOOLS,
    tool("get_charging_specs_and_status"),
    tool(
        "search_poi_along_the_route",
        {
            "route_id": {"type": "string"},
            "filters": {"type": "array", "items": {"type": "string"}},
            "category_poi": {"type": "string"},
            "at_kilometer": {"type": "integer"},
        },
        ["route_id", "filters", "category_poi", "at_kilometer"],
    ),
)


def _base88_navigation_state(*, after_delete: bool) -> dict[str, Any]:
    if after_delete:
        waypoint_ids = ["loc_bru_597661", "loc_ber_217736", "loc_lei_519681"]
        route_ids = ["rll_bru_ber_407820", "rll_ber_lei_896859"]
        names = ["Brussels", "Berlin", "Leipzig"]
        routes = [
            {
                "route_id": "rll_bru_ber_407820",
                "start_id": "loc_bru_597661",
                "destination_id": "loc_ber_217736",
                "name_via": "L556, K463, K440",
                "distance_km": 750.11,
                "duration_hours": 9,
                "duration_minutes": 30,
                "road_types": ["country road", "urban"],
                "includes_toll": False,
                "alias": ["fastest", "first", "shortest"],
            },
            {
                "route_id": "rll_ber_lei_896859",
                "start_id": "loc_ber_217736",
                "destination_id": "loc_lei_519681",
                "name_via": "A59, K617, L843",
                "distance_km": 177.9,
                "duration_hours": 2,
                "duration_minutes": 16,
                "road_types": ["country road", "highway", "urban"],
                "includes_toll": False,
                "alias": ["third", "shortest"],
            },
        ]
    else:
        waypoint_ids = [
            "loc_bru_597661",
            "loc_bon_490528",
            "loc_ber_217736",
            "loc_lei_519681",
        ]
        route_ids = [
            "rll_bru_bon_361072",
            "rll_bon_ber_593219",
            "rll_ber_lei_896859",
        ]
        names = ["Brussels", "Bonn", "Berlin", "Leipzig"]
        routes = []
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": waypoint_ids,
            "routes_to_final_destination_id": route_ids,
            "details": {
                "waypoints": [
                    {"id": waypoint_id, "name": name}
                    for waypoint_id, name in zip(waypoint_ids, names, strict=True)
                ],
                "routes": routes,
            },
        },
    }


def _base88_direct_routes() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": "rll_bru_ber_407820",
                    "start_id": "loc_bru_597661",
                    "destination_id": "loc_ber_217736",
                    "name_via": "L556, K463, K440",
                    "distance_km": 750.11,
                    "duration_hours": 9,
                    "duration_minutes": 30,
                    "road_types": ["country road", "urban"],
                    "includes_toll": False,
                    "alias": ["fastest", "first", "shortest"],
                },
                {
                    "route_id": "rll_bru_ber_770681",
                    "start_id": "loc_bru_597661",
                    "destination_id": "loc_ber_217736",
                    "name_via": "L147",
                    "distance_km": 776.53,
                    "duration_hours": 9,
                    "duration_minutes": 35,
                    "road_types": ["country road", "urban"],
                    "includes_toll": False,
                    "alias": ["second"],
                },
                {
                    "route_id": "rll_bru_ber_968663",
                    "start_id": "loc_bru_597661",
                    "destination_id": "loc_ber_217736",
                    "name_via": "L811",
                    "distance_km": 790.79,
                    "duration_hours": 9,
                    "duration_minutes": 44,
                    "road_types": ["country road", "highway", "urban"],
                    "includes_toll": False,
                    "alias": ["third"],
                },
            ]
        },
    }


def _base88_charger_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "pois_found_along_route": [
                {
                    "id": "poi_cha_920428",
                    "name": "EV Power",
                    "category": "charging_stations",
                    "opening_hours": "00:00h - 24:00h",
                    "phone_number": "+49 574 6218638",
                    "corresponding_route_ids": ["rll_bru_ber_407820"],
                    "route_positions": {"rll_bru_ber": {"at_route_kilometer": 300.1}},
                    "charging_plugs": [
                        {
                            "plug_id": "plg_cha_143424",
                            "power_type": "DC",
                            "power_kw": 300,
                            "availability": "available",
                        }
                    ],
                    "detour_from_route_km": {"detour": 6.4, "unit": "km"},
                    "detour_from_route_time": {"hour": 0, "minutes": 8},
                }
            ]
        },
    }


@pytest.mark.parametrize(
    "initial_text",
    [
        (
            "Hey there! I'm on my business trip route from Brussels to Bonn to "
            "Berlin to Leipzig, but I actually don't need to stop in Bonn anymore. "
            "Can you remove Bonn from my route?"
        ),
        (
            "Hey there! I'm currently navigating to Berlin, but I've got Bonn as a "
            "stop on my route, and I actually don't need to go there anymore. Can "
            "you remove Bonn from my current route?"
        ),
        (
            "Hey there! I'm currently navigating to Berlin, but I actually don't "
            "need to stop in Bonn anymore. Can you remove Bonn from my route?"
        ),
    ],
)
def test_base88_named_delete_then_evidence_bound_charger_search(
    initial_text: str,
) -> None:
    client = SequentialIntentClient([])
    runtime = runtime_for(client)
    context_id = f"base88-{abs(hash(initial_text))}"

    outbound = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=BASE_88_POLICY,
            user_text=initial_text,
            live_tools=BASE_88_TOOLS,
        )
    )
    assert outbound.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=_base88_navigation_state(after_delete=False),
        )
    )
    assert outbound.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_bru_597661",
                "destination_id": "loc_ber_217736",
            },
        },
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=_base88_direct_routes(),
        )
    )
    assert outbound.tool_calls == (
        {
            "tool_name": "navigation_delete_waypoint",
            "arguments": {
                "waypoint_id_to_delete": "loc_bon_490528",
                "route_id_without_waypoint": "rll_bru_ber_407820",
            },
        },
    )
    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-delete",
            tool_name="navigation_delete_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_deleted": True,
                    "new_waypoints": [
                        "loc_bru_597661",
                        "loc_ber_217736",
                        "loc_lei_519681",
                    ],
                    "new_routes": [
                        "rll_bru_ber_407820",
                        "rll_ber_lei_896859",
                    ],
                },
            },
        )
    )
    assert completed.text is not None
    for expected in ("Bonn", "Berlin", "Leipzig", "L556", "alternative"):
        assert expected in completed.text
    assert "fastest route for the new direct segment" in completed.text
    assert "fastest route for each segment" not in completed.text

    outbound = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-charge-question",
            context_id=context_id,
            user_text=(
                "Okay, perfect! Thanks for removing Bonn. Also, this is a pretty "
                "long drive, do I need to charge along the way? If so, can you find "
                "some charging stations for me?"
            ),
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_charging_specs_and_status", "arguments": {}},
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-charging",
            tool_name="get_charging_specs_and_status",
            content={
                "status": "SUCCESS",
                "result": {
                    "battery_capacity_kwh": 60.0,
                    "max_charging_power_ac": 11,
                    "max_charging_power_dc": 300,
                    "state_of_charge": 85.0,
                    "remaining_range": "323.0km",
                },
            },
        )
    )
    assert outbound.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-fresh-navigation",
            tool_name="get_current_navigation_state",
            content=_base88_navigation_state(after_delete=True),
        )
    )
    assert outbound.tool_calls == (
        {
            "tool_name": "search_poi_along_the_route",
            "arguments": {
                "route_id": "rll_bru_ber_407820",
                "filters": ["charging_stations::has_available_plug"],
                "category_poi": "charging_stations",
                "at_kilometer": 300,
            },
        },
    )
    found = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-charger",
            tool_name="search_poi_along_the_route",
            content=_base88_charger_result(),
        )
    )
    assert found.tool_calls == ()
    assert found.text is not None
    for expected in ("EV Power", "300.1", "300 kilowatt DC", "did not add"):
        assert expected in found.text

    decline_texts = (
        (
            "That sounds good, I'm satisfied with that option. But no, don't add "
            "it to the route just yet. I want to check later if the plugs are still "
            "available."
        ),
        (
            "Thanks for the options! I'm good with these for now, but I don't want "
            "to add them to the route just yet."
        ),
        ("Okay, those sound good. I don't want to add any to the route just yet."),
    )
    for index, decline_text in enumerate(decline_texts):
        declined = runtime.handle_event(
            InboundEvent(
                message_id=f"{context_id}-decline-{index}",
                context_id=context_id,
                user_text=decline_text,
            )
        )
        assert declined.tool_calls == ()
        assert declined.text is not None and "did not add" in declined.text
    assert client.intent_calls == 0
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "text",
    [
        (
            "Do I need to charge along the way? If so, do not find charging "
            "stations for me."
        ),
        (
            "Will I need to charge along the way? Please don't search for charging "
            "stations."
        ),
    ],
)
def test_base88_negated_charger_search_remains_assessment_only(text: str) -> None:
    assert _active_route_charging_request(text) == (True, False)


def test_base88_separate_charge_assessment_then_search_reuses_fresh_evidence() -> None:
    client = SequentialIntentClient([])
    runtime = runtime_for(client)
    context_id = "base88-split-charge-request"
    outbound = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-delete-request",
            context_id=context_id,
            system_policy=BASE_88_POLICY,
            user_text="Remove Bonn from my trip.",
            live_tools=BASE_88_TOOLS,
        )
    )
    assert outbound.tool_calls
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=_base88_navigation_state(after_delete=False),
        )
    )
    assert outbound.tool_calls
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=_base88_direct_routes(),
        )
    )
    assert outbound.tool_calls
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-deleted",
            tool_name="navigation_delete_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_deleted": True,
                    "new_waypoints": [
                        "loc_bru_597661",
                        "loc_ber_217736",
                        "loc_lei_519681",
                    ],
                    "new_routes": [
                        "rll_bru_ber_407820",
                        "rll_ber_lei_896859",
                    ],
                },
            },
        )
    )
    outbound = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-assessment",
            context_id=context_id,
            user_text=(
                "That sounds good, thanks! With such a long drive, will I need to "
                "charge along the way?"
            ),
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_charging_specs_and_status", "arguments": {}},
    )
    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-charging",
            tool_name="get_charging_specs_and_status",
            content={
                "status": "SUCCESS",
                "result": {"state_of_charge": 85.0, "remaining_range": "323.0km"},
            },
        )
    )
    assert outbound.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    assessed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-fresh-navigation",
            tool_name="get_current_navigation_state",
            content=_base88_navigation_state(after_delete=True),
        )
    )
    assert assessed.tool_calls == ()
    assert assessed.text is not None and "will need to charge" in assessed.text

    search = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-search",
            context_id=context_id,
            user_text="Yes, please search for charging stations along the route.",
        )
    )
    assert search.tool_calls == (
        {
            "tool_name": "search_poi_along_the_route",
            "arguments": {
                "route_id": "rll_bru_ber_407820",
                "filters": ["charging_stations::has_available_plug"],
                "category_poi": "charging_stations",
                "at_kilometer": 300,
            },
        },
    )
    session = runtime.sessions.get(context_id)
    assert session is not None
    tool_names = [result.tool_name for result in session.successful_read_results]
    assert tool_names.count("get_charging_specs_and_status") == 1
    assert tool_names.count("get_current_navigation_state") == 2
    assert client.intent_calls == 0
    assert client.action_calls == 0


BASE_54_OFFICIAL_TEXT = (
    "Set the climate temperature to 22 degrees for all zones. Increase the seat "
    "heating by two levels for occupied seats. Turn on the steering wheel heating "
    "to level 2."
)
BASE_54_TOOLS = (
    SEAT_HEATING_LEVEL_TOOL,
    SEAT_OCCUPANCY_TOOL,
    CLIMATE_TEMPERATURE_TOOL,
    SEAT_HEATING_TOOL,
    STEERING_WHEEL_HEATING_TOOL,
)


def _base54_model_intent_without_relative_seat_goal() -> dict[str, Any]:
    return action_intent(
        [
            {
                "semantic_operation": "set_climate_temperature",
                "desired_outcome": {
                    "temperature": 22,
                    "seat_zone": "ALL_ZONES",
                },
            },
            {
                "semantic_operation": "set_steering_wheel_heating",
                "desired_outcome": {"level": 2},
            },
        ],
        explicit_slots={
            "temperature": 22,
            "seat_zone": "ALL_ZONES",
            "level": 2,
        },
    )


def test_base54_exact_official_relative_occupied_seat_heating_compound() -> None:
    client = FailingActionClient(
        _base54_model_intent_without_relative_seat_goal(),
        lambda payload: pytest.fail(f"planner must not run: {payload}"),
    )
    runtime = runtime_for(client)
    context_id = "base54-exact-official"

    outbound = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=BASE_54_OFFICIAL_TEXT,
            tools=BASE_54_TOOLS,
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_seat_heating_level", "arguments": {}},
    )

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-seat-levels",
            tool_name="get_seat_heating_level",
            content={
                "status": "SUCCESS",
                "result": {
                    "seat_heating_driver": 0,
                    "seat_heating_passenger": 0,
                },
            },
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_seats_occupancy", "arguments": {}},
    )

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-occupancy",
            tool_name="get_seats_occupancy",
            content={
                "status": "SUCCESS",
                "result": {
                    "seats_occupied": {
                        "driver": True,
                        "passenger": True,
                        "driver_rear": False,
                        "passenger_rear": False,
                    }
                },
            },
        )
    )
    expected_sets = (
        (
            "set_climate_temperature",
            {"temperature": 22, "seat_zone": "ALL_ZONES"},
        ),
        ("set_seat_heating", {"level": 2, "seat_zone": "ALL_ZONES"}),
        ("set_steering_wheel_heating", {"level": 2}),
    )
    for index, (tool_name, arguments) in enumerate(expected_sets):
        assert outbound.tool_calls == (
            {"tool_name": tool_name, "arguments": arguments},
        )
        outbound = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"{context_id}-set-{index}",
                tool_name=tool_name,
                content={"status": "SUCCESS", "result": arguments},
            )
        )

    assert outbound.tool_calls == ()
    assert outbound.text is not None and outbound.text.startswith("Done")
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert [goal.semantic_operation for goal in session.intent.goals] == [
        "set_climate_temperature",
        "set_seat_heating",
        "set_steering_wheel_heating",
    ]
    seat_goal = session.intent.goals[1]
    assert seat_goal.desired_outcome == {"level": 2, "seat_zone": "ALL_ZONES"}
    derived_ids = session.derived_value_evidence_by_goal[seat_goal.goal_id]
    assert set(derived_ids) == {"level", "seat_zone"}
    for evidence_id in derived_ids.values():
        evidence = session.evidence.evidence[evidence_id]
        assert evidence.derivation == "relative_occupied_seat_heating_v1"
        assert len(evidence.derived_from) == 2
    assert client.action_calls == 0


@pytest.mark.parametrize(
    ("first_text", "second_text"),
    [
        (
            (
                "Set the climate temperature to 22 degrees. Also, increase the "
                "seat heating and turn on the steering wheel heating."
            ),
            (
                "For the climate temperature, set it for all zones. For seat "
                "heating, increase it by two levels for occupied seats. Turn on "
                "the steering wheel heating to level 2."
            ),
        ),
        (
            (
                "Set the climate temperature. Increase the seat heating. Turn on "
                "the steering wheel heating."
            ),
            BASE_54_OFFICIAL_TEXT,
        ),
    ],
)
def test_base54_complete_second_turn_supersedes_stale_seat_clarification(
    first_text: str, second_text: str
) -> None:
    vague = {
        **action_intent(
            [
                {
                    "semantic_operation": "set_climate_temperature",
                    "desired_outcome": {"temperature": 22},
                },
                {
                    "semantic_operation": "set_seat_heating",
                    "desired_outcome": {},
                },
                {
                    "semantic_operation": "set_steering_wheel_heating",
                    "desired_outcome": {},
                },
            ],
            explicit_slots={"temperature": 22},
        ),
        "ambiguities": [
            {
                "name": "seat_zone",
                "candidate_values": ["ALL_ZONES", "DRIVER", "PASSENGER"],
                "goal_index": 0,
            }
        ],
    }
    client = SequentialIntentClient([vague])
    runtime = runtime_for(client)
    context_id = "base54-second-turn-clarification"

    first = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-first",
            text=first_text,
            tools=BASE_54_TOOLS,
        )
    )
    assert first.tool_calls == ()
    assert first.text is not None and "Which option" in first.text

    second = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-second",
            user_text=second_text,
        )
    )
    assert second.tool_calls == (
        {"tool_name": "get_seat_heating_level", "arguments": {}},
    )

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-seat-levels",
            tool_name="get_seat_heating_level",
            content={
                "status": "SUCCESS",
                "result": {
                    "seat_heating_driver": 0,
                    "seat_heating_passenger": 0,
                },
            },
        )
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_seats_occupancy", "arguments": {}},
    )

    outbound = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-occupancy",
            tool_name="get_seats_occupancy",
            content={
                "status": "SUCCESS",
                "result": {
                    "seats_occupied": {
                        "driver": True,
                        "passenger": True,
                        "driver_rear": False,
                        "passenger_rear": False,
                    }
                },
            },
        )
    )
    expected_sets = (
        (
            "set_climate_temperature",
            {"temperature": 22, "seat_zone": "ALL_ZONES"},
        ),
        ("set_seat_heating", {"level": 2, "seat_zone": "ALL_ZONES"}),
        ("set_steering_wheel_heating", {"level": 2}),
    )
    for index, (tool_name, arguments) in enumerate(expected_sets):
        assert outbound.tool_calls == (
            {"tool_name": tool_name, "arguments": arguments},
        )
        outbound = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"{context_id}-set-{index}",
                tool_name=tool_name,
                content={"status": "SUCCESS", "result": arguments},
            )
        )

    assert outbound.tool_calls == ()
    assert outbound.text is not None and outbound.text.startswith("Done")
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.pending_clarifications == {}
    assert session.relative_seat_heating_deltas_by_goal
    assert session.occupied_seat_heating_goal_ids == set(
        session.relative_seat_heating_deltas_by_goal
    )
    assert [goal.semantic_operation for goal in session.intent.goals] == [
        "set_climate_temperature",
        "set_seat_heating",
        "set_steering_wheel_heating",
    ]
    assert client.intent_calls == 1
    assert client.action_calls == 0


BASE_56_POLICY = """
AUT-POL:017: Tools to delete, replace, or add a waypoint or a destination can
only be used when the navigation system is already active and a route is set.
AUT-POL:018: If navigation is active, edit the active route with the
corresponding tool and do not reload the navigation system.
CURRENT_LOCATION = {"id":"loc_wie_683071","name":"Wiesbaden"}
"""
BASE_56_OFFICIAL_TEXT = "Remove Nuremberg from my route. Go straight to Paris."


def _base56_model_intent() -> dict[str, Any]:
    return action_intent(
        [
            {
                "semantic_operation": "navigation_delete_one_waypoint",
                "desired_outcome": {
                    "waypoint_id": "loc_model_guess",
                    "replacement_route_id": "rll_model_guess",
                },
            }
        ],
        explicit_slots={
            "waypoint_id": "loc_model_guess",
            "replacement_route_id": "rll_model_guess",
        },
    )


def _base56_navigation_state() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [
                "loc_wie_683071",
                "loc_nur_485085",
                "loc_par_405686",
            ],
            "routes_to_final_destination_id": [
                "rll_wie_nur_519252",
                "rll_nur_par_739533",
            ],
            "details": {
                "waypoints": [
                    {"id": "loc_wie_683071", "name": "Wiesbaden"},
                    {"id": "loc_nur_485085", "name": "Nuremberg"},
                    {"id": "loc_par_405686", "name": "Paris"},
                ],
                "routes": [
                    {"id": "rll_wie_nur_519252", "name_via": "A3"},
                    {"id": "rll_nur_par_739533", "name_via": "A6"},
                ],
            },
        },
    }


def _base56_route_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": "rll_wie_par_899896",
                    "start_id": "loc_wie_683071",
                    "destination_id": "loc_par_405686",
                    "name_via": "A11, A51",
                    "distance_km": 523.12,
                    "duration_hours": 6,
                    "duration_minutes": 32,
                    "road_types": ["highway"],
                    "includes_toll": False,
                    "alias": ["fastest", "first", "shortest"],
                },
                {
                    "route_id": "rll_wie_par_204756",
                    "start_id": "loc_wie_683071",
                    "destination_id": "loc_par_405686",
                    "name_via": "A58, K423, L578",
                    "distance_km": 536.12,
                    "duration_hours": 6,
                    "duration_minutes": 46,
                    "road_types": ["highway", "urban"],
                    "includes_toll": False,
                    "alias": ["second"],
                },
                {
                    "route_id": "rll_wie_par_985173",
                    "start_id": "loc_wie_683071",
                    "destination_id": "loc_par_405686",
                    "name_via": "L840",
                    "distance_km": 556.41,
                    "duration_hours": 7,
                    "duration_minutes": 2,
                    "road_types": ["urban"],
                    "includes_toll": False,
                    "alias": ["third"],
                },
            ]
        },
    }


def _base56_start(
    runtime: CARGuardOrchestrator, *, context_id: str, text: str = BASE_56_OFFICIAL_TEXT
):
    return runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=BASE_56_POLICY,
            user_text=text,
            live_tools=NAVIGATION_REPLACEMENT_TOOLS,
        )
    )


def test_base56_exact_named_waypoint_delete_uses_verified_direct_route() -> None:
    client = SequentialIntentClient([_base56_model_intent()])
    runtime = runtime_for(client)
    context_id = "base56-exact"

    detailed = _base56_start(runtime, context_id=context_id)
    assert detailed.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )

    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=_base56_navigation_state(),
        )
    )
    assert routes.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_wie_683071",
                "destination_id": "loc_par_405686",
            },
        },
    )

    delete = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=_base56_route_result(),
        )
    )
    assert delete.tool_calls == (
        {
            "tool_name": "navigation_delete_waypoint",
            "arguments": {
                "waypoint_id_to_delete": "loc_nur_485085",
                "route_id_without_waypoint": "rll_wie_par_899896",
            },
        },
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    goal = session.intent.goals[0]
    assert goal.desired_outcome["waypoint_name_to_delete"] == "Nuremberg"
    assert goal.desired_outcome["remaining_destination_name"] == "Paris"
    derived = session.derived_value_evidence_by_goal[goal.goal_id]
    for name in (
        "waypoint_id",
        "waypoint_name_to_delete",
        "remaining_destination_name",
        "replacement_route_id",
        "replacement_route_via",
        "replacement_route_includes_toll",
        "replacement_route_is_fastest",
    ):
        evidence = session.evidence.evidence[derived[name]]
        assert len(evidence.derived_from) == 2
    # The bounded named-delete grammar is grounded without an LLM extraction.
    assert client.intent_calls == 0
    assert client.action_calls == 0

    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-delete",
            tool_name="navigation_delete_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_deleted": True,
                    "new_waypoints": [
                        "loc_wie_683071",
                        "loc_par_405686",
                    ],
                    "new_routes": ["rll_wie_par_899896"],
                },
            },
        )
    )
    assert completed.tool_calls == ()
    assert completed.text == (
        "Done. I removed Nuremberg from the route. Navigation now continues "
        "directly to Paris via A11, A51. I used the fastest route for each "
        "segment. Would you like information about the alternative routes?"
    )


def test_hallucination54_missing_detailed_navigation_fields_stops_closed() -> None:
    client = SequentialIntentClient([_base56_model_intent()])
    runtime = runtime_for(client)
    context_id = "hallucination54-missing-navigation-fields"
    assert _base56_start(runtime, context_id=context_id).tool_calls

    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content={
                "status": "SUCCESS",
                "result": {
                    "navigation_active": True,
                    "details": {},
                },
            },
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert not any(
        read.tool_name == "get_routes_from_start_to_destination"
        for read in session.successful_read_results
    )
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "unsafe_via",
    [
        "POL-017",
        "A11.road",
        "A11. I also opened the trunk",
        "A11\x00A51",
        "A11,\nA51",
        "A11,\rA51",
        "A11,\x0bA51",
        "A11,\tA51",
    ],
)
def test_base56_unsafe_route_label_stops_before_delete(unsafe_via: str) -> None:
    client = SequentialIntentClient([_base56_model_intent()])
    runtime = runtime_for(client)
    context_id = f"base56-unsafe-via-{len(unsafe_via)}"
    assert _base56_start(runtime, context_id=context_id).tool_calls
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=_base56_navigation_state(),
        )
    )
    assert routes.tool_calls
    route_result = _base56_route_result()
    route_result["result"]["routes"][0]["name_via"] = unsafe_via

    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=route_result,
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base56_inconsistent_post_state_is_not_completed() -> None:
    client = SequentialIntentClient([_base56_model_intent()])
    runtime = runtime_for(client)
    context_id = "base56-inconsistent-post-state"
    assert _base56_start(runtime, context_id=context_id).tool_calls
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=_base56_navigation_state(),
        )
    )
    assert routes.tool_calls
    delete = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=_base56_route_result(),
        )
    )
    assert delete.tool_calls

    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-delete",
            tool_name="navigation_delete_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_deleted": True,
                    "new_waypoints": [],
                    "new_routes": ["rll_wie_par_899896"],
                },
            },
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None and "continues directly" not in blocked.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.goal_dag.goals[0].status.value != "done"


def test_base56_four_waypoint_delete_validates_full_post_state() -> None:
    client = SequentialIntentClient([_base56_model_intent()])
    runtime = runtime_for(client)
    context_id = "base56-four-waypoint-post-state"
    detailed = _base56_start(
        runtime,
        context_id=context_id,
        text="Remove the Nuremberg stop.",
    )
    assert detailed.tool_calls
    state = _base56_navigation_state()
    state["result"]["waypoints_id"] = [
        "loc_wie_683071",
        "loc_nur_485085",
        "loc_col_464166",
        "loc_par_405686",
    ]
    state["result"]["routes_to_final_destination_id"] = [
        "rll_wie_nur_519252",
        "rll_nur_col_739533",
        "rll_col_par_882834",
    ]
    state["result"]["details"]["waypoints"] = [
        {"id": "loc_wie_683071", "name": "Wiesbaden"},
        {"id": "loc_nur_485085", "name": "Nuremberg"},
        {"id": "loc_col_464166", "name": "Cologne"},
        {"id": "loc_par_405686", "name": "Paris"},
    ]
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=state,
        )
    )
    assert routes.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_wie_683071",
                "destination_id": "loc_col_464166",
            },
        },
    )
    route_result = _base56_route_result()
    for route in route_result["result"]["routes"]:
        route["destination_id"] = "loc_col_464166"
        route["route_id"] = route["route_id"].replace("_par_", "_col_")
    delete = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=route_result,
        )
    )
    assert delete.tool_calls == (
        {
            "tool_name": "navigation_delete_waypoint",
            "arguments": {
                "waypoint_id_to_delete": "loc_nur_485085",
                "route_id_without_waypoint": "rll_wie_col_899896",
            },
        },
    )
    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-delete",
            tool_name="navigation_delete_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_deleted": True,
                    "new_waypoints": [
                        "loc_wie_683071",
                        "loc_col_464166",
                        "loc_par_405686",
                    ],
                    "new_routes": [
                        "rll_wie_col_899896",
                        "rll_col_par_882834",
                    ],
                },
            },
        )
    )

    assert completed.text == (
        "Done. I removed Nuremberg from the route. Navigation now continues "
        "directly to Cologne via A11, A51. It then follows the existing route to "
        "Paris. I used the fastest route for the new direct segment. Would you like "
        "information about the alternative routes?"
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.goal_dag.goals[0].status.value == "done"


def test_base56_direct_destination_request_rejects_intervening_waypoint() -> None:
    client = SequentialIntentClient([_base56_model_intent()])
    runtime = runtime_for(client)
    context_id = "base56-direct-with-intervening-waypoint"
    assert _base56_start(runtime, context_id=context_id).tool_calls
    state = _base56_navigation_state()
    state["result"]["waypoints_id"] = [
        "loc_wie_683071",
        "loc_nur_485085",
        "loc_col_464166",
        "loc_par_405686",
    ]
    state["result"]["routes_to_final_destination_id"] = [
        "rll_wie_nur_519252",
        "rll_nur_col_739533",
        "rll_col_par_882834",
    ]
    state["result"]["details"]["waypoints"] = [
        {"id": "loc_wie_683071", "name": "Wiesbaden"},
        {"id": "loc_nur_485085", "name": "Nuremberg"},
        {"id": "loc_col_464166", "name": "Cologne"},
        {"id": "loc_par_405686", "name": "Paris"},
    ]

    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=state,
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert not any(
        read.tool_name == "get_routes_from_start_to_destination"
        for read in session.successful_read_results
    )
    assert session.budget.attempted_sets == set()


def test_base56_named_delete_retry_starts_verified_workflow() -> None:
    client = SequentialIntentClient([_empty_base50_intent()] * 4)
    runtime = runtime_for(client)
    context_id = "base56-delete-retry"

    vague = _base56_start(
        runtime,
        context_id=context_id,
        text="Remove the intermediate stop from my route.",
    )
    assert vague.tool_calls == ()
    assert vague.text is not None

    retry = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-retry",
            user_text="Remove the Nuremberg stop.",
        )
    )
    assert retry.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=_base56_navigation_state(),
        )
    )
    assert routes.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_wie_683071",
                "destination_id": "loc_par_405686",
            },
        },
    )
    delete = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=_base56_route_result(),
        )
    )
    assert delete.tool_calls == (
        {
            "tool_name": "navigation_delete_waypoint",
            "arguments": {
                "waypoint_id_to_delete": "loc_nur_485085",
                "route_id_without_waypoint": "rll_wie_par_899896",
            },
        },
    )
    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-delete",
            tool_name="navigation_delete_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_deleted": True,
                    "new_waypoints": ["loc_wie_683071", "loc_par_405686"],
                    "new_routes": ["rll_wie_par_899896"],
                },
            },
        )
    )
    assert completed.text == (
        "Done. I removed Nuremberg from the route. Navigation now continues "
        "directly to Paris via A11, A51. I used the fastest route for each "
        "segment. Would you like information about the alternative routes?"
    )
    # Only the initial vague request needs model interpretation; the named retry
    # is handled by the deterministic verified-delete workflow.
    assert client.intent_calls == 2
    assert client.action_calls == 0


def test_base56_route_clarification_accepts_exact_a11_a51_via() -> None:
    client = SequentialIntentClient([_base56_model_intent()])
    runtime = runtime_for(client)
    context_id = "base56-route-via-choice"
    assert _base56_start(runtime, context_id=context_id).tool_calls
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=_base56_navigation_state(),
        )
    )
    assert routes.tool_calls

    split = _base56_route_result()
    route_options = split["result"]["routes"]
    route_options[0]["alias"] = ["fastest", "first"]
    route_options[1]["alias"] = ["second", "shortest"]
    route_options[1]["distance_km"] = 520.0
    clarify = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=split,
        )
    )
    assert clarify.tool_calls == ()
    assert clarify.text is not None and "A11" in clarify.text

    delete = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-route-choice",
            user_text="Choose the one via A11/A51.",
        )
    )
    assert delete.tool_calls == (
        {
            "tool_name": "navigation_delete_waypoint",
            "arguments": {
                "waypoint_id_to_delete": "loc_nur_485085",
                "route_id_without_waypoint": "rll_wie_par_899896",
            },
        },
    )
    # Both the named delete and the exact displayed-route reply are deterministic.
    assert client.intent_calls == 0
    assert client.action_calls == 0
