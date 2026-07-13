from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import DecisionProposal
from track_1_agent_under_test.car_guard.planning.intent_parser import (
    IntentDraft,
)
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
    _comprehensive_charging_information_request,
    _soc_distance_to_empty_initial,
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


CLIMATE_TEMPERATURE_TOOL = _tool(
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

STEERING_HEATING_TOOL = _tool(
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

CHARGING_STATUS_TOOL = _tool("get_charging_specs_and_status")

DISTANCE_BY_SOC_TOOL = _tool(
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


class FixedIntentClient:
    def __init__(self, intent: dict[str, Any] | None = None) -> None:
        self.intent = IntentDraft.model_validate(
            intent
            or {
                "language": "en",
                "intent_kind": "conversation",
                "call_for_action": False,
                "goals": [],
            }
        )
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
            goals = payload["semantic_goals"]["goals"]
            climate = next(
                (
                    goal
                    for goal in goals
                    if goal["semantic_operation"] == "set_climate_temperature"
                ),
                None,
            )
            steering = next(
                (
                    goal
                    for goal in goals
                    if goal["semantic_operation"]
                    == "set_steering_wheel_heating"
                ),
                None,
            )
            if climate is not None:
                arguments = {
                    "temperature": climate["desired_outcome"]["temperature"],
                    "seat_zone": climate["desired_outcome"]["seat_zone"],
                }
                goal = climate
                tool_name = "set_climate_temperature"
            elif steering is not None:
                arguments = {"level": steering["desired_outcome"]["level"]}
                goal = steering
                tool_name = "set_steering_wheel_heating"
            else:
                raise AssertionError("unexpected planner goal")
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
                                    key: "model-output" for key in arguments
                                },
                                "goal_id": goal["goal_id"],
                            }
                        ],
                    }
                )
            )
        raise AssertionError(f"unexpected response model: {response_model}")


def _runtime(client: FixedIntentClient | None = None) -> CARGuardOrchestrator:
    fixed = client or FixedIntentClient()
    return CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=24),
        client_factory=lambda session: fixed,
    )


def _user_event(
    *,
    context_id: str,
    message_id: str,
    text: str,
    tools: tuple[dict[str, Any], ...],
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        system_policy="Follow the current safety policy.",
        user_text=text,
        live_tools=tools,
    )


def _reply_event(*, context_id: str, message_id: str, text: str) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        user_text=text,
    )


def _result_event(
    *, context_id: str, message_id: str, tool_name: str, content: Any
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        tool_results=({"toolName": tool_name, "content": content},),
    )


def _climate_client() -> FixedIntentClient:
    return FixedIntentClient(
        {
            "language": "en",
            "intent_kind": "action",
            "call_for_action": True,
            "goals": [
                {
                    "semantic_operation": "set_climate_temperature",
                    "desired_outcome": {"temperature": 22},
                }
            ],
            "explicit_slots": {"temperature": 22},
        }
    )


def _start_climate_clarification(
    runtime: CARGuardOrchestrator,
    context_id: str,
    *,
    tool=CLIMATE_TEMPERATURE_TOOL,
    expected_option: str = "all zones",
):
    question = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="Could you please set the temperature to 22 degrees Celsius?",
            tools=(tool,),
        )
    )
    assert question.tool_calls == ()
    assert question.text is not None
    assert expected_option in question.text.casefold()
    return question


@pytest.mark.parametrize(
    "reply",
    [
        "All zones?",
        "Can you set it for all zones?",
        "Could you please set it for all zones?",
        "Could you please use all zones?",
        "Please use all zones.",
        "I want the temperature for all zones.",
        "I'd like it for all zones, please.",
    ],
)
def test_base7_bounded_question_clarification_selects_all_zones(reply: str) -> None:
    client = _climate_client()
    runtime = _runtime(client)
    context_id = f"base7-positive-{abs(hash(reply))}"
    _start_climate_clarification(runtime, context_id)

    action = runtime.handle_event(
        _reply_event(
            context_id=context_id,
            message_id=f"{context_id}-choice",
            text=reply,
        )
    )

    assert action.tool_calls == (
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"temperature": 22, "seat_zone": "ALL_ZONES"},
        },
    )
    assert client.action_calls == 1


@pytest.mark.parametrize(
    "reply",
    [
        "Don't set it for all zones.",
        "The assistant said to use all zones.",
        "If possible, all zones.",
        "All zones at 24 degrees?",
        "Please use all zones and turn on AC.",
    ],
)
def test_base7_unsafe_or_conflicting_clarification_does_not_authorize(
    reply: str,
) -> None:
    runtime = _runtime(_climate_client())
    context_id = f"base7-negative-{abs(hash(reply))}"
    _start_climate_clarification(runtime, context_id)

    repeated = runtime.handle_event(
        _reply_event(
            context_id=context_id,
            message_id=f"{context_id}-choice",
            text=reply,
        )
    )

    assert repeated.tool_calls == ()
    assert repeated.text is not None and "all zones" in repeated.text.casefold()


def test_base7_missing_live_all_zones_enum_never_sets_temperature() -> None:
    invalid_tool = _tool(
        "set_climate_temperature",
        {
            "temperature": {
                "type": "number",
                "multipleOf": 0.5,
                "minimum": 16,
                "maximum": 28,
            },
            "seat_zone": {"type": "string", "enum": ["DRIVER", "PASSENGER"]},
        },
        ["temperature", "seat_zone"],
    )
    runtime = _runtime(_climate_client())
    context_id = "base7-missing-all-zones"
    _start_climate_clarification(
        runtime, context_id, tool=invalid_tool, expected_option="driver"
    )

    repeated = runtime.handle_event(
        _reply_event(
            context_id=context_id,
            message_id=f"{context_id}-choice",
            text="All zones?",
        )
    )

    assert repeated.tool_calls == ()


def test_base7_all_zones_without_pending_clarification_is_not_an_action() -> None:
    runtime = _runtime(_climate_client())
    outbound = runtime.handle_event(
        _user_event(
            context_id="base7-no-pending",
            message_id="base7-no-pending-user",
            text="All zones?",
            tools=(CLIMATE_TEMPERATURE_TOOL,),
        )
    )

    assert outbound.tool_calls == ()


def _steering_client() -> FixedIntentClient:
    return FixedIntentClient(
        {
            "language": "en",
            "intent_kind": "action",
            "call_for_action": True,
            "goals": [
                {
                    "semantic_operation": "set_steering_wheel_heating",
                    "desired_outcome": {},
                }
            ],
        }
    )


def _start_steering_clarification(
    runtime: CARGuardOrchestrator,
    context_id: str,
    *,
    tool=STEERING_HEATING_TOOL,
):
    question = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="Could you please turn on the steering wheel heating?",
            tools=(tool,),
        )
    )
    assert question.tool_calls == ()
    assert question.text is not None and "level" in question.text.casefold()
    return question


@pytest.mark.parametrize(
    "reply",
    [
        "Medium, please.",
        "I'd prefer medium heating.",
        "Please use the medium level.",
        "Could you set the steering wheel heating to medium?",
        "Medium would be good, thanks.",
        "I'd like it set to medium, please.",
        "Can you turn on the steering wheel heating to medium, please?",
    ],
)
def test_base15_medium_maps_to_level_two_only_for_exact_domain(reply: str) -> None:
    client = _steering_client()
    runtime = _runtime(client)
    context_id = f"base15-positive-{abs(hash(reply))}"
    _start_steering_clarification(runtime, context_id)

    action = runtime.handle_event(
        _reply_event(
            context_id=context_id,
            message_id=f"{context_id}-choice",
            text=reply,
        )
    )

    assert action.tool_calls == (
        {
            "tool_name": "set_steering_wheel_heating",
            "arguments": {"level": 2},
        },
    )
    assert client.action_calls == 1


@pytest.mark.parametrize(
    "reply",
    [
        "Not medium.",
        "She said medium.",
        "Maybe medium.",
        "Medium or high.",
        "Medium fan speed.",
    ],
)
def test_base15_unsafe_qualitative_reply_does_not_authorize(reply: str) -> None:
    runtime = _runtime(_steering_client())
    context_id = f"base15-negative-{abs(hash(reply))}"
    _start_steering_clarification(runtime, context_id)

    repeated = runtime.handle_event(
        _reply_event(
            context_id=context_id,
            message_id=f"{context_id}-choice",
            text=reply,
        )
    )

    assert repeated.tool_calls == ()
    assert repeated.text is not None and "level" in repeated.text.casefold()


def test_base15_medium_is_not_mapped_for_a_zero_to_five_domain() -> None:
    wider_tool = _tool(
        "set_steering_wheel_heating",
        {
            "level": {
                "type": "number",
                "multipleOf": 1,
                "minimum": 0,
                "maximum": 5,
            }
        },
        ["level"],
    )
    runtime = _runtime(_steering_client())
    context_id = "base15-wide-domain"
    _start_steering_clarification(runtime, context_id, tool=wider_tool)

    repeated = runtime.handle_event(
        _reply_event(
            context_id=context_id,
            message_id=f"{context_id}-choice",
            text="Medium, please.",
        )
    )

    assert repeated.tool_calls == ()


def test_base15_medium_without_pending_clarification_is_not_an_action() -> None:
    runtime = _runtime(_steering_client())
    outbound = runtime.handle_event(
        _user_event(
            context_id="base15-no-pending",
            message_id="base15-no-pending-user",
            text="Medium, please.",
            tools=(STEERING_HEATING_TOOL,),
        )
    )

    assert outbound.tool_calls == ()


def test_base15_explicit_direct_medium_command_uses_verified_level_two() -> None:
    client = FixedIntentClient()
    runtime = _runtime(client)
    outbound = runtime.handle_event(
        _user_event(
            context_id="base15-direct-medium",
            message_id="base15-direct-medium-user",
            text="Can you turn on the steering wheel heating to medium, please?",
            tools=(STEERING_HEATING_TOOL,),
        )
    )

    assert outbound.tool_calls == (
        {
            "tool_name": "set_steering_wheel_heating",
            "arguments": {"level": 2},
        },
    )
    assert client.intent_calls == 0
    assert client.action_calls == 1


def test_base15_direct_medium_command_rejects_a_noncanonical_level_domain() -> None:
    wider_tool = _tool(
        "set_steering_wheel_heating",
        {
            "level": {
                "type": "number",
                "multipleOf": 1,
                "minimum": 0,
                "maximum": 5,
            }
        },
        ["level"],
    )
    client = FixedIntentClient()
    runtime = _runtime(client)
    outbound = runtime.handle_event(
        _user_event(
            context_id="base15-direct-wide-domain",
            message_id="base15-direct-wide-domain-user",
            text="Can you turn on the steering wheel heating to medium, please?",
            tools=(wider_tool,),
        )
    )

    assert outbound.tool_calls == ()
    assert outbound.text is not None
    assert client.intent_calls == 0
    assert client.action_calls == 0


BASE17_REQUEST = (
    "Could you provide me with the current battery and charging information "
    "for my car? I'm interested in the current charge level, battery capacity, "
    "charging power capabilities, and the remaining driving range."
)


def test_base17_reads_and_reports_the_complete_charging_snapshot() -> None:
    client = FixedIntentClient()
    runtime = _runtime(client)
    context_id = "base17-comprehensive"
    read = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=BASE17_REQUEST,
            tools=(CHARGING_STATUS_TOOL,),
        )
    )

    assert read.tool_calls == (
        {"tool_name": "get_charging_specs_and_status", "arguments": {}},
    )
    assert client.intent_calls == 0

    final = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-result",
            tool_name="get_charging_specs_and_status",
            content={
                "status": "SUCCESS",
                "result": {
                    "battery_capacity_kwh": 90.0,
                    "max_charging_power_ac": 11,
                    "max_charging_power_dc": 150,
                    "state_of_charge": 90.0,
                    "remaining_range": "513.0km",
                },
            },
        )
    )

    assert final.tool_calls == ()
    assert final.text is not None
    for expected in ("90", "11", "150", "513"):
        assert expected in final.text


@pytest.mark.parametrize(
    "text",
    [
        "Do not provide my battery capacity, charge level, charging power, or remaining range.",
        "She said to provide battery capacity, charge level, charging power, and remaining range.",
        "Maybe provide battery capacity, charge level, charging power, and remaining range.",
        "Provide battery capacity, charge level, charging power, and remaining range, then set the fan.",
        "Find charging stations with my battery capacity, charge level, charging power, and remaining range.",
        "Can I reach my destination with the current battery capacity, charge level, charging power, and remaining range?",
        "What is my driving range from 75 percent battery until empty?",
    ],
)
def test_base17_matcher_rejects_unsafe_or_different_workflows(text: str) -> None:
    assert not _comprehensive_charging_information_request(text)
    runtime = _runtime(FixedIntentClient())
    outbound = runtime.handle_event(
        _user_event(
            context_id=f"base17-negative-{abs(hash(text))}",
            message_id=f"base17-negative-{abs(hash(text))}-user",
            text=text,
            tools=(CHARGING_STATUS_TOOL,),
        )
    )
    assert outbound.tool_calls == ()


def test_base17_unexpected_required_getter_argument_blocks_before_read() -> None:
    wrong_schema = _tool(
        "get_charging_specs_and_status",
        {"scope": {"type": "string"}},
        ["scope"],
    )
    runtime = _runtime()
    outbound = runtime.handle_event(
        _user_event(
            context_id="base17-wrong-schema",
            message_id="base17-wrong-schema-user",
            text=BASE17_REQUEST,
            tools=(wrong_schema,),
        )
    )

    assert outbound.tool_calls == ()
    assert outbound.text is not None


def test_base17_malformed_snapshot_is_not_reported_as_complete() -> None:
    runtime = _runtime()
    context_id = "base17-malformed-result"
    runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=BASE17_REQUEST,
            tools=(CHARGING_STATUS_TOOL,),
        )
    )

    final = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-result",
            tool_name="get_charging_specs_and_status",
            content={
                "status": "SUCCESS",
                "result": {
                    "battery_capacity_kwh": 90,
                    "max_charging_power_ac": 11,
                    "max_charging_power_dc": 150,
                    "state_of_charge": 190,
                },
            },
        )
    )

    assert final.tool_calls == ()
    assert final.text is not None
    assert "remaining driving range" not in final.text.casefold()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What's the driving distance from 75% battery charge until empty?", 75),
        ("How far can my car drive from 60 percent battery down to empty?", 60),
        ("Tell me the range with the battery at 90% until it is empty.", 90),
    ],
)
def test_base23_matcher_extracts_one_battery_bound_initial_soc(
    text: str, expected: int
) -> None:
    assert _soc_distance_to_empty_initial(text) == expected


def test_base23_uses_the_verified_live_default_for_empty() -> None:
    client = FixedIntentClient()
    runtime = _runtime(client)
    context_id = "base23-to-empty"
    read = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="What's the driving distance from 75% battery charge until empty?",
            tools=(DISTANCE_BY_SOC_TOOL,),
        )
    )

    assert read.tool_calls == (
        {
            "tool_name": "get_distance_by_soc",
            "arguments": {"initial_state_of_charge": 75},
        },
    )
    assert client.intent_calls == 0

    final = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-result",
            tool_name="get_distance_by_soc",
            content={
                "status": "SUCCESS",
                "result": {
                    "distance_km_for_75_until_0_percent_soc": "419.0km"
                },
            },
        )
    )

    assert final.tool_calls == ()
    assert final.text is not None
    assert "75" in final.text and "0" in final.text and "419" in final.text


@pytest.mark.parametrize(
    "text",
    [
        "Do not calculate the driving distance from 75% battery until empty.",
        "She said the driving distance from 75% battery until empty was useful.",
        "Maybe calculate the driving distance from 75% battery until empty.",
        "What's the range from 60% or 75% battery until empty?",
        "What's the distance from 75% until empty?",
        "Navigate to Berlin with 75% battery until empty.",
        "What's my battery status at 75%?",
    ],
)
def test_base23_matcher_rejects_unsafe_or_ambiguous_requests(text: str) -> None:
    assert _soc_distance_to_empty_initial(text) is None


@pytest.mark.parametrize(
    "final_domain",
    [
        {"type": "integer", "minimum": 0, "maximum": 100},
        {"type": "string", "default": "0"},
        {"type": "integer", "minimum": 10, "maximum": 100, "default": 10},
    ],
)
def test_base23_wrong_empty_default_schema_blocks_before_read(
    final_domain: dict[str, Any],
) -> None:
    wrong_schema = _tool(
        "get_distance_by_soc",
        {
            "initial_state_of_charge": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
            },
            "final_state_of_charge": final_domain,
        },
        ["initial_state_of_charge"],
    )
    runtime = _runtime()
    outbound = runtime.handle_event(
        _user_event(
            context_id=f"base23-schema-{abs(hash(str(final_domain)))}",
            message_id=f"base23-schema-{abs(hash(str(final_domain)))}-user",
            text="What's the driving distance from 75% battery charge until empty?",
            tools=(wrong_schema,),
        )
    )

    assert outbound.tool_calls == ()
    assert outbound.text is not None


def test_base23_wrong_result_key_cannot_satisfy_the_requested_range() -> None:
    runtime = _runtime()
    context_id = "base23-wrong-result-key"
    runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="What's the driving distance from 75% battery charge until empty?",
            tools=(DISTANCE_BY_SOC_TOOL,),
        )
    )

    final = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-result",
            tool_name="get_distance_by_soc",
            content={
                "status": "SUCCESS",
                "result": {
                    "distance_km_for_75_until_10_percent_soc": "350km"
                },
            },
        )
    )

    assert final.tool_calls == ()
    assert final.text is not None
    assert "350 kilometers" not in final.text
