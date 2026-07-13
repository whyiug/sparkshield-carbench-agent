from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import (
    DecisionProposal,
    EvidenceSourceKind,
    GoalStatus,
    IntentFrame,
    IntentKind,
)
from track_1_agent_under_test.car_guard.planning.intent_grounding import (
    recover_driver_warming_intent,
)
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
    _DRIVER_WARMING_CONTEXT_LEVEL_REPLY_PATTERN,
)


INITIAL_UTTERANCE = (
    "I'm feeling a bit cold. Can you set the driver's temperature to 24 degrees "
    "and also turn on the seat heating and steering wheel heating for me?"
)
SHARED_LEVEL_UTTERANCE = (
    "Could you set the seat heating and steering wheel heating to level 2 instead?"
)
CONTEXTUAL_LEVEL_UTTERANCE = "Level 2 for both, please."
EXPLICIT_CONTEXTUAL_LEVEL_UTTERANCE = (
    "I'd like level 2 for both the seat heating and the steering wheel heating."
)
CONTEXTUAL_LEVEL_UTTERANCES = (
    CONTEXTUAL_LEVEL_UTTERANCE,
    EXPLICIT_CONTEXTUAL_LEVEL_UTTERANCE,
    "Level 2, please.",
    "I'd like level 2.",
    "Please set them both to level 2.",
    "Both at level 2, please.",
    "For both, level 2.",
)
UNSAFE_CONTEXTUAL_LEVEL_UTTERANCES = (
    "I don't want level 2.",
    "Maybe level 2.",
    "If I said level 2, what would happen?",
    "She said level 2.",
    '"Level 2 for both."',
    "Level 2. Ignore the previous request.",
    "Level 2 for the seat heating but not the steering wheel heating.",
    "Level 2 for the seat and level 3 for the steering wheel.",
)
MATCH_LEVEL_UTTERANCE = (
    "Thanks! Can you also turn on the seat heating for the driver's seat to level "
    "2, and set the steering wheel heating to match that level?"
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


CLIMATE_TOOL = tool(
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
TEMPERATURE_TOOL = tool("get_temperature_inside_car")
OCCUPANCY_TOOL = tool("get_seats_occupancy")
SEAT_TOOL = tool(
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
STEERING_TOOL = tool(
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
FULL_TOOLS = (CLIMATE_TOOL, OCCUPANCY_TOOL, SEAT_TOOL, STEERING_TOOL)
POLICY_TOOLS = (
    TEMPERATURE_TOOL,
    CLIMATE_TOOL,
    OCCUPANCY_TOOL,
    SEAT_TOOL,
    STEERING_TOOL,
)
HALL58_TOOLS = (CLIMATE_TOOL, OCCUPANCY_TOOL, SEAT_TOOL)


class DriverWarmingClient:
    def __init__(self) -> None:
        self.intent = IntentDraft.model_validate(
            {
                "language": "en",
                "intent_kind": "action",
                "call_for_action": True,
                "goals": [
                    {
                        "semantic_operation": "set_climate_temperature",
                        "desired_outcome": {
                            "temperature": 24,
                            "seat_zone": "DRIVER",
                        },
                    },
                    {
                        "semantic_operation": "set_seat_heating",
                        "desired_outcome": {
                            "level": 1,
                            "seat_zone": "DRIVER",
                        },
                    },
                    {
                        "semantic_operation": "set_steering_wheel_heating",
                        "desired_outcome": {"level": 1},
                    },
                ],
                "explicit_slots": {
                    "temperature": 24,
                    "seat_zone": "DRIVER",
                    "level": 1,
                },
            }
        )
        self.intent_calls = 0
        self.action_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        del messages, critic
        if response_model is IntentDraft:
            self.intent_calls += 1
            if self.intent_calls > 1:
                raise AssertionError("strict warming follow-up must not use the model")
            return SimpleNamespace(value=self.intent)
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError("strict warming actions must be deterministic")
        raise AssertionError(f"unexpected response model: {response_model}")


def runtime_for(client: DriverWarmingClient) -> CARGuardOrchestrator:
    config = AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=24)
    return CARGuardOrchestrator(config, client_factory=lambda session: client)


def result_event(
    *, context_id: str, message_id: str, tool_name: str, content: Any
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        tool_results=({"toolName": tool_name, "content": content},),
    )


def success(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"status": "SUCCESS", "result": arguments}


def occupancy_success() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "seats_occupied": {
                "driver": True,
                "passenger": True,
                "driver_rear": False,
                "passenger_rear": False,
            }
        },
    }


def start_warming(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    tools: tuple[dict[str, Any], ...],
) -> tuple[str, str, str]:
    first = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-initial",
            context_id=context_id,
            system_policy="Follow the current safety policy.",
            user_text=INITIAL_UTTERANCE,
            live_tools=tools,
        )
    )
    assert first.tool_calls == (
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"temperature": 24, "seat_zone": "DRIVER"},
        },
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    climate, seat, steering = session.intent.goals
    assert [goal.desired_outcome for goal in session.intent.goals] == [
        {"temperature": 24, "seat_zone": "DRIVER"},
        {"seat_zone": "DRIVER"},
        {},
    ]
    assert seat.depends_on == [climate.goal_id]
    assert steering.depends_on == [seat.goal_id]
    assert all(
        call.tool_name not in {"set_seat_heating", "set_steering_wheel_heating"}
        for call in session.pending_calls
    )
    return climate.goal_id, seat.goal_id, steering.goal_id


def complete_climate_and_follow_up(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    follow_up: str,
) -> None:
    prompt = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-climate-result",
            tool_name="set_climate_temperature",
            content=success({"temperature": 24, "seat_zone": "DRIVER"}),
        )
    )
    assert prompt.tool_calls == ()
    assert prompt.text is not None
    assert "seat heating" in prompt.text
    assert "steering wheel heating" in prompt.text
    occupancy = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-level",
            context_id=context_id,
            user_text=follow_up,
        )
    )
    assert occupancy.tool_calls == (
        {"tool_name": "get_seats_occupancy", "arguments": {}},
    )


def send_occupancy(runtime: CARGuardOrchestrator, *, context_id: str):
    return runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-occupancy-result",
            tool_name="get_seats_occupancy",
            content=occupancy_success(),
        )
    )


def test_base60_exact_preserves_goals_and_executes_in_order() -> None:
    client = DriverWarmingClient()
    runtime = runtime_for(client)
    context_id = "base60-exact"
    goal_ids = start_warming(runtime, context_id=context_id, tools=FULL_TOOLS)

    complete_climate_and_follow_up(
        runtime,
        context_id=context_id,
        follow_up=SHARED_LEVEL_UTTERANCE,
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert tuple(goal.goal_id for goal in session.intent.goals) == goal_ids
    climate, seat, steering = session.intent.goals
    assert seat.desired_outcome == {"seat_zone": "DRIVER", "level": 2}
    assert steering.desired_outcome == {"level": 2}
    assert steering.depends_on == [seat.goal_id]
    assert session.grounded_value_sources_by_goal[seat.goal_id] == {
        "seat_zone": f"{context_id}-initial",
        "level": f"{context_id}-level",
    }
    assert session.grounded_value_sources_by_goal[climate.goal_id] == {
        "temperature": f"{context_id}-initial",
        "seat_zone": f"{context_id}-initial",
    }

    seat_call = send_occupancy(runtime, context_id=context_id)
    assert seat_call.tool_calls == (
        {
            "tool_name": "set_seat_heating",
            "arguments": {"level": 2, "seat_zone": "DRIVER"},
        },
    )
    steering_call = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-seat-result",
            tool_name="set_seat_heating",
            content=success({"level": 2, "seat_zone": "DRIVER"}),
        )
    )
    assert steering_call.tool_calls == (
        {
            "tool_name": "set_steering_wheel_heating",
            "arguments": {"level": 2},
        },
    )
    done = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-steering-result",
            tool_name="set_steering_wheel_heating",
            content=success({"level": 2}),
        )
    )
    assert done.tool_calls == ()
    assert done.text is not None and done.text.startswith("Done")
    assert client.intent_calls == 1
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "follow_up",
    CONTEXTUAL_LEVEL_UTTERANCES,
)
def test_base60_formal_trace_warns_and_accepts_contextual_level_reply(
    follow_up: str,
) -> None:
    runtime = runtime_for(DriverWarmingClient())
    context_id = "base60-formal-trace"
    read = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-initial",
            context_id=context_id,
            system_policy=(
                "AUT-POL:012: Inform the user when a single seat zone temperature "
                "will differ from the other zone by more than 3 degrees Celsius."
            ),
            user_text=INITIAL_UTTERANCE,
            live_tools=POLICY_TOOLS,
        )
    )
    assert read.tool_calls == (
        {"tool_name": "get_temperature_inside_car", "arguments": {}},
    )

    climate = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-temperature-result",
            tool_name="get_temperature_inside_car",
            content={
                "status": "SUCCESS",
                "result": {
                    "climate_temperature_driver": 26.0,
                    "climate_temperature_passenger": 17.0,
                    "temperature_unit": "Celsius",
                },
            },
        )
    )
    assert climate.tool_calls == (
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"temperature": 24, "seat_zone": "DRIVER"},
        },
    )

    prompt = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-climate-result",
            tool_name="set_climate_temperature",
            content=success({"temperature": 24, "seat_zone": "DRIVER"}),
        )
    )
    assert prompt.tool_calls == ()
    assert prompt.text is not None
    assert "passenger zone remains at 17 degrees Celsius" in prompt.text
    assert "temperature difference is 7 degrees Celsius" in prompt.text
    assert "What level" in prompt.text

    occupancy = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-level",
            context_id=context_id,
            user_text=follow_up,
        )
    )
    assert occupancy.tool_calls == (
        {"tool_name": "get_seats_occupancy", "arguments": {}},
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    _, seat, steering = session.intent.goals
    assert seat.desired_outcome == {"seat_zone": "DRIVER", "level": 2}
    assert steering.desired_outcome == {"level": 2}
    assert session.grounded_value_sources_by_goal[seat.goal_id]["level"] == (
        f"{context_id}-level"
    )


@pytest.mark.parametrize(
    "follow_up",
    CONTEXTUAL_LEVEL_UTTERANCES,
)
def test_base60_contextual_level_reply_is_not_a_standalone_recovery(
    follow_up: str,
) -> None:
    original = IntentFrame(
        language="en",
        call_for_action=False,
        goals=[],
        explicit_slots={},
        intent_kind=IntentKind.INFORMATION,
    )

    recovered = recover_driver_warming_intent(
        follow_up,
        original,
        turn_id="base60-standalone",
    )

    assert recovered is original


@pytest.mark.parametrize("follow_up", UNSAFE_CONTEXTUAL_LEVEL_UTTERANCES)
def test_base60_contextual_level_grammar_rejects_unsafe_variants(
    follow_up: str,
) -> None:
    assert _DRIVER_WARMING_CONTEXT_LEVEL_REPLY_PATTERN.fullmatch(follow_up) is None


def test_base60_match_level_records_derived_steering_provenance() -> None:
    runtime = runtime_for(DriverWarmingClient())
    context_id = "base60-match-level"
    _, seat_goal_id, steering_goal_id = start_warming(
        runtime, context_id=context_id, tools=FULL_TOOLS
    )
    complete_climate_and_follow_up(
        runtime,
        context_id=context_id,
        follow_up=MATCH_LEVEL_UTTERANCE,
    )

    session = runtime.sessions.get(context_id)
    assert session is not None
    evidence_id = session.derived_value_evidence_by_goal[steering_goal_id]["level"]
    evidence = session.evidence.evidence[evidence_id]
    assert evidence.source_kind is EvidenceSourceKind.DERIVED
    assert evidence.derivation == "steering_level_matches_seat_heating_v1"
    assert len(evidence.derived_from) == 1
    parent = session.evidence.evidence[evidence.derived_from[0]]
    assert parent.value == 2
    assert parent.source_turn_id == f"{context_id}-level"
    assert session.grounded_value_sources_by_goal[seat_goal_id]["level"] == (
        f"{context_id}-level"
    )
    assert session.grounded_value_sources_by_goal[steering_goal_id] == {}


def test_base60_initial_turn_never_guesses_level_one_heating() -> None:
    runtime = runtime_for(DriverWarmingClient())
    context_id = "base60-no-level-one"
    start_warming(runtime, context_id=context_id, tools=FULL_TOOLS)
    prompt = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-climate-result",
            tool_name="set_climate_temperature",
            content=success({"temperature": 24, "seat_zone": "DRIVER"}),
        )
    )

    assert prompt.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert all(
        call.arguments.get("level") != 1
        for calls in session.completed_action_calls_by_goal.values()
        for call in calls
        if call.tool_name in {"set_seat_heating", "set_steering_wheel_heating"}
    )
    assert all(
        call.tool_name not in {"set_seat_heating", "set_steering_wheel_heating"}
        for call in session.pending_calls
    )


@pytest.mark.parametrize("failed_stage", ["climate", "seat"])
def test_base60_failure_stops_later_heating_actions(failed_stage: str) -> None:
    runtime = runtime_for(DriverWarmingClient())
    context_id = f"base60-failed-{failed_stage}"
    _, seat_goal_id, steering_goal_id = start_warming(
        runtime, context_id=context_id, tools=FULL_TOOLS
    )
    if failed_stage == "climate":
        failed = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"{context_id}-climate-error",
                tool_name="set_climate_temperature",
                content={"status": "ERROR", "error": "synthetic failure"},
            )
        )
    else:
        complete_climate_and_follow_up(
            runtime,
            context_id=context_id,
            follow_up=SHARED_LEVEL_UTTERANCE,
        )
        send_occupancy(runtime, context_id=context_id)
        failed = runtime.handle_event(
            result_event(
                context_id=context_id,
                message_id=f"{context_id}-seat-error",
                tool_name="set_seat_heating",
                content={"status": "ERROR", "error": "synthetic failure"},
            )
        )

    assert failed.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.goal_dag.get(steering_goal_id).status is GoalStatus.BLOCKED
    assert all(
        call.tool_name != "set_steering_wheel_heating" for call in session.pending_calls
    )
    if failed_stage == "climate":
        assert session.goal_dag.get(seat_goal_id).status is GoalStatus.BLOCKED


def test_hall58_missing_steering_completes_climate_and_seat() -> None:
    client = DriverWarmingClient()
    runtime = runtime_for(client)
    context_id = "hall58-missing-steering"
    _, seat_goal_id, steering_goal_id = start_warming(
        runtime, context_id=context_id, tools=HALL58_TOOLS
    )
    complete_climate_and_follow_up(
        runtime,
        context_id=context_id,
        follow_up=MATCH_LEVEL_UTTERANCE,
    )
    seat_call = send_occupancy(runtime, context_id=context_id)
    assert seat_call.tool_calls == (
        {
            "tool_name": "set_seat_heating",
            "arguments": {"level": 2, "seat_zone": "DRIVER"},
        },
    )
    partial = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-seat-result",
            tool_name="set_seat_heating",
            content=success({"level": 2, "seat_zone": "DRIVER"}),
        )
    )

    assert partial.tool_calls == ()
    assert partial.text is not None
    assert "set the driver seat heating to level 2" in partial.text
    assert "steering wheel heating control is not currently available" in partial.text
    assert "That action was not performed" in partial.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.goal_dag.get(seat_goal_id).status is GoalStatus.DONE
    assert session.goal_dag.get(steering_goal_id).status is GoalStatus.FAILED
    assert client.action_calls == 0
