from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.llm.client import LLMFailure
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
    _current_weather_fog_lights_request,
    _dependent_seat_heating_capability_request,
    _seat_heating_capability_request,
    _seat_occupancy_information_request,
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


OCCUPANCY = tool("get_seats_occupancy")
CLIMATE = tool("get_climate_settings")
SEAT_HEATING_READ = tool("get_seat_heating_level")
SEAT_HEATING = tool(
    "set_seat_heating",
    {
        "level": {"type": "integer", "minimum": 0, "maximum": 3},
        "seat_zone": {
            "type": "string",
            "enum": ["ALL_ZONES", "DRIVER", "PASSENGER"],
        },
    },
    ["level", "seat_zone"],
)
SEAT_HEATING_WITHOUT_ZONE = tool(
    "set_seat_heating",
    {"level": {"type": "integer", "minimum": 0, "maximum": 3}},
    ["level"],
)
CLIMATE_SET = tool(
    "set_climate_temperature",
    {
        "temperature": {"type": "number", "minimum": 16, "maximum": 30},
        "seat_zone": {
            "type": "string",
            "enum": ["ALL_ZONES", "DRIVER", "PASSENGER"],
        },
    },
    ["temperature", "seat_zone"],
)
STEERING_SET = tool(
    "set_steering_wheel_heating",
    {"level": {"type": "integer", "minimum": 0, "maximum": 3}},
    ["level"],
)

EMPTY_SEAT_REQUEST = (
    "Okay, that helps. Since it is just me, can you please turn off the seat "
    "heating for all the other seats that are empty? There is no need to heat "
    "empty seats."
)
MATCHED_HEATING_REQUEST = (
    "I am feeling cold. Could you please set the passenger side temperature to "
    "24 degrees Celsius? Also, I would like the passenger seat heating to match "
    "the driver's seat heating level, and the steering wheel heating to match "
    "that same level."
)
OCCUPANCY_REQUEST = (
    "Can you check if there are any passengers in the car right now? I am trying "
    "to figure out if we are heating empty seats."
)
COMPOUND_INFORMATION_REQUEST = (
    "Could you tell me what the current climate settings are and who is in the "
    "car right now? I am trying to save some energy."
)
PUBLIC_SHAPED_COMPOUND_INFORMATION_REQUEST = (
    "Hey there! I'm trying to save a bit of energy. Can you tell me what the "
    "current climate settings are and who's in the car right now?"
)
DEPENDENT_PASSENGER_WARMING_REQUEST = (
    "I'm feeling a bit cold. Could you please set the passenger side temperature "
    "to 24 degrees Celsius? Also, I'd like the passenger seat heating to match "
    "the driver's seat heating level, and the steering wheel heating to be at "
    "the same level as the passenger seat heating."
)


class CountingFailureClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs: Any) -> Any:
        del kwargs
        self.calls += 1
        raise LLMFailure("synthetic planner unavailable")


class NoModelClient:
    def generate(self, **kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("deterministic seat workflows must not call a model")


class FogIntentClient:
    def generate(self, *, response_model: Any, **kwargs: Any) -> Any:
        del kwargs
        if response_model is IntentDraft:
            return SimpleNamespace(
                value=IntentDraft.model_validate(
                    {
                        "language": "en",
                        "intent_kind": "action",
                        "call_for_action": True,
                        "goals": [
                            {
                                "semantic_operation": "set_fog_lights",
                                "desired_outcome": {"enabled": True},
                            }
                        ],
                        "explicit_slots": {"enabled": True},
                    }
                )
            )
        raise LLMFailure("synthetic action planner unavailable")


class TemperatureClarificationThenFailureClient:
    def __init__(self) -> None:
        self.intent_calls = 0

    def generate(self, *, response_model: Any, **kwargs: Any) -> Any:
        del kwargs
        if response_model is IntentDraft:
            self.intent_calls += 1
            if self.intent_calls == 1:
                return SimpleNamespace(
                    value=IntentDraft.model_validate(
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
                            "ambiguities": [
                                {
                                    "name": "seat_zone",
                                    "candidate_values": [
                                        "ALL_ZONES",
                                        "DRIVER",
                                        "PASSENGER",
                                    ],
                                    "goal_index": 0,
                                }
                            ],
                        }
                    )
                )
        raise LLMFailure("synthetic planner unavailable")


def runtime(client: Any) -> CARGuardOrchestrator:
    return CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=24),
        client_factory=lambda session: client,
    )


def user_event(
    context_id: str,
    message_id: str,
    text: str,
    tools: tuple[dict[str, Any], ...],
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        system_policy="Use only available controls and report missing capabilities.",
        user_text=text,
        live_tools=tools,
    )


def result_event(
    context_id: str,
    message_id: str,
    *results: tuple[str, dict[str, Any]],
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        tool_results=tuple(
            {"toolName": tool_name, "content": content}
            for tool_name, content in results
        ),
    )


def occupancy_result(*, extra: bool = False) -> dict[str, Any]:
    seats: dict[str, Any] = {
        "driver": True,
        "passenger": False,
        "driver_rear": False,
        "passenger_rear": False,
    }
    if extra:
        seats["cargo"] = False
    return {"status": "SUCCESS", "result": {"seats_occupied": seats}}


def climate_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "fan_speed": 0,
            "fan_airflow_direction": "WINDSHIELD_FEET",
            "air_conditioning": False,
            "air_circulation": "AUTO",
            "window_front_defrost": False,
            "window_rear_defrost": False,
        },
    }


@pytest.mark.parametrize(
    "utterance",
    (EMPTY_SEAT_REQUEST, "Turn off the passenger seat heating."),
)
def test_missing_seat_heating_control_is_reported_before_model_or_tools(
    utterance: str,
) -> None:
    client = CountingFailureClient()
    agent = runtime(client)
    blocked = agent.handle_event(
        user_event(
            "missing-seat-control",
            "missing-seat-control-user",
            utterance,
            (OCCUPANCY,),
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "seat heating control is not currently available" in blocked.text.casefold()
    assert "did not set seat heating" in blocked.text.casefold()
    assert client.calls == 0
    session = agent.sessions.get("missing-seat-control")
    assert session is not None
    assert session.authorized_action_goal_ids == set()
    assert session.budget.attempted_sets == set()


def test_missing_explicit_seat_zone_parameter_is_reported_before_clarification() -> None:
    client = CountingFailureClient()
    agent = runtime(client)
    blocked = agent.handle_event(
        user_event(
            "missing-seat-zone",
            "missing-seat-zone-user",
            "Turn off the passenger seat heating.",
            (SEAT_HEATING_WITHOUT_ZONE,),
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    lowered = blocked.text.casefold()
    assert "seat heating control" in lowered
    assert "required seat zone setting" in lowered
    assert "which option" not in lowered
    assert "did not set seat heating" in lowered
    assert client.calls == 0


@pytest.mark.parametrize(
    "seat_controls",
    ((), (SEAT_HEATING_WITHOUT_ZONE,)),
)
def test_dependent_seat_sync_stops_as_one_unit_when_target_control_is_missing(
    seat_controls: tuple[dict[str, Any], ...],
) -> None:
    client = CountingFailureClient()
    agent = runtime(client)
    blocked = agent.handle_event(
        user_event(
            "dependent-seat-sync-missing",
            "dependent-seat-sync-missing-user",
            DEPENDENT_PASSENGER_WARMING_REQUEST,
            (CLIMATE_SET, SEAT_HEATING_READ, *seat_controls, STEERING_SET),
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    lowered = blocked.text.casefold()
    assert "cannot safely match the passenger seat heating" in lowered
    assert "passenger temperature" in lowered
    assert "steering-wheel heating" in lowered
    assert "which option" not in lowered
    assert client.calls == 0
    session = agent.sessions.get("dependent-seat-sync-missing")
    assert session is not None
    assert session.budget.attempted_sets == set()


@pytest.mark.parametrize(
    "controls",
    (
        (CLIMATE_SET, SEAT_HEATING, STEERING_SET),
        (CLIMATE_SET, SEAT_HEATING_READ, SEAT_HEATING),
        (
            tool(
                "set_climate_temperature",
                {
                    "temperature": {"type": "number", "enum": [22]},
                    "seat_zone": {"type": "string", "enum": ["PASSENGER"]},
                },
                ["temperature", "seat_zone"],
            ),
            SEAT_HEATING_READ,
            SEAT_HEATING,
            STEERING_SET,
        ),
        (
            CLIMATE_SET,
            tool(
                "get_seat_heating_level",
                {"scope": {"type": "string"}},
                ["scope"],
            ),
            SEAT_HEATING,
            STEERING_SET,
        ),
        (
            CLIMATE_SET,
            SEAT_HEATING_READ,
            tool(
                "set_seat_heating",
                {
                    "level": {"type": "integer", "enum": [2]},
                    "seat_zone": {
                        "type": "string",
                        "enum": ["PASSENGER"],
                    },
                },
                ["level", "seat_zone"],
            ),
            STEERING_SET,
        ),
        (
            CLIMATE_SET,
            SEAT_HEATING_READ,
            SEAT_HEATING,
            tool(
                "set_steering_wheel_heating",
                {
                    "level": {"type": "integer", "minimum": 0, "maximum": 3},
                    "mode": {"type": "string"},
                },
                ["level", "mode"],
            ),
        ),
    ),
)
def test_dependent_seat_sync_atomic_gate_rejects_any_incomplete_dependency(
    controls: tuple[dict[str, Any], ...],
) -> None:
    client = CountingFailureClient()
    agent = runtime(client)
    blocked = agent.handle_event(
        user_event(
            "dependent-seat-sync-atomic",
            "dependent-seat-sync-atomic-user",
            DEPENDENT_PASSENGER_WARMING_REQUEST,
            controls,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "did not change the passenger temperature" in blocked.text.casefold()
    assert client.calls == 0
    session = agent.sessions.get("dependent-seat-sync-atomic")
    assert session is not None
    assert session.budget.attempted_sets == set()


def test_dependent_seat_sync_falls_through_when_target_schema_is_complete() -> None:
    client = CountingFailureClient()
    agent = runtime(client)
    response = agent.handle_event(
        user_event(
            "dependent-seat-sync-complete",
            "dependent-seat-sync-complete-user",
            DEPENDENT_PASSENGER_WARMING_REQUEST,
            (CLIMATE_SET, SEAT_HEATING_READ, SEAT_HEATING, STEERING_SET),
        )
    )

    assert response.tool_calls == ()
    assert response.text is not None
    assert "cannot safely match the passenger seat heating" not in response.text.casefold()
    assert client.calls == 1


@pytest.mark.parametrize(
    ("case", "superseding_request", "controls", "limitation_text"),
    (
        (
            "fog",
            "Check the current weather and then turn on the fog lights.",
            (CLIMATE_SET,),
            "current location and time",
        ),
        (
            "dependent-seat",
            DEPENDENT_PASSENGER_WARMING_REQUEST,
            (CLIMATE_SET, SEAT_HEATING_READ, SEAT_HEATING),
            "cannot safely match the passenger seat heating",
        ),
        (
            "occupancy",
            PUBLIC_SHAPED_COMPOUND_INFORMATION_REQUEST,
            (CLIMATE_SET, CLIMATE),
            "seat occupancy information is unavailable",
        ),
    ),
)
def test_recognized_limitation_supersedes_stale_temperature_clarification(
    case: str,
    superseding_request: str,
    controls: tuple[dict[str, Any], ...],
    limitation_text: str,
) -> None:
    client = TemperatureClarificationThenFailureClient()
    agent = runtime(client)
    context_id = f"supersede-stale-temperature-{case}"
    first = agent.handle_event(
        user_event(
            context_id,
            f"{context_id}-temperature",
            "Set the climate temperature to 22 degrees.",
            controls,
        )
    )
    assert first.tool_calls == ()
    assert first.text is not None and "Which option" in first.text
    session = agent.sessions.get(context_id)
    assert session is not None and session.intent is not None
    old_goal_ids = {goal.goal_id for goal in session.intent.goals}
    assert session.pending_clarifications

    blocked = agent.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-replacement",
            user_text=superseding_request,
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert limitation_text in blocked.text.casefold()
    assert session.intent is None
    assert session.pending_clarifications == {}
    assert session.authorized_action_goal_ids == set()
    assert session.grounded_desired_values_by_goal == {}
    assert session.grounded_value_sources_by_goal == {}
    assert session.derived_value_evidence_by_goal == {}
    assert list(session.goal_dag.goals) == []
    assert all(
        need.required_for_goal_id not in old_goal_ids
        for need in session.evidence.needs.values()
    )

    stale_reply = agent.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-stale-reply",
            user_text="Passenger.",
        )
    )
    assert stale_reply.tool_calls == ()
    assert session.budget.attempted_sets == set()


def test_dependent_seat_sync_rejects_a_live_domain_without_the_requested_target() -> None:
    driver_only = tool(
        "set_seat_heating",
        {
            "level": {"type": "integer", "minimum": 0, "maximum": 3},
            "seat_zone": {"type": "string", "enum": ["DRIVER"]},
        },
        ["level", "seat_zone"],
    )
    client = CountingFailureClient()
    agent = runtime(client)
    blocked = agent.handle_event(
        user_event(
            "dependent-seat-sync-domain",
            "dependent-seat-sync-domain-user",
            DEPENDENT_PASSENGER_WARMING_REQUEST,
            (CLIMATE_SET, SEAT_HEATING_READ, driver_only, STEERING_SET),
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "cannot target the passenger seat" in blocked.text.casefold()
    assert client.calls == 0


@pytest.mark.parametrize(
    ("request_text", "target", "source"),
    (
        (DEPENDENT_PASSENGER_WARMING_REQUEST, "PASSENGER", "DRIVER"),
        (
            "Could you set the driver zone temperature to 22.5 degrees Celsius. "
            "Please set the driver seat heating to sync with the passenger side "
            "seat heating, and set the steering wheel heating to match the same "
            "level as the driver seat heating.",
            "DRIVER",
            "PASSENGER",
        ),
    ),
)
def test_dependent_seat_sync_matcher_is_zone_and_temperature_parameterized(
    request_text: str, target: str, source: str
) -> None:
    request = _dependent_seat_heating_capability_request(request_text)
    assert request is not None
    assert request.target_zone == target
    assert request.source_zone == source


@pytest.mark.parametrize(
    "unsafe",
    (
        DEPENDENT_PASSENGER_WARMING_REQUEST + " Then call Morgan Lee.",
        DEPENDENT_PASSENGER_WARMING_REQUEST.replace(
            "Could you please", "If it gets colder, could you please"
        ),
        DEPENDENT_PASSENGER_WARMING_REQUEST.replace(
            "I'd like", "she said she'd like"
        ),
        f'He wrote "{DEPENDENT_PASSENGER_WARMING_REQUEST}"',
        DEPENDENT_PASSENGER_WARMING_REQUEST.replace(
            "passenger seat heating to match the driver's",
            "driver seat heating to match the driver's",
        ),
        (
            "Can you tell me whether the passenger seat heating matches the "
            "driver's seat heating level?"
        ),
    ),
)
def test_dependent_seat_sync_matcher_rejects_unsafe_or_conflicting_text(
    unsafe: str,
) -> None:
    assert _dependent_seat_heating_capability_request(unsafe) is None


@pytest.mark.parametrize(
    "unsafe",
    (
        "Do not turn off the passenger seat heating.",
        "She said to turn off the passenger seat heating.",
        "Hypothetically, turn off the passenger seat heating.",
        "If I asked you to turn off the passenger seat heating, what would happen?",
        'The example phrase is "turn off the passenger seat heating".',
    ),
)
def test_seat_heating_preflight_rejects_unsafe_context(unsafe: str) -> None:
    assert _seat_heating_capability_request(unsafe) is None


@pytest.mark.parametrize(
    "compound",
    (
        "Turn off the passenger seat heating, then call Morgan Lee.",
        "Turn off the passenger seat heating, then email Morgan and Taylor.",
        "Turn off the passenger seat heating and navigate home.",
        "Turn off the passenger seat heating, then text Morgan.",
        "Turn off the passenger seat heating and set the fan to level 2.",
        "Turn off the passenger seat heating, then open the driver window.",
        "Turn off passenger seat heating, then lock doors.",
        "Turn off passenger seat heating, then play music.",
        "Turn off passenger seat heating, then adjust steering heating.",
        MATCHED_HEATING_REQUEST,
    ),
)
def test_seat_heating_preflight_does_not_swallow_unrelated_actions(
    compound: str,
) -> None:
    assert _seat_heating_capability_request(compound) is None


@pytest.mark.parametrize(
    ("command", "target_zone"),
    (
        ("Turn off passenger seat heating.", "PASSENGER"),
        ("Can you please set the driver's seat heating to level 2?", "DRIVER"),
        ("Please increase the heated seats by one level.", None),
        ("I would like passenger seat heating turned off.", "PASSENGER"),
        ("Switch seat heating for the passenger seat on.", "PASSENGER"),
        (EMPTY_SEAT_REQUEST, None),
        (
            "Because no one else is in the vehicle, please turn off seat heating "
            "for all unoccupied seats to save energy.",
            None,
        ),
    ),
)
def test_seat_heating_preflight_accepts_one_complete_command(
    command: str, target_zone: str | None
) -> None:
    request = _seat_heating_capability_request(command)
    assert request is not None
    assert request.target_zone == target_zone


@pytest.mark.parametrize(
    "read_only",
    (
        "Can you tell me what level the passenger seat heating is at?",
        "I need to know whether the passenger seat heating is on.",
        "I need information about the current passenger seat heating level.",
        "Show me the passenger seat heating level.",
        "What level is displayed for the passenger seat heating?",
        "Could you check the passenger seat heating for me?",
        "I would like the current passenger seat heating level.",
    ),
)
def test_seat_heating_preflight_rejects_read_only_requests(read_only: str) -> None:
    assert _seat_heating_capability_request(read_only) is None


def test_read_only_seat_request_with_only_read_tool_bypasses_write_preflight() -> None:
    client = CountingFailureClient()
    agent = runtime(client)
    response = agent.handle_event(
        user_event(
            "seat-heating-read",
            "seat-heating-read-user",
            "Can you tell me what level the passenger seat heating is at?",
            (SEAT_HEATING_READ,),
        )
    )

    assert response.tool_calls == ()
    assert client.calls == 1
    assert response.text is not None
    assert "seat heating control is not currently available" not in response.text.casefold()


def test_direct_need_you_command_reaches_missing_write_preflight() -> None:
    client = CountingFailureClient()
    agent = runtime(client)
    blocked = agent.handle_event(
        user_event(
            "seat-heating-command",
            "seat-heating-command-user",
            "I need you to turn off the passenger seat heating.",
            (SEAT_HEATING_READ,),
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "seat heating control is not currently available" in blocked.text.casefold()
    assert client.calls == 0


def test_full_seat_heating_schema_falls_through_to_existing_intent_path() -> None:
    client = CountingFailureClient()
    agent = runtime(client)
    response = agent.handle_event(
        user_event(
            "full-seat-schema",
            "full-seat-schema-user",
            "Turn off the passenger seat heating.",
            (SEAT_HEATING,),
        )
    )

    assert response.tool_calls == ()
    assert client.calls == 1
    assert response.text is not None
    assert "seat heating control is not currently available" not in response.text.casefold()


def test_standalone_occupancy_read_is_exact_and_never_authorizes_a_write() -> None:
    agent = runtime(NoModelClient())
    first = agent.handle_event(
        user_event(
            "seat-occupancy",
            "seat-occupancy-user",
            OCCUPANCY_REQUEST,
            (OCCUPANCY,),
        )
    )
    assert first.tool_calls == ({"tool_name": "get_seats_occupancy", "arguments": {}},)

    answer = agent.handle_event(
        result_event(
            "seat-occupancy",
            "seat-occupancy-result",
            ("get_seats_occupancy", occupancy_result()),
        )
    )
    assert answer.tool_calls == ()
    assert answer.text is not None
    lowered = answer.text.casefold()
    assert "driver seat" in lowered and "occupied" in lowered
    assert "front passenger seat" in lowered and "unoccupied" in lowered
    session = agent.sessions.get("seat-occupancy")
    assert session is not None
    assert session.authorized_action_goal_ids == set()
    assert session.budget.attempted_sets == set()


def test_compound_climate_and_occupancy_read_uses_only_the_two_reads() -> None:
    agent = runtime(NoModelClient())
    first = agent.handle_event(
        user_event(
            "climate-occupancy",
            "climate-occupancy-user",
            COMPOUND_INFORMATION_REQUEST,
            (CLIMATE, OCCUPANCY),
        )
    )
    assert {call["tool_name"] for call in first.tool_calls} == {
        "get_climate_settings",
        "get_seats_occupancy",
    }

    answer = agent.handle_event(
        result_event(
            "climate-occupancy",
            "climate-occupancy-result",
            ("get_climate_settings", climate_result()),
            ("get_seats_occupancy", occupancy_result()),
        )
    )
    assert answer.tool_calls == ()
    assert answer.text is not None
    lowered = answer.text.casefold()
    assert "air conditioning off" in lowered
    assert "driver seat" in lowered
    session = agent.sessions.get("climate-occupancy")
    assert session is not None
    assert session.authorized_action_goal_ids == set()
    assert session.budget.attempted_sets == set()


def test_public_shaped_climate_and_occupancy_preface_stays_deterministic() -> None:
    request = _seat_occupancy_information_request(
        PUBLIC_SHAPED_COMPOUND_INFORMATION_REQUEST
    )
    assert request is not None and request.include_climate

    agent = runtime(NoModelClient())
    first = agent.handle_event(
        user_event(
            "public-shaped-climate-occupancy",
            "public-shaped-climate-occupancy-user",
            PUBLIC_SHAPED_COMPOUND_INFORMATION_REQUEST,
            (CLIMATE, OCCUPANCY),
        )
    )
    assert {call["tool_name"] for call in first.tool_calls} == {
        "get_climate_settings",
        "get_seats_occupancy",
    }


def test_occupancy_result_with_extra_seat_is_rejected_without_claiming_state() -> None:
    agent = runtime(NoModelClient())
    agent.handle_event(
        user_event(
            "bad-occupancy",
            "bad-occupancy-user",
            OCCUPANCY_REQUEST,
            (OCCUPANCY,),
        )
    )
    blocked = agent.handle_event(
        result_event(
            "bad-occupancy",
            "bad-occupancy-result",
            ("get_seats_occupancy", occupancy_result(extra=True)),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "driver seat is occupied" not in blocked.text.casefold()


def test_occupancy_answer_then_empty_seat_action_reaches_missing_control_preflight() -> None:
    agent = runtime(NoModelClient())
    agent.handle_event(
        user_event(
            "occupancy-followup",
            "occupancy-followup-user",
            OCCUPANCY_REQUEST,
            (OCCUPANCY,),
        )
    )
    agent.handle_event(
        result_event(
            "occupancy-followup",
            "occupancy-followup-result",
            ("get_seats_occupancy", occupancy_result()),
        )
    )
    blocked = agent.handle_event(
        InboundEvent(
            context_id="occupancy-followup",
            message_id="occupancy-followup-action",
            user_text=EMPTY_SEAT_REQUEST,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "seat heating control is not currently available" in blocked.text.casefold()


def test_occupancy_matcher_rejects_write_and_meta_requests() -> None:
    assert _seat_occupancy_information_request(OCCUPANCY_REQUEST) is not None
    assert _seat_occupancy_information_request(COMPOUND_INFORMATION_REQUEST) is not None
    assert (
        _seat_occupancy_information_request(
            PUBLIC_SHAPED_COMPOUND_INFORMATION_REQUEST + " Then call Morgan Lee."
        )
        is None
    )
    assert (
        _seat_occupancy_information_request(
            "Tell me who is in the car, then turn off the passenger seat heating."
        )
        is None
    )
    assert (
        _seat_occupancy_information_request(
            "Hypothetically, who is in the car right now?"
        )
        is None
    )


@pytest.mark.parametrize(
    "declarative",
    (
        "There are passengers in the car.",
        "There are any passengers in the car right now.",
    ),
)
def test_occupancy_matcher_rejects_declarative_passenger_state(
    declarative: str,
) -> None:
    assert _seat_occupancy_information_request(declarative) is None


@pytest.mark.parametrize(
    ("request_text", "include_climate"),
    (
        ("Tell me who is in the car.", False),
        ("Who is inside the vehicle right now?", False),
        ("Which seats are currently occupied?", False),
        ("Are there passengers in the car?", False),
        ("Can you check if there are any passengers in the cabin?", False),
        ("Please show me the current seat occupancy.", False),
        (COMPOUND_INFORMATION_REQUEST, True),
        (
            "Could you tell me who is in the car and what the current climate "
            "settings are?",
            True,
        ),
    ),
)
def test_occupancy_matcher_accepts_only_bounded_read_compositions(
    request_text: str, include_climate: bool
) -> None:
    request = _seat_occupancy_information_request(request_text)
    assert request is not None
    assert request.include_climate is include_climate


@pytest.mark.parametrize(
    "compound",
    (
        "Tell me who is in the car, then call Morgan Lee.",
        "Tell me who is in the car, then email Morgan and Taylor.",
        "Tell me who is in the car and navigate home.",
        "Tell me who is in the car, then text Morgan.",
        "Tell me who is in the car and set the temperature to 22 degrees.",
        "Tell me which seats are occupied, then set the fan to level 2.",
        "Tell me who is in car, then lock doors.",
        "Tell me who is in car, then play music.",
        "Tell me who is in car, then unlock doors.",
        "Tell me who is in car, then check tire pressure.",
    ),
)
def test_occupancy_matcher_does_not_swallow_compound_actions(compound: str) -> None:
    assert _seat_occupancy_information_request(compound) is None


def test_current_weather_fog_lights_uses_policy_read_and_stops_on_missing_condition() -> None:
    weather = tool(
        "get_weather",
        {
            "location_or_poi_id": {"type": "string"},
            "month": {"type": "integer"},
            "day": {"type": "integer"},
            "time_hour_24hformat": {"type": "integer"},
            "time_minutes": {"type": "integer"},
        },
        ["location_or_poi_id", "month", "day", "time_hour_24hformat"],
    )
    fog_control = tool(
        "set_fog_lights", {"on": {"type": "boolean"}}, ["on"]
    )
    policy = (
        'CURRENT_LOCATION = {"id":"loc_demo","name":"Demo City"}\n'
        'DATETIME = {"year":2025,"month":1,"day":10,"hour":19,"minute":30}\n'
        "AUT-POL:009: Weather must be checked manually before the action is "
        "performed.\n"
        "AUT-POL:013: When activating fog lights, check low beam and high beam."
    )
    agent = runtime(NoModelClient())
    read = agent.handle_event(
        InboundEvent(
            context_id="current-weather-fog-lights",
            message_id="current-weather-fog-lights-user",
            system_policy=policy,
            user_text=(
                "Hey there! It's getting a bit hard to see out here in Demo City. "
                "Could you check the weather for me and then turn on the fog lights?"
            ),
            live_tools=(weather, fog_control),
        )
    )
    assert read.tool_calls == (
        {
            "tool_name": "get_weather",
            "arguments": {
                "location_or_poi_id": "loc_demo",
                "month": 1,
                "day": 10,
                "time_hour_24hformat": 19,
                "time_minutes": 30,
            },
        },
    )

    blocked = agent.handle_event(
        result_event(
            "current-weather-fog-lights",
            "current-weather-fog-lights-result",
            (
                "get_weather",
                {
                    "status": "SUCCESS",
                    "result": {
                        "current_slot": {"condition": "unknown"},
                        "next_slot": None,
                    },
                },
            ),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    lowered = blocked.text.casefold()
    assert "current weather condition information" in lowered
    assert "did not change the fog lights" in lowered
    session = agent.sessions.get("current-weather-fog-lights")
    assert session is not None
    assert session.budget.attempted_sets == set()


def test_current_weather_fog_lights_never_reuses_prior_same_state_weather() -> None:
    weather = tool(
        "get_weather",
        {
            "location_or_poi_id": {"type": "string"},
            "month": {"type": "integer"},
            "day": {"type": "integer"},
            "time_hour_24hformat": {"type": "integer"},
            "time_minutes": {"type": "integer"},
        },
        ["location_or_poi_id", "month", "day", "time_hour_24hformat"],
    )
    fog_control = tool(
        "set_fog_lights", {"on": {"type": "boolean"}}, ["on"]
    )
    policy = (
        'CURRENT_LOCATION = {"id":"loc_demo","name":"Demo City"}\n'
        'DATETIME = {"year":2025,"month":1,"day":10,"hour":19,"minute":30}\n'
        "AUT-POL:009: Weather must be checked manually before the action is "
        "performed.\n"
        "AUT-POL:013: When activating fog lights, check low beam and high beam."
    )
    request = "Check the current weather and then turn on the fog lights."
    agent = runtime(NoModelClient())
    first = agent.handle_event(
        InboundEvent(
            context_id="fresh-weather-repeat",
            message_id="fresh-weather-repeat-first",
            system_policy=policy,
            user_text=request,
            live_tools=(weather, fog_control),
        )
    )
    assert first.tool_calls[0]["tool_name"] == "get_weather"
    agent.handle_event(
        result_event(
            "fresh-weather-repeat",
            "fresh-weather-repeat-first-result",
            (
                "get_weather",
                {
                    "status": "SUCCESS",
                    "result": {
                        "current_slot": {"condition": "rainy"},
                        "next_slot": None,
                    },
                },
            ),
        )
    )

    repeated = agent.handle_event(
        InboundEvent(
            context_id="fresh-weather-repeat",
            message_id="fresh-weather-repeat-second",
            user_text=request,
        )
    )
    assert repeated.tool_calls == first.tool_calls
    session = agent.sessions.get("fresh-weather-repeat")
    assert session is not None
    assert session.budget.attempted_sets == set()


@pytest.mark.parametrize(
    "unsafe",
    (
        "Check the weather and turn off the fog lights.",
        "If the weather is bad, turn on the fog lights.",
        "She said to check the weather and turn on the fog lights.",
        'The example phrase is "check the weather and turn on the fog lights".',
        "Check the weather and turn on the fog lights, then call Morgan.",
        "Check the weather tomorrow and turn on the fog lights.",
    ),
)
def test_current_weather_fog_lights_matcher_rejects_unsafe_compositions(
    unsafe: str,
) -> None:
    assert _current_weather_fog_lights_request(unsafe) is None


@pytest.mark.parametrize(
    ("request_text", "location"),
    (
        (
            "Hey there! It's getting a bit hard to see out here in Demo City. "
            "Could you check the weather for me and then turn on the fog lights?",
            "Demo City",
        ),
        (
            "Please check the current forecast at my current location and enable "
            "the fog lights.",
            None,
        ),
        (
            "Hey, can you turn on the fog lights for me? It's becoming difficult "
            "to see outside.",
            None,
        ),
    ),
)
def test_current_weather_fog_lights_matcher_accepts_only_bounded_shapes(
    request_text: str, location: str | None
) -> None:
    request = _current_weather_fog_lights_request(request_text)
    assert request is not None
    assert request.location_name == location


def test_current_weather_fog_lights_does_not_claim_a_different_location() -> None:
    client = CountingFailureClient()
    agent = runtime(client)
    response = agent.handle_event(
        InboundEvent(
            context_id="different-weather-location",
            message_id="different-weather-location-user",
            system_policy=(
                'CURRENT_LOCATION = {"id":"loc_demo","name":"Demo City"}\n'
                'DATETIME = {"year":2025,"month":1,"day":10,"hour":19,"minute":30}'
            ),
            user_text=(
                "It's getting hard to see out here in Other City. Check the "
                "weather and then turn on the fog lights."
            ),
            live_tools=(),
        )
    )

    assert response.tool_calls == ()
    assert client.calls == 1
    assert response.text is not None
    assert "current location and time" not in response.text.casefold()


def test_missing_weather_condition_regression_remains_transparent() -> None:
    weather = tool(
        "get_weather",
        {
            "location_or_poi_id": {"type": "string"},
            "month": {"type": "integer"},
            "day": {"type": "integer"},
            "time_hour_24hformat": {"type": "integer"},
            "time_minutes": {"type": "integer"},
        },
        ["location_or_poi_id", "month", "day", "time_hour_24hformat"],
    )
    fog_control = tool(
        "set_fog_lights", {"on": {"type": "boolean"}}, ["on"]
    )
    policy = (
        'CURRENT_LOCATION = {"id":"loc_demo","name":"Demo City"}\n'
        'DATETIME = {"year":2025,"month":1,"day":10,"hour":19,"minute":30}\n'
        "AUT-POL:009: Weather must be checked manually before the action is "
        "performed.\n"
        "AUT-POL:013: When activating fog lights, check low beam and high beam."
    )
    agent = runtime(FogIntentClient())
    read = agent.handle_event(
        InboundEvent(
            context_id="missing-weather-field",
            message_id="missing-weather-field-user",
            system_policy=policy,
            user_text="Turn on the fog lights.",
            live_tools=(weather, fog_control),
        )
    )
    assert read.tool_calls == (
        {
            "tool_name": "get_weather",
            "arguments": {
                "location_or_poi_id": "loc_demo",
                "month": 1,
                "day": 10,
                "time_hour_24hformat": 19,
                "time_minutes": 30,
            },
        },
    )

    blocked = agent.handle_event(
        result_event(
            "missing-weather-field",
            "missing-weather-field-result",
            (
                "get_weather",
                {
                    "status": "SUCCESS",
                    "result": {
                        "current_slot": {"condition": "unknown"},
                        "next_slot": None,
                    },
                },
            ),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    lowered = blocked.text.casefold()
    assert "current weather condition information" in lowered
    assert "did not change the fog lights" in lowered
