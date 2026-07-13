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
    ResultIssue,
    _absolute_climate_temperature_intent,
    _absolute_fan_speed_intent,
    _missing_capability_followup,
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


UNRELATED_READ = tool("get_climate_settings")
TEMPERATURE_SET = tool(
    "set_climate_temperature",
    {
        "temperature": {"type": "number", "minimum": 16, "maximum": 28},
        "seat_zone": {
            "type": "string",
            "enum": ["ALL_ZONES", "DRIVER", "PASSENGER"],
        },
    },
    ["temperature", "seat_zone"],
)
FAN_SET = tool(
    "set_fan_speed",
    {"level": {"type": "integer", "minimum": 0, "maximum": 5}},
    ["level"],
)
FAN_SET_WITH_WRONG_PARAMETER = tool(
    "set_fan_speed",
    {"speed": {"type": "integer", "minimum": 0, "maximum": 5}},
    ["speed"],
)
AIR_CONDITIONING_SET = tool(
    "set_air_conditioning",
    {"on": {"type": "boolean"}},
    ["on"],
)
SEAT_HEATING_WITH_WRONG_PARAMETERS = tool(
    "set_seat_heating",
    {
        "heat": {"type": "integer", "minimum": 0, "maximum": 3},
        "seat": {"type": "string", "enum": ["DRIVER", "PASSENGER"]},
    },
    ["heat", "seat"],
)
SEAT_HEATING_WITH_INVALID_LEVEL_RANGE = tool(
    "set_seat_heating",
    {
        "level": {"type": "integer", "minimum": 0, "maximum": 1},
        "seat_zone": {"type": "string", "enum": ["DRIVER", "PASSENGER"]},
    },
    ["level", "seat_zone"],
)
SEAT_HEATING_WITH_INCOMPATIBLE_ZONE = tool(
    "set_seat_heating",
    {
        "level": {"type": "integer", "minimum": 0, "maximum": 3},
        "seat_zone": {"type": "string", "enum": ["PASSENGER"]},
    },
    ["level", "seat_zone"],
)
WINDOW_POSITIONS = tool("get_vehicle_window_positions")
WINDOW_SET = tool(
    "open_close_window",
    {
        "window": {
            "type": "string",
            "enum": ["DRIVER", "PASSENGER", "DRIVER_REAR", "PASSENGER_REAR"],
        },
        "percentage": {"type": "number", "minimum": 0, "maximum": 100},
    },
    ["window", "percentage"],
)
CHARGING_STATUS = tool("get_charging_specs_and_status")
READING_SET = tool(
    "set_reading_light",
    {
        "position": {
            "type": "string",
            "enum": [
                "ALL",
                "DRIVER",
                "PASSENGER",
                "DRIVER_REAR",
                "PASSENGER_REAR",
            ],
        },
        "on": {"type": "boolean"},
    },
    ["position", "on"],
)
READING_STATUS = tool("get_reading_lights_status")
OCCUPANCY = tool("get_seats_occupancy")

TEMPERATURE_REQUEST = (
    "Could you please set the temperature to 22 degrees Celsius for all zones?"
)
FAN_REQUEST = (
    "Hey there! It's a bit stuffy in here. Could you turn on the fan for me and "
    "set it to level 3?"
)
CHARGING_REQUEST = (
    "Could you tell me about my car's battery and charging status, including the "
    "current charge level, battery capacity, AC and DC charging power capabilities, "
    "and remaining driving range?"
)
READING_REQUEST = (
    "Can you adjust the reading lights to only illuminate occupied seats and turn "
    "them off for any unoccupied seats?"
)


class NoModelClient:
    def generate(self, **kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("strict Hall workflows must not require a model")


class PlannerFailureClient:
    def generate(self, **kwargs: Any) -> Any:
        del kwargs
        raise LLMFailure("synthetic planner unavailable")


class FixedIntentClient:
    def __init__(self, goals: list[dict[str, Any]], explicit_slots: dict[str, Any]):
        self.intent = IntentDraft.model_validate(
            {
                "language": "en",
                "intent_kind": "action",
                "call_for_action": True,
                "goals": goals,
                "explicit_slots": explicit_slots,
            }
        )

    def generate(self, *, response_model: Any, **kwargs: Any) -> Any:
        del kwargs
        if response_model is IntentDraft:
            return SimpleNamespace(value=self.intent)
        raise LLMFailure("synthetic planner unavailable")


class CapturingOrchestrator(CARGuardOrchestrator):
    captured_result_issue: ResultIssue | None = None

    def _result_issue_limitation_plan(self, session: Any, issue: ResultIssue) -> Any:
        self.captured_result_issue = issue
        return super()._result_issue_limitation_plan(session, issue)


def runtime(
    client: Any | None = None, *, capture_result_issue: bool = False
) -> CARGuardOrchestrator:
    orchestrator_type = CapturingOrchestrator if capture_result_issue else CARGuardOrchestrator
    selected_client = client or NoModelClient()
    return orchestrator_type(
        AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=24),
        client_factory=lambda session: selected_client,
    )


def user_event(
    context_id: str,
    message_id: str,
    text: str,
    tools: tuple[dict[str, Any], ...] = (),
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        system_policy="Follow the current safety policy."
        if message_id.endswith("-1")
        else None,
        user_text=text,
        live_tools=tools,
    )


def result_event(
    context_id: str,
    message_id: str,
    *results: tuple[str, Any],
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        tool_results=tuple(
            {"toolName": tool_name, "content": content}
            for tool_name, content in results
        ),
    )


def valid_reading_status() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "reading_light_driver": True,
            "reading_light_passenger": True,
            "reading_light_driver_rear": False,
            "reading_light_passenger_rear": False,
        },
    }


def assert_human_limitation(
    outbound: Any, *, phrase: str, forbidden_tool_name: str
) -> None:
    assert outbound.tool_calls == ()
    assert outbound.text is not None
    lowered = outbound.text.casefold()
    assert phrase in lowered
    assert "did not" in lowered
    assert forbidden_tool_name not in outbound.text


@pytest.mark.parametrize(
    ("text", "matcher", "operation", "desired"),
    [
        (
            TEMPERATURE_REQUEST,
            _absolute_climate_temperature_intent,
            "set_climate_temperature",
            {"temperature": 22, "seat_zone": "ALL_ZONES"},
        ),
        (
            FAN_REQUEST,
            _absolute_fan_speed_intent,
            "set_fan_speed",
            {"level": 3},
        ),
    ],
)
def test_absolute_control_matchers_recover_explicit_semantic_goal(
    text: str, matcher: Any, operation: str, desired: dict[str, Any]
) -> None:
    intent = matcher(text, turn_id="turn-1")
    assert intent is not None
    assert intent.call_for_action
    assert len(intent.goals) == 1
    assert intent.goals[0].semantic_operation == operation
    assert intent.goals[0].desired_outcome == desired


@pytest.mark.parametrize(
    ("matcher", "text"),
    [
        (_absolute_climate_temperature_intent, "Set the temperature to 22."),
        (_absolute_fan_speed_intent, "Could you turn on the fan?"),
        (
            _absolute_climate_temperature_intent,
            "Do not set the temperature to 22 degrees Celsius.",
        ),
        (
            _absolute_fan_speed_intent,
            "She said to set the fan to level 3.",
        ),
        (
            _absolute_climate_temperature_intent,
            "If it gets cold, set the temperature to 22 degrees Celsius.",
        ),
        (
            _absolute_fan_speed_intent,
            "Maybe set the fan to level 3.",
        ),
        (
            _absolute_climate_temperature_intent,
            "Set the temperature to 22 degrees Celsius, then open the window.",
        ),
        (
            _absolute_fan_speed_intent,
            "Set the fan to level 3 and turn on the reading lights.",
        ),
    ],
)
def test_absolute_control_matchers_reject_ambiguous_or_noncurrent_language(
    matcher: Any, text: str
) -> None:
    assert matcher(text, turn_id="turn-1") is None


def test_missing_temperature_control_preserves_goal_and_never_sets() -> None:
    agent = runtime()
    context_id = "hall-temperature-missing"
    blocked = agent.handle_event(
        user_event(
            context_id, f"{context_id}-1", TEMPERATURE_REQUEST, (UNRELATED_READ,)
        )
    )
    assert_human_limitation(
        blocked,
        phrase="cabin temperature control",
        forbidden_tool_name="set_climate_temperature",
    )
    session = agent.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert [
        (goal.semantic_operation, goal.desired_outcome) for goal in session.intent.goals
    ] == [("set_climate_temperature", {"temperature": 22, "seat_zone": "ALL_ZONES"})]


def test_missing_fan_control_preserves_goal_and_never_sets() -> None:
    agent = runtime()
    context_id = "hall-fan-missing"
    blocked = agent.handle_event(
        user_event(context_id, f"{context_id}-1", FAN_REQUEST, (UNRELATED_READ,))
    )
    assert_human_limitation(
        blocked,
        phrase="fan speed control",
        forbidden_tool_name="set_fan_speed",
    )
    session = agent.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert [
        (goal.semantic_operation, goal.desired_outcome) for goal in session.intent.goals
    ] == [("set_fan_speed", {"level": 3})]


def test_missing_charging_information_preserves_goal_and_never_substitutes() -> None:
    agent = runtime()
    context_id = "hall-charging-missing"
    blocked = agent.handle_event(
        user_event(context_id, f"{context_id}-1", CHARGING_REQUEST, (UNRELATED_READ,))
    )
    assert_human_limitation(
        blocked,
        phrase="battery and charging status information",
        forbidden_tool_name="get_charging_specs_and_status",
    )
    session = agent.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert [goal.semantic_operation for goal in session.intent.goals] == [
        "read_charging_status"
    ]


@pytest.mark.parametrize(
    ("tools", "phrase", "forbidden"),
    [
        (
            (READING_SET, READING_STATUS),
            "seat occupancy information",
            "get_seats_occupancy",
        ),
        (
            (READING_SET, OCCUPANCY),
            "reading light status information",
            "get_reading_lights_status",
        ),
        ((READING_STATUS, OCCUPANCY), "reading light control", "set_reading_light"),
        (
            (
                READING_SET,
                READING_STATUS,
                tool("get_seats_occupancy", {"scope": {"type": "string"}}, ["scope"]),
            ),
            "seat occupancy information",
            "get_seats_occupancy",
        ),
        (
            (
                READING_SET,
                OCCUPANCY,
                tool(
                    "get_reading_lights_status",
                    {"scope": {"type": "string"}},
                    ["scope"],
                ),
            ),
            "reading light status information",
            "get_reading_lights_status",
        ),
        (
            (
                tool(
                    "set_reading_light",
                    {"on": {"type": "boolean"}},
                    ["on"],
                ),
                READING_STATUS,
                OCCUPANCY,
            ),
            "reading light control",
            "set_reading_light",
        ),
    ],
)
def test_occupancy_reading_lights_preflight_names_each_missing_dependency(
    tools: tuple[dict[str, Any], ...], phrase: str, forbidden: str
) -> None:
    agent = runtime()
    context_id = f"hall-reading-{abs(hash(phrase + forbidden + str(len(tools))))}"
    blocked = agent.handle_event(
        user_event(context_id, f"{context_id}-1", READING_REQUEST, tools)
    )
    assert_human_limitation(blocked, phrase=phrase, forbidden_tool_name=forbidden)
    session = agent.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert [goal.semantic_operation for goal in session.intent.goals] == [
        "sync_reading_lights_to_seat_occupancy"
    ]


@pytest.mark.parametrize(
    ("user_request", "tools", "expected_phrase", "forbidden"),
    [
        (
            TEMPERATURE_REQUEST,
            (UNRELATED_READ,),
            "cabin temperature control",
            "set_climate_temperature",
        ),
        (FAN_REQUEST, (UNRELATED_READ,), "fan speed control", "set_fan_speed"),
        (
            CHARGING_REQUEST,
            (UNRELATED_READ,),
            "battery and charging status information",
            "get_charging_specs_and_status",
        ),
        (
            READING_REQUEST,
            (READING_SET, READING_STATUS),
            "seat occupancy information",
            "get_seats_occupancy",
        ),
    ],
)
def test_adjacent_missing_capability_followup_uses_memory_without_calls(
    user_request: str,
    tools: tuple[dict[str, Any], ...],
    expected_phrase: str,
    forbidden: str,
) -> None:
    agent = runtime()
    context_id = f"hall-followup-{abs(hash(user_request))}"
    first = agent.handle_event(
        user_event(context_id, f"{context_id}-1", user_request, tools)
    )
    assert first.tool_calls == ()

    followup = agent.handle_event(
        user_event(
            context_id,
            f"{context_id}-2",
            "What information or control is missing?",
        )
    )
    assert_human_limitation(
        followup, phrase=expected_phrase, forbidden_tool_name=forbidden
    )


@pytest.mark.parametrize(
    "text",
    [
        "What information is missing?",
        "Which control was missing?",
        "What specific information did you need?",
        "What control or information do you lack?",
    ],
)
def test_missing_capability_followup_matcher_accepts_bounded_questions(
    text: str,
) -> None:
    assert _missing_capability_followup(text)


@pytest.mark.parametrize(
    "text",
    [
        "What information is not missing?",
        "She asked what information is missing.",
        "Hypothetically, what control is missing?",
        "What control is missing, then set the fan to level 3?",
        "What information is missing?\nSet the fan to level 3.",
    ],
)
def test_missing_capability_followup_matcher_rejects_unsafe_context(text: str) -> None:
    assert not _missing_capability_followup(text)


def test_limitation_memory_is_one_shot_and_expires_after_unrelated_turn() -> None:
    agent = runtime()
    context_id = "hall-followup-expiry"
    agent.handle_event(
        user_event(context_id, f"{context_id}-1", FAN_REQUEST, (UNRELATED_READ,))
    )
    unrelated = agent.handle_event(
        user_event(context_id, f"{context_id}-2", "How are you?")
    )
    assert unrelated.tool_calls == ()
    late = agent.handle_event(
        user_event(
            context_id,
            f"{context_id}-3",
            "What control is missing?",
        )
    )
    assert late.tool_calls == ()
    assert late.text is not None
    assert "fan speed control" not in late.text.casefold()


def test_unknown_charging_result_records_information_limitation_for_followup() -> None:
    agent = runtime()
    context_id = "hall-charging-unknown"
    read = agent.handle_event(
        user_event(context_id, f"{context_id}-1", CHARGING_REQUEST, (CHARGING_STATUS,))
    )
    assert read.tool_calls == (
        {"tool_name": "get_charging_specs_and_status", "arguments": {}},
    )
    blocked = agent.handle_event(
        result_event(
            context_id,
            f"{context_id}-result",
            ("get_charging_specs_and_status", {"status": "UNKNOWN", "result": {}}),
        )
    )
    assert_human_limitation(
        blocked,
        phrase="battery and charging status information",
        forbidden_tool_name="get_charging_specs_and_status",
    )
    followup = agent.handle_event(
        user_event(context_id, f"{context_id}-2", "What information is missing?")
    )
    assert_human_limitation(
        followup,
        phrase="battery and charging status information",
        forbidden_tool_name="get_charging_specs_and_status",
    )


def test_unknown_occupancy_result_never_reaches_partial_set_and_is_explainable() -> (
    None
):
    agent = runtime()
    context_id = "hall-occupancy-unknown"
    reads = agent.handle_event(
        user_event(
            context_id,
            f"{context_id}-1",
            READING_REQUEST,
            (READING_SET, READING_STATUS, OCCUPANCY),
        )
    )
    assert reads.tool_calls == (
        {"tool_name": "get_seats_occupancy", "arguments": {}},
        {"tool_name": "get_reading_lights_status", "arguments": {}},
    )
    blocked = agent.handle_event(
        result_event(
            context_id,
            f"{context_id}-result",
            ("get_seats_occupancy", {"status": "UNKNOWN", "result": {}}),
            ("get_reading_lights_status", valid_reading_status()),
        )
    )
    assert_human_limitation(
        blocked,
        phrase="seat occupancy information",
        forbidden_tool_name="get_seats_occupancy",
    )
    followup = agent.handle_event(
        user_event(context_id, f"{context_id}-2", "What information is missing?")
    )
    assert_human_limitation(
        followup,
        phrase="seat occupancy information",
        forbidden_tool_name="get_seats_occupancy",
    )


def test_unknown_reading_light_status_never_reaches_partial_set() -> None:
    agent = runtime()
    context_id = "hall-reading-status-unknown"
    reads = agent.handle_event(
        user_event(
            context_id,
            f"{context_id}-1",
            READING_REQUEST,
            (READING_SET, READING_STATUS, OCCUPANCY),
        )
    )
    assert len(reads.tool_calls) == 2
    blocked = agent.handle_event(
        result_event(
            context_id,
            f"{context_id}-result",
            (
                "get_seats_occupancy",
                {
                    "status": "SUCCESS",
                    "result": {
                        "seats_occupied": {
                            "driver": True,
                            "passenger": False,
                            "driver_rear": False,
                            "passenger_rear": True,
                        }
                    },
                },
            ),
            ("get_reading_lights_status", {"status": "UNKNOWN", "result": {}}),
        )
    )
    assert_human_limitation(
        blocked,
        phrase="reading light status information",
        forbidden_tool_name="set_reading_light",
    )


def test_partial_occupancy_snapshot_never_reaches_set_or_loses_context() -> None:
    agent = runtime()
    context_id = "hall-occupancy-partial"
    agent.handle_event(
        user_event(
            context_id,
            f"{context_id}-1",
            READING_REQUEST,
            (READING_SET, READING_STATUS, OCCUPANCY),
        )
    )
    blocked = agent.handle_event(
        result_event(
            context_id,
            f"{context_id}-result",
            (
                "get_seats_occupancy",
                {
                    "status": "SUCCESS",
                    "result": {"seats_occupied": {"driver": True}},
                },
            ),
            ("get_reading_lights_status", valid_reading_status()),
        )
    )
    assert_human_limitation(
        blocked,
        phrase="seat occupancy information",
        forbidden_tool_name="set_reading_light",
    )
    followup = agent.handle_event(
        user_event(context_id, f"{context_id}-2", "Which information was missing?")
    )
    assert_human_limitation(
        followup,
        phrase="seat occupancy information",
        forbidden_tool_name="set_reading_light",
    )


def _air_conditioning_client() -> FixedIntentClient:
    return FixedIntentClient(
        [
            {
                "semantic_operation": "set_air_conditioning",
                "desired_outcome": {"enabled": True},
            }
        ],
        {"enabled": True},
    )


def _air_conditioning_tools() -> tuple[dict[str, Any], ...]:
    return (
        UNRELATED_READ,
        WINDOW_POSITIONS,
        WINDOW_SET,
        FAN_SET,
        AIR_CONDITIONING_SET,
    )


def test_available_fan_control_keeps_existing_base_action_behavior() -> None:
    agent = runtime(PlannerFailureClient())
    context_id = "hall-fan-green-control"
    outbound = agent.handle_event(
        user_event(context_id, f"{context_id}-1", FAN_REQUEST, (FAN_SET,))
    )

    assert outbound.text is None
    assert outbound.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 3}},
    )


def test_irrelevant_tool_presence_and_order_do_not_change_missing_control() -> None:
    irrelevant = tool("get_current_navigation_state")
    variants = (
        (UNRELATED_READ,),
        (irrelevant, UNRELATED_READ),
        (UNRELATED_READ, irrelevant),
    )
    responses: list[str] = []
    for index, tools in enumerate(variants):
        agent = runtime()
        context_id = f"hall-missing-control-invariance-{index}"
        outbound = agent.handle_event(
            user_event(context_id, f"{context_id}-1", FAN_REQUEST, tools)
        )
        assert_human_limitation(
            outbound,
            phrase="fan speed control",
            forbidden_tool_name="set_fan_speed",
        )
        assert outbound.text is not None
        responses.append(outbound.text)

    assert len(set(responses)) == 1


def test_recipe_mapped_parameter_limitation_is_order_invariant() -> None:
    irrelevant = tool("get_current_navigation_state")
    variants = (
        (FAN_SET_WITH_WRONG_PARAMETER,),
        (irrelevant, FAN_SET_WITH_WRONG_PARAMETER),
        (FAN_SET_WITH_WRONG_PARAMETER, irrelevant),
    )
    responses: list[str] = []
    for index, tools in enumerate(variants):
        agent = runtime()
        context_id = f"hall-missing-parameter-invariance-{index}"
        outbound = agent.handle_event(
            user_event(context_id, f"{context_id}-1", FAN_REQUEST, tools)
        )
        assert outbound.tool_calls == ()
        assert outbound.text is not None
        lowered = outbound.text.casefold()
        assert "fan speed control" in lowered
        assert "level setting" in lowered
        assert "set_fan_speed" not in outbound.text
        responses.append(outbound.text)

    assert len(set(responses)) == 1


@pytest.mark.parametrize(
    ("content", "expected_code"),
    [
        ({"status": "SUCCESS"}, "extractor_path_missing"),
        (
            {"status": "SUCCESS", "result": {"fan_speed": "UNKNOWN"}},
            "insufficient_value",
        ),
    ],
)
def test_matched_declared_result_field_limitation_preserves_paired_issue(
    content: dict[str, Any], expected_code: str
) -> None:
    agent = runtime(_air_conditioning_client(), capture_result_issue=True)
    context_id = f"hall-declared-field-{expected_code}"
    read = agent.handle_event(
        user_event(
            context_id,
            f"{context_id}-1",
            "Turn on the air conditioning.",
            _air_conditioning_tools(),
        )
    )
    assert read.tool_calls == (
        {"tool_name": "get_climate_settings", "arguments": {}},
    )

    blocked = agent.handle_event(
        result_event(
            context_id,
            f"{context_id}-result",
            ("get_climate_settings", content),
        )
    )
    assert_human_limitation(
        blocked,
        phrase="current fan speed information",
        forbidden_tool_name="set_air_conditioning",
    )
    assert isinstance(agent, CapturingOrchestrator)
    issue = agent.captured_result_issue
    assert issue is not None
    assert [(item.need_id, item.code) for item in issue.validation_issues] == [
        (issue.validation_issues[0].need_id, expected_code)
    ]
    session = agent.sessions.get(context_id)
    assert session is not None
    need = session.evidence.needs[issue.validation_issues[0].need_id]
    assert need.proposition == "fan_speed"


def test_internal_extractor_error_uses_generic_limitation_without_leakage() -> None:
    from track_1_agent_under_test.car_guard.capability import SemanticResultValidator

    agent = runtime(_air_conditioning_client(), capture_result_issue=True)
    context_id = "hall-internal-extractor"
    read = agent.handle_event(
        user_event(
            context_id,
            f"{context_id}-1",
            "Turn on the air conditioning.",
            _air_conditioning_tools(),
        )
    )
    assert read.tool_calls
    agent.result_validator = SemanticResultValidator()

    blocked = agent.handle_event(
        result_event(
            context_id,
            f"{context_id}-result",
            (
                "get_climate_settings",
                {"status": "SUCCESS", "result": {"fan_speed": 0}},
            ),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    lowered = blocked.text.casefold()
    assert "extractor" not in lowered
    assert "fan speed information" not in lowered
    assert isinstance(agent, CapturingOrchestrator)
    issue = agent.captured_result_issue
    assert issue is not None
    assert [item.code for item in issue.validation_issues] == [
        "extractor_not_registered"
    ]


def test_unrelated_unknown_result_field_does_not_change_declared_evidence() -> None:
    agent = runtime(_air_conditioning_client())
    context_id = "hall-unrelated-unknown"
    first = agent.handle_event(
        user_event(
            context_id,
            f"{context_id}-1",
            "Turn on the air conditioning.",
            _air_conditioning_tools(),
        )
    )
    assert first.tool_calls[0]["tool_name"] == "get_climate_settings"

    next_read = agent.handle_event(
        result_event(
            context_id,
            f"{context_id}-result",
            (
                "get_climate_settings",
                {
                    "status": "SUCCESS",
                    "result": {"fan_speed": 0, "unrelated": "UNKNOWN"},
                },
            ),
        )
    )
    assert next_read.text is None
    assert next_read.tool_calls == (
        {"tool_name": "get_vehicle_window_positions", "arguments": {}},
    )


def test_later_ready_goal_missing_capability_blocks_before_earlier_set() -> None:
    client = FixedIntentClient(
        [
            {
                "semantic_operation": "set_fan_speed",
                "desired_outcome": {"level": 3},
            },
            {
                "semantic_operation": "set_ambient_lights",
                "desired_outcome": {"color": "PURPLE", "enabled": True},
            },
        ],
        {"level": 3, "color": "PURPLE", "enabled": True},
    )
    agent = runtime(client)
    context_id = "hall-later-goal-missing"
    blocked = agent.handle_event(
        user_event(
            context_id,
            f"{context_id}-1",
            "Set the fan speed to level 3 and change the ambient lights to purple.",
            (FAN_SET,),
        )
    )

    assert_human_limitation(
        blocked,
        phrase="ambient light control",
        forbidden_tool_name="set_ambient_lights",
    )
    session = agent.sessions.get(context_id)
    assert session is not None
    assert [goal.status.value for goal in session.goal_dag.goals] == [
        "pending",
        "pending",
    ]


def test_early_target_scan_blocks_specialized_climate_read_for_later_bad_schema() -> (
    None
):
    client = FixedIntentClient(
        [
            {
                "semantic_operation": "set_air_conditioning",
                "desired_outcome": {"enabled": True},
            },
            {
                "semantic_operation": "set_seat_heating",
                "desired_outcome": {"level": 2, "seat_zone": "DRIVER"},
            },
        ],
        {"enabled": True, "level": 2, "seat_zone": "DRIVER"},
    )
    agent = runtime(client)
    context_id = "hall-early-multigoal-schema"
    blocked = agent.handle_event(
        user_event(
            context_id,
            f"{context_id}-1",
            "Turn on the air conditioning and set the driver seat heating to level 2.",
            (*_air_conditioning_tools(), SEAT_HEATING_WITH_WRONG_PARAMETERS),
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    lowered = blocked.text.casefold()
    assert "seat heating control" in lowered
    assert "level and seat zone settings" in lowered
    assert "get_climate_settings" not in blocked.text


@pytest.mark.parametrize(
    "invalid_seat_heating_control",
    (SEAT_HEATING_WITH_INVALID_LEVEL_RANGE, SEAT_HEATING_WITH_INCOMPATIBLE_ZONE),
)
def test_early_target_scan_blocks_climate_read_for_later_invalid_arguments(
    invalid_seat_heating_control: dict[str, Any],
) -> None:
    client = FixedIntentClient(
        [
            {
                "semantic_operation": "set_air_conditioning",
                "desired_outcome": {"enabled": True},
            },
            {
                "semantic_operation": "set_seat_heating",
                "desired_outcome": {"level": 2, "seat_zone": "DRIVER"},
            },
        ],
        {"enabled": True, "level": 2, "seat_zone": "DRIVER"},
    )
    agent = runtime(client)
    context_id = "hall-early-invalid-target-arguments"
    blocked = agent.handle_event(
        user_event(
            context_id,
            f"{context_id}-1",
            "Turn on the air conditioning and set the driver seat heating to level 2.",
            (*_air_conditioning_tools(), invalid_seat_heating_control),
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    lowered = blocked.text.casefold()
    assert "available control cannot express the requested setting" in lowered
    assert "get_climate_settings" not in blocked.text
    assert "validation" not in lowered
    assert "enum" not in lowered
    assert "maximum" not in lowered
    assert "$." not in blocked.text
