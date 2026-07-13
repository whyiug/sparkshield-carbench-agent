from __future__ import annotations

from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.llm.client import LLMFailure
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
    _fully_open_window_close_intent,
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


WINDOW_READ = tool("get_vehicle_window_positions")
WINDOW_SET = tool(
    "open_close_window",
    {
        "window": {
            "type": "string",
            "enum": ["ALL", "DRIVER", "PASSENGER", "DRIVER_REAR", "PASSENGER_REAR"],
        },
        "percentage": {"type": "number", "minimum": 0, "maximum": 100},
    },
    ["window", "percentage"],
)
TEMPERATURE_READ = tool("get_temperature_inside_car")
TEMPERATURE_SET = tool(
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


class UnavailableModel:
    def generate(self, **kwargs: Any) -> Any:
        del kwargs
        raise LLMFailure("synthetic model unavailable")


def runtime() -> CARGuardOrchestrator:
    return CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=24),
        client_factory=lambda session: UnavailableModel(),
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
        system_policy="Use only available controls and current verified state.",
        user_text=text,
        live_tools=tools,
    )


def result_event(
    context_id: str, message_id: str, tool_name: str, result: dict[str, Any]
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        tool_results=(
            {
                "toolName": tool_name,
                "content": {"status": "SUCCESS", "result": result},
            },
        ),
    )


@pytest.mark.parametrize(
    ("request_text", "expected_window"),
    (
        ("Close the window that is fully open.", "PASSENGER_REAR"),
        ("Could you please close the completely open window?", "PASSENGER_REAR"),
        ("Please close the wide open window.", "PASSENGER_REAR"),
        ("Close the fully-open window.", "PASSENGER_REAR"),
        ("Close the window that is currently fully open.", "PASSENGER_REAR"),
        ("Close whichever window is fully open.", "PASSENGER_REAR"),
        ("Check which window is fully open and close it.", "PASSENGER_REAR"),
    ),
)
def test_unique_fully_open_window_is_read_and_bound(
    request_text: str, expected_window: str
) -> None:
    agent = runtime()
    context_id = f"unique-window-{expected_window}-{len(request_text)}"

    read = agent.handle_event(
        user_event(
            context_id, "window-user", request_text, (WINDOW_READ, WINDOW_SET)
        )
    )
    assert read.tool_calls == (
        {"tool_name": "get_vehicle_window_positions", "arguments": {}},
    )

    action = agent.handle_event(
        result_event(
            context_id,
            "window-state",
            "get_vehicle_window_positions",
            {
                "window_driver_position": 0,
                "window_passenger_position": 20,
                "window_driver_rear_position": 50,
                "window_passenger_rear_position": 100,
            },
        )
    )
    assert action.tool_calls == (
        {
            "tool_name": "open_close_window",
            "arguments": {"window": expected_window, "percentage": 0},
        },
    )
    session = agent.sessions.get(context_id)
    assert session is not None and session.intent is not None
    goal = session.intent.goals[0]
    evidence_id = session.derived_value_evidence_by_goal[goal.goal_id]["window"]
    evidence = session.evidence.evidence[evidence_id]
    assert evidence.value == expected_window
    assert evidence.derivation == "unique_fully_open_window_v1"
    assert len(evidence.derived_from) == 1


@pytest.mark.parametrize(
    "positions",
    (
        {
            "window_driver_position": 0,
            "window_passenger_position": 20,
            "window_driver_rear_position": 50,
            "window_passenger_rear_position": 80,
        },
        {
            "window_driver_position": 100,
            "window_passenger_position": 20,
            "window_driver_rear_position": 50,
            "window_passenger_rear_position": 100,
        },
    ),
)
def test_nonunique_fully_open_window_remains_a_clarification(
    positions: dict[str, Any],
) -> None:
    agent = runtime()
    context_id = f"nonunique-window-{positions['window_driver_position']}"
    read = agent.handle_event(
        user_event(
            context_id,
            "window-user",
            "Close the fully open window.",
            (WINDOW_READ, WINDOW_SET),
        )
    )
    assert read.tool_calls

    clarification = agent.handle_event(
        result_event(
            context_id,
            "window-state",
            "get_vehicle_window_positions",
            positions,
        )
    )
    assert clarification.tool_calls == ()
    assert clarification.text is not None and "Which option" in clarification.text
    session = agent.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert session.derived_value_evidence_by_goal == {}


@pytest.mark.parametrize(
    "positions",
    (
        {
            "window_driver_position": 100,
            "window_passenger_position": 0,
            "window_driver_rear_position": 0,
        },
        {
            "window_driver_position": float("nan"),
            "window_passenger_position": 0,
            "window_driver_rear_position": 0,
            "window_passenger_rear_position": 0,
        },
        {
            "window_driver_position": 101,
            "window_passenger_position": 0,
            "window_driver_rear_position": 0,
            "window_passenger_rear_position": 0,
        },
    ),
)
def test_malformed_window_snapshot_does_not_bind_or_set(
    positions: dict[str, Any],
) -> None:
    agent = runtime()
    context_id = "malformed-window"
    read = agent.handle_event(
        user_event(
            context_id,
            "window-user",
            "Close the fully open window.",
            (WINDOW_READ, WINDOW_SET),
        )
    )
    assert read.tool_calls

    blocked = agent.handle_event(
        result_event(
            context_id,
            "window-state",
            "get_vehicle_window_positions",
            positions,
        )
    )
    assert blocked.tool_calls == ()
    session = agent.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert session.derived_value_evidence_by_goal == {}


def test_unique_window_not_in_live_enum_fails_closed() -> None:
    driver_only_set = tool(
        "open_close_window",
        {
            "window": {"type": "string", "enum": ["DRIVER"]},
            "percentage": {"type": "number", "minimum": 0, "maximum": 100},
        },
        ["window", "percentage"],
    )
    agent = runtime()
    context_id = "unique-window-outside-live-enum"
    read = agent.handle_event(
        user_event(
            context_id,
            "window-user",
            "Close the fully open window.",
            (WINDOW_READ, driver_only_set),
        )
    )
    assert read.tool_calls

    blocked = agent.handle_event(
        result_event(
            context_id,
            "window-state",
            "get_vehicle_window_positions",
            {
                "window_driver_position": 0,
                "window_passenger_position": 0,
                "window_driver_rear_position": 0,
                "window_passenger_rear_position": 100,
            },
        )
    )
    assert blocked.tool_calls == ()
    session = agent.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert session.derived_value_evidence_by_goal == {}


@pytest.mark.parametrize(
    ("driver", "passenger", "expected_zone"),
    ((20, 22, "DRIVER"), (22, 24, "PASSENGER")),
)
def test_absolute_temperature_binds_the_unique_mismatching_zone(
    driver: int, passenger: int, expected_zone: str
) -> None:
    agent = runtime()
    context_id = f"unique-temperature-{expected_zone.casefold()}"
    request_text = (
        "Set the temperature to 22 degrees."
        if expected_zone == "DRIVER"
        else "Could you please set the temperature to 22 degrees?"
    )
    read = agent.handle_event(
        user_event(
            context_id,
            "temperature-user",
            request_text,
            (TEMPERATURE_READ, TEMPERATURE_SET),
        )
    )
    assert read.tool_calls == (
        {"tool_name": "get_temperature_inside_car", "arguments": {}},
    )

    action = agent.handle_event(
        result_event(
            context_id,
            "temperature-state",
            "get_temperature_inside_car",
            {
                "climate_temperature_driver": driver,
                "climate_temperature_passenger": passenger,
                "temperature_unit": "Celsius",
            },
        )
    )
    assert action.tool_calls == (
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"temperature": 22, "seat_zone": expected_zone},
        },
    )
    session = agent.sessions.get(context_id)
    assert session is not None and session.intent is not None
    goal = session.intent.goals[0]
    evidence_id = session.derived_value_evidence_by_goal[goal.goal_id]["seat_zone"]
    evidence = session.evidence.evidence[evidence_id]
    assert evidence.value == expected_zone
    assert evidence.derivation == "unique_climate_target_mismatch_zone_v1"


@pytest.mark.parametrize(("driver", "passenger"), ((22, 22), (20, 24)))
def test_absolute_temperature_without_unique_mismatch_remains_a_clarification(
    driver: int, passenger: int
) -> None:
    agent = runtime()
    context_id = f"nonunique-temperature-{driver}-{passenger}"
    read = agent.handle_event(
        user_event(
            context_id,
            "temperature-user",
            "Set the temperature to 22 degrees.",
            (TEMPERATURE_READ, TEMPERATURE_SET),
        )
    )
    assert read.tool_calls

    clarification = agent.handle_event(
        result_event(
            context_id,
            "temperature-state",
            "get_temperature_inside_car",
            {
                "climate_temperature_driver": driver,
                "climate_temperature_passenger": passenger,
                "temperature_unit": "Celsius",
            },
        )
    )
    assert clarification.tool_calls == ()
    assert clarification.text is not None and "Which option" in clarification.text
    session = agent.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert session.derived_value_evidence_by_goal == {}


@pytest.mark.parametrize(
    "snapshot",
    (
        {
            "climate_temperature_driver": 20,
            "climate_temperature_passenger": "22",
            "temperature_unit": "Celsius",
        },
        {
            "climate_temperature_driver": 20,
            "climate_temperature_passenger": 22,
            "temperature_unit": "Fahrenheit",
        },
    ),
)
def test_malformed_temperature_snapshot_does_not_bind_or_set(
    snapshot: dict[str, Any],
) -> None:
    agent = runtime()
    context_id = "malformed-temperature"
    read = agent.handle_event(
        user_event(
            context_id,
            "temperature-user",
            "Set the temperature to 22 degrees.",
            (TEMPERATURE_READ, TEMPERATURE_SET),
        )
    )
    assert read.tool_calls

    blocked = agent.handle_event(
        result_event(
            context_id,
            "temperature-state",
            "get_temperature_inside_car",
            snapshot,
        )
    )
    assert blocked.tool_calls == ()
    session = agent.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert session.derived_value_evidence_by_goal == {}


def test_explicit_temperature_zone_does_not_trigger_contextual_read() -> None:
    agent = runtime()
    action = agent.handle_event(
        user_event(
            "explicit-temperature-zone",
            "temperature-user",
            "Set the driver zone temperature to 22 degrees.",
            (TEMPERATURE_READ, TEMPERATURE_SET),
        )
    )
    assert action.tool_calls == (
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"temperature": 22, "seat_zone": "DRIVER"},
        },
    )


def test_missing_context_read_falls_back_to_user_zone_without_stale_gate() -> None:
    agent = runtime()
    context_id = "temperature-context-read-unavailable"
    clarification = agent.handle_event(
        user_event(
            context_id,
            "temperature-user",
            "Set the temperature to 22 degrees.",
            (TEMPERATURE_SET,),
        )
    )
    assert clarification.tool_calls == ()
    assert clarification.text is not None and "Which option" in clarification.text

    action = agent.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id="temperature-zone-reply",
            user_text="Passenger.",
        )
    )
    assert action.tool_calls == (
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"temperature": 22, "seat_zone": "PASSENGER"},
        },
    )


@pytest.mark.parametrize(
    "request_text",
    (
        "Do not set the temperature to 22 degrees.",
        "If it gets warmer, set the temperature to 22 degrees.",
        'The passenger said "set the temperature to 22 degrees".',
        "The passenger reported that we should set the temperature to 22 degrees.",
        "You shouldn't set the temperature to 22 degrees.",
        "There is no need to set the temperature to 22 degrees.",
        "You are not allowed to set the temperature to 22 degrees.",
        "You must not set the temperature to 22 degrees.",
        "Can you tell me how to set the temperature to 22 degrees?",
        "Explain how to set the temperature to 22 degrees.",
        "Is it possible to set the temperature to 22 degrees?",
        "I want to know whether you can set the temperature to 22 degrees.",
    ),
)
def test_unsafe_temperature_scope_does_not_trigger_contextual_read(
    request_text: str,
) -> None:
    agent = runtime()
    outbound = agent.handle_event(
        user_event(
            f"unsafe-temperature-{len(request_text)}",
            "temperature-user",
            request_text,
            (TEMPERATURE_READ, TEMPERATURE_SET),
        )
    )
    assert outbound.tool_calls == ()


@pytest.mark.parametrize(
    "request_text",
    (
        "Do not close the fully open window.",
        "If it rains, close the fully open window.",
        'The driver said "close the fully open window".',
        "The driver reported that we should close the fully open window.",
        "Close the window that is not fully open.",
        "Close the driver window that is fully open.",
    ),
)
def test_contextual_window_matcher_rejects_unsafe_or_explicit_scope(
    request_text: str,
) -> None:
    assert (
        _fully_open_window_close_intent(request_text, turn_id="synthetic") is None
    )
