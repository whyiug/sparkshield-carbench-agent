from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import (
    DecisionProposal,
    GoalStatus,
    OfficialToolCall,
)
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
)


ESSEN_ID = "loc_ess_699309"
MANNHEIM_ID = "loc_man_660365"
COLOGNE_ID = "loc_col_464166"
ALTERNATE_MANNHEIM_ID = "loc_man_123456"
ALTERNATE_COLOGNE_ID = "loc_col_123456"
MANNHEIM_PROBE_ROUTE_ID = "rll_ess_man_621643"
COLOGNE_FASTEST_ROUTE_ID = "rll_ess_col_838235"
COLOGNE_SHORTEST_ROUTE_ID = "rll_ess_col_645120"

REQUEST = (
    "Navigate to a charging station in Mannheim if it's not raining there. "
    "Otherwise, navigate to Cologne. Take the shortest route."
)
FORMAL_REQUEST = (
    "Navigate to a charging station in Mannheim, but only if it's not raining "
    "there. Otherwise, navigate to Cologne."
)
POLICY = (
    "Use the metric system and 24h times. "
    "AUT-POL:004: If a tool description starts with REQUIRES_CONFIRMATION, "
    "describe the exact action and parameters and obtain an explicit yes before "
    "calling it. AUT-POL:016: The start of a new route must be the current car "
    "location. AUT-POL:024: Weather may only be requested for the current day. "
    'CURRENT_LOCATION={"id":"loc_ess_699309","name":"Essen"}. '
    'CURRENT_DATETIME={"year":2025,"month":3,"day":13,"hour":16,"minute":0}.'
)


def _tool(
    name: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    *,
    description: str = "",
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


LOCATION_TOOL = _tool(
    "get_location_id_by_location_name",
    {"location": {"type": "string"}},
    ["location"],
)
ROUTES_TOOL = _tool(
    "get_routes_from_start_to_destination",
    {
        "start_id": {"type": "string"},
        "destination_id": {"type": "string"},
    },
    ["start_id", "destination_id"],
)
CURRENT_NAVIGATION_TOOL = _tool(
    "get_current_navigation_state",
    {"detailed_information": {"type": "boolean"}},
)
WEATHER_TOOL = _tool(
    "get_weather",
    {
        "location_or_poi_id": {"type": "string"},
        "month": {"type": "integer", "minimum": 1, "maximum": 12},
        "day": {"type": "integer", "minimum": 1, "maximum": 31},
        "time_hour_24hformat": {"type": "integer", "minimum": 0, "maximum": 23},
    },
    ["location_or_poi_id", "month", "day", "time_hour_24hformat"],
)
POI_SEARCH_TOOL = _tool(
    "search_poi_at_location",
    {
        "location_id": {"type": "string"},
        "category_poi": {"type": "string"},
        "filters": {"type": "array", "items": {"type": "string"}},
    },
    ["location_id", "category_poi"],
)
SET_NAVIGATION_TOOL = _tool(
    "set_new_navigation",
    {"route_ids": {"type": "array", "items": {"type": "string"}}},
    ["route_ids"],
    description="REQUIRES_CONFIRMATION: Start navigation on the supplied routes.",
)
FULL_TOOLS = (
    LOCATION_TOOL,
    ROUTES_TOOL,
    CURRENT_NAVIGATION_TOOL,
    WEATHER_TOOL,
    POI_SEARCH_TOOL,
    SET_NAVIGATION_TOOL,
)


def location_result(location_id: str) -> dict[str, Any]:
    return {"status": "SUCCESS", "result": {"id": location_id}}


def mannheim_probe_routes() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": MANNHEIM_PROBE_ROUTE_ID,
                    "start_id": ESSEN_ID,
                    "destination_id": MANNHEIM_ID,
                    "name_via": "A3, A67",
                    "distance_km": 276.38,
                    "duration_hours": 3,
                    "duration_minutes": 23,
                    "road_types": ["highway"],
                    "includes_toll": False,
                    "alias": ["fastest", "first", "shortest"],
                }
            ]
        },
    }


def mannheim_weather(condition: Any = "rainy") -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "current_slot": {
                "start_time": "18:00",
                "end_time": "21:00",
                "temperature_c": 7,
                "wind_speed_kph": 18,
                "humidity_percent": 91,
                "condition": condition,
            },
            "next_slot": None,
        },
    }


def cologne_routes() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": COLOGNE_FASTEST_ROUTE_ID,
                    "start_id": ESSEN_ID,
                    "destination_id": COLOGNE_ID,
                    "name_via": "A3",
                    "distance_km": 79.4,
                    "duration_hours": 1,
                    "duration_minutes": 3,
                    "road_types": ["highway"],
                    "includes_toll": False,
                    "alias": ["fastest", "first"],
                },
                {
                    "route_id": COLOGNE_SHORTEST_ROUTE_ID,
                    "start_id": ESSEN_ID,
                    "destination_id": COLOGNE_ID,
                    "name_via": "A42, A57",
                    "distance_km": 72.8,
                    "duration_hours": 1,
                    "duration_minutes": 11,
                    "road_types": ["highway", "urban"],
                    "includes_toll": False,
                    "alias": ["second", "shortest"],
                },
            ]
        },
    }


def inactive_navigation_state() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": False,
            "waypoints_id": [],
            "routes_to_final_destination_id": [],
        },
    }


def active_navigation_state() -> dict[str, Any]:
    active_route = deepcopy(cologne_routes()["result"]["routes"][0])
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [ESSEN_ID, COLOGNE_ID],
            "routes_to_final_destination_id": [active_route["route_id"]],
            "details": {
                "waypoints": [
                    {"id": ESSEN_ID, "name": "Essen"},
                    {"id": COLOGNE_ID, "name": "Cologne"},
                ],
                "routes": [active_route],
            },
        },
    }


class Base96Client:
    def __init__(self) -> None:
        self.intent_calls = 0
        self.action_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        del messages, critic
        if response_model is IntentDraft:
            self.intent_calls += 1
            if self.intent_calls > 1:
                raise AssertionError(
                    "Base96 must retain one grounded conditional intent"
                )
            # Deliberately unsafe: deterministic recovery must not trust the guessed
            # branch or route preference before current route/weather evidence.
            return SimpleNamespace(
                value=IntentDraft.model_validate(
                    {
                        "language": "en",
                        "intent_kind": "action",
                        "call_for_action": True,
                        "goals": [
                            {
                                "semantic_operation": "navigation_create",
                                "desired_outcome": {
                                    "location": "Mannheim",
                                    "route_choice_alias": "fastest",
                                },
                            }
                        ],
                        "explicit_slots": {
                            "location": "Mannheim",
                            "route_choice_alias": "fastest",
                        },
                    }
                )
            )
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError(
                "Base96 reads, branch choice, and SET must be deterministic"
            )
        raise AssertionError(f"unexpected response model: {response_model}")


def _runtime() -> tuple[CARGuardOrchestrator, Base96Client]:
    client = Base96Client()
    config = AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=32)
    return CARGuardOrchestrator(config, client_factory=lambda session: client), client


def _user_event(
    context_id: str,
    message_id: str,
    text: str,
    *,
    tools: tuple[dict[str, Any], ...] = (),
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        system_policy=POLICY if tools else None,
        user_text=text,
        live_tools=tools,
    )


def _result_event(
    context_id: str,
    message_id: str,
    calls: tuple[dict[str, Any], ...],
    results: dict[str, Any],
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        tool_results=tuple(
            {
                "toolName": call["tool_name"],
                "content": deepcopy(results[call["tool_name"]]),
            }
            for call in calls
        ),
    )


def _assert_one_call(
    outbound: Any, name: str, arguments: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    assert outbound.tool_calls == ({"tool_name": name, "arguments": arguments},)
    return outbound.tool_calls


def _run_to_mannheim_route(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    request: str = REQUEST,
    tools: tuple[dict[str, Any], ...] = FULL_TOOLS,
) -> Any:
    lookup = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-user",
            request,
            tools=tools,
        )
    )
    lookup_calls = _assert_one_call(
        lookup,
        "get_location_id_by_location_name",
        {"location": "Mannheim"},
    )
    probe = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-mannheim-location",
            lookup_calls,
            {"get_location_id_by_location_name": location_result(MANNHEIM_ID)},
        )
    )
    _assert_one_call(
        probe,
        "get_routes_from_start_to_destination",
        {"start_id": ESSEN_ID, "destination_id": MANNHEIM_ID},
    )
    return probe


def _run_to_weather(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    route_result: dict[str, Any] | None = None,
    request: str = REQUEST,
    tools: tuple[dict[str, Any], ...] = FULL_TOOLS,
) -> Any:
    probe = _run_to_mannheim_route(
        runtime,
        context_id=context_id,
        request=request,
        tools=tools,
    )
    probe_calls = _assert_one_call(
        probe,
        "get_routes_from_start_to_destination",
        {"start_id": ESSEN_ID, "destination_id": MANNHEIM_ID},
    )
    weather = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-mannheim-route",
            probe_calls,
            {
                "get_routes_from_start_to_destination": route_result
                or mannheim_probe_routes()
            },
        )
    )
    _assert_one_call(
        weather,
        "get_weather",
        {
            "location_or_poi_id": MANNHEIM_ID,
            "month": 3,
            "day": 13,
            "time_hour_24hformat": 19,
        },
    )
    return weather


@pytest.mark.parametrize(
    "route_reply",
    [
        "Use the shortest route.",
        "Take the shortest route, not the fastest.",
        "Shortest route.",
    ],
)
def test_conditional_navigation_without_initial_route_choice_binds_fresh_reply(
    route_reply: str,
) -> None:
    runtime, client = _runtime()
    context_id = f"conditional-route-reply-{abs(hash(route_reply))}"
    weather = _run_to_weather(
        runtime,
        context_id=context_id,
        request=FORMAL_REQUEST,
    )
    fallback_location = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-weather",
            weather.tool_calls,
            {"get_weather": mannheim_weather("cloudy_and_rain_and_hail")},
        )
    )
    fallback = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-fallback-location",
            fallback_location.tool_calls,
            {"get_location_id_by_location_name": location_result(COLOGNE_ID)},
        )
    )
    clarification = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-fallback-routes",
            fallback.tool_calls,
            {"get_routes_from_start_to_destination": cologne_routes()},
        )
    )

    assert clarification.tool_calls == ()
    assert clarification.text is not None
    assert "cologne" in clarification.text.casefold()
    assert "fastest" in clarification.text.casefold()
    assert "shortest" in clarification.text.casefold()

    snapshot = runtime.handle_event(
        _user_event(context_id, f"{context_id}-route-choice", route_reply)
    )
    _assert_one_call(
        snapshot,
        "get_current_navigation_state",
        {"detailed_information": True},
    )
    confirmation = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-inactive-navigation",
            snapshot.tool_calls,
            {"get_current_navigation_state": inactive_navigation_state()},
        )
    )
    assert confirmation.tool_calls == ()
    assert confirmation.text is not None
    assert "cologne" in confirmation.text.casefold()
    assert "shortest" in confirmation.text.casefold()

    started = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-yes",
            "Yes, start navigation to Cologne on the shortest route.",
        )
    )
    _assert_one_call(
        started,
        "set_new_navigation",
        {"route_ids": [COLOGNE_SHORTEST_ROUTE_ID]},
    )
    assert client.intent_calls == 1
    assert client.action_calls == 0


def _run_to_navigation_snapshot(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    route_result: dict[str, Any] | None = None,
    weather_result: dict[str, Any] | None = None,
    tools: tuple[dict[str, Any], ...] = FULL_TOOLS,
) -> Any:
    weather = _run_to_weather(
        runtime,
        context_id=context_id,
        route_result=route_result,
        tools=tools,
    )
    weather_calls = weather.tool_calls
    fallback_location = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-weather",
            weather_calls,
            {"get_weather": weather_result or mannheim_weather()},
        )
    )
    fallback_location_calls = _assert_one_call(
        fallback_location,
        "get_location_id_by_location_name",
        {"location": "Cologne"},
    )
    fallback = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-cologne-location",
            fallback_location_calls,
            {"get_location_id_by_location_name": location_result(COLOGNE_ID)},
        )
    )
    fallback_calls = _assert_one_call(
        fallback,
        "get_routes_from_start_to_destination",
        {"start_id": ESSEN_ID, "destination_id": COLOGNE_ID},
    )
    navigation_snapshot = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-cologne-route",
            fallback_calls,
            {"get_routes_from_start_to_destination": cologne_routes()},
        )
    )
    _assert_one_call(
        navigation_snapshot,
        "get_current_navigation_state",
        {"detailed_information": True},
    )
    return navigation_snapshot


def _run_to_confirmation(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    route_result: dict[str, Any] | None = None,
    weather_result: dict[str, Any] | None = None,
    tools: tuple[dict[str, Any], ...] = FULL_TOOLS,
) -> Any:
    navigation_snapshot = _run_to_navigation_snapshot(
        runtime,
        context_id=context_id,
        route_result=route_result,
        weather_result=weather_result,
        tools=tools,
    )
    return runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-inactive-navigation",
            navigation_snapshot.tool_calls,
            {"get_current_navigation_state": inactive_navigation_state()},
        )
    )


def test_base96_rain_branch_uses_arrival_weather_then_exact_confirmed_shortest_set() -> (
    None
):
    runtime, client = _runtime()
    context_id = "base96-rain-success"

    confirmation = _run_to_confirmation(runtime, context_id=context_id)

    assert confirmation.tool_calls == ()
    assert confirmation.text is not None
    assert "cologne" in confirmation.text.casefold()
    assert "shortest" in confirmation.text.casefold()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    navigation_reads = [
        result
        for result in session.successful_read_results
        if result.tool_name == "get_current_navigation_state"
        and result.arguments == {"detailed_information": True}
        and result.state_version == session.evidence.current_state_version
    ]
    assert len(navigation_reads) == 1
    assert navigation_reads[0].value == inactive_navigation_state()["result"]
    expected = OfficialToolCall(
        tool_name="set_new_navigation",
        arguments={"route_ids": [COLOGNE_SHORTEST_ROUTE_ID]},
    )
    assert session.execution_bundle.calls == (expected,)
    active = session.confirmation_latch.active
    assert len(active) == 1
    assert active[0].scope.goal_ids == list(session.execution_bundle.goal_ids)
    assert active[0].scope.ordered_actions == [expected]
    assert active[0].scope.requested_at_user_turn == 1

    started = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-yes",
            "Yes, start navigation to Cologne on that shortest route.",
        )
    )
    set_calls = _assert_one_call(
        started,
        "set_new_navigation",
        {"route_ids": [COLOGNE_SHORTEST_ROUTE_ID]},
    )
    completed = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-set-result",
            set_calls,
            {
                "set_new_navigation": {
                    "status": "SUCCESS",
                    "result": {
                        "navigation_set": True,
                        "start_id": ESSEN_ID,
                        "waypoints": [ESSEN_ID, COLOGNE_ID],
                        "destination_id": COLOGNE_ID,
                    },
                }
            },
        )
    )

    assert completed.tool_calls == ()
    assert completed.text is not None
    assert "started navigation" in completed.text.casefold()
    assert all(goal.status is GoalStatus.DONE for goal in session.goal_dag.goals)
    assert client.intent_calls == 1
    assert client.action_calls == 0


def test_base96_active_navigation_fails_before_bundle_or_confirmation() -> None:
    runtime, client = _runtime()
    context_id = "base96-active-navigation"
    snapshot = _run_to_navigation_snapshot(runtime, context_id=context_id)

    blocked = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-active-navigation",
            snapshot.tool_calls,
            {"get_current_navigation_state": active_navigation_state()},
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.execution_bundle is None
    assert not session.confirmation_latch.active
    assert session.budget.attempted_sets == set()
    assert client.intent_calls == 1
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "reply",
    [
        "No, don't start navigation.",
        'She said, "Yes, start navigation to Cologne."',
        "If I confirmed navigation to Cologne, what would happen?",
        "Yes, start navigation to Mannheim instead.",
    ],
)
def test_base96_confirmation_is_fresh_affirmative_and_exact_scope(reply: str) -> None:
    runtime, client = _runtime()
    context_id = f"base96-confirmation-{abs(hash(reply))}"
    confirmation = _run_to_confirmation(runtime, context_id=context_id)
    assert confirmation.tool_calls == ()

    rejected = runtime.handle_event(
        _user_event(context_id, f"{context_id}-reply", reply)
    )

    assert rejected.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "conflicting_reply",
    [
        "Yes, start navigation to Mannheim.",
        "Yes, use the fastest route.",
    ],
)
def test_base96_confirmation_rejects_explicitly_conflicting_scope(
    conflicting_reply: str,
) -> None:
    runtime, client = _runtime()
    context_id = f"base96-conflicting-confirmation-{abs(hash(conflicting_reply))}"
    confirmation = _run_to_confirmation(runtime, context_id=context_id)
    assert confirmation.tool_calls == ()

    rejected = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-reply",
            conflicting_reply,
        )
    )

    assert rejected.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


@pytest.mark.parametrize("case", ["wrong_endpoint", "missing_duration"])
def test_base96_malformed_eta_route_never_reaches_weather_or_set(case: str) -> None:
    runtime, client = _runtime()
    context_id = f"base96-bad-eta-{case}"
    malformed = mannheim_probe_routes()
    route = malformed["result"]["routes"][0]
    if case == "wrong_endpoint":
        route["destination_id"] = COLOGNE_ID
    else:
        route.pop("duration_hours")

    probe = _run_to_mannheim_route(runtime, context_id=context_id)
    calls = probe.tool_calls
    blocked = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-route",
            calls,
            {"get_routes_from_start_to_destination": malformed},
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "condition",
    ["unknown", "drizzle", "showers", "thunderstorm"],
)
def test_base96_unrecognized_weather_condition_fails_closed(condition: str) -> None:
    runtime, client = _runtime()
    context_id = f"base96-unrecognized-weather-{condition}"
    weather = _run_to_weather(runtime, context_id=context_id)

    blocked = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-weather",
            weather.tool_calls,
            {"get_weather": mannheim_weather(condition)},
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


@pytest.mark.parametrize(
    ("condition", "expected_tool", "expected_arguments"),
    [
        (
            "sunny",
            "search_poi_at_location",
            {
                "location_id": MANNHEIM_ID,
                "category_poi": "charging_stations",
            },
        ),
        (
            "cloudy",
            "search_poi_at_location",
            {
                "location_id": MANNHEIM_ID,
                "category_poi": "charging_stations",
            },
        ),
        (
            "partly_cloudy",
            "search_poi_at_location",
            {
                "location_id": MANNHEIM_ID,
                "category_poi": "charging_stations",
            },
        ),
        (
            "rainy",
            "get_location_id_by_location_name",
            {"location": "Cologne"},
        ),
        (
            "snowy",
            "search_poi_at_location",
            {
                "location_id": MANNHEIM_ID,
                "category_poi": "charging_stations",
            },
        ),
        (
            "cloudy_and_thunderstorm",
            "search_poi_at_location",
            {
                "location_id": MANNHEIM_ID,
                "category_poi": "charging_stations",
            },
        ),
        (
            "cloudy_and_hail",
            "search_poi_at_location",
            {
                "location_id": MANNHEIM_ID,
                "category_poi": "charging_stations",
            },
        ),
    ],
)
def test_base96_official_weather_conditions_select_expected_branch(
    condition: str,
    expected_tool: str,
    expected_arguments: dict[str, Any],
) -> None:
    runtime, client = _runtime()
    context_id = f"base96-official-weather-{condition}"
    weather = _run_to_weather(runtime, context_id=context_id)

    branch = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-weather",
            weather.tool_calls,
            {"get_weather": mannheim_weather(condition)},
        )
    )

    _assert_one_call(branch, expected_tool, expected_arguments)
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


@pytest.mark.parametrize("case", ["malformed_condition", "stale_slot"])
def test_base96_unsafe_destination_weather_never_selects_a_branch_or_set(
    case: str,
) -> None:
    runtime, client = _runtime()
    context_id = f"base96-bad-weather-{case}"
    weather_result = mannheim_weather()
    slot = weather_result["result"]["current_slot"]
    if case == "malformed_condition":
        slot["condition"] = ["rainy", "cloudy"]
    else:
        slot["start_time"] = "15:00"
        slot["end_time"] = "18:00"

    weather = _run_to_weather(runtime, context_id=context_id)
    weather_calls = _assert_one_call(
        weather,
        "get_weather",
        {
            "location_or_poi_id": MANNHEIM_ID,
            "month": 3,
            "day": 13,
            "time_hour_24hformat": 19,
        },
    )
    blocked = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-weather",
            weather_calls,
            {"get_weather": weather_result},
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base96_clear_weather_selects_mannheim_charging_search_not_cologne() -> None:
    runtime, client = _runtime()
    context_id = "base96-clear-branch"
    weather = _run_to_weather(runtime, context_id=context_id)
    weather_calls = weather.tool_calls
    clear = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-weather",
            weather_calls,
            {"get_weather": mannheim_weather("cloudy")},
        )
    )

    assert len(clear.tool_calls) == 1
    call = clear.tool_calls[0]
    assert call["tool_name"] == "search_poi_at_location"
    assert call["arguments"]["location_id"] == MANNHEIM_ID
    assert call["arguments"]["category_poi"] == "charging_stations"
    assert call["arguments"].get("destination_id") != COLOGNE_ID
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "missing_tool",
    [LOCATION_TOOL, WEATHER_TOOL],
    ids=["location", "weather"],
)
def test_base96_missing_required_lookup_tool_fails_before_any_read(
    missing_tool: dict[str, Any],
) -> None:
    runtime, client = _runtime()
    tool_name = missing_tool["function"]["name"]
    context_id = f"base96-missing-{tool_name}"
    tools = tuple(tool for tool in FULL_TOOLS if tool is not missing_tool)
    outbound = runtime.handle_event(
        _user_event(context_id, f"{context_id}-user", REQUEST, tools=tools)
    )

    assert outbound.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base96_mannheim_route_uses_live_resolved_id_not_static_mapping() -> None:
    runtime, client = _runtime()
    context_id = "base96-live-mannheim-resolution"
    lookup = runtime.handle_event(
        _user_event(context_id, f"{context_id}-user", REQUEST, tools=FULL_TOOLS)
    )
    calls = _assert_one_call(
        lookup,
        "get_location_id_by_location_name",
        {"location": "Mannheim"},
    )

    route = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-location",
            calls,
            {
                "get_location_id_by_location_name": location_result(
                    ALTERNATE_MANNHEIM_ID
                )
            },
        )
    )

    _assert_one_call(
        route,
        "get_routes_from_start_to_destination",
        {"start_id": ESSEN_ID, "destination_id": ALTERNATE_MANNHEIM_ID},
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base96_fallback_route_uses_live_resolved_id_not_static_mapping() -> None:
    runtime, client = _runtime()
    context_id = "base96-live-cologne-resolution"
    weather = _run_to_weather(runtime, context_id=context_id)
    fallback_location = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-weather",
            weather.tool_calls,
            {"get_weather": mannheim_weather()},
        )
    )
    calls = _assert_one_call(
        fallback_location,
        "get_location_id_by_location_name",
        {"location": "Cologne"},
    )

    fallback_route = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-location",
            calls,
            {"get_location_id_by_location_name": location_result(ALTERNATE_COLOGNE_ID)},
        )
    )

    _assert_one_call(
        fallback_route,
        "get_routes_from_start_to_destination",
        {"start_id": ESSEN_ID, "destination_id": ALTERNATE_COLOGNE_ID},
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0
