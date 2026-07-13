from __future__ import annotations

from copy import deepcopy
import re
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


MANNHEIM_ID = "loc_man_660365"
STUTTGART_ID = "loc_stu_828398"
ALTERNATE_STUTTGART_ID = "loc_stu_928398"
FASTNED_ID = "poi_cha_363177"
FASTNED_PLUG_ID = "plg_cha_564167"
FASTNED_PHONE = "+49 110 1244459"

DIRECT_ROUTE_ID = "rll_man_stu_853906"
TO_CHARGER_ROUTE_ID = "rlp_man_cha_455314"
FROM_CHARGER_ROUTE_ID = "rpl_cha_stu_597902"
ALTERNATE_DIRECT_ROUTE_ID = "rll_man_stu_953906"
ALTERNATE_FROM_CHARGER_ROUTE_ID = "rpl_cha_stu_697902"

INITIAL_REQUEST = (
    "What are the minimum and maximum charging times that would still get me "
    "to my next meeting on time, with a 5-minute arrival buffer and 40 "
    "kilometers of range left, if I use the fastest charger around my current "
    "location?"
)
OCCUPIED_PLUG_SELECTION = (
    "Use Fastned's fastest 300 kW plug for the calculation even though it is "
    "occupied. For any route selection, use the fastest route."
)
OCCUPIED_PLUG_SELECTION_VARIANTS = (
    (
        "Okay, let's go with the 300 kW plug at Fastned then. I understand it's "
        "currently occupied, but please use it for the calculation."
    ),
    (
        "Alright, I select the 300 kW plug at Fastned. I understand and accept "
        "that it's currently occupied, please proceed with the calculation using "
        "this plug."
    ),
    (
        "Okay, I select the 300 kW plug at Fastned. I explicitly accept its "
        "current availability."
    ),
)
NAVIGATION_REQUEST = (
    "The minimum charging time is under 12 minutes, so set navigation first to "
    "that Fastned charger and then to my Stuttgart meeting, using the fastest "
    "route for both segments."
)
POST_ANALYSIS_NAVIGATION_REQUESTS = (
    (
        "Okay, since the minimum charging time is less than 12 minutes, please "
        "set up navigation first to the charging station and then to my meeting "
        "location. Make sure to use the fastest route for both segments."
    ),
    (
        "I asked you to set up navigation first to the charging station and then "
        "to my meeting location, using the fastest route for both segments. Can "
        "you please do that?"
    ),
    (
        "I need you to set up navigation. First to the Fastned charging station, "
        "and then to my meeting location. Please use the fastest route for both "
        "parts of the journey."
    ),
    (
        "Please set up navigation to the Fastned charging station, and then to my "
        "meeting location. Fastest route for both, please."
    ),
)
CALL_REQUEST = (
    "Now that navigation is fully set, if the maximum charging time is more "
    "than 30 minutes and less than 50 minutes, call the charging station "
    "provider to reserve the plug."
)
NEGATED_NAVIGATION_REQUEST = (
    "The minimum charging time is under 12 minutes, so do not set navigation "
    "first to that Fastned charger and then to my Stuttgart meeting, using the "
    "fastest route for both segments."
)
DESCRIPTIVE_NAVIGATION_REQUEST = (
    "The proposed plan would set navigation first to that Fastned charger and "
    "then to my Stuttgart meeting because the minimum charging time is under "
    "12 minutes, using the fastest route for both segments."
)
NEGATED_CALL_REQUEST = (
    "Now that navigation is fully set, if the maximum charging time is more "
    "than 30 minutes and less than 50 minutes, do not call the charging station "
    "provider to reserve the plug."
)
DESCRIPTIVE_CALL_REQUEST = (
    "Now that navigation is fully set, if the maximum charging time is more "
    "than 30 minutes and less than 50 minutes, the app would call the charging "
    "station provider to reserve the plug."
)
POLICY = (
    "Use metric units and 24h times. Only current-day calendar entries may be "
    "read. The start of a new route must be the current car location. If "
    "navigation is already active, edit it in sequence; a new navigation may "
    "only be set while navigation is inactive. A phone call ends the "
    "conversation. "
    'CURRENT_LOCATION={"id":"loc_man_660365","name":"Mannheim"}. '
    'CURRENT_DATETIME={"year":2025,"month":1,"day":10,"hour":13,"minute":20}.'
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


CALENDAR_TOOL = _tool(
    "get_entries_from_calendar",
    {
        "month": {"type": "integer", "minimum": 1, "maximum": 12},
        "day": {"type": "integer", "minimum": 1, "maximum": 31},
    },
    ["month", "day"],
)
LOCATION_TOOL = _tool(
    "get_location_id_by_location_name",
    {"location": {"type": "string"}},
    ["location"],
)
CHARGING_STATUS_TOOL = _tool("get_charging_specs_and_status")
CURRENT_NAVIGATION_TOOL = _tool(
    "get_current_navigation_state",
    {"detailed_information": {"type": "boolean"}},
)
ROUTES_TOOL = _tool(
    "get_routes_from_start_to_destination",
    {
        "start_id": {"type": "string"},
        "destination_id": {"type": "string"},
    },
    ["start_id", "destination_id"],
)
LOCAL_CHARGER_TOOL = _tool(
    "search_poi_at_location",
    {
        "location_id": {"type": "string"},
        "category_poi": {"type": "string"},
        "filters": {"type": "array", "items": {"type": "string"}},
    },
    ["location_id", "category_poi"],
)
CHARGING_TIME_TOOL = _tool(
    "calculate_charging_time_by_soc",
    {
        "charging_station_id": {"type": "string"},
        "charging_station_plug_id": {"type": "string"},
        "start_state_of_charge": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "target_state_of_charge": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
    },
    [
        "charging_station_id",
        "charging_station_plug_id",
        "start_state_of_charge",
    ],
)
SET_NAVIGATION_TOOL = _tool(
    "set_new_navigation",
    {"route_ids": {"type": "array", "items": {"type": "string"}}},
    ["route_ids"],
)
CALL_TOOL = _tool(
    "call_phone_by_number",
    {"phone_number": {"type": "string"}},
    ["phone_number"],
)
FULL_TOOLS = (
    CALENDAR_TOOL,
    LOCATION_TOOL,
    CHARGING_STATUS_TOOL,
    CURRENT_NAVIGATION_TOOL,
    ROUTES_TOOL,
    LOCAL_CHARGER_TOOL,
    CHARGING_TIME_TOOL,
    SET_NAVIGATION_TOOL,
    CALL_TOOL,
)


def _calendar_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "date": {"year": 2025, "month": 1, "day": 10},
            "meetings": [
                {
                    "start": {"hour": "15", "minute": "30"},
                    "duration": "30min",
                    "location": "Stuttgart",
                    "attendees": ["con_4304", "con_2224"],
                    "topic": "Partnership Discussion",
                },
                {
                    "start": {"hour": "17", "minute": "30"},
                    "duration": 30,
                    "location": "Barcelona",
                    "attendees": ["con_7515"],
                    "topic": "Financial Planning",
                },
            ],
        },
    }


def _charging_status_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "battery_capacity_kwh": 80.0,
            "max_charging_power_ac": 11,
            "max_charging_power_dc": 200,
            "state_of_charge": 20.0,
            "remaining_range": "101.0km",
        },
    }


def _route(
    route_id: str,
    start_id: str,
    destination_id: str,
    *,
    distance_km: float,
    duration_hours: int,
    duration_minutes: int,
    aliases: list[str],
) -> dict[str, Any]:
    return {
        "route_id": route_id,
        "start_id": start_id,
        "destination_id": destination_id,
        "name_via": "B592, B313",
        "distance_km": distance_km,
        "duration_hours": duration_hours,
        "duration_minutes": duration_minutes,
        "road_types": ["country road"],
        "includes_toll": False,
        "alias": aliases,
    }


def _direct_routes_result(
    *,
    destination_id: str = STUTTGART_ID,
    route_id: str = DIRECT_ROUTE_ID,
) -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                _route(
                    route_id,
                    MANNHEIM_ID,
                    destination_id,
                    distance_km=110.8,
                    duration_hours=1,
                    duration_minutes=25,
                    aliases=["fastest", "first", "shortest"],
                ),
                _route(
                    "rll_man_stu_174385",
                    MANNHEIM_ID,
                    destination_id,
                    distance_km=110.97,
                    duration_hours=1,
                    duration_minutes=25,
                    aliases=["second"],
                ),
            ]
        },
    }


def _to_charger_routes_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                _route(
                    TO_CHARGER_ROUTE_ID,
                    MANNHEIM_ID,
                    FASTNED_ID,
                    distance_km=7.43,
                    duration_hours=0,
                    duration_minutes=9,
                    aliases=["fastest", "first", "shortest"],
                ),
                _route(
                    "rlp_man_cha_474281",
                    MANNHEIM_ID,
                    FASTNED_ID,
                    distance_km=7.58,
                    duration_hours=0,
                    duration_minutes=10,
                    aliases=["second"],
                ),
            ]
        },
    }


def _from_charger_routes_result(
    *,
    destination_id: str = STUTTGART_ID,
    route_id: str = FROM_CHARGER_ROUTE_ID,
) -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                _route(
                    route_id,
                    FASTNED_ID,
                    destination_id,
                    distance_km=110.8,
                    duration_hours=1,
                    duration_minutes=25,
                    aliases=["fastest", "first", "shortest"],
                ),
                _route(
                    "rpl_cha_stu_875054",
                    FASTNED_ID,
                    destination_id,
                    distance_km=110.97,
                    duration_hours=1,
                    duration_minutes=25,
                    aliases=["second"],
                ),
            ]
        },
    }


def _navigation_state_result(
    *,
    active: bool = False,
    destination_id: str = STUTTGART_ID,
    direct_route_id: str = DIRECT_ROUTE_ID,
) -> dict[str, Any]:
    if not active:
        return {
            "status": "SUCCESS",
            "result": {
                "navigation_active": False,
                "waypoints_id": [],
                "routes_to_final_destination_id": [],
                "details": {"waypoints": [], "routes": []},
            },
        }
    direct_route = _direct_routes_result(
        destination_id=destination_id,
        route_id=direct_route_id,
    )["result"]["routes"][0]
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [MANNHEIM_ID, destination_id],
            "routes_to_final_destination_id": [direct_route_id],
            "details": {
                "waypoints": [
                    {"id": MANNHEIM_ID, "name": "Mannheim"},
                    {"id": destination_id, "name": "Stuttgart"},
                ],
                "routes": [direct_route],
            },
        },
    }


def _local_chargers_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "pois_found": [
                {
                    "id": "poi_cha_425198",
                    "name": "EnBW",
                    "category": "charging_stations",
                    "phone_number": "+49 142 5946143",
                    "corresponding_location_id": MANNHEIM_ID,
                    "charging_plugs": [
                        {
                            "plug_id": "plg_cha_839238",
                            "power_type": "AC",
                            "power_kw": 22,
                            "availability": "available",
                        },
                        {
                            "plug_id": "plg_cha_938199",
                            "power_type": "AC",
                            "power_kw": 22,
                            "availability": "maintenance",
                        },
                    ],
                },
                {
                    "id": FASTNED_ID,
                    "name": "Fastned",
                    "category": "charging_stations",
                    "phone_number": FASTNED_PHONE,
                    "corresponding_location_id": MANNHEIM_ID,
                    "charging_plugs": [
                        {
                            "plug_id": "plg_cha_961357",
                            "power_type": "DC",
                            "power_kw": 50,
                            "availability": "available",
                        },
                        {
                            "plug_id": FASTNED_PLUG_ID,
                            "power_type": "DC",
                            "power_kw": 300,
                            "availability": "occupied",
                        },
                    ],
                },
            ]
        },
    }


class Base98Client:
    """Base98's calculations and writes may not come from an action planner."""

    def __init__(self) -> None:
        self.intent_calls = 0
        self.action_calls = 0
        self.empty_intent = IntentDraft.model_validate(
            {
                "language": "en",
                "intent_kind": "information",
                "call_for_action": False,
                "goals": [],
                "explicit_slots": {},
            }
        )

    def generate(self, *, messages, response_model, critic=False):
        del messages, critic
        if response_model is IntentDraft:
            self.intent_calls += 1
            return SimpleNamespace(value=self.empty_intent)
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError(
                "Base98 reads, arithmetic branches, and writes must be deterministic"
            )
        raise AssertionError(f"unexpected response model: {response_model}")


class Base98ToolHarness:
    def __init__(
        self,
        *,
        calendar_result: dict[str, Any] | None = None,
        charging_time_result: dict[str, Any] | None = None,
        navigation_result: dict[str, Any] | None = None,
        meeting_location_id: str = STUTTGART_ID,
        direct_route_id: str = DIRECT_ROUTE_ID,
        from_charger_route_id: str = FROM_CHARGER_ROUTE_ID,
    ) -> None:
        self.meeting_location_id = meeting_location_id
        self.direct_route_id = direct_route_id
        self.from_charger_route_id = from_charger_route_id
        self.calendar_result = calendar_result or _calendar_result()
        self.charging_time_result = charging_time_result or {
            "status": "SUCCESS",
            "result": {"time_from_19_until_30_percent_soc": "3min"},
        }
        self.navigation_result = navigation_result or _navigation_state_result(
            destination_id=meeting_location_id,
            direct_route_id=direct_route_id,
        )
        self.calls: list[dict[str, Any]] = []

    def record(self, calls: tuple[dict[str, Any], ...]) -> None:
        self.calls.extend(deepcopy(list(calls)))

    def result_for(self, call: dict[str, Any]) -> dict[str, Any]:
        name = call["tool_name"]
        arguments = call["arguments"]
        if name == "get_entries_from_calendar":
            assert arguments == {"month": 1, "day": 10}
            return deepcopy(self.calendar_result)
        if name == "get_location_id_by_location_name":
            assert arguments == {"location": "Stuttgart"}
            return {
                "status": "SUCCESS",
                "result": {"id": self.meeting_location_id},
            }
        if name == "get_charging_specs_and_status":
            assert arguments == {}
            return _charging_status_result()
        if name == "get_current_navigation_state":
            assert arguments == {"detailed_information": True}
            return deepcopy(self.navigation_result)
        if name == "search_poi_at_location":
            # An availability filter would incorrectly hide the explicitly
            # acceptable occupied 300 kW plug.
            assert arguments == {
                "location_id": MANNHEIM_ID,
                "category_poi": "charging_stations",
            }
            return _local_chargers_result()
        if name == "get_routes_from_start_to_destination":
            endpoints = (arguments.get("start_id"), arguments.get("destination_id"))
            if endpoints == (MANNHEIM_ID, self.meeting_location_id):
                return _direct_routes_result(
                    destination_id=self.meeting_location_id,
                    route_id=self.direct_route_id,
                )
            if endpoints == (MANNHEIM_ID, FASTNED_ID):
                return _to_charger_routes_result()
            if endpoints == (FASTNED_ID, self.meeting_location_id):
                return _from_charger_routes_result(
                    destination_id=self.meeting_location_id,
                    route_id=self.from_charger_route_id,
                )
            raise AssertionError(f"unexpected Base98 route endpoints: {endpoints}")
        if name == "calculate_charging_time_by_soc":
            assert arguments == {
                "charging_station_id": FASTNED_ID,
                "charging_station_plug_id": FASTNED_PLUG_ID,
                "start_state_of_charge": 19,
                "target_state_of_charge": 30,
            }
            return deepcopy(self.charging_time_result)
        raise AssertionError(f"unexpected Base98 read tool: {name}")

    def count(self, name: str, arguments: dict[str, Any] | None = None) -> int:
        return sum(
            call["tool_name"] == name
            and (arguments is None or call["arguments"] == arguments)
            for call in self.calls
        )


def _runtime() -> tuple[CARGuardOrchestrator, Base98Client]:
    client = Base98Client()
    runtime = CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=48),
        client_factory=lambda session: client,
    )
    return runtime, client


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


def _results_event(
    context_id: str,
    message_id: str,
    calls: tuple[dict[str, Any], ...],
    harness: Base98ToolHarness,
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        tool_results=tuple(
            {
                "toolName": call["tool_name"],
                "content": harness.result_for(call),
            }
            for call in calls
        ),
    )


def _settle_reads(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    outbound: Any,
    harness: Base98ToolHarness,
    forbidden: frozenset[str],
    phase: str,
) -> Any:
    for step in range(24):
        if not outbound.tool_calls:
            return outbound
        calls = outbound.tool_calls
        names = {call["tool_name"] for call in calls}
        assert names.isdisjoint(forbidden), (
            f"{phase} emitted forbidden tools: {sorted(names & forbidden)}"
        )
        harness.record(calls)
        outbound = runtime.handle_event(
            _results_event(
                context_id,
                f"{context_id}-{phase}-result-{step}",
                calls,
                harness,
            )
        )
    raise AssertionError(f"{phase} did not settle within the step budget")


def _assert_called_once(
    harness: Base98ToolHarness, name: str, arguments: dict[str, Any]
) -> None:
    assert harness.count(name, arguments) == 1


def _start_to_station_choice(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    harness: Base98ToolHarness,
) -> Any:
    outbound = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-initial",
            INITIAL_REQUEST,
            tools=FULL_TOOLS,
        )
    )
    choice = _settle_reads(
        runtime,
        context_id=context_id,
        outbound=outbound,
        harness=harness,
        forbidden=frozenset(
            {
                "calculate_charging_time_by_soc",
                "set_new_navigation",
                "call_phone_by_number",
            }
        ),
        phase="station-choice",
    )

    _assert_called_once(harness, "get_entries_from_calendar", {"month": 1, "day": 10})
    _assert_called_once(harness, "get_charging_specs_and_status", {})
    _assert_called_once(
        harness,
        "search_poi_at_location",
        {"location_id": MANNHEIM_ID, "category_poi": "charging_stations"},
    )
    assert choice.text is not None
    lowered = choice.text.casefold()
    assert "fastned" in lowered
    assert "300" in lowered
    assert "occupied" in lowered
    return choice


def _continue_to_charging_report(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    harness: Base98ToolHarness,
) -> Any:
    _start_to_station_choice(runtime, context_id=context_id, harness=harness)
    selected = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-plug-selection",
            OCCUPIED_PLUG_SELECTION,
        )
    )
    report = _settle_reads(
        runtime,
        context_id=context_id,
        outbound=selected,
        harness=harness,
        forbidden=frozenset({"set_new_navigation", "call_phone_by_number"}),
        phase="charging-calculation",
    )

    _assert_called_once(
        harness,
        "get_location_id_by_location_name",
        {"location": "Stuttgart"},
    )
    _assert_called_once(
        harness,
        "get_routes_from_start_to_destination",
        {
            "start_id": MANNHEIM_ID,
            "destination_id": harness.meeting_location_id,
        },
    )
    _assert_called_once(
        harness,
        "get_routes_from_start_to_destination",
        {"start_id": MANNHEIM_ID, "destination_id": FASTNED_ID},
    )
    _assert_called_once(
        harness,
        "get_routes_from_start_to_destination",
        {
            "start_id": FASTNED_ID,
            "destination_id": harness.meeting_location_id,
        },
    )
    _assert_called_once(
        harness,
        "calculate_charging_time_by_soc",
        {
            "charging_station_id": FASTNED_ID,
            "charging_station_plug_id": FASTNED_PLUG_ID,
            "start_state_of_charge": 19,
            "target_state_of_charge": 30,
        },
    )
    calendar_index = next(
        index
        for index, call in enumerate(harness.calls)
        if call["tool_name"] == "get_entries_from_calendar"
    )
    location_index = next(
        index
        for index, call in enumerate(harness.calls)
        if call["tool_name"] == "get_location_id_by_location_name"
    )
    destination_route_indexes = [
        index
        for index, call in enumerate(harness.calls)
        if call["tool_name"] == "get_routes_from_start_to_destination"
        and call["arguments"].get("destination_id") == harness.meeting_location_id
    ]
    assert destination_route_indexes
    assert calendar_index < location_index < min(destination_route_indexes)
    assert len(harness.calls) == 8
    assert report.text is not None
    lowered = report.text.casefold()
    assert "minimum" in lowered and "maximum" in lowered
    assert re.search(r"\b3(?:\s+|-)?(?:minutes?|min)\b", lowered)
    assert re.search(r"\b31(?:\s+|-)?(?:minutes?|min)\b", lowered)
    return report


def _reach_pending_navigation_set(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    harness: Base98ToolHarness,
    navigation_request: str = NAVIGATION_REQUEST,
) -> Any:
    _continue_to_charging_report(runtime, context_id=context_id, harness=harness)
    snapshot = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-navigation-request",
            navigation_request,
        )
    )
    assert snapshot.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    harness.record(snapshot.tool_calls)
    pending = runtime.handle_event(
        _results_event(
            context_id,
            f"{context_id}-fresh-navigation-state",
            snapshot.tool_calls,
            harness,
        )
    )
    assert pending.tool_calls == (
        {
            "tool_name": "set_new_navigation",
            "arguments": {
                "route_ids": [TO_CHARGER_ROUTE_ID, harness.from_charger_route_id]
            },
        },
    )
    harness.record(pending.tool_calls)
    assert (
        harness.count("get_current_navigation_state", {"detailed_information": True})
        == 1
    )
    assert harness.count("set_new_navigation") == 1
    assert [call["tool_name"] for call in harness.calls[-2:]] == [
        "get_current_navigation_state",
        "set_new_navigation",
    ]
    return pending


def _navigation_success_payload(harness: Base98ToolHarness) -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_set": True,
            "start_id": MANNHEIM_ID,
            "waypoints": [
                MANNHEIM_ID,
                FASTNED_ID,
                harness.meeting_location_id,
            ],
            "destination_id": harness.meeting_location_id,
        },
    }


def _complete_navigation(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    harness: Base98ToolHarness,
    payload: dict[str, Any] | None = None,
) -> Any:
    return runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-navigation-result",
            tool_results=(
                {
                    "toolName": "set_new_navigation",
                    "content": payload or _navigation_success_payload(harness),
                },
            ),
        )
    )


def _read_current_navigation_after_completion(
    runtime: CARGuardOrchestrator,
    client: Base98Client,
    *,
    context_id: str,
    harness: Base98ToolHarness,
) -> Any:
    client.empty_intent = IntentDraft.model_validate(
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
        }
    )
    current = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-current-navigation",
            "What is my current navigation route?",
        )
    )
    assert current.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    harness.record(current.tool_calls)
    return runtime.handle_event(
        _results_event(
            context_id,
            f"{context_id}-current-navigation-result",
            current.tool_calls,
            harness,
        )
    )


def _active_meeting_navigation_result(harness: Base98ToolHarness) -> dict[str, Any]:
    first = _to_charger_routes_result()["result"]["routes"][0]
    second = _from_charger_routes_result(
        destination_id=harness.meeting_location_id,
        route_id=harness.from_charger_route_id,
    )["result"]["routes"][0]
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [
                MANNHEIM_ID,
                FASTNED_ID,
                harness.meeting_location_id,
            ],
            "routes_to_final_destination_id": [
                TO_CHARGER_ROUTE_ID,
                harness.from_charger_route_id,
            ],
            "details": {
                "waypoints": [
                    {"id": MANNHEIM_ID, "name": "Mannheim"},
                    {"id": FASTNED_ID, "name": "Fastned"},
                    {"id": harness.meeting_location_id, "name": "Stuttgart"},
                ],
                "routes": [first, second],
            },
        },
    }


def test_base98_exact_success_uses_occupied_fastest_plug_then_sets_and_calls() -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = "base98-exact-success"

    _reach_pending_navigation_set(runtime, context_id=context_id, harness=harness)
    completed = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-navigation-result",
            tool_results=(
                {
                    "toolName": "set_new_navigation",
                    "content": {
                        "status": "SUCCESS",
                        "result": {
                            "navigation_set": True,
                            "start_id": MANNHEIM_ID,
                            "waypoints": [
                                MANNHEIM_ID,
                                FASTNED_ID,
                                harness.meeting_location_id,
                            ],
                            "destination_id": harness.meeting_location_id,
                        },
                    },
                },
            ),
        )
    )
    assert completed.tool_calls == ()
    assert completed.text is not None
    assert "fastned" in completed.text.casefold()
    assert "stuttgart" in completed.text.casefold()

    called = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-call-request",
            CALL_REQUEST,
        )
    )
    assert called.tool_calls == (
        {
            "tool_name": "call_phone_by_number",
            "arguments": {"phone_number": FASTNED_PHONE},
        },
    )
    harness.record(called.tool_calls)
    assert called.terminal
    assert harness.count("call_phone_by_number") == 1
    assert len(harness.calls) == 11
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "utterance",
    POST_ANALYSIS_NAVIGATION_REQUESTS,
    ids=[
        "condition-context-please",
        "prior-request-current-confirmation",
        "need-you-multi-sentence",
        "please-without-first",
    ],
)
def test_base98_post_analysis_navigation_request_variants_set_fastest_segments(
    utterance: str,
) -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = f"base98-post-analysis-navigation-{len(utterance)}"

    pending = _reach_pending_navigation_set(
        runtime,
        context_id=context_id,
        harness=harness,
        navigation_request=utterance,
    )

    assert pending.tool_calls == (
        {
            "tool_name": "set_new_navigation",
            "arguments": {
                "route_ids": [TO_CHARGER_ROUTE_ID, harness.from_charger_route_id]
            },
        },
    )
    assert harness.count("set_new_navigation") == 1
    assert client.action_calls == 0


def test_base98_alternate_resolved_meeting_id_propagates_to_routes_and_set() -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness(
        meeting_location_id=ALTERNATE_STUTTGART_ID,
        direct_route_id=ALTERNATE_DIRECT_ROUTE_ID,
        from_charger_route_id=ALTERNATE_FROM_CHARGER_ROUTE_ID,
    )
    context_id = "base98-alternate-meeting-location-id"

    pending = _reach_pending_navigation_set(
        runtime,
        context_id=context_id,
        harness=harness,
    )

    _assert_called_once(
        harness,
        "get_location_id_by_location_name",
        {"location": "Stuttgart"},
    )
    assert (
        harness.count(
            "get_routes_from_start_to_destination",
            {"start_id": MANNHEIM_ID, "destination_id": STUTTGART_ID},
        )
        == 0
    )
    assert (
        harness.count(
            "get_routes_from_start_to_destination",
            {"start_id": FASTNED_ID, "destination_id": STUTTGART_ID},
        )
        == 0
    )
    assert pending.tool_calls == (
        {
            "tool_name": "set_new_navigation",
            "arguments": {
                "route_ids": [
                    TO_CHARGER_ROUTE_ID,
                    ALTERNATE_FROM_CHARGER_ROUTE_ID,
                ]
            },
        },
    )
    assert client.action_calls == 0


def test_base98_marker_only_navigation_success_remains_compatible() -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = "base98-marker-only-navigation-success"
    _reach_pending_navigation_set(runtime, context_id=context_id, harness=harness)
    _complete_navigation(
        runtime,
        context_id=context_id,
        harness=harness,
        payload={"status": "SUCCESS", "result": {"navigation_set": True}},
    )

    called = runtime.handle_event(
        _user_event(context_id, f"{context_id}-call", CALL_REQUEST)
    )

    assert called.tool_calls == (
        {
            "tool_name": "call_phone_by_number",
            "arguments": {"phone_number": FASTNED_PHONE},
        },
    )
    assert called.terminal
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "utterance",
    OCCUPIED_PLUG_SELECTION_VARIANTS,
    ids=["understand-occupied", "understand-and-accept", "accept-availability"],
)
def test_base98_explicit_occupied_plug_selection_variants_reach_analysis(
    utterance: str,
) -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = f"base98-selection-variant-{len(utterance)}"
    _start_to_station_choice(runtime, context_id=context_id, harness=harness)

    selected = runtime.handle_event(
        _user_event(context_id, f"{context_id}-selection", utterance)
    )
    report = _settle_reads(
        runtime,
        context_id=context_id,
        outbound=selected,
        harness=harness,
        forbidden=frozenset({"set_new_navigation", "call_phone_by_number"}),
        phase="selection-variant",
    )

    assert harness.count("calculate_charging_time_by_soc") == 1
    assert report.text is not None
    assert "minimum" in report.text.casefold()
    assert "maximum" in report.text.casefold()
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "exact_selection",
    OCCUPIED_PLUG_SELECTION_VARIANTS,
    ids=["understand-occupied", "understand-and-accept", "accept-availability"],
)
@pytest.mark.parametrize(
    "vague_selection",
    [
        (
            "I don't mind if it's occupied, please use the fastest charger for "
            "the time calculation."
        ),
        (
            "I do not mind if it's occupied, please use the fastest charger for "
            "the time calculation."
        ),
    ],
    ids=["dont-mind", "do-not-mind"],
)
def test_base98_dont_mind_vague_selection_prompts_before_exact_analysis(
    exact_selection: str, vague_selection: str
) -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = f"base98-vague-selection-{len(exact_selection)}"
    _start_to_station_choice(runtime, context_id=context_id, harness=harness)

    prompt = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-vague-selection",
            vague_selection,
        )
    )

    assert prompt.tool_calls == ()
    assert prompt.text is not None
    assert "exact" in prompt.text.casefold()
    assert "plug" in prompt.text.casefold()
    assert harness.count("calculate_charging_time_by_soc") == 0

    selected = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-exact-selection",
            exact_selection,
        )
    )
    report = _settle_reads(
        runtime,
        context_id=context_id,
        outbound=selected,
        harness=harness,
        forbidden=frozenset({"set_new_navigation", "call_phone_by_number"}),
        phase="exact-selection-after-prompt",
    )

    assert harness.count("calculate_charging_time_by_soc") == 1
    assert report.text is not None
    assert "minimum" in report.text.casefold()
    assert "maximum" in report.text.casefold()
    assert client.action_calls == 0


def test_base98_still_want_vague_selection_prompts_before_exact_analysis() -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = "base98-still-want-vague-selection"
    _start_to_station_choice(runtime, context_id=context_id, harness=harness)

    prompt = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-vague-selection",
            (
                "I still want to select the fastest charger for the time "
                "calculation, even if it's currently occupied."
            ),
        )
    )

    assert prompt.tool_calls == ()
    assert prompt.text is not None
    assert "exact" in prompt.text.casefold()
    assert "plug" in prompt.text.casefold()
    assert harness.count("calculate_charging_time_by_soc") == 0

    selected = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-exact-selection",
            OCCUPIED_PLUG_SELECTION_VARIANTS[2],
        )
    )
    report = _settle_reads(
        runtime,
        context_id=context_id,
        outbound=selected,
        harness=harness,
        forbidden=frozenset({"set_new_navigation", "call_phone_by_number"}),
        phase="still-want-exact-selection",
    )

    assert harness.count("calculate_charging_time_by_soc") == 1
    assert report.text is not None
    assert "minimum" in report.text.casefold()
    assert "maximum" in report.text.casefold()
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "I would select the 300 kW plug at Fastned for the calculation, even "
            "though it is occupied."
        ),
        (
            "I do not select the 300 kW plug at Fastned for the calculation, even "
            "though it is occupied."
        ),
        (
            "I don't want to use the 300 kW plug at Fastned for the calculation, "
            "even though it is occupied."
        ),
        (
            "I don't use the 300 kW plug at Fastned for the calculation, even "
            "though it is occupied."
        ),
        (
            "I don't select the 300 kW plug at Fastned for the calculation, even "
            "though it is occupied."
        ),
        (
            "I might select the 300 kW plug at Fastned for the calculation, even "
            "though it is occupied."
        ),
        (
            "I might want to select the 300 kW plug at Fastned for the calculation, "
            "even though it is occupied."
        ),
        (
            "I would consider selecting the 300 kW plug at Fastned for the "
            "calculation, even though it is occupied."
        ),
        (
            "I was thinking about selecting the 300 kW plug at Fastned for the "
            "calculation, even though it is occupied."
        ),
    ],
    ids=[
        "first-person-description",
        "do-not-select",
        "dont-want",
        "dont-use",
        "dont-select",
        "might-select",
        "might-want",
        "would-consider",
        "was-thinking",
    ],
)
def test_base98_non_request_plug_selections_never_calculate(utterance: str) -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = f"base98-selection-non-request-{len(utterance)}"
    _start_to_station_choice(runtime, context_id=context_id, harness=harness)

    blocked = runtime.handle_event(
        _user_event(context_id, f"{context_id}-selection", utterance)
    )

    assert blocked.tool_calls == ()
    assert not blocked.terminal
    assert harness.count("calculate_charging_time_by_soc") == 0
    assert harness.count("set_new_navigation") == 0
    assert harness.count("call_phone_by_number") == 0
    assert client.action_calls == 0


def test_base98_wrong_day_calendar_never_calculates_sets_or_calls() -> None:
    runtime, client = _runtime()
    malformed_calendar = _calendar_result()
    malformed_calendar["result"]["date"]["day"] = 11
    harness = Base98ToolHarness(calendar_result=malformed_calendar)
    context_id = "base98-wrong-day-calendar"

    first = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-initial",
            INITIAL_REQUEST,
            tools=FULL_TOOLS,
        )
    )
    blocked = _settle_reads(
        runtime,
        context_id=context_id,
        outbound=first,
        harness=harness,
        forbidden=frozenset(
            {
                "calculate_charging_time_by_soc",
                "set_new_navigation",
                "call_phone_by_number",
            }
        ),
        phase="bad-calendar",
    )
    assert harness.count("get_entries_from_calendar") == 1
    assert harness.count("get_location_id_by_location_name") == 0
    assert blocked.tool_calls == ()

    attempted = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-conditional-navigation",
            NAVIGATION_REQUEST,
        )
    )
    _settle_reads(
        runtime,
        context_id=context_id,
        outbound=attempted,
        harness=harness,
        forbidden=frozenset(
            {
                "calculate_charging_time_by_soc",
                "set_new_navigation",
                "call_phone_by_number",
            }
        ),
        phase="bad-calendar-followup",
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base98_conflicting_charging_time_never_authorizes_navigation() -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness(
        charging_time_result={
            "status": "SUCCESS",
            "result": {"time_from_19_until_30_percent_soc": ["3min", "13min"]},
        }
    )
    context_id = "base98-bad-charging-calculation"

    _start_to_station_choice(runtime, context_id=context_id, harness=harness)
    selected = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-plug-selection",
            OCCUPIED_PLUG_SELECTION,
        )
    )
    _settle_reads(
        runtime,
        context_id=context_id,
        outbound=selected,
        harness=harness,
        forbidden=frozenset({"set_new_navigation", "call_phone_by_number"}),
        phase="bad-calculation",
    )
    assert harness.count("calculate_charging_time_by_soc") >= 1

    attempted = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-conditional-navigation",
            NAVIGATION_REQUEST,
        )
    )
    _settle_reads(
        runtime,
        context_id=context_id,
        outbound=attempted,
        harness=harness,
        forbidden=frozenset({"set_new_navigation", "call_phone_by_number"}),
        phase="bad-calculation-followup",
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base98_call_waits_for_successful_navigation_result() -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = "base98-call-after-navigation"
    pending = _reach_pending_navigation_set(
        runtime, context_id=context_id, harness=harness
    )

    premature = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-premature-call",
            CALL_REQUEST,
        )
    )
    assert premature.tool_calls == ()
    assert "waiting" in (premature.text or "").casefold()

    failed = runtime.handle_event(
        InboundEvent(
            context_id=context_id,
            message_id=f"{context_id}-navigation-failure",
            tool_results=(
                {
                    "toolName": pending.tool_calls[0]["tool_name"],
                    "content": {
                        "status": "FAILURE",
                        "error": "navigation service unavailable",
                    },
                },
            ),
        )
    )
    assert failed.tool_calls == ()

    retried = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-call-after-failure",
            CALL_REQUEST,
        )
    )
    assert retried.tool_calls == ()
    assert not retried.terminal
    assert harness.count("call_phone_by_number") == 0
    assert client.action_calls == 0


def test_base98_active_navigation_blocks_new_navigation_and_call() -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness(navigation_result=_navigation_state_result(active=True))
    context_id = "base98-active-navigation"
    _continue_to_charging_report(runtime, context_id=context_id, harness=harness)

    snapshot = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-navigation-request",
            NAVIGATION_REQUEST,
        )
    )
    assert snapshot.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    harness.record(snapshot.tool_calls)
    blocked = runtime.handle_event(
        _results_event(
            context_id,
            f"{context_id}-active-navigation-result",
            snapshot.tool_calls,
            harness,
        )
    )
    assert blocked.tool_calls == ()
    assert harness.count("set_new_navigation") == 0

    call_attempt = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-call-request",
            CALL_REQUEST,
        )
    )
    blocked_call = _settle_reads(
        runtime,
        context_id=context_id,
        outbound=call_attempt,
        harness=harness,
        forbidden=frozenset({"set_new_navigation", "call_phone_by_number"}),
        phase="active-navigation-call",
    )
    assert not blocked_call.terminal
    assert harness.count("call_phone_by_number") == 0
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "utterance",
    [NEGATED_NAVIGATION_REQUEST, DESCRIPTIVE_NAVIGATION_REQUEST],
    ids=["negated", "third-person-description"],
)
def test_base98_non_request_navigation_followups_never_set(
    utterance: str,
) -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = f"base98-navigation-non-request-{utterance[:8]}"
    _continue_to_charging_report(runtime, context_id=context_id, harness=harness)

    attempted = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-navigation-followup",
            utterance,
        )
    )
    blocked = _settle_reads(
        runtime,
        context_id=context_id,
        outbound=attempted,
        harness=harness,
        forbidden=frozenset({"set_new_navigation", "call_phone_by_number"}),
        phase="non-request-navigation",
    )

    assert blocked.tool_calls == ()
    assert not blocked.terminal
    assert harness.count("set_new_navigation") == 0
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "utterance",
    [NEGATED_CALL_REQUEST, DESCRIPTIVE_CALL_REQUEST],
    ids=["negated", "third-person-description"],
)
def test_base98_non_request_call_followups_never_dial(utterance: str) -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = f"base98-call-non-request-{utterance[:8]}"
    _reach_pending_navigation_set(runtime, context_id=context_id, harness=harness)
    _complete_navigation(runtime, context_id=context_id, harness=harness)

    blocked = runtime.handle_event(
        _user_event(context_id, f"{context_id}-call-followup", utterance)
    )

    assert blocked.tool_calls == ()
    assert not blocked.terminal
    assert harness.count("call_phone_by_number") == 0
    assert client.action_calls == 0


def test_base98_fresh_contradictory_navigation_read_invalidates_provider_call() -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = "base98-stale-navigation-completion"
    _reach_pending_navigation_set(runtime, context_id=context_id, harness=harness)
    _complete_navigation(runtime, context_id=context_id, harness=harness)
    observed = _read_current_navigation_after_completion(
        runtime,
        client,
        context_id=context_id,
        harness=harness,
    )
    assert observed.tool_calls == ()

    blocked = runtime.handle_event(
        _user_event(context_id, f"{context_id}-call", CALL_REQUEST)
    )

    assert blocked.tool_calls == ()
    assert not blocked.terminal
    assert harness.count("call_phone_by_number") == 0
    assert client.action_calls == 0


def test_base98_matching_post_navigation_read_keeps_provider_call_authorized() -> None:
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = "base98-current-navigation-completion"
    _reach_pending_navigation_set(runtime, context_id=context_id, harness=harness)
    _complete_navigation(runtime, context_id=context_id, harness=harness)
    harness.navigation_result = _active_meeting_navigation_result(harness)
    observed = _read_current_navigation_after_completion(
        runtime,
        client,
        context_id=context_id,
        harness=harness,
    )
    assert observed.tool_calls == ()

    called = runtime.handle_event(
        _user_event(context_id, f"{context_id}-call", CALL_REQUEST)
    )

    assert called.tool_calls == (
        {
            "tool_name": "call_phone_by_number",
            "arguments": {"phone_number": FASTNED_PHONE},
        },
    )
    assert called.terminal
    assert client.action_calls == 0


def test_base98_contradictory_navigation_success_payload_never_authorizes_call() -> (
    None
):
    runtime, client = _runtime()
    harness = Base98ToolHarness()
    context_id = "base98-contradictory-navigation-success"
    _reach_pending_navigation_set(runtime, context_id=context_id, harness=harness)
    contradictory = _navigation_success_payload(harness)
    contradictory["result"]["waypoints"] = [MANNHEIM_ID, STUTTGART_ID]
    contradictory["result"]["destination_id"] = ALTERNATE_STUTTGART_ID
    _complete_navigation(
        runtime,
        context_id=context_id,
        harness=harness,
        payload=contradictory,
    )

    blocked = runtime.handle_event(
        _user_event(context_id, f"{context_id}-call", CALL_REQUEST)
    )

    assert blocked.tool_calls == ()
    assert not blocked.terminal
    assert harness.count("call_phone_by_number") == 0
    assert client.action_calls == 0
