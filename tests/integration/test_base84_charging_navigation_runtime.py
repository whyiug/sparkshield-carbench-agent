from __future__ import annotations

import json
from copy import deepcopy
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


WARSAW_ID = "loc_war_429257"
HAMBURG_ID = "loc_ham_166665"
IONITY_ID = "poi_cha_948882"
IONITY_DC_PLUG_ID = "plg_cha_947862"
WARSAW_HAMBURG_SECOND_ROUTE_ID = "rll_war_ham_553572"
WARSAW_IONITY_FASTEST_ROUTE_ID = "rlp_war_cha_224861"
IONITY_HAMBURG_SECOND_ROUTE_ID = "rpl_cha_ham_429250"

INITIAL_REQUEST = (
    "I need to get to Hamburg, but I'm not sure if I have enough battery. "
    "Can you check that for me? Please don't start navigation yet."
)
FORMAL_INITIAL_REQUEST = (
    "I need to get to Hamburg, but I'm not sure if I have enough battery. "
    "Can you check that for me?"
)
LOCAL_CHARGER_REQUEST = (
    "No, I want to find a charging station nearby in Warsaw first, before I "
    "even start the trip."
)
CHARGING_TIME_REQUEST = (
    "I want to go to the Ionity charging station. How long will it take to "
    "charge from my current 35 percent to 95 percent?"
)
FORMAL_CHARGING_TIME_REQUEST = (
    "Okay, I want to go to the Ionity charging station. Can you tell me how "
    "long it will take to charge to 95% there?"
)
SET_NAVIGATION_REQUEST = (
    "Yes, please set up navigation to Hamburg, but with the charging stop at "
    "Ionity first. For the route to Ionity, I want the fastest option. And for "
    "the route to Hamburg, I want the second route option, the one via B432, "
    "B132."
)
FORMAL_SET_NAVIGATION_REQUEST = (
    "Okay, let's set up navigation to Hamburg with the charging stop at "
    "Ionity. For the charging station, I want the fastest route. And for "
    "Hamburg, please use the second route option, the one via B432, B132."
)
FORMAL_SEQUENCED_NAVIGATION_REQUEST = (
    "Okay, that sounds good. Please set up the navigation now. I want the "
    "fastest route to the Ionity charging station, and then for the route to "
    "Hamburg, please use the second option, the one via B432, B132."
)
FORMAL_AFTER_THAT_NAVIGATION_REQUEST = (
    "Okay, that sounds good. Can you set up the navigation now? I want to go "
    "to the Ionity charging station first, taking the fastest route there. "
    "After that, please navigate to Hamburg, but for that part, I want to take "
    "the second route option, the one via B432, B132."
)
POLICY = (
    "Use metric units. The start of a new route must be the current car "
    "location. Do not start navigation until the user asks. "
    'CURRENT_LOCATION={"id":"loc_war_429257","name":"Warsaw"}'
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
LOCAL_POI_TOOL = _tool(
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
        "filters": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "any::currently_open",
                    "charging_stations::has_available_plug",
                    "charging_stations::has_dc_plug",
                ],
            },
        },
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
        "target_state_of_charge",
    ],
)
SET_NAVIGATION_TOOL = _tool(
    "set_new_navigation",
    {"route_ids": {"type": "array", "items": {"type": "string"}}},
    ["route_ids"],
)
FULL_TOOLS = (
    LOCATION_TOOL,
    CHARGING_STATUS_TOOL,
    CURRENT_NAVIGATION_TOOL,
    ROUTES_TOOL,
    LOCAL_POI_TOOL,
    CHARGING_TIME_TOOL,
    SET_NAVIGATION_TOOL,
)


def _intent(
    operation: str,
    desired_outcome: dict[str, Any],
    *,
    action: bool,
) -> IntentDraft:
    return IntentDraft.model_validate(
        {
            "language": "en",
            "intent_kind": "action" if action else "information",
            "call_for_action": action,
            "goals": [
                {
                    "semantic_operation": operation,
                    "desired_outcome": desired_outcome,
                }
            ],
            "explicit_slots": desired_outcome,
        }
    )


class Base84Client:
    def __init__(self) -> None:
        self.intent_calls = 0
        self.action_calls = 0

    @staticmethod
    def _latest_user_text(messages: list[dict[str, Any]]) -> str:
        return next(
            (
                str(message.get("content", ""))
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        ).casefold()

    def generate(self, *, messages, response_model, critic=False):
        del critic
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError(
                "Base84 actions must be evidence-bound and deterministic"
            )
        if response_model is not IntentDraft:
            raise AssertionError(f"unexpected response model: {response_model}")

        self.intent_calls += 1
        text = self._latest_user_text(messages)
        if "set up navigation" in text:
            intent = _intent(
                "set_new_navigation",
                {
                    "destination_name": "Hamburg",
                    "intermediate_stop_name": "Ionity",
                    "first_segment_selection": "fastest",
                    "second_segment_selection": "second",
                },
                action=True,
            )
        elif "95 percent" in text:
            intent = _intent(
                "calculate_charging_time_by_soc",
                {
                    "charging_station_name": "Ionity",
                    "start_state_of_charge": 35,
                    "target_state_of_charge": 95,
                },
                action=False,
            )
        elif "nearby in warsaw" in text:
            intent = _intent(
                "search_poi_at_location",
                {
                    "location_name": "Warsaw",
                    "category": "charging_stations",
                    "filters": ["charging_stations::has_available_plug"],
                },
                action=False,
            )
        else:
            intent = _intent(
                "assess_battery_charge_trip_range",
                {"destination_name": "Hamburg"},
                action=False,
            )
        return SimpleNamespace(value=intent)


def _runtime() -> tuple[CARGuardOrchestrator, Base84Client]:
    client = Base84Client()
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
    tools: tuple[dict[str, Any], ...] | None = None,
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        system_policy=POLICY if tools is not None else None,
        user_text=text,
        live_tools=tools,
    )


def _result_event(
    context_id: str,
    message_id: str,
    results: list[tuple[str, Any]],
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        tool_results=tuple(
            {"toolName": tool_name, "content": content}
            for tool_name, content in results
        ),
    )


def charging_status() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "battery_capacity_kwh": 70.0,
            "max_charging_power_ac": 11,
            "max_charging_power_dc": 150,
            "state_of_charge": 35.0,
            "remaining_range": "155.0km",
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
    active_route = deepcopy(hamburg_routes()["result"]["routes"][0])
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [WARSAW_ID, HAMBURG_ID],
            "routes_to_final_destination_id": [active_route["route_id"]],
            "details": {
                "waypoints": [
                    {"id": WARSAW_ID, "name": "Warsaw"},
                    {"id": HAMBURG_ID, "name": "Hamburg"},
                ],
                "routes": [active_route],
            },
        },
    }


def hamburg_routes() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": "rll_war_ham_503836",
                    "start_id": WARSAW_ID,
                    "destination_id": HAMBURG_ID,
                    "name_via": "L419, K819, K221",
                    "distance_km": 899.13,
                    "duration_hours": 11,
                    "duration_minutes": 7,
                    "road_types": ["country road", "highway", "urban"],
                    "includes_toll": False,
                    "alias": ["fastest", "first"],
                },
                {
                    "route_id": WARSAW_HAMBURG_SECOND_ROUTE_ID,
                    "start_id": WARSAW_ID,
                    "destination_id": HAMBURG_ID,
                    "name_via": "B432, B132",
                    "distance_km": 895.38,
                    "duration_hours": 11,
                    "duration_minutes": 9,
                    "road_types": ["country road"],
                    "includes_toll": False,
                    "alias": ["second"],
                },
                {
                    "route_id": "rll_war_ham_618038",
                    "start_id": WARSAW_ID,
                    "destination_id": HAMBURG_ID,
                    "name_via": "L257, A82, A10",
                    "distance_km": 883.39,
                    "duration_hours": 11,
                    "duration_minutes": 10,
                    "road_types": ["highway", "urban"],
                    "includes_toll": False,
                    "alias": ["third", "shortest"],
                },
            ]
        },
    }


def warsaw_chargers() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "pois_found": [
                {
                    "id": IONITY_ID,
                    "name": "Ionity",
                    "category": "charging_stations",
                    "position": {
                        "longitude": 21.020239112441047,
                        "latitude": 52.21042202453828,
                    },
                    "opening_hours": "00:00h - 24:00h",
                    "phone_number": "+49 753 8042345",
                    "corresponding_location_id": WARSAW_ID,
                    "charging_plugs": [
                        {
                            "plug_id": "plg_cha_664037",
                            "power_type": "AC",
                            "power_kw": 11,
                            "availability": "available",
                        },
                        {
                            "plug_id": IONITY_DC_PLUG_ID,
                            "power_type": "DC",
                            "power_kw": 100,
                            "availability": "available",
                        },
                        {
                            "plug_id": "plg_cha_541904",
                            "power_type": "DC",
                            "power_kw": 50,
                            "availability": "occupied",
                        },
                    ],
                },
                {
                    "id": "poi_cha_483074",
                    "name": "Tesla Supercharger",
                    "category": "charging_stations",
                    "position": {
                        "longitude": 20.98920912293031,
                        "latitude": 52.17461193916646,
                    },
                    "opening_hours": "00:00h - 24:00h",
                    "phone_number": "+49 483 1667515",
                    "corresponding_location_id": WARSAW_ID,
                    "charging_plugs": [
                        {
                            "plug_id": "plg_cha_226343",
                            "power_type": "AC",
                            "power_kw": 11,
                            "availability": "available",
                        },
                        {
                            "plug_id": "plg_cha_522841",
                            "power_type": "DC",
                            "power_kw": 350,
                            "availability": "occupied",
                        },
                    ],
                },
            ]
        },
    }


def warsaw_ionity_routes() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": WARSAW_IONITY_FASTEST_ROUTE_ID,
                    "start_id": WARSAW_ID,
                    "destination_id": IONITY_ID,
                    "name_via": "A60",
                    "distance_km": 2.73,
                    "duration_hours": 0,
                    "duration_minutes": 4,
                    "road_types": ["highway"],
                    "includes_toll": False,
                    "alias": ["fastest", "first"],
                },
                {
                    "route_id": "rlp_war_cha_455230",
                    "start_id": WARSAW_ID,
                    "destination_id": IONITY_ID,
                    "name_via": "K174, K265, K675",
                    "distance_km": 2.71,
                    "duration_hours": 0,
                    "duration_minutes": 4,
                    "road_types": ["country road", "highway", "urban"],
                    "includes_toll": False,
                    "alias": ["second", "shortest"],
                },
            ]
        },
    }


def ionity_hamburg_routes() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": "rpl_cha_ham_286806",
                    "start_id": IONITY_ID,
                    "destination_id": HAMBURG_ID,
                    "name_via": "L419, K819, K221",
                    "distance_km": 899.13,
                    "duration_hours": 11,
                    "duration_minutes": 7,
                    "road_types": ["country road", "highway", "urban"],
                    "includes_toll": False,
                    "alias": ["fastest", "first"],
                    "base_route_id": "rll_war_ham_503836",
                },
                {
                    "route_id": IONITY_HAMBURG_SECOND_ROUTE_ID,
                    "start_id": IONITY_ID,
                    "destination_id": HAMBURG_ID,
                    "name_via": "B432, B132",
                    "distance_km": 895.38,
                    "duration_hours": 11,
                    "duration_minutes": 9,
                    "road_types": ["country road"],
                    "includes_toll": False,
                    "alias": ["second"],
                    "base_route_id": WARSAW_HAMBURG_SECOND_ROUTE_ID,
                },
                {
                    "route_id": "rpl_cha_ham_286696",
                    "start_id": IONITY_ID,
                    "destination_id": HAMBURG_ID,
                    "name_via": "L257, A82, A10",
                    "distance_km": 883.39,
                    "duration_hours": 11,
                    "duration_minutes": 10,
                    "road_types": ["highway", "urban"],
                    "includes_toll": False,
                    "alias": ["third", "shortest"],
                    "base_route_id": "rll_war_ham_618038",
                },
            ]
        },
    }


def charging_time_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {"time_from_35_until_95_percent_soc": "28min"},
    }


def _calls_by_name(outbound: Any) -> dict[str, dict[str, Any]]:
    calls = {call["tool_name"]: call for call in outbound.tool_calls}
    assert len(calls) == len(outbound.tool_calls), "unexpected duplicate tool calls"
    return calls


def _assert_one_call(outbound: Any, name: str, arguments: dict[str, Any]) -> None:
    assert outbound.tool_calls == ({"tool_name": name, "arguments": arguments},)


def _append_calls(transcript: list[dict[str, Any]], outbound: Any) -> None:
    transcript.extend(deepcopy(list(outbound.tool_calls)))


def test_base84_formal_initial_information_request_starts_parallel_reads() -> None:
    runtime, client = _runtime()
    outbound = runtime.handle_event(
        _user_event(
            "base84-formal-initial",
            "base84-formal-initial-user",
            FORMAL_INITIAL_REQUEST,
            tools=FULL_TOOLS,
        )
    )

    calls = _calls_by_name(outbound)
    assert set(calls) == {
        "get_location_id_by_location_name",
        "get_charging_specs_and_status",
    }
    assert calls["get_location_id_by_location_name"]["arguments"] == {
        "location": "Hamburg"
    }
    assert calls["get_charging_specs_and_status"]["arguments"] == {}
    assert client.action_calls == 0


def _run_to_charger_search(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    transcript: list[dict[str, Any]] | None = None,
) -> Any:
    first = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-initial",
            INITIAL_REQUEST,
            tools=FULL_TOOLS,
        )
    )
    calls = _calls_by_name(first)
    assert set(calls) == {
        "get_location_id_by_location_name",
        "get_charging_specs_and_status",
    }
    assert calls["get_location_id_by_location_name"]["arguments"] == {
        "location": "Hamburg"
    }
    assert calls["get_charging_specs_and_status"]["arguments"] == {}
    if transcript is not None:
        _append_calls(transcript, first)

    route = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-identity-status",
            [
                (
                    "get_location_id_by_location_name",
                    {"status": "SUCCESS", "result": {"id": HAMBURG_ID}},
                ),
                ("get_charging_specs_and_status", charging_status()),
            ],
        )
    )
    _assert_one_call(
        route,
        "get_routes_from_start_to_destination",
        {"start_id": WARSAW_ID, "destination_id": HAMBURG_ID},
    )
    if transcript is not None:
        _append_calls(transcript, route)

    assessment = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-hamburg-routes",
            [("get_routes_from_start_to_destination", hamburg_routes())],
        )
    )
    assert assessment.tool_calls == ()
    assert assessment.text is not None
    assert "155" in assessment.text
    assert "not enough" in assessment.text.casefold() or "need to charge" in (
        assessment.text.casefold()
    )

    search = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-local-search",
            LOCAL_CHARGER_REQUEST,
        )
    )
    _assert_one_call(
        search,
        "search_poi_at_location",
        {
            "location_id": WARSAW_ID,
            "category_poi": "charging_stations",
            "filters": ["charging_stations::has_available_plug"],
        },
    )
    if transcript is not None:
        _append_calls(transcript, search)
    return search


def _run_to_charger_presentation(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    transcript: list[dict[str, Any]] | None = None,
) -> Any:
    _run_to_charger_search(
        runtime,
        context_id=context_id,
        transcript=transcript,
    )
    presented = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-charger-results",
            [("search_poi_at_location", warsaw_chargers())],
        )
    )
    assert presented.tool_calls == ()
    assert presented.text is not None
    assert "Ionity" in presented.text
    assert "100" in presented.text and "DC" in presented.text
    return presented


def _request_charging_time(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    transcript: list[dict[str, Any]] | None = None,
    request_text: str = CHARGING_TIME_REQUEST,
) -> Any:
    requested = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-charging-time",
            request_text,
        )
    )
    calls = _calls_by_name(requested)
    assert set(calls) == {
        "calculate_charging_time_by_soc",
        "get_routes_from_start_to_destination",
    }
    assert calls["calculate_charging_time_by_soc"]["arguments"] == {
        "charging_station_id": IONITY_ID,
        "charging_station_plug_id": IONITY_DC_PLUG_ID,
        "start_state_of_charge": 35,
        "target_state_of_charge": 95,
    }
    assert calls["get_routes_from_start_to_destination"]["arguments"] == {
        "start_id": WARSAW_ID,
        "destination_id": IONITY_ID,
    }
    if transcript is not None:
        _append_calls(transcript, requested)
    return requested


def test_base84_formal_target_percent_uses_verified_current_soc() -> None:
    runtime, _ = _runtime()
    context_id = "base84-formal-charge-target"
    _run_to_charger_presentation(runtime, context_id=context_id)

    _request_charging_time(
        runtime,
        context_id=context_id,
        request_text=FORMAL_CHARGING_TIME_REQUEST,
    )


def _run_to_charging_answer(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    transcript: list[dict[str, Any]] | None = None,
) -> Any:
    _run_to_charger_presentation(
        runtime,
        context_id=context_id,
        transcript=transcript,
    )
    _request_charging_time(
        runtime,
        context_id=context_id,
        transcript=transcript,
    )
    answer = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-charging-and-route",
            [
                (
                    "get_routes_from_start_to_destination",
                    warsaw_ionity_routes(),
                ),
                ("calculate_charging_time_by_soc", charging_time_result()),
            ],
        )
    )
    assert answer.tool_calls == ()
    assert answer.text is not None
    assert "28" in answer.text and "95" in answer.text
    return answer


def _call_signature(call: dict[str, Any]) -> tuple[str, str]:
    return call["tool_name"], json.dumps(call["arguments"], sort_keys=True)


def test_base84_canonical_nine_actions_end_in_exact_multisegment_navigation() -> None:
    runtime, client = _runtime()
    context_id = "base84-canonical"
    transcript: list[dict[str, Any]] = []
    _run_to_charging_answer(
        runtime,
        context_id=context_id,
        transcript=transcript,
    )

    navigation_state = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-set-navigation",
            SET_NAVIGATION_REQUEST,
        )
    )
    _assert_one_call(
        navigation_state,
        "get_current_navigation_state",
        {"detailed_information": True},
    )
    _append_calls(transcript, navigation_state)

    second_segment = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-fresh-navigation",
            [("get_current_navigation_state", inactive_navigation_state())],
        )
    )
    _assert_one_call(
        second_segment,
        "get_routes_from_start_to_destination",
        {"start_id": IONITY_ID, "destination_id": HAMBURG_ID},
    )
    _append_calls(transcript, second_segment)

    start = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-second-segment",
            [("get_routes_from_start_to_destination", ionity_hamburg_routes())],
        )
    )
    _assert_one_call(
        start,
        "set_new_navigation",
        {
            "route_ids": [
                WARSAW_IONITY_FASTEST_ROUTE_ID,
                IONITY_HAMBURG_SECOND_ROUTE_ID,
            ]
        },
    )
    _append_calls(transcript, start)

    completed = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-navigation-result",
            [
                (
                    "set_new_navigation",
                    {
                        "status": "SUCCESS",
                        "result": {
                            "navigation_set": True,
                            "start_id": WARSAW_ID,
                            "waypoints": [WARSAW_ID, IONITY_ID, HAMBURG_ID],
                            "destination_id": HAMBURG_ID,
                        },
                    },
                )
            ],
        )
    )
    assert completed.tool_calls == ()
    assert completed.text is not None and "navigation" in completed.text.casefold()

    expected = [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Hamburg"},
        },
        {"tool_name": "get_charging_specs_and_status", "arguments": {}},
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {"start_id": WARSAW_ID, "destination_id": HAMBURG_ID},
        },
        {
            "tool_name": "search_poi_at_location",
            "arguments": {
                "location_id": WARSAW_ID,
                "category_poi": "charging_stations",
                "filters": ["charging_stations::has_available_plug"],
            },
        },
        {
            "tool_name": "calculate_charging_time_by_soc",
            "arguments": {
                "charging_station_id": IONITY_ID,
                "charging_station_plug_id": IONITY_DC_PLUG_ID,
                "start_state_of_charge": 35,
                "target_state_of_charge": 95,
            },
        },
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {"start_id": WARSAW_ID, "destination_id": IONITY_ID},
        },
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {"start_id": IONITY_ID, "destination_id": HAMBURG_ID},
        },
        {
            "tool_name": "set_new_navigation",
            "arguments": {
                "route_ids": [
                    WARSAW_IONITY_FASTEST_ROUTE_ID,
                    IONITY_HAMBURG_SECOND_ROUTE_ID,
                ]
            },
        },
    ]
    assert len(transcript) == 9
    assert sorted(map(_call_signature, transcript)) == sorted(
        map(_call_signature, expected)
    )
    set_index = next(
        index
        for index, call in enumerate(transcript)
        if call["tool_name"] == "set_new_navigation"
    )
    calculation_index = next(
        index
        for index, call in enumerate(transcript)
        if call["tool_name"] == "calculate_charging_time_by_soc"
    )
    navigation_read_index = next(
        index
        for index, call in enumerate(transcript)
        if call["tool_name"] == "get_current_navigation_state"
    )
    second_segment_index = next(
        index
        for index, call in enumerate(transcript)
        if call["tool_name"] == "get_routes_from_start_to_destination"
        and call["arguments"] == {"start_id": IONITY_ID, "destination_id": HAMBURG_ID}
    )
    assert calculation_index < navigation_read_index < second_segment_index < set_index
    assert client.action_calls == 0


def test_base84_formal_multisegment_wording_starts_fresh_navigation_read() -> None:
    runtime, _ = _runtime()
    context_id = "base84-formal-navigation-request"
    _run_to_charging_answer(runtime, context_id=context_id)

    navigation_state = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-set-navigation",
            FORMAL_SET_NAVIGATION_REQUEST,
        )
    )

    _assert_one_call(
        navigation_state,
        "get_current_navigation_state",
        {"detailed_information": True},
    )


def test_base84_sequenced_multileg_wording_starts_fresh_navigation_read() -> None:
    runtime, _ = _runtime()
    context_id = "base84-formal-sequenced-navigation"
    _run_to_charging_answer(runtime, context_id=context_id)

    navigation_state = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-set-navigation",
            FORMAL_SEQUENCED_NAVIGATION_REQUEST,
        )
    )

    _assert_one_call(
        navigation_state,
        "get_current_navigation_state",
        {"detailed_information": True},
    )


def test_base84_after_that_multileg_wording_starts_fresh_navigation_read() -> None:
    runtime, _ = _runtime()
    context_id = "base84-formal-after-that-navigation"
    _run_to_charging_answer(runtime, context_id=context_id)

    navigation_state = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-set-navigation",
            FORMAL_AFTER_THAT_NAVIGATION_REQUEST,
        )
    )

    _assert_one_call(
        navigation_state,
        "get_current_navigation_state",
        {"detailed_information": True},
    )


@pytest.mark.parametrize("bad_evidence", ["wrong_station_scope", "partial_first_route"])
def test_base84_malformed_or_partial_evidence_never_reaches_set(
    bad_evidence: str,
) -> None:
    runtime, client = _runtime()
    context_id = f"base84-bad-{bad_evidence}"
    _run_to_charger_search(runtime, context_id=context_id)

    chargers = warsaw_chargers()
    if bad_evidence == "wrong_station_scope":
        chargers["result"]["pois_found"][0]["corresponding_location_id"] = HAMBURG_ID
    presented = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-charger-results",
            [("search_poi_at_location", chargers)],
        )
    )

    if bad_evidence == "wrong_station_scope":
        assert presented.tool_calls == ()
        blocked = runtime.handle_event(
            _user_event(
                context_id,
                f"{context_id}-charging-time",
                CHARGING_TIME_REQUEST,
            )
        )
        assert blocked.tool_calls == ()
    else:
        assert presented.tool_calls == ()
        _request_charging_time(runtime, context_id=context_id)
        partial_routes = warsaw_ionity_routes()
        del partial_routes["result"]["routes"][0]["alias"]
        blocked = runtime.handle_event(
            _result_event(
                context_id,
                f"{context_id}-partial-route",
                [
                    (
                        "get_routes_from_start_to_destination",
                        partial_routes,
                    ),
                    ("calculate_charging_time_by_soc", charging_time_result()),
                ],
            )
        )
        assert blocked.tool_calls == ()

    attempted = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-set-navigation",
            SET_NAVIGATION_REQUEST,
        )
    )
    assert attempted.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base84_does_not_set_while_95_percent_calculation_is_pending() -> None:
    runtime, client = _runtime()
    context_id = "base84-calculation-pending"
    _run_to_charger_presentation(runtime, context_id=context_id)
    _request_charging_time(runtime, context_id=context_id)

    route_only = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-first-segment-only",
            [
                (
                    "get_routes_from_start_to_destination",
                    warsaw_ionity_routes(),
                )
            ],
        )
    )
    assert all(
        call["tool_name"] != "set_new_navigation" for call in route_only.tool_calls
    )

    premature = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-premature-set",
            SET_NAVIGATION_REQUEST,
        )
    )
    assert all(
        call["tool_name"] != "set_new_navigation" for call in premature.tool_calls
    )
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert any(
        call.tool_name == "calculate_charging_time_by_soc"
        for call in session.pending_calls
    )
    assert client.action_calls == 0


def test_base84_active_navigation_fails_before_second_segment_or_set() -> None:
    runtime, client = _runtime()
    context_id = "base84-active-navigation"
    _run_to_charging_answer(runtime, context_id=context_id)

    navigation_state = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-set-navigation",
            SET_NAVIGATION_REQUEST,
        )
    )
    _assert_one_call(
        navigation_state,
        "get_current_navigation_state",
        {"detailed_information": True},
    )

    blocked = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-active-navigation",
            [("get_current_navigation_state", active_navigation_state())],
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert not any(
        result.tool_name == "get_routes_from_start_to_destination"
        and result.arguments == {"start_id": IONITY_ID, "destination_id": HAMBURG_ID}
        for result in session.successful_read_results
    )
    assert client.action_calls == 0


def test_base84_single_quoted_reported_navigation_is_not_action_authority() -> None:
    runtime, client = _runtime()
    context_id = "base84-quoted-navigation"
    _run_to_charging_answer(runtime, context_id=context_id)

    blocked = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-quoted",
            (
                "The dashboard displays 'Yes, please set up navigation to Hamburg "
                "with the charging stop at Ionity first. For the route to Ionity, "
                "use the fastest option. For the route to Hamburg, use the second "
                "route option via B432, B132.'"
            ),
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base84_comparison_with_negated_station_never_starts_multileg_action() -> None:
    runtime, client = _runtime()
    context_id = "base84-negated-station-comparison"
    _run_to_charging_answer(runtime, context_id=context_id)

    blocked = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-comparison",
            (
                "Navigate to Hamburg, not Ionity. For comparison, first show the "
                "fastest route to Ionity, and then use the second route to Hamburg "
                "via B432, B132."
            ),
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base84_first_segment_via_never_binds_to_second_segment() -> None:
    runtime, client = _runtime()
    context_id = "base84-segment-scoped-via"
    _run_to_charging_answer(runtime, context_id=context_id)

    blocked = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-set-navigation",
            (
                "Set up navigation to Hamburg with the charging stop at Ionity "
                "first. For the route to Ionity, use the fastest option via B432, "
                "B132. For the route to Hamburg, use the second route option."
            ),
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base84_conflicting_explicit_current_soc_blocks_calculation() -> None:
    runtime, client = _runtime()
    context_id = "base84-conflicting-current-soc"
    _run_to_charger_presentation(runtime, context_id=context_id)

    blocked = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-charging-time",
            (
                "I'm currently at 40%. I'll use Ionity. How long will it take to "
                "charge to 95%?"
            ),
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.pending_calls == []
    assert client.action_calls == 0


def test_base84_negated_local_search_never_calls_poi_tool() -> None:
    runtime, client = _runtime()
    context_id = "base84-negated-local-search"
    _run_to_charger_presentation(runtime, context_id=context_id)

    blocked = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-negated-search",
            "Don't search for any more charging stations nearby.",
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.pending_calls == []
    assert client.action_calls == 0
