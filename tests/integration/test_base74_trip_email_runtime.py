from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import (
    DecisionProposal,
    Evidence,
    EvidenceSourceKind,
    EvidenceStatus,
    GoalStatus,
)
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
)


MADRID_ID = "loc_mad_180891"
MONACO_ID = "loc_mon_279370"
RACHEL_ID = "con_5111"
RACHEL_EMAIL = "rachel.clark9106@hotmail.com"
FASTEST_ROUTE_ID = "rll_mad_mon_479762"
REPSOL_ID = "poi_cha_127836"
REPSOL_PLUG_ID = "plg_cha_143855"

INITIAL_REQUEST = (
    "Hey there! I need to email Rachel Clark about my upcoming business trip to "
    "Monaco. Can you help me plan the route and include all the travel details in "
    "the email? I don't want to set it in the navigation system, just gather the "
    "info for the email."
)
POLICY = (
    "Use metric units and do not change navigation unless the user asks. If a tool "
    "description starts with REQUIRES_CONFIRMATION, list the exact action details "
    "and obtain a fresh explicit yes before calling it. "
    'CURRENT_LOCATION={"id":"loc_mad_180891","name":"Madrid"}'
)
EMAIL_BODY = (
    "Hi Rachel,\n\n"
    "I wanted to share the details of my upcoming business trip to Monaco. I'll be "
    "driving from Madrid via K139, K466, and L772. The fastest route is "
    "approximately 1168.11 kilometers and should take about 14 hours and 38 "
    "minutes, with no toll roads.\n\n"
    "My current range of 443 kilometers is not enough for the trip. Before "
    "departure, I plan to charge at the Repsol station in Madrid using its "
    "available 350 kilowatt DC plug. Charging from 70 to 100 percent should take "
    "about 10 minutes and provide an estimated full-charge range of 633 "
    "kilometers.\n\n"
    "Because that is still less than the route distance, I'll need at least one "
    "additional charging stop along the way. I haven't planned that stop yet.\n\n"
    "I'll keep you updated on my travel plans. Let me know if you need any "
    "additional information.\n\n"
    "Best regards"
)

NAVIGATION_MUTATIONS = {
    "set_new_navigation",
    "navigation_add_one_waypoint",
    "navigation_replace_final_destination",
    "navigation_replace_one_waypoint",
    "navigation_delete_destination",
    "navigation_delete_waypoint",
}


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


CHARGING_STATUS_TOOL = _tool("get_charging_specs_and_status")
LOCATION_TOOL = _tool(
    "get_location_id_by_location_name",
    {"location": {"type": "string"}},
    ["location"],
)
CONTACT_LOOKUP_TOOL = _tool(
    "get_contact_id_by_contact_name",
    {
        "contact_first_name": {"type": "string"},
        "contact_last_name": {"type": "string"},
    },
)
CONTACT_INFORMATION_TOOL = _tool(
    "get_contact_information",
    {"contact_ids": {"type": "array", "items": {"type": "string"}}},
    ["contact_ids"],
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
    ],
)
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
SEND_EMAIL_TOOL = _tool(
    "send_email",
    {
        "content_message": {"type": "string"},
        "email_addresses": {"type": "array", "items": {"type": "string"}},
    },
    ["content_message", "email_addresses"],
    description=(
        "REQUIRES_CONFIRMATION, Email Tool: sends an email with the specified "
        "message to the specified email addresses."
    ),
)
SET_NAVIGATION_TOOL = _tool(
    "set_new_navigation",
    {"route_ids": {"type": "array", "items": {"type": "string"}}},
    ["route_ids"],
)
ADD_WAYPOINT_TOOL = _tool(
    "navigation_add_one_waypoint",
    {
        "waypoint_id_to_add": {"type": "string"},
        "route_id_leading_to_new_waypoint": {"type": "string"},
    },
)

FULL_TOOLS = (
    CHARGING_STATUS_TOOL,
    LOCATION_TOOL,
    CONTACT_LOOKUP_TOOL,
    CONTACT_INFORMATION_TOOL,
    ROUTES_TOOL,
    LOCAL_POI_TOOL,
    CHARGING_TIME_TOOL,
    DISTANCE_BY_SOC_TOOL,
    SEND_EMAIL_TOOL,
    SET_NAVIGATION_TOOL,
    ADD_WAYPOINT_TOOL,
)
RANGE_TOOLS = (CHARGING_STATUS_TOOL, LOCATION_TOOL, ROUTES_TOOL)
HALL72_TOOLS = tuple(
    tool for tool in FULL_TOOLS if tool["function"]["name"] != "get_distance_by_soc"
)


@pytest.fixture
def charging_status() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "battery_capacity_kwh": 100.0,
            "max_charging_power_ac": 11,
            "max_charging_power_dc": 350,
            "state_of_charge": 70.0,
            "remaining_range": "443.0km",
        },
    }


@pytest.fixture
def hall72_unknown_status() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "battery_capacity_kwh": "unknown",
            "max_charging_power_ac": 11,
            "max_charging_power_dc": 350,
            "state_of_charge": 70.0,
            "remaining_range": "unknown",
        },
    }


@pytest.fixture
def routes() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": FASTEST_ROUTE_ID,
                    "start_id": MADRID_ID,
                    "destination_id": MONACO_ID,
                    "name_via": "K139, K466, L772",
                    "distance_km": 1168.11,
                    "duration_hours": 14,
                    "duration_minutes": 38,
                    "road_types": ["country road", "highway", "urban"],
                    "includes_toll": False,
                    "alias": ["fastest", "first", "shortest"],
                },
                {
                    "route_id": "rll_mad_mon_394363",
                    "start_id": MADRID_ID,
                    "destination_id": MONACO_ID,
                    "name_via": "K581, K631, K98",
                    "distance_km": 1185.93,
                    "duration_hours": 14,
                    "duration_minutes": 47,
                    "road_types": [
                        "country road",
                        "includes toll road",
                        "urban",
                    ],
                    "includes_toll": True,
                    "alias": ["second"],
                },
                {
                    "route_id": "rll_mad_mon_924075",
                    "start_id": MADRID_ID,
                    "destination_id": MONACO_ID,
                    "name_via": "B209, K454, K795",
                    "distance_km": 1187.29,
                    "duration_hours": 14,
                    "duration_minutes": 54,
                    "road_types": ["country road", "highway", "urban"],
                    "includes_toll": False,
                    "alias": ["third"],
                },
            ]
        },
    }


@pytest.fixture
def contact_lookup() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {"matches": {RACHEL_ID: "rachel clark"}},
    }


@pytest.fixture
def contact_information() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            RACHEL_ID: {
                "id": RACHEL_ID,
                "name": {"first_name": "Rachel", "last_name": "Clark"},
                "phone_number": "+49 697 486628",
                "email": RACHEL_EMAIL,
            }
        },
    }


@pytest.fixture
def madrid_chargers() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "pois_found": [
                {
                    "id": REPSOL_ID,
                    "name": "Repsol",
                    "category": "charging_stations",
                    "position": {
                        "longitude": -3.713069501861441,
                        "latitude": 40.361736248282796,
                    },
                    "opening_hours": "00:00h - 24:00h",
                    "phone_number": "+49 932 7332036",
                    "corresponding_location_id": MADRID_ID,
                    "charging_plugs": [
                        {
                            "plug_id": REPSOL_PLUG_ID,
                            "power_type": "DC",
                            "power_kw": 350,
                            "availability": "available",
                        }
                    ],
                },
                {
                    "id": "poi_cha_966627",
                    "name": "Endesa X",
                    "category": "charging_stations",
                    "position": {
                        "longitude": -3.7489342511906587,
                        "latitude": 40.4144035828974,
                    },
                    "opening_hours": "00:00h - 24:00h",
                    "phone_number": "+49 761 8507307",
                    "corresponding_location_id": MADRID_ID,
                    "charging_plugs": [
                        {
                            "plug_id": "plg_cha_207514",
                            "power_type": "AC",
                            "power_kw": 22,
                            "availability": "available",
                        },
                        {
                            "plug_id": "plg_cha_916600",
                            "power_type": "DC",
                            "power_kw": 250,
                            "availability": "occupied",
                        },
                        {
                            "plug_id": "plg_cha_381092",
                            "power_type": "DC",
                            "power_kw": 150,
                            "availability": "available",
                        },
                    ],
                },
            ]
        },
    }


def _intent(
    goals: list[dict[str, Any]],
    *,
    slots: dict[str, Any],
    action: bool,
) -> IntentDraft:
    return IntentDraft.model_validate(
        {
            "language": "en",
            "intent_kind": "action" if action else "information",
            "call_for_action": action,
            "goals": goals,
            "explicit_slots": slots,
        }
    )


class TripRangeClient:
    def __init__(self) -> None:
        self.intent = _intent(
            [
                {
                    "semantic_operation": "assess_battery_charge_trip_range",
                    "desired_outcome": {"destination_name": "Monaco"},
                }
            ],
            slots={"destination_name": "Monaco"},
            action=False,
        )
        self.action_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        del messages, critic
        if response_model is IntentDraft:
            return SimpleNamespace(value=self.intent)
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError("the trip range control must be deterministic")
        raise AssertionError(f"unexpected response model: {response_model}")


class Base74Client:
    def __init__(self) -> None:
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
            raise AssertionError("the Base74 trip email must be deterministic")
        if response_model is not IntentDraft:
            raise AssertionError(f"unexpected response model: {response_model}")

        text = self._latest_user_text(messages)
        if "charging stations" in text:
            intent = _intent(
                [
                    {
                        "semantic_operation": "search_poi_at_location",
                        "desired_outcome": {
                            "location_name": "Madrid",
                            "category": "charging_stations",
                            "filters": ["charging_stations::has_available_plug"],
                        },
                    }
                ],
                slots={
                    "location_name": "Madrid",
                    "category": "charging_stations",
                },
                action=False,
            )
        elif "repsol" in text:
            intent = _intent(
                [
                    {
                        "semantic_operation": "get_ev_range",
                        "desired_outcome": {"initial_soc": 100, "final_soc": 0},
                    }
                ],
                slots={"initial_soc": 100, "final_soc": 0},
                action=False,
            )
        elif "mention" in text and "email" in text:
            intent = _intent(
                [
                    {
                        "semantic_operation": "send_email",
                        "desired_outcome": {
                            "first_name": "Rachel",
                            "last_name": "Clark",
                            "message": "business trip route and charging details",
                        },
                    }
                ],
                slots={"first_name": "Rachel", "last_name": "Clark"},
                action=True,
            )
        elif "remaining range" in text or "fastest route" in text:
            intent = _intent(
                [
                    {
                        "semantic_operation": "assess_battery_charge_trip_range",
                        "desired_outcome": {"destination_name": "Monaco"},
                    }
                ],
                slots={"destination_name": "Monaco"},
                action=False,
            )
        else:
            intent = _intent(
                [
                    {
                        "semantic_operation": "find_routes",
                        "desired_outcome": {
                            "start_id": MADRID_ID,
                            "destination_name": "Monaco",
                        },
                    },
                    {
                        "semantic_operation": "send_email",
                        "desired_outcome": {
                            "first_name": "Rachel",
                            "last_name": "Clark",
                            "message": "business trip travel details",
                        },
                    },
                ],
                slots={
                    "destination_name": "Monaco",
                    "first_name": "Rachel",
                    "last_name": "Clark",
                },
                action=True,
            )
        return SimpleNamespace(value=intent)


def _runtime(client: TripRangeClient | Base74Client) -> CARGuardOrchestrator:
    config = AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=48)
    return CARGuardOrchestrator(config, client_factory=lambda session: client)


def _user_event(
    *,
    context_id: str,
    message_id: str,
    text: str,
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
    *,
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


def _evidence(proposition: str, value: Any) -> Evidence:
    return Evidence(
        proposition=proposition,
        value=value,
        status=EvidenceStatus.KNOWN,
        source_kind=EvidenceSourceKind.SYSTEM,
        source_turn_id=f"base74-{proposition}-fixture",
        confidence=1.0,
    )


def _calls_by_name(outbound: Any) -> dict[str, dict[str, Any]]:
    _assert_no_navigation_mutation(outbound)
    calls = {call["tool_name"]: call for call in outbound.tool_calls}
    assert len(calls) == len(outbound.tool_calls), "unexpected duplicate tool calls"
    return calls


def _assert_no_navigation_mutation(outbound: Any) -> None:
    assert all(
        call["tool_name"] not in NAVIGATION_MUTATIONS for call in outbound.tool_calls
    )


def _run_range_control(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    status_result: dict[str, Any],
    route_result: dict[str, Any],
) -> Any:
    status = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="Can you tell me if my remaining range is enough to reach Monaco?",
            tools=RANGE_TOOLS,
        )
    )
    assert status.tool_calls == (
        {"tool_name": "get_charging_specs_and_status", "arguments": {}},
    )
    location = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-status",
            results=[("get_charging_specs_and_status", status_result)],
        )
    )
    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Monaco"},
        },
    )
    route = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            results=[
                (
                    "get_location_id_by_location_name",
                    {"status": "SUCCESS", "result": {"id": MONACO_ID}},
                )
            ],
        )
    )
    assert route.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {"start_id": MADRID_ID, "destination_id": MONACO_ID},
        },
    )
    return runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            results=[("get_routes_from_start_to_destination", route_result)],
        )
    )


def _run_to_initial_preview(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    route_result: dict[str, Any],
    contact_result: dict[str, Any],
    contact_info_result: dict[str, Any],
) -> Any:
    first = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-initial",
            text=INITIAL_REQUEST,
            tools=FULL_TOOLS,
        )
    )
    calls = _calls_by_name(first)
    assert set(calls) == {
        "get_location_id_by_location_name",
        "get_contact_id_by_contact_name",
    }
    assert calls["get_location_id_by_location_name"]["arguments"] == {
        "location": "Monaco"
    }
    assert calls["get_contact_id_by_contact_name"]["arguments"] == {
        "contact_first_name": "Rachel",
        "contact_last_name": "Clark",
    }

    second = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-identity-results",
            results=[
                (
                    "get_location_id_by_location_name",
                    {"status": "SUCCESS", "result": {"id": MONACO_ID}},
                ),
                ("get_contact_id_by_contact_name", contact_result),
            ],
        )
    )
    calls = _calls_by_name(second)
    assert set(calls) == {
        "get_routes_from_start_to_destination",
        "get_contact_information",
    }
    assert calls["get_routes_from_start_to_destination"]["arguments"] == {
        "start_id": MADRID_ID,
        "destination_id": MONACO_ID,
    }
    assert calls["get_contact_information"]["arguments"] == {
        "contact_ids": [RACHEL_ID]
    }

    preview = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-trip-results",
            results=[
                ("get_routes_from_start_to_destination", route_result),
                ("get_contact_information", contact_info_result),
            ],
        )
    )
    _assert_no_navigation_mutation(preview)
    assert preview.tool_calls == ()
    assert preview.text is not None
    assert "fastest and shortest" in preview.text
    assert "K139" in preview.text and "14 hours and 38 minutes" in preview.text
    assert "2" in preview.text and "alternative" in preview.text.casefold()
    assert "K581" not in preview.text and "B209" not in preview.text
    assert RACHEL_EMAIL in preview.text
    return preview


def _run_to_range_answer(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    route_result: dict[str, Any],
    contact_result: dict[str, Any],
    contact_info_result: dict[str, Any],
    status_result: dict[str, Any],
) -> Any:
    _run_to_initial_preview(
        runtime,
        context_id=context_id,
        route_result=route_result,
        contact_result=contact_result,
        contact_info_result=contact_info_result,
    )
    status = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-range-request",
            text=(
                "Okay, I'll go with the fastest route you mentioned. Before we send "
                "this, can you tell me what my current range is and if it's enough "
                "for this trip?"
            ),
        )
    )
    assert status.tool_calls == (
        {"tool_name": "get_charging_specs_and_status", "arguments": {}},
    )
    answer = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-range-result",
            results=[("get_charging_specs_and_status", status_result)],
        )
    )
    _assert_no_navigation_mutation(answer)
    assert answer.tool_calls == ()
    assert answer.text is not None
    assert "443" in answer.text and "1168" in answer.text
    assert "not enough" in answer.text.casefold()
    return answer


def _run_to_charger_search(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    route_result: dict[str, Any],
    contact_result: dict[str, Any],
    contact_info_result: dict[str, Any],
    status_result: dict[str, Any],
) -> Any:
    _run_to_range_answer(
        runtime,
        context_id=context_id,
        route_result=route_result,
        contact_result=contact_result,
        contact_info_result=contact_info_result,
        status_result=status_result,
    )
    search = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-charger-request",
            text=(
                "Okay, let's charge here in Madrid before I head out. Can you find "
                "some charging stations in Madrid?"
            ),
        )
    )
    _assert_no_navigation_mutation(search)
    assert search.tool_calls == (
        {
            "tool_name": "search_poi_at_location",
            "arguments": {
                "location_id": MADRID_ID,
                "category_poi": "charging_stations",
                "filters": ["charging_stations::has_available_plug"],
            },
        },
    )
    return search


def _run_to_charger_presentation(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    route_result: dict[str, Any],
    contact_result: dict[str, Any],
    contact_info_result: dict[str, Any],
    status_result: dict[str, Any],
    charger_result: dict[str, Any],
) -> Any:
    _run_to_charger_search(
        runtime,
        context_id=context_id,
        route_result=route_result,
        contact_result=contact_result,
        contact_info_result=contact_info_result,
        status_result=status_result,
    )
    presented = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-chargers",
            results=[("search_poi_at_location", charger_result)],
        )
    )
    _assert_no_navigation_mutation(presented)
    assert presented.tool_calls == ()
    assert presented.text is not None
    assert "Repsol" in presented.text and "Endesa X" in presented.text
    assert "350" in presented.text and "150" in presented.text
    assert "occupied" in presented.text.casefold()
    return presented


def _run_to_full_range_answer(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    route_result: dict[str, Any],
    contact_result: dict[str, Any],
    contact_info_result: dict[str, Any],
    status_result: dict[str, Any],
    charger_result: dict[str, Any],
) -> Any:
    _run_to_charger_presentation(
        runtime,
        context_id=context_id,
        route_result=route_result,
        contact_result=contact_result,
        contact_info_result=contact_info_result,
        status_result=status_result,
        charger_result=charger_result,
    )
    assessment = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-repsol",
            text=(
                "Yeah, let's go with the Repsol station. How long would it take to "
                "charge to 100% there? And after that, will I need another charging "
                "stop along the way to Monaco?"
            ),
        )
    )
    calls = _calls_by_name(assessment)
    assert set(calls) == {
        "calculate_charging_time_by_soc",
        "get_distance_by_soc",
    }
    assert calls["calculate_charging_time_by_soc"]["arguments"] == {
        "charging_station_id": REPSOL_ID,
        "charging_station_plug_id": REPSOL_PLUG_ID,
        "start_state_of_charge": 70,
        "target_state_of_charge": 100,
    }
    assert calls["get_distance_by_soc"]["arguments"] == {
        "initial_state_of_charge": 100,
        "final_state_of_charge": 0,
    }

    answer = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-charge-assessment",
            results=[
                (
                    "calculate_charging_time_by_soc",
                    {
                        "status": "SUCCESS",
                        "result": {
                            "time_from_70_until_100_percent_soc": "10min"
                        },
                    },
                ),
                (
                    "get_distance_by_soc",
                    {
                        "status": "SUCCESS",
                        "result": {
                            "distance_km_for_100_until_0_percent_soc": "633.0km"
                        },
                    },
                ),
            ],
        )
    )
    _assert_no_navigation_mutation(answer)
    assert answer.tool_calls == ()
    assert answer.text is not None
    assert "10" in answer.text and "633" in answer.text
    assert "507" not in answer.text
    assert "additional" in answer.text.casefold() or "another" in answer.text.casefold()
    return answer


def _run_to_frozen_draft(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    route_result: dict[str, Any],
    contact_result: dict[str, Any],
    contact_info_result: dict[str, Any],
    status_result: dict[str, Any],
    charger_result: dict[str, Any],
) -> Any:
    _run_to_full_range_answer(
        runtime,
        context_id=context_id,
        route_result=route_result,
        contact_result=contact_result,
        contact_info_result=contact_info_result,
        status_result=status_result,
        charger_result=charger_result,
    )
    draft = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-draft-request",
            text=(
                "Okay, so another stop is needed. I don't want to plan that one yet. "
                "Please mention it in the email to Rachel, update the email with all "
                "the verified information, and then send it."
            ),
        )
    )
    _assert_no_navigation_mutation(draft)
    assert draft.tool_calls == ()
    assert draft.text is not None
    for marker in (
        RACHEL_EMAIL,
        "K139",
        "1168.11",
        "14 hours and 38 minutes",
        "Repsol",
        "10 minutes",
        "633",
        "additional charging stop",
    ):
        assert marker in draft.text
    assert "507" not in draft.text
    assert "confirm" in draft.text.casefold() or "send" in draft.text.casefold()
    return draft


def _confirm_frozen_email(runtime: CARGuardOrchestrator, *, context_id: str) -> Any:
    sent = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes, please send it!",
        )
    )
    _assert_no_navigation_mutation(sent)
    assert sent.tool_calls == (
        {
            "tool_name": "send_email",
            "arguments": {
                "content_message": EMAIL_BODY,
                "email_addresses": [RACHEL_EMAIL],
            },
        },
    )
    return sent


def test_base74_live_fixtures_are_green_controls(
    charging_status: dict[str, Any],
    routes: dict[str, Any],
    madrid_chargers: dict[str, Any],
) -> None:
    options = CARGuardOrchestrator._destination_route_options_from_evidence(
        _evidence("route_options", routes["result"]["routes"]),
        start_id=MADRID_ID,
        destination_id=MONACO_ID,
    )
    charging = CARGuardOrchestrator._trip_charging_values(
        _evidence("trip_charging_status", charging_status["result"])
    )
    poi_names = CARGuardOrchestrator._poi_names_from_evidence(
        _evidence("poi_candidates", madrid_chargers["result"]["pois_found"]),
        location_id=MADRID_ID,
        category="charging_stations",
    )

    assert options is not None and options[0].route_id == FASTEST_ROUTE_ID
    assert options[0].duration_minutes == 878
    assert charging is not None
    assert tuple(map(str, charging)) == ("70.0", "443.0")
    assert poi_names == ("Repsol", "Endesa X")


def test_base74_current_range_control_uses_443_not_canonical_full_range(
    charging_status: dict[str, Any], routes: dict[str, Any]
) -> None:
    client = TripRangeClient()
    answer = _run_range_control(
        _runtime(client),
        context_id="base74-range-control",
        status_result=charging_status,
        route_result=routes,
    )

    assert answer.tool_calls == ()
    assert answer.text is not None
    assert "443" in answer.text and "1168.11" in answer.text
    assert "need to charge" in answer.text.casefold()
    assert client.action_calls == 0


def test_hall72_unknown_range_stops_before_conditional_reads(
    hall72_unknown_status: dict[str, Any], routes: dict[str, Any]
) -> None:
    client = TripRangeClient()
    runtime = _runtime(client)
    context_id = "hall72-unknown-range"
    status = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=(
                "Before sending the trip email, can you tell me if my remaining "
                "range is enough to reach Monaco?"
            ),
            tools=HALL72_TOOLS,
        )
    )
    assert status.tool_calls == (
        {"tool_name": "get_charging_specs_and_status", "arguments": {}},
    )

    blocked = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-status",
            results=[("get_charging_specs_and_status", hall72_unknown_status)],
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None and "couldn't verify" in blocked.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert len(session.successful_read_results) == 1
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0
    assert routes["result"]["routes"][0]["distance_km"] == 1168.11


def test_base74_full_name_contact_and_route_preview_never_mutate_navigation(
    routes: dict[str, Any],
    contact_lookup: dict[str, Any],
    contact_information: dict[str, Any],
) -> None:
    client = Base74Client()
    runtime = _runtime(client)
    context_id = "base74-initial-preview"

    preview = _run_to_initial_preview(
        runtime,
        context_id=context_id,
        route_result=routes,
        contact_result=contact_lookup,
        contact_info_result=contact_information,
    )

    assert preview.text is not None and "no toll" in preview.text.casefold()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base74_repsol_and_plug_are_derived_from_displayed_live_evidence(
    charging_status: dict[str, Any],
    routes: dict[str, Any],
    contact_lookup: dict[str, Any],
    contact_information: dict[str, Any],
    madrid_chargers: dict[str, Any],
) -> None:
    client = Base74Client()
    runtime = _runtime(client)

    answer = _run_to_full_range_answer(
        runtime,
        context_id="base74-repsol-provenance",
        route_result=routes,
        contact_result=contact_lookup,
        contact_info_result=contact_information,
        status_result=charging_status,
        charger_result=madrid_chargers,
    )

    assert answer.text is not None and "633" in answer.text
    assert "507" not in answer.text
    assert client.action_calls == 0


def test_base74_wrong_location_charger_never_reaches_calculation(
    charging_status: dict[str, Any],
    routes: dict[str, Any],
    contact_lookup: dict[str, Any],
    contact_information: dict[str, Any],
    madrid_chargers: dict[str, Any],
) -> None:
    client = Base74Client()
    runtime = _runtime(client)
    context_id = "base74-poisoned-repsol"
    poisoned = deepcopy(madrid_chargers)
    poisoned["result"]["pois_found"][0]["corresponding_location_id"] = MONACO_ID

    _run_to_charger_search(
        runtime,
        context_id=context_id,
        route_result=routes,
        contact_result=contact_lookup,
        contact_info_result=contact_information,
        status_result=charging_status,
    )
    blocked = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-chargers",
            results=[("search_poi_at_location", poisoned)],
        )
    )

    _assert_no_navigation_mutation(blocked)
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "couldn't verify" in blocked.text
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "draft_request",
    [
        (
            "Can you draft an email to Rachel Clark with all these details? Please "
            "include an overview of the trip, route, current range, charging plan, "
            "and additional stop. Make it sound professional but friendly."
        ),
        (
            "Could you please draft that email to Rachel Clark for me now? It needs "
            "to include the verified route, current range, charging plan, and "
            "additional stop."
        ),
        (
            "I need you to draft that email to Rachel Clark. Please include the "
            "trip overview, route distance and duration, current range, local "
            "charging plan, and additional stop."
        ),
    ],
)
def test_base74_full_name_draft_paraphrases_freeze_verified_email(
    draft_request: str,
    charging_status: dict[str, Any],
    routes: dict[str, Any],
    contact_lookup: dict[str, Any],
    contact_information: dict[str, Any],
    madrid_chargers: dict[str, Any],
) -> None:
    runtime = _runtime(Base74Client())
    context_id = f"base74-draft-{abs(hash(draft_request))}"
    _run_to_full_range_answer(
        runtime,
        context_id=context_id,
        route_result=routes,
        contact_result=contact_lookup,
        contact_info_result=contact_information,
        status_result=charging_status,
        charger_result=madrid_chargers,
    )

    draft = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-draft",
            text=draft_request,
        )
    )

    _assert_no_navigation_mutation(draft)
    assert draft.tool_calls == ()
    assert draft.text is not None
    for marker in (
        RACHEL_EMAIL,
        "K139",
        "1168.11",
        "14 hours and 38 minutes",
        "Repsol",
        "10 minutes",
        "633",
        "additional charging stop",
    ):
        assert marker in draft.text


def test_base74_fresh_yes_sends_exact_frozen_633_km_draft(
    charging_status: dict[str, Any],
    routes: dict[str, Any],
    contact_lookup: dict[str, Any],
    contact_information: dict[str, Any],
    madrid_chargers: dict[str, Any],
) -> None:
    client = Base74Client()
    runtime = _runtime(client)
    context_id = "base74-frozen-email"

    _run_to_frozen_draft(
        runtime,
        context_id=context_id,
        route_result=routes,
        contact_result=contact_lookup,
        contact_info_result=contact_information,
        status_result=charging_status,
        charger_result=madrid_chargers,
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()

    sent = _confirm_frozen_email(runtime, context_id=context_id)

    assert "633" in EMAIL_BODY and "507" not in EMAIL_BODY
    assert sent.text is None
    assert client.action_calls == 0


def test_base74_failed_send_never_claims_email_completion(
    charging_status: dict[str, Any],
    routes: dict[str, Any],
    contact_lookup: dict[str, Any],
    contact_information: dict[str, Any],
    madrid_chargers: dict[str, Any],
) -> None:
    client = Base74Client()
    runtime = _runtime(client)
    context_id = "base74-email-failure"
    _run_to_frozen_draft(
        runtime,
        context_id=context_id,
        route_result=routes,
        contact_result=contact_lookup,
        contact_info_result=contact_information,
        status_result=charging_status,
        charger_result=madrid_chargers,
    )
    _confirm_frozen_email(runtime, context_id=context_id)
    session = runtime.sessions.get(context_id)
    assert session is not None
    pending_goals = tuple(
        goal_id for call in session.pending_calls for goal_id in call.goal_ids
    )
    assert pending_goals

    failed = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-send-result",
            results=[
                (
                    "send_email",
                    {
                        "status": "FAILURE",
                        "errors": {"SEND_EMAIL_001": "email rejected"},
                    },
                )
            ],
        )
    )

    assert failed.tool_calls == ()
    assert failed.text is not None
    assert "sent" not in failed.text.casefold()
    assert "couldn't verify" in failed.text.casefold()
    for goal_id in pending_goals:
        assert session.goal_dag.get(goal_id).status is GoalStatus.FAILED
    assert client.action_calls == 0
