from __future__ import annotations

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


BELGRADE_ID = "loc_bel_437063"
BUDAPEST_ID = "loc_bud_247915"
FASTEST_ROUTE_ID = "rll_bel_bud_590769"
PRE_ID = "poi_cha_429428"
GRACE_NELSON_ID = "con_3790"
GRACE_EMAIL = "grace.nelson3334@gmail.com"
POLICY = (
    "Follow the current safety policy. "
    'CURRENT_LOCATION={"id":"loc_bel_437063","name":"Belgrade"}'
)
EMAIL_BODY = (
    "Hi Grace,\n\n"
    "I wanted to update you on my travel time for our meeting in Budapest. "
    "I'll be driving from Belgrade and the route is approximately 369 "
    "kilometers, which should take about 4 hours and 41 minutes.\n\n"
    "Since I'm driving an electric vehicle, I'll need a charging stop. PRE is "
    "around kilometer 169, with a 5.1 kilometer detour that adds about 7 "
    "minutes. It is open 24 hours. Its DC plugs are currently occupied, and "
    "an 11 kilowatt AC plug is available.\n\n"
    "I'll keep you posted if there are any changes to my arrival time.\n\n"
    "Best regards"
)


def tool(
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


CHARGING_STATUS_TOOL = tool("get_charging_specs_and_status")
LOCATION_TOOL = tool(
    "get_location_id_by_location_name",
    {"location": {"type": "string"}},
    ["location"],
)
ROUTES_TOOL = tool(
    "get_routes_from_start_to_destination",
    {
        "start_id": {"type": "string"},
        "destination_id": {"type": "string"},
    },
    ["start_id", "destination_id"],
)
CHARGER_SEARCH_TOOL = tool(
    "search_poi_along_the_route",
    {
        "route_id": {"type": "string"},
        "category_poi": {"type": "string"},
        "at_kilometer": {"type": "integer"},
        "filters": {"type": "array", "items": {"type": "string"}},
    },
    ["route_id", "category_poi", "at_kilometer"],
)
CONTACT_LOOKUP_TOOL = tool(
    "get_contact_id_by_contact_name",
    {
        "contact_first_name": {"type": "string"},
        "contact_last_name": {"type": "string"},
    },
)
CONTACT_INFORMATION_TOOL = tool(
    "get_contact_information",
    {"contact_ids": {"type": "array", "items": {"type": "string"}}},
    ["contact_ids"],
)
SEND_EMAIL_TOOL = tool(
    "send_email",
    {
        "email_addresses": {"type": "array", "items": {"type": "string"}},
        "content_message": {"type": "string"},
    },
    ["email_addresses", "content_message"],
    description=(
        "REQUIRES_CONFIRMATION: Send the exact content to the listed recipients."
    ),
)
SET_NAVIGATION_TOOL = tool(
    "set_new_navigation",
    {"route_ids": {"type": "array", "items": {"type": "string"}}},
    ["route_ids"],
)
ADD_WAYPOINT_TOOL = tool(
    "navigation_add_one_waypoint",
    {
        "waypoint_id_before_new_waypoint": {"type": "string"},
        "route_id_leading_to_new_waypoint": {"type": "string"},
        "route_id_leading_away_from_new_waypoint": {"type": "string"},
        "waypoint_id_to_add": {"type": "string"},
        "waypoint_id_after_new_waypoint": {"type": "string"},
    },
)
FULL_TOOLS = (
    CHARGING_STATUS_TOOL,
    LOCATION_TOOL,
    ROUTES_TOOL,
    CHARGER_SEARCH_TOOL,
    CONTACT_LOOKUP_TOOL,
    CONTACT_INFORMATION_TOOL,
    SEND_EMAIL_TOOL,
    SET_NAVIGATION_TOOL,
    ADD_WAYPOINT_TOOL,
)
HALL68_TOOLS = tuple(
    spec for spec in FULL_TOOLS if spec["function"]["name"] != "send_email"
)
HALL_NO_CONTACT_INFO_TOOLS = tuple(
    spec for spec in FULL_TOOLS if spec["function"]["name"] != "get_contact_information"
)


def trip_range_intent() -> dict[str, Any]:
    return {
        "language": "en",
        "intent_kind": "information",
        "call_for_action": False,
        "goals": [
            {
                "semantic_operation": "assess_battery_charge_trip_range",
                "desired_outcome": {"destination_name": "Budapest"},
            }
        ],
        "explicit_slots": {"destination_name": "Budapest"},
    }


def route_preview_intent() -> dict[str, Any]:
    return {
        "language": "en",
        "intent_kind": "information",
        "call_for_action": False,
        "goals": [
            {
                "semantic_operation": "find_routes",
                "desired_outcome": {
                    "start_id": BELGRADE_ID,
                    "destination_name": "Budapest",
                },
            }
        ],
        "explicit_slots": {"destination_name": "Budapest"},
    }


def hall68_intent() -> dict[str, Any]:
    return {
        "language": "en",
        "intent_kind": "action",
        "call_for_action": True,
        "goals": [
            *trip_range_intent()["goals"],
            {
                "semantic_operation": "send_email",
                "desired_outcome": {
                    "first_name": "Grace",
                    "last_name": "Nelson",
                    "message": "travel time and charging station details",
                },
            },
        ],
        "explicit_slots": {
            "destination_name": "Budapest",
            "first_name": "Grace",
            "last_name": "Nelson",
            "message": "travel time and charging station details",
        },
    }


class StaticIntentClient:
    def __init__(self, intent: dict[str, Any]) -> None:
        self.intent = IntentDraft.model_validate(intent)
        self.intent_calls = 0
        self.action_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        del messages, critic
        if response_model is IntentDraft:
            self.intent_calls += 1
            return SimpleNamespace(value=self.intent)
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError("the business trip workflow must be deterministic")
        raise AssertionError(f"unexpected response model: {response_model}")


def runtime_for(client: StaticIntentClient) -> CARGuardOrchestrator:
    config = AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=48)
    return CARGuardOrchestrator(config, client_factory=lambda session: client)


def user_event(
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
        live_tools=tools or (),
    )


def result_event(
    *, context_id: str, message_id: str, tool_name: str, content: Any
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        tool_results=({"toolName": tool_name, "content": content},),
    )


def charging_status() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "battery_capacity_kwh": 90.0,
            "max_charging_power_ac": 22,
            "max_charging_power_dc": 268,
            "state_of_charge": 72.0,
            "remaining_range": "324.0km",
        },
    }


def route_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": FASTEST_ROUTE_ID,
                    "start_id": BELGRADE_ID,
                    "destination_id": BUDAPEST_ID,
                    "name_via": "B2, B197, B127",
                    "distance_km": 368.81,
                    "duration_hours": 4,
                    "duration_minutes": 41,
                    "road_types": ["country road", "highway"],
                    "includes_toll": False,
                    "alias": ["fastest", "first", "shortest"],
                },
                {
                    "route_id": "rll_bel_bud_989600",
                    "start_id": BELGRADE_ID,
                    "destination_id": BUDAPEST_ID,
                    "name_via": "A85, A25",
                    "distance_km": 378.03,
                    "duration_hours": 4,
                    "duration_minutes": 43,
                    "road_types": ["highway"],
                    "includes_toll": False,
                    "alias": ["second"],
                },
                {
                    "route_id": "rll_bel_bud_409991",
                    "start_id": BELGRADE_ID,
                    "destination_id": BUDAPEST_ID,
                    "name_via": "L845, L191",
                    "distance_km": 392.27,
                    "duration_hours": 4,
                    "duration_minutes": 56,
                    "road_types": ["country road", "highway", "urban"],
                    "includes_toll": False,
                    "alias": ["third"],
                },
            ]
        },
    }


def charger_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "pois_found_along_route": [
                {
                    "id": PRE_ID,
                    "name": "PRE",
                    "category": "charging_stations",
                    "opening_hours": "00:00h - 24:00h",
                    "phone_number": "+49 469 8225496",
                    "corresponding_route_ids": [FASTEST_ROUTE_ID],
                    "route_positions": {"rll_bel_bud": {"at_route_kilometer": 168.8}},
                    "charging_plugs": [
                        {
                            "plug_id": "plg_cha_292542",
                            "power_type": "AC",
                            "power_kw": 11,
                            "availability": "available",
                        },
                        {
                            "plug_id": "plg_cha_657487",
                            "power_type": "DC",
                            "power_kw": 50,
                            "availability": "occupied",
                        },
                        {
                            "plug_id": "plg_cha_763675",
                            "power_type": "DC",
                            "power_kw": 150,
                            "availability": "occupied",
                        },
                    ],
                    "detour_from_route_km": {"detour": 5.1, "unit": "km"},
                    "detour_from_route_time": {"hour": 0, "minutes": 7},
                }
            ]
        },
    }


def contact_matches() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "matches": {
                GRACE_NELSON_ID: "grace nelson",
                "con_6876": "grace young",
                "con_2738": "grace lewis",
                "con_4826": "grace martin",
                "con_8151": "grace green",
            }
        },
    }


def contact_information() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            GRACE_NELSON_ID: {
                "id": GRACE_NELSON_ID,
                "name": {"first_name": "Grace", "last_name": "Nelson"},
                "phone_number": "+49 544 826700",
                "email": GRACE_EMAIL,
            }
        },
    }


def assert_no_navigation_set(outbound: Any) -> None:
    forbidden = {"set_new_navigation", "navigation_add_one_waypoint"}
    assert all(call["tool_name"] not in forbidden for call in outbound.tool_calls)


def run_trip_range(runtime: CARGuardOrchestrator, *, context_id: str) -> Any:
    status = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=(
                "Can I reach Budapest from Belgrade on my current charge? "
                "I only need the route information."
            ),
            tools=FULL_TOOLS,
        )
    )
    assert status.tool_calls == (
        {"tool_name": "get_charging_specs_and_status", "arguments": {}},
    )
    location = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-status",
            tool_name="get_charging_specs_and_status",
            content=charging_status(),
        )
    )
    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Budapest"},
        },
    )
    route = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": BUDAPEST_ID}},
        )
    )
    assert route.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": BELGRADE_ID,
                "destination_id": BUDAPEST_ID,
            },
        },
    )
    answer = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-route",
            tool_name="get_routes_from_start_to_destination",
            content=route_result(),
        )
    )
    assert answer.tool_calls == ()
    assert answer.text is not None
    assert_no_navigation_set(answer)
    return answer


def request_charger(runtime: CARGuardOrchestrator, *, context_id: str) -> Any:
    run_trip_range(runtime, context_id=context_id)
    search = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-charger-request",
            text=(
                "Use the fastest route. Find a DC charging station around the "
                "150-kilometer mark, but do not add it to navigation."
            ),
        )
    )
    assert search.tool_calls == (
        {
            "tool_name": "search_poi_along_the_route",
            "arguments": {
                "route_id": FASTEST_ROUTE_ID,
                "filters": ["charging_stations::has_dc_plug"],
                "category_poi": "charging_stations",
                "at_kilometer": 150,
            },
        },
    )
    assert_no_navigation_set(search)
    return search


def run_to_charger_presentation(
    runtime: CARGuardOrchestrator, *, context_id: str
) -> Any:
    request_charger(runtime, context_id=context_id)
    presented = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-charger",
            tool_name="search_poi_along_the_route",
            content=charger_result(),
        )
    )
    assert presented.tool_calls == ()
    assert presented.text is not None and "PRE" in presented.text
    assert "occupied" in presented.text.casefold()
    assert_no_navigation_set(presented)
    return presented


def run_to_email_confirmation(runtime: CARGuardOrchestrator, *, context_id: str) -> Any:
    run_to_charger_presentation(runtime, context_id=context_id)
    lookup = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-email",
            text=(
                "Do not add the station as a waypoint. Send an email to Grace "
                "with my travel time and the charging station details."
            ),
        )
    )
    assert lookup.tool_calls == (
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {"contact_first_name": "Grace"},
        },
    )
    choices = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-contacts",
            tool_name="get_contact_id_by_contact_name",
            content=contact_matches(),
        )
    )
    assert choices.tool_calls == ()
    assert choices.text is not None
    assert "Grace Nelson" in choices.text and "Grace Young" in choices.text
    selected = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-nelson",
            text="Grace Nelson.",
        )
    )
    assert selected.tool_calls == (
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": [GRACE_NELSON_ID]},
        },
    )
    draft = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-contact-info",
            tool_name="get_contact_information",
            content=contact_information(),
        )
    )
    assert draft.tool_calls == ()
    assert draft.text is not None
    assert GRACE_EMAIL in draft.text
    assert "4 hours and 41 minutes" in draft.text
    assert "PRE" in draft.text and "11 kilowatt" in draft.text
    assert "Tesla" not in draft.text and "30-45" not in draft.text
    return draft


def test_base70_trip_range_fixture_is_green_control() -> None:
    client = StaticIntentClient(trip_range_intent())
    runtime = runtime_for(client)
    answer = run_trip_range(runtime, context_id="base70-range-control")

    assert "72 percent" in answer.text
    assert "324" in answer.text and "368.81" in answer.text
    assert "need to charge" in answer.text.casefold()
    assert client.action_calls == 0


def test_base70_route_preview_never_sets_navigation() -> None:
    client = StaticIntentClient(route_preview_intent())
    runtime = runtime_for(client)
    context_id = "base70-route-preview"

    location = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=(
                "Plan a route to Budapest from Belgrade and tell me the travel "
                "time. Do not start navigation."
            ),
            tools=FULL_TOOLS,
        )
    )
    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Budapest"},
        },
    )
    route = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": BUDAPEST_ID}},
        )
    )
    assert route.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": BELGRADE_ID,
                "destination_id": BUDAPEST_ID,
            },
        },
    )
    preview = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-route",
            tool_name="get_routes_from_start_to_destination",
            content=route_result(),
        )
    )
    assert preview.tool_calls == ()
    assert preview.text is not None
    assert "B2" in preview.text and "4 hours and 41 minutes" in preview.text
    assert "2" in preview.text and "alternative" in preview.text.casefold()
    assert_no_navigation_set(preview)
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()


def test_base70_range_shortfall_searches_dc_charger_at_150() -> None:
    runtime = runtime_for(StaticIntentClient(trip_range_intent()))

    search = request_charger(runtime, context_id="base70-at-150")

    assert search.text is None


def test_base70_grace_nelson_frozen_draft_requires_yes_before_send() -> None:
    client = StaticIntentClient(trip_range_intent())
    runtime = runtime_for(client)
    context_id = "base70-email"
    run_to_email_confirmation(runtime, context_id=context_id)
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()

    sent = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes, send it.",
        )
    )

    assert sent.tool_calls == (
        {
            "tool_name": "send_email",
            "arguments": {
                "email_addresses": [GRACE_EMAIL],
                "content_message": EMAIL_BODY,
            },
        },
    )
    assert_no_navigation_set(sent)
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "email_request",
    [
        (
            "Plan a route to Budapest, check if I need to charge, and send "
            "Grace Nelson an email with the details."
        ),
        (
            "Okay, the fastest route sounds good! Also, since I'm driving an "
            "electric car, will I need to stop for charging? If so, can you find "
            "a charging station around the 150-kilometer mark? And I also need "
            "to email Grace Nelson to let her know my estimated travel time."
        ),
        (
            "Okay, let's go with the fastest route you mentioned. Also, since "
            "I'm driving an electric car, will I need to stop for charging? If "
            "so, can you find a charging station around the 150-kilometer mark "
            "of the journey? And I also need to email Grace Nelson to let her "
            "know my estimated travel time."
        ),
        (
            "Okay, no worries about the charging station then. Can you just help "
            "me send that email to Grace Nelson with the travel time?"
        ),
        (
            "Okay, thanks for the charging station info. Now, about that email "
            "to Grace Nelson, can you send her my estimated travel time?"
        ),
    ],
)
def test_hallucination68_missing_send_email_stops_before_all_reads(
    email_request: str,
) -> None:
    client = StaticIntentClient(hall68_intent())
    runtime = runtime_for(client)
    context_id = "hall68-no-email"

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=email_request,
            tools=HALL68_TOOLS,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.successful_read_results == []
    assert session.pending_calls == []
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "non_request",
    [
        "I do not need to email Grace Nelson.",
        "Do not send that email to Grace Nelson.",
        'He said, "I need to email Grace Nelson."',
        'The example phrase is "send that email to Grace Nelson."',
    ],
)
def test_hallucination68_email_preflight_ignores_negated_and_quoted_text(
    non_request: str,
) -> None:
    client = StaticIntentClient(hall68_intent())
    runtime = runtime_for(client)

    runtime.handle_event(
        user_event(
            context_id="hall68-non-request",
            message_id="hall68-non-request-user",
            text=non_request,
            tools=HALL68_TOOLS,
        )
    )

    assert client.intent_calls == 1


def test_combined_trip_email_missing_contact_schema_stops_before_all_reads() -> None:
    client = StaticIntentClient(hall68_intent())
    runtime = runtime_for(client)
    context_id = "trip-email-no-contact-info"

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=(
                "Okay, the fastest route sounds good! Also, since I'm driving an "
                "electric car, will I need to stop for charging? If so, can you "
                "find a charging station around the 150-kilometer mark? And I "
                "also need to email Grace Nelson to let her know my estimated "
                "travel time."
            ),
            tools=HALL_NO_CONTACT_INFO_TOOLS,
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.pending_calls == []
    assert session.successful_read_results == []
    assert session.budget.attempted_sets == set()
    assert client.intent_calls == 0


@pytest.mark.parametrize("case", ["wrong-route", "all-plugs-occupied"])
def test_wrong_or_unusable_charger_never_advances_to_email(case: str) -> None:
    runtime = runtime_for(StaticIntentClient(trip_range_intent()))
    context_id = f"base70-bad-charger-{case}"
    request_charger(runtime, context_id=context_id)
    bad = deepcopy(charger_result())
    station = bad["result"]["pois_found_along_route"][0]
    if case == "wrong-route":
        station["corresponding_route_ids"] = ["rll_bel_vie_000001"]
        station["route_positions"] = {"rll_bel_vie": {"at_route_kilometer": 168.8}}
    else:
        for plug in station["charging_plugs"]:
            plug["availability"] = "occupied"

    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-result",
            tool_name="search_poi_along_the_route",
            content=bad,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert any(
        marker in blocked.text.casefold()
        for marker in ("verify", "available", "occupied", "route")
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()


def test_stale_yes_cannot_send_frozen_email() -> None:
    runtime = runtime_for(StaticIntentClient(trip_range_intent()))
    context_id = "base70-stale-yes"
    run_to_email_confirmation(runtime, context_id=context_id)
    session = runtime.sessions.get(context_id)
    assert session is not None
    session.evidence.invalidate_tool_state()

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes.",
        )
    )

    assert all(call["tool_name"] != "send_email" for call in blocked.tool_calls)
    assert_no_navigation_set(blocked)
    assert session.budget.attempted_sets == set()


def test_same_state_trip_summary_tamper_cannot_send_frozen_email() -> None:
    runtime = runtime_for(StaticIntentClient(trip_range_intent()))
    context_id = "base70-same-state-summary-tamper"
    run_to_email_confirmation(runtime, context_id=context_id)
    session = runtime.sessions.get(context_id)
    assert session is not None
    summaries = session.evidence.latest_for_proposition(
        "trip_route_assessment_summary", current_state_only=True
    )
    assert summaries and isinstance(summaries[-1].value, dict)
    summaries[-1].value["route_distance_km"] = "999"

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes.",
        )
    )

    assert all(call["tool_name"] != "send_email" for call in blocked.tool_calls)
    assert_no_navigation_set(blocked)
    assert session.budget.attempted_sets == set()


def test_recorded_short_route_preview_is_information_only() -> None:
    runtime = runtime_for(StaticIntentClient(route_preview_intent()))
    context_id = "base70-recorded-short-route"

    location = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="Plan a route to Budapest.",
            tools=FULL_TOOLS,
        )
    )

    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Budapest"},
        },
    )
    assert_no_navigation_set(location)


@pytest.mark.parametrize(
    "text",
    [
        (
            "Hey there! I'm planning a business trip to Budapest from Belgrade. "
            "Can you help me with the route?"
        ),
        (
            "Yeah, I need to navigate to Budapest from Belgrade. Can you find me "
            "the fastest route?"
        ),
        (
            "Okay, so I need to navigate to Budapest from Belgrade. Can you find "
            "me the fastest route?"
        ),
        (
            "Could you please plan a route for me from Belgrade to Budapest? "
            "I'd prefer the fastest route available."
        ),
        (
            "Hey, can you navigate me to Budapest? I'm starting from Belgrade, "
            "and I want the fastest way there."
        ),
        (
            "Okay, let's try this again. Can you please plan a route for me from "
            "Belgrade to Budapest? I'd like the fastest route, please."
        ),
        (
            "Okay, can you plan a route for me? My starting point is Belgrade and "
            "my destination is Budapest. I'd like the fastest route, please."
        ),
        (
            "Hey, can you plan a route for me from Belgrade to Budapest? I want "
            "the fastest one. And also, can you check if I'll need to charge my "
            "EV on the way? If so, find a fast charging station about 150 "
            "kilometers in."
        ),
    ],
)
def test_formal_route_phrasings_start_information_only_preview(text: str) -> None:
    runtime = runtime_for(StaticIntentClient(route_preview_intent()))
    context_id = f"base70-formal-route-{abs(hash(text))}"

    location = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=text,
            tools=FULL_TOOLS,
        )
    )

    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Budapest"},
        },
    )
    assert_no_navigation_set(location)


@pytest.mark.parametrize(
    "text",
    [
        (
            '"Could you please plan a route for me from Belgrade to Budapest? '
            "I'd prefer the fastest route available.\""
        ),
        (
            "Do not plan a route for me from Belgrade to Budapest. I'd prefer "
            "the fastest route available."
        ),
        (
            "Could you please plan a route for me from Belgrade to Budapest? "
            "I'd prefer the fastest route available. Then email everyone."
        ),
    ],
)
def test_formal_route_grammar_rejects_quotes_negation_and_extra_commands(
    text: str,
) -> None:
    runtime = runtime_for(
        StaticIntentClient(
            {
                "language": "en",
                "intent_kind": "conversation",
                "call_for_action": False,
                "goals": [],
            }
        )
    )
    context_id = f"base70-rejected-route-{abs(hash(text))}"

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=text,
            tools=FULL_TOOLS,
        )
    )

    assert blocked.tool_calls == ()
    assert_no_navigation_set(blocked)


def test_formal_route_grammar_rejects_noncurrent_explicit_start() -> None:
    runtime = runtime_for(StaticIntentClient(route_preview_intent()))
    context_id = "base70-wrong-formal-start"

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=(
                "Could you please plan a route for me from Vienna to Budapest? "
                "I'd prefer the fastest route available."
            ),
            tools=FULL_TOOLS,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert_no_navigation_set(blocked)


def test_short_route_with_explicit_current_start_is_information_only() -> None:
    client = StaticIntentClient(route_preview_intent())
    runtime = runtime_for(client)
    context_id = "base70-short-explicit-start"

    location = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="Plan a route to Budapest from Belgrade.",
            tools=FULL_TOOLS,
        )
    )

    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Budapest"},
        },
    )
    assert client.intent_calls == 0
    assert_no_navigation_set(location)


@pytest.mark.parametrize(
    "text",
    [
        "Plan a route to Budapest from Belgrade and start navigation.",
        "Plan a route to Budapest and turn on the air conditioning.",
    ],
)
def test_short_route_grammar_does_not_intercept_extra_commands(text: str) -> None:
    client = StaticIntentClient(
        {
            "language": "en",
            "intent_kind": "conversation",
            "call_for_action": False,
            "goals": [],
        }
    )
    runtime = runtime_for(client)
    context_id = f"base70-short-rejected-{abs(hash(text))}"

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=text,
            tools=FULL_TOOLS,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert client.intent_calls >= 1
    assert_no_navigation_set(blocked)


def test_short_route_grammar_rejects_noncurrent_explicit_start() -> None:
    client = StaticIntentClient(route_preview_intent())
    runtime = runtime_for(client)
    context_id = "base70-short-wrong-start"

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="Plan a route to Budapest from Vienna.",
            tools=FULL_TOOLS,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert client.intent_calls == 0
    assert_no_navigation_set(blocked)


@pytest.mark.parametrize(
    "charger_text",
    [
        (
            "Use the fastest route. Do I need to stop for charging? If so, "
            "find a charging station around the 150-kilometer mark."
        ),
        (
            "Okay, the fastest route sounds good! Also, since I'm driving an "
            "electric car, will I need to stop for charging? If so, can you find "
            "a charging station around the 150-kilometer mark? And I also need "
            "to email Grace Nelson to let her know my estimated travel time."
        ),
        (
            "Okay, let's go with the fastest route you mentioned. Also, since "
            "I'm driving an electric car, will I need to stop for charging? If "
            "so, can you find a charging station around the 150-kilometer mark "
            "of the journey? And I also need to email Grace Nelson to let her "
            "know my estimated travel time."
        ),
    ],
)
def test_recorded_conditional_charger_followup_uses_verified_fastest_route(
    charger_text: str,
) -> None:
    runtime = runtime_for(StaticIntentClient(trip_range_intent()))
    context_id = "base70-recorded-conditional-charger"
    location = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-route",
            text="Plan a route to Budapest.",
            tools=FULL_TOOLS,
        )
    )
    assert location.tool_calls[0]["tool_name"] == "get_location_id_by_location_name"
    route = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": BUDAPEST_ID}},
        )
    )
    assert route.tool_calls[0]["tool_name"] == "get_routes_from_start_to_destination"
    preview = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=route_result(),
        )
    )
    assert preview.tool_calls == ()

    status = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-charger",
            text=charger_text,
        )
    )
    assert status.tool_calls == (
        {"tool_name": "get_charging_specs_and_status", "arguments": {}},
    )
    assert_no_navigation_set(status)
    search = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-status",
            tool_name="get_charging_specs_and_status",
            content=charging_status(),
        )
    )

    assert search.tool_calls == (
        {
            "tool_name": "search_poi_along_the_route",
            "arguments": {
                "route_id": FASTEST_ROUTE_ID,
                "filters": ["charging_stations::has_dc_plug"],
                "category_poi": "charging_stations",
                "at_kilometer": 150,
            },
        },
    )
    assert_no_navigation_set(search)


def test_recorded_combined_request_continues_to_frozen_email_without_reprompt() -> None:
    client = StaticIntentClient(trip_range_intent())
    runtime = runtime_for(client)
    context_id = "base70-recorded-combined-email-chain"

    location = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-route",
            text=(
                "Hey there! I'm planning a business trip to Budapest from "
                "Belgrade. Can you help me with the route?"
            ),
            tools=FULL_TOOLS,
        )
    )
    assert location.tool_calls[0]["tool_name"] == "get_location_id_by_location_name"
    route = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": BUDAPEST_ID}},
        )
    )
    assert route.tool_calls[0]["tool_name"] == "get_routes_from_start_to_destination"
    preview = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=route_result(),
        )
    )
    assert preview.tool_calls == ()

    status = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-combined",
            text=(
                "Okay, the fastest route sounds good! Also, since I'm driving an "
                "electric car, will I need to stop for charging? If so, can you "
                "find a charging station around the 150-kilometer mark? And I "
                "also need to email Grace Nelson to let her know my estimated "
                "travel time."
            ),
        )
    )
    assert status.tool_calls == (
        {"tool_name": "get_charging_specs_and_status", "arguments": {}},
    )
    search = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-status",
            tool_name="get_charging_specs_and_status",
            content=charging_status(),
        )
    )
    assert search.tool_calls[0]["tool_name"] == "search_poi_along_the_route"
    lookup = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-charger",
            tool_name="search_poi_along_the_route",
            content=charger_result(),
        )
    )
    assert lookup.tool_calls == (
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {"contact_first_name": "Grace"},
        },
    )
    info = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-contacts",
            tool_name="get_contact_id_by_contact_name",
            content=contact_matches(),
        )
    )
    assert info.tool_calls == (
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": [GRACE_NELSON_ID]},
        },
    )
    draft = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-info",
            tool_name="get_contact_information",
            content=contact_information(),
        )
    )

    assert draft.tool_calls == ()
    assert draft.text is not None
    assert "4 hours and 41 minutes" in draft.text
    assert "PRE" in draft.text
    assert GRACE_EMAIL in draft.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0

    sent = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-yes",
            text="Yes, send it.",
        )
    )
    assert sent.tool_calls == (
        {
            "tool_name": "send_email",
            "arguments": {
                "email_addresses": [GRACE_EMAIL],
                "content_message": EMAIL_BODY,
            },
        },
    )
    assert_no_navigation_set(sent)


@pytest.mark.parametrize(
    "email_text",
    [
        (
            "No, don't add it as a waypoint. Send an email to Grace Nelson "
            "with my estimated travel time and the charging station details."
        ),
        (
            "Okay, thanks for the charging station info. Now, about that email "
            "to Grace Nelson, can you send her my estimated travel time?"
        ),
        (
            "Okay, no worries about the charging station then. Can you just help "
            "me send that email to Grace Nelson with the travel time?"
        ),
        (
            "Yes, I want to send an email to Grace Nelson. Can you find her "
            "contact and draft an email with the travel time?"
        ),
        (
            "Okay, I really need to send an email to Grace Nelson. Can you please "
            "try to find her contact again and draft an email with the travel "
            "time?"
        ),
    ],
)
def test_recorded_named_email_binds_only_the_unique_displayed_contact(
    email_text: str,
) -> None:
    runtime = runtime_for(StaticIntentClient(trip_range_intent()))
    context_id = "base70-recorded-named-email"
    run_to_charger_presentation(runtime, context_id=context_id)

    lookup = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-email",
            text=email_text,
        )
    )
    assert lookup.tool_calls == (
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {"contact_first_name": "Grace"},
        },
    )
    selected = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-contacts",
            tool_name="get_contact_id_by_contact_name",
            content=contact_matches(),
        )
    )
    assert selected.tool_calls == (
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": [GRACE_NELSON_ID]},
        },
    )
    assert_no_navigation_set(selected)


def test_multiple_chargers_choose_unique_highest_available_power() -> None:
    runtime = runtime_for(StaticIntentClient(trip_range_intent()))
    context_id = "base70-multiple-chargers"
    request_charger(runtime, context_id=context_id)
    result = charger_result()
    pre = result["result"]["pois_found_along_route"][0]
    second = deepcopy(pre)
    second["id"] = "poi_cha_555555"
    second["name"] = "Power Hub"
    second["phone_number"] = "+49 469 8225500"
    second["route_positions"] = {
        "rll_bel_bud": {"at_route_kilometer": 170.0},
        "rll_bud_bel": {"at_route_kilometer": 198.0},
    }
    second["corresponding_route_ids"] = [
        FASTEST_ROUTE_ID,
        "rll_bud_bel_111111",
    ]
    second["charging_plugs"] = [
        {
            "plug_id": "plg_cha_555551",
            "power_type": "AC",
            "power_kw": 22,
            "availability": "available",
        },
        {
            "plug_id": "plg_cha_555552",
            "power_type": "DC",
            "power_kw": 150,
            "availability": "occupied",
        },
    ]
    pre["route_positions"]["rll_bud_bel"] = {"at_route_kilometer": 200.0}
    pre["corresponding_route_ids"].append("rll_bud_bel_222222")
    result["result"]["pois_found_along_route"].append(second)

    presented = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-result",
            tool_name="search_poi_along_the_route",
            content=result,
        )
    )

    assert presented.tool_calls == ()
    assert presented.text is not None and "Power Hub" in presented.text
    assert_no_navigation_set(presented)


def test_multiple_chargers_with_tied_available_power_fail_closed() -> None:
    runtime = runtime_for(StaticIntentClient(trip_range_intent()))
    context_id = "base70-tied-chargers"
    request_charger(runtime, context_id=context_id)
    result = charger_result()
    second = deepcopy(result["result"]["pois_found_along_route"][0])
    second["id"] = "poi_cha_555556"
    second["name"] = "Tie Station"
    second["phone_number"] = "+49 469 8225501"
    for index, plug in enumerate(second["charging_plugs"]):
        plug["plug_id"] = f"plg_cha_66666{index}"
    result["result"]["pois_found_along_route"].append(second)

    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-result",
            tool_name="search_poi_along_the_route",
            content=result,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert_no_navigation_set(blocked)
