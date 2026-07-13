from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import (
    DecisionProposal,
    EvidenceSourceKind,
    EvidenceStatus,
)
from track_1_agent_under_test.car_guard.llm.client import LLMFailure
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
    _active_route_buffer_search_soc,
    _active_route_range_question_soc,
)


LEIPZIG_ID = "loc_lei_519681"
FRANKFURT_ID = "loc_fra_178468"
HAMBURG_ID = "loc_ham_166665"
BARCELONA_ID = "loc_bar_223644"
LEIPZIG_FRANKFURT_ROUTE_ID = "rll_lei_fra_659595"
FRANKFURT_HAMBURG_ROUTE_ID = "rll_fra_ham_782021"
FRANKFURT_BARCELONA_ROUTE_ID = "rll_fra_bar_981238"
FASTNED_ID = "poi_cha_713487"
FASTNED_PHONE = "+49 358 8158348"

INITIAL_REQUEST = (
    "I'm currently navigating to Hamburg, but I've decided I want to go to "
    "Barcelona instead. Can you show me multiple route options to Barcelona?"
)
ROUTE_SELECTION = (
    "I'll take the shortest route, the one that's 1257.75 kilometers with no "
    "tolls. Please set up the new navigation for me."
)
DIRECT_ROUTE_SELECTION = (
    "I'll take the shortest route, 1257.75 kilometers with no tolls. Please set "
    "up the new navigation for me."
)
RANGE_QUESTION = (
    "I have 98% battery charge right now, which is fine for Frankfurt, but I'm "
    "concerned about the much longer journey to Barcelona. Will I need to stop "
    "for charging to reach Barcelona safely?"
)
FORMAL_RANGE_QUESTION = (
    "I have 98% battery, which is fine for Frankfurt, but I'm concerned about "
    "the range for Barcelona. Can you help me find a charging station to ensure "
    "I can reach Barcelona safely?"
)
BUFFER_SEARCH_REQUEST = (
    "Yes, please search for charging stations. I want to maintain at least 15% "
    "battery charge as a safety buffer, so please find a charging station at a "
    "point where approximately 15% is reached."
)
FORMAL_BUFFER_SEARCH_REQUEST = (
    "Yes, please. I want to maintain at least a 15% battery charge as a safety "
    "buffer, so please find a charging station where approximately 15% battery "
    "is reached."
)
PHONE_QUESTION = (
    "Is there a phone number available for this Fastned station? I'd like to "
    "call them to see if I can reserve a charging spot."
)
FORMAL_PHONE_QUESTION = (
    "Is there a phone number available for Fastned? I'd like to ask if I can "
    "reserve a charging spot."
)
FORMAL_CALL_REQUEST = "Please call the number for Fastned once you have it."
FORMAL_DEMONSTRATIVE_CALL = "Yes, please call that number."
POLICY = (
    "Use metric units. Edit an active route with the corresponding navigation "
    "tool and do not replace the whole route. A phone call ends the conversation. "
    'CURRENT_LOCATION={"id":"loc_lei_519681","name":"Leipzig"}'
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


CURRENT_NAVIGATION_TOOL = tool(
    "get_current_navigation_state",
    {"detailed_information": {"type": "boolean"}},
)
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
REPLACE_DESTINATION_TOOL = tool(
    "navigation_replace_final_destination",
    {
        "new_destination_id": {"type": "string"},
        "route_id_leading_to_new_destination": {"type": "string"},
    },
    ["new_destination_id", "route_id_leading_to_new_destination"],
)
CHARGING_STATUS_TOOL = tool("get_charging_specs_and_status")
DISTANCE_BY_SOC_TOOL = tool(
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
        },
    },
    ["initial_state_of_charge"],
)
SEARCH_ALONG_ROUTE_TOOL = tool(
    "search_poi_along_the_route",
    {
        "route_id": {"type": "string"},
        "category_poi": {"type": "string"},
        "at_kilometer": {"type": "integer"},
        "filters": {"type": "array", "items": {"type": "string"}},
    },
    ["route_id", "category_poi", "at_kilometer"],
)
CALL_TOOL = tool(
    "call_phone_by_number",
    {"phone_number": {"type": "string"}},
    ["phone_number"],
)
FULL_TOOLS = (
    CURRENT_NAVIGATION_TOOL,
    LOCATION_TOOL,
    ROUTES_TOOL,
    REPLACE_DESTINATION_TOOL,
    CHARGING_STATUS_TOOL,
    DISTANCE_BY_SOC_TOOL,
    SEARCH_ALONG_ROUTE_TOOL,
    CALL_TOOL,
)


class Base86Client:
    """The LLM only proposes the initial informational replacement goal."""

    def __init__(self) -> None:
        self.intent_calls = 0
        self.action_calls = 0
        self.initial_intent = IntentDraft.model_validate(
            {
                "language": "en",
                "intent_kind": "information",
                "call_for_action": False,
                "goals": [
                    {
                        "semantic_operation": "navigation_replace_final_destination",
                        "desired_outcome": {
                            "new_destination_name": "Barcelona",
                        },
                    }
                ],
                "explicit_slots": {"new_destination_name": "Barcelona"},
            }
        )
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
            value = self.initial_intent if self.intent_calls == 1 else self.empty_intent
            return SimpleNamespace(value=value)
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError(
                "Base86 actions must be evidence-bound and deterministic"
            )
        raise AssertionError(f"unexpected response model: {response_model}")


class FailingInitialIntentClient(Base86Client):
    def generate(self, *, messages, response_model, critic=False):
        if response_model is IntentDraft:
            self.intent_calls += 1
            raise LLMFailure("synthetic initial intent failure")
        return super().generate(
            messages=messages,
            response_model=response_model,
            critic=critic,
        )


class EmptyInitialIntentClient(Base86Client):
    def generate(self, *, messages, response_model, critic=False):
        if response_model is IntentDraft:
            self.intent_calls += 1
            return SimpleNamespace(value=self.empty_intent)
        return super().generate(
            messages=messages,
            response_model=response_model,
            critic=critic,
        )


def runtime_for(client: Base86Client) -> CARGuardOrchestrator:
    return CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=48),
        client_factory=lambda session: client,
    )


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


def old_navigation_state() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [LEIPZIG_ID, FRANKFURT_ID, HAMBURG_ID],
            "routes_to_final_destination_id": [
                LEIPZIG_FRANKFURT_ROUTE_ID,
                FRANKFURT_HAMBURG_ROUTE_ID,
            ],
            "details": {
                "waypoints": [
                    {"id": LEIPZIG_ID, "name": "Leipzig"},
                    {"id": FRANKFURT_ID, "name": "Frankfurt"},
                    {"id": HAMBURG_ID, "name": "Hamburg"},
                ],
                "routes": [
                    {
                        "route_id": LEIPZIG_FRANKFURT_ROUTE_ID,
                        "start_id": LEIPZIG_ID,
                        "destination_id": FRANKFURT_ID,
                        "name_via": "B599",
                        "distance_km": 336.33,
                        "duration_hours": 4,
                        "duration_minutes": 9,
                        "road_types": ["country road", "urban"],
                        "includes_toll": False,
                        "alias": ["fastest", "first", "shortest"],
                    },
                    {
                        "route_id": FRANKFURT_HAMBURG_ROUTE_ID,
                        "start_id": FRANKFURT_ID,
                        "destination_id": HAMBURG_ID,
                        "name_via": "L849, A97, A58",
                        "distance_km": 466.0,
                        "duration_hours": 5,
                        "duration_minutes": 53,
                        "road_types": ["highway", "urban"],
                        "includes_toll": False,
                        "alias": ["fastest", "first", "shortest"],
                    },
                ],
            },
        },
    }


def barcelona_routes() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": FRANKFURT_BARCELONA_ROUTE_ID,
                    "start_id": FRANKFURT_ID,
                    "destination_id": BARCELONA_ID,
                    "name_via": "K105, K121, L558",
                    "distance_km": 1257.75,
                    "duration_hours": 15,
                    "duration_minutes": 46,
                    "road_types": ["urban"],
                    "includes_toll": False,
                    "alias": ["fastest", "first", "shortest"],
                },
                {
                    "route_id": "rll_fra_bar_271975",
                    "start_id": FRANKFURT_ID,
                    "destination_id": BARCELONA_ID,
                    "name_via": "B479, L2",
                    "distance_km": 1301.8,
                    "duration_hours": 16,
                    "duration_minutes": 34,
                    "road_types": ["country road", "highway", "urban"],
                    "includes_toll": False,
                    "alias": ["second"],
                },
                {
                    "route_id": "rll_fra_bar_603500",
                    "start_id": FRANKFURT_ID,
                    "destination_id": BARCELONA_ID,
                    "name_via": "B235, B823",
                    "distance_km": 1325.07,
                    "duration_hours": 16,
                    "duration_minutes": 55,
                    "road_types": ["country road", "highway"],
                    "includes_toll": False,
                    "alias": ["third"],
                },
            ]
        },
    }


def new_navigation_state() -> dict[str, Any]:
    result = deepcopy(old_navigation_state())
    state = result["result"]
    state["waypoints_id"] = [LEIPZIG_ID, FRANKFURT_ID, BARCELONA_ID]
    state["routes_to_final_destination_id"] = [
        LEIPZIG_FRANKFURT_ROUTE_ID,
        FRANKFURT_BARCELONA_ROUTE_ID,
    ]
    state["details"]["waypoints"][-1] = {
        "id": BARCELONA_ID,
        "name": "Barcelona",
    }
    state["details"]["routes"][-1] = barcelona_routes()["result"]["routes"][0]
    return result


def charging_status(*, remaining_range_km: float = 466.0) -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "battery_capacity_kwh": 70.0,
            "max_charging_power_ac": 22,
            "max_charging_power_dc": 1000,
            "state_of_charge": 98.0,
            "remaining_range": f"{remaining_range_km:.1f}km",
        },
    }


def distance_to_buffer() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {"distance_km_for_98_until_15_percent_soc": "394.0km"},
    }


def fastned_result(**overrides: Any) -> dict[str, Any]:
    station = {
        "id": FASTNED_ID,
        "name": "Fastned",
        "category": "charging_stations",
        "opening_hours": "00:00h - 24:00h",
        "phone_number": FASTNED_PHONE,
        "corresponding_route_ids": [
            FRANKFURT_BARCELONA_ROUTE_ID,
            "rll_fra_bar_271975",
            "rll_fra_bar_603500",
        ],
        "route_positions": {
            "rll_fra_bar": {"at_route_kilometer": 50.0},
        },
        "charging_plugs": [
            {
                "plug_id": "plg_cha_293832",
                "power_type": "DC",
                "power_kw": 100,
                "availability": "available",
            },
            {
                "plug_id": "plg_cha_671040",
                "power_type": "AC",
                "power_kw": 22,
                "availability": "available",
            },
            {
                "plug_id": "plg_cha_666324",
                "power_type": "DC",
                "power_kw": 150,
                "availability": "occupied",
            },
        ],
        "detour_from_route_km": {"detour": 7.8, "unit": "km"},
        "detour_from_route_time": {"hour": 0, "minutes": 10},
    }
    station.update(overrides)
    return {
        "status": "SUCCESS",
        "result": {"pois_found_along_route": [station]},
    }


def _assert_call(outbound: Any, name: str, arguments: dict[str, Any]) -> None:
    assert outbound.tool_calls == ({"tool_name": name, "arguments": arguments},)


def run_to_route_prompt(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    tools: tuple[dict[str, Any], ...] = FULL_TOOLS,
) -> Any:
    current = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-initial",
            text=INITIAL_REQUEST,
            tools=tools,
        )
    )
    _assert_call(
        current,
        "get_current_navigation_state",
        {"detailed_information": True},
    )
    location = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-old-navigation",
            tool_name="get_current_navigation_state",
            content=old_navigation_state(),
        )
    )
    _assert_call(
        location,
        "get_location_id_by_location_name",
        {"location": "Barcelona"},
    )
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-barcelona-id",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": BARCELONA_ID}},
        )
    )
    _assert_call(
        routes,
        "get_routes_from_start_to_destination",
        {"start_id": FRANKFURT_ID, "destination_id": BARCELONA_ID},
    )
    prompt = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-route-options",
            tool_name="get_routes_from_start_to_destination",
            content=barcelona_routes(),
        )
    )
    assert prompt.tool_calls == ()
    assert prompt.text is not None
    for marker in ("1257.75", "fastest and shortest", "2"):
        assert marker in prompt.text
    return prompt


def run_to_replaced_navigation(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    tools: tuple[dict[str, Any], ...] = FULL_TOOLS,
    selection_text: str = ROUTE_SELECTION,
) -> Any:
    run_to_route_prompt(runtime, context_id=context_id, tools=tools)
    replacement = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-select-route",
            text=selection_text,
        )
    )
    _assert_call(
        replacement,
        "navigation_replace_final_destination",
        {
            "new_destination_id": BARCELONA_ID,
            "route_id_leading_to_new_destination": FRANKFURT_BARCELONA_ROUTE_ID,
        },
    )
    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-destination-replaced",
            tool_name="navigation_replace_final_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "destination_replaced": True,
                    "new_waypoints": [LEIPZIG_ID, FRANKFURT_ID, BARCELONA_ID],
                    "new_routes": [
                        LEIPZIG_FRANKFURT_ROUTE_ID,
                        FRANKFURT_BARCELONA_ROUTE_ID,
                    ],
                },
            },
        )
    )
    assert completed.tool_calls == ()
    assert completed.text is not None
    assert "replaced the destination" in completed.text.casefold()
    return completed


def run_to_range_assessment(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    tools: tuple[dict[str, Any], ...] = FULL_TOOLS,
    remaining_range_km: float = 466.0,
    range_question: str = RANGE_QUESTION,
) -> Any:
    run_to_replaced_navigation(runtime, context_id=context_id, tools=tools)
    status = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-range-question",
            text=range_question,
        )
    )
    _assert_call(status, "get_charging_specs_and_status", {})
    navigation = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-charging-status",
            tool_name="get_charging_specs_and_status",
            content=charging_status(remaining_range_km=remaining_range_km),
        )
    )
    _assert_call(
        navigation,
        "get_current_navigation_state",
        {"detailed_information": True},
    )
    assessment = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-fresh-navigation",
            tool_name="get_current_navigation_state",
            content=new_navigation_state(),
        )
    )
    assert assessment.tool_calls == ()
    assert assessment.text is not None
    assert "need to charge" in assessment.text.casefold()
    return assessment


def run_to_charger_display(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    tools: tuple[dict[str, Any], ...] = FULL_TOOLS,
    charger_result: dict[str, Any] | None = None,
    range_question: str = RANGE_QUESTION,
    buffer_search_request: str = BUFFER_SEARCH_REQUEST,
) -> Any:
    run_to_range_assessment(
        runtime,
        context_id=context_id,
        tools=tools,
        range_question=range_question,
    )
    distance = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-buffer-search",
            text=buffer_search_request,
        )
    )
    _assert_call(
        distance,
        "get_distance_by_soc",
        {"initial_state_of_charge": 98, "final_state_of_charge": 15},
    )
    search = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-buffer-distance",
            tool_name="get_distance_by_soc",
            content=distance_to_buffer(),
        )
    )
    _assert_call(
        search,
        "search_poi_along_the_route",
        {
            "route_id": FRANKFURT_BARCELONA_ROUTE_ID,
            "category_poi": "charging_stations",
            "at_kilometer": 50,
            "filters": ["charging_stations::has_available_plug"],
        },
    )
    displayed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-fastned",
            tool_name="search_poi_along_the_route",
            content=charger_result or fastned_result(),
        )
    )
    assert displayed.tool_calls == ()
    assert displayed.text is not None and "Fastned" in displayed.text
    return displayed


def test_base86_exact_success_chain_uses_buffer_evidence_then_calls_displayed_charger() -> (
    None
):
    client = Base86Client()
    runtime = runtime_for(client)
    context_id = "base86-exact-success"

    displayed = run_to_charger_display(runtime, context_id=context_id)
    assert "50" in displayed.text
    # The initial 466 km range covers the 336.33 km first leg, so this result
    # proves the assessment used the full 1594.08 km remaining route.
    session = runtime.sessions.get(context_id)
    assert session is not None
    route_summaries = session.evidence.for_proposition(
        "active_route_charging_summary", current_state_only=True
    )
    assert route_summaries
    assert route_summaries[-1].value["remaining_route_distance_km"] == "1594.08"
    summaries = session.evidence.for_proposition(
        "active_route_charger_summary", current_state_only=True
    )
    assert summaries
    summary = summaries[-1]
    assert summary.status is EvidenceStatus.KNOWN
    assert summary.source_kind is EvidenceSourceKind.DERIVED
    assert summary.value["station_id"] == FASTNED_ID
    assert summary.value["station_name"] == "Fastned"
    assert summary.value["phone_number"] == FASTNED_PHONE
    assert any(
        session.evidence.evidence[parent_id].source_kind is EvidenceSourceKind.TOOL
        and session.evidence.evidence[parent_id].proposition
        == "active_route_charging_station_candidates"
        for parent_id in summary.derived_from
    )

    phone = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-phone-question",
            text=PHONE_QUESTION,
        )
    )
    assert phone.tool_calls == ()
    assert phone.text is not None and FASTNED_PHONE in phone.text

    call = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-call-confirmation",
            text="Yes, please call them.",
        )
    )
    _assert_call(call, "call_phone_by_number", {"phone_number": FASTNED_PHONE})
    assert call.terminal
    assert runtime.sessions.get(context_id) is None
    assert runtime.sessions.get_tombstone(context_id) is not None
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "client_type",
    [FailingInitialIntentClient, EmptyInitialIntentClient],
    ids=["llm-failure", "empty-intent"],
)
def test_base86_full_chain_does_not_depend_on_initial_intent_llm(
    client_type: type[Base86Client],
) -> None:
    client = client_type()
    runtime = runtime_for(client)
    context_id = f"base86-deterministic-{client_type.__name__.casefold()}"

    run_to_charger_display(runtime, context_id=context_id)
    phone = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-phone-question",
            text=PHONE_QUESTION,
        )
    )
    assert phone.tool_calls == ()
    assert phone.text is not None and FASTNED_PHONE in phone.text

    call = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-call-confirmation",
            text="Yes, please call them.",
        )
    )
    _assert_call(call, "call_phone_by_number", {"phone_number": FASTNED_PHONE})
    assert call.terminal
    assert client.intent_calls == 0
    assert client.action_calls == 0


def test_base86_direct_route_selection_without_relative_clause() -> None:
    runtime = runtime_for(Base86Client())

    completed = run_to_replaced_navigation(
        runtime,
        context_id="base86-direct-route-selection",
        selection_text=DIRECT_ROUTE_SELECTION,
    )

    assert completed.text is not None
    assert "replaced the destination" in completed.text.casefold()


def test_base86_formal_range_and_buffer_wording_uses_active_route_workflow() -> None:
    runtime = runtime_for(Base86Client())
    context_id = "base86-formal-range-buffer"

    displayed = run_to_charger_display(
        runtime,
        context_id=context_id,
        range_question=FORMAL_RANGE_QUESTION,
        buffer_search_request=FORMAL_BUFFER_SEARCH_REQUEST,
    )

    assert displayed.text is not None and "Fastned" in displayed.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    summaries = session.evidence.for_proposition(
        "active_route_charging_summary", current_state_only=True
    )
    assert summaries
    assert summaries[-1].value["state_of_charge"] == "98"
    assert summaries[-1].value["needs_charging"] is True

    phone = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-phone-question",
            text=FORMAL_PHONE_QUESTION,
        )
    )
    assert phone.tool_calls == ()
    assert phone.text is not None and FASTNED_PHONE in phone.text
    offers = session.evidence.for_proposition(
        "active_route_charger_call_offer", current_state_only=True
    )
    assert len(offers) == 1
    assert offers[0].value == {
        "station_name": "Fastned",
        "phone_number": FASTNED_PHONE,
    }

    call = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-call-request",
            text=FORMAL_DEMONSTRATIVE_CALL,
        )
    )
    _assert_call(call, "call_phone_by_number", {"phone_number": FASTNED_PHONE})
    assert call.terminal


def test_base86_active_route_range_requires_completed_destination_replacement() -> None:
    runtime = runtime_for(Base86Client())
    context_id = "base86-range-without-completed-replacement"
    run_to_route_prompt(runtime, context_id=context_id)
    replacement = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-select-route",
            text=ROUTE_SELECTION,
        )
    )
    assert replacement.tool_calls[0]["tool_name"] == (
        "navigation_replace_final_destination"
    )

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-range-question",
            text=FORMAL_RANGE_QUESTION,
        )
    )

    assert blocked.tool_calls == ()
    assert "waiting" in (blocked.text or "").casefold()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert all(
        goal.desired_outcome.get("active_route_charging_workflow") is not True
        for goal in session.intent.goals
    )


@pytest.mark.parametrize(
    "unsafe_text",
    [
        (
            'The driver said, "I have 98% battery and I am concerned about the '
            "range for Barcelona. Can you help me find a charging station to reach "
            'Barcelona safely?"'
        ),
        (
            "I have 98% battery and I am concerned about the range for Barcelona. "
            "Please do not help me find a charging station to reach Barcelona "
            "safely."
        ),
    ],
    ids=["quoted-report", "negated"],
)
def test_base86_active_route_range_parser_rejects_quoted_or_negated_text(
    unsafe_text: str,
) -> None:
    assert (
        _active_route_range_question_soc(
            unsafe_text,
            destination_name="Barcelona",
        )
        is None
    )


def test_base86_preview_declares_fastest_default_without_navigation_change() -> None:
    runtime = runtime_for(Base86Client())

    prompt = run_to_route_prompt(runtime, context_id="base86-policy-wording")

    assert prompt.text is not None
    folded = prompt.text.casefold()
    assert "selected the fastest route by default for this segment" in folded
    assert "it is also the shortest route" in folded
    assert "there are 2 further route alternatives" in folded
    assert "have not changed navigation yet" in folded


@pytest.mark.parametrize(
    "unsafe_request",
    [
        "Please do not call the number for Fastned once you have it.",
        ('The driver said, "Please call the number for Fastned once you have it."'),
        "Please call the number for Ionity once you have it.",
        'The driver said, "Yes, please call that number."',
        f"Yes, please call that number at {FASTNED_PHONE}.",
    ],
    ids=[
        "negated",
        "quoted-report",
        "station-name-mismatch",
        "quoted-demonstrative",
        "extra-number",
    ],
)
def test_base86_phone_offer_rejects_unsafe_or_mismatched_call_request(
    unsafe_request: str,
) -> None:
    runtime = runtime_for(EmptyInitialIntentClient())
    context_id = f"base86-phone-reject-{abs(hash(unsafe_request))}"
    run_to_charger_display(runtime, context_id=context_id)
    phone = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-phone-question",
            text=FORMAL_PHONE_QUESTION,
        )
    )
    assert phone.tool_calls == ()
    assert phone.text is not None and FASTNED_PHONE in phone.text

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-unsafe-call",
            text=unsafe_request,
        )
    )

    assert blocked.tool_calls == ()
    assert not blocked.terminal


def test_base86_phone_offer_must_be_from_immediately_preceding_user_request() -> None:
    runtime = runtime_for(EmptyInitialIntentClient())
    context_id = "base86-non-adjacent-phone-offer"
    run_to_charger_display(runtime, context_id=context_id)
    phone = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-phone-question",
            text=FORMAL_PHONE_QUESTION,
        )
    )
    assert phone.tool_calls == ()
    assert phone.text is not None and FASTNED_PHONE in phone.text

    intervening = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-intervening",
            text="Thanks, I will think about it.",
        )
    )
    assert intervening.tool_calls == ()

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-late-call",
            text=FORMAL_DEMONSTRATIVE_CALL,
        )
    )
    assert blocked.tool_calls == ()
    assert not blocked.terminal
    assert "immediately preceding" in (blocked.text or "").casefold()


def test_base86_stale_phone_offer_cannot_authorize_named_call() -> None:
    runtime = runtime_for(EmptyInitialIntentClient())
    context_id = "base86-stale-phone-offer"
    run_to_charger_display(runtime, context_id=context_id)
    phone = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-phone-question",
            text=FORMAL_PHONE_QUESTION,
        )
    )
    assert phone.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None
    session.evidence.invalidate_tool_state()

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-call-request",
            text=FORMAL_CALL_REQUEST,
        )
    )

    assert blocked.tool_calls == ()
    assert not blocked.terminal
    assert "no longer current" in (blocked.text or "").casefold()


@pytest.mark.parametrize(
    "unsafe_text",
    [
        (
            "Do not search for a charging station. I want to maintain at least "
            "15% battery charge as a safety buffer near where it reaches 15%."
        ),
        (
            'The driver said, "Please search for a charging station where my 15% '
            'safety buffer reaches approximately 15%."'
        ),
    ],
    ids=["negated", "quoted-report"],
)
def test_base86_buffer_parser_rejects_negated_or_quoted_text(
    unsafe_text: str,
) -> None:
    assert _active_route_buffer_search_soc(unsafe_text) is None


def test_base86_search_waits_for_successful_navigation_replacement_result() -> None:
    runtime = runtime_for(Base86Client())
    context_id = "base86-pending-replacement"
    run_to_route_prompt(runtime, context_id=context_id)
    pending = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-select-route",
            text=ROUTE_SELECTION,
        )
    )
    assert pending.tool_calls[0]["tool_name"] == "navigation_replace_final_destination"

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-premature-search",
            text=BUFFER_SEARCH_REQUEST,
        )
    )
    assert blocked.tool_calls == ()
    assert not blocked.terminal
    assert "waiting" in (blocked.text or "").casefold()


def test_base86_stale_charger_evidence_cannot_authorize_call() -> None:
    runtime = runtime_for(Base86Client())
    context_id = "base86-stale-charger"
    run_to_charger_display(runtime, context_id=context_id)
    session = runtime.sessions.get(context_id)
    assert session is not None
    session.evidence.invalidate_tool_state()

    blocked = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-call",
            text="Please call Fastned.",
        )
    )
    assert blocked.tool_calls == ()
    assert not blocked.terminal
