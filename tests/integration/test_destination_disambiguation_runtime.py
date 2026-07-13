from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import (
    DecisionProposal,
    IntentFrame,
    IntentKind,
)
from track_1_agent_under_test.car_guard.planning.intent_grounding import (
    parse_adjacent_navigation_destination_reply,
    parse_descriptive_navigation_destination_request,
    recover_named_navigation_destination_replacement_intent,
    recover_named_navigation_waypoint_delete_intent,
)
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
)


POLICY = "Follow the current navigation safety policy."
START_ID = "loc_origin_100"
OLD_DESTINATION_ID = "loc_old_200"
NEW_DESTINATION_ID = "loc_new_300"
FASTEST_ROUTE_ID = "rll_origin_new_fast"
SHORTEST_ROUTE_ID = "rll_origin_new_short"
GENERIC_ADJACENT_REPLY = (
    "Oh, I'm excited to visit Riverton's electric mobility galleries! Yes, Riverton, "
    "please. And remember, I want the shortest route."
)
GENERIC_MEAN_ADJACENT_REPLY = (
    "Oh, I mean Riverton! I'm really excited to visit the electric mobility "
    "galleries there. Please find the shortest route to Riverton."
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
REPLACE_TOOL = tool(
    "navigation_replace_final_destination",
    {
        "new_destination_id": {"type": "string"},
        "route_id_leading_to_new_destination": {"type": "string"},
    },
    ["new_destination_id", "route_id_leading_to_new_destination"],
)
DELETE_TOOL = tool(
    "navigation_delete_waypoint",
    {
        "waypoint_id_to_delete": {"type": "string"},
        "route_id_without_waypoint": {"type": "string"},
    },
    ["waypoint_id_to_delete", "route_id_without_waypoint"],
)
FULL_TOOLS = (
    CURRENT_NAVIGATION_TOOL,
    LOCATION_TOOL,
    ROUTES_TOOL,
    REPLACE_TOOL,
    DELETE_TOOL,
)


class EmptyClient:
    def __init__(self) -> None:
        self.intent_calls = 0
        self.action_calls = 0
        self.intent = IntentDraft.model_validate(
            {
                "language": "en",
                "intent_kind": "conversation",
                "call_for_action": False,
                "goals": [],
            }
        )

    def generate(self, *, messages, response_model, critic=False):
        del messages, critic
        if response_model is IntentDraft:
            self.intent_calls += 1
            return SimpleNamespace(value=self.intent)
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError("destination replacement must remain deterministic")
        raise AssertionError(f"unexpected response model: {response_model}")


def runtime_for(client: EmptyClient) -> CARGuardOrchestrator:
    return CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=24),
        client_factory=lambda session: client,
    )


def result_event(
    *, context_id: str, message_id: str, tool_name: str, content: Any
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        tool_results=({"toolName": tool_name, "content": content},),
    )


def navigation_state() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [START_ID, OLD_DESTINATION_ID],
            "routes_to_final_destination_id": ["rll_origin_old_1"],
            "details": {
                "waypoints": [
                    {"id": START_ID, "name": "Originburg"},
                    {"id": OLD_DESTINATION_ID, "name": "Oldtown"},
                ]
            },
        },
    }


def route_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": FASTEST_ROUTE_ID,
                    "start_id": START_ID,
                    "destination_id": NEW_DESTINATION_ID,
                    "name_via": "A10",
                    "distance_km": 210.0,
                    "duration_hours": 2,
                    "duration_minutes": 0,
                    "road_types": ["highway"],
                    "includes_toll": False,
                    "alias": ["first", "fastest"],
                },
                {
                    "route_id": SHORTEST_ROUTE_ID,
                    "start_id": START_ID,
                    "destination_id": NEW_DESTINATION_ID,
                    "name_via": "B20",
                    "distance_km": 190.0,
                    "duration_hours": 2,
                    "duration_minutes": 20,
                    "road_types": ["country road"],
                    "includes_toll": False,
                    "alias": ["second", "shortest"],
                },
            ]
        },
    }


@pytest.mark.parametrize(
    ("route_choice", "expected_route_id"),
    [("shortest", SHORTEST_ROUTE_ID), ("fastest", FASTEST_ROUTE_ID)],
)
def test_compound_named_replacement_preserves_and_selects_route_alias(
    route_choice: str, expected_route_id: str
) -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = f"compound-replacement-{route_choice}"

    current = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=POLICY,
            user_text=(
                "I want to change my final destination to Riverton. "
                f"Can you please find the {route_choice} route to Riverton?"
            ),
            live_tools=FULL_TOOLS,
        )
    )
    assert current.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    assert client.intent_calls == 0

    location = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=navigation_state(),
        )
    )
    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Riverton"},
        },
    )
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": NEW_DESTINATION_ID}},
        )
    )
    assert routes.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": START_ID,
                "destination_id": NEW_DESTINATION_ID,
            },
        },
    )
    replacement = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=route_result(),
        )
    )
    assert replacement.tool_calls == (
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": NEW_DESTINATION_ID,
                "route_id_leading_to_new_destination": expected_route_id,
            },
        },
    )
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "I want to change my final destination to Riverton. "
            "Can you find the shortest route to Laketown?"
        ),
        (
            "If needed, I want to change my final destination to Riverton. "
            "Can you find the shortest route to Riverton?"
        ),
        (
            'She said, "Change my final destination to Riverton and find the '
            'shortest route."'
        ),
        (
            "I do not want to change my final destination to Riverton. "
            "Can you find the shortest route to Riverton?"
        ),
        (
            "I need to change my final destination. Please change it from Oldtown "
            "to Riverton Lock Doors. Can you find the shortest route to Riverton "
            "Lock Doors?"
        ),
        (
            "I want to change my final destination to Riverton Play Music. Can "
            "you find the shortest route to Riverton Play Music?"
        ),
        "Please change my final destination to Riverton Check Tire Pressure.",
        (
            "I want to change my final destination to Riverton Call Alice. Can "
            "you find the shortest route to Riverton Call Alice?"
        ),
        (
            "I want to change my final destination to Riverton Turn On Seat "
            "Heating. Can you find the shortest route to Riverton Turn On Seat "
            "Heating?"
        ),
        "Please change my final destination to Riverton Open Sunroof.",
    ],
)
def test_compound_named_replacement_rejects_conflicting_or_unsafe_text(
    utterance: str,
) -> None:
    seed = IntentFrame(
        language="en",
        call_for_action=False,
        goals=[],
        intent_kind=IntentKind.CONVERSATION,
    )
    recovered = recover_named_navigation_destination_replacement_intent(
        utterance, seed, turn_id="synthetic-turn"
    )
    assert recovered is seed


def test_direct_from_to_replacement_preserves_previous_destination() -> None:
    seed = IntentFrame(
        language="en",
        call_for_action=False,
        goals=[],
        intent_kind=IntentKind.CONVERSATION,
    )
    recovered = recover_named_navigation_destination_replacement_intent(
        "I need to change my final destination. Please change it from Oldtown "
        "to Riverton. Can you find the shortest route to Riverton?",
        seed,
        turn_id="direct-from-to",
    )
    assert recovered is not seed
    assert recovered.goals[0].desired_outcome == {
        "new_destination_name": "Riverton",
        "previous_destination_name": "Oldtown",
        "route_choice_alias": "shortest",
    }


@pytest.mark.parametrize(
    "utterance",
    [
        "Please remove Brookfield from my current route, then turn on the headlights.",
        "Please remove Brookfield from my current route, then lock the doors.",
        "Please remove Brookfield from my current route, then navigate to Laketown.",
        "Please remove Brookfield from my current route, then play music.",
        "Please remove Brookfield from my current route, then turn on seat heating.",
    ],
)
def test_named_waypoint_delete_rejects_unconsumed_action_tail(
    utterance: str,
) -> None:
    seed = IntentFrame(
        language="en",
        call_for_action=False,
        goals=[],
        intent_kind=IntentKind.CONVERSATION,
    )
    recovered = recover_named_navigation_waypoint_delete_intent(
        utterance,
        seed,
        turn_id="waypoint-delete-tail",
    )
    assert recovered is seed


def test_named_waypoint_delete_rejects_route_destination_mismatching_fresh_state() -> (
    None
):
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = "waypoint-delete-route-destination-mismatch"
    request = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-request",
            context_id=context_id,
            system_policy=POLICY,
            user_text=(
                "Can you please remove Brookfield from my route and then show me "
                "the shortest route to Laketown?"
            ),
            live_tools=FULL_TOOLS,
        )
    )
    assert request.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )

    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content={
                "status": "SUCCESS",
                "result": {
                    "navigation_active": True,
                    "waypoints_id": [
                        "loc_alpha_101",
                        "loc_brook_202",
                        "loc_gamma_303",
                        "loc_river_404",
                    ],
                    "routes_to_final_destination_id": [
                        "rll_alpha_brook_1",
                        "rll_brook_gamma_1",
                        "rll_gamma_river_1",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_alpha_101", "name": "Alphaville"},
                            {"id": "loc_brook_202", "name": "Brookfield"},
                            {"id": "loc_gamma_303", "name": "Gammaton"},
                            {"id": "loc_river_404", "name": "Riverton"},
                        ]
                    },
                },
            },
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert client.intent_calls == 0
    assert client.action_calls == 0


def test_descriptive_request_and_adjacent_reply_parsers_are_bounded() -> None:
    initial = parse_descriptive_navigation_destination_request(
        "I need to change my final destination. I'd like to go to a European "
        "city known for electric vehicles instead of Oldtown. Can you find the "
        "shortest route there?"
    )
    assert initial is not None
    assert initial.previous_destination_name == "Oldtown"
    assert initial.route_choice_alias == "shortest"

    reply = parse_adjacent_navigation_destination_reply(
        "Yes, Riverton, please. I want the shortest route to Riverton.",
        inherited_route_choice=initial.route_choice_alias,
    )
    assert reply is not None
    assert reply.destination_name == "Riverton"
    assert reply.route_choice_alias == "shortest"

    descriptive_reply = parse_adjacent_navigation_destination_reply(
        "I'm excited about visiting Riverton's botanical garden. So yes, "
        "Riverton. And please, make sure it's the shortest route.",
        inherited_route_choice=initial.route_choice_alias,
    )
    assert descriptive_reply is not None
    assert descriptive_reply.destination_name == "Riverton"
    assert descriptive_reply.route_choice_alias == "shortest"

    generalized_reply = parse_adjacent_navigation_destination_reply(
        "I'm interested in visiting Laketown's historic library. So yes, "
        "Laketown. And please, make sure it's the shortest route.",
        inherited_route_choice=initial.route_choice_alias,
    )
    assert generalized_reply is not None
    assert generalized_reply.destination_name == "Laketown"
    assert generalized_reply.route_choice_alias == "shortest"

    multiword_city = parse_adjacent_navigation_destination_reply(
        "New Riverton.",
        inherited_route_choice="shortest",
    )
    assert multiword_city is not None
    assert multiword_city.destination_name == "New Riverton"

    multiword_visit_object = parse_adjacent_navigation_destination_reply(
        "I'm interested in visiting Riverton's natural history archive. So yes, "
        "Riverton. And please, make sure it's the shortest route.",
        inherited_route_choice="shortest",
    )
    assert multiword_visit_object is not None
    assert multiword_visit_object.destination_name == "Riverton"

    action_homonym_visit_object = parse_adjacent_navigation_destination_reply(
        "I'm interested in visiting Riverton's open air markets. So yes, "
        "Riverton. And please, make sure it's the shortest route.",
        inherited_route_choice="shortest",
    )
    assert action_homonym_visit_object is not None
    assert action_homonym_visit_object.destination_name == "Riverton"

    assert (
        parse_adjacent_navigation_destination_reply(
            "Riverton or Laketown, whichever is closer.",
            inherited_route_choice="shortest",
        )
        is None
    )


def test_adjacent_destination_accepts_generic_excited_visit_reply() -> None:
    reply = parse_adjacent_navigation_destination_reply(
        GENERIC_ADJACENT_REPLY,
        inherited_route_choice="shortest",
    )
    assert reply is not None
    assert reply.destination_name == "Riverton"
    assert reply.route_choice_alias == "shortest"
    assert reply.route_choice_from_reply


@pytest.mark.parametrize(
    ("reply_text", "destination"),
    [
        (GENERIC_MEAN_ADJACENT_REPLY, "Riverton"),
        (
            "Oh, I mean New Riverton! I am excited to visit the mobility gallery "
            "there. Please take the fastest route to New Riverton.",
            "New Riverton",
        ),
    ],
)
def test_adjacent_destination_accepts_mean_city_then_route_reply(
    reply_text: str, destination: str
) -> None:
    route = "fastest" if "fastest route" in reply_text else "shortest"
    reply = parse_adjacent_navigation_destination_reply(
        reply_text,
        inherited_route_choice=route,
    )
    assert reply is not None
    assert reply.destination_name == destination
    assert reply.route_choice_alias == route
    assert reply.route_choice_from_reply


@pytest.mark.parametrize(
    ("reply_text", "inherited_route"),
    [
        (
            GENERIC_MEAN_ADJACENT_REPLY.replace(
                "route to Riverton", "route to Laketown"
            ),
            "shortest",
        ),
        (
            GENERIC_MEAN_ADJACENT_REPLY.replace(
                "electric mobility galleries", "lock the doors"
            ),
            "shortest",
        ),
        (GENERIC_MEAN_ADJACENT_REPLY + " Then play music.", "shortest"),
        (GENERIC_MEAN_ADJACENT_REPLY, "fastest"),
        (f'"{GENERIC_MEAN_ADJACENT_REPLY}"', "shortest"),
        (f"If I decide to travel, {GENERIC_MEAN_ADJACENT_REPLY}", "shortest"),
    ],
)
def test_mean_city_then_route_reply_rejects_conflict_or_scope_expansion(
    reply_text: str, inherited_route: str
) -> None:
    assert (
        parse_adjacent_navigation_destination_reply(
            reply_text,
            inherited_route_choice=inherited_route,
        )
        is None
    )


@pytest.mark.parametrize(
    ("reply", "inherited_route"),
    [
        (
            "Oh, I'm excited to visit Riverton's mobility galleries! Yes, Laketown, "
            "please. And remember, I want the shortest route.",
            "shortest",
        ),
        (GENERIC_ADJACENT_REPLY, "fastest"),
        (GENERIC_ADJACENT_REPLY + " Then lock the doors.", "shortest"),
        (f'"{GENERIC_ADJACENT_REPLY}"', "shortest"),
        (f"If I decide to travel, {GENERIC_ADJACENT_REPLY}", "shortest"),
    ],
)
def test_generic_excited_visit_reply_rejects_unsafe_or_conflicting_scope(
    reply: str,
    inherited_route: str,
) -> None:
    assert (
        parse_adjacent_navigation_destination_reply(
            reply,
            inherited_route_choice=inherited_route,
        )
        is None
    )


@pytest.mark.parametrize(
    "reply",
    [
        (
            "I'm excited about visiting Riverton's electric vehicle museum and "
            "lock the doors. So yes, Riverton. And please, make sure it's the "
            "shortest route."
        ),
        (
            "I'm excited about visiting Riverton's electric vehicle museum and "
            "play music. So yes, Riverton. And please, make sure it's the shortest "
            "route."
        ),
        (
            "I'm excited about visiting Riverton's electric vehicle museum and "
            "check tire pressure. So yes, Riverton. And please, make sure it's the "
            "shortest route."
        ),
        (
            "I'm excited about visiting Riverton's electric vehicle museum and "
            "unlock the car. So yes, Riverton. And please, make sure it's the "
            "shortest route."
        ),
        (
            "I'm interested in visiting Riverton's lock the doors. So yes, "
            "Riverton. And please, make sure it's the shortest route."
        ),
        (
            "I'm interested in visiting Riverton's play music. So yes, Riverton. "
            "And please, make sure it's the shortest route."
        ),
        (
            "I'm interested in visiting Riverton's check tire pressure. So yes, "
            "Riverton. And please, make sure it's the shortest route."
        ),
        (
            "I'm interested in visiting Riverton's call Alice. So yes, Riverton. "
            "And please, make sure it's the shortest route."
        ),
        (
            "I'm interested in visiting Riverton's turn on seat heating. So yes, "
            "Riverton. And please, make sure it's the shortest route."
        ),
        (
            "I'm interested in visiting Riverton's open sunroof. So yes, "
            "Riverton. And please, make sure it's the shortest route."
        ),
    ],
)
def test_adjacent_interest_description_rejects_embedded_action(reply: str) -> None:
    assert (
        parse_adjacent_navigation_destination_reply(
            reply,
            inherited_route_choice="shortest",
        )
        is None
    )


@pytest.mark.parametrize(
    "action_tail",
    [
        "lock the doors",
        "play music",
        "check tire pressure",
        "unlock the car",
        "adjust the climate",
        "navigate home",
    ],
)
def test_initial_destination_description_rejects_embedded_action(
    action_tail: str,
) -> None:
    request = (
        "I need to change my final destination. I'd like to go to a European city "
        f"known for electric vehicles and {action_tail} instead of Oldtown. Can "
        "you find the shortest route there?"
    )
    assert parse_descriptive_navigation_destination_request(request) is None


@pytest.mark.parametrize(
    "description",
    [
        "a European city with call Alice",
        "a European city with lock doors",
        "a European city known for play music",
        "a European city known for check tire pressure",
        "a European city known for turn on seat heating",
        "a European city known for open sunroof",
    ],
)
def test_initial_destination_description_rejects_control_words_without_connector(
    description: str,
) -> None:
    request = (
        "I need to change my final destination. I'd like to go to "
        f"{description} instead of Oldtown. Can you find the shortest route there?"
    )
    assert parse_descriptive_navigation_destination_request(request) is None


@pytest.mark.parametrize(
    "description_tail",
    ["open air markets", "lower emissions", "call centers", "heat pumps"],
)
def test_initial_destination_description_allows_action_word_homonyms(
    description_tail: str,
) -> None:
    request = (
        "I need to change my final destination. I'd like to go to a European city "
        f"known for {description_tail} instead of Oldtown. Can you find the "
        "shortest route there?"
    )
    parsed = parse_descriptive_navigation_destination_request(request)
    assert parsed is not None
    assert parsed.previous_destination_name == "Oldtown"


@pytest.mark.parametrize(
    "name",
    [
        "Open Lake",
        "Lower Saxony",
        "Open Sunroof Museum",
        "Lower Easton",
        "Call Center City",
        "Heat Springs",
        "Change Islands",
        "Email Innovation Center",
        "Call Innovation Center",
        "Phone History Museum",
        "Message Research Lab",
    ],
)
def test_destination_names_allow_action_word_homonyms(name: str) -> None:
    adjacent = parse_adjacent_navigation_destination_reply(
        f"{name}.",
        inherited_route_choice="shortest",
    )
    assert adjacent is not None
    assert adjacent.destination_name == name

    seed = IntentFrame(
        language="en",
        call_for_action=False,
        goals=[],
        intent_kind=IntentKind.CONVERSATION,
    )
    recovered = recover_named_navigation_destination_replacement_intent(
        f"I want to change my final destination to {name}. Can you find the "
        f"shortest route to {name}?",
        seed,
        turn_id=f"homonym-{name}",
    )
    assert recovered is not seed
    assert recovered.goals[0].desired_outcome["new_destination_name"] == name


@pytest.mark.parametrize(
    "action_phrase",
    [
        "email Alice",
        "send email Alice",
        "text Alice",
        "phone Alice",
        "close doors",
        "raise windows",
        "lower volume",
        "honk horn",
        "cool cabin",
        "tell a joke",
        "start engine",
        "enable air conditioning",
        "disable seat heating",
        "activate cruise control",
        "stop navigation",
        "defrost front window",
        "increase fan speed",
        "decrease fan speed",
        "warm driver seat",
        "cool driver seat",
        "ventilate passenger seat",
        "warm cabin",
    ],
)
def test_destination_action_phrase_matrix_rejects_every_entry_point(
    action_phrase: str,
) -> None:
    descriptive = (
        "I need to change my final destination. I'd like to go to a European city "
        f"known for {action_phrase} instead of Oldtown. Can you find the shortest "
        "route there?"
    )
    assert parse_descriptive_navigation_destination_request(descriptive) is None

    title_phrase = action_phrase.title()
    assert (
        parse_adjacent_navigation_destination_reply(
            f"Riverton {title_phrase}.",
            inherited_route_choice="shortest",
        )
        is None
    )
    assert (
        parse_adjacent_navigation_destination_reply(
            f"I'm interested in visiting Riverton's {action_phrase}. So yes, "
            "Riverton. And please, make sure it's the shortest route.",
            inherited_route_choice="shortest",
        )
        is None
    )

    seed = IntentFrame(
        language="en",
        call_for_action=False,
        goals=[],
        intent_kind=IntentKind.CONVERSATION,
    )
    direct = recover_named_navigation_destination_replacement_intent(
        f"I want to change my final destination to Riverton {title_phrase}. Can "
        f"you find the shortest route to Riverton {title_phrase}?",
        seed,
        turn_id=f"action-matrix-{action_phrase}",
    )
    assert direct is seed


@pytest.mark.parametrize(
    "reply",
    [
        "Okay.",
        "Yes.",
        "Thanks.",
        "Cancel.",
        "Help.",
        "What?",
        "Tomorrow.",
        "Shortest.",
        "Use Riverton.",
        "Riverton Lock Doors.",
        "Riverton Play Music.",
        "Riverton Check Tire Pressure.",
        "Riverton Call Alice.",
        "Riverton Turn On Seat Heating.",
        "Riverton Open Sunroof.",
        "Never mind. Yes, Riverton.",
    ],
)
def test_adjacent_destination_standalone_rejects_control_or_noop_reply(
    reply: str,
) -> None:
    assert (
        parse_adjacent_navigation_destination_reply(
            reply,
            inherited_route_choice="shortest",
        )
        is None
    )


def _begin_open_destination_clarification(
    runtime: CARGuardOrchestrator, *, context_id: str
) -> Any:
    return runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-initial",
            context_id=context_id,
            system_policy=POLICY,
            user_text=(
                "I need to change my final destination. I'd like to go to a "
                "European city known for electric vehicles instead of Oldtown. "
                "Can you find the shortest route there?"
            ),
            live_tools=FULL_TOOLS,
        )
    )


def _begin_direct_from_to_replacement(
    runtime: CARGuardOrchestrator, *, context_id: str
) -> Any:
    return runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-request",
            context_id=context_id,
            system_policy=POLICY,
            user_text=(
                "I need to change my final destination. Please change it from "
                "Oldtown to Riverton. Can you find the shortest route to Riverton?"
            ),
            live_tools=FULL_TOOLS,
        )
    )


def test_open_destination_clarification_never_guesses_then_resolves_one_city() -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = "open-destination-valid"

    question = _begin_open_destination_clarification(runtime, context_id=context_id)
    assert question.tool_calls == ()
    assert question.text is not None and "which city" in question.text.casefold()
    assert client.intent_calls == 0
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.pending_destination_clarification is not None
    assert session.authorized_action_goal_ids == set()
    assert session.budget.attempted_sets == set()

    current = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-city",
            context_id=context_id,
            user_text=(
                "I'm excited about visiting Riverton's botanical garden. "
                "So yes, Riverton. And please, make sure it's the shortest route."
            ),
        )
    )
    assert current.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    assert session.pending_destination_clarification is None
    assert session.intent is not None and len(session.intent.goals) == 1
    goal = session.intent.goals[0]
    assert goal.semantic_operation == "navigation_replace_final_destination"
    assert goal.desired_outcome == {
        "new_destination_name": "Riverton",
        "previous_destination_name": "Oldtown",
        "route_choice_alias": "shortest",
    }
    assert session.grounded_desired_values_by_goal[goal.goal_id] == (
        goal.desired_outcome
    )
    assert (
        session.grounded_value_sources_by_goal[goal.goal_id][
            "previous_destination_name"
        ]
        == session.intent.intent_source_turn_ids[0]
    )
    assert session.authorized_action_goal_ids == {goal.goal_id}
    assert session.budget.attempted_sets == set()
    assert client.intent_calls == 0


def test_generic_adjacent_reply_preserves_fresh_reverify_before_replacement() -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = "generic-adjacent-fresh-reverify"

    question = _begin_open_destination_clarification(runtime, context_id=context_id)
    assert question.tool_calls == ()
    current = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-city",
            context_id=context_id,
            user_text=GENERIC_ADJACENT_REPLY,
        )
    )
    assert current.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )

    location = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=navigation_state(),
        )
    )
    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Riverton"},
        },
    )
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": NEW_DESTINATION_ID}},
        )
    )
    assert routes.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": START_ID,
                "destination_id": NEW_DESTINATION_ID,
            },
        },
    )

    coalesced_routes = route_result()
    coalesced_routes["result"]["routes"][0]["alias"] = [
        "first",
        "fastest",
        "shortest",
    ]
    coalesced_routes["result"]["routes"][0]["distance_km"] = 180.0
    coalesced_routes["result"]["routes"][1]["alias"] = ["second"]
    recheck = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=coalesced_routes,
        )
    )
    assert recheck.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    replacement = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state-recheck",
            tool_name="get_current_navigation_state",
            content=navigation_state(),
        )
    )
    assert replacement.tool_calls == (
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": NEW_DESTINATION_ID,
                "route_id_leading_to_new_destination": FASTEST_ROUTE_ID,
            },
        },
    )
    assert client.intent_calls == 0
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "reply",
    [
        "Riverton or Laketown, whichever is closer.",
        "Yes, Riverton, but use the fastest route.",
        "Thanks for explaining that.",
        "Cancel.",
        "Okay.",
        "Yes, Riverton, please. Turn on seat heating.",
        "Yes, Riverton, please. Lock doors.",
        "Yes, Riverton, please. Navigate to Laketown.",
        "Yes, Riverton, please. Turn on the headlights.",
        'She said, "Use Riverton."',
        "If Riverton works, use it.",
    ],
)
def test_open_destination_clarification_rejects_unsafe_adjacent_reply(
    reply: str,
) -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = f"open-destination-invalid-{abs(hash(reply))}"
    _begin_open_destination_clarification(runtime, context_id=context_id)

    blocked = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-reply",
            context_id=context_id,
            user_text=reply,
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "did not change" in blocked.text.casefold()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.pending_destination_clarification is None
    assert session.authorized_action_goal_ids == set()
    assert session.pending_calls == []
    assert session.budget.attempted_sets == set()
    assert client.intent_calls == 0


def test_pending_destination_stops_when_current_final_name_mismatches() -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = "open-destination-current-mismatch"
    _begin_open_destination_clarification(runtime, context_id=context_id)
    current = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-city",
            context_id=context_id,
            user_text="Riverton.",
        )
    )
    assert current.tool_calls[0]["tool_name"] == "get_current_navigation_state"

    wrong_state = navigation_state()
    wrong_state["result"]["details"]["waypoints"][-1]["name"] = "Wrongtown"
    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-wrong-state",
            tool_name="get_current_navigation_state",
            content=wrong_state,
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "stopped" in blocked.text.casefold()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.intent is None
    assert session.authorized_action_goal_ids == set()
    assert client.action_calls == 0


def test_direct_from_to_stops_before_lookup_when_current_final_mismatches() -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = "direct-from-to-current-mismatch"
    current = _begin_direct_from_to_replacement(runtime, context_id=context_id)
    assert current.tool_calls[0]["tool_name"] == "get_current_navigation_state"
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    goal = session.intent.goals[0]
    assert goal.desired_outcome["previous_destination_name"] == "Oldtown"
    assert session.grounded_desired_values_by_goal[goal.goal_id] == (
        goal.desired_outcome
    )

    wrong_state = navigation_state()
    wrong_state["result"]["details"]["waypoints"][-1]["name"] = "Wrongtown"
    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=wrong_state,
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "stopped" in blocked.text.casefold()
    assert session.intent is None
    assert session.authorized_action_goal_ids == set()
    assert client.action_calls == 0


def test_direct_from_to_rechecks_matching_previous_destination_before_set() -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = "direct-from-to-success"
    _begin_direct_from_to_replacement(runtime, context_id=context_id)
    location = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=navigation_state(),
        )
    )
    assert location.tool_calls[0]["tool_name"] == "get_location_id_by_location_name"
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": NEW_DESTINATION_ID}},
        )
    )
    assert routes.tool_calls[0]["tool_name"] == "get_routes_from_start_to_destination"
    recheck = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=route_result(),
        )
    )
    assert recheck.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    replacement = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state-recheck",
            tool_name="get_current_navigation_state",
            content=navigation_state(),
        )
    )
    assert replacement.tool_calls == (
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": NEW_DESTINATION_ID,
                "route_id_leading_to_new_destination": SHORTEST_ROUTE_ID,
            },
        },
    )
    assert client.action_calls == 0


def test_pending_destination_stops_when_previous_name_provenance_mismatches() -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = "open-destination-provenance-mismatch"
    _begin_open_destination_clarification(runtime, context_id=context_id)
    runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-city",
            context_id=context_id,
            user_text="Riverton.",
        )
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    goal_id = session.intent.goals[0].goal_id
    session.grounded_desired_values_by_goal[goal_id]["previous_destination_name"] = (
        "Wrongtown"
    )

    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=navigation_state(),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "stopped" in blocked.text.casefold()
    assert session.intent is None
    assert session.authorized_action_goal_ids == set()
    assert client.action_calls == 0


def test_pending_destination_rechecks_state_after_routes_before_set() -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = "open-destination-stale-before-set"
    _begin_open_destination_clarification(runtime, context_id=context_id)
    runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-city",
            context_id=context_id,
            user_text="Riverton.",
        )
    )
    location = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=navigation_state(),
        )
    )
    assert location.tool_calls[0]["tool_name"] == "get_location_id_by_location_name"
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": NEW_DESTINATION_ID}},
        )
    )
    assert routes.tool_calls[0]["tool_name"] == "get_routes_from_start_to_destination"
    recheck = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=route_result(),
        )
    )
    assert recheck.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )

    stale_state = navigation_state()
    stale_state["result"]["waypoints_id"][0] = "loc_changed_900"
    stale_state["result"]["routes_to_final_destination_id"][0] = "rll_changed_old_900"
    stale_state["result"]["details"]["waypoints"][0] = {
        "id": "loc_changed_900",
        "name": "Changedburg",
    }
    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-stale-state",
            tool_name="get_current_navigation_state",
            content=stale_state,
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "stopped" in blocked.text.casefold()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.intent is None
    assert session.authorized_action_goal_ids == set()
    assert client.action_calls == 0


@pytest.mark.parametrize("route_choice", ["shortest", "fastest"])
def test_destination_routes_reject_requested_metric_tie(
    route_choice: str,
) -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = f"destination-route-{route_choice}-tie"
    runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=POLICY,
            user_text=(
                "I want to change my final destination to Riverton. "
                f"Can you please find the {route_choice} route to Riverton?"
            ),
            live_tools=FULL_TOOLS,
        )
    )
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=navigation_state(),
        )
    )
    runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": NEW_DESTINATION_ID}},
        )
    )
    tied_routes = route_result()
    if route_choice == "shortest":
        tied_routes["result"]["routes"][0]["distance_km"] = 190.0
    else:
        tied_routes["result"]["routes"][1]["duration_hours"] = 2
        tied_routes["result"]["routes"][1]["duration_minutes"] = 0
    blocked = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=tied_routes,
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "delete_request",
    [
        (
            "Now the Brookfield stop is no longer needed. Please remove "
            "Brookfield from the route and use the shortest route after that."
        ),
        (
            "Can you please remove Brookfield from the route and then find the "
            "shortest route for me?"
        ),
        (
            "Now that's done, I also need to remove Brookfield from my route. "
            "It's no longer necessary. Can you find the shortest route after "
            "removing Brookfield?"
        ),
        (
            "I'm sorry, I thought you had already removed Brookfield. Can you "
            "please remove Brookfield from the route and then find the shortest "
            "route for me?"
        ),
        (
            "Okay, I understand. So, can you please remove Brookfield from my route "
            "now, and then show me the shortest route to Riverton?"
        ),
    ],
)
def test_open_clarification_replacement_then_named_stop_deletion(
    delete_request: str,
) -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = "open-destination-two-phase"
    origin_id = "loc_alpha_101"
    deleted_id = "loc_brook_202"
    segment_start_id = "loc_gamma_303"
    old_id = "loc_old_404"
    replacement_id = "loc_river_505"
    replacement_route_id = "rll_gamma_river_short"
    direct_route_id = "rll_alpha_gamma_short"

    _begin_open_destination_clarification(runtime, context_id=context_id)
    state_read = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-city",
            context_id=context_id,
            user_text=("Yes, Riverton, please. I want the shortest route to Riverton."),
        )
    )
    assert state_read.tool_calls[0]["tool_name"] == "get_current_navigation_state"
    location = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content={
                "status": "SUCCESS",
                "result": {
                    "navigation_active": True,
                    "waypoints_id": [
                        origin_id,
                        deleted_id,
                        segment_start_id,
                        old_id,
                    ],
                    "routes_to_final_destination_id": [
                        "rll_alpha_brook_1",
                        "rll_brook_gamma_1",
                        "rll_gamma_old_1",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": origin_id, "name": "Alphaville"},
                            {"id": deleted_id, "name": "Brookfield"},
                            {"id": segment_start_id, "name": "Gammaton"},
                            {"id": old_id, "name": "Oldtown"},
                        ],
                        "routes": [
                            {
                                "route_id": "rll_alpha_brook_1",
                                "start_id": origin_id,
                                "destination_id": deleted_id,
                                "includes_toll": True,
                            },
                            {
                                "route_id": "rll_brook_gamma_1",
                                "start_id": deleted_id,
                                "destination_id": segment_start_id,
                                "includes_toll": False,
                            },
                            {
                                "route_id": "rll_gamma_old_1",
                                "start_id": segment_start_id,
                                "destination_id": old_id,
                                "includes_toll": False,
                            },
                        ],
                    },
                },
            },
        )
    )
    assert location.tool_calls[0]["tool_name"] == "get_location_id_by_location_name"
    routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": replacement_id}},
        )
    )
    assert routes.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": segment_start_id,
                "destination_id": replacement_id,
            },
        },
    )
    replace = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-replacement-routes",
            tool_name="get_routes_from_start_to_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "routes": [
                        {
                            "route_id": "rll_gamma_river_fast",
                            "start_id": segment_start_id,
                            "destination_id": replacement_id,
                            "name_via": "A11",
                            "distance_km": 220,
                            "duration_hours": 2,
                            "duration_minutes": 0,
                            "road_types": ["highway"],
                            "includes_toll": False,
                            "alias": ["first", "fastest"],
                        },
                        {
                            "route_id": replacement_route_id,
                            "start_id": segment_start_id,
                            "destination_id": replacement_id,
                            "name_via": "B12",
                            "distance_km": 200,
                            "duration_hours": 2,
                            "duration_minutes": 20,
                            "road_types": ["country road"],
                            "includes_toll": False,
                            "alias": ["second", "shortest"],
                        },
                    ]
                },
            },
        )
    )
    assert replace.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    replace = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-replacement-state-recheck",
            tool_name="get_current_navigation_state",
            content={
                "status": "SUCCESS",
                "result": {
                    "navigation_active": True,
                    "waypoints_id": [
                        origin_id,
                        deleted_id,
                        segment_start_id,
                        old_id,
                    ],
                    "routes_to_final_destination_id": [
                        "rll_alpha_brook_1",
                        "rll_brook_gamma_1",
                        "rll_gamma_old_1",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": origin_id, "name": "Alphaville"},
                            {"id": deleted_id, "name": "Brookfield"},
                            {"id": segment_start_id, "name": "Gammaton"},
                            {"id": old_id, "name": "Oldtown"},
                        ],
                        "routes": [
                            {
                                "route_id": "rll_alpha_brook_1",
                                "start_id": origin_id,
                                "destination_id": deleted_id,
                                "includes_toll": False,
                            },
                            {
                                "route_id": "rll_brook_gamma_1",
                                "start_id": deleted_id,
                                "destination_id": segment_start_id,
                                "includes_toll": False,
                            },
                            {
                                "route_id": "rll_gamma_old_1",
                                "start_id": segment_start_id,
                                "destination_id": old_id,
                                "includes_toll": False,
                            },
                        ],
                    },
                },
            },
        )
    )
    assert replace.tool_calls == (
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": replacement_id,
                "route_id_leading_to_new_destination": replacement_route_id,
            },
        },
    )
    replacement_completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-replacement-done",
            tool_name="navigation_replace_final_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "destination_replaced": True,
                    "new_waypoints": [
                        origin_id,
                        deleted_id,
                        segment_start_id,
                        replacement_id,
                    ],
                    "new_routes": [
                        "rll_alpha_brook_1",
                        "rll_brook_gamma_1",
                        replacement_route_id,
                    ],
                },
            },
        )
    )
    assert replacement_completed.text is not None
    assert "toll" not in replacement_completed.text.casefold()

    delete_state = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-delete",
            context_id=context_id,
            user_text=delete_request,
        )
    )
    assert delete_state.tool_calls[0]["tool_name"] == "get_current_navigation_state"
    direct_routes = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-delete-state",
            tool_name="get_current_navigation_state",
            content={
                "status": "SUCCESS",
                "result": {
                    "navigation_active": True,
                    "waypoints_id": [
                        origin_id,
                        deleted_id,
                        segment_start_id,
                        replacement_id,
                    ],
                    "routes_to_final_destination_id": [
                        "rll_alpha_brook_1",
                        "rll_brook_gamma_1",
                        replacement_route_id,
                    ],
                    "details": {
                        "waypoints": [
                            {"id": origin_id, "name": "Alphaville"},
                            {"id": deleted_id, "name": "Brookfield"},
                            {"id": segment_start_id, "name": "Gammaton"},
                            {"id": replacement_id, "name": "Riverton"},
                        ]
                    },
                },
            },
        )
    )
    assert direct_routes.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": origin_id,
                "destination_id": segment_start_id,
            },
        },
    )
    delete = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-direct-routes",
            tool_name="get_routes_from_start_to_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "routes": [
                        {
                            "route_id": "rll_alpha_gamma_fast",
                            "start_id": origin_id,
                            "destination_id": segment_start_id,
                            "name_via": "A20",
                            "distance_km": 310,
                            "duration_hours": 3,
                            "duration_minutes": 0,
                            "road_types": ["highway"],
                            "includes_toll": False,
                            "alias": ["first", "fastest"],
                        },
                        {
                            "route_id": direct_route_id,
                            "start_id": origin_id,
                            "destination_id": segment_start_id,
                            "name_via": "B21",
                            "distance_km": 280,
                            "duration_hours": 3,
                            "duration_minutes": 30,
                            "road_types": ["country road"],
                            "includes_toll": True,
                            "alias": ["second", "shortest"],
                        },
                    ]
                },
            },
        )
    )
    assert delete.tool_calls == (
        {
            "tool_name": "navigation_delete_waypoint",
            "arguments": {
                "waypoint_id_to_delete": deleted_id,
                "route_id_without_waypoint": direct_route_id,
            },
        },
    )
    assert client.intent_calls == 0
    assert client.action_calls == 0
    completed_delete = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-delete-done",
            tool_name="navigation_delete_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_deleted": True,
                    "new_waypoints": [
                        origin_id,
                        segment_start_id,
                        replacement_id,
                    ],
                    "new_routes": [direct_route_id, replacement_route_id],
                },
            },
        )
    )
    assert completed_delete.text is not None
    assert "updated route includes toll roads" in completed_delete.text
    assert (
        parse_adjacent_navigation_destination_reply(
            "Yes, Riverton, but use the fastest route.",
            inherited_route_choice="shortest",
        )
        is None
    )
