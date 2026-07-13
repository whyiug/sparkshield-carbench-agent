from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import DecisionProposal, GoalStatus
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
)


REQUEST = "change my navigation destination to Munich, please"
POLICY = "Follow the current safety policy."
START_ID = "loc_dor_399984"
OLD_DESTINATION_ID = "loc_dus_892560"
MUNICH_ID = "loc_mun_9995"
FASTEST_ROUTE_ID = "rll_dor_mun_199807"


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
FULL_TOOLS = (
    CURRENT_NAVIGATION_TOOL,
    LOCATION_TOOL,
    ROUTES_TOOL,
    REPLACE_TOOL,
)
HALL64_TOOLS = (CURRENT_NAVIGATION_TOOL, LOCATION_TOOL, ROUTES_TOOL)


class DestinationReplaceClient:
    def __init__(self, intent: dict[str, Any] | None = None) -> None:
        self.intent = IntentDraft.model_validate(
            intent
            or {
                "language": "en",
                "intent_kind": "information",
                "call_for_action": False,
                "goals": [],
                "explicit_slots": {},
            }
        )
        self.intent_calls = 0
        self.action_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        del messages, critic
        if response_model is IntentDraft:
            self.intent_calls += 1
            return SimpleNamespace(value=self.intent)
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError(
                "destination replacement actions must be deterministic"
            )
        raise AssertionError(f"unexpected response model: {response_model}")


def runtime_for(
    client: DestinationReplaceClient,
) -> CARGuardOrchestrator:
    config = AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=24)
    return CARGuardOrchestrator(config, client_factory=lambda session: client)


def result_event(
    *, context_id: str, message_id: str, tool_name: str, content: Any
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        tool_results=({"toolName": tool_name, "content": content},),
    )


def navigation_state(
    *, existing_route_includes_toll: bool | None = None
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [START_ID, OLD_DESTINATION_ID],
            "routes_to_final_destination_id": ["rll_dor_dus_836702"],
            "details": {
                "waypoints": [
                    {"id": START_ID, "name": "Dortmund"},
                    {"id": OLD_DESTINATION_ID, "name": "Dusseldorf"},
                ]
            },
        },
    }
    if existing_route_includes_toll is not None:
        state["result"]["details"]["routes"] = [
            {
                "route_id": "rll_dor_dus_836702",
                "start_id": START_ID,
                "destination_id": OLD_DESTINATION_ID,
                "includes_toll": existing_route_includes_toll,
            }
        ]
    return state


def multi_segment_navigation_state(
    *, retained_route_includes_toll: bool
) -> dict[str, Any]:
    origin_id = "loc_essen_711"
    state = navigation_state()
    state["result"]["waypoints_id"] = [origin_id, START_ID, OLD_DESTINATION_ID]
    state["result"]["routes_to_final_destination_id"] = [
        "rll_ess_dor_711",
        "rll_dor_dus_836702",
    ]
    state["result"]["details"] = {
        "waypoints": [
            {"id": origin_id, "name": "Essen"},
            {"id": START_ID, "name": "Dortmund"},
            {"id": OLD_DESTINATION_ID, "name": "Dusseldorf"},
        ],
        "routes": [
            {
                "route_id": "rll_ess_dor_711",
                "start_id": origin_id,
                "destination_id": START_ID,
                "includes_toll": retained_route_includes_toll,
            },
            {
                "route_id": "rll_dor_dus_836702",
                "start_id": START_ID,
                "destination_id": OLD_DESTINATION_ID,
                "includes_toll": False,
            },
        ],
    }
    return state


def route_result(
    *,
    first_hours: Any = 7,
    first_minutes: Any = 0,
    distinct_highlights: bool = False,
) -> dict[str, Any]:
    first_aliases = ["first", "fastest"]
    second_aliases = ["second"]
    first_distance = 560.0
    second_distance = 575.0
    second_hours = 7
    second_minutes = 15
    if distinct_highlights:
        second_aliases.append("shortest")
        first_distance = 570.0
        second_distance = 560.0
    else:
        first_aliases.append("shortest")
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": FASTEST_ROUTE_ID,
                    "start_id": START_ID,
                    "destination_id": MUNICH_ID,
                    "name_via": "A9",
                    "distance_km": first_distance,
                    "duration_hours": first_hours,
                    "duration_minutes": first_minutes,
                    "road_types": ["highway"],
                    "includes_toll": False,
                    "alias": first_aliases,
                },
                {
                    "route_id": "rll_dor_mun_475855",
                    "start_id": START_ID,
                    "destination_id": MUNICH_ID,
                    "name_via": "A8",
                    "distance_km": second_distance,
                    "duration_hours": second_hours,
                    "duration_minutes": second_minutes,
                    "road_types": ["highway", "urban"],
                    "includes_toll": False,
                    "alias": second_aliases,
                },
                {
                    "route_id": "rll_dor_mun_750110",
                    "start_id": START_ID,
                    "destination_id": MUNICH_ID,
                    "name_via": "B2",
                    "distance_km": 590.0,
                    "duration_hours": 7,
                    "duration_minutes": 30,
                    "road_types": ["country road"],
                    "includes_toll": False,
                    "alias": ["third"],
                },
            ]
        },
    }


def formal_route_result() -> dict[str, Any]:
    routes = route_result()
    vias = ("L505, K873, B560", "L178, K912", "B156, A90, A59")
    for option, via in zip(routes["result"]["routes"], vias, strict=True):
        option["name_via"] = via
    return routes


def begin_replacement(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    tools: tuple[dict[str, Any], ...] = FULL_TOOLS,
) -> str:
    current = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=POLICY,
            user_text=REQUEST,
            live_tools=tools,
        )
    )
    assert current.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert len(session.intent.goals) == 1
    goal = session.intent.goals[0]
    assert goal.semantic_operation == "navigation_replace_final_destination"
    assert goal.desired_outcome == {"new_destination_name": "Munich"}
    return goal.goal_id


def run_to_route_prompt(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    routes: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    goal_id = begin_replacement(runtime, context_id=context_id)
    location = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=state or navigation_state(),
        )
    )
    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Munich"},
        },
    )
    route = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": MUNICH_ID}},
        )
    )
    assert route.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {"start_id": START_ID, "destination_id": MUNICH_ID},
        },
    )
    prompt = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            tool_name="get_routes_from_start_to_destination",
            content=routes or route_result(),
        )
    )
    return goal_id, prompt


def assert_no_replace(outbound: Any) -> None:
    assert all(
        call["tool_name"] != "navigation_replace_final_destination"
        for call in outbound.tool_calls
    )


def test_base66_exact_chain_selects_fastest_route_and_preserves_provenance() -> None:
    client = DestinationReplaceClient()
    runtime = runtime_for(client)
    context_id = "base66-exact-fastest"
    goal_id, prompt = run_to_route_prompt(runtime, context_id=context_id)

    assert prompt.tool_calls == ()
    assert prompt.text is not None
    assert "fastest and shortest" in prompt.text
    selected = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text="Use the fastest route.",
        )
    )

    assert selected.tool_calls == (
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": MUNICH_ID,
                "route_id_leading_to_new_destination": FASTEST_ROUTE_ID,
            },
        },
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert session.intent.goals[0].goal_id == goal_id
    assert session.grounded_value_sources_by_goal[goal_id]["route_choice_alias"] == (
        f"{context_id}-choice"
    )
    assert client.action_calls == 0

    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-done",
            tool_name="navigation_replace_final_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "destination_replaced": True,
                    "new_waypoints": [START_ID, MUNICH_ID],
                    "new_routes": [FASTEST_ROUTE_ID],
                },
            },
        )
    )
    assert completed.tool_calls == ()
    assert completed.text is not None and "replaced the destination" in completed.text


@pytest.mark.parametrize(
    ("existing_toll", "replacement_toll", "expects_notice"),
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_destination_replacement_discloses_only_evidence_backed_route_tolls(
    existing_toll: bool,
    replacement_toll: bool,
    expects_notice: bool,
) -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = f"replacement-toll-{existing_toll}-{replacement_toll}"
    routes = route_result()
    routes["result"]["routes"][0]["includes_toll"] = replacement_toll
    state = multi_segment_navigation_state(
        retained_route_includes_toll=existing_toll
    )
    _, prompt = run_to_route_prompt(
        runtime,
        context_id=context_id,
        routes=routes,
        state=state,
    )
    assert prompt.text is not None
    replace = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text="Use the fastest route.",
        )
    )
    assert replace.tool_calls[0]["tool_name"] == "navigation_replace_final_destination"
    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-done",
            tool_name="navigation_replace_final_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "destination_replaced": True,
                    "new_waypoints": [
                        *state["result"]["waypoints_id"][:-1],
                        MUNICH_ID,
                    ],
                    "new_routes": [
                        *state["result"]["routes_to_final_destination_id"][:-1],
                        FASTEST_ROUTE_ID,
                    ],
                },
            },
        )
    )
    assert completed.text is not None
    assert ("updated route includes toll roads" in completed.text) is expects_notice


def test_destination_replacement_does_not_claim_tolls_from_malformed_details() -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = "replacement-toll-malformed-details"
    state = multi_segment_navigation_state(retained_route_includes_toll=True)
    state["result"]["details"]["routes"][0]["includes_toll"] = "true"
    _, prompt = run_to_route_prompt(runtime, context_id=context_id, state=state)
    assert prompt.text is not None
    replace = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text="Use the fastest route.",
        )
    )
    assert replace.tool_calls[0]["tool_name"] == "navigation_replace_final_destination"
    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-done",
            tool_name="navigation_replace_final_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "destination_replaced": True,
                    "new_waypoints": [
                        *state["result"]["waypoints_id"][:-1],
                        MUNICH_ID,
                    ],
                    "new_routes": [
                        *state["result"]["routes_to_final_destination_id"][:-1],
                        FASTEST_ROUTE_ID,
                    ],
                },
            },
        )
    )
    assert completed.text is not None
    assert "toll" not in completed.text.casefold()


def test_destination_replacement_rejects_a_result_that_drops_retained_segments() -> (
    None
):
    runtime = runtime_for(DestinationReplaceClient())
    context_id = "replacement-dropped-retained-segment"
    state = multi_segment_navigation_state(retained_route_includes_toll=True)
    _, prompt = run_to_route_prompt(runtime, context_id=context_id, state=state)
    assert prompt.text is not None
    replace = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text="Use the fastest route.",
        )
    )
    assert replace.tool_calls[0]["tool_name"] == "navigation_replace_final_destination"
    rejected = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-done",
            tool_name="navigation_replace_final_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "destination_replaced": True,
                    "new_waypoints": [START_ID, MUNICH_ID],
                    "new_routes": [FASTEST_ROUTE_ID],
                },
            },
        )
    )
    assert rejected.text is not None
    assert "couldn't verify" in rejected.text
    assert "toll" not in rejected.text.casefold()


def test_destination_replacement_excludes_the_replaced_old_segment_toll() -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = "replacement-excludes-old-segment-toll"
    _, prompt = run_to_route_prompt(
        runtime,
        context_id=context_id,
        state=navigation_state(existing_route_includes_toll=True),
    )
    assert prompt.text is not None
    replace = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text="Use the fastest route.",
        )
    )
    assert replace.tool_calls[0]["tool_name"] == "navigation_replace_final_destination"
    completed = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-done",
            tool_name="navigation_replace_final_destination",
            content={
                "status": "SUCCESS",
                "result": {
                    "destination_replaced": True,
                    "new_waypoints": [START_ID, MUNICH_ID],
                    "new_routes": [FASTEST_ROUTE_ID],
                },
            },
        )
    )
    assert completed.text is not None
    assert "toll" not in completed.text.casefold()


def test_base66_fresh_command_replaces_a_completed_current_navigation_goal() -> None:
    client = DestinationReplaceClient(
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
    runtime = runtime_for(client)
    context_id = "base66-fresh-after-current-navigation"
    current = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-current-user",
            context_id=context_id,
            system_policy=POLICY,
            user_text="What is my current navigation route?",
            live_tools=FULL_TOOLS,
        )
    )
    assert current.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and len(session.goal_dag.goals) == 1
    old_goal_id = session.goal_dag.goals[0].goal_id

    summary = runtime.handle_event(
        result_event(
            context_id=context_id,
            message_id=f"{context_id}-current-state",
            tool_name="get_current_navigation_state",
            content=navigation_state(),
        )
    )
    assert summary.tool_calls == ()
    assert summary.text is not None
    assert session.goal_dag.get(old_goal_id).status is GoalStatus.DONE
    assert client.intent_calls == 1

    replacement = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-replacement-user",
            context_id=context_id,
            user_text="Please change my navigation destination to Munich.",
        )
    )

    assert replacement.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    assert session.intent is not None and len(session.intent.goals) == 1
    fresh_goal = session.intent.goals[0]
    assert fresh_goal.goal_id != old_goal_id
    assert fresh_goal.semantic_operation == "navigation_replace_final_destination"
    assert fresh_goal.desired_outcome == {"new_destination_name": "Munich"}
    assert fresh_goal.goal_id == session.goal_dag.goals[0].goal_id
    assert session.goal_dag.get(fresh_goal.goal_id).status is GoalStatus.PENDING
    assert client.intent_calls == 1


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "Hey there! I'm actually heading to Munich now instead of Paris. "
            "Can you change my navigation destination for me?"
        ),
        (
            "Hey there! I've changed my mind about going to Paris. Can you "
            "switch my navigation destination to Munich instead?"
        ),
        (
            "Hey there! I've changed my mind about Paris. Can you switch my "
            "navigation to Munich instead?"
        ),
        (
            "Yep, that's the current one. So, can you change the destination "
            "to Munich for me?"
        ),
        (
            "Okay, I understand that's the current route. But I want to change "
            "the destination to Munich, please."
        ),
        "Okay, I need to change my navigation destination. Please set it to Munich.",
        "I need to change my navigation. Please change the destination to Munich.",
        "Please change my navigation destination to Munich.",
        (
            "I want to change my navigation. My new destination is Munich. "
            "Can you please set that up?"
        ),
        (
            "I want to change my navigation destination. Please set the new "
            "destination to Munich."
        ),
        "Please change my current navigation destination to Munich.",
    ],
    ids=[
        "heading-new-instead-of-old",
        "changed-mind-switch",
        "changed-mind-structured-switch",
        "current-one-ack",
        "current-route-ack",
        "set-it-anaphora",
        "navigation-then-destination",
        "direct-navigation-destination",
        "declared-new-destination",
        "set-new-destination",
        "direct-current-navigation-destination",
    ],
)
def test_base66_formal_fresh_commands_bypass_wrong_navigation_read_model(
    utterance: str,
) -> None:
    client = DestinationReplaceClient(
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
    runtime = runtime_for(client)
    context_id = f"base66-formal-fresh-{abs(hash(utterance))}"

    first = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=POLICY,
            user_text=utterance,
            live_tools=FULL_TOOLS,
        )
    )

    assert first.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert len(session.intent.goals) == 1
    fresh_goal = session.intent.goals[0]
    assert fresh_goal.goal_id.startswith("goal-named-destination-replacement-")
    assert fresh_goal.semantic_operation == "navigation_replace_final_destination"
    assert fresh_goal.desired_outcome == {"new_destination_name": "Munich"}
    assert session.goal_dag.get(fresh_goal.goal_id).status is GoalStatus.PENDING
    assert client.intent_calls == 0


def test_base66_fresh_command_supersedes_stale_route_clarification() -> None:
    client = DestinationReplaceClient()
    runtime = runtime_for(client)
    context_id = "base66-fresh-over-stale-route-choice"
    old_goal_id, prompt = run_to_route_prompt(runtime, context_id=context_id)
    assert prompt.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None and len(session.pending_clarifications) == 1

    replacement = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-replacement-user",
            context_id=context_id,
            user_text="Please change my navigation destination to Berlin.",
        )
    )

    assert replacement.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    assert session.pending_clarifications == {}
    assert session.intent is not None and len(session.intent.goals) == 1
    fresh_goal = session.intent.goals[0]
    assert fresh_goal.goal_id != old_goal_id
    assert fresh_goal.desired_outcome == {"new_destination_name": "Berlin"}
    assert session.goal_dag.get(fresh_goal.goal_id).status is GoalStatus.PENDING


def test_base66_fresh_command_does_not_supersede_a_pending_tool_result() -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = "base66-fresh-while-state-pending"
    goal_id = begin_replacement(runtime, context_id=context_id)
    session = runtime.sessions.get(context_id)
    assert session is not None and len(session.pending_calls) == 1

    waiting = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-replacement-user",
            context_id=context_id,
            user_text="Please change my navigation destination to Berlin.",
        )
    )

    assert waiting.tool_calls == ()
    assert waiting.text is not None and "still waiting" in waiting.text
    assert session.intent is not None and session.intent.goals[0].goal_id == goal_id
    assert session.intent.goals[0].desired_outcome == {"new_destination_name": "Munich"}
    assert len(session.pending_calls) == 1


def test_base66_ordinal_control_reaches_exact_replacement_set() -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = "base66-first-route-control"
    _, prompt = run_to_route_prompt(runtime, context_id=context_id)
    assert prompt.text is not None and "first route" in prompt.text

    selected = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text="Use the first route.",
        )
    )

    assert selected.tool_calls == (
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": MUNICH_ID,
                "route_id_leading_to_new_destination": FASTEST_ROUTE_ID,
            },
        },
    )


def test_base66_normalizes_sixty_route_minutes_before_presentation() -> None:
    runtime = runtime_for(DestinationReplaceClient())
    _, prompt = run_to_route_prompt(
        runtime,
        context_id="base66-sixty-minutes",
        routes=route_result(first_hours=6, first_minutes=60),
    )

    assert prompt.tool_calls == ()
    assert prompt.text is not None
    assert "about 7 hours" in prompt.text
    assert "60 minutes" not in prompt.text


@pytest.mark.parametrize(
    "choice",
    [
        "Use the fastest one.",
        "The first route you mentioned.",
        "That one.",
    ],
)
def test_base66_contextual_fastest_route_choices_select_unique_highlight(
    choice: str,
) -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = f"base66-choice-{len(choice)}"
    _, prompt = run_to_route_prompt(runtime, context_id=context_id)
    assert prompt.text is not None and "fastest and shortest" in prompt.text

    selected = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text=choice,
        )
    )

    assert selected.tool_calls == (
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": MUNICH_ID,
                "route_id_leading_to_new_destination": FASTEST_ROUTE_ID,
            },
        },
    )


@pytest.mark.parametrize(
    "choice",
    [
        "Okay, so I'll take the fastest route then.",
        "I'll take the first route, the fastest one.",
        "Yeah, the first route, please.",
        "Yes, the first route, the fastest one.",
    ],
)
def test_base66_official_contextual_route_choice_variants(choice: str) -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = f"base66-official-choice-{abs(hash(choice))}"
    _, prompt = run_to_route_prompt(runtime, context_id=context_id)
    assert prompt.text is not None

    selected = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text=choice,
        )
    )

    assert selected.tool_calls == (
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": MUNICH_ID,
                "route_id_leading_to_new_destination": FASTEST_ROUTE_ID,
            },
        },
    )


@pytest.mark.parametrize(
    ("choice", "accepted"),
    [
        ("Okay, I'll take the fastest one then.", True),
        ("Yeah, the first one, please.", True),
        ("CONTINUE", False),
        (
            "Yes, the first route, please. The one via L505, K873, B560.",
            True,
        ),
        (
            "Yes, the first route. Please confirm that you've set the navigation "
            "to that one.",
            True,
        ),
        (
            "I've said the first route a few times now. Can you please set the "
            "navigation to the first route, the one via L505, K873, B560?",
            True,
        ),
    ],
    ids=[
        "take-fastest-one",
        "ack-first-one",
        "continue-control-token",
        "first-with-redundant-via",
        "first-with-bounded-confirmation",
        "history-with-explicit-set-navigation",
    ],
)
def test_base66_fourth_formal_route_choice_family(choice: str, accepted: bool) -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = f"base66-fourth-choice-{abs(hash(choice))}"
    _, prompt = run_to_route_prompt(
        runtime,
        context_id=context_id,
        routes=formal_route_result(),
    )
    assert prompt.text is not None

    selected = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text=choice,
        )
    )

    if accepted:
        assert selected.tool_calls == (
            {
                "tool_name": "navigation_replace_final_destination",
                "arguments": {
                    "new_destination_id": MUNICH_ID,
                    "route_id_leading_to_new_destination": FASTEST_ROUTE_ID,
                },
            },
        )
    else:
        assert_no_replace(selected)
        session = runtime.sessions.get(context_id)
        assert session is not None and len(session.pending_clarifications) == 1


@pytest.mark.parametrize(
    "choice",
    [
        "Should I take the first route via L505, K873, B560?",
        "Do you recommend I take the first route via L505, K873, B560?",
        "Could I take the first route via L505, K873, B560?",
        "Don't use the first route via L505, K873, B560.",
        "If needed, use the first route via L505, K873, B560.",
        "I'll take the second route, the fastest one.",
        "Yes, the first route. The one via L178, K912.",
        "Yes, the first route. Open the sunroof.",
        "Yes, the first route. The one via L505, K873, B560 and open the sunroof.",
        "!!! Use the first route.",
        "@@ Use the first route.",
        "Use the first route $$$.",
        "Use 'the first route' please.",
        "Use the first route. !!!",
        ("I've said the first route a few times now? Can you use the first route?"),
    ],
)
def test_base66_evidence_convergent_route_choice_rejects_unsafe_or_conflicting_text(
    choice: str,
) -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = f"base66-convergent-reject-{abs(hash(choice))}"
    _, prompt = run_to_route_prompt(
        runtime,
        context_id=context_id,
        routes=formal_route_result(),
    )
    assert prompt.text is not None

    rejected = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text=choice,
        )
    )

    assert_no_replace(rejected)
    assert rejected.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and len(session.pending_clarifications) == 1


def test_hall64_missing_replace_tool_blocks_before_all_reads() -> None:
    client = DestinationReplaceClient()
    runtime = runtime_for(client)
    context_id = "hall64-no-replace"

    blocked = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=POLICY,
            user_text=REQUEST,
            live_tools=HALL64_TOOLS,
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
    ("field", "value"),
    [
        ("duration_minutes", "60"),
        ("duration_minutes", True),
        ("duration_minutes", -1),
        ("duration_minutes", 1.5),
        ("duration_hours", "7"),
        ("duration_hours", 241),
    ],
)
def test_base66_malformed_route_durations_fail_closed(field: str, value: Any) -> None:
    routes = route_result()
    routes["result"]["routes"][0][field] = value
    runtime = runtime_for(DestinationReplaceClient())
    _, blocked = run_to_route_prompt(
        runtime,
        context_id=f"base66-bad-duration-{field}-{type(value).__name__}-{value}",
        routes=routes,
    )

    assert_no_replace(blocked)
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "couldn't verify" in blocked.text


@pytest.mark.parametrize(
    "case",
    ["duplicate_fastest", "duplicate_ordinal", "missing_ordinal", "unknown_alias"],
)
def test_base66_malformed_route_aliases_fail_closed(case: str) -> None:
    routes = route_result()
    options = routes["result"]["routes"]
    if case == "duplicate_fastest":
        options[1]["alias"].append("fastest")
    elif case == "duplicate_ordinal":
        options[2]["alias"] = ["second"]
    elif case == "missing_ordinal":
        options[0]["alias"] = ["fastest", "shortest"]
    else:
        options[2]["alias"].append("quickest")
    runtime = runtime_for(DestinationReplaceClient())
    _, blocked = run_to_route_prompt(
        runtime,
        context_id=f"base66-bad-alias-{case}",
        routes=routes,
    )

    assert_no_replace(blocked)
    assert blocked.tool_calls == ()
    assert blocked.text is not None and "couldn't verify" in blocked.text


@pytest.mark.parametrize("choice", ["That one.", "The first route you mentioned."])
def test_base66_contextual_reference_is_rejected_after_alternative_details(
    choice: str,
) -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = "base66-stale-that-after-details"
    _, prompt = run_to_route_prompt(runtime, context_id=context_id)
    assert prompt.text is not None and "fastest and shortest" in prompt.text
    details = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-details",
            context_id=context_id,
            user_text="Tell me more about the alternative routes.",
        )
    )
    assert details.tool_calls == ()
    assert details.text is not None and "second route" in details.text

    rejected = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-contextual-choice",
            context_id=context_id,
            user_text=choice,
        )
    )

    assert_no_replace(rejected)
    assert rejected.tool_calls == ()
    assert rejected.text is not None


def test_base66_that_one_is_rejected_when_two_routes_are_highlighted() -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = "base66-ambiguous-that"
    _, prompt = run_to_route_prompt(
        runtime,
        context_id=context_id,
        routes=route_result(
            first_hours=6,
            first_minutes=50,
            distinct_highlights=True,
        ),
    )
    assert prompt.text is not None
    assert "fastest" in prompt.text and "shortest" in prompt.text

    rejected = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-that",
            context_id=context_id,
            user_text="That one.",
        )
    )

    assert_no_replace(rejected)
    assert rejected.tool_calls == ()
    assert rejected.text is not None


@pytest.mark.parametrize(
    "choice",
    [
        '"Use the first route."',
        "“Use the fastest route.”",
        "'That one.'",
        "‘That one.’",
        'She said, "Use the fastest route."',
        "Use the first route. That one.",
        "The phrase use the fastest route is an example.",
        '"Okay, so I\'ll take the fastest route then."',
        'She said, "Yeah, the first route, please."',
        "Okay, so I'll take the fastest route then. Show the alternatives.",
        "Could I take the first route, the fastest one?",
    ],
)
def test_base66_quoted_reported_or_meta_route_choice_is_rejected(
    choice: str,
) -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = f"base66-unsafe-choice-{abs(hash(choice))}"
    _, prompt = run_to_route_prompt(runtime, context_id=context_id)
    assert prompt.text is not None

    rejected = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text=choice,
        )
    )

    assert_no_replace(rejected)
    assert rejected.tool_calls == ()
    assert rejected.text is not None


def test_base66_joint_first_fastest_choice_rejects_conflicting_route_labels() -> None:
    routes = route_result()
    options = routes["result"]["routes"]
    options[0]["alias"] = ["first", "shortest"]
    options[0]["duration_minutes"] = 20
    options[1]["alias"] = ["second", "fastest"]
    runtime = runtime_for(DestinationReplaceClient())
    context_id = "base66-conflicting-first-fastest"
    _, prompt = run_to_route_prompt(
        runtime,
        context_id=context_id,
        routes=routes,
    )
    assert prompt.text is not None

    rejected = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text="I'll take the first route, the fastest one.",
        )
    )

    assert_no_replace(rejected)
    assert rejected.tool_calls == ()
    assert rejected.text is not None


def test_base66_fresh_deictic_cannot_authorize_model_invented_replacement() -> None:
    malicious_intent = {
        "language": "en",
        "intent_kind": "action",
        "call_for_action": True,
        "goals": [
            {
                "semantic_operation": "navigation_replace_final_destination",
                "desired_outcome": {
                    "new_destination_id": MUNICH_ID,
                    "route_id": FASTEST_ROUTE_ID,
                },
            }
        ],
        "explicit_slots": {
            "new_destination_id": MUNICH_ID,
            "route_id": FASTEST_ROUTE_ID,
        },
    }
    client = DestinationReplaceClient(malicious_intent)
    runtime = runtime_for(client)
    context_id = "base66-fresh-that"

    rejected = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-user",
            context_id=context_id,
            system_policy=POLICY,
            user_text="That one.",
            live_tools=FULL_TOOLS,
        )
    )

    assert rejected.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.authorized_action_goal_ids == set()
    assert session.intent is not None and session.intent.goals == []
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base66_stale_route_evidence_is_not_used_after_state_change() -> None:
    runtime = runtime_for(DestinationReplaceClient())
    context_id = "base66-route-toctou"
    _, prompt = run_to_route_prompt(runtime, context_id=context_id)
    assert prompt.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    session.evidence.invalidate_tool_state()

    refreshed = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-choice",
            context_id=context_id,
            user_text="Use the first route.",
        )
    )

    assert refreshed.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    assert session.budget.attempted_sets == set()
