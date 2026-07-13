from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import (
    DecisionProposal,
    GoalStatus,
    IntentFrame,
    IntentKind,
)
from track_1_agent_under_test.car_guard.planning import (
    recover_navigation_resume_intent,
)
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
)


START_ID = "loc_alpha_100100"
DESTINATION_ID = "loc_beta_200200"
ROUTE_ID = "rll_alpha_beta_300300"
POLICY = (
    "The start of the overall route set always has to be the current car location. "
    "A new navigation should only be set while navigation is inactive. "
    'CURRENT_LOCATION={"id":"loc_alpha_100100","name":"Alpha"}'
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


CURRENT_NAVIGATION_TOOL = _tool(
    "get_current_navigation_state",
    {"detailed_information": {"type": "boolean"}},
)
SET_NAVIGATION_TOOL = _tool(
    "set_new_navigation",
    {"route_ids": {"type": "array", "items": {"type": "string"}}},
    ["route_ids"],
)
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
RESUME_TOOLS = (CURRENT_NAVIGATION_TOOL, SET_NAVIGATION_TOOL)


def _stopped_navigation() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": False,
            "waypoints_id": [START_ID, DESTINATION_ID],
            "routes_to_final_destination_id": [ROUTE_ID],
            "details": {
                "waypoints": [
                    {"id": START_ID, "name": "Alpha"},
                    {"id": DESTINATION_ID, "name": "Beta"},
                ],
                "routes": [
                    {
                        "route_id": ROUTE_ID,
                        "start_id": START_ID,
                        "destination_id": DESTINATION_ID,
                        "name_via": "K123",
                        "distance_km": 12.5,
                        "duration_hours": 1,
                        "duration_minutes": 5,
                        "road_types": ["country road"],
                        "includes_toll": False,
                        "alias": ["previous"],
                    }
                ],
            },
        },
    }


class DeterministicNavigationClient:
    def __init__(self) -> None:
        self.intent_calls = 0
        self.action_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        del messages, critic
        if response_model is IntentDraft:
            self.intent_calls += 1
            return SimpleNamespace(
                value=IntentDraft.model_validate(
                    {
                        "language": "en",
                        "intent_kind": "action",
                        "call_for_action": True,
                        "goals": [
                            {
                                "semantic_operation": "navigate_to",
                                "desired_outcome": {"location": "Gamma"},
                            }
                        ],
                        "explicit_slots": {"location": "Gamma"},
                    }
                )
            )
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError("navigation workflows must remain deterministic")
        raise AssertionError(f"unexpected response model: {response_model}")


def _runtime() -> tuple[CARGuardOrchestrator, DeterministicNavigationClient]:
    client = DeterministicNavigationClient()
    config = AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=24)
    return (
        CARGuardOrchestrator(config, client_factory=lambda session: client),
        client,
    )


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
    context_id: str, message_id: str, tool_name: str, content: Any
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        tool_results=({"toolName": tool_name, "content": content},),
    )


def _run_to_confirmation(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    result: dict[str, Any] | None = None,
):
    read = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-user",
            "Could you resume my previous navigation?",
            tools=RESUME_TOOLS,
        )
    )
    assert read.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    return runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-state",
            "get_current_navigation_state",
            result or _stopped_navigation(),
        )
    )


@pytest.mark.parametrize(
    "utterance",
    [
        "Restart my last navigation route.",
        "Could you resume my previous navigation?",
        "Okay, how about resuming my previous navigation then?",
        "Could you start my previous navigation route then?",
    ],
)
def test_resume_intent_recovers_only_the_generic_previous_route(
    utterance: str,
) -> None:
    seed = IntentFrame(
        language="en",
        call_for_action=False,
        goals=[],
        intent_kind=IntentKind.CONVERSATION,
    )

    recovered = recover_navigation_resume_intent(utterance, seed, turn_id="turn-user")

    assert recovered.call_for_action
    assert recovered.intent_kind is IntentKind.ACTION
    assert [
        (goal.semantic_operation, goal.desired_outcome) for goal in recovered.goals
    ] == [("resume_previous_navigation", {})]


@pytest.mark.parametrize(
    "utterance",
    [
        "Do not resume my previous navigation.",
        'She said "resume my previous navigation."',
        'The example is "restart my last navigation route."',
        "If the road clears, resume my previous navigation.",
        "Can I resume my previous navigation?",
        "Restart navigation to Gamma.",
        "Resume my previous navigation and call the office.",
    ],
)
def test_resume_intent_rejects_non_commands_and_compound_requests(
    utterance: str,
) -> None:
    seed = IntentFrame(
        language="en",
        call_for_action=False,
        goals=[],
        intent_kind=IntentKind.CONVERSATION,
    )

    recovered = recover_navigation_resume_intent(utterance, seed, turn_id="turn-user")

    assert recovered is seed


def test_resume_reads_details_confirms_exact_snapshot_and_sets_observed_route() -> None:
    runtime, client = _runtime()
    context_id = "resume-success"

    confirmation = _run_to_confirmation(runtime, context_id=context_id)

    assert confirmation.tool_calls == ()
    assert confirmation.text is not None
    for detail in (
        "Alpha",
        "Beta",
        "K123",
        "12.5 kilometers",
        "1 hour and 5 minutes",
        "no toll roads",
        "exact route",
    ):
        assert detail in confirmation.text
    session = runtime.sessions.get(context_id)
    assert session is not None and session.execution_bundle is not None
    goal_id = session.execution_bundle.goal_ids[0]
    assert session.execution_bundle.confirmation_id is not None
    assert client.intent_calls == 0
    assert client.action_calls == 0

    started = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-confirm",
            "Yes, please restart the previous navigation.",
        )
    )

    assert started.tool_calls == (
        {
            "tool_name": "set_new_navigation",
            "arguments": {"route_ids": [ROUTE_ID]},
        },
    )
    completed = runtime.handle_event(
        _result_event(
            context_id,
            f"{context_id}-set",
            "set_new_navigation",
            {"status": "SUCCESS", "result": {"navigation_set": True}},
        )
    )
    assert completed.tool_calls == ()
    assert completed.text is not None
    assert "started navigation" in completed.text.casefold()
    assert session.goal_dag.get(goal_id).status is GoalStatus.DONE


@pytest.mark.parametrize(
    "confirmation_text",
    [
        "Yes, restart the 12.5 kilometer route that takes 1 hour and 5 minutes.",
        "Yes, restart route 123.",
    ],
)
def test_resume_accepts_numbers_repeated_from_the_frozen_route_description(
    confirmation_text: str,
) -> None:
    runtime, _ = _runtime()
    context_id = f"resume-description-{len(confirmation_text)}"

    confirmation = _run_to_confirmation(runtime, context_id=context_id)
    assert confirmation.text is not None

    started = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-confirm",
            confirmation_text,
        )
    )

    assert started.tool_calls == (
        {
            "tool_name": "set_new_navigation",
            "arguments": {"route_ids": [ROUTE_ID]},
        },
    )


def test_resume_confirmation_fails_closed_when_snapshot_becomes_stale() -> None:
    runtime, _ = _runtime()
    context_id = "resume-stale"
    confirmation = _run_to_confirmation(runtime, context_id=context_id)
    assert confirmation.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    session.evidence.invalidate_tool_state()

    blocked = runtime.handle_event(
        _user_event(context_id, f"{context_id}-confirm", "Yes, restart it.")
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert session.budget.attempted_sets == set()


@pytest.mark.parametrize("malformation", ["missing_details", "wrong_chain"])
def test_resume_rejects_malformed_stopped_navigation_snapshot(
    malformation: str,
) -> None:
    runtime, client = _runtime()
    state = _stopped_navigation()
    if malformation == "missing_details":
        del state["result"]["details"]
    else:
        state["result"]["details"]["routes"][0]["start_id"] = DESTINATION_ID

    blocked = _run_to_confirmation(
        runtime,
        context_id=f"resume-malformed-{malformation}",
        result=state,
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert client.intent_calls == 0
    assert client.action_calls == 0


@pytest.mark.parametrize(
    "tools",
    [
        (CURRENT_NAVIGATION_TOOL,),
        (
            CURRENT_NAVIGATION_TOOL,
            _tool(
                "set_new_navigation", {"route_ids": {"type": "string"}}, ["route_ids"]
            ),
        ),
    ],
)
def test_resume_missing_or_invalid_set_schema_stops_before_read(
    tools: tuple[dict[str, Any], ...],
) -> None:
    runtime, client = _runtime()
    context_id = f"resume-schema-{len(tools)}"

    blocked = runtime.handle_event(
        _user_event(
            context_id,
            f"{context_id}-user",
            "Restart my last navigation route.",
            tools=tools,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.successful_read_results == []
    assert session.budget.attempted_sets == set()
    assert client.intent_calls == 0


def test_ordinary_navigation_create_uses_deterministic_grounded_workflow() -> None:
    runtime, client = _runtime()
    tools = (
        CURRENT_NAVIGATION_TOOL,
        LOCATION_TOOL,
        ROUTES_TOOL,
        SET_NAVIGATION_TOOL,
    )

    outbound = runtime.handle_event(
        _user_event(
            "ordinary-create",
            "ordinary-create-user",
            "Navigate to Gamma.",
            tools=tools,
        )
    )

    assert outbound.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    assert client.intent_calls == 0
    assert client.action_calls == 0
