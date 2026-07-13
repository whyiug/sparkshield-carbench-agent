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


BONN_ID = "loc_bon_490528"
FRANKFURT_ID = "loc_fra_178468"
FASTEST_ROUTE_ID = "rll_bon_fra_349364"
REQUEST = "I need to set up navigation to Frankfurt."
POLICY = (
    "The start of the overall route set always has to be the current car location. "
    "If the navigation system is already active, edit the active route. If multiple "
    "navigation edits are needed, perform them in sequence. A new navigation should "
    "only be set while navigation is inactive. If there are multiple alternative "
    "routes, describe the fastest route, the shortest route, and the number of "
    "further route alternatives. "
    'CURRENT_LOCATION={"id":"loc_bon_490528","name":"Bonn"}'
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
SET_NEW_NAVIGATION_TOOL = _tool(
    "set_new_navigation",
    {"route_ids": {"type": "array", "items": {"type": "string"}}},
    ["route_ids"],
)
SET_NEW_NAVIGATION_WITHOUT_ROUTE_IDS = _tool("set_new_navigation")


@pytest.fixture
def current_navigation_tool() -> dict[str, Any]:
    return deepcopy(CURRENT_NAVIGATION_TOOL)


@pytest.fixture
def set_new_navigation_tool() -> dict[str, Any]:
    return deepcopy(SET_NEW_NAVIGATION_TOOL)


@pytest.fixture
def full_tools(
    current_navigation_tool: dict[str, Any],
    set_new_navigation_tool: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    return (
        current_navigation_tool,
        deepcopy(LOCATION_TOOL),
        deepcopy(ROUTES_TOOL),
        set_new_navigation_tool,
    )


@pytest.fixture
def hall70_tools(
    current_navigation_tool: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    return (
        current_navigation_tool,
        deepcopy(LOCATION_TOOL),
        deepcopy(ROUTES_TOOL),
        deepcopy(SET_NEW_NAVIGATION_WITHOUT_ROUTE_IDS),
    )


@pytest.fixture
def inactive_navigation() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": False,
            "waypoints_id": [],
            "routes_to_final_destination_id": [],
        },
    }


@pytest.fixture
def active_navigation() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [BONN_ID, "loc_col_464166"],
            "routes_to_final_destination_id": ["rll_bon_col_344373"],
            "details": {
                "waypoints": [
                    {"id": BONN_ID, "name": "Bonn"},
                    {"id": "loc_col_464166", "name": "Cologne"},
                ]
            },
        },
    }


@pytest.fixture
def base72_routes() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": FASTEST_ROUTE_ID,
                    "start_id": BONN_ID,
                    "destination_id": FRANKFURT_ID,
                    "name_via": "B303, A82",
                    "distance_km": 153.91,
                    "duration_hours": 1,
                    "duration_minutes": 54,
                    "road_types": ["country road", "highway"],
                    "includes_toll": False,
                    "alias": ["fastest", "first", "shortest"],
                },
                {
                    "route_id": "rll_bon_fra_239238",
                    "start_id": BONN_ID,
                    "destination_id": FRANKFURT_ID,
                    "name_via": "L38, K919",
                    "distance_km": 157.14,
                    "duration_hours": 1,
                    "duration_minutes": 58,
                    "road_types": ["urban"],
                    "includes_toll": False,
                    "alias": ["second"],
                },
                {
                    "route_id": "rll_bon_fra_965555",
                    "start_id": BONN_ID,
                    "destination_id": FRANKFURT_ID,
                    "name_via": "L842, K751, K700",
                    "distance_km": 160.84,
                    "duration_hours": 2,
                    "duration_minutes": 3,
                    "road_types": ["urban"],
                    "includes_toll": False,
                    "alias": ["third"],
                },
            ]
        },
    }


class NavigationCreateClient:
    def __init__(self) -> None:
        self.intent = IntentDraft.model_validate(
            {
                "language": "en",
                "intent_kind": "action",
                "call_for_action": True,
                "goals": [
                    {
                        "semantic_operation": "navigate_to",
                        "desired_outcome": {"location": "Frankfurt"},
                    }
                ],
                "explicit_slots": {"location": "Frankfurt"},
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
            raise AssertionError("navigation creation must use deterministic routing")
        raise AssertionError(f"unexpected response model: {response_model}")


@pytest.fixture
def client() -> NavigationCreateClient:
    return NavigationCreateClient()


@pytest.fixture
def runtime(client: NavigationCreateClient) -> CARGuardOrchestrator:
    config = AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=24)
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


def _route_evidence(route_result: dict[str, Any]) -> Evidence:
    return Evidence(
        proposition="route_options",
        value=route_result["result"]["routes"],
        status=EvidenceStatus.KNOWN,
        source_kind=EvidenceSourceKind.SYSTEM,
        source_turn_id="base72-route-fixture",
        confidence=1.0,
    )


def _malform_routes(route_result: dict[str, Any], case: str) -> None:
    routes = route_result["result"]["routes"]
    if case == "wrong_endpoint":
        routes[0]["start_id"] = "loc_col_464166"
    elif case == "duplicate_fastest":
        routes[1]["alias"].append("fastest")
    elif case == "invalid_toll_type":
        routes[0]["includes_toll"] = "false"
    else:
        raise AssertionError(f"unknown malformed route case: {case}")


def _goal_id(runtime: CARGuardOrchestrator, context_id: str) -> str:
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    assert len(session.intent.goals) == 1
    return session.intent.goals[0].goal_id


def _begin_to_route_request(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    tools: tuple[dict[str, Any], ...],
    navigation_result: dict[str, Any],
) -> tuple[str, Any]:
    outbound = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=REQUEST,
            tools=tools,
        )
    )
    goal_id = _goal_id(runtime, context_id)
    observed_reads: list[str] = []

    for step in range(6):
        assert outbound.tool_calls, (
            "navigation creation stopped before obtaining the current state, exact "
            f"destination, and route options: {outbound.text!r}"
        )
        results: list[tuple[str, Any]] = []
        for call in outbound.tool_calls:
            name = call["tool_name"]
            arguments = call["arguments"]
            assert name != "set_new_navigation", "route choice must precede the SET"
            observed_reads.append(name)
            if name == "get_current_navigation_state":
                assert arguments == {"detailed_information": True}
                results.append((name, navigation_result))
            elif name == "get_location_id_by_location_name":
                assert arguments == {"location": "Frankfurt"}
                results.append(
                    (
                        name,
                        {"status": "SUCCESS", "result": {"id": FRANKFURT_ID}},
                    )
                )
            elif name == "get_routes_from_start_to_destination":
                assert arguments == {
                    "start_id": BONN_ID,
                    "destination_id": FRANKFURT_ID,
                }
                assert observed_reads.count("get_current_navigation_state") == 1
                assert observed_reads.count("get_location_id_by_location_name") == 1
                return goal_id, outbound
            else:
                pytest.fail(f"unexpected navigation-create read: {name}")
        outbound = runtime.handle_event(
            _result_event(
                context_id=context_id,
                message_id=f"{context_id}-read-{step}",
                results=results,
            )
        )
    pytest.fail("navigation creation did not reach the route lookup")


def _run_to_route_prompt(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    tools: tuple[dict[str, Any], ...],
    navigation_result: dict[str, Any],
    route_result: dict[str, Any],
) -> tuple[str, Any]:
    goal_id, route_call = _begin_to_route_request(
        runtime,
        context_id=context_id,
        tools=tools,
        navigation_result=navigation_result,
    )
    assert len(route_call.tool_calls) == 1
    prompt = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-routes",
            results=[("get_routes_from_start_to_destination", route_result)],
        )
    )
    return goal_id, prompt


def test_base72_route_fixture_is_accepted_by_strict_parser(
    base72_routes: dict[str, Any],
) -> None:
    options = CARGuardOrchestrator._destination_route_options_from_evidence(
        _route_evidence(base72_routes),
        start_id=BONN_ID,
        destination_id=FRANKFURT_ID,
    )

    assert options is not None
    assert [option.route_id for option in options] == [
        FASTEST_ROUTE_ID,
        "rll_bon_fra_239238",
        "rll_bon_fra_965555",
    ]
    assert options[0].ordinal == "first"
    assert options[0].aliases == ("fastest", "first", "shortest")
    assert options[0].duration_minutes == 114
    assert not options[0].includes_toll


@pytest.mark.parametrize(
    "case",
    ["wrong_endpoint", "duplicate_fastest", "invalid_toll_type"],
)
def test_base72_strict_parser_rejects_malformed_routes(
    base72_routes: dict[str, Any], case: str
) -> None:
    _malform_routes(base72_routes, case)

    options = CARGuardOrchestrator._destination_route_options_from_evidence(
        _route_evidence(base72_routes),
        start_id=BONN_ID,
        destination_id=FRANKFURT_ID,
    )

    assert options is None


@pytest.mark.parametrize(
    "selection",
    [
        "Yes, please start navigation with the fastest route.",
        "Yes, let's take the fastest route. Please start navigation.",
        "Yes, please start navigation on the fastest route.",
        "Please start navigation on the fastest route.",
    ],
)
def test_base72_public_fastest_phrases_start_exact_route(
    runtime: CARGuardOrchestrator,
    client: NavigationCreateClient,
    full_tools: tuple[dict[str, Any], ...],
    inactive_navigation: dict[str, Any],
    base72_routes: dict[str, Any],
    selection: str,
) -> None:
    context_id = f"base72-public-fastest-{len(selection)}"
    goal_id, prompt = _run_to_route_prompt(
        runtime,
        context_id=context_id,
        tools=full_tools,
        navigation_result=inactive_navigation,
        route_result=base72_routes,
    )

    assert prompt.tool_calls == ()
    assert prompt.text is not None
    assert "fastest and shortest" in prompt.text
    assert "B303" in prompt.text and "A82" in prompt.text
    assert "153.91 kilometers" in prompt.text
    assert "1 hour and 54 minutes" in prompt.text
    assert "no toll roads" in prompt.text
    assert "2 further route alternatives" in prompt.text
    assert "L38" not in prompt.text and "L842" not in prompt.text

    selection_turn = f"{context_id}-selection"
    started = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=selection_turn,
            text=selection,
        )
    )

    assert started.tool_calls == (
        {
            "tool_name": "set_new_navigation",
            "arguments": {"route_ids": [FASTEST_ROUTE_ID]},
        },
    )
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.grounded_value_sources_by_goal[goal_id]["route_choice_alias"] == (
        selection_turn
    )
    route_evidence_id = session.derived_value_evidence_by_goal[goal_id]["route_ids"]
    route_evidence = session.evidence.evidence[route_evidence_id]
    assert route_evidence.value == [FASTEST_ROUTE_ID]
    assert route_evidence.derived_from
    assert client.action_calls == 0

    completed = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-set-result",
            results=[
                (
                    "set_new_navigation",
                    {
                        "status": "SUCCESS",
                        "result": {
                            "navigation_set": True,
                            "start_id": BONN_ID,
                            "waypoints": [BONN_ID, FRANKFURT_ID],
                            "destination_id": FRANKFURT_ID,
                        },
                    },
                )
            ],
        )
    )

    assert completed.tool_calls == ()
    assert completed.text is not None
    assert "started navigation" in completed.text.casefold()
    assert session.goal_dag.get(goal_id).status is GoalStatus.DONE


@pytest.mark.parametrize(
    "case",
    ["wrong_endpoint", "duplicate_fastest", "invalid_toll_type"],
)
def test_base72_malformed_route_results_fail_closed_before_set(
    runtime: CARGuardOrchestrator,
    client: NavigationCreateClient,
    full_tools: tuple[dict[str, Any], ...],
    inactive_navigation: dict[str, Any],
    base72_routes: dict[str, Any],
    case: str,
) -> None:
    context_id = f"base72-malformed-{case}"
    _malform_routes(base72_routes, case)

    _, blocked = _run_to_route_prompt(
        runtime,
        context_id=context_id,
        tools=full_tools,
        navigation_result=inactive_navigation,
        route_result=base72_routes,
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None and "couldn't verify" in blocked.text
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_hall70_missing_route_ids_blocks_before_all_reads(
    runtime: CARGuardOrchestrator,
    client: NavigationCreateClient,
    hall70_tools: tuple[dict[str, Any], ...],
) -> None:
    context_id = "hall70-missing-route-ids"
    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="Navigate to Frankfurt.",
            tools=hall70_tools,
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


def test_navigation_create_recipe_id_uses_deterministic_workflow(
    runtime: CARGuardOrchestrator,
    client: NavigationCreateClient,
    full_tools: tuple[dict[str, Any], ...],
) -> None:
    client.intent = IntentDraft.model_validate(
        {
            "language": "en",
            "intent_kind": "action",
            "call_for_action": True,
            "goals": [
                {
                    "semantic_operation": "navigation_create",
                    "desired_outcome": {"location": "Frankfurt"},
                }
            ],
            "explicit_slots": {"location": "Frankfurt"},
        }
    )

    outbound = runtime.handle_event(
        _user_event(
            context_id="base72-recipe-id",
            message_id="base72-recipe-id-user",
            text=REQUEST,
            tools=full_tools,
        )
    )

    assert outbound.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    assert client.action_calls == 0


def test_active_navigation_never_uses_set_new_navigation(
    runtime: CARGuardOrchestrator,
    client: NavigationCreateClient,
    full_tools: tuple[dict[str, Any], ...],
    active_navigation: dict[str, Any],
) -> None:
    context_id = "base72-active-navigation"
    current = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=REQUEST,
            tools=full_tools,
        )
    )

    assert current.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )
    blocked = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            results=[("get_current_navigation_state", active_navigation)],
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_inactive_navigation_with_stale_route_data_fails_closed(
    runtime: CARGuardOrchestrator,
    client: NavigationCreateClient,
    full_tools: tuple[dict[str, Any], ...],
) -> None:
    context_id = "base72-contradictory-inactive-navigation"
    current = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=REQUEST,
            tools=full_tools,
        )
    )
    assert current.tool_calls
    blocked = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            results=[
                (
                    "get_current_navigation_state",
                    {
                        "status": "SUCCESS",
                        "result": {
                            "navigation_active": False,
                            "waypoints_id": [BONN_ID, FRANKFURT_ID],
                            "routes_to_final_destination_id": [FASTEST_ROUTE_ID],
                        },
                    },
                )
            ],
        )
    )

    assert blocked.tool_calls == ()
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base72_unsafe_route_choices_keep_set_blocked(
    runtime: CARGuardOrchestrator,
    client: NavigationCreateClient,
    full_tools: tuple[dict[str, Any], ...],
    inactive_navigation: dict[str, Any],
    base72_routes: dict[str, Any],
) -> None:
    context_id = "base72-unsafe-route-choice"
    _, prompt = _run_to_route_prompt(
        runtime,
        context_id=context_id,
        tools=full_tools,
        navigation_result=inactive_navigation,
        route_result=base72_routes,
    )
    assert prompt.text is not None

    for index, unsafe in enumerate(
        (
            "Don't start navigation on the fastest route.",
            'She said, "Please start navigation on the fastest route."',
            "Please start navigation on the fastest route. Open the sunroof.",
        )
    ):
        rejected = runtime.handle_event(
            _user_event(
                context_id=context_id,
                message_id=f"{context_id}-unsafe-{index}",
                text=unsafe,
            )
        )
        assert rejected.tool_calls == ()
        assert rejected.text is not None
        session = runtime.sessions.get(context_id)
        assert session is not None and session.budget.attempted_sets == set()

    started = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-valid",
            text="Please start navigation on the fastest route.",
        )
    )
    assert started.tool_calls == (
        {
            "tool_name": "set_new_navigation",
            "arguments": {"route_ids": [FASTEST_ROUTE_ID]},
        },
    )
    assert client.action_calls == 0


def test_failed_navigation_set_never_claims_completion(
    runtime: CARGuardOrchestrator,
    client: NavigationCreateClient,
    full_tools: tuple[dict[str, Any], ...],
    inactive_navigation: dict[str, Any],
    base72_routes: dict[str, Any],
) -> None:
    context_id = "base72-set-failure"
    goal_id, prompt = _run_to_route_prompt(
        runtime,
        context_id=context_id,
        tools=full_tools,
        navigation_result=inactive_navigation,
        route_result=base72_routes,
    )
    assert prompt.text is not None

    started = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-selection",
            text="Yes, please start navigation with the fastest route.",
        )
    )
    assert started.tool_calls == (
        {
            "tool_name": "set_new_navigation",
            "arguments": {"route_ids": [FASTEST_ROUTE_ID]},
        },
    )

    failed = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-set-result",
            results=[
                (
                    "set_new_navigation",
                    {
                        "status": "FAILURE",
                        "errors": {"SET_NAVIGATION_001": "navigation rejected"},
                    },
                )
            ],
        )
    )

    assert failed.tool_calls == ()
    assert failed.text is not None
    assert "started navigation" not in failed.text.casefold()
    assert "couldn't verify" in failed.text.casefold()
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.goal_dag.get(goal_id).status is GoalStatus.FAILED
    assert client.action_calls == 0
