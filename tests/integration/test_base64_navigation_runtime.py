from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import DecisionProposal
from track_1_agent_under_test.car_guard.llm import LLMFailure
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
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


BASE64_TOOLS = (
    _tool(
        "get_current_navigation_state",
        {"detailed_information": {"type": "boolean"}},
    ),
    _tool(
        "get_location_id_by_location_name",
        {"location": {"type": "string"}},
        ["location"],
    ),
    _tool(
        "get_routes_from_start_to_destination",
        {
            "start_id": {"type": "string"},
            "destination_id": {"type": "string"},
        },
        ["start_id", "destination_id"],
    ),
    _tool(
        "navigation_add_one_waypoint",
        {
            "waypoint_id_before_new_waypoint": {"type": "string"},
            "route_id_leading_to_new_waypoint": {"type": "string"},
            "route_id_leading_away_from_new_waypoint": {"type": "string"},
            "waypoint_id_to_add": {"type": "string"},
            "waypoint_id_after_new_waypoint": {"type": "string"},
        },
        [
            "waypoint_id_before_new_waypoint",
            "route_id_leading_to_new_waypoint",
            "route_id_leading_away_from_new_waypoint",
            "waypoint_id_to_add",
            "waypoint_id_after_new_waypoint",
        ],
    ),
    _tool(
        "search_poi_at_location",
        {
            "location_id": {"type": "string"},
            "category_poi": {"type": "string"},
        },
        ["location_id", "category_poi"],
    ),
    _tool(
        "navigation_replace_one_waypoint",
        {
            "waypoint_id_to_replace": {"type": "string"},
            "new_waypoint_id": {"type": "string"},
            "route_id_leading_to_new_waypoint": {"type": "string"},
            "route_id_leading_away_from_new_waypoint": {"type": "string"},
        },
        [
            "waypoint_id_to_replace",
            "new_waypoint_id",
            "route_id_leading_to_new_waypoint",
            "route_id_leading_away_from_new_waypoint",
        ],
    ),
)

HALL62_TOOLS = tuple(
    spec
    for spec in BASE64_TOOLS
    if spec["function"]["name"] != "navigation_replace_one_waypoint"
)


def _action_intent(
    semantic_operation: str, desired_outcome: dict[str, Any]
) -> dict[str, Any]:
    return {
        "language": "en",
        "intent_kind": "action",
        "call_for_action": True,
        "goals": [
            {
                "semantic_operation": semantic_operation,
                "desired_outcome": desired_outcome,
            }
        ],
        "explicit_slots": dict(desired_outcome),
    }


def _base64_intents() -> list[dict[str, Any]]:
    return [
        _action_intent(
            "navigation_add_one_waypoint",
            {
                "new_waypoint_name": "Stuttgart",
                "previous_waypoint_name": "Cologne",
            },
        ),
        _action_intent(
            "search_poi_at_next_navigation_stop",
            {"category": "restaurants"},
        ),
        _action_intent(
            "navigation_replace_one_waypoint",
            {
                "waypoint_name_to_replace": "Cologne",
                "new_waypoint_name": "Karlsruhe",
                "route_choice_alias": "fastest",
            },
        ),
    ]


class _SequentialIntentClient:
    def __init__(self, intents: list[dict[str, Any]] | None = None) -> None:
        self.intents = [
            IntentDraft.model_validate(intent)
            for intent in (intents if intents is not None else _base64_intents())
        ]
        self.intent_calls = 0
        self.action_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        del messages, critic
        if response_model is IntentDraft:
            if self.intent_calls >= len(self.intents):
                raise AssertionError("unexpected additional intent extraction")
            intent = self.intents[self.intent_calls]
            self.intent_calls += 1
            return SimpleNamespace(value=intent)
        if response_model is DecisionProposal:
            self.action_calls += 1
            pytest.fail("the deterministic base64 workflow must not call the planner")
        raise AssertionError(f"unexpected response model: {response_model}")


class _StructuredFailureClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, messages, response_model, critic=False):
        del messages, response_model, critic
        self.calls += 1
        raise LLMFailure("synthetic structured-output failure")


def _runtime(
    client: _SequentialIntentClient | _StructuredFailureClient,
) -> CARGuardOrchestrator:
    config = AgentConfig(llm="test/model", soft_max_steps=48)
    return CARGuardOrchestrator(config, client_factory=lambda session: client)


def _user_event(
    *,
    context_id: str,
    message_id: str,
    text: str,
    tools: tuple[dict[str, Any], ...],
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        system_policy=(
            "Follow the current safety policy. "
            'CURRENT_LOCATION={"id":"loc_mon_279370","name":"Monaco"}'
        ),
        user_text=text,
        live_tools=tools,
    )


def _result_event(
    *, context_id: str, message_id: str, tool_name: str, content: Any
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        tool_results=({"toolName": tool_name, "content": content},),
    )


def _navigation_state(*, added: bool) -> dict[str, Any]:
    waypoint_rows = [
        ("loc_mon_279370", "Monaco"),
        ("loc_col_464166", "Cologne"),
    ]
    route_ids = ["rll_mon_col_344373"]
    if added:
        waypoint_rows.append(("loc_stu_828398", "Stuttgart"))
        route_ids.append("rll_col_stu_882834")
        route_ids.append("rll_stu_lux_346041")
    else:
        route_ids.append("rll_col_lux_460833")
    waypoint_rows.append(("loc_lux_222378", "Luxembourg"))
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [row[0] for row in waypoint_rows],
            "routes_to_final_destination_id": route_ids,
            "details": {
                "waypoints": [
                    {"id": location_id, "name": name}
                    for location_id, name in waypoint_rows
                ]
            },
        },
    }


_ROUTES: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {
    ("loc_col_464166", "loc_stu_828398"): (
        {
            "route_id": "rll_col_stu_882834",
            "name_via": "K329",
            "distance_km": 348.77,
            "duration_hours": 4,
            "duration_minutes": 20,
            "road_types": ["highway", "urban"],
            "includes_toll": False,
            "alias": ["fastest", "first", "shortest"],
        },
        {
            "route_id": "rll_col_stu_501376",
            "name_via": "A77, A21",
            "distance_km": 351.61,
            "duration_hours": 4,
            "duration_minutes": 25,
            "road_types": ["country road", "highway", "urban"],
            "includes_toll": False,
            "alias": ["second"],
        },
        {
            "route_id": "rll_col_stu_840248",
            "name_via": "L348",
            "distance_km": 358.45,
            "duration_hours": 4,
            "duration_minutes": 26,
            "road_types": ["country road", "highway", "urban"],
            "includes_toll": False,
            "alias": ["third"],
        },
    ),
    ("loc_stu_828398", "loc_lux_222378"): (
        {
            "route_id": "rll_stu_lux_346041",
            "name_via": "A53, B742, L991",
            "distance_km": 279.56,
            "duration_hours": 3,
            "duration_minutes": 33,
            "road_types": ["country road", "highway", "urban"],
            "includes_toll": False,
            "alias": ["fastest", "first", "shortest"],
        },
        {
            "route_id": "rll_stu_lux_239924",
            "name_via": "A57, A97",
            "distance_km": 288.95,
            "duration_hours": 3,
            "duration_minutes": 39,
            "road_types": ["country road", "highway", "urban"],
            "includes_toll": False,
            "alias": ["second"],
        },
        {
            "route_id": "rll_stu_lux_312704",
            "name_via": "L696",
            "distance_km": 296.75,
            "duration_hours": 3,
            "duration_minutes": 39,
            "road_types": ["country road", "urban"],
            "includes_toll": False,
            "alias": ["third"],
        },
    ),
    ("loc_mon_279370", "loc_kar_304825"): (
        {
            "route_id": "rll_mon_kar_837395",
            "name_via": "A67, A35, A20",
            "distance_km": 692.63,
            "duration_hours": 8,
            "duration_minutes": 37,
            "road_types": ["highway"],
            "includes_toll": False,
            "alias": ["fastest", "first", "shortest"],
        },
        {
            "route_id": "rll_mon_kar_348702",
            "name_via": "B754",
            "distance_km": 707.63,
            "duration_hours": 8,
            "duration_minutes": 47,
            "road_types": ["country road", "highway"],
            "includes_toll": False,
            "alias": ["second"],
        },
        {
            "route_id": "rll_mon_kar_544135",
            "name_via": "A2",
            "distance_km": 741.27,
            "duration_hours": 9,
            "duration_minutes": 13,
            "road_types": ["highway"],
            "includes_toll": False,
            "alias": ["third"],
        },
    ),
    ("loc_kar_304825", "loc_stu_828398"): (
        {
            "route_id": "rll_kar_stu_956053",
            "name_via": "B866, B555, B213",
            "distance_km": 72.49,
            "duration_hours": 0,
            "duration_minutes": 54,
            "road_types": ["country road"],
            "includes_toll": False,
            "alias": ["fastest", "first", "shortest"],
        },
        {
            "route_id": "rll_kar_stu_265298",
            "name_via": "K194, A80, A10",
            "distance_km": 72.84,
            "duration_hours": 0,
            "duration_minutes": 54,
            "road_types": ["highway", "urban"],
            "includes_toll": False,
            "alias": ["second"],
        },
        {
            "route_id": "rll_kar_stu_579129",
            "name_via": "K447, A8",
            "distance_km": 75.1,
            "duration_hours": 0,
            "duration_minutes": 57,
            "road_types": ["highway", "urban"],
            "includes_toll": False,
            "alias": ["third"],
        },
    ),
}


def _route_result(start_id: str, destination_id: str) -> dict[str, Any]:
    rows = _ROUTES[(start_id, destination_id)]
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    **row,
                    "start_id": start_id,
                    "destination_id": destination_id,
                }
                for row in rows
            ]
        },
    }


def _poi_result() -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "pois_found": [
                {
                    "id": "poi_res_561973",
                    "name": "Zum Goldenen Hahn",
                    "category": "restaurants",
                    "phone_number": "+49 849 3333006",
                    "corresponding_location_id": "loc_col_464166",
                },
                {
                    "id": "poi_res_224640",
                    "name": "Brauhaus Germania",
                    "category": "restaurants",
                    "phone_number": "+49 621 3893169",
                    "corresponding_location_id": "loc_col_464166",
                },
            ]
        },
    }


def _expect_call(outbound, tool_name: str, arguments: dict[str, Any]) -> None:
    assert outbound.tool_calls == ({"tool_name": tool_name, "arguments": arguments},)


def _drive_add_and_poi(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    tools: tuple[dict[str, Any], ...],
    post_add_state: dict[str, Any] | None = None,
    expect_poi_search: bool = True,
    forge_add_route_evidence: bool = False,
    add_route_includes_toll: bool = False,
) -> None:
    current = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-add-user",
            text="I want to add Stuttgart after Cologne.",
            tools=tools,
        )
    )
    _expect_call(
        current,
        "get_current_navigation_state",
        {"detailed_information": True},
    )
    location = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-initial-state",
            tool_name="get_current_navigation_state",
            content=_navigation_state(added=False),
        )
    )
    _expect_call(
        location,
        "get_location_id_by_location_name",
        {"location": "Stuttgart"},
    )
    route_to = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-stuttgart-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": "loc_stu_828398"}},
        )
    )
    _expect_call(
        route_to,
        "get_routes_from_start_to_destination",
        {"start_id": "loc_col_464166", "destination_id": "loc_stu_828398"},
    )
    route_to_result = _route_result("loc_col_464166", "loc_stu_828398")
    if add_route_includes_toll:
        route_to_result["result"]["routes"][0]["includes_toll"] = True
    route_from = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-route-to-stuttgart",
            tool_name="get_routes_from_start_to_destination",
            content=route_to_result,
        )
    )
    _expect_call(
        route_from,
        "get_routes_from_start_to_destination",
        {"start_id": "loc_stu_828398", "destination_id": "loc_lux_222378"},
    )
    add = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-route-from-stuttgart",
            tool_name="get_routes_from_start_to_destination",
            content=_route_result("loc_stu_828398", "loc_lux_222378"),
        )
    )
    _expect_call(
        add,
        "navigation_add_one_waypoint",
        {
            "waypoint_id_before_new_waypoint": "loc_col_464166",
            "route_id_leading_to_new_waypoint": "rll_col_stu_882834",
            "route_id_leading_away_from_new_waypoint": "rll_stu_lux_346041",
            "waypoint_id_to_add": "loc_stu_828398",
            "waypoint_id_after_new_waypoint": "loc_lux_222378",
        },
    )
    if forge_add_route_evidence:
        session = runtime.sessions.get(context_id)
        assert session is not None
        add_goal_id = session.goal_dag.goals[0].goal_id
        evidence_id = session.derived_value_evidence_by_goal[add_goal_id][
            "route_id_leading_away_from_waypoint"
        ]
        evidence = session.evidence.evidence[evidence_id]
        session.evidence.evidence[evidence_id] = evidence.model_copy(
            update={"derivation": "forged_fastest_route_v1"}
        )
    added = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-add-result",
            tool_name="navigation_add_one_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_added": True,
                    "new_waypoints_id": [
                        "loc_mon_279370",
                        "loc_col_464166",
                        "loc_stu_828398",
                        "loc_lux_222378",
                    ],
                    "new_routes_id": [
                        "rll_mon_col_344373",
                        "rll_col_stu_882834",
                        "rll_stu_lux_346041",
                    ],
                },
            },
        )
    )
    assert added.tool_calls == ()
    assert added.text is not None
    if forge_add_route_evidence:
        assert "fastest route for each segment" not in added.text
        assert "alternative routes" not in added.text
        assert "toll roads" not in added.text
    else:
        assert "fastest route for each segment" in added.text
        assert "alternative routes" in added.text
        if add_route_includes_toll:
            assert "includes toll roads" in added.text
        else:
            assert "toll roads" not in added.text

    fresh = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-poi-user",
            text="I'm hungry. Find a restaurant at my next stop.",
            tools=tools,
        )
    )
    _expect_call(
        fresh,
        "get_current_navigation_state",
        {"detailed_information": True},
    )
    search = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-added-state",
            tool_name="get_current_navigation_state",
            content=post_add_state or _navigation_state(added=True),
        )
    )
    if not expect_poi_search:
        assert search.tool_calls == ()
        assert search.text is not None
        return
    _expect_call(
        search,
        "search_poi_at_location",
        {"location_id": "loc_col_464166", "category_poi": "restaurants"},
    )
    displayed = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-poi-result",
            tool_name="search_poi_at_location",
            content=_poi_result(),
        )
    )
    assert displayed.tool_calls == ()
    assert displayed.text is not None
    assert "Zum Goldenen Hahn" in displayed.text
    assert "Brauhaus Germania" in displayed.text


def test_base64_exact_add_next_stop_poi_then_fastest_replace() -> None:
    client = _SequentialIntentClient()
    runtime = _runtime(client)
    context_id = "base64-exact"
    _drive_add_and_poi(runtime, context_id=context_id, tools=BASE64_TOOLS)

    location = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-replace-user",
            text=(
                "Neither restaurant works. Replace Cologne with Karlsruhe instead. "
                "I know a restaurant there, so don't search for one. Use the fastest route."
            ),
            tools=BASE64_TOOLS,
        )
    )
    _expect_call(
        location,
        "get_location_id_by_location_name",
        {"location": "Karlsruhe"},
    )
    route_to = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-karlsruhe-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": "loc_kar_304825"}},
        )
    )
    _expect_call(
        route_to,
        "get_routes_from_start_to_destination",
        {"start_id": "loc_mon_279370", "destination_id": "loc_kar_304825"},
    )
    route_from = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-route-to-karlsruhe",
            tool_name="get_routes_from_start_to_destination",
            content=_route_result("loc_mon_279370", "loc_kar_304825"),
        )
    )
    _expect_call(
        route_from,
        "get_routes_from_start_to_destination",
        {"start_id": "loc_kar_304825", "destination_id": "loc_stu_828398"},
    )
    replace = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-route-from-karlsruhe",
            tool_name="get_routes_from_start_to_destination",
            content=_route_result("loc_kar_304825", "loc_stu_828398"),
        )
    )
    _expect_call(
        replace,
        "navigation_replace_one_waypoint",
        {
            "waypoint_id_to_replace": "loc_col_464166",
            "new_waypoint_id": "loc_kar_304825",
            "route_id_leading_to_new_waypoint": "rll_mon_kar_837395",
            "route_id_leading_away_from_new_waypoint": "rll_kar_stu_956053",
        },
    )
    completed = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-replace-result",
            tool_name="navigation_replace_one_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_replaced": True,
                    "new_waypoints": [
                        "loc_mon_279370",
                        "loc_kar_304825",
                        "loc_stu_828398",
                        "loc_lux_222378",
                    ],
                    "new_routes": [
                        "rll_mon_kar_837395",
                        "rll_kar_stu_956053",
                        "rll_stu_lux_346041",
                    ],
                },
            },
        )
    )
    assert completed.tool_calls == ()
    assert completed.text is not None
    assert client.intent_calls == 0
    assert client.action_calls == 0

    session = runtime.sessions.get(context_id)
    assert session is not None
    all_calls = [
        call.tool_name
        for calls in session.completed_action_calls_by_goal.values()
        for call in calls
    ]
    assert "navigation_add_one_waypoint" in all_calls
    assert "navigation_replace_one_waypoint" in all_calls
    assert all(
        not (
            result.tool_name == "search_poi_at_location"
            and result.arguments.get("location_id") == "loc_kar_304825"
        )
        for result in session.successful_read_results
    )


def test_hallucination62_missing_replace_tool_stops_after_completed_prefix() -> None:
    client = _SequentialIntentClient()
    runtime = _runtime(client)
    context_id = "hallucination62-exact"
    _drive_add_and_poi(runtime, context_id=context_id, tools=HALL62_TOOLS)
    session = runtime.sessions.get(context_id)
    assert session is not None
    read_count = len(session.successful_read_results)
    attempted_sets = set(session.budget.attempted_sets)
    completed_actions = dict(session.completed_action_calls_by_goal)

    blocked = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-replace-user",
            text="Replace Cologne with Karlsruhe. I want the fastest route.",
            tools=HALL62_TOOLS,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None and "required control" in blocked.text
    assert len(session.successful_read_results) == read_count
    assert session.budget.attempted_sets == attempted_sets
    assert session.completed_action_calls_by_goal == completed_actions
    assert any(
        call.tool_name == "navigation_add_one_waypoint"
        for calls in completed_actions.values()
        for call in calls
    )
    assert client.intent_calls == 0
    assert client.action_calls == 0


def test_base64_add_rejects_malformed_detailed_navigation_state() -> None:
    client = _SequentialIntentClient()
    runtime = _runtime(client)
    context_id = "base64-malformed-state"
    current = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="I want to add Stuttgart after Cologne.",
            tools=BASE64_TOOLS,
        )
    )
    _expect_call(
        current,
        "get_current_navigation_state",
        {"detailed_information": True},
    )
    malformed = _navigation_state(added=False)
    malformed["result"].pop("details")
    blocked = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=malformed,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base64_strict_add_skips_failing_structured_model() -> None:
    client = _StructuredFailureClient()
    runtime = _runtime(client)

    current = runtime.handle_event(
        _user_event(
            context_id="base64-deterministic-intent",
            message_id="base64-deterministic-intent-user",
            text="I want to add Stuttgart after Cologne.",
            tools=BASE64_TOOLS,
        )
    )

    _expect_call(
        current,
        "get_current_navigation_state",
        {"detailed_information": True},
    )
    assert client.calls == 0


def test_base64_add_rejects_false_fastest_route_metadata() -> None:
    client = _SequentialIntentClient()
    runtime = _runtime(client)
    context_id = "base64-false-fastest"
    current = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="I want to add Stuttgart after Cologne.",
            tools=BASE64_TOOLS,
        )
    )
    assert current.tool_calls
    location = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=_navigation_state(added=False),
        )
    )
    assert location.tool_calls
    route_to = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": "loc_stu_828398"}},
        )
    )
    assert route_to.tool_calls
    false_fastest = _route_result("loc_col_464166", "loc_stu_828398")
    false_fastest["result"]["routes"][0]["duration_minutes"] = 30
    blocked = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-route",
            tool_name="get_routes_from_start_to_destination",
            content=false_fastest,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None and "couldn't verify" in blocked.text
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


def test_base64_next_stop_rejects_state_that_disagrees_with_completed_add() -> None:
    client = _SequentialIntentClient()
    runtime = _runtime(client)
    wrong_state = _navigation_state(added=True)
    waypoint_ids = wrong_state["result"]["waypoints_id"]
    waypoint_ids[1], waypoint_ids[2] = waypoint_ids[2], waypoint_ids[1]
    waypoint_rows = wrong_state["result"]["details"]["waypoints"]
    waypoint_rows[1], waypoint_rows[2] = waypoint_rows[2], waypoint_rows[1]

    _drive_add_and_poi(
        runtime,
        context_id="base64-wrong-post-add-state",
        tools=BASE64_TOOLS,
        post_add_state=wrong_state,
        expect_poi_search=False,
    )

    session = runtime.sessions.get("base64-wrong-post-add-state")
    assert session is not None
    assert all(
        result.tool_name != "search_poi_at_location"
        for result in session.successful_read_results
    )
    assert client.action_calls == 0


def test_base64_add_completion_rejects_forged_fastest_route_evidence() -> None:
    client = _SequentialIntentClient()
    runtime = _runtime(client)

    _drive_add_and_poi(
        runtime,
        context_id="base64-forged-add-route-evidence",
        tools=BASE64_TOOLS,
        forge_add_route_evidence=True,
    )

    assert client.intent_calls == 0
    assert client.action_calls == 0


def test_base64_add_completion_aggregates_toll_across_both_segments() -> None:
    client = _SequentialIntentClient()
    runtime = _runtime(client)

    _drive_add_and_poi(
        runtime,
        context_id="base64-add-route-with-toll",
        tools=BASE64_TOOLS,
        add_route_includes_toll=True,
    )

    assert client.intent_calls == 0
    assert client.action_calls == 0


def test_base64_replace_does_not_reuse_stale_displayed_poi_state() -> None:
    client = _SequentialIntentClient()
    runtime = _runtime(client)
    context_id = "base64-stale-displayed-state"
    _drive_add_and_poi(runtime, context_id=context_id, tools=BASE64_TOOLS)
    session = runtime.sessions.get(context_id)
    assert session is not None and session.displayed_destination_pois
    session.evidence.invalidate_tool_state()

    current = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-replace-user",
            text="Replace Cologne with Karlsruhe. I want the fastest route.",
            tools=BASE64_TOOLS,
        )
    )

    assert current.tool_calls != (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Karlsruhe"},
        },
    )
    if current.tool_calls:
        _expect_call(
            current,
            "get_current_navigation_state",
            {"detailed_information": True},
        )
    else:
        assert current.text is not None
    assert client.action_calls == 0


def test_base64_replace_without_alias_rejects_false_fastest_metadata() -> None:
    client = _SequentialIntentClient(
        [
            _action_intent(
                "navigation_replace_one_waypoint",
                {
                    "waypoint_name_to_replace": "Cologne",
                    "new_waypoint_name": "Karlsruhe",
                },
            )
        ]
    )
    runtime = _runtime(client)
    context_id = "base64-replace-no-alias-false-fastest"
    current = runtime.handle_event(
        _user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text="Replace Cologne with Karlsruhe.",
            tools=BASE64_TOOLS,
        )
    )
    assert current.tool_calls
    location = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-state",
            tool_name="get_current_navigation_state",
            content=_navigation_state(added=True),
        )
    )
    assert location.tool_calls
    route_to = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": "loc_kar_304825"}},
        )
    )
    assert route_to.tool_calls
    false_fastest = _route_result("loc_mon_279370", "loc_kar_304825")
    false_fastest["result"]["routes"][0]["duration_minutes"] = 57
    false_fastest["result"]["routes"][1]["duration_minutes"] = 47
    blocked = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id=f"{context_id}-route",
            tool_name="get_routes_from_start_to_destination",
            content=false_fastest,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None and "couldn't verify" in blocked.text
    session = runtime.sessions.get(context_id)
    assert session is not None and session.budget.attempted_sets == set()
    assert client.action_calls == 0


SYNTHETIC_NAVIGATION_EDIT_TOOLS = (
    _tool(
        "get_current_navigation_state",
        {"detailed_information": {"type": "boolean"}},
    ),
    _tool(
        "get_location_id_by_location_name",
        {"location": {"type": "string"}},
        ["location"],
    ),
    _tool(
        "get_routes_from_start_to_destination",
        {
            "start_id": {"type": "string"},
            "destination_id": {"type": "string"},
        },
        ["start_id", "destination_id"],
    ),
    _tool(
        "navigation_add_one_waypoint",
        {
            "waypoint_id_before_new_waypoint": {"type": "string"},
            "route_id_leading_to_new_waypoint": {"type": "string"},
            "route_id_leading_away_from_new_waypoint": {"type": "string"},
            "waypoint_id_to_add": {"type": "string"},
            "waypoint_id_after_new_waypoint": {"type": "string"},
        },
        [
            "waypoint_id_before_new_waypoint",
            "route_id_leading_to_new_waypoint",
            "waypoint_id_to_add",
        ],
    ),
    _tool(
        "navigation_replace_one_waypoint",
        {
            "waypoint_id_to_replace": {"type": "string"},
            "new_waypoint_id": {"type": "string"},
            "route_id_leading_to_new_waypoint": {"type": "string"},
            "route_id_leading_away_from_new_waypoint": {"type": "string"},
        },
        [
            "waypoint_id_to_replace",
            "new_waypoint_id",
            "route_id_leading_to_new_waypoint",
            "route_id_leading_away_from_new_waypoint",
        ],
    ),
)


def _synthetic_navigation_state(*, with_intermediate: bool) -> dict[str, Any]:
    waypoints = [("loc_alpha", "Alpha")]
    routes: list[str] = []
    if with_intermediate:
        waypoints.append(("loc_beta", "Beta"))
        routes.append("rsy_alpha_beta")
    waypoints.append(("loc_omega", "Omega"))
    routes.append("rsy_beta_omega" if with_intermediate else "rsy_alpha_omega")
    return {
        "status": "SUCCESS",
        "result": {
            "navigation_active": True,
            "waypoints_id": [identifier for identifier, _ in waypoints],
            "routes_to_final_destination_id": routes,
            "details": {
                "waypoints": [
                    {"id": identifier, "name": name}
                    for identifier, name in waypoints
                ]
            },
        },
    }


def _synthetic_route_result(start_id: str, destination_id: str) -> dict[str, Any]:
    suffix = f"{start_id.removeprefix('loc_')}_{destination_id.removeprefix('loc_')}"
    rows = (
        (f"rsy_{suffix}_fast", "A1", 12, 10, ["first", "fastest"]),
        (f"rsy_{suffix}_short", "B2", 9, 12, ["second", "shortest"]),
        (f"rsy_{suffix}_third", "K3", 15, 14, ["third"]),
    )
    return {
        "status": "SUCCESS",
        "result": {
            "routes": [
                {
                    "route_id": route_id,
                    "start_id": start_id,
                    "destination_id": destination_id,
                    "name_via": via,
                    "distance_km": distance,
                    "duration_hours": 0,
                    "duration_minutes": minutes,
                    "road_types": ["synthetic road"],
                    "includes_toll": False,
                    "alias": aliases,
                }
                for route_id, via, distance, minutes, aliases in rows
            ]
        },
    }


def _synthetic_user_event(
    *, context_id: str, message_id: str, text: str
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        system_policy=(
            "Follow the current safety policy. "
            'CURRENT_LOCATION={"id":"loc_alpha","name":"Alpha"}'
        ),
        user_text=text,
        live_tools=SYNTHETIC_NAVIGATION_EDIT_TOOLS,
    )


@pytest.mark.parametrize(
    ("route_request", "expected_route_id", "expect_fastest_notice"),
    [
        (" Use the second route.", "rsy_omega_gamma_short", False),
        ("", "rsy_omega_gamma_fast", True),
    ],
)
def test_synthetic_add_after_final_appends_with_one_selected_route(
    route_request: str,
    expected_route_id: str,
    expect_fastest_notice: bool,
) -> None:
    client = _StructuredFailureClient()
    runtime = _runtime(client)
    context_id = "synthetic-append-final"

    current = runtime.handle_event(
        _synthetic_user_event(
            context_id=context_id,
            message_id="append-user",
            text=f"Add Gamma as a new waypoint after Omega.{route_request}",
        )
    )
    _expect_call(
        current, "get_current_navigation_state", {"detailed_information": True}
    )
    location = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id="append-state",
            tool_name="get_current_navigation_state",
            content=_synthetic_navigation_state(with_intermediate=False),
        )
    )
    _expect_call(
        location,
        "get_location_id_by_location_name",
        {"location": "Gamma"},
    )
    route = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id="append-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": "loc_gamma"}},
        )
    )
    _expect_call(
        route,
        "get_routes_from_start_to_destination",
        {"start_id": "loc_omega", "destination_id": "loc_gamma"},
    )
    route_result = _synthetic_route_result("loc_omega", "loc_gamma")
    if expect_fastest_notice:
        route_result["result"]["routes"][0]["includes_toll"] = True
    add = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id="append-routes",
            tool_name="get_routes_from_start_to_destination",
            content=route_result,
        )
    )
    _expect_call(
        add,
        "navigation_add_one_waypoint",
        {
            "waypoint_id_before_new_waypoint": "loc_omega",
            "route_id_leading_to_new_waypoint": expected_route_id,
            "waypoint_id_to_add": "loc_gamma",
        },
    )
    completed = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id="append-complete",
            tool_name="navigation_add_one_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_added": True,
                    "new_waypoints_id": [
                        "loc_alpha",
                        "loc_omega",
                        "loc_gamma",
                    ],
                    "new_routes_id": [
                        "rsy_alpha_omega",
                        expected_route_id,
                    ],
                },
            },
        )
    )

    assert completed.tool_calls == ()
    assert completed.text is not None
    assert (
        "fastest route for each segment" in completed.text
    ) is expect_fastest_notice
    assert ("includes toll roads" in completed.text) is expect_fastest_notice
    assert client.calls == 0


def test_synthetic_replace_uses_unique_shortest_route_for_both_segments() -> None:
    client = _StructuredFailureClient()
    runtime = _runtime(client)
    context_id = "synthetic-replace-shortest"

    current = runtime.handle_event(
        _synthetic_user_event(
            context_id=context_id,
            message_id="replace-user",
            text="Replace Beta with Gamma. Use the shortest route.",
        )
    )
    _expect_call(
        current, "get_current_navigation_state", {"detailed_information": True}
    )
    location = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id="replace-state",
            tool_name="get_current_navigation_state",
            content=_synthetic_navigation_state(with_intermediate=True),
        )
    )
    _expect_call(
        location,
        "get_location_id_by_location_name",
        {"location": "Gamma"},
    )
    route_to = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id="replace-location",
            tool_name="get_location_id_by_location_name",
            content={"status": "SUCCESS", "result": {"id": "loc_gamma"}},
        )
    )
    _expect_call(
        route_to,
        "get_routes_from_start_to_destination",
        {"start_id": "loc_alpha", "destination_id": "loc_gamma"},
    )
    route_from = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id="replace-route-to",
            tool_name="get_routes_from_start_to_destination",
            content=_synthetic_route_result("loc_alpha", "loc_gamma"),
        )
    )
    _expect_call(
        route_from,
        "get_routes_from_start_to_destination",
        {"start_id": "loc_gamma", "destination_id": "loc_omega"},
    )
    replace = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id="replace-route-from",
            tool_name="get_routes_from_start_to_destination",
            content=_synthetic_route_result("loc_gamma", "loc_omega"),
        )
    )
    _expect_call(
        replace,
        "navigation_replace_one_waypoint",
        {
            "waypoint_id_to_replace": "loc_beta",
            "new_waypoint_id": "loc_gamma",
            "route_id_leading_to_new_waypoint": "rsy_alpha_gamma_short",
            "route_id_leading_away_from_new_waypoint": "rsy_gamma_omega_short",
        },
    )
    completed = runtime.handle_event(
        _result_event(
            context_id=context_id,
            message_id="replace-complete",
            tool_name="navigation_replace_one_waypoint",
            content={
                "status": "SUCCESS",
                "result": {
                    "waypoint_replaced": True,
                    "new_waypoints": ["loc_alpha", "loc_gamma", "loc_omega"],
                    "new_routes": [
                        "rsy_alpha_gamma_short",
                        "rsy_gamma_omega_short",
                    ],
                },
            },
        )
    )

    assert completed.tool_calls == ()
    assert completed.text is not None
    assert "fastest route for each segment" not in completed.text
    assert client.calls == 0
