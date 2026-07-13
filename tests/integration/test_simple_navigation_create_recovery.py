from __future__ import annotations

from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import IntentFrame, IntentKind
from track_1_agent_under_test.car_guard.planning.intent_grounding import (
    recover_simple_navigation_create_intent,
)
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
)


START_ID = "loc_wes_111222"
DESTINATION_ID = "loc_nor_333444"
FIRST_ROUTE_ID = "rll_wes_nor_555666"
SECOND_ROUTE_ID = "rll_wes_nor_777888"
POLICY = (
    "A new navigation may only be set while navigation is inactive. The overall "
    "route must start at the current car location. Present route alternatives "
    "before setting navigation. "
    'CURRENT_LOCATION={"id":"loc_wes_111222","name":"Westhaven"}'
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


TOOLS = (
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
        "set_new_navigation",
        {"route_ids": {"type": "array", "items": {"type": "string"}}},
        ["route_ids"],
    ),
)


class NoModelClient:
    def generate(self, **_: Any) -> Any:
        raise AssertionError("a strict simple-navigation command must bypass the model")


def _seed() -> IntentFrame:
    return IntentFrame(
        language="en",
        call_for_action=False,
        goals=[],
        intent_source_turn_ids=["turn-simple"],
        intent_kind=IntentKind.CONVERSATION,
    )


@pytest.mark.parametrize(
    ("utterance", "destination"),
    (
        ("Navigate to Northbridge.", "Northbridge"),
        ("Please drive me to Northbridge Research Park.", "Northbridge Research Park"),
        ("Could you take me to the Northbridge Science Center?", "Northbridge Science Center"),
    ),
)
def test_strict_simple_navigation_create_recovery(
    utterance: str, destination: str
) -> None:
    recovered = recover_simple_navigation_create_intent(
        utterance,
        _seed(),
        turn_id="turn-simple",
    )

    assert recovered.call_for_action
    assert recovered.intent_kind is IntentKind.ACTION
    assert len(recovered.goals) == 1
    assert recovered.goals[0].semantic_operation == "create_navigation"
    assert recovered.goals[0].desired_outcome == {"location": destination}
    assert recovered.explicit_slots == {"location": destination}


@pytest.mark.parametrize(
    "utterance",
    (
        "Don't navigate to Northbridge.",
        "If the road is clear, navigate to Northbridge.",
        'She said, "Navigate to Northbridge."',
        "Navigate to Northbridge or Eastmere.",
        "Navigate to Northbridge and Eastmere.",
        "Navigate to Northbridge and open the sunroof.",
        "Navigate to Northbridge. Then open the sunroof.",
        "Navigate there.",
        "Take me to my office.",
        "Drive to Current Destination.",
    ),
)
def test_strict_simple_navigation_create_recovery_rejects_unsafe_forms(
    utterance: str,
) -> None:
    seed = _seed()

    assert (
        recover_simple_navigation_create_intent(
            utterance,
            seed,
            turn_id="turn-simple",
        )
        is seed
    )


def _result_event(
    context_id: str,
    message_id: str,
    tool_name: str,
    content: dict[str, Any],
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        tool_results=({"toolName": tool_name, "content": content},),
    )


def test_simple_navigation_create_uses_existing_route_choice_workflow_without_model() -> None:
    runtime = CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=24),
        client_factory=lambda session: NoModelClient(),
    )
    context_id = "simple-navigation-create-synthetic"

    current = runtime.handle_event(
        InboundEvent(
            message_id="simple-user",
            context_id=context_id,
            system_policy=POLICY,
            user_text="Drive me to Northbridge Research Park.",
            live_tools=TOOLS,
        )
    )
    assert current.tool_calls == (
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        },
    )

    location = runtime.handle_event(
        _result_event(
            context_id,
            "simple-state",
            "get_current_navigation_state",
            {
                "status": "SUCCESS",
                "result": {
                    "navigation_active": False,
                    "waypoints_id": [],
                    "routes_to_final_destination_id": [],
                },
            },
        )
    )
    assert location.tool_calls == (
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Northbridge Research Park"},
        },
    )

    routes = runtime.handle_event(
        _result_event(
            context_id,
            "simple-location",
            "get_location_id_by_location_name",
            {"status": "SUCCESS", "result": {"id": DESTINATION_ID}},
        )
    )
    assert routes.tool_calls == (
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": START_ID,
                "destination_id": DESTINATION_ID,
            },
        },
    )

    route_prompt = runtime.handle_event(
        _result_event(
            context_id,
            "simple-routes",
            "get_routes_from_start_to_destination",
            {
                "status": "SUCCESS",
                "result": {
                    "routes": [
                        {
                            "route_id": FIRST_ROUTE_ID,
                            "start_id": START_ID,
                            "destination_id": DESTINATION_ID,
                            "name_via": "A12",
                            "distance_km": 41.2,
                            "duration_hours": 0,
                            "duration_minutes": 38,
                            "road_types": ["highway"],
                            "includes_toll": False,
                            "alias": ["fastest", "first"],
                        },
                        {
                            "route_id": SECOND_ROUTE_ID,
                            "start_id": START_ID,
                            "destination_id": DESTINATION_ID,
                            "name_via": "B34",
                            "distance_km": 39.8,
                            "duration_hours": 0,
                            "duration_minutes": 44,
                            "road_types": ["urban"],
                            "includes_toll": False,
                            "alias": ["shortest", "second"],
                        },
                    ]
                },
            },
        )
    )
    assert route_prompt.tool_calls == ()
    assert route_prompt.text is not None and "Northbridge Research Park" in route_prompt.text

    selected = runtime.handle_event(
        InboundEvent(
            message_id="simple-selection",
            context_id=context_id,
            user_text="Please start navigation on the second route.",
        )
    )
    assert selected.tool_calls == (
        {
            "tool_name": "set_new_navigation",
            "arguments": {"route_ids": [SECOND_ROUTE_ID]},
        },
    )
