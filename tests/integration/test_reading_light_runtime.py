from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import DecisionProposal
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
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


POSITIONS = [
    "ALL",
    "DRIVER",
    "PASSENGER",
    "DRIVER_REAR",
    "PASSENGER_REAR",
    "RIGHT_REAR",
    "LEFT_REAR",
]
READING_SET_TOOL = tool(
    "set_reading_light",
    {
        "position": {"type": "string", "enum": POSITIONS},
        "on": {"type": "boolean"},
    },
    ["position", "on"],
)
READING_STATE_TOOL = tool("get_reading_lights_status")
OCCUPANCY_TOOL = tool("get_seats_occupancy")
READING_TOOLS = (READING_SET_TOOL, READING_STATE_TOOL, OCCUPANCY_TOOL)


class NoModelClient:
    def generate(self, **kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("the occupancy workflow must not call a model")


class SimpleReadingClient:
    def __init__(self, *, position: str = "DRIVER", enabled: bool = True) -> None:
        self.position = position
        self.enabled = enabled
        self.intent = IntentDraft.model_validate(
            {
                "language": "en",
                "intent_kind": "action",
                "call_for_action": True,
                "goals": [
                    {
                        "semantic_operation": "set_reading_light",
                        "desired_outcome": {
                            "position": position,
                            "enabled": enabled,
                        },
                    }
                ],
                "explicit_slots": {"position": position, "enabled": enabled},
            }
        )

    def generate(self, *, messages, response_model, critic=False):
        del critic
        if response_model is IntentDraft:
            return SimpleNamespace(value=self.intent)
        if response_model is DecisionProposal:
            payload = json.loads(messages[-1]["content"])
            goal = payload["semantic_goals"]["goals"][0]
            return SimpleNamespace(
                value=DecisionProposal.model_validate(
                    {
                        "kind": "tool_set",
                        "goal_ids": [goal["goal_id"]],
                        "tool_calls": [
                            {
                                "tool_name": "set_reading_light",
                                "arguments": {
                                    "position": self.position,
                                    "on": self.enabled,
                                },
                                "argument_sources": {
                                    "position": "model-output",
                                    "on": "model-output",
                                },
                                "goal_id": goal["goal_id"],
                            }
                        ],
                    }
                )
            )
        raise AssertionError(f"unexpected response model: {response_model}")


def runtime(client: Any) -> CARGuardOrchestrator:
    return CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False),
        client_factory=lambda session: client,
    )


def user_event(
    context_id: str,
    text: str,
    *,
    tools: tuple[dict[str, Any], ...] = READING_TOOLS,
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=f"{context_id}-user",
        system_policy="Follow the current safety policy.",
        user_text=text,
        live_tools=tools,
    )


def results_event(
    context_id: str,
    suffix: str,
    *results: tuple[str, Any],
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=f"{context_id}-{suffix}",
        tool_results=tuple(
            {"toolName": tool_name, "content": content}
            for tool_name, content in results
        ),
    )


def light_state(
    *,
    driver: bool = True,
    passenger: bool = True,
    driver_rear: bool = False,
    passenger_rear: bool = False,
) -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "result": {
            "reading_light_driver": driver,
            "reading_light_passenger": passenger,
            "reading_light_driver_rear": driver_rear,
            "reading_light_passenger_rear": passenger_rear,
        },
    }


def test_simple_reading_light_requires_matching_post_set_readback() -> None:
    context_id = "simple-reading"
    agent = runtime(SimpleReadingClient())
    action = agent.handle_event(
        user_event(context_id, "Turn on the driver reading light.")
    )
    assert action.tool_calls == (
        {
            "tool_name": "set_reading_light",
            "arguments": {"position": "DRIVER", "on": True},
        },
    )

    readback = agent.handle_event(
        results_event(
            context_id,
            "set",
            (
                "set_reading_light",
                {"status": "SUCCESS", "result": {"position": "DRIVER"}},
            ),
        )
    )
    assert readback.tool_calls == (
        {"tool_name": "get_reading_lights_status", "arguments": {}},
    )

    completed = agent.handle_event(
        results_event(
            context_id,
            "readback",
            ("get_reading_lights_status", light_state(driver=True)),
        )
    )
    assert completed.tool_calls == ()
    assert completed.text is not None
    assert "turned on" in completed.text.casefold()


def test_readback_mismatch_aborts_without_claiming_completion() -> None:
    context_id = "reading-mismatch"
    agent = runtime(SimpleReadingClient())
    action = agent.handle_event(
        user_event(context_id, "Turn on the driver reading light.")
    )
    assert action.tool_calls
    readback = agent.handle_event(
        results_event(
            context_id,
            "set",
            (
                "set_reading_light",
                {"status": "SUCCESS", "result": {"position": "DRIVER"}},
            ),
        )
    )
    assert readback.tool_calls[0]["tool_name"] == "get_reading_lights_status"

    blocked = agent.handle_event(
        results_event(
            context_id,
            "readback",
            ("get_reading_lights_status", light_state(driver=False)),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "turned on" not in blocked.text.casefold()


def test_missing_readback_tool_blocks_before_simple_set() -> None:
    agent = runtime(SimpleReadingClient())
    blocked = agent.handle_event(
        user_event(
            "missing-readback",
            "Turn on the driver reading light.",
            tools=(READING_SET_TOOL,),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None


def test_occupancy_workflow_derives_only_required_deltas_and_reads_each_set_back() -> (
    None
):
    context_id = "occupancy-reading"
    agent = runtime(NoModelClient())
    reads = agent.handle_event(
        user_event(
            context_id,
            "Adjust the reading lights for occupied seats and turn them off for unoccupied seats.",
        )
    )
    assert reads.tool_calls == (
        {"tool_name": "get_seats_occupancy", "arguments": {}},
        {"tool_name": "get_reading_lights_status", "arguments": {}},
    )

    first_set = agent.handle_event(
        results_event(
            context_id,
            "initial-state",
            (
                "get_seats_occupancy",
                {
                    "status": "SUCCESS",
                    "result": {
                        "seats_occupied": {
                            "driver": True,
                            "passenger": False,
                            "driver_rear": False,
                            "passenger_rear": True,
                        }
                    },
                },
            ),
            ("get_reading_lights_status", light_state()),
        )
    )
    assert first_set.tool_calls == (
        {
            "tool_name": "set_reading_light",
            "arguments": {"position": "PASSENGER_REAR", "on": True},
        },
    )

    first_readback = agent.handle_event(
        results_event(
            context_id,
            "first-set",
            (
                "set_reading_light",
                {"status": "SUCCESS", "result": {"position": "PASSENGER_REAR"}},
            ),
        )
    )
    assert first_readback.tool_calls == (
        {"tool_name": "get_reading_lights_status", "arguments": {}},
    )
    second_set = agent.handle_event(
        results_event(
            context_id,
            "first-readback",
            (
                "get_reading_lights_status",
                light_state(passenger=True, passenger_rear=True),
            ),
        )
    )
    assert second_set.tool_calls == (
        {
            "tool_name": "set_reading_light",
            "arguments": {"position": "PASSENGER", "on": False},
        },
    )

    second_readback = agent.handle_event(
        results_event(
            context_id,
            "second-set",
            (
                "set_reading_light",
                {"status": "SUCCESS", "result": {"position": "PASSENGER"}},
            ),
        )
    )
    assert second_readback.tool_calls[0]["tool_name"] == "get_reading_lights_status"
    completed = agent.handle_event(
        results_event(
            context_id,
            "second-readback",
            (
                "get_reading_lights_status",
                light_state(passenger=False, passenger_rear=True),
            ),
        )
    )
    assert completed.tool_calls == ()
    assert completed.text is not None


def test_partial_occupancy_result_never_reaches_set() -> None:
    context_id = "partial-occupancy"
    agent = runtime(NoModelClient())
    reads = agent.handle_event(
        user_event(
            context_id,
            "Optimize the reading lights based on which seats are occupied.",
        )
    )
    assert reads.tool_calls
    blocked = agent.handle_event(
        results_event(
            context_id,
            "state",
            (
                "get_seats_occupancy",
                {
                    "status": "SUCCESS",
                    "result": {"seats_occupied": {"driver": True}},
                },
            ),
            ("get_reading_lights_status", light_state()),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None


def test_missing_position_schema_blocks_occupancy_reads_and_sets() -> None:
    malformed_set = tool(
        "set_reading_light", {"on": {"type": "boolean"}}, ["on"]
    )
    agent = runtime(NoModelClient())
    blocked = agent.handle_event(
        user_event(
            "missing-position",
            "Adjust reading lights based on occupied and unoccupied seats.",
            tools=(malformed_set, READING_STATE_TOOL, OCCUPANCY_TOOL),
        )
    )
    assert blocked.tool_calls == ()
    assert blocked.text is not None
