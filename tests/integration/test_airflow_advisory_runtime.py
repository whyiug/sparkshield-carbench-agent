from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import DecisionProposal
from track_1_agent_under_test.car_guard.planning.intent_parser import (
    IntentDraft,
)
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
    _airflow_advisory_request,
)


POLICY = "Follow the current vehicle safety policy."


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


CLIMATE_TOOL = tool("get_climate_settings")
FAN_TOOL = tool(
    "set_fan_speed",
    {"level": {"type": "integer", "minimum": 0, "maximum": 5}},
    ["level"],
)
CIRCULATION_TOOL = tool(
    "set_air_circulation",
    {"mode": {"type": "string", "enum": ["AUTO", "FRESH_AIR", "RECIRCULATION"]}},
    ["mode"],
)
WINDOW_TOOL = tool(
    "open_close_window",
    {
        "window": {
            "type": "string",
            "enum": ["DRIVER", "PASSENGER", "DRIVER_REAR", "PASSENGER_REAR"],
        },
        "percentage": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    ["window", "percentage"],
)
FULL_TOOLS = (CLIMATE_TOOL, FAN_TOOL, CIRCULATION_TOOL, WINDOW_TOOL)
CLIMATE_RESULT = {
    "status": "SUCCESS",
    "result": {
        "fan_speed": 0,
        "fan_airflow_direction": "HEAD_FEET",
        "air_conditioning": False,
        "air_circulation": "FRESH_AIR",
        "window_front_defrost": False,
        "window_rear_defrost": False,
    },
}


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
        del critic
        if response_model is IntentDraft:
            self.intent_calls += 1
            return SimpleNamespace(value=self.intent)
        if response_model is DecisionProposal:
            self.action_calls += 1
            payload = json.loads(messages[-1]["content"])
            goal = next(
                goal
                for goal in payload["semantic_goals"]["goals"]
                if goal["semantic_operation"] == "set_fan_speed"
            )
            return SimpleNamespace(
                value=DecisionProposal.model_validate(
                    {
                        "kind": "tool_set",
                        "goal_ids": [goal["goal_id"]],
                        "tool_calls": [
                            {
                                "tool_name": "set_fan_speed",
                                "arguments": {"level": 2},
                                "argument_sources": {"level": "model-output"},
                                "goal_id": goal["goal_id"],
                            }
                        ],
                    }
                )
            )
        raise AssertionError(f"unexpected response model: {response_model}")


def runtime_for(client: EmptyClient) -> CARGuardOrchestrator:
    return CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=24),
        client_factory=lambda session: client,
    )


def user_event(
    *, context_id: str, message_id: str, text: str, tools: tuple[dict[str, Any], ...]
) -> InboundEvent:
    return InboundEvent(
        message_id=message_id,
        context_id=context_id,
        system_policy=POLICY,
        user_text=text,
        live_tools=tools,
    )


@pytest.mark.parametrize(
    "utterance",
    [
        "The air in the cabin feels stagnant. What can be done about that?",
        "The car air feels stuffy. What options do I have?",
        "The vehicle air is stale. What could be done about it?",
        "The airflow in the car is not circulating. What options do we have?",
        "What are the options to improve the air circulation?",
        "How can I improve the airflow in the cabin?",
        "What are some ways to improve air circulation in the vehicle?",
    ],
)
def test_airflow_advice_reads_state_without_authorizing_a_write(
    utterance: str,
) -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = f"airflow-{abs(hash(utterance))}"

    read = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id=f"{context_id}-user",
            text=utterance,
            tools=FULL_TOOLS,
        )
    )

    assert read.tool_calls == (
        {"tool_name": "get_climate_settings", "arguments": {}},
    )
    assert client.intent_calls == 0
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert session.authorized_action_goal_ids == set()
    assert session.budget.attempted_sets == set()

    answer = runtime.handle_event(
        InboundEvent(
            message_id=f"{context_id}-result",
            context_id=context_id,
            tool_results=(
                {"toolName": "get_climate_settings", "content": CLIMATE_RESULT},
            ),
        )
    )

    assert answer.tool_calls == ()
    assert answer.text is not None
    lowered = answer.text.casefold()
    assert "off at level 0" in lowered
    assert "fresh air" in lowered
    assert "fan speed" in lowered
    assert "air circulation mode" in lowered
    assert "windows" in lowered
    assert session.authorized_action_goal_ids == set()
    assert session.budget.attempted_sets == set()


def test_explicit_followup_fan_level_uses_the_existing_exact_action() -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = "airflow-explicit-followup"
    runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="airflow-user",
            text="The cabin air feels stale. What can be done?",
            tools=FULL_TOOLS,
        )
    )
    runtime.handle_event(
        InboundEvent(
            message_id="airflow-result",
            context_id=context_id,
            tool_results=(
                {"toolName": "get_climate_settings", "content": CLIMATE_RESULT},
            ),
        )
    )

    action = runtime.handle_event(
        InboundEvent(
            message_id="airflow-level",
            context_id=context_id,
            user_text="Please set the fan to level 2.",
        )
    )

    assert action.tool_calls == (
        {"tool_name": "set_fan_speed", "arguments": {"level": 2}},
    )
    assert client.intent_calls == 0


@pytest.mark.parametrize(
    "non_request",
    [
        'He said, "The cabin air feels stagnant. What can be done?"',
        "The cabin air does not feel stagnant.",
        "If the cabin air feels stale, what can be done?",
        "Stagnant traffic near the airport is frustrating.",
        "The cabin air feels stale. Open the passenger window.",
        "What are the options to improve airflow and start navigation?",
        "The cabin air feels stale. What can be done? Who is in the car?",
        "The air in the cabin feels stagnant. What can be done, and who is in the car?",
        "The cabin air feels stale. What can be done, and what is the battery level?",
        (
            "The air in the cabin feels stagnant. What can be done? "
            "Also, what is my battery level?"
        ),
        "What are the options to improve airflow? Play some music.",
        "The air in the cabin feels stagnant. What can be done, and play music?",
        "The car air feels stuffy. What options do I have? Tell me a joke.",
        "The air in the cabin feels stagnant. What can be done, and tell me a joke?",
        "How can I improve air circulation in the cabin, and check my calendar?",
        "The air in the cabin feels stagnant. What can be done, and check the calendar?",
    ],
)
def test_airflow_advice_rejects_unbounded_or_state_changing_text(
    non_request: str,
) -> None:
    assert not _airflow_advisory_request(non_request)

    client = EmptyClient()
    runtime = runtime_for(client)
    outbound = runtime.handle_event(
        user_event(
            context_id=f"airflow-negative-{abs(hash(non_request))}",
            message_id="negative-user",
            text=non_request,
            tools=FULL_TOOLS,
        )
    )

    assert all(
        call["tool_name"] != "get_climate_settings" for call in outbound.tool_calls
    )


def test_airflow_advice_missing_read_control_stops_before_the_llm() -> None:
    client = EmptyClient()
    runtime = runtime_for(client)

    blocked = runtime.handle_event(
        user_event(
            context_id="airflow-missing-read",
            message_id="airflow-missing-user",
            text="The cabin air feels stuffy. What can be done?",
            tools=(FAN_TOOL,),
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None and "unavailable" in blocked.text.casefold()
    assert client.intent_calls == 0
    session = runtime.sessions.get("airflow-missing-read")
    assert session is not None and session.budget.attempted_sets == set()


def test_airflow_advice_only_offers_controls_present_in_the_live_schema() -> None:
    client = EmptyClient()
    runtime = runtime_for(client)
    context_id = "airflow-live-schema-options"

    read = runtime.handle_event(
        user_event(
            context_id=context_id,
            message_id="airflow-live-schema-user",
            text="The cabin air feels stale. What can be done?",
            tools=(CLIMATE_TOOL, CIRCULATION_TOOL),
        )
    )
    assert read.tool_calls == (
        {"tool_name": "get_climate_settings", "arguments": {}},
    )

    answer = runtime.handle_event(
        InboundEvent(
            message_id="airflow-live-schema-result",
            context_id=context_id,
            tool_results=(
                {"toolName": "get_climate_settings", "content": CLIMATE_RESULT},
            ),
        )
    )

    assert answer.text is not None
    lowered = answer.text.casefold()
    assert "i can change the air circulation mode" in lowered
    assert "adjust the fan speed" not in lowered
    assert "window" not in lowered
