from __future__ import annotations

from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.llm.client import LLMFailure
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
    _broad_headlights_request,
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


EXTERIOR_READ = tool("get_exterior_lights_status")
LOW_SET = tool("set_head_lights_low_beams", {"on": {"type": "boolean"}}, ["on"])
HIGH_SET = tool("set_head_lights_high_beams", {"on": {"type": "boolean"}}, ["on"])
FOG_SET = tool("set_fog_lights", {"on": {"type": "boolean"}}, ["on"])
ALL_TOOLS = (EXTERIOR_READ, LOW_SET, HIGH_SET, FOG_SET)


class UnavailableModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs: Any) -> Any:
        del kwargs
        self.calls += 1
        raise LLMFailure("the deterministic headlight workflow must not call the model")


def runtime(model: UnavailableModel | None = None) -> CARGuardOrchestrator:
    selected = model or UnavailableModel()
    return CARGuardOrchestrator(
        AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=20),
        client_factory=lambda session: selected,
    )


def user_event(
    context_id: str,
    message_id: str,
    text: str,
    tools: tuple[dict[str, Any], ...] = ALL_TOOLS,
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        system_policy="POL-014:",
        user_text=text,
        live_tools=tools,
    )


def result_event(
    context_id: str,
    message_id: str,
    tool_name: str,
    result: dict[str, Any],
) -> InboundEvent:
    return InboundEvent(
        context_id=context_id,
        message_id=message_id,
        tool_results=(
            {
                "toolName": tool_name,
                "content": {"status": "SUCCESS", "result": result},
            },
        ),
    )


def begin(
    agent: CARGuardOrchestrator,
    context_id: str,
    tools: tuple[dict[str, Any], ...] = ALL_TOOLS,
):
    outbound = agent.handle_event(
        user_event(context_id, f"{context_id}-user", "Turn on the headlights.", tools)
    )
    assert outbound.tool_calls == (
        {"tool_name": "get_exterior_lights_status", "arguments": {}},
    )
    return outbound


@pytest.mark.parametrize(
    "text",
    (
        "Turn on the headlights.",
        "Please switch my head lights on.",
        "Could you enable the headlights for me?",
    ),
)
def test_broad_headlight_matcher_accepts_only_one_current_request(text: str) -> None:
    assert _broad_headlights_request(text)


@pytest.mark.parametrize(
    "text",
    (
        "Turn on the high beams.",
        "Turn on the low beams.",
        "Turn on the fog lights.",
        "Turn on the lights.",
        "Do not turn on the headlights.",
        "If it gets dark, turn on the headlights.",
        'The driver said "turn on the headlights."',
        "Turn on the headlights and open the window.",
        "Turn on the headlights. Then call someone.",
    ),
)
def test_broad_headlight_matcher_rejects_specific_unsafe_or_composed_scope(
    text: str,
) -> None:
    assert not _broad_headlights_request(text)


def test_unique_unlit_low_beam_is_selected_from_complete_snapshot() -> None:
    agent = runtime()
    begin(agent, "select-low")

    action = agent.handle_event(
        result_event(
            "select-low",
            "select-low-state",
            "get_exterior_lights_status",
            {
                "fog_lights": False,
                "head_lights_low_beams": False,
                "head_lights_high_beams": True,
            },
        )
    )

    assert action.tool_calls == (
        {"tool_name": "set_head_lights_low_beams", "arguments": {"on": True}},
    )
    session = agent.sessions.get("select-low")
    assert session is not None and session.intent is not None
    assert session.intent.goals[0].semantic_operation == "set_low_beams"
    selections = session.evidence.for_proposition(
        f"derived_selection:{session.intent.goals[0].goal_id}:headlight_beam"
    )
    assert len(selections) == 1
    assert selections[0].value == "low"
    assert selections[0].derivation == (
        "unlit_headlight_sibling_from_complete_snapshot_v1"
    )


def test_both_beams_off_requires_an_exact_user_choice() -> None:
    agent = runtime()
    begin(agent, "choose-beam")

    clarification = agent.handle_event(
        result_event(
            "choose-beam",
            "choose-beam-state",
            "get_exterior_lights_status",
            {
                "fog_lights": False,
                "head_lights_low_beams": False,
                "head_lights_high_beams": False,
            },
        )
    )
    assert clarification.tool_calls == ()
    assert clarification.text is not None
    assert "high" in clarification.text.casefold()
    assert "low" in clarification.text.casefold()

    still_ambiguous = agent.handle_event(
        user_event("choose-beam", "choose-beam-ambiguous", "Both beams.")
    )
    assert still_ambiguous.tool_calls == ()
    assert still_ambiguous.text is not None

    action = agent.handle_event(
        user_event("choose-beam", "choose-beam-low", "Low beams, please.")
    )
    assert action.tool_calls == (
        {"tool_name": "set_head_lights_low_beams", "arguments": {"on": True}},
    )


def test_direct_low_beam_command_resolves_the_pending_choice() -> None:
    agent = runtime()
    begin(agent, "direct-choice")
    agent.handle_event(
        result_event(
            "direct-choice",
            "direct-choice-state",
            "get_exterior_lights_status",
            {
                "fog_lights": False,
                "head_lights_low_beams": False,
                "head_lights_high_beams": False,
            },
        )
    )

    action = agent.handle_event(
        user_event("direct-choice", "direct-choice-low", "Turn on the low beams.")
    )

    assert action.tool_calls == (
        {"tool_name": "set_head_lights_low_beams", "arguments": {"on": True}},
    )


def test_independent_direct_action_supersedes_the_custom_choice() -> None:
    model = UnavailableModel()
    agent = runtime(model)
    begin(agent, "new-scope")
    agent.handle_event(
        result_event(
            "new-scope",
            "new-scope-state",
            "get_exterior_lights_status",
            {
                "fog_lights": False,
                "head_lights_low_beams": False,
                "head_lights_high_beams": False,
            },
        )
    )

    routed = agent.handle_event(
        user_event("new-scope", "new-scope-window", "Open the driver window instead.")
    )

    assert routed.tool_calls == ()
    assert routed.text is not None
    assert "choose exactly high beams or low beams" not in routed.text.casefold()
    assert model.calls == 1
    session = agent.sessions.get("new-scope")
    assert session is not None
    assert session.intent is None or all(
        goal.semantic_operation != "set_broad_headlights"
        for goal in session.intent.goals
    )


@pytest.mark.parametrize(
    "unsafe",
    (
        "Do not open the driver window instead.",
        "If it gets warm, open the driver window instead.",
        'The driver said "open the driver window instead."',
        "Open the driver window and turn on the low beams.",
    ),
)
def test_unsafe_or_compound_scope_does_not_escape_pending_choice(unsafe: str) -> None:
    model = UnavailableModel()
    agent = runtime(model)
    begin(agent, f"unsafe-new-scope-{len(unsafe)}")
    agent.handle_event(
        result_event(
            f"unsafe-new-scope-{len(unsafe)}",
            f"unsafe-new-scope-state-{len(unsafe)}",
            "get_exterior_lights_status",
            {
                "fog_lights": False,
                "head_lights_low_beams": False,
                "head_lights_high_beams": False,
            },
        )
    )

    blocked = agent.handle_event(
        user_event(
            f"unsafe-new-scope-{len(unsafe)}",
            f"unsafe-new-scope-user-{len(unsafe)}",
            unsafe,
        )
    )

    assert blocked.tool_calls == ()
    assert blocked.text is not None
    assert "high" in blocked.text.casefold() and "low" in blocked.text.casefold()
    assert model.calls == 0


def test_both_beams_already_on_does_not_repeat_a_set() -> None:
    agent = runtime()
    begin(agent, "already-on")

    completed = agent.handle_event(
        result_event(
            "already-on",
            "already-on-state",
            "get_exterior_lights_status",
            {
                "fog_lights": False,
                "head_lights_low_beams": True,
                "head_lights_high_beams": True,
            },
        )
    )

    assert completed.tool_calls == ()
    assert completed.text is not None and "already on" in completed.text
    session = agent.sessions.get("already-on")
    assert session is not None
    assert session.budget.attempted_sets == set()


@pytest.mark.parametrize(
    "snapshot",
    (
        {
            "fog_lights": False,
            "head_lights_low_beams": False,
        },
        {
            "fog_lights": False,
            "head_lights_low_beams": False,
            "head_lights_high_beams": "true",
        },
        {
            "fog_lights": False,
            "head_lights_low_beams": False,
            "head_lights_high_beams": True,
            "unexpected": False,
        },
    ),
)
def test_malformed_or_incomplete_snapshot_fails_closed(
    snapshot: dict[str, Any],
) -> None:
    agent = runtime()
    begin(agent, "malformed")

    blocked = agent.handle_event(
        result_event(
            "malformed",
            "malformed-state",
            "get_exterior_lights_status",
            snapshot,
        )
    )

    assert blocked.tool_calls == ()
    session = agent.sessions.get("malformed")
    assert session is not None
    assert session.budget.attempted_sets == set()


def test_missing_selected_set_control_reports_unavailable_after_read() -> None:
    agent = runtime()
    begin(agent, "missing-high", tools=(EXTERIOR_READ, LOW_SET))

    blocked = agent.handle_event(
        result_event(
            "missing-high",
            "missing-high-state",
            "get_exterior_lights_status",
            {
                "fog_lights": False,
                "head_lights_low_beams": True,
                "head_lights_high_beams": False,
            },
        )
    )

    assert blocked.tool_calls == ()
    assert (
        blocked.text is not None
        and "not currently available" in blocked.text.casefold()
    )
    session = agent.sessions.get("missing-high")
    assert session is not None
    assert session.budget.attempted_sets == set()


def test_derived_high_beam_uses_the_standard_exact_confirmation_bundle() -> None:
    agent = runtime()
    begin(agent, "confirm-high")

    confirmation = agent.handle_event(
        result_event(
            "confirm-high",
            "confirm-high-state",
            "get_exterior_lights_status",
            {
                "fog_lights": True,
                "head_lights_low_beams": True,
                "head_lights_high_beams": False,
            },
        )
    )
    assert confirmation.tool_calls == ()
    assert confirmation.text is not None
    assert "shall i go ahead" in confirmation.text.casefold()
    session = agent.sessions.get("confirm-high")
    assert session is not None and len(session.confirmation_latch.pending) == 1

    mismatched = agent.handle_event(
        user_event(
            "confirm-high",
            "confirm-high-mismatch",
            "Yes, turn on the low beams instead.",
        )
    )
    assert mismatched.tool_calls == ()
    assert len(session.confirmation_latch.pending) == 1

    first = agent.handle_event(
        user_event("confirm-high", "confirm-high-yes", "Yes, please proceed.")
    )
    assert first.tool_calls == (
        {"tool_name": "set_fog_lights", "arguments": {"on": False}},
    )
    assert session.confirmation_latch.pending == []
    assert len(session.confirmation_latch.active) == 1

    second = agent.handle_event(
        result_event(
            "confirm-high",
            "confirm-high-fog-off",
            "set_fog_lights",
            {"on": False},
        )
    )
    assert second.tool_calls == (
        {"tool_name": "set_head_lights_high_beams", "arguments": {"on": True}},
    )
