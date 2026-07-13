from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from a2a.helpers.proto_helpers import new_data_part, new_text_part
from google.protobuf.json_format import MessageToDict

from track_1_agent_under_test.car_guard.a2a.adapter import A2AAdapter
from track_1_agent_under_test.car_guard.a2a.result_matching import (
    PendingToolCall,
    match_tool_results,
)
from track_1_agent_under_test.car_guard.a2a.session_store import (
    DisplayedPOICandidate,
    ExecutionBundle,
    SessionClosedError,
    SessionLifecycle,
    SessionStore,
)
from track_1_agent_under_test.car_guard.domain import (
    CommitDecision,
    GateOutcome,
    OfficialToolCall,
)
from track_1_agent_under_test.car_guard.observability.turn_metrics import (
    TurnMetricsAccumulator,
)
from track_1_agent_under_test.car_guard.runtime.budget import RuntimeBudget


@dataclass
class FakeMessage:
    parts: list
    message_id: str = "message-1"


def test_default_runtime_budget_reserves_the_fiftieth_evaluator_step() -> None:
    assert RuntimeBudget().soft_max_steps == 49
    assert SessionStore().get_or_create("default-budget").budget.soft_max_steps == 49


def test_initial_message_and_tools_round_trip() -> None:
    message = FakeMessage(
        parts=[
            new_text_part("System: policy text\n\nUser: open the window"),
            new_data_part(
                {"tools": [{"type": "function", "function": {"name": "get_windows"}}]}
            ),
        ]
    )

    event = A2AAdapter().parse(message)

    assert event.system_policy == "policy text"
    assert event.user_text == "open the window"
    assert event.live_tools[0]["function"]["name"] == "get_windows"


def test_adapter_preserves_context_task_metadata_and_lifecycle_signals() -> None:
    message = A2AAdapter.to_message(
        A2AAdapter.text(
            "finish",
            metadata={"source": "message", "lifecycle": {"terminal": False}},
        ),
        context_id="message-context",
    )
    task = SimpleNamespace(
        id="task-1",
        context_id="task-context",
        status=SimpleNamespace(state=3),
        metadata={"task_marker": "preserved"},
    )

    event = A2AAdapter().parse(
        message,
        task=task,
        request_metadata={"request_marker": "preserved"},
    )

    assert event.context_id == "message-context"
    assert event.task_id == "task-1"
    assert event.metadata["source"] == "message"
    assert event.task_metadata == {"task_marker": "preserved"}
    assert event.request_metadata == {"request_marker": "preserved"}
    assert event.task_state == "completed"
    assert event.terminal
    assert not event.cancel_requested
    assert event.should_close_session


def test_adapter_parses_cancel_from_metadata_and_explicit_context_wins() -> None:
    message = FakeMessage(parts=[new_text_part("cancel")])

    event = A2AAdapter().parse(
        message,
        context_id="request-context",
        task_id="request-task",
        request_metadata={"cancelRequested": True},
    )

    assert event.context_id == "request-context"
    assert event.task_id == "request-task"
    assert event.cancel_requested
    assert event.terminal


def test_metadata_task_state_overrides_unspecified_task_state() -> None:
    task = SimpleNamespace(
        id="task-1",
        context_id="ctx",
        status=SimpleNamespace(state=0),
        metadata={"taskState": "TASK_STATE_CANCELED"},
    )

    event = A2AAdapter().parse(FakeMessage(parts=[]), task=task)

    assert event.task_state == "canceled"
    assert event.cancel_requested
    assert event.terminal


def test_outbound_message_keeps_metadata_at_message_scope() -> None:
    outbound = A2AAdapter.text(
        "Done.",
        terminal=True,
        metadata={"turn_metrics": {"num_llm_calls": 2}},
    )

    message = A2AAdapter.to_message(
        outbound,
        context_id="ctx-1",
        task_id="task-1",
    )
    serialized = MessageToDict(message)

    assert serialized["contextId"] == "ctx-1"
    assert serialized["taskId"] == "task-1"
    assert serialized["parts"] == [{"text": "Done."}]
    assert serialized["metadata"]["terminal"] is True
    assert serialized["metadata"]["turn_metrics"]["num_llm_calls"] == 2


def test_outbound_message_accepts_explicit_stable_message_id() -> None:
    message = A2AAdapter.to_message(
        A2AAdapter.calls([{"tool_name": "set_fan", "arguments": {"level": 2}}]),
        message_id="stable-agent-response",
        context_id="ctx-1",
        task_id="task-1",
    )

    assert message.message_id == "stable-agent-response"
    with pytest.raises(ValueError, match="message_id cannot be blank"):
        A2AAdapter.to_message(A2AAdapter.text("Safe response."), message_id="  ")


def test_non_object_data_list_items_are_reported_as_malformed() -> None:
    valid_tool = {"type": "function", "function": {"name": "get_status"}}
    valid_result = {"toolName": "get_status", "content": "ok"}
    message = FakeMessage(
        parts=[
            new_data_part(
                {
                    "tools": [valid_tool, "bad-tool", None],
                    "tool_results": [valid_result, 42],
                }
            )
        ]
    )

    event = A2AAdapter().parse(message)

    assert event.live_tools == (valid_tool,)
    assert event.tool_results == (valid_result,)
    assert event.malformed_parts == (
        "part[0]:tools[1]-not-object",
        "part[0]:tools[2]-not-object",
        "part[0]:results[1]-not-object",
    )


def test_tool_results_prefer_id_then_name_order() -> None:
    pending = [
        PendingToolCall("a", "get_status", {}, 0),
        PendingToolCall("b", "get_status", {"zone": "rear"}, 1),
    ]
    results = [
        {"toolCallId": "b", "toolName": "get_status", "content": "rear"},
        {"toolName": "get_status", "content": "front"},
    ]

    matched = match_tool_results(results, pending)

    assert matched[0].pending_call == pending[1]
    assert matched[0].matched_by == "call_id"
    assert matched[1].pending_call == pending[0]
    assert matched[1].matched_by == "tool_name_order"


def test_evaluator_generated_id_is_bound_then_deduplicated() -> None:
    pending = [PendingToolCall("local-a", "get_status", {}, 0)]
    seen: set[str] = set()
    external: dict[str, str] = {}

    first = match_tool_results(
        [{"toolCallId": "opaque-1", "toolName": "get_status", "content": "ok"}],
        pending,
        seen_result_ids=seen,
        external_call_ids=external,
    )[0]
    replay = match_tool_results(
        [{"toolCallId": "opaque-1", "toolName": "get_status", "content": "ok"}],
        pending,
        seen_result_ids=seen,
        external_call_ids=external,
    )[0]

    assert first.pending_call == pending[0]
    assert first.is_actionable
    assert first.matched_by == "tool_name_order_bound_external_id"
    assert external == {"opaque-1": "local-a"}
    assert replay.pending_call == pending[0]
    assert replay.matched_by == "call_id"
    assert replay.duplicate
    assert not replay.is_actionable


def test_evaluator_reused_id_rebinds_after_the_prior_call_completed() -> None:
    seen: set[str] = set()
    external: dict[str, str] = {}
    first_pending = [PendingToolCall("local-a", "get_status", {}, 0)]
    second_pending = [PendingToolCall("local-b", "get_status", {}, 0)]

    first = match_tool_results(
        [{"toolCallId": "opaque-1", "toolName": "get_status", "content": "one"}],
        first_pending,
        seen_result_ids=seen,
        external_call_ids=external,
    )[0]
    second = match_tool_results(
        [{"toolCallId": "opaque-1", "toolName": "get_status", "content": "two"}],
        second_pending,
        seen_result_ids=seen,
        external_call_ids=external,
    )[0]
    replay = match_tool_results(
        [{"toolCallId": "opaque-1", "toolName": "get_status", "content": "two"}],
        second_pending,
        seen_result_ids=seen,
        external_call_ids=external,
    )[0]

    assert first.pending_call == first_pending[0]
    assert second.pending_call == second_pending[0]
    assert second.matched_by == "tool_name_order_rebound_external_id"
    assert second.is_actionable
    assert external == {"opaque-1": "local-b"}
    assert replay.pending_call == second_pending[0]
    assert replay.duplicate
    assert not replay.is_actionable


def test_result_id_and_tool_name_conflict_fails_closed() -> None:
    pending = [PendingToolCall("local-a", "get_status", {}, 0)]
    seen: set[str] = set()

    conflict = match_tool_results(
        [
            {
                "toolCallId": "local-a",
                "toolName": "get_other_status",
                "content": "wrong",
            }
        ],
        pending,
        seen_result_ids=seen,
    )[0]
    replay = match_tool_results(
        [{"toolCallId": "local-a", "toolName": "get_status", "content": "ok"}],
        pending,
        seen_result_ids=seen,
    )[0]

    assert conflict.pending_call is None
    assert conflict.tool_name_conflict
    assert conflict.matched_by == "call_id_tool_name_mismatch"
    assert not conflict.is_actionable
    assert replay.pending_call == pending[0]
    assert replay.duplicate


def test_pending_result_call_ids_must_be_unique() -> None:
    pending = [
        PendingToolCall("same", "get_first", {}, 0),
        PendingToolCall("same", "get_second", {}, 1),
    ]

    with pytest.raises(ValueError, match="IDs must be unique"):
        match_tool_results([], pending)


def test_outbound_enforces_xor_and_duplicate_name_rule() -> None:
    adapter = A2AAdapter()
    text = adapter.text("  Please   choose one. ")
    assert text.text == "Please choose one."

    try:
        adapter.calls(
            [
                {"tool_name": "get_status", "arguments": {}},
                {"tool_name": "get_status", "arguments": {"zone": "rear"}},
            ]
        )
    except ValueError as exc:
        assert "duplicate tool names" in str(exc)
    else:
        raise AssertionError("duplicate names should be rejected")


def test_adapter_serializes_only_allowed_commit_decision() -> None:
    adapter = A2AAdapter()
    allowed = CommitDecision(outcome=GateOutcome.ALLOW, user_text="Ready.")

    serialized = adapter.serialize(
        allowed,
        terminal=True,
        metadata={"turn_metrics": {"num_llm_calls": 1}},
    )
    assert serialized.text == "Ready."
    assert serialized.terminal
    assert serialized.metadata["turn_metrics"]["num_llm_calls"] == 1
    try:
        adapter.serialize(CommitDecision(outcome=GateOutcome.NEED_READ))
    except ValueError as exc:
        assert "allowed" in str(exc)
    else:
        raise AssertionError("blocked decisions must be routed before serialization")


def test_session_survives_text_and_expires_by_ttl() -> None:
    now = [0.0]
    store = SessionStore(ttl_seconds=10, max_sessions=2, clock=lambda: now[0])
    session = store.get_or_create("ctx")
    session.append("assistant", "Which option?")

    now[0] = 9.0
    assert store.get("ctx") is session
    now[0] = 20.0
    assert store.get("ctx") is None


def test_session_lru_and_cancel_are_isolated() -> None:
    store = SessionStore(max_sessions=2)
    first = store.get_or_create("first")
    store.get_or_create("second")
    assert store.get("first") is first
    store.get_or_create("third")

    assert store.get("second") is None
    assert store.get("first") is first
    store.cancel("first")
    assert store.get("first") is None
    assert store.get_tombstone("first").lifecycle is SessionLifecycle.CANCELLED


def test_terminal_session_removes_state_and_replays_from_tombstone_until_ttl() -> None:
    now = [0.0]
    store = SessionStore(ttl_seconds=10, clock=lambda: now[0])
    session = store.get_or_create("phone")
    outbound = A2AAdapter.text("Calling now.", terminal=True)
    session.accept_message("message-1")
    session.cache_outbound("message-1", outbound)

    tombstone = store.mark_terminal("phone")

    assert store.get("phone") is None
    assert session.terminal
    assert session.closed
    assert tombstone.terminal
    assert store.replay_for("phone", "message-1") is outbound
    with pytest.raises(SessionClosedError) as exc_info:
        store.get_or_create("phone")
    assert exc_info.value.tombstone is tombstone
    now[0] = 11.0
    assert store.get_tombstone("phone") is None
    assert store.get_or_create("phone") is not session


def test_cancel_tombstone_does_not_replay_stale_action() -> None:
    store = SessionStore()
    session = store.get_or_create("ctx")
    stale = A2AAdapter.calls([{"tool_name": "set_old", "arguments": {}}])
    cancelled = A2AAdapter.text("Cancelled.", terminal=True)
    session.accept_message("old-message")
    session.cache_outbound("old-message", stale)

    tombstone = store.cancel("ctx", message_id="cancel-message", outbound=cancelled)

    assert tombstone.cancelled
    assert store.replay_for("ctx", "old-message") is None
    assert store.replay_for("ctx", "cancel-message") is cancelled
    assert store.reopen("ctx") is not session
    assert store.get_tombstone("ctx") is None


def test_terminal_tombstone_never_replays_tool_calls() -> None:
    store = SessionStore()
    session = store.get_or_create("phone")
    phone_call = A2AAdapter.calls(
        [
            {
                "tool_name": "call_phone_by_number",
                "arguments": {"phone_number": "+1-555-0100"},
            }
        ],
        terminal=True,
    )
    session.accept_message("message-1")
    session.cache_outbound("message-1", phone_call)

    tombstone = store.mark_terminal("phone")

    assert "message-1" in tombstone.seen_message_ids
    assert tombstone.replay_for("message-1") is None
    assert tombstone.last_outbound is None


def test_active_session_cannot_be_reopened_accidentally() -> None:
    store = SessionStore()
    session = store.get_or_create("ctx")

    with pytest.raises(ValueError, match="active session"):
        store.reopen("ctx")

    assert store.get("ctx") is session


def test_duplicate_message_replays_cached_outbound_only() -> None:
    session = SessionStore().get_or_create("ctx")
    outbound = A2AAdapter.text("Which route?")

    assert session.accept_message("message-1")
    session.cache_outbound("message-1", outbound)
    assert not session.accept_message("message-1")
    assert session.replay_for("message-1") is outbound


def test_session_lock_helper_makes_accept_and_replay_atomic() -> None:
    store = SessionStore()
    outbound = A2AAdapter.text("One response.")

    with store.locked("ctx") as session:
        accepted, replay = session.accept_or_replay("message-1")
        assert accepted
        assert replay is None
        session.cache_outbound("message-1", outbound)

    with store.locked("ctx") as session:
        accepted, replay = session.accept_or_replay("message-1")
        assert not accepted
        assert replay is outbound
        with pytest.raises(ValueError, match="different outbound"):
            session.cache_outbound("message-1", A2AAdapter.text("Changed."))


def test_execution_bundle_advances_serially_with_monotonic_call_ids() -> None:
    session = SessionStore().get_or_create("ctx")
    session.execution_bundle = ExecutionBundle(
        goal_ids=("goal-1",),
        calls=(
            OfficialToolCall(tool_name="set_first", arguments={}),
            OfficialToolCall(tool_name="set_second", arguments={}),
        ),
        policy_operations=("prerequisite", None),
    )

    assert session.execution_bundle.current_call.tool_name == "set_first"
    session.execution_bundle.advance()
    assert session.execution_bundle.current_call.tool_name == "set_second"
    assert session.next_call_id() == "agent-call-1"
    assert session.next_call_id() == "agent-call-2"


def test_relative_climate_temperature_delta_is_session_scoped() -> None:
    store = SessionStore()
    first = store.get_or_create("first")
    second = store.get_or_create("second")

    first.relative_climate_temperature_deltas_by_goal["temperature"] = -4

    assert store.get("first") is first
    assert first.relative_climate_temperature_deltas_by_goal == {"temperature": -4}
    assert second.relative_climate_temperature_deltas_by_goal == {}


def test_resolved_location_binding_is_normalized_provenanced_and_versioned() -> None:
    session = SessionStore().get_or_create("location")

    binding = session.remember_resolved_location(
        literal_name="  MUNICH  ",
        location_id="loc_mun_9995",
        source_call_id="agent-call-1",
        source_evidence_id="evidence-location-1",
        state_version=3,
    )

    assert binding.normalized_name == "munich"
    assert binding.source_call_id == "agent-call-1"
    assert binding.source_evidence_id == "evidence-location-1"
    assert session.resolved_location_ids_by_name == {"munich": binding}
    assert session.resolved_location_binding("Munich", state_version=3) is binding
    assert session.resolved_location_binding("Munich", state_version=4) is None


def test_resolved_location_binding_rejects_poi_and_same_version_conflicts() -> None:
    session = SessionStore().get_or_create("location")

    with pytest.raises(ValueError, match=r"loc_\*"):
        session.remember_resolved_location(
            literal_name="Munich",
            location_id="poi_res_123",
            source_call_id="agent-call-1",
            source_evidence_id="evidence-location-1",
            state_version=0,
        )

    session.remember_resolved_location(
        literal_name="Munich",
        location_id="loc_mun_9995",
        source_call_id="agent-call-1",
        source_evidence_id="evidence-location-1",
        state_version=0,
    )
    with pytest.raises(ValueError, match="conflicting IDs"):
        session.remember_resolved_location(
            literal_name="munich",
            location_id="loc_other_123",
            source_call_id="agent-call-2",
            source_evidence_id="evidence-location-2",
            state_version=0,
        )
    assert session.resolved_location_ids_by_name == {}
    assert session.poisoned_resolved_location_bindings == {("munich", 0)}
    assert session.resolved_location_binding("Munich", state_version=0) is None

    with pytest.raises(ValueError, match="poisoned"):
        session.remember_resolved_location(
            literal_name="Munich",
            location_id="loc_mun_9995",
            source_call_id="agent-call-3",
            source_evidence_id="evidence-location-3",
            state_version=0,
        )

    replacement = session.remember_resolved_location(
        literal_name="Munich",
        location_id="loc_mun_9995",
        source_call_id="agent-call-4",
        source_evidence_id="evidence-location-4",
        state_version=1,
    )
    assert session.poisoned_resolved_location_bindings == set()
    assert session.resolved_location_binding("Munich", state_version=1) is replacement


def test_displayed_poi_candidate_opening_hours_are_backward_compatible() -> None:
    candidate = DisplayedPOICandidate(
        normalized_name="meson el segoviano",
        display_name="Meson El Segoviano",
        poi_id="poi_res_651583",
        phone_number="+49 503 3108973",
        location_id="loc_mad_732100",
        source_call_id="agent-call-1",
        source_poi_evidence_id="evidence-poi-1",
        source_location_evidence_id="evidence-location-1",
        displayed_evidence_id="evidence-displayed-1",
        state_version=0,
    )

    assert candidate.opening_hours is None
    enriched = replace(candidate, opening_hours="09:00h - 21:00h")
    assert enriched.opening_hours == "09:00h - 21:00h"

    session = SessionStore().get_or_create("displayed-poi")
    session.remember_displayed_destination_pois((enriched,))
    assert session.displayed_destination_pois == (enriched,)


@pytest.mark.parametrize(
    "opening_hours",
    [
        "",
        "   ",
        " 09:00h - 21:00h",
        "09:00h - 21:00h ",
        "09:00h\n21:00h",
        "09:00h\x0021:00h",
        "09:00h\u202821:00h",
        "x" * 121,
    ],
)
def test_displayed_poi_candidate_rejects_unsafe_opening_hours(
    opening_hours: str,
) -> None:
    with pytest.raises(ValueError, match="opening hours"):
        DisplayedPOICandidate(
            normalized_name="meson el segoviano",
            display_name="Meson El Segoviano",
            poi_id="poi_res_651583",
            phone_number="+49 503 3108973",
            location_id="loc_mad_732100",
            source_call_id="agent-call-1",
            source_poi_evidence_id="evidence-poi-1",
            source_location_evidence_id="evidence-location-1",
            displayed_evidence_id="evidence-displayed-1",
            state_version=0,
            opening_hours=opening_hours,
        )


def test_budget_prevents_repeated_successful_set_and_bounds_gets() -> None:
    budget = RuntimeBudget(max_identical_gets=2)
    assert budget.allow_call("get", {}, state_changing=False)
    budget.record_outbound("get", {}, state_changing=False)
    budget.record_outbound("get", {}, state_changing=False)
    assert not budget.allow_call("get", {}, state_changing=False)

    assert budget.allow_call("set", {"on": True}, state_changing=True)
    budget.record_outbound("set", {"on": True}, state_changing=True)
    assert not budget.allow_call("set", {"on": True}, state_changing=True)
    budget.record_success("set", {"on": True}, state_changing=True)
    assert not budget.allow_call("set", {"on": True}, state_changing=True)


def test_metrics_accumulate_and_reset_without_touching_session() -> None:
    metrics = TurnMetricsAccumulator()
    metrics.add(prompt_tokens=10, completion_tokens=4, elapsed_ms=20, model="one")
    metrics.add(prompt_tokens=3, thinking_tokens=2, elapsed_ms=40, model="two")

    snapshot = metrics.snapshot(reset=True)

    assert snapshot["num_llm_calls"] == 2
    assert snapshot["prompt_tokens"] == 13
    assert snapshot["avg_llm_call_time_ms"] == 30.0
    assert metrics.llm_calls == 0
