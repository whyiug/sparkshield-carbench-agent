"""Deterministic matching between outbound calls and environment results."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PendingToolCall:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    outbound_index: int
    goal_ids: tuple[str, ...] = ()
    evidence_needs: tuple[str, ...] = ()
    state_changing: bool = False
    terminal_after_success: bool = False
    policy_operation: str | None = None
    bundle_index: int | None = None
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class MatchedToolResult:
    pending_call: PendingToolCall | None
    tool_name: str
    content: Any
    provided_call_id: str | None
    matched_by: str
    duplicate: bool = False
    tool_name_conflict: bool = False

    @property
    def is_actionable(self) -> bool:
        return (
            self.pending_call is not None
            and not self.duplicate
            and not self.tool_name_conflict
        )


def _value(item: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in item:
            return item[name]
    return None


def match_tool_results(
    results: list[dict[str, Any]],
    pending_calls: list[PendingToolCall],
    *,
    seen_result_ids: set[str] | None = None,
    external_call_ids: dict[str, str] | None = None,
) -> list[MatchedToolResult]:
    """Match known IDs first and bind new evaluator IDs by name/order.

    CAR-bench emits an evaluator-generated opaque call ID in results even though
    that ID is absent from the outbound A2A payload. The first result therefore
    establishes an external-to-internal mapping; later deliveries can use it.
    """

    seen = seen_result_ids if seen_result_ids is not None else set()
    external = external_call_ids if external_call_ids is not None else {}
    by_id = {call.call_id: call for call in pending_calls}
    if len(by_id) != len(pending_calls):
        raise ValueError("pending tool call IDs must be unique")
    unmatched_ids = {call.call_id for call in pending_calls}
    by_name: dict[str, deque[PendingToolCall]] = defaultdict(deque)
    for pending_call in sorted(pending_calls, key=lambda value: value.outbound_index):
        by_name[pending_call.tool_name].append(pending_call)

    matched: list[MatchedToolResult] = []
    deferred: list[dict[str, Any]] = []

    for result in results:
        call_id = _value(result, "tool_call_id", "toolCallId", "call_id", "callId")
        if call_id:
            call_id = str(call_id)
            internal_id = external.get(call_id, call_id)
            matched_call: PendingToolCall | None = by_id.get(internal_id)
            provided_name = str(_value(result, "tool_name", "toolName") or "")
            tool_name = provided_name or (
                matched_call.tool_name if matched_call else ""
            )
            seen_before = call_id in seen
            reusable_external_id = (
                seen_before and call_id in external and matched_call is None
            )
            duplicate = seen_before and not reusable_external_id
            matched_by = "call_id"
            name_conflict = bool(
                matched_call is not None
                and provided_name
                and provided_name != matched_call.tool_name
            )
            if name_conflict:
                matched_call = None
                matched_by = "call_id_tool_name_mismatch"
            if matched_call is None and not duplicate:
                if not name_conflict:
                    queue = by_name.get(tool_name, deque())
                    while queue and queue[0].call_id not in unmatched_ids:
                        queue.popleft()
                    matched_call = queue.popleft() if queue else None
                    if matched_call is not None:
                        external[call_id] = matched_call.call_id
                        matched_by = (
                            "tool_name_order_rebound_external_id"
                            if reusable_external_id
                            else "tool_name_order_bound_external_id"
                        )
            if matched_call is not None:
                unmatched_ids.discard(matched_call.call_id)
            seen.add(call_id)
            matched.append(
                MatchedToolResult(
                    pending_call=matched_call,
                    tool_name=tool_name,
                    content=_value(result, "content", "result", "value"),
                    provided_call_id=call_id,
                    matched_by=(
                        matched_by
                        if matched_call is not None or name_conflict
                        else "unmatched_call_id"
                    ),
                    duplicate=duplicate,
                    tool_name_conflict=name_conflict,
                )
            )
        else:
            deferred.append(result)

    for result in deferred:
        tool_name = str(_value(result, "tool_name", "toolName") or "")
        queue = by_name.get(tool_name, deque())
        while queue and queue[0].call_id not in unmatched_ids:
            queue.popleft()
        deferred_call: PendingToolCall | None = queue.popleft() if queue else None
        if deferred_call is not None:
            unmatched_ids.discard(deferred_call.call_id)
        matched.append(
            MatchedToolResult(
                pending_call=deferred_call,
                tool_name=tool_name,
                content=_value(result, "content", "result", "value"),
                provided_call_id=None,
                matched_by=(
                    "tool_name_order" if deferred_call else "unmatched_tool_name"
                ),
            )
        )

    return matched
