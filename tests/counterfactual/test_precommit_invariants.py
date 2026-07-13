from __future__ import annotations

from typing import Any

from track_1_agent_under_test.car_guard.capability import (
    FeasibilityProof,
    LiveBinding,
    LiveToolRegistry,
)
from track_1_agent_under_test.car_guard.domain import (
    AmbiguitySlot,
    Candidate,
    DecisionProposal,
    Evidence,
    EvidenceNeed,
    EvidenceSourceKind,
    EvidenceStatus,
    EvidenceStore,
    GateOutcome,
    OfficialToolCall,
)
from track_1_agent_under_test.car_guard.policy import (
    GateContext,
    PolicyCompiler,
    PreCommitGate,
    SetOrderEntry,
    provenance_scope_key,
    stable_set_order,
)
from track_1_agent_under_test.car_guard.runtime.budget import RuntimeBudget


def tool(name: str, *, state_changing: bool = False) -> dict[str, Any]:
    properties = {"on": {"type": "boolean"}} if state_changing else {}
    required = ["on"] if state_changing else []
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def proposal() -> DecisionProposal:
    return DecisionProposal.model_validate(
        {
            "kind": "tool_set",
            "goal_ids": ["target"],
            "tool_calls": [
                {
                    "tool_name": "set_target",
                    "arguments": {"on": True},
                    "argument_sources": {"on": "user-turn"},
                    "goal_id": "target",
                }
            ],
        }
    )


def proof() -> FeasibilityProof:
    return FeasibilityProof(
        goal_id="target",
        feasible=True,
        outcome=GateOutcome.ALLOW,
        bindings=[
            LiveBinding(
                recipe_id="target",
                semantic_operation="set_target",
                tool_name="set_target",
                arguments={"on": True},
                required_parameters={"on"},
            )
        ],
        checked_operations=["set_target"],
        reasons=["feasible"],
    )


def context(
    evidence: EvidenceStore,
    *,
    tools: list[dict[str, Any]] | None = None,
    ambiguities: tuple[AmbiguitySlot, ...] = (),
) -> GateContext:
    target_call = OfficialToolCall(tool_name="set_target", arguments={"on": True})
    scope = provenance_scope_key("target", target_call.tool_name, target_call.arguments)
    return GateContext(
        live_tools=LiveToolRegistry(tools or [tool("set_target", state_changing=True)]),
        evidence=evidence,
        budget=RuntimeBudget(),
        compiled_policy=PolicyCompiler().compile(""),
        authorized_goal_ids=frozenset({"target"}),
        valid_provenance_ids=frozenset({"user-turn"}),
        explicit_call_argument_values={scope: {"on": True}},
        feasibility_proofs={"target": proof()},
        ambiguities=ambiguities,
        set_order_entries=(
            SetOrderEntry(
                call=OfficialToolCall(
                    tool_name="set_target", arguments={"on": True}
                ),
                goal_id="target",
            ),
        ),
    )


def unrelated_unknown_store() -> EvidenceStore:
    need = EvidenceNeed(
        proposition="unrelated state",
        required_for_goal_id="other-goal",
        acceptable_sources={"system_context"},
    )
    store = EvidenceStore()
    store.register_need(need)
    store.add(
        Evidence(
            proposition="unrelated state",
            value=None,
            status=EvidenceStatus.UNKNOWN,
            source_kind=EvidenceSourceKind.SYSTEM,
            source_turn_id="system-turn",
            confidence=1.0,
        )
    )
    return store


def test_unrelated_unknown_evidence_does_not_change_target_decision() -> None:
    gate = PreCommitGate()
    empty = gate.validate(proposal(), context(EvidenceStore()))
    unrelated = gate.validate(proposal(), context(unrelated_unknown_store()))
    assert empty == unrelated
    assert unrelated.outcome is GateOutcome.ALLOW


def test_unrelated_ambiguity_does_not_change_target_decision() -> None:
    slot = AmbiguitySlot(
        name="recipient",
        goal_id="other-goal",
        candidates=[
            Candidate(candidate_id="one", value="one"),
            Candidate(candidate_id="two", value="two"),
        ],
    )
    gate = PreCommitGate()
    baseline = gate.validate(proposal(), context(EvidenceStore()))
    transformed = gate.validate(
        proposal(), context(EvidenceStore(), ambiguities=(slot,))
    )
    assert baseline == transformed


def test_irrelevant_live_tool_addition_and_order_do_not_change_decision() -> None:
    relevant = tool("set_target", state_changing=True)
    unrelated = tool("get_unrelated")
    gate = PreCommitGate()
    expected = gate.validate(
        proposal(), context(EvidenceStore(), tools=[relevant])
    )
    added_before = gate.validate(
        proposal(), context(EvidenceStore(), tools=[unrelated, relevant])
    )
    added_after = gate.validate(
        proposal(), context(EvidenceStore(), tools=[relevant, unrelated])
    )
    assert expected == added_before == added_after


def test_set_order_is_invariant_to_entry_input_order() -> None:
    prerequisite = SetOrderEntry(
        call=OfficialToolCall(tool_name="set_policy", arguments={"on": False}),
        policy_dependency_rank=0,
        user_mention_rank=2,
        goal_topological_rank=2,
        recipe_rank=2,
    )
    first_mentioned = SetOrderEntry(
        call=OfficialToolCall(tool_name="set_first", arguments={"on": True}),
        user_mention_rank=0,
        goal_topological_rank=1,
        recipe_rank=1,
    )
    second_mentioned = SetOrderEntry(
        call=OfficialToolCall(tool_name="set_second", arguments={"on": True}),
        user_mention_rank=1,
        goal_topological_rank=0,
        recipe_rank=0,
    )
    entries = (second_mentioned, prerequisite, first_mentioned)
    reversed_entries = tuple(reversed(entries))
    expected = (
        prerequisite.call,
        first_mentioned.call,
        second_mentioned.call,
    )
    assert stable_set_order(entries) == expected
    assert stable_set_order(reversed_entries) == expected


def test_required_unknown_evidence_blocks_target_set() -> None:
    need = EvidenceNeed(
        proposition="target state",
        required_for_goal_id="target",
        acceptable_sources={"system_context"},
    )
    store = EvidenceStore()
    store.register_need(need)
    store.add(
        Evidence(
            proposition="target state",
            value=None,
            status=EvidenceStatus.UNKNOWN,
            source_kind=EvidenceSourceKind.SYSTEM,
            source_turn_id="system-turn",
            confidence=1.0,
        )
    )
    decision = PreCommitGate().validate(proposal(), context(store))
    assert decision.outcome is GateOutcome.UNAVAILABLE_EVIDENCE
    assert decision.normalized_calls == []
