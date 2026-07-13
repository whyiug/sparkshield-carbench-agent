"""Deterministic twelve-stage validation before any outbound action."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import Field, ValidationError

from ..capability.feasibility import FeasibilityProof
from ..capability.live_tool_registry import (
    LiveToolRegistry,
    ToolArgumentValidationError,
)
from ..domain import (
    AmbiguitySlot,
    CommitDecision,
    DecisionProposal,
    EvidenceNeed,
    EvidenceCardinality,
    EvidenceStatus,
    EvidenceStore,
    EvidenceSourceKind,
    GateOutcome,
    GateReason,
    GoalDAG,
    OfficialToolCall,
    ProposalKind,
)
from ..domain.types import DomainModel, NonEmptyStr
from ..runtime.budget import RuntimeBudget, call_fingerprint
from ..response.tts_guard import TTSGuard, TTSViolation
from .confirmation_latch import ConfirmationLatch
from .compiler import CompiledPolicy
from .rules import (
    ActionAuthorization,
    ConfirmationRequirement,
    ExecutionMode,
    PolicyArgumentDecision,
    PolicyConflict,
    PolicyDecision,
    PolicyRequest,
    SemanticPolicyCall,
)


class SetOrderEntry(DomainModel):
    """Ordering facts fixed before a state-changing proposal is emitted."""

    call: OfficialToolCall
    goal_id: NonEmptyStr | None = None
    policy_dependency_rank: int | None = Field(default=None, ge=0)
    user_mention_rank: int = Field(default=0, ge=0)
    goal_topological_rank: int = Field(default=0, ge=0)
    recipe_rank: int = Field(default=0, ge=0)


def stable_set_order(entries: Sequence[SetOrderEntry]) -> tuple[OfficialToolCall, ...]:
    """Apply policy, user mention, DAG, then recipe ordering with stable ties."""

    indexed = tuple(enumerate(entries))
    ordered = sorted(
        indexed,
        key=lambda item: (
            item[1].policy_dependency_rank is None,
            (
                item[1].policy_dependency_rank
                if item[1].policy_dependency_rank is not None
                else 0
            ),
            item[1].user_mention_rank,
            item[1].goal_topological_rank,
            item[1].recipe_rank,
            call_fingerprint(item[1].call.tool_name, item[1].call.arguments),
            item[0],
        ),
    )
    return tuple(entry.call for _, entry in ordered)


@dataclass(slots=True)
class GateContext:
    """All current-session facts that the gate is permitted to inspect."""

    live_tools: LiveToolRegistry
    evidence: EvidenceStore
    budget: RuntimeBudget
    compiled_policy: CompiledPolicy
    # Additional restrictions may be supplied, but can never replace the
    # action-bound decision evaluated from ``compiled_policy`` below.
    policy_decision: PolicyDecision = field(default_factory=PolicyDecision)
    policy_decision_digest: str | None = None
    confirmation_latch: ConfirmationLatch = field(default_factory=ConfirmationLatch)
    current_user_turn: int = 0
    authorized_goal_ids: frozenset[str] = frozenset()
    valid_provenance_ids: frozenset[str] = frozenset()
    explicit_call_argument_values: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    call_argument_evidence_ids: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )
    ambiguities: tuple[AmbiguitySlot, ...] = ()
    required_evidence_needs: tuple[EvidenceNeed, ...] = ()
    feasibility_proofs: Mapping[str, FeasibilityProof] = field(default_factory=dict)
    policy_call_bindings: Mapping[str, OfficialToolCall] = field(default_factory=dict)
    # Values must be created with ``policy_operation_completion_key``.  A bare
    # semantic operation is intentionally not sufficient because it can be
    # replayed with different arguments, evidence, or goals.
    completed_policy_operations: frozenset[str] = frozenset()
    # Retained for source compatibility only.  Text authorization is based on
    # deterministic validation below, never on caller-asserted format codes.
    satisfied_format_codes: frozenset[str] = frozenset()
    # ``bundle_goal_ids`` is the full frozen execution/confirmation scope;
    # proposal/call goal IDs remain the scope of the one action being emitted.
    bundle_goal_ids: tuple[str, ...] = ()
    execution_bundle: tuple[OfficialToolCall, ...] = ()
    confirmation_bundle: tuple[OfficialToolCall, ...] = ()
    set_order_entries: tuple[SetOrderEntry, ...] = ()
    goal_dag: GoalDAG | None = None
    terminal_after_success: bool = False
    policy_state: Mapping[str, Any] = field(default_factory=dict)
    policy_facts: Mapping[str, Any] = field(default_factory=dict)
    policy_authorization: ActionAuthorization = ActionAuthorization.NONE
    semantic_operations: Mapping[str, str] = field(default_factory=dict)
    tool_descriptions: Mapping[str, str] = field(default_factory=dict)
    substitution_authorized_by_user: bool = False
    response_language: str | None = None

    def __post_init__(self) -> None:
        if self.current_user_turn < 0:
            raise ValueError("current_user_turn must be non-negative")
        if len(self.bundle_goal_ids) != len(set(self.bundle_goal_ids)):
            raise ValueError("bundle_goal_ids must be unique")


class PreCommitGate:
    """Validate proposals in the exact order required by the runtime design."""

    def validate(
        self,
        proposal: DecisionProposal | Mapping[str, Any],
        context: GateContext,
    ) -> CommitDecision:
        # 1. Output structure and tool/text XOR.
        try:
            candidate = (
                proposal
                if isinstance(proposal, DecisionProposal)
                else DecisionProposal.model_validate(proposal)
            )
        except (ValidationError, TypeError, ValueError) as exc:
            return _blocked(
                GateOutcome.INVALID_PROPOSAL,
                "structure.invalid",
                "The proposal does not contain exactly one valid output mode.",
                details={"validation_error": str(exc)},
            )

        if not candidate.tool_calls:
            context = replace(
                context,
                policy_decision=_effective_policy_decision(candidate, (), context),
            )
            return self._validate_text(candidate, context)

        # 2. Reactive goal authorization for every SET.
        action_goal_ids = _action_goal_ids(candidate)
        call_goal_ids = {
            call.goal_id for call in candidate.tool_calls if call.goal_id is not None
        }
        if not call_goal_ids.issubset(candidate.goal_ids):
            return _blocked(
                GateOutcome.INVALID_PROPOSAL,
                "authorization.goal_mismatch",
                "A proposed call is bound to a different goal.",
                goal_ids=candidate.goal_ids,
            )
        if candidate.kind is ProposalKind.TOOL_SET:
            if not action_goal_ids:
                return _blocked(
                    GateOutcome.POLICY_CONFLICT,
                    "authorization.action_goal_required",
                    "A state-changing call must identify its one action goal.",
                )
            if not set(action_goal_ids).issubset(_bundle_goal_ids(candidate, context)):
                return _blocked(
                    GateOutcome.INVALID_PROPOSAL,
                    "authorization.bundle_goal_mismatch",
                    "The action goal is outside the frozen bundle goal scope.",
                    goal_ids=action_goal_ids,
                )
            authorized_by_request = set(action_goal_ids).issubset(
                context.authorized_goal_ids
            )
            authorized_by_confirmation = self._authorized_confirmation(
                candidate, context
            )
            if not authorized_by_request and not authorized_by_confirmation:
                return _blocked(
                    GateOutcome.POLICY_CONFLICT,
                    "authorization.reactive_only",
                    "The state-changing action has no user authorization.",
                    goal_ids=action_goal_ids,
                )

        # 3 and 4. Current live callable and exact live JSON Schema.
        normalized: list[OfficialToolCall] = []
        for call in candidate.tool_calls:
            if not context.live_tools.has_tool(call.tool_name):
                return _blocked(
                    GateOutcome.UNSUPPORTED_CAPABILITY,
                    "capability.not_live",
                    "A goal-required action is not callable in the current live tools.",
                    goal_ids=candidate.goal_ids,
                    call_ids=_call_ids(call.call_id),
                )
            try:
                arguments = context.live_tools.validate_arguments(
                    call.tool_name, call.arguments
                )
            except ToolArgumentValidationError as exc:
                return _blocked(
                    GateOutcome.UNSUPPORTED_PARAMETER,
                    f"schema.{exc.code}",
                    "Proposed arguments do not satisfy the current live schema.",
                    goal_ids=candidate.goal_ids,
                    call_ids=_call_ids(call.call_id),
                    details={"path": exc.path},
                )
            normalized.append(
                OfficialToolCall(
                    tool_name=call.tool_name,
                    arguments=arguments,
                    tool_call_id=call.call_id,
                )
            )

        # Evaluate policy against the normalized candidate. A caller-provided
        # decision is additive and therefore cannot erase an applicable rule.
        context = replace(
            context,
            policy_decision=_effective_policy_decision(
                candidate, tuple(normalized), context
            ),
        )

        # 5. Every SET argument has a goal-bound source whose value is verified.
        if candidate.kind is ProposalKind.TOOL_SET:
            for call in candidate.tool_calls:
                if set(call.argument_sources) != set(call.arguments):
                    return _blocked(
                        GateOutcome.INVALID_PROPOSAL,
                        "provenance.incomplete",
                        "Every state-changing argument requires provenance.",
                        goal_ids=candidate.goal_ids,
                        call_ids=_call_ids(call.call_id),
                    )
                goal_id = call.goal_id
                assert goal_id is not None
                if any(
                    not _provenance_matches(
                        argument=name,
                        value=value,
                        source_id=call.argument_sources[name],
                        goal_id=goal_id,
                        call=OfficialToolCall(
                            tool_name=call.tool_name, arguments=call.arguments
                        ),
                        context=context,
                    )
                    for name, value in call.arguments.items()
                ):
                    return _blocked(
                        GateOutcome.INVALID_PROPOSAL,
                        "provenance.unverifiable",
                        "State-changing argument provenance is not verifiable.",
                        goal_ids=action_goal_ids,
                        call_ids=_call_ids(call.call_id),
                    )

        # 6. Only unresolved required slots for these goals block the action.
        required_parameters = set().union(
            *(
                context.live_tools.required_parameters(call.tool_name)
                for call in candidate.tool_calls
            )
        )
        unresolved = [
            slot
            for slot in context.ambiguities
            if not slot.is_resolved
            and (
                slot.goal_id in action_goal_ids
                or (slot.goal_id is None and slot.name in required_parameters)
            )
        ]
        if unresolved:
            return _blocked(
                GateOutcome.NEED_USER_DISAMBIGUATION,
                "ambiguity.required_slot",
                "A required action parameter still has multiple valid candidates.",
                goal_ids=action_goal_ids,
                details={"slots": [slot.name for slot in unresolved]},
            )

        # 7. Goal-conditioned evidence is required only before a SET.
        if candidate.kind is ProposalKind.TOOL_SET:
            evidence_decision = self._validate_evidence(action_goal_ids, context)
            if evidence_decision is not None:
                return evidence_decision

        # 8. Policy reads, prerequisites, arguments, conflicts, and formats.
        policy_decision = self._validate_policy(candidate, tuple(normalized), context)
        if policy_decision is not None:
            return policy_decision

        # 9. A policy-required confirmation matches the exact action bundle.
        if (
            candidate.kind is ProposalKind.TOOL_SET
            and context.policy_decision.confirmation is not None
            and not self._authorized_confirmation(candidate, context)
        ):
            return _blocked(
                GateOutcome.NEED_CONFIRMATION,
                "confirmation.exact_bundle_required",
                "The exact ordered action bundle requires current user confirmation.",
                goal_ids=action_goal_ids,
            )

        # 10. Every SET goal has an affirmative conditional closure proof.
        if candidate.kind is ProposalKind.TOOL_SET:
            for goal_id in action_goal_ids:
                proof = context.feasibility_proofs.get(goal_id)
                if proof is None:
                    return _blocked(
                        GateOutcome.INVALID_PROPOSAL,
                        "closure.proof_required",
                        "A conditional capability closure proof is required before SET.",
                        goal_ids=[goal_id],
                    )
                if proof.goal_id != goal_id:
                    return _blocked(
                        GateOutcome.INVALID_PROPOSAL,
                        "closure.goal_mismatch",
                        "The capability closure proof belongs to another goal.",
                        goal_ids=[goal_id],
                    )
                if not proof.first_set_allowed:
                    outcome = proof.outcome
                    if outcome is GateOutcome.ALLOW:
                        outcome = GateOutcome.INVALID_PROPOSAL
                    return _blocked(
                        outcome,
                        "closure.not_feasible",
                        "The full triggered action closure is not feasible.",
                        goal_ids=[goal_id],
                    )
                expected = _next_proof_call(proof, context.budget)
                if expected is None and not _proof_contains_call(proof, normalized[0]):
                    return _blocked(
                        GateOutcome.INVALID_PROPOSAL,
                        "closure.next_binding_required",
                        "The proof does not contain a pending state-changing binding.",
                        goal_ids=[goal_id],
                    )
                if expected is not None and not _same_call(normalized[0], expected):
                    return _blocked(
                        GateOutcome.INVALID_PROPOSAL,
                        "closure.next_call_mismatch",
                        "The proposed SET is not the next call proven by the closure.",
                        goal_ids=[goal_id],
                    )
                if context.set_order_entries and not _proof_order_entries_match(
                    proof, goal_id, context.set_order_entries
                ):
                    return _blocked(
                        GateOutcome.INVALID_PROPOSAL,
                        "closure.order_binding_mismatch",
                        "The deterministic SET order does not cover exactly the proof bindings.",
                        goal_ids=[goal_id],
                    )

        # 11. GET independence, SET seriality, and stable side-effect order.
        order_decision = self._validate_order(candidate, tuple(normalized), context)
        if order_decision is not None:
            return order_decision

        # 12. Idempotency and bounded room for the frozen bundle and final text.
        for normalized_call in normalized:
            state_changing = candidate.kind is ProposalKind.TOOL_SET
            if not context.budget.allow_call(
                normalized_call.tool_name,
                normalized_call.arguments,
                state_changing=state_changing,
            ):
                return _blocked(
                    GateOutcome.INVALID_PROPOSAL,
                    "budget.duplicate_or_exhausted_call",
                    "The call is a duplicate or exceeds its retry budget.",
                    goal_ids=action_goal_ids or candidate.goal_ids,
                    call_ids=_call_ids(normalized_call.tool_call_id),
                )
        remaining_calls = (
            _remaining_bundle_calls(context)
            if candidate.kind is ProposalKind.TOOL_SET
            else tuple(normalized)
        )
        if candidate.kind is ProposalKind.TOOL_SET:
            if not any(_same_call(normalized[0], call) for call in remaining_calls):
                return _blocked(
                    GateOutcome.INVALID_PROPOSAL,
                    "budget.action_outside_frozen_bundle",
                    "The proposed action is not pending in the frozen execution bundle.",
                    goal_ids=action_goal_ids,
                )
            remaining_fingerprints = [
                call_fingerprint(call.tool_name, call.arguments)
                for call in remaining_calls
            ]
            if len(remaining_fingerprints) != len(set(remaining_fingerprints)):
                return _blocked(
                    GateOutcome.INVALID_PROPOSAL,
                    "budget.duplicate_bundle_call",
                    "The frozen execution bundle contains a duplicate side effect.",
                    goal_ids=_bundle_goal_ids(candidate, context),
                )
            unavailable_future = next(
                (
                    call
                    for call in remaining_calls
                    if not context.budget.allow_call(
                        call.tool_name,
                        call.arguments,
                        state_changing=True,
                    )
                ),
                None,
            )
            if unavailable_future is not None:
                return _blocked(
                    GateOutcome.INVALID_PROPOSAL,
                    "budget.bundle_call_not_executable",
                    "A remaining frozen-bundle action is duplicate or already attempted.",
                    goal_ids=_bundle_goal_ids(candidate, context),
                )
        # A final natural-language result is mandatory even when the current SET
        # is the last action.  The gate is pure: this is capacity validation, not
        # budget mutation or an optimistic reservation counter.
        reserved_steps = len(remaining_calls) + 1
        if context.budget.steps + reserved_steps > context.budget.soft_max_steps:
            return _blocked(
                GateOutcome.INVALID_PROPOSAL,
                "budget.insufficient_steps",
                "The remaining step budget cannot execute the frozen bundle and final response.",
                goal_ids=action_goal_ids or candidate.goal_ids,
                details={
                    "remaining_bundle_calls": len(remaining_calls),
                    "final_response_steps": 1,
                },
            )

        return CommitDecision(outcome=GateOutcome.ALLOW, normalized_calls=normalized)

    def _validate_text(
        self, proposal: DecisionProposal, context: GateContext
    ) -> CommitDecision:
        assert proposal.user_text is not None
        conflicts = context.policy_decision.conflicts
        if conflicts:
            return _policy_conflict(proposal.goal_ids, conflicts[0].code)
        format_failures = _response_format_failures(
            proposal.user_text,
            context.policy_decision,
            response_language=context.response_language,
        )
        if format_failures:
            return _blocked(
                GateOutcome.POLICY_CONFLICT,
                "policy.format_unsatisfied",
                "The response has not passed every active policy format check.",
                goal_ids=proposal.goal_ids,
                details={"format_codes": sorted(format_failures)},
            )
        return CommitDecision(
            outcome=GateOutcome.ALLOW,
            user_text=TTSGuard().ensure(proposal.user_text),
        )

    def _validate_evidence(
        self,
        action_goal_ids: Sequence[str],
        context: GateContext,
    ) -> CommitDecision | None:
        needs: dict[str, EvidenceNeed] = dict(context.evidence.needs)
        for need in context.required_evidence_needs:
            assert need.need_id is not None
            needs[need.need_id] = need
        relevant = [
            need
            for need in needs.values()
            if need.required_for_goal_id in action_goal_ids
            and need.required_before_set
        ]
        for need in relevant:
            if context.evidence.is_satisfied(need):
                continue
            observations = context.evidence.for_proposition(need.proposition)
            if not observations:
                return _blocked(
                    GateOutcome.NEED_READ,
                    "evidence.read_required",
                    "Required goal evidence has not been observed.",
                    goal_ids=[need.required_for_goal_id],
                )
            statuses = {observation.status for observation in observations}
            unavailable = statuses.intersection(
                {
                    EvidenceStatus.UNKNOWN,
                    EvidenceStatus.ERROR,
                    EvidenceStatus.CONFLICT,
                    EvidenceStatus.STALE,
                }
            )
            conflicting_known_values = (
                need.cardinality is EvidenceCardinality.UNIQUE
                and bool(context.evidence.satisfying_evidence(need))
            )
            return _blocked(
                (
                    GateOutcome.UNAVAILABLE_EVIDENCE
                    if unavailable or conflicting_known_values
                    else GateOutcome.NEED_READ
                ),
                "evidence.insufficient",
                "Observed evidence does not safely satisfy the goal requirement.",
                goal_ids=[need.required_for_goal_id],
            )
        return None

    def _validate_policy(
        self,
        proposal: DecisionProposal,
        normalized: tuple[OfficialToolCall, ...],
        context: GateContext,
    ) -> CommitDecision | None:
        policy = context.policy_decision
        if policy.conflicts:
            return _policy_conflict(proposal.goal_ids, policy.conflicts[0].code)

        pending_reads = [
            call
            for call in policy.required_reads
            if not _policy_operation_completed(call, proposal, context)
        ]
        if pending_reads:
            if proposal.kind is not ProposalKind.TOOL_GET or not any(
                _matches_policy_call(call, normalized, context)
                for call in pending_reads
            ):
                return _blocked(
                    GateOutcome.NEED_READ,
                    "policy.read_required",
                    "A policy-required read must complete before the action.",
                    goal_ids=proposal.goal_ids,
                )

        pending_prerequisites = [
            call
            for call in policy.prerequisite_calls
            if not _policy_operation_completed(call, proposal, context)
        ]
        if pending_prerequisites:
            if context.execution_bundle:
                ranked: list[tuple[int, SemanticPolicyCall]] = []
                for policy_call in pending_prerequisites:
                    bound = _bound_policy_call(policy_call, context)
                    rank = next(
                        (
                            index
                            for index, bundle_call in enumerate(
                                context.execution_bundle
                            )
                            if bound is not None and _same_call(bound, bundle_call)
                        ),
                        None,
                    )
                    if rank is None:
                        return _blocked(
                            GateOutcome.POLICY_CONFLICT,
                            "policy.prerequisite_not_frozen",
                            (
                                "A policy prerequisite is outside the frozen "
                                "execution bundle."
                            ),
                            goal_ids=proposal.goal_ids,
                        )
                    ranked.append((rank, policy_call))
                pending_prerequisites = [
                    policy_call
                    for _, policy_call in sorted(ranked, key=lambda item: item[0])
                ]
            first = pending_prerequisites[0]
            if (
                proposal.kind is not ProposalKind.TOOL_SET
                or len(normalized) != 1
                or not _matches_policy_call(first, normalized, context)
            ):
                return _blocked(
                    GateOutcome.POLICY_CONFLICT,
                    "policy.prerequisite_order",
                    "A policy prerequisite must execute before the requested action.",
                    goal_ids=proposal.goal_ids,
                )

        for decision in policy.argument_decisions:
            invalid = _invalid_policy_argument(decision, normalized)
            if invalid:
                return _blocked(
                    GateOutcome.POLICY_CONFLICT,
                    "policy.argument_constraint",
                    "An action argument violates an active policy constraint.",
                    goal_ids=proposal.goal_ids,
                )

        if policy.execution_mode is ExecutionMode.SERIAL and len(normalized) > 1:
            return _blocked(
                GateOutcome.POLICY_CONFLICT,
                "policy.serial_execution_required",
                "The active policy requires serial execution.",
                goal_ids=proposal.goal_ids,
            )
        return None

    def _validate_order(
        self,
        proposal: DecisionProposal,
        normalized: tuple[OfficialToolCall, ...],
        context: GateContext,
    ) -> CommitDecision | None:
        if proposal.kind is ProposalKind.TOOL_GET and len(proposal.tool_calls) > 1:
            if any(call.depends_on for call in proposal.tool_calls):
                return _blocked(
                    GateOutcome.INVALID_PROPOSAL,
                    "order.parallel_get_dependency",
                    "Only independent reads may execute in parallel.",
                    goal_ids=proposal.goal_ids,
                )
            if context.goal_dag is not None:
                parallel_goal_ids = {
                    call.goal_id
                    for call in proposal.tool_calls
                    if call.goal_id is not None
                }
                known_goal_ids = {goal.goal_id for goal in context.goal_dag.goals}
                if not parallel_goal_ids.issubset(known_goal_ids):
                    return _blocked(
                        GateOutcome.INVALID_PROPOSAL,
                        "order.parallel_get_unknown_goal",
                        "A parallel read references a goal outside the current DAG.",
                        goal_ids=proposal.goal_ids,
                    )
                has_dependency = any(
                    dependency in parallel_goal_ids
                    for goal_id in parallel_goal_ids
                    for dependency in context.goal_dag.get(goal_id).depends_on
                )
                if has_dependency:
                    return _blocked(
                        GateOutcome.INVALID_PROPOSAL,
                        "order.parallel_get_goal_dependency",
                        "Reads for dependent goals cannot execute in parallel.",
                        goal_ids=proposal.goal_ids,
                    )

        if proposal.kind is ProposalKind.TOOL_SET:
            if len(normalized) != 1:
                return _blocked(
                    GateOutcome.INVALID_PROPOSAL,
                    "order.set_must_be_serial",
                    "State-changing calls must execute one at a time.",
                    goal_ids=proposal.goal_ids,
                )
            if not context.set_order_entries:
                return _blocked(
                    GateOutcome.INVALID_PROPOSAL,
                    "order.set_order_required",
                    "Every SET requires a precomputed deterministic order entry.",
                    goal_ids=proposal.goal_ids,
                )
            if context.execution_bundle and not _same_call_sequence(
                stable_set_order(context.set_order_entries),
                context.execution_bundle,
            ):
                return _blocked(
                    GateOutcome.INVALID_PROPOSAL,
                    "order.execution_bundle_mismatch",
                    "The frozen execution bundle differs from deterministic SET order.",
                    goal_ids=proposal.goal_ids,
                )
            completed = context.budget.successful_sets
            expected = next(
                (
                    call
                    for call in stable_set_order(context.set_order_entries)
                    if call_fingerprint(call.tool_name, call.arguments) not in completed
                ),
                None,
            )
            proposed_is_ordered = any(
                _same_call(normalized[0], entry.call)
                for entry in context.set_order_entries
            )
            if (expected is None and not proposed_is_ordered) or (
                expected is not None and not _same_call(normalized[0], expected)
            ):
                return _blocked(
                    GateOutcome.INVALID_PROPOSAL,
                    "order.unstable_set_order",
                    "The proposed SET is not the next deterministic side effect.",
                    goal_ids=proposal.goal_ids,
                )
        return None

    def _authorized_confirmation(
        self, proposal: DecisionProposal, context: GateContext
    ) -> bool:
        return _has_authorized_confirmation(proposal, context)


def _action_goal_ids(proposal: DecisionProposal) -> tuple[str, ...]:
    if proposal.kind is not ProposalKind.TOOL_SET:
        return tuple(proposal.goal_ids)
    if len(proposal.tool_calls) != 1 or proposal.tool_calls[0].goal_id is None:
        return ()
    return (proposal.tool_calls[0].goal_id,)


def _bundle_goal_ids(
    proposal: DecisionProposal, context: GateContext
) -> tuple[str, ...]:
    return context.bundle_goal_ids or tuple(proposal.goal_ids)


def _frozen_execution_bundle(context: GateContext) -> tuple[OfficialToolCall, ...]:
    if context.execution_bundle:
        return context.execution_bundle
    if context.set_order_entries:
        return stable_set_order(context.set_order_entries)
    return context.confirmation_bundle


def _remaining_bundle_calls(context: GateContext) -> tuple[OfficialToolCall, ...]:
    return tuple(
        call
        for call in _frozen_execution_bundle(context)
        if call_fingerprint(call.tool_name, call.arguments)
        not in context.budget.successful_sets
    )


def _confirmation_bundle(
    proposal: DecisionProposal, context: GateContext
) -> tuple[OfficialToolCall, ...]:
    if context.confirmation_bundle:
        frozen = _frozen_execution_bundle(context)
        if frozen and not _same_call_sequence(context.confirmation_bundle, frozen):
            return ()
        return context.confirmation_bundle
    if context.execution_bundle:
        return context.execution_bundle
    if context.set_order_entries:
        bundle_goals = set(_bundle_goal_ids(proposal, context))
        relevant = tuple(
            entry
            for entry in context.set_order_entries
            if entry.goal_id is None or entry.goal_id in bundle_goals
        )
        return stable_set_order(relevant)
    translated: list[OfficialToolCall] = []
    confirmation = context.policy_decision.confirmation
    if confirmation is not None:
        for policy_call in confirmation.bundle_operations:
            bound = _bound_policy_call(policy_call, context)
            if bound is None:
                return ()
            translated.append(bound)
    return tuple(translated)


def _has_authorized_confirmation(
    proposal: DecisionProposal, context: GateContext
) -> bool:
    bundle = _confirmation_bundle(proposal, context)
    if not bundle:
        return False
    return (
        context.confirmation_latch.authorized_confirmation(
            goal_ids=list(_bundle_goal_ids(proposal, context)),
            ordered_actions=list(bundle),
            current_user_turn=context.current_user_turn,
        )
        is not None
    )


def _effective_policy_decision(
    proposal: DecisionProposal,
    normalized: tuple[OfficialToolCall, ...],
    context: GateContext,
) -> PolicyDecision:
    decisions: list[PolicyDecision] = []
    digest_actions: tuple[OfficialToolCall, ...] = ()
    authorization = context.policy_authorization
    action_goal_ids = _action_goal_ids(proposal)
    policy_goal_ids = list(action_goal_ids or tuple(proposal.goal_ids))
    if authorization is ActionAuthorization.NONE and set(policy_goal_ids).issubset(
        context.authorized_goal_ids
    ):
        authorization = ActionAuthorization.USER_REQUEST
    elif authorization is ActionAuthorization.NONE and _has_authorized_confirmation(
        proposal, context
    ):
        authorization = ActionAuthorization.USER_CONFIRMATION

    if normalized:
        bundle_goals = set(_bundle_goal_ids(proposal, context))
        relevant_order_entries = tuple(
            entry
            for entry in context.set_order_entries
            if entry.goal_id is None or entry.goal_id in bundle_goals
        )
        ordered_actions = list(
            context.execution_bundle
            or context.confirmation_bundle
            or (
                stable_set_order(relevant_order_entries)
                if proposal.kind is ProposalKind.TOOL_SET and relevant_order_entries
                else normalized
            )
        )
        digest_actions = tuple(ordered_actions)
        policy_actions = list(normalized)
        if proposal.kind is ProposalKind.TOOL_SET and ordered_actions:
            final_action = ordered_actions[-1]
            if not any(_same_call(final_action, call) for call in policy_actions):
                policy_actions.append(final_action)
        for call in policy_actions:
            semantic_operation = context.semantic_operations.get(
                call.tool_call_id or "",
                context.semantic_operations.get(call.tool_name, call.tool_name),
            )
            decisions.append(
                context.compiled_policy.evaluate(
                    PolicyRequest(
                        goal_ids=policy_goal_ids,
                        action=call,
                        semantic_operation=semantic_operation,
                        ordered_actions=ordered_actions,
                        tool_description=context.tool_descriptions.get(call.tool_name),
                        state_changing=proposal.kind is ProposalKind.TOOL_SET,
                        authorization=authorization,
                        state=dict(context.policy_state),
                        facts=dict(context.policy_facts),
                        requested_tool_available=True,
                        requested_parameters_available=True,
                        substitution_authorized_by_user=(
                            context.substitution_authorized_by_user
                        ),
                    )
                )
            )
    else:
        decisions.append(
            context.compiled_policy.evaluate(
                PolicyRequest(
                    goal_ids=policy_goal_ids,
                    state_changing=False,
                    authorization=authorization,
                    state=dict(context.policy_state),
                    facts=dict(context.policy_facts),
                )
            )
        )
    additional = context.policy_decision
    if not _policy_decision_is_empty(additional):
        expected_digest = policy_action_digest(
            proposal,
            context.compiled_policy,
            ordered_actions=digest_actions,
        )
        if context.policy_decision_digest != expected_digest:
            additional = PolicyDecision(
                conflicts=[
                    PolicyConflict(
                        code="precomputed_policy_binding_mismatch",
                        message=(
                            "A precomputed policy decision is not bound to the "
                            "normalized action bundle."
                        ),
                    )
                ]
            )
    decisions.append(additional)
    return _merge_policy_decisions(decisions)


def _merge_policy_decisions(decisions: Sequence[PolicyDecision]) -> PolicyDecision:
    confirmations = [
        decision.confirmation
        for decision in decisions
        if decision.confirmation is not None
    ]
    confirmation: ConfirmationRequirement | None = None
    if confirmations:
        confirmation = ConfirmationRequirement(
            explicit_yes=any(item.explicit_yes for item in confirmations),
            describe_action_and_parameters=any(
                item.describe_action_and_parameters for item in confirmations
            ),
            include_full_bundle=any(item.include_full_bundle for item in confirmations),
            rule_ids=_ordered_unique(
                rule_id for item in confirmations for rule_id in item.rule_ids
            ),
            reason_codes=_ordered_unique(
                reason for item in confirmations for reason in item.reason_codes
            ),
            warnings=_ordered_unique(
                warning for item in confirmations for warning in item.warnings
            ),
            bundle_operations=[
                operation
                for item in confirmations
                for operation in item.bundle_operations
            ],
        )
    return PolicyDecision(
        applied_rule_ids=_ordered_unique(
            rule_id for decision in decisions for rule_id in decision.applied_rule_ids
        ),
        required_reads=[
            item for decision in decisions for item in decision.required_reads
        ],
        prerequisite_calls=[
            item for decision in decisions for item in decision.prerequisite_calls
        ],
        confirmation=confirmation,
        format_decisions=[
            item for decision in decisions for item in decision.format_decisions
        ],
        argument_decisions=[
            item for decision in decisions for item in decision.argument_decisions
        ],
        conflicts=[item for decision in decisions for item in decision.conflicts],
        execution_mode=(
            ExecutionMode.SERIAL
            if any(
                decision.execution_mode is ExecutionMode.SERIAL
                for decision in decisions
            )
            else ExecutionMode.DEFAULT
        ),
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _response_format_failures(
    text: str,
    policy: PolicyDecision,
    *,
    response_language: str | None,
) -> set[str]:
    failures: set[str] = set()
    try:
        normalized = TTSGuard().ensure(text)
    except TTSViolation:
        normalized = text
        failures.add("tts_safe")
        failures.add("tts_no_markdown_or_visual_lists")

    for decision in policy.format_decisions:
        if not _response_format_satisfied(
            normalized,
            decision.code,
            decision.details,
            response_language=response_language,
        ):
            failures.add(decision.code)
    return failures


def _response_format_satisfied(
    text: str,
    code: str,
    details: Mapping[str, Any],
    *,
    response_language: str | None,
) -> bool:
    folded = text.casefold()
    language = (response_language or "").casefold().replace("_", "-")
    has_question = "?" in text or "？" in text
    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in text)
    letter_count = sum(char.isalpha() for char in text)

    if code in {"tts_safe", "tts_no_markdown_or_visual_lists"}:
        try:
            TTSGuard().ensure(text)
        except TTSViolation:
            return False
        return True
    if code == "natural_first_person":
        if language.startswith("zh"):
            return bool(text.strip())
        if language.startswith("de"):
            return bool(
                re.search(r"\b(?:ich|wir|mein(?:e[rmns]?)?)\b", folded)
            ) or folded.startswith("erledigt")
        if not language and cjk_count:
            return True
        return bool(re.search(r"\b(?:i|we|my|our)\b", folded)) or folded.startswith(
            ("done", "sorry")
        )
    if code == "respond_in_user_language":
        if not language:
            return True
        if language.startswith("zh"):
            return cjk_count > 0
        return letter_count == 0 or cjk_count / letter_count < 0.5
    if code == "metric_distance_km_and_m":
        return not _contains_imperial_distance(text)
    if code == "temperature_celsius":
        return not _contains_fahrenheit(text)
    if code == "datetime_24h":
        return not _contains_twelve_hour_time(text)
    if code in {
        "ask_before_any_substitution",
        "ask_which_poi_to_navigate_to",
    }:
        return has_question
    if code == "transparently_report_unavailable":
        return _contains_any(
            folded,
            (
                "can't",
                "cannot",
                "not available",
                "unavailable",
                "couldn't",
                "无法",
                "不能",
                "不可用",
                "nicht verfügbar",
            ),
        )
    if code == "describe_action_and_all_parameters_before_confirmation":
        arguments = details.get("arguments", {})
        return (
            has_question
            and isinstance(arguments, Mapping)
            and all(_text_contains_value(folded, value) for value in arguments.values())
        )
    if code == "state_weather_warning_before_confirmation":
        return has_question and _text_contains_value(folded, details.get("condition"))
    if code == "warn_energy_inefficiency":
        return _contains_any(
            folded,
            ("energy", "inefficient", "efficiency", "能耗", "能源", "energie"),
        )
    if code == "inform_temperature_zone_difference":
        return _text_contains_value(folded, details.get("resulting_difference_celsius"))
    if code == "describe_conflicting_lights_bundle":
        return _contains_any(folded, ("fog", "雾灯", "nebel")) and _contains_any(
            folded, ("high beam", "远光", "fernlicht")
        )
    if code == "disclose_route_toll":
        return _contains_any(folded, ("toll", "收费", "maut"))
    if code == "present_all_poi_names":
        names = details.get("names", [])
        return isinstance(names, list) and all(
            str(name).casefold() in folded for name in names
        )
    if code == "route_alternatives_fastest_and_shortest":
        route_terms = _contains_any(folded, ("fastest", "最快", "schnellste"))
        if not bool(details.get("same_route")):
            route_terms = route_terms and _contains_any(
                folded, ("shortest", "最短", "kürzeste", "kuerzeste")
            )
        return route_terms and has_question
    if code == "select_fastest_route_per_segment":
        return _contains_any(folded, ("fastest", "最快", "schnellste"))
    if code == "announce_fastest_default_and_offer_alternatives":
        return _contains_any(
            folded, ("fastest", "最快", "schnellste")
        ) and _contains_any(
            folded, ("alternative", "option", "其他路线", "备选", "alternative")
        )
    # Unknown policy format codes are fail-closed.  Adding a policy format must
    # therefore add a deterministic validator instead of a caller-side claim.
    return False


def _contains_any(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


def _contains_imperial_distance(text: str) -> bool:
    quantity = (
        r"(?:\b\d+(?:\.\d+)?\s*|"
        r"\b(?:a|an|half|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"few|several|many)\s+)"
    )
    return bool(
        re.search(
            rf"{quantity}(?:miles?|feet|foot|ft)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _contains_fahrenheit(text: str) -> bool:
    return bool(
        re.search(
            r"(?:°\s*f\b|\bfahrenheit\b|\bdegrees?\s+f\b)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _contains_twelve_hour_time(text: str) -> bool:
    return bool(
        re.search(
            r"\b\d{1,2}(?::[0-5]\d)?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)(?![A-Za-z])",
            text,
            flags=re.IGNORECASE,
        )
    )


def _text_contains_value(text: str, value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        word_markers = (
            ("true", "on", "enable", "enabled", "an")
            if value
            else ("false", "off", "disable", "disabled", "aus")
        )
        language_markers = ("开启", "打开") if value else ("关闭", "关掉")
        return bool(
            re.search(
                rf"\b(?:{'|'.join(re.escape(item) for item in word_markers)})\b",
                text,
            )
        ) or _contains_any(text, language_markers)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    rendered = str(value).casefold().replace("_", " ")
    return rendered in text


def _policy_decision_is_empty(decision: PolicyDecision) -> bool:
    return not decision.model_dump(exclude_defaults=True, exclude_none=True)


def policy_action_digest(
    proposal: DecisionProposal,
    compiled_policy: CompiledPolicy,
    *,
    ordered_actions: Sequence[OfficialToolCall] = (),
) -> str:
    calls = (
        [call.to_a2a_payload() for call in ordered_actions]
        if ordered_actions
        else [
            {"tool_name": call.tool_name, "arguments": call.arguments}
            for call in proposal.tool_calls
        ]
    )
    payload = {
        "policy_digest": compiled_policy.source_digest,
        "proposal_kind": proposal.kind.value,
        "goal_ids": proposal.goal_ids,
        "ordered_actions": calls,
        "user_text": proposal.user_text,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bound_policy_call(
    policy_call: SemanticPolicyCall, context: GateContext
) -> OfficialToolCall | None:
    bound = context.policy_call_bindings.get(
        policy_call_binding_key(policy_call.semantic_operation, policy_call.arguments)
    )
    if bound is None:
        bound = context.policy_call_bindings.get(policy_call.semantic_operation)
    if bound is not None:
        return bound
    if context.live_tools.has_tool(policy_call.semantic_operation):
        try:
            arguments = context.live_tools.validate_arguments(
                policy_call.semantic_operation, policy_call.arguments
            )
        except ToolArgumentValidationError:
            return None
        return OfficialToolCall(
            tool_name=policy_call.semantic_operation, arguments=arguments
        )
    return None


def policy_call_binding_key(
    semantic_operation: str, arguments: Mapping[str, Any]
) -> str:
    encoded = json.dumps(
        dict(arguments),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return f"{semantic_operation}:args-{hashlib.sha256(encoded).hexdigest()[:20]}"


def _matches_policy_call(
    policy_call: SemanticPolicyCall,
    normalized: Sequence[OfficialToolCall],
    context: GateContext,
) -> bool:
    bound = _bound_policy_call(policy_call, context)
    return bound is not None and any(_same_call(bound, call) for call in normalized)


def _invalid_policy_argument(
    decision: PolicyArgumentDecision, normalized: Sequence[OfficialToolCall]
) -> bool:
    matching = [call for call in normalized if decision.name in call.arguments]
    if decision.presence_required and not matching:
        return True
    return decision.has_fixed_value and not any(
        call.arguments[decision.name] == decision.fixed_value for call in matching
    )


def _provenance_matches(
    *,
    argument: str,
    value: Any,
    source_id: str,
    goal_id: str,
    call: OfficialToolCall,
    context: GateContext,
) -> bool:
    scope_key = provenance_scope_key(goal_id, call.tool_name, call.arguments)
    explicit = context.explicit_call_argument_values.get(scope_key, {})
    if (
        source_id in context.valid_provenance_ids
        and argument in explicit
        and _strict_value_equal(value, explicit[argument])
    ):
        return True

    for decision in context.policy_decision.argument_decisions:
        accepted_policy_sources = {
            item
            for item in (
                decision.source,
                *(f"policy:{rule_id}" for rule_id in decision.rule_ids),
            )
            if item is not None
        }
        if (
            decision.name == argument
            and decision.has_fixed_value
            and source_id in accepted_policy_sources
            and _strict_value_equal(value, decision.fixed_value)
        ):
            return True

    for requirement in context.policy_decision.prerequisite_calls:
        bound = _bound_policy_call(requirement, context)
        accepted_rule_sources = {
            source
            for rule_id in requirement.rule_ids
            for source in (rule_id, f"policy:{rule_id}")
        }
        if (
            bound is not None
            and _same_call(call, bound)
            and argument in requirement.arguments
            and _strict_value_equal(value, requirement.arguments[argument])
            and source_id in accepted_rule_sources
        ):
            return True

    bound_evidence_id = context.call_argument_evidence_ids.get(scope_key, {}).get(
        argument
    )
    return bound_evidence_id == source_id and _validated_evidence_value(
        source_id, value, context.evidence, seen=set()
    )


def _validated_evidence_value(
    evidence_id: str,
    expected_value: Any,
    store: EvidenceStore,
    *,
    seen: set[str],
) -> bool:
    if evidence_id in seen:
        return False
    observation = store.evidence.get(evidence_id)
    if observation is None or observation.status is not EvidenceStatus.KNOWN:
        return False
    if not _strict_value_equal(observation.value, expected_value):
        return False
    if observation.source_kind is not EvidenceSourceKind.DERIVED:
        return True
    next_seen = {*seen, evidence_id}
    return bool(observation.derived_from) and all(
        _validated_evidence_input(item, store, seen=next_seen)
        for item in observation.derived_from
    )


def _validated_evidence_input(
    evidence_id: str, store: EvidenceStore, *, seen: set[str]
) -> bool:
    if evidence_id in seen:
        return False
    observation = store.evidence.get(evidence_id)
    if observation is None or observation.status is not EvidenceStatus.KNOWN:
        return False
    if observation.source_kind is not EvidenceSourceKind.DERIVED:
        return True
    next_seen = {*seen, evidence_id}
    return bool(observation.derived_from) and all(
        _validated_evidence_input(item, store, seen=next_seen)
        for item in observation.derived_from
    )


def _next_proof_call(
    proof: FeasibilityProof, budget: RuntimeBudget
) -> OfficialToolCall | None:
    for binding in proof.bindings:
        if binding.is_read:
            continue
        if not binding.is_complete:
            return None
        call = OfficialToolCall(
            tool_name=binding.tool_name, arguments=binding.arguments
        )
        if (
            call_fingerprint(call.tool_name, call.arguments)
            not in budget.successful_sets
        ):
            return call
    return None


def _proof_contains_call(proof: FeasibilityProof, call: OfficialToolCall) -> bool:
    return any(
        not binding.is_read
        and binding.is_complete
        and _same_call(
            call,
            OfficialToolCall(tool_name=binding.tool_name, arguments=binding.arguments),
        )
        for binding in proof.bindings
    )


def _proof_order_entries_match(
    proof: FeasibilityProof,
    goal_id: str,
    entries: Sequence[SetOrderEntry],
) -> bool:
    proof_fingerprints = sorted(
        call_fingerprint(binding.tool_name, binding.arguments)
        for binding in proof.bindings
        if not binding.is_read and binding.is_complete
    )
    proof_set = set(proof_fingerprints)
    ordered_fingerprints: list[str] = []
    for entry in entries:
        fingerprint = call_fingerprint(entry.call.tool_name, entry.call.arguments)
        if entry.goal_id == goal_id or (
            entry.goal_id is None and fingerprint in proof_set
        ):
            ordered_fingerprints.append(fingerprint)
    ordered_fingerprints.sort()
    return proof_fingerprints == ordered_fingerprints


def policy_operation_completion_key(
    semantic_operation: str,
    bound_call: OfficialToolCall,
    *,
    evidence_state_version: int,
    bundle_goal_ids: Sequence[str],
) -> str:
    """Bind a completed policy operation to all state-sensitive inputs."""

    if evidence_state_version < 0:
        raise ValueError("evidence_state_version must be non-negative")
    if not semantic_operation.strip() or not bundle_goal_ids:
        raise ValueError("policy completion requires an operation and goal bundle")
    goal_payload = json.dumps(
        list(bundle_goal_ids),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    goal_digest = hashlib.sha256(goal_payload).hexdigest()[:20]
    arguments_fingerprint = call_fingerprint(bound_call.tool_name, bound_call.arguments)
    return (
        f"policy-complete:{semantic_operation}:{arguments_fingerprint}:"
        f"state-{evidence_state_version}:goals-{goal_digest}"
    )


def _policy_operation_completed(
    policy_call: SemanticPolicyCall,
    proposal: DecisionProposal,
    context: GateContext,
) -> bool:
    bound = _bound_policy_call(policy_call, context)
    if bound is None:
        return False
    bundle_goals = _bundle_goal_ids(proposal, context)
    if not bundle_goals:
        return False
    completion_key = policy_operation_completion_key(
        policy_call.semantic_operation,
        bound,
        evidence_state_version=context.evidence.current_state_version,
        bundle_goal_ids=bundle_goals,
    )
    return completion_key in context.completed_policy_operations


def provenance_scope_key(
    goal_id: str, tool_name: str, arguments: Mapping[str, Any]
) -> str:
    return f"{goal_id}:{call_fingerprint(tool_name, dict(arguments))}"


def _strict_value_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _same_call(left: OfficialToolCall, right: OfficialToolCall) -> bool:
    return left.to_a2a_payload() == right.to_a2a_payload()


def _same_call_sequence(
    left: Sequence[OfficialToolCall], right: Sequence[OfficialToolCall]
) -> bool:
    return len(left) == len(right) and all(
        _same_call(left_call, right_call)
        for left_call, right_call in zip(left, right, strict=True)
    )


def _call_ids(call_id: str | None) -> list[str]:
    return [] if call_id is None else [call_id]


def _policy_conflict(goal_ids: Sequence[str], code: str) -> CommitDecision:
    return _blocked(
        GateOutcome.POLICY_CONFLICT,
        f"policy.{code}",
        "An active policy blocks the proposed output.",
        goal_ids=goal_ids,
    )


def _blocked(
    outcome: GateOutcome,
    code: str,
    message: str,
    *,
    goal_ids: Sequence[str] = (),
    call_ids: Sequence[str] = (),
    details: Mapping[str, Any] | None = None,
) -> CommitDecision:
    return CommitDecision(
        outcome=outcome,
        reasons=[
            GateReason(
                code=code,
                message=message,
                goal_ids=list(goal_ids),
                tool_call_ids=list(call_ids),
                details=dict(details or {}),
            )
        ],
    )


__all__ = [
    "GateContext",
    "PreCommitGate",
    "SetOrderEntry",
    "policy_action_digest",
    "policy_operation_completion_key",
    "provenance_scope_key",
    "stable_set_order",
]
