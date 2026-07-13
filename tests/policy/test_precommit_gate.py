from __future__ import annotations

from typing import Any

import pytest

from track_1_agent_under_test.car_guard.capability import (
    FeasibilityProof,
    LiveBinding,
    LiveToolRegistry,
)
from track_1_agent_under_test.car_guard.domain import (
    AmbiguitySlot,
    Candidate,
    ConfirmationScope,
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
    ConfirmationLatch,
    ConfirmationRequirement,
    ExecutionMode,
    GateContext,
    PolicyArgumentDecision,
    PolicyConflict,
    PolicyCompiler,
    PolicyDecision,
    PolicyFormatDecision,
    PreCommitGate,
    SemanticPolicyCall,
    SetOrderEntry,
    policy_action_digest,
    policy_operation_completion_key,
    provenance_scope_key,
    stable_set_order,
)
from track_1_agent_under_test.car_guard.runtime.budget import RuntimeBudget


def function_tool(
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


def control_tool(name: str = "set_control") -> dict[str, Any]:
    return function_tool(name, {"on": {"type": "boolean"}}, ["on"])


def set_proposal(
    *,
    tool_name: str = "set_control",
    arguments: dict[str, Any] | None = None,
    sources: dict[str, str] | None = None,
    goal_id: str = "goal-1",
) -> DecisionProposal:
    values = {"on": True} if arguments is None else arguments
    provenance = {"on": "user-turn-1"} if sources is None else sources
    return DecisionProposal.model_validate(
        {
            "kind": "tool_set",
            "goal_ids": [goal_id],
            "tool_calls": [
                {
                    "call_id": "call-set",
                    "tool_name": tool_name,
                    "arguments": values,
                    "argument_sources": provenance,
                    "goal_id": goal_id,
                }
            ],
        }
    )


def allow_proof(goal_id: str = "goal-1") -> FeasibilityProof:
    return FeasibilityProof(
        goal_id=goal_id,
        feasible=True,
        outcome=GateOutcome.ALLOW,
        bindings=[
            LiveBinding(
                recipe_id="control",
                semantic_operation="set_control",
                tool_name="set_control",
                arguments={"on": True},
                required_parameters={"on"},
            )
        ],
        checked_operations=["set_control"],
        reasons=["closure is feasible"],
    )


def base_context(
    *,
    tools: list[dict[str, Any]] | None = None,
    policy: PolicyDecision | None = None,
    evidence: EvidenceStore | None = None,
    budget: RuntimeBudget | None = None,
    policy_proposal: DecisionProposal | None = None,
    **updates: Any,
) -> GateContext:
    default_call = OfficialToolCall(tool_name="set_control", arguments={"on": True})
    explicit_scope = provenance_scope_key(
        "goal-1", default_call.tool_name, default_call.arguments
    )
    data: dict[str, Any] = {
        "live_tools": LiveToolRegistry(tools or [control_tool()]),
        "evidence": evidence or EvidenceStore(),
        "budget": budget or RuntimeBudget(),
        "compiled_policy": PolicyCompiler().compile(""),
        "policy_decision": policy or PolicyDecision(),
        "authorized_goal_ids": frozenset({"goal-1"}),
        "valid_provenance_ids": frozenset({"user-turn-1"}),
        "explicit_call_argument_values": {explicit_scope: {"on": True}},
        "feasibility_proofs": {"goal-1": allow_proof()},
        "set_order_entries": (
            SetOrderEntry(
                call=OfficialToolCall(tool_name="set_control", arguments={"on": True}),
                goal_id="goal-1",
            ),
        ),
    }
    data.update(updates)
    if policy is not None and "policy_decision_digest" not in data:
        data["policy_decision_digest"] = policy_action_digest(
            policy_proposal or set_proposal(),
            data["compiled_policy"],
            ordered_actions=data.get("confirmation_bundle", ()),
        )
    return GateContext(**data)


def test_valid_set_passes_all_twelve_stages_without_mutating_context() -> None:
    context = base_context()
    proposal = set_proposal()

    decision = PreCommitGate().validate(proposal, context)

    assert decision.outcome is GateOutcome.ALLOW
    assert decision.normalized_calls == [
        OfficialToolCall(
            tool_name="set_control", arguments={"on": True}, tool_call_id="call-set"
        )
    ]
    assert context.budget.steps == 0
    assert context.budget.successful_sets == set()


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kind": "respond",
            "user_text": "Done.",
            "tool_calls": [{"tool_name": "set_control", "arguments": {"on": True}}],
        },
        {
            "kind": "tool_set",
            "goal_ids": ["goal-1"],
            "tool_calls": [
                {"tool_name": "set_control", "arguments": {"on": True}},
                {"tool_name": "set_other", "arguments": {"on": True}},
            ],
        },
    ],
)
def test_stage_1_rejects_invalid_output_structure(payload: dict[str, Any]) -> None:
    decision = PreCommitGate().validate(payload, base_context())
    assert decision.outcome is GateOutcome.INVALID_PROPOSAL
    assert decision.reasons[0].code == "structure.invalid"


def test_stage_2_rejects_set_without_reactive_goal_authorization() -> None:
    decision = PreCommitGate().validate(
        set_proposal(), base_context(authorized_goal_ids=frozenset())
    )
    assert decision.outcome is GateOutcome.POLICY_CONFLICT
    assert decision.reasons[0].code == "authorization.reactive_only"


def test_stage_2_requires_action_level_goal_binding_for_each_set() -> None:
    proposal = DecisionProposal.model_validate(
        {
            "kind": "tool_set",
            "goal_ids": ["goal-1"],
            "tool_calls": [
                {
                    "call_id": "call-set",
                    "tool_name": "set_control",
                    "arguments": {"on": True},
                    "argument_sources": {"on": "user-turn-1"},
                }
            ],
        }
    )
    decision = PreCommitGate().validate(proposal, base_context())
    assert decision.outcome is GateOutcome.POLICY_CONFLICT
    assert decision.reasons[0].code == "authorization.action_goal_required"


def test_stage_3_rejects_non_live_tool_without_inspecting_other_tools() -> None:
    decision = PreCommitGate().validate(
        set_proposal(tool_name="set_absent"), base_context()
    )
    assert decision.outcome is GateOutcome.UNSUPPORTED_CAPABILITY
    assert decision.reasons[0].code == "capability.not_live"


@pytest.mark.parametrize(
    ("arguments", "expected_code"),
    [({}, "schema.required"), ({"on": "yes"}, "schema.type")],
)
def test_stage_4_rejects_live_schema_violation(
    arguments: dict[str, Any], expected_code: str
) -> None:
    decision = PreCommitGate().validate(
        set_proposal(arguments=arguments, sources={}), base_context()
    )
    assert decision.outcome is GateOutcome.UNSUPPORTED_PARAMETER
    assert decision.reasons[0].code == expected_code


@pytest.mark.parametrize(
    ("sources", "expected_code"),
    [({}, "provenance.incomplete"), ({"on": "invented"}, "provenance.unverifiable")],
)
def test_stage_5_requires_complete_verifiable_set_provenance(
    sources: dict[str, str], expected_code: str
) -> None:
    decision = PreCommitGate().validate(set_proposal(sources=sources), base_context())
    assert decision.outcome is GateOutcome.INVALID_PROPOSAL
    assert decision.reasons[0].code == expected_code


def test_stage_5_source_id_alone_cannot_authorize_an_argument() -> None:
    decision = PreCommitGate().validate(
        set_proposal(),
        base_context(
            valid_provenance_ids=frozenset({"user-turn-1"}),
            explicit_call_argument_values={},
        ),
    )
    assert decision.outcome is GateOutcome.INVALID_PROPOSAL
    assert decision.reasons[0].code == "provenance.unverifiable"


def _system_evidence(value: Any, status: EvidenceStatus) -> Evidence:
    return Evidence(
        proposition="argument value for on",
        value=value,
        status=status,
        source_kind=EvidenceSourceKind.SYSTEM,
        source_turn_id=f"system-{status.value}-{value}",
        confidence=1.0,
    )


@pytest.mark.parametrize(
    ("observation", "binding_goal"),
    [
        (_system_evidence(None, EvidenceStatus.UNKNOWN), "goal-1"),
        (_system_evidence(False, EvidenceStatus.KNOWN), "goal-1"),
        (_system_evidence(True, EvidenceStatus.KNOWN), "other-goal"),
    ],
)
def test_stage_5_unknown_mismatched_or_unrelated_evidence_cannot_authorize(
    observation: Evidence, binding_goal: str
) -> None:
    store = EvidenceStore()
    store.add(observation)
    assert observation.evidence_id is not None
    decision = PreCommitGate().validate(
        set_proposal(sources={"on": observation.evidence_id}),
        base_context(
            evidence=store,
            valid_provenance_ids=frozenset(),
            explicit_call_argument_values={},
            call_argument_evidence_ids={
                provenance_scope_key(binding_goal, "set_control", {"on": True}): {
                    "on": observation.evidence_id
                }
            },
        ),
    )
    assert decision.outcome is GateOutcome.INVALID_PROPOSAL
    assert decision.reasons[0].code == "provenance.unverifiable"


def test_stage_5_goal_argument_bound_known_evidence_authorizes_exact_value() -> None:
    observation = _system_evidence(True, EvidenceStatus.KNOWN)
    store = EvidenceStore()
    store.add(observation)
    stored = next(iter(store.evidence.values()))
    assert stored.evidence_id is not None
    decision = PreCommitGate().validate(
        set_proposal(sources={"on": stored.evidence_id}),
        base_context(
            evidence=store,
            valid_provenance_ids=frozenset(),
            explicit_call_argument_values={},
            call_argument_evidence_ids={
                provenance_scope_key("goal-1", "set_control", {"on": True}): {
                    "on": stored.evidence_id
                }
            },
        ),
    )
    assert decision.outcome is GateOutcome.ALLOW


def test_stage_5_derived_evidence_requires_known_validated_inputs() -> None:
    source = _system_evidence(None, EvidenceStatus.UNKNOWN)
    assert source.evidence_id is not None
    derived = Evidence(
        proposition="derived on argument",
        value=True,
        status=EvidenceStatus.KNOWN,
        source_kind=EvidenceSourceKind.DERIVED,
        source_turn_id="derived-turn",
        confidence=1.0,
        derived_from=[source.evidence_id],
        derivation="on_from_vehicle_state_v1",
    )
    store = EvidenceStore()
    store.update([source, derived])
    assert derived.evidence_id is not None
    decision = PreCommitGate().validate(
        set_proposal(sources={"on": derived.evidence_id}),
        base_context(
            evidence=store,
            valid_provenance_ids=frozenset(),
            explicit_call_argument_values={},
            call_argument_evidence_ids={
                provenance_scope_key("goal-1", "set_control", {"on": True}): {
                    "on": derived.evidence_id
                }
            },
        ),
    )
    assert decision.outcome is GateOutcome.INVALID_PROPOSAL
    assert decision.reasons[0].code == "provenance.unverifiable"


def test_stage_5_policy_fixed_value_is_action_bound_provenance() -> None:
    policy = PolicyDecision(
        argument_decisions=[
            PolicyArgumentDecision(
                name="on",
                rule_ids=["POL-X"],
                has_fixed_value=True,
                fixed_value=True,
                source="policy-fixed",
            )
        ]
    )
    decision = PreCommitGate().validate(
        set_proposal(sources={"on": "policy-fixed"}),
        base_context(
            policy=policy,
            valid_provenance_ids=frozenset(),
            explicit_call_argument_values={},
        ),
    )
    assert decision.outcome is GateOutcome.ALLOW


def test_stage_6_blocks_only_goal_required_unresolved_ambiguity() -> None:
    slot = AmbiguitySlot(
        name="on",
        goal_id="goal-1",
        candidates=[
            Candidate(candidate_id="enable", value=True),
            Candidate(candidate_id="disable", value=False),
        ],
    )
    decision = PreCommitGate().validate(
        set_proposal(), base_context(ambiguities=(slot,))
    )
    assert decision.outcome is GateOutcome.NEED_USER_DISAMBIGUATION
    assert decision.reasons[0].details == {"slots": ["on"]}


def evidence_need(goal_id: str = "goal-1") -> EvidenceNeed:
    return EvidenceNeed(
        proposition="vehicle state is safe",
        required_for_goal_id=goal_id,
        acceptable_sources={"system_context"},
    )


def unknown_evidence() -> Evidence:
    return Evidence(
        proposition="vehicle state is safe",
        value=None,
        status=EvidenceStatus.UNKNOWN,
        source_kind=EvidenceSourceKind.SYSTEM,
        source_turn_id="system-turn",
        confidence=1.0,
    )


def test_stage_7_distinguishes_unread_from_observed_unavailable_evidence() -> None:
    need = evidence_need()
    unread = EvidenceStore()
    unread.register_need(need)
    decision = PreCommitGate().validate(set_proposal(), base_context(evidence=unread))
    assert decision.outcome is GateOutcome.NEED_READ

    observed = EvidenceStore()
    observed.register_need(need)
    observed.add(unknown_evidence())
    decision = PreCommitGate().validate(set_proposal(), base_context(evidence=observed))
    assert decision.outcome is GateOutcome.UNAVAILABLE_EVIDENCE


def test_advisory_evidence_need_cannot_replace_argument_provenance() -> None:
    advisory = EvidenceNeed(
        proposition="optional contextual state",
        required_for_goal_id="goal-1",
        acceptable_sources={"tool_result"},
        required_before_set=False,
    )
    store = EvidenceStore()
    store.register_need(advisory)

    decision = PreCommitGate().validate(
        set_proposal(sources={}),
        base_context(evidence=store),
    )

    assert decision.outcome is GateOutcome.INVALID_PROPOSAL
    assert decision.reasons[0].code == "provenance.incomplete"


def test_stage_8_blocks_policy_conflict_required_read_and_prerequisite_order() -> None:
    conflict = PolicyDecision(
        conflicts=[PolicyConflict(code="unsafe", message="unsafe action")]
    )
    decision = PreCommitGate().validate(set_proposal(), base_context(policy=conflict))
    assert decision.outcome is GateOutcome.POLICY_CONFLICT
    assert decision.reasons[0].code == "policy.unsafe"

    read_required = PolicyDecision(
        required_reads=[SemanticPolicyCall(semantic_operation="get_status")]
    )
    decision = PreCommitGate().validate(
        set_proposal(), base_context(policy=read_required)
    )
    assert decision.outcome is GateOutcome.NEED_READ
    assert decision.reasons[0].code == "policy.read_required"

    prerequisite = PolicyDecision(
        prerequisite_calls=[
            SemanticPolicyCall(
                semantic_operation="set_prerequisite", arguments={"on": False}
            )
        ]
    )
    decision = PreCommitGate().validate(
        set_proposal(),
        base_context(
            policy=prerequisite,
            tools=[control_tool(), control_tool("set_prerequisite")],
            policy_call_bindings={
                "set_prerequisite": OfficialToolCall(
                    tool_name="set_prerequisite", arguments={"on": False}
                )
            },
        ),
    )
    assert decision.outcome is GateOutcome.POLICY_CONFLICT
    assert decision.reasons[0].code == "policy.prerequisite_order"


def test_completed_policy_read_is_bound_to_args_state_and_goal_bundle() -> None:
    policy_call = SemanticPolicyCall(
        semantic_operation="get_status", arguments={"zone": "ALL"}
    )
    policy = PolicyDecision(required_reads=[policy_call])
    bound = OfficialToolCall(tool_name="get_status", arguments={"zone": "ALL"})
    valid_key = policy_operation_completion_key(
        policy_call.semantic_operation,
        bound,
        evidence_state_version=0,
        bundle_goal_ids=("goal-1",),
    )
    shared = {
        "policy": policy,
        "policy_call_bindings": {policy_call.semantic_operation: bound},
        "completed_policy_operations": frozenset({valid_key}),
    }

    allowed = PreCommitGate().validate(set_proposal(), base_context(**shared))
    assert allowed.outcome is GateOutcome.ALLOW

    bare_semantic_name = PreCommitGate().validate(
        set_proposal(),
        base_context(
            **{
                **shared,
                "completed_policy_operations": frozenset({"get_status"}),
            }
        ),
    )
    assert bare_semantic_name.outcome is GateOutcome.NEED_READ

    wrong_arguments = OfficialToolCall(
        tool_name="get_status", arguments={"zone": "DRIVER"}
    )
    wrong_argument_key = policy_operation_completion_key(
        policy_call.semantic_operation,
        wrong_arguments,
        evidence_state_version=0,
        bundle_goal_ids=("goal-1",),
    )
    stale_arguments = PreCommitGate().validate(
        set_proposal(),
        base_context(
            **{
                **shared,
                "completed_policy_operations": frozenset({wrong_argument_key}),
            }
        ),
    )
    assert stale_arguments.outcome is GateOutcome.NEED_READ

    changed_state = EvidenceStore()
    changed_state.invalidate_tool_state()
    stale_state = PreCommitGate().validate(
        set_proposal(), base_context(evidence=changed_state, **shared)
    )
    assert stale_state.outcome is GateOutcome.NEED_READ

    other_bundle_key = policy_operation_completion_key(
        policy_call.semantic_operation,
        bound,
        evidence_state_version=0,
        bundle_goal_ids=("goal-1", "goal-2"),
    )
    stale_bundle = PreCommitGate().validate(
        set_proposal(),
        base_context(
            **{
                **shared,
                "completed_policy_operations": frozenset({other_bundle_key}),
            }
        ),
    )
    assert stale_bundle.outcome is GateOutcome.NEED_READ


def test_stage_8_enforces_fixed_policy_argument_and_text_format_checks() -> None:
    policy = PolicyDecision(
        argument_decisions=[
            PolicyArgumentDecision(name="on", has_fixed_value=True, fixed_value=False)
        ]
    )
    decision = PreCommitGate().validate(set_proposal(), base_context(policy=policy))
    assert decision.outcome is GateOutcome.POLICY_CONFLICT
    assert decision.reasons[0].code == "policy.argument_constraint"

    response = DecisionProposal(kind="respond", user_text="I found the result.")
    format_policy = PolicyDecision(
        format_decisions=[PolicyFormatDecision(code="tts_safe")]
    )
    decision = PreCommitGate().validate(
        response, base_context(policy=format_policy, policy_proposal=response)
    )
    assert decision.outcome is GateOutcome.ALLOW

    unsafe_response = DecisionProposal(
        kind="respond", user_text="I found *the result*."
    )
    rejected = PreCommitGate().validate(
        unsafe_response,
        base_context(
            policy=format_policy,
            policy_proposal=unsafe_response,
            satisfied_format_codes=frozenset({"tts_safe"}),
        ),
    )
    assert rejected.outcome is GateOutcome.POLICY_CONFLICT
    assert rejected.reasons[0].code == "policy.format_unsatisfied"
    assert "tts_safe" in rejected.reasons[0].details["format_codes"]


@pytest.mark.parametrize(
    ("format_code", "allowed_text", "rejected_text"),
    [
        (
            "datetime_24h",
            "Done. I turned on the purple ambient lights.",
            "I scheduled it for 8 PM.",
        ),
        (
            "temperature_celsius",
            "I confirmed the zones differ by 4 degrees from each other.",
            "I set the temperature to 70 degrees F.",
        ),
        (
            "metric_distance_km_and_m",
            "Done. I directed the airflow to your feet.",
            "I found the destination 10 feet away.",
        ),
    ],
)
def test_unit_format_checks_require_actual_unit_or_clock_markers(
    format_code: str, allowed_text: str, rejected_text: str
) -> None:
    policy = PolicyDecision(
        format_decisions=[PolicyFormatDecision(code=format_code)]
    )
    allowed = DecisionProposal(kind="respond", user_text=allowed_text)
    allowed_decision = PreCommitGate().validate(
        allowed,
        base_context(policy=policy, policy_proposal=allowed),
    )
    assert allowed_decision.outcome is GateOutcome.ALLOW

    rejected = DecisionProposal(kind="respond", user_text=rejected_text)
    rejected_decision = PreCommitGate().validate(
        rejected,
        base_context(policy=policy, policy_proposal=rejected),
    )
    assert rejected_decision.outcome is GateOutcome.POLICY_CONFLICT
    assert rejected_decision.reasons[0].details["format_codes"] == [format_code]


def test_text_format_validation_checks_required_content_not_claimed_codes() -> None:
    policy = PolicyDecision(
        format_decisions=[
            PolicyFormatDecision(
                code="present_all_poi_names",
                details={"names": ["Alpha Cafe", "Beta Library"]},
            ),
            PolicyFormatDecision(code="ask_which_poi_to_navigate_to"),
        ]
    )
    missing_name = DecisionProposal(
        kind="respond",
        user_text="I found Alpha Cafe. Which place should I navigate to?",
    )
    rejected = PreCommitGate().validate(
        missing_name,
        base_context(
            policy=policy,
            policy_proposal=missing_name,
            satisfied_format_codes=frozenset(
                {"present_all_poi_names", "ask_which_poi_to_navigate_to"}
            ),
        ),
    )
    assert rejected.outcome is GateOutcome.POLICY_CONFLICT
    assert rejected.reasons[0].details["format_codes"] == ["present_all_poi_names"]

    complete = DecisionProposal(
        kind="respond",
        user_text=(
            "I found Alpha Cafe and Beta Library. Which place should I navigate to?"
        ),
    )
    allowed = PreCommitGate().validate(
        complete, base_context(policy=policy, policy_proposal=complete)
    )
    assert allowed.outcome is GateOutcome.ALLOW


def test_precomputed_policy_decision_requires_exact_action_bundle_digest() -> None:
    supplemental = PolicyDecision(
        conflicts=[PolicyConflict(code="supplemental", message="bound restriction")]
    )
    decision = PreCommitGate().validate(
        set_proposal(),
        base_context(
            policy=supplemental,
            policy_decision_digest="0" * 64,
        ),
    )
    assert decision.outcome is GateOutcome.POLICY_CONFLICT
    assert decision.reasons[0].code == "policy.precomputed_policy_binding_mismatch"


def confirmed_latch(call: OfficialToolCall) -> ConfirmationLatch:
    latch = ConfirmationLatch()
    latch.arm(
        ConfirmationScope(
            goal_ids=["goal-1"],
            ordered_actions=[call],
            requested_at_user_turn=0,
            expires_after_user_turn=2,
        )
    )
    resolution = latch.resolve(
        "yes",
        current_user_turn=1,
        source_turn_id="user-turn-confirmation",
        goal_ids=["goal-1"],
        ordered_actions=[call],
    )
    assert resolution.confirmation is not None
    return latch


def test_stage_9_requires_confirmation_for_exact_ordered_bundle() -> None:
    policy = PolicyDecision(confirmation=ConfirmationRequirement(rule_ids=["POL-X"]))
    expected = OfficialToolCall(tool_name="set_control", arguments={"on": True})
    without = PreCommitGate().validate(
        set_proposal(),
        base_context(policy=policy, confirmation_bundle=(expected,)),
    )
    assert without.outcome is GateOutcome.NEED_CONFIRMATION

    latch = confirmed_latch(expected)
    allowed = PreCommitGate().validate(
        set_proposal(),
        base_context(
            policy=policy,
            confirmation_latch=latch,
            confirmation_bundle=(expected,),
            current_user_turn=1,
        ),
    )
    assert allowed.outcome is GateOutcome.ALLOW

    changed = OfficialToolCall(tool_name="set_control", arguments={"on": False})
    rejected = PreCommitGate().validate(
        set_proposal(),
        base_context(
            policy=policy,
            confirmation_latch=latch,
            confirmation_bundle=(changed,),
            current_user_turn=1,
        ),
    )
    assert rejected.outcome is GateOutcome.NEED_CONFIRMATION


def test_bundle_confirmation_does_not_expand_action_goal_proof_scope() -> None:
    first = OfficialToolCall(tool_name="set_control", arguments={"on": True})
    second = OfficialToolCall(tool_name="set_other", arguments={"on": False})
    ordered = (first, second)
    latch = ConfirmationLatch()
    latch.arm(
        ConfirmationScope(
            goal_ids=["goal-1", "goal-2"],
            ordered_actions=list(ordered),
            requested_at_user_turn=0,
            expires_after_user_turn=2,
        )
    )
    resolution = latch.resolve(
        "yes",
        current_user_turn=1,
        source_turn_id="user-confirmation",
        goal_ids=["goal-1", "goal-2"],
        ordered_actions=list(ordered),
    )
    assert resolution.confirmation is not None

    policy = PolicyDecision(
        confirmation=ConfirmationRequirement(rule_ids=["POL-BUNDLE"])
    )
    decision = PreCommitGate().validate(
        set_proposal(),
        base_context(
            tools=[control_tool(), control_tool("set_other")],
            policy=policy,
            confirmation_latch=latch,
            current_user_turn=1,
            authorized_goal_ids=frozenset(),
            bundle_goal_ids=("goal-1", "goal-2"),
            execution_bundle=ordered,
            confirmation_bundle=ordered,
            # Only the current action goal needs an affirmative proof.
            feasibility_proofs={"goal-1": allow_proof("goal-1")},
            set_order_entries=(
                SetOrderEntry(call=first, goal_id="goal-1", user_mention_rank=0),
                SetOrderEntry(call=second, goal_id="goal-2", user_mention_rank=1),
            ),
        ),
    )
    assert decision.outcome is GateOutcome.ALLOW


def test_stage_10_requires_goal_matching_affirmative_feasibility_proof() -> None:
    missing = PreCommitGate().validate(
        set_proposal(), base_context(feasibility_proofs={})
    )
    assert missing.outcome is GateOutcome.INVALID_PROPOSAL
    assert missing.reasons[0].code == "closure.proof_required"

    blocked_proof = FeasibilityProof(
        goal_id="goal-1",
        feasible=False,
        outcome=GateOutcome.UNSUPPORTED_CAPABILITY,
        checked_operations=["set_control"],
        reasons=["triggered prerequisite is not bindable"],
    )
    blocked = PreCommitGate().validate(
        set_proposal(),
        base_context(feasibility_proofs={"goal-1": blocked_proof}),
    )
    assert blocked.outcome is GateOutcome.UNSUPPORTED_CAPABILITY
    assert blocked.reasons[0].code == "closure.not_feasible"


def test_stage_10_feasible_flag_cannot_authorize_a_different_or_empty_binding() -> None:
    empty = FeasibilityProof(
        goal_id="goal-1",
        feasible=True,
        outcome=GateOutcome.ALLOW,
        checked_operations=["set_control"],
        reasons=["claimed feasible without a binding"],
    )
    decision = PreCommitGate().validate(
        set_proposal(), base_context(feasibility_proofs={"goal-1": empty})
    )
    assert decision.outcome is GateOutcome.INVALID_PROPOSAL
    assert decision.reasons[0].code == "closure.next_binding_required"

    mismatched = FeasibilityProof(
        goal_id="goal-1",
        feasible=True,
        outcome=GateOutcome.ALLOW,
        bindings=[
            LiveBinding(
                recipe_id="control",
                semantic_operation="set_control",
                tool_name="set_control",
                arguments={"on": False},
                required_parameters={"on"},
            )
        ],
        checked_operations=["set_control"],
        reasons=["proof is for a different argument value"],
    )
    decision = PreCommitGate().validate(
        set_proposal(), base_context(feasibility_proofs={"goal-1": mismatched})
    )
    assert decision.outcome is GateOutcome.INVALID_PROPOSAL
    assert decision.reasons[0].code == "closure.next_call_mismatch"


def test_stage_11_enforces_stable_set_order_and_independent_parallel_gets() -> None:
    target = OfficialToolCall(tool_name="set_control", arguments={"on": True})
    prerequisite = OfficialToolCall(
        tool_name="set_prerequisite", arguments={"on": False}
    )
    entries = (
        SetOrderEntry(call=target, goal_id="goal-1", user_mention_rank=0),
        SetOrderEntry(
            call=prerequisite,
            goal_id="goal-1",
            policy_dependency_rank=0,
            user_mention_rank=1,
        ),
    )
    order_proof = FeasibilityProof(
        goal_id="goal-1",
        feasible=True,
        outcome=GateOutcome.ALLOW,
        bindings=[
            LiveBinding(
                recipe_id="order",
                semantic_operation="set_control",
                tool_name=target.tool_name,
                arguments=target.arguments,
                required_parameters={"on"},
            ),
            LiveBinding(
                recipe_id="order",
                semantic_operation="set_prerequisite",
                tool_name=prerequisite.tool_name,
                arguments=prerequisite.arguments,
                required_parameters={"on"},
            ),
        ],
        checked_operations=["set_control", "set_prerequisite"],
        reasons=["both calls are feasible"],
    )
    assert stable_set_order(entries) == (prerequisite, target)
    decision = PreCommitGate().validate(
        set_proposal(),
        base_context(
            set_order_entries=entries,
            feasibility_proofs={"goal-1": order_proof},
        ),
    )
    assert decision.outcome is GateOutcome.INVALID_PROPOSAL
    assert decision.reasons[0].code == "order.unstable_set_order"

    dependent_gets = DecisionProposal.model_validate(
        {
            "kind": "tool_get",
            "goal_ids": ["goal-1"],
            "tool_calls": [
                {"call_id": "one", "tool_name": "get_one"},
                {
                    "call_id": "two",
                    "tool_name": "get_two",
                    "depends_on": ["one"],
                },
            ],
        }
    )
    decision = PreCommitGate().validate(
        dependent_gets,
        base_context(tools=[function_tool("get_one"), function_tool("get_two")]),
    )
    assert decision.outcome is GateOutcome.INVALID_PROPOSAL
    assert decision.reasons[0].code == "order.parallel_get_dependency"


def test_stage_11_rejects_set_without_mandatory_order_metadata() -> None:
    decision = PreCommitGate().validate(
        set_proposal(), base_context(set_order_entries=())
    )
    assert decision.outcome is GateOutcome.INVALID_PROPOSAL
    assert decision.reasons[0].code == "order.set_order_required"


@pytest.mark.parametrize(
    ("compiled_text", "tool_name", "state"),
    [
        (
            "POL-011:",
            "set_air_conditioning",
            {
                "window_driver_position": 40,
                "window_passenger_position": 0,
                "window_driver_rear_position": 0,
                "window_passenger_rear_position": 0,
                "fan_speed": 1,
            },
        ),
        ("POL-014:", "set_head_lights_high_beams", {"fog_lights": True}),
    ],
)
def test_action_bound_policy_evaluation_cannot_be_bypassed_by_empty_decision(
    compiled_text: str, tool_name: str, state: dict[str, Any]
) -> None:
    call = OfficialToolCall(tool_name=tool_name, arguments={"on": True})
    explicit_scope = provenance_scope_key("goal-1", call.tool_name, call.arguments)
    action_proof = FeasibilityProof(
        goal_id="goal-1",
        feasible=True,
        outcome=GateOutcome.ALLOW,
        bindings=[
            LiveBinding(
                recipe_id="action",
                semantic_operation=tool_name,
                tool_name=tool_name,
                arguments={"on": True},
                required_parameters={"on"},
            )
        ],
        checked_operations=[tool_name],
        reasons=["target alone appears feasible"],
    )
    decision = PreCommitGate().validate(
        set_proposal(tool_name=tool_name),
        base_context(
            tools=[control_tool(tool_name)],
            policy=PolicyDecision(),
            compiled_policy=PolicyCompiler().compile(compiled_text),
            policy_state=state,
            explicit_call_argument_values={explicit_scope: {"on": True}},
            feasibility_proofs={"goal-1": action_proof},
            set_order_entries=(SetOrderEntry(call=call, goal_id="goal-1"),),
        ),
    )
    assert decision.outcome is GateOutcome.POLICY_CONFLICT
    assert decision.reasons[0].code == "policy.prerequisite_order"


def test_pol_014_false_then_true_bundle_has_call_scoped_provenance() -> None:
    fog_off = OfficialToolCall(tool_name="set_fog_lights", arguments={"on": False})
    high_on = OfficialToolCall(
        tool_name="set_head_lights_high_beams", arguments={"on": True}
    )
    proof = FeasibilityProof(
        goal_id="goal-1",
        feasible=True,
        outcome=GateOutcome.ALLOW,
        bindings=[
            LiveBinding(
                recipe_id="high-beam",
                semantic_operation="turn_off_fog_lights",
                tool_name=fog_off.tool_name,
                arguments=fog_off.arguments,
                required_parameters={"on"},
            ),
            LiveBinding(
                recipe_id="high-beam",
                semantic_operation="turn_on_high_beam",
                tool_name=high_on.tool_name,
                arguments=high_on.arguments,
                required_parameters={"on"},
            ),
        ],
        checked_operations=["turn_off_fog_lights", "turn_on_high_beam"],
        reasons=["the full ordered bundle is feasible"],
    )
    order = (
        SetOrderEntry(
            call=fog_off,
            goal_id="goal-1",
            policy_dependency_rank=0,
        ),
        SetOrderEntry(call=high_on, goal_id="goal-1"),
    )
    latch = confirmed_latch_for_bundle([fog_off, high_on])
    budget = RuntimeBudget()
    high_scope = provenance_scope_key("goal-1", high_on.tool_name, high_on.arguments)
    shared = {
        "live_tools": LiveToolRegistry(
            [control_tool(fog_off.tool_name), control_tool(high_on.tool_name)]
        ),
        "evidence": EvidenceStore(),
        "budget": budget,
        "compiled_policy": PolicyCompiler().compile("POL-014:"),
        "confirmation_latch": latch,
        "current_user_turn": 1,
        "authorized_goal_ids": frozenset({"goal-1"}),
        "valid_provenance_ids": frozenset({"user-turn-1"}),
        "explicit_call_argument_values": {high_scope: {"on": True}},
        "feasibility_proofs": {"goal-1": proof},
        "policy_call_bindings": {"vehicle.fog_lights.set": fog_off},
        "confirmation_bundle": (fog_off, high_on),
        "set_order_entries": order,
        "policy_state": {"fog_lights": True},
    }
    fog_proposal = set_proposal(
        tool_name=fog_off.tool_name,
        arguments=fog_off.arguments,
        sources={"on": "policy:POL-014"},
    )
    first = PreCommitGate().validate(fog_proposal, GateContext(**shared))
    assert first.outcome is GateOutcome.ALLOW

    budget.record_success(fog_off.tool_name, fog_off.arguments, state_changing=True)
    shared["completed_policy_operations"] = frozenset(
        {
            policy_operation_completion_key(
                "vehicle.fog_lights.set",
                fog_off,
                evidence_state_version=0,
                bundle_goal_ids=("goal-1",),
            )
        }
    )
    high_proposal = set_proposal(
        tool_name=high_on.tool_name,
        arguments=high_on.arguments,
        sources={"on": "user-turn-1"},
    )
    second = PreCommitGate().validate(high_proposal, GateContext(**shared))
    assert second.outcome is GateOutcome.ALLOW


def confirmed_latch_for_bundle(calls: list[OfficialToolCall]) -> ConfirmationLatch:
    latch = ConfirmationLatch()
    latch.arm(
        ConfirmationScope(
            goal_ids=["goal-1"],
            ordered_actions=calls,
            requested_at_user_turn=0,
            expires_after_user_turn=2,
        )
    )
    resolution = latch.resolve(
        "yes",
        current_user_turn=1,
        source_turn_id="user-turn-confirmation",
        goal_ids=["goal-1"],
        ordered_actions=calls,
    )
    assert resolution.confirmation is not None
    return latch


def test_stage_11_honors_policy_serial_execution_mode() -> None:
    parallel_gets = DecisionProposal.model_validate(
        {
            "kind": "tool_get",
            "tool_calls": [
                {"tool_name": "get_one"},
                {"tool_name": "get_two"},
            ],
        }
    )
    policy = PolicyDecision(execution_mode=ExecutionMode.SERIAL)
    decision = PreCommitGate().validate(
        parallel_gets,
        base_context(
            policy=policy,
            policy_proposal=parallel_gets,
            tools=[function_tool("get_one"), function_tool("get_two")],
        ),
    )
    assert decision.outcome is GateOutcome.POLICY_CONFLICT
    assert decision.reasons[0].code == "policy.serial_execution_required"


def test_stage_12_blocks_replay_and_reserves_a_finishing_step() -> None:
    duplicate_budget = RuntimeBudget()
    duplicate_budget.record_success("set_control", {"on": True}, state_changing=True)
    duplicate = PreCommitGate().validate(
        set_proposal(), base_context(budget=duplicate_budget)
    )
    assert duplicate.outcome is GateOutcome.INVALID_PROPOSAL
    assert duplicate.reasons[0].code == "budget.duplicate_or_exhausted_call"

    exhausted = RuntimeBudget(soft_max_steps=36, steps=35)
    no_room = PreCommitGate().validate(set_proposal(), base_context(budget=exhausted))
    assert no_room.outcome is GateOutcome.INVALID_PROPOSAL
    assert no_room.reasons[0].code == "budget.insufficient_steps"

    terminal = PreCommitGate().validate(
        set_proposal(),
        base_context(budget=exhausted, terminal_after_success=True),
    )
    assert terminal.outcome is GateOutcome.INVALID_PROPOSAL
    assert terminal.reasons[0].code == "budget.insufficient_steps"


def test_stage_12_reserves_full_frozen_bundle_before_first_set() -> None:
    first = OfficialToolCall(tool_name="set_control", arguments={"on": True})
    second = OfficialToolCall(tool_name="set_other", arguments={"on": False})
    ordered = (first, second)
    context_updates = {
        "tools": [control_tool(), control_tool("set_other")],
        "bundle_goal_ids": ("goal-1", "goal-2"),
        "execution_bundle": ordered,
        "set_order_entries": (
            SetOrderEntry(call=first, goal_id="goal-1", user_mention_rank=0),
            SetOrderEntry(call=second, goal_id="goal-2", user_mention_rank=1),
        ),
    }

    insufficient = PreCommitGate().validate(
        set_proposal(),
        base_context(
            budget=RuntimeBudget(soft_max_steps=36, steps=34),
            **context_updates,
        ),
    )
    assert insufficient.outcome is GateOutcome.INVALID_PROPOSAL
    assert insufficient.reasons[0].code == "budget.insufficient_steps"
    assert insufficient.reasons[0].details == {
        "remaining_bundle_calls": 2,
        "final_response_steps": 1,
    }

    enough = PreCommitGate().validate(
        set_proposal(),
        base_context(
            budget=RuntimeBudget(soft_max_steps=36, steps=33),
            **context_updates,
        ),
    )
    assert enough.outcome is GateOutcome.ALLOW


def test_stage_12_rejects_duplicate_side_effect_inside_frozen_bundle() -> None:
    repeated = OfficialToolCall(tool_name="set_control", arguments={"on": True})
    decision = PreCommitGate().validate(
        set_proposal(),
        base_context(
            bundle_goal_ids=("goal-1", "goal-2"),
            execution_bundle=(repeated, repeated),
            set_order_entries=(
                SetOrderEntry(call=repeated, goal_id="goal-1", user_mention_rank=0),
                SetOrderEntry(call=repeated, goal_id="goal-2", user_mention_rank=1),
            ),
        ),
    )
    assert decision.outcome is GateOutcome.INVALID_PROPOSAL
    assert decision.reasons[0].code == "budget.duplicate_bundle_call"
