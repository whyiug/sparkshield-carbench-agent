from __future__ import annotations

import pytest

from track_1_agent_under_test.car_guard.disambiguation import (
    CandidateEliminationRule,
    ClarificationAnswer,
    DisambiguationResolver,
)
from track_1_agent_under_test.car_guard.domain import (
    AmbiguitySlot,
    Candidate,
    GateOutcome,
    ResolutionLevel,
)


def _slot(name: str = "contact") -> AmbiguitySlot:
    return AmbiguitySlot(
        name=name,
        goal_id="goal-call",
        candidates=[
            Candidate(candidate_id="alex-home", value="contact-1", label="Alex home"),
            Candidate(candidate_id="alex-work", value="contact-2", label="Alex work"),
            Candidate(candidate_id="alex-mobile", value="contact-3", label="Alex mobile"),
        ],
    )


@pytest.mark.parametrize(
    "level",
    [
        ResolutionLevel.STRICT_POLICY,
        ResolutionLevel.EXPLICIT_USER,
        ResolutionLevel.PREFERENCE,
        ResolutionLevel.HEURISTIC,
        ResolutionLevel.CONTEXT,
    ],
)
def test_each_internal_precedence_level_can_resolve(level: ResolutionLevel) -> None:
    result = DisambiguationResolver().resolve(
        _slot(),
        eliminations=[
            CandidateEliminationRule(
                level=level,
                eliminate_candidate_ids=["alex-work", "alex-mobile"],
                reason=f"Evidence at {level.value} selects the home contact",
                evidence_ids=[f"ev-{level.value}"],
            )
        ],
    )

    assert result.outcome is GateOutcome.ALLOW
    assert result.slot.chosen.candidate_id == "alex-home"
    assert result.slot.chosen_by is level
    assert [item.candidate_id for item in result.slot.eliminated] == [
        "alex-work",
        "alex-mobile",
    ]


def test_precedence_is_strict_even_when_rules_arrive_in_reverse_order() -> None:
    slot = _slot()
    lower_precedence = CandidateEliminationRule(
        level=ResolutionLevel.EXPLICIT_USER,
        eliminate_candidate_ids=["alex-home", "alex-mobile"],
        reason="A lower-precedence interpretation would select work",
    )
    policy = CandidateEliminationRule(
        level=ResolutionLevel.STRICT_POLICY,
        eliminate_candidate_ids=["alex-work", "alex-mobile"],
        reason="Policy permits only the home contact",
        rule_id="POL-contact",
    )

    result = DisambiguationResolver().resolve(
        slot,
        eliminations=[lower_precedence, policy],
    )

    assert result.slot.chosen.candidate_id == "alex-home"
    assert result.slot.chosen_by is ResolutionLevel.STRICT_POLICY
    assert {item.eliminated_by for item in result.slot.eliminated} == {
        ResolutionLevel.STRICT_POLICY
    }


def test_resolver_continues_through_preference_and_context_before_asking() -> None:
    result = DisambiguationResolver().resolve(
        _slot(),
        eliminations=[
            CandidateEliminationRule(
                level=ResolutionLevel.PREFERENCE,
                eliminate_candidate_ids=["alex-mobile"],
                reason="The learned preference excludes mobile",
            ),
            CandidateEliminationRule(
                level=ResolutionLevel.CONTEXT,
                eliminate_candidate_ids=["alex-work"],
                reason="Only the home contact is reachable in current context",
            ),
        ],
    )

    assert result.outcome is GateOutcome.ALLOW
    assert result.slot.chosen.candidate_id == "alex-home"
    assert result.slot.chosen_by is ResolutionLevel.CONTEXT
    assert [item.eliminated_by for item in result.slot.eliminated] == [
        ResolutionLevel.PREFERENCE,
        ResolutionLevel.CONTEXT,
    ]


def test_candidate_and_option_order_remains_original_and_stable() -> None:
    result = DisambiguationResolver().resolve(
        _slot(),
        eliminations=[
            CandidateEliminationRule(
                level=ResolutionLevel.HEURISTIC,
                eliminate_candidate_ids=["alex-work"],
                reason="Default excludes the work entry",
            )
        ],
    )

    assert result.outcome is GateOutcome.NEED_USER_DISAMBIGUATION
    assert [candidate.candidate_id for candidate in result.slot.candidates] == [
        "alex-home",
        "alex-work",
        "alex-mobile",
    ]
    assert [
        option.candidate_id for option in result.clarification_request.options
    ] == ["alex-home", "alex-mobile"]


def test_two_or_more_candidates_produce_one_precise_bound_question() -> None:
    result = DisambiguationResolver().resolve(_slot())

    assert result.outcome is GateOutcome.NEED_USER_DISAMBIGUATION
    assert result.slot.chosen is None
    assert result.clarification_request.slot_name == "contact"
    assert result.clarification_request.goal_id == "goal-call"
    assert result.clarification_request.prompt == (
        "Which contact did you mean: Alex home, Alex work, or Alex mobile?"
    )


def test_zero_candidates_is_a_conflict_and_never_a_guess() -> None:
    result = DisambiguationResolver().resolve(
        _slot(),
        eliminations=[
            CandidateEliminationRule(
                level=ResolutionLevel.STRICT_POLICY,
                eliminate_candidate_ids=[
                    "alex-mobile",
                    "alex-home",
                    "alex-work",
                ],
                reason="Policy constraints conflict with every candidate",
            )
        ],
    )

    assert result.outcome is GateOutcome.POLICY_CONFLICT
    assert result.slot.chosen is None
    assert result.clarification_request is None
    assert [item.candidate_id for item in result.slot.eliminated] == [
        "alex-home",
        "alex-work",
        "alex-mobile",
    ]


def test_user_clarification_is_bound_to_the_original_slot_snapshot() -> None:
    resolver = DisambiguationResolver()
    contact_request = resolver.resolve(_slot("contact")).clarification_request
    route_request = resolver.resolve(_slot("route")).clarification_request
    wrong_answer = ClarificationAnswer(
        request_id=route_request.request_id,
        slot_name="route",
        candidate_id="alex-home",
        source_turn_id="turn-4",
    )

    rejected = resolver.resolve(_slot("contact"), clarification=wrong_answer)

    assert rejected.outcome is GateOutcome.NEED_USER_DISAMBIGUATION
    assert rejected.slot.chosen is None
    assert rejected.clarification_request.request_id == contact_request.request_id
    assert rejected.reasons[0].code == "ambiguity.clarification_binding_mismatch"


def test_bound_user_clarification_is_the_sixth_and_final_level() -> None:
    resolver = DisambiguationResolver()
    pending = resolver.resolve(_slot())
    answer = ClarificationAnswer(
        request_id=pending.clarification_request.request_id,
        slot_name="contact",
        candidate_id="alex-work",
        source_turn_id="turn-4",
    )

    resolved = resolver.resolve(_slot(), clarification=answer)

    assert resolved.outcome is GateOutcome.ALLOW
    assert resolved.slot.chosen.candidate_id == "alex-work"
    assert resolved.slot.chosen_by is ResolutionLevel.USER_CLARIFICATION
    assert [item.candidate_id for item in resolved.slot.eliminated] == [
        "alex-home",
        "alex-mobile",
    ]
    assert all(
        item.eliminated_by is ResolutionLevel.USER_CLARIFICATION
        for item in resolved.slot.eliminated
    )


@pytest.mark.parametrize(
    "outcome",
    [
        GateOutcome.NEED_READ,
        GateOutcome.UNSUPPORTED_PARAMETER,
        GateOutcome.UNAVAILABLE_EVIDENCE,
    ],
)
def test_upstream_parameter_or_system_evidence_outcomes_never_ask_user(
    outcome: GateOutcome,
) -> None:
    result = DisambiguationResolver().resolve(
        _slot(),
        upstream_outcome=outcome,
    )

    assert result.outcome is outcome
    assert result.slot.chosen is None
    assert result.clarification_request is None
