"""Strict six-level ambiguity resolution by candidate elimination."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self, TypeAlias

from pydantic import Field, model_validator

from ..domain.types import (
    AmbiguitySlot,
    DomainModel,
    Elimination,
    GateOutcome,
    GateReason,
    ResolutionLevel,
)
from .candidates import (
    CandidateEliminationRule,
    ClarificationAnswer,
    ClarificationRequest,
)


EliminationInput: TypeAlias = CandidateEliminationRule | Elimination

RESOLUTION_PRECEDENCE: tuple[ResolutionLevel, ...] = (
    ResolutionLevel.STRICT_POLICY,
    ResolutionLevel.EXPLICIT_USER,
    ResolutionLevel.PREFERENCE,
    ResolutionLevel.HEURISTIC,
    ResolutionLevel.CONTEXT,
    ResolutionLevel.USER_CLARIFICATION,
)

_UPSTREAM_BLOCKING_OUTCOMES = {
    GateOutcome.NEED_READ,
    GateOutcome.UNSUPPORTED_CAPABILITY,
    GateOutcome.UNSUPPORTED_PARAMETER,
    GateOutcome.UNAVAILABLE_EVIDENCE,
    GateOutcome.POLICY_CONFLICT,
}


class DisambiguationResolution(DomainModel):
    """One deterministic outcome, with no text/tool side effects."""

    outcome: GateOutcome
    slot: AmbiguitySlot
    clarification_request: ClarificationRequest | None = None
    reasons: list[GateReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.outcome is GateOutcome.ALLOW and not self.slot.is_resolved:
            raise ValueError("allow resolution requires a chosen candidate")
        if self.outcome is GateOutcome.NEED_USER_DISAMBIGUATION:
            if self.clarification_request is None or self.slot.is_resolved:
                raise ValueError(
                    "user disambiguation requires an unresolved slot and a bound request"
                )
        elif self.clarification_request is not None:
            raise ValueError(
                "clarification request is valid only for need_user_disambiguation"
            )
        return self


class DisambiguationResolver:
    """Apply policy through context before asking one precise user question."""

    def resolve(
        self,
        slot: AmbiguitySlot,
        *,
        eliminations: Sequence[EliminationInput] = (),
        clarification: ClarificationAnswer | None = None,
        clarification_prompt: str | None = None,
        upstream_outcome: GateOutcome | str | None = None,
        upstream_reason: str | None = None,
        sole_candidate_level: ResolutionLevel = ResolutionLevel.CONTEXT,
    ) -> DisambiguationResolution:
        outcome = (
            GateOutcome(upstream_outcome) if upstream_outcome is not None else None
        )
        if outcome in _UPSTREAM_BLOCKING_OUTCOMES:
            return DisambiguationResolution(
                outcome=outcome,
                slot=slot,
                reasons=[
                    GateReason(
                        code=f"ambiguity.upstream.{outcome.value}",
                        message=upstream_reason
                        or "An upstream capability or evidence gate blocks resolution.",
                        goal_ids=[slot.goal_id] if slot.goal_id else [],
                    )
                ],
            )
        if outcome not in {None, GateOutcome.ALLOW}:
            assert outcome is not None
            raise ValueError(
                f"unsupported upstream outcome for disambiguation: {outcome.value}"
            )

        if slot.is_resolved:
            return self._resolved(slot)

        candidate_order = [candidate.candidate_id for candidate in slot.candidates]
        if not candidate_order:
            return self._conflict(slot, "No semantically valid candidate exists.")
        if len(candidate_order) == 1:
            return self._resolved(
                slot.with_choice(candidate_order[0], sole_candidate_level)
            )

        facts = self._normalise_eliminations(slot, eliminations)
        applied: list[Elimination] = []
        remaining = set(candidate_order)

        for level in RESOLUTION_PRECEDENCE[:-1]:
            by_candidate: dict[str, Elimination] = {}
            for fact in facts:
                if fact.eliminated_by is level:
                    by_candidate.setdefault(fact.candidate_id, fact)

            for candidate_id in candidate_order:
                selected_fact = by_candidate.get(candidate_id)
                if selected_fact is not None and candidate_id in remaining:
                    applied.append(selected_fact)
                    remaining.remove(candidate_id)

            working = slot.model_copy(update={"eliminated": applied}, deep=True)
            if not remaining:
                return self._conflict(
                    working,
                    f"{level.value} constraints eliminate every candidate.",
                )
            if len(remaining) == 1:
                chosen_id = next(
                    candidate_id
                    for candidate_id in candidate_order
                    if candidate_id in remaining
                )
                return self._resolved(working.with_choice(chosen_id, level))

        working = slot.model_copy(update={"eliminated": applied}, deep=True)
        request = ClarificationRequest.for_slot(
            working,
            prompt=clarification_prompt,
        )
        if clarification is None:
            return self._ask(working, request)

        if (
            clarification.request_id != request.request_id
            or clarification.slot_name != request.slot_name
        ):
            return self._ask(
                working,
                request,
                reason=(
                    "The clarification answer is bound to another or stale "
                    "ambiguity slot."
                ),
                code="ambiguity.clarification_binding_mismatch",
            )

        if clarification.candidate_id not in remaining:
            return self._ask(
                working,
                request,
                reason="The selected option is no longer a valid candidate.",
                code="ambiguity.clarification_candidate_invalid",
            )

        for candidate_id in candidate_order:
            if candidate_id in remaining and candidate_id != clarification.candidate_id:
                applied.append(
                    Elimination(
                        candidate_id=candidate_id,
                        eliminated_by=ResolutionLevel.USER_CLARIFICATION,
                        reason=(
                            "User selected another option in the clarification "
                            f"bound to {request.request_id}."
                        ),
                        rule_id=f"clarification:{request.request_id}",
                    )
                )
        clarified = slot.model_copy(update={"eliminated": applied}, deep=True)
        return self._resolved(
            clarified.with_choice(
                clarification.candidate_id,
                ResolutionLevel.USER_CLARIFICATION,
            )
        )

    def resolve_slots(
        self,
        slots: Sequence[AmbiguitySlot],
        *,
        eliminations_by_slot: Mapping[str, Sequence[EliminationInput]] | None = None,
        clarifications_by_slot: Mapping[str, ClarificationAnswer] | None = None,
        upstream_outcomes: Mapping[str, GateOutcome | str] | None = None,
    ) -> list[DisambiguationResolution]:
        """Resolve independent slots in caller order without sharing answers."""

        rules = eliminations_by_slot or {}
        answers = clarifications_by_slot or {}
        outcomes = upstream_outcomes or {}
        return [
            self.resolve(
                slot,
                eliminations=rules.get(slot.name, ()),
                clarification=answers.get(slot.name),
                upstream_outcome=outcomes.get(slot.name),
            )
            for slot in slots
        ]

    def _normalise_eliminations(
        self,
        slot: AmbiguitySlot,
        inputs: Sequence[EliminationInput],
    ) -> list[Elimination]:
        known = {candidate.candidate_id for candidate in slot.candidates}
        facts = list(slot.eliminated)
        for item in inputs:
            if isinstance(item, CandidateEliminationRule):
                facts.extend(item.to_eliminations(slot))
            else:
                facts.append(item)

        for fact in facts:
            if fact.candidate_id not in known:
                raise ValueError(
                    f"elimination references unknown candidate: {fact.candidate_id}"
                )
            if fact.eliminated_by is ResolutionLevel.USER_CLARIFICATION:
                raise ValueError(
                    "user clarification must be supplied as a bound ClarificationAnswer"
                )
        return facts

    def _resolved(self, slot: AmbiguitySlot) -> DisambiguationResolution:
        assert slot.chosen is not None
        return DisambiguationResolution(
            outcome=GateOutcome.ALLOW,
            slot=slot,
            reasons=[
                GateReason(
                    code="ambiguity.resolved",
                    message=(
                        f"Slot {slot.name!r} resolved at "
                        f"{slot.chosen_by.value if slot.chosen_by else 'unknown'} precedence."
                    ),
                    goal_ids=[slot.goal_id] if slot.goal_id else [],
                    details={"candidate_id": slot.chosen.candidate_id},
                )
            ],
        )

    def _conflict(
        self,
        slot: AmbiguitySlot,
        message: str,
    ) -> DisambiguationResolution:
        return DisambiguationResolution(
            outcome=GateOutcome.POLICY_CONFLICT,
            slot=slot,
            reasons=[
                GateReason(
                    code="ambiguity.no_candidate",
                    message=message,
                    goal_ids=[slot.goal_id] if slot.goal_id else [],
                )
            ],
        )

    def _ask(
        self,
        slot: AmbiguitySlot,
        request: ClarificationRequest,
        *,
        reason: str = "Multiple candidates remain after policy, intent, and context.",
        code: str = "ambiguity.user_clarification_required",
    ) -> DisambiguationResolution:
        return DisambiguationResolution(
            outcome=GateOutcome.NEED_USER_DISAMBIGUATION,
            slot=slot,
            clarification_request=request,
            reasons=[
                GateReason(
                    code=code,
                    message=reason,
                    goal_ids=[slot.goal_id] if slot.goal_id else [],
                    details={"request_id": request.request_id},
                )
            ],
        )


__all__ = [
    "DisambiguationResolution",
    "DisambiguationResolver",
    "EliminationInput",
    "RESOLUTION_PRECEDENCE",
]
