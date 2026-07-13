"""Auditable inputs and user-binding records for candidate elimination."""

from __future__ import annotations

import hashlib
import json
from typing import Self

from pydantic import Field, model_validator

from ..domain.types import (
    AmbiguitySlot,
    DomainModel,
    Elimination,
    NonEmptyStr,
    ResolutionLevel,
)


class CandidateEliminationRule(DomainModel):
    """A fact that removes named candidates at exactly one precedence level."""

    level: ResolutionLevel
    eliminate_candidate_ids: list[NonEmptyStr]
    reason: NonEmptyStr
    evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    rule_id: NonEmptyStr | None = None
    slot_name: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_rule(self) -> Self:
        if self.level is ResolutionLevel.USER_CLARIFICATION:
            raise ValueError(
                "user_clarification eliminations require a bound clarification answer"
            )
        if not self.eliminate_candidate_ids:
            raise ValueError("an elimination rule must name at least one candidate")
        if len(self.eliminate_candidate_ids) != len(set(self.eliminate_candidate_ids)):
            raise ValueError("elimination rule candidate IDs must be unique")
        return self

    def to_eliminations(self, slot: AmbiguitySlot) -> list[Elimination]:
        if self.slot_name is not None and self.slot_name != slot.name:
            raise ValueError(
                f"elimination rule for slot {self.slot_name!r} cannot apply to {slot.name!r}"
            )
        known = {candidate.candidate_id for candidate in slot.candidates}
        unknown = set(self.eliminate_candidate_ids).difference(known)
        if unknown:
            raise ValueError(
                f"elimination rule references unknown candidates: {sorted(unknown)}"
            )
        requested = set(self.eliminate_candidate_ids)
        return [
            Elimination(
                candidate_id=candidate.candidate_id,
                eliminated_by=self.level,
                reason=self.reason,
                evidence_ids=list(self.evidence_ids),
                rule_id=self.rule_id,
            )
            for candidate in slot.candidates
            if candidate.candidate_id in requested
        ]


class ClarificationOption(DomainModel):
    candidate_id: NonEmptyStr
    label: NonEmptyStr


class ClarificationRequest(DomainModel):
    """A minimal question bound to one exact slot and candidate snapshot."""

    request_id: NonEmptyStr
    slot_name: NonEmptyStr
    goal_id: NonEmptyStr | None = None
    options: list[ClarificationOption]
    prompt: NonEmptyStr

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        candidate_ids = [option.candidate_id for option in self.options]
        if len(candidate_ids) < 2:
            raise ValueError("clarification requires at least two candidate options")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("clarification candidate IDs must be unique")
        return self

    @classmethod
    def for_slot(
        cls,
        slot: AmbiguitySlot,
        *,
        prompt: str | None = None,
    ) -> ClarificationRequest:
        remaining = slot.remaining_candidates
        if len(remaining) < 2:
            raise ValueError("only unresolved slots can produce clarification requests")
        options = [
            ClarificationOption(
                candidate_id=candidate.candidate_id,
                label=(candidate.label or str(candidate.value)).strip()
                or candidate.candidate_id,
            )
            for candidate in remaining
        ]
        payload = {
            "slot_name": slot.name,
            "goal_id": slot.goal_id,
            "candidate_ids": [option.candidate_id for option in options],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        request_id = f"clarify-{hashlib.sha256(encoded).hexdigest()[:20]}"
        if prompt is None:
            labels = [option.label for option in options]
            if len(labels) == 2:
                natural_options = f"{labels[0]} or {labels[1]}"
            else:
                natural_options = f"{', '.join(labels[:-1])}, or {labels[-1]}"
            prompt = f"Which {slot.name} did you mean: {natural_options}?"
        return cls(
            request_id=request_id,
            slot_name=slot.name,
            goal_id=slot.goal_id,
            options=options,
            prompt=prompt,
        )


class ClarificationAnswer(DomainModel):
    """A parsed user selection that cannot float to another pending slot."""

    request_id: NonEmptyStr
    slot_name: NonEmptyStr
    candidate_id: NonEmptyStr
    source_turn_id: NonEmptyStr


__all__ = [
    "CandidateEliminationRule",
    "ClarificationAnswer",
    "ClarificationOption",
    "ClarificationRequest",
]
