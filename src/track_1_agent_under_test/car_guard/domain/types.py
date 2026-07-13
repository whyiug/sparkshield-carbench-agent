"""Shared domain contracts for planning and pre-commit validation.

These models deliberately contain only runtime state derived from the policy,
conversation, live tools, and observed tool results.  They are also the narrow
boundary between model-generated proposals and deterministic validation code.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DomainModel(BaseModel):
    """Base configuration shared by all CAR-Guard domain models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProposalKind(str, Enum):
    TOOL_GET = "tool_get"
    TOOL_SET = "tool_set"
    ASK_USER = "ask_user"
    RESPOND = "respond"
    COMPLETE = "complete"


class GateOutcome(str, Enum):
    ALLOW = "allow"
    NEED_READ = "need_read"
    NEED_USER_DISAMBIGUATION = "need_user_disambiguation"
    NEED_CONFIRMATION = "need_confirmation"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    UNAVAILABLE_EVIDENCE = "unavailable_evidence"
    POLICY_CONFLICT = "policy_conflict"
    INVALID_PROPOSAL = "invalid_proposal"


class ResolutionLevel(str, Enum):
    STRICT_POLICY = "strict_policy"
    EXPLICIT_USER = "explicit_user"
    PREFERENCE = "preference"
    HEURISTIC = "heuristic"
    CONTEXT = "context"
    USER_CLARIFICATION = "user_clarification"


class ConfirmationStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class Candidate(DomainModel):
    """One semantically valid value for an unresolved intent slot."""

    candidate_id: NonEmptyStr
    value: Any
    label: str | None = None
    evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Elimination(DomainModel):
    """Auditable removal of a candidate at one resolver precedence level."""

    candidate_id: NonEmptyStr
    eliminated_by: ResolutionLevel
    reason: NonEmptyStr
    evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    rule_id: NonEmptyStr | None = None


class AmbiguitySlot(DomainModel):
    """Candidate set and its deterministic disambiguation history."""

    name: NonEmptyStr
    candidates: list[Candidate]
    eliminated: list[Elimination] = Field(default_factory=list)
    chosen: Candidate | None = None
    chosen_by: ResolutionLevel | None = None
    goal_id: NonEmptyStr | None = None
    source_turn_ids: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_resolution(self) -> Self:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("ambiguity candidate_id values must be unique")

        eliminated_ids = [item.candidate_id for item in self.eliminated]
        if len(eliminated_ids) != len(set(eliminated_ids)):
            raise ValueError("a candidate may be eliminated only once")
        unknown = set(eliminated_ids).difference(candidate_ids)
        if unknown:
            raise ValueError(
                f"eliminations reference unknown candidates: {sorted(unknown)}"
            )

        if (self.chosen is None) != (self.chosen_by is None):
            raise ValueError("chosen and chosen_by must be set together")
        if self.chosen is not None:
            if self.chosen.candidate_id not in candidate_ids:
                raise ValueError("chosen candidate must belong to the slot")
            if self.chosen.candidate_id in eliminated_ids:
                raise ValueError("an eliminated candidate cannot be chosen")
        return self

    @property
    def remaining_candidates(self) -> list[Candidate]:
        eliminated_ids = {item.candidate_id for item in self.eliminated}
        return [
            candidate
            for candidate in self.candidates
            if candidate.candidate_id not in eliminated_ids
        ]

    @property
    def is_resolved(self) -> bool:
        return self.chosen is not None

    def with_choice(self, candidate_id: str, chosen_by: ResolutionLevel) -> Self:
        candidate = next(
            (
                item
                for item in self.remaining_candidates
                if item.candidate_id == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise ValueError(f"candidate is absent or eliminated: {candidate_id}")
        return self.model_copy(
            update={"chosen": candidate, "chosen_by": chosen_by},
            deep=True,
        )


class ProposedToolCall(DomainModel):
    """Planner-produced call, including internal provenance annotations."""

    tool_name: NonEmptyStr
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: NonEmptyStr | None = None
    goal_id: NonEmptyStr | None = None
    argument_sources: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    depends_on: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_argument_sources(self) -> Self:
        unknown = set(self.argument_sources).difference(self.arguments)
        if unknown:
            raise ValueError(
                f"argument provenance references absent arguments: {sorted(unknown)}"
            )
        return self


class OfficialToolCall(DomainModel):
    """A schema-normalized call using names from the current live inventory."""

    tool_name: NonEmptyStr
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: NonEmptyStr | None = None

    def to_a2a_payload(self) -> dict[str, Any]:
        """Return only fields accepted by the benchmark tool-call data part."""

        return {
            "tool_name": self.tool_name,
            "arguments": self.model_dump(mode="json")["arguments"],
        }


class DecisionProposal(DomainModel):
    """A single planner step that is structurally safe to pass to the gate."""

    kind: ProposalKind
    goal_ids: list[NonEmptyStr] = Field(default_factory=list)
    tool_calls: list[ProposedToolCall] = Field(default_factory=list)
    user_text: str | None = None
    evidence_used: list[NonEmptyStr] = Field(default_factory=list)
    policy_rules_used: list[NonEmptyStr] = Field(default_factory=list)
    needs_critic: bool = False

    @model_validator(mode="after")
    def enforce_output_mode(self) -> Self:
        has_calls = bool(self.tool_calls)
        has_text = self.user_text is not None and bool(self.user_text.strip())
        if has_calls == has_text:
            raise ValueError(
                "proposal must contain exactly one of tool_calls or user_text"
            )

        tool_kind = self.kind in {ProposalKind.TOOL_GET, ProposalKind.TOOL_SET}
        if tool_kind != has_calls:
            raise ValueError("proposal kind does not match its output mode")
        if self.kind is ProposalKind.TOOL_SET and len(self.tool_calls) != 1:
            raise ValueError("state-changing proposals must contain exactly one call")

        if self.user_text is not None:
            object.__setattr__(self, "user_text", self.user_text.strip())

        call_ids = [
            call.call_id for call in self.tool_calls if call.call_id is not None
        ]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("proposal call_id values must be unique")
        if self.kind is ProposalKind.TOOL_GET:
            names = [call.tool_name for call in self.tool_calls]
            if len(names) != len(set(names)):
                raise ValueError("parallel reads cannot contain duplicate tool names")
        return self


class GateReason(DomainModel):
    """Machine-readable and human-explainable reason emitted by a gate check."""

    code: NonEmptyStr
    message: NonEmptyStr
    goal_ids: list[NonEmptyStr] = Field(default_factory=list)
    tool_call_ids: list[NonEmptyStr] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False


class CommitDecision(DomainModel):
    """Deterministic gate result ready for routing or outbound serialization."""

    outcome: GateOutcome
    normalized_calls: list[OfficialToolCall] = Field(default_factory=list)
    user_text: str | None = None
    reasons: list[GateReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_output(self) -> Self:
        has_calls = bool(self.normalized_calls)
        has_text = self.user_text is not None and bool(self.user_text.strip())
        if has_calls and has_text:
            raise ValueError("commit decision cannot contain calls and text together")
        if self.outcome is GateOutcome.ALLOW and not (has_calls or has_text):
            raise ValueError("allow decision requires calls or user text")
        if self.user_text is not None:
            if not has_text:
                raise ValueError("user_text cannot be blank")
            object.__setattr__(self, "user_text", self.user_text.strip())
        return self

    def to_outbound_payload(self) -> dict[str, Any]:
        if self.normalized_calls:
            return {
                "tool_calls": [call.to_a2a_payload() for call in self.normalized_calls]
            }
        if self.user_text is not None:
            return {"text": self.user_text}
        return {}


class ConfirmationScope(DomainModel):
    """Exact action bundle to which a user confirmation can apply."""

    goal_ids: list[NonEmptyStr]
    ordered_actions: list[OfficialToolCall]
    normalized_arguments: list[dict[str, Any]] = Field(default_factory=list)
    requested_at_user_turn: int = Field(ge=0)
    expires_after_user_turn: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if not self.goal_ids:
            raise ValueError("confirmation scope requires at least one goal")
        if not self.ordered_actions:
            raise ValueError("confirmation scope requires at least one action")
        if self.expires_after_user_turn < self.requested_at_user_turn:
            raise ValueError("confirmation expiration precedes its request")

        expected_arguments = [dict(action.arguments) for action in self.ordered_actions]
        if not self.normalized_arguments:
            object.__setattr__(self, "normalized_arguments", expected_arguments)
        elif self.normalized_arguments != expected_arguments:
            raise ValueError(
                "normalized_arguments must exactly match the ordered action bundle"
            )
        return self

    @property
    def fingerprint(self) -> str:
        payload = {
            "goal_ids": self.goal_ids,
            "ordered_actions": [
                action.to_a2a_payload() for action in self.ordered_actions
            ],
            "normalized_arguments": self.normalized_arguments,
            "requested_at_user_turn": self.requested_at_user_turn,
            "expires_after_user_turn": self.expires_after_user_turn,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def matches(
        self,
        *,
        goal_ids: list[str],
        ordered_actions: list[OfficialToolCall],
        current_user_turn: int,
    ) -> bool:
        if current_user_turn > self.expires_after_user_turn:
            return False
        return (
            goal_ids == self.goal_ids
            and [call.to_a2a_payload() for call in ordered_actions]
            == [call.to_a2a_payload() for call in self.ordered_actions]
            and [call.arguments for call in ordered_actions]
            == self.normalized_arguments
        )


class Confirmation(DomainModel):
    """A user response bound to one exact pending confirmation scope."""

    confirmation_id: NonEmptyStr
    scope: ConfirmationScope
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    source_turn_id: NonEmptyStr | None = None
    user_response: str | None = None
    resolved_at_user_turn: int | None = Field(default=None, ge=0)

    @property
    def is_terminal(self) -> bool:
        return self.status is not ConfirmationStatus.PENDING

    def authorizes(
        self,
        *,
        goal_ids: list[str],
        ordered_actions: list[OfficialToolCall],
        current_user_turn: int,
    ) -> bool:
        return self.status is ConfirmationStatus.CONFIRMED and self.scope.matches(
            goal_ids=goal_ids,
            ordered_actions=ordered_actions,
            current_user_turn=current_user_turn,
        )


__all__ = [
    "AmbiguitySlot",
    "Candidate",
    "CommitDecision",
    "Confirmation",
    "ConfirmationScope",
    "ConfirmationStatus",
    "DecisionProposal",
    "DomainModel",
    "Elimination",
    "GateOutcome",
    "GateReason",
    "NonEmptyStr",
    "OfficialToolCall",
    "ProposalKind",
    "ProposedToolCall",
    "ResolutionLevel",
]
