"""Structured general critic with a deterministic single-replan boundary."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator
from typing_extensions import Annotated

from .critic_input import CriticInput


_GENERAL_CRITIC_PROMPT = """You review one candidate action for a reliable in-car
assistant. Use only the supplied policy, visible dialogue, semantic goals, referenced
live schemas, validated known evidence, unresolved choices, and candidate action.
Assess only general intent alignment, policy safety, unresolved ambiguity, and missing
declared dependencies. Interface binding and evidence availability have already been
checked deterministically; do not speculate about other interfaces or values.

Return exactly one structured decision:
- accept: the candidate is supported as written;
- ask: user clarification is required;
- decline: policy or user intent rules out the candidate;
- replan_once: the candidate can be corrected using only the supplied information.

Use only an allowed reason code. For accept, omit the reason code and reason. For every
other decision, provide a short reason about intent, policy, ambiguity, dependency, or
plan structure. Do not introduce new tools, parameters, evidence, goals, or facts."""


NonEmptyReason = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)
]


class CriticDecision(str, Enum):
    ACCEPT = "accept"
    ASK = "ask"
    DECLINE = "decline"
    REPLAN_ONCE = "replan_once"


class CriticReasonCode(str, Enum):
    INTENT_MISMATCH = "intent_mismatch"
    POLICY_RISK = "policy_risk"
    UNRESOLVED_AMBIGUITY = "unresolved_ambiguity"
    DEPENDENCY_GAP = "dependency_gap"
    PLAN_STRUCTURE = "plan_structure"
    REPLAN_LIMIT_REACHED = "replan_limit_reached"


class CriticVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: CriticDecision
    reason_code: CriticReasonCode | None = None
    reason: NonEmptyReason | None = None

    @model_validator(mode="after")
    def validate_reason(self) -> CriticVerdict:
        if self.decision is CriticDecision.ACCEPT:
            if self.reason_code is not None or self.reason is not None:
                raise ValueError("accept cannot include a reason")
        elif self.reason_code is None or self.reason is None:
            raise ValueError("non-accept decisions require a code and reason")
        return self

    @property
    def requires_revision(self) -> bool:
        return self.decision is CriticDecision.REPLAN_ONCE


class CriticClient(Protocol):
    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        critic: bool = False,
    ) -> Any: ...


class GeneralCritic:
    """Review one safe DTO and deterministically cap revision at one attempt."""

    def __init__(self, client: CriticClient) -> None:
        self.client = client

    def review(
        self, view: CriticInput, *, replan_already_used: bool = False
    ) -> CriticVerdict:
        prompt = _GENERAL_CRITIC_PROMPT
        if replan_already_used:
            prompt += (
                "\nA revision has already been used for this decision. "
                "Do not request another revision."
            )
        response = self.client.generate(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": view.model_dump_json()},
            ],
            response_model=CriticVerdict,
            critic=True,
        )
        payload = response.value if hasattr(response, "value") else response
        verdict = CriticVerdict.model_validate(payload)
        if replan_already_used and verdict.decision is CriticDecision.REPLAN_ONCE:
            return CriticVerdict(
                decision=CriticDecision.DECLINE,
                reason_code=CriticReasonCode.REPLAN_LIMIT_REACHED,
                reason="A revised candidate has already used the allowed revision.",
            )
        return verdict


__all__ = [
    "CriticClient",
    "CriticDecision",
    "CriticReasonCode",
    "CriticVerdict",
    "GeneralCritic",
]
