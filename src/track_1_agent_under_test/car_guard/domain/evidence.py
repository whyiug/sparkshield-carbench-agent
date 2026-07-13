"""Goal-conditioned evidence requirements and provenance store."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from enum import Enum
from typing import Any, Self

from pydantic import Field, field_serializer, model_validator

from .types import DomainModel, NonEmptyStr


class EvidenceCardinality(str, Enum):
    ONE = "one"
    UNIQUE = "unique"
    ALL = "all"


class EvidenceStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    ERROR = "error"
    CONFLICT = "conflict"
    STALE = "stale"


class EvidenceSourceKind(str, Enum):
    SYSTEM = "system"
    USER = "user"
    TOOL = "tool"
    PREFERENCE = "preference"
    DERIVED = "derived"


SOURCE_KINDS_BY_ACCEPTABLE_SOURCE: dict[str, set[EvidenceSourceKind]] = {
    "system_context": {EvidenceSourceKind.SYSTEM},
    "user_explicit": {EvidenceSourceKind.USER},
    "tool_result": {EvidenceSourceKind.TOOL},
    "preference": {EvidenceSourceKind.PREFERENCE},
    "derived": {EvidenceSourceKind.DERIVED},
}


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:20]}"


class EvidenceNeed(DomainModel):
    need_id: NonEmptyStr | None = None
    proposition: NonEmptyStr
    required_for_goal_id: NonEmptyStr
    acceptable_sources: set[NonEmptyStr]
    freshness: NonEmptyStr | None = None
    cardinality: EvidenceCardinality = EvidenceCardinality.ONE
    required_before_set: bool = True

    @model_validator(mode="after")
    def validate_need(self) -> Self:
        if not self.acceptable_sources:
            raise ValueError("evidence need requires at least one acceptable source")
        unsupported = set(self.acceptable_sources).difference(
            SOURCE_KINDS_BY_ACCEPTABLE_SOURCE
        )
        if unsupported:
            raise ValueError(f"unsupported evidence sources: {sorted(unsupported)}")
        if self.need_id is None:
            object.__setattr__(
                self,
                "need_id",
                _stable_id(
                    "need",
                    {
                        "proposition": self.proposition,
                        "required_for_goal_id": self.required_for_goal_id,
                        "acceptable_sources": sorted(self.acceptable_sources),
                        "freshness": self.freshness,
                        "cardinality": self.cardinality.value,
                        "required_before_set": self.required_before_set,
                    },
                ),
            )
        return self

    @field_serializer("acceptable_sources", when_used="json")
    def serialize_sources(self, sources: set[str]) -> list[str]:
        return sorted(sources)


class Evidence(DomainModel):
    evidence_id: NonEmptyStr | None = None
    proposition: NonEmptyStr
    value: Any | None = None
    status: EvidenceStatus
    source_kind: EvidenceSourceKind
    source_turn_id: NonEmptyStr
    source_tool_call_id: NonEmptyStr | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    observation_index: int = Field(default=0, ge=0)
    state_version: int = Field(default=0, ge=0)
    derived_from: list[NonEmptyStr] = Field(default_factory=list)
    derivation: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.status is EvidenceStatus.KNOWN:
            unavailable = self.value is None or (
                isinstance(self.value, str)
                and (not self.value.strip() or self.value.strip().lower() == "unknown")
            )
            if unavailable:
                raise ValueError("known evidence requires a concrete value")

        if self.source_kind is EvidenceSourceKind.DERIVED:
            if not self.derived_from or self.derivation is None:
                raise ValueError(
                    "derived evidence requires input evidence IDs and a derivation"
                )
            if self.source_tool_call_id is not None:
                raise ValueError(
                    "derived evidence cannot claim a tool call as its source"
                )
        elif self.derived_from or self.derivation is not None:
            raise ValueError("derivation provenance is valid only for derived evidence")

        if (
            self.source_kind is EvidenceSourceKind.TOOL
            and self.source_tool_call_id is None
        ):
            raise ValueError("tool evidence requires a source_tool_call_id")
        if (
            self.source_kind is not EvidenceSourceKind.TOOL
            and self.source_tool_call_id is not None
        ):
            raise ValueError("only tool evidence can claim a source_tool_call_id")

        if self.evidence_id is None:
            object.__setattr__(
                self,
                "evidence_id",
                _stable_id(
                    "evidence",
                    {
                        "proposition": self.proposition,
                        "value": self.value,
                        "status": self.status.value,
                        "source_kind": self.source_kind.value,
                        "source_turn_id": self.source_turn_id,
                        "source_tool_call_id": self.source_tool_call_id,
                        "observation_index": self.observation_index,
                        "state_version": self.state_version,
                        "derived_from": self.derived_from,
                        "derivation": self.derivation,
                    },
                ),
            )
        return self


class EvidenceStore(DomainModel):
    """Structured index of declared needs and observations for one session."""

    needs: dict[str, EvidenceNeed] = Field(default_factory=dict)
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    next_observation_index: int = Field(default=1, ge=1)
    current_state_version: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_keys(self) -> Self:
        for key, need in self.needs.items():
            if key != need.need_id:
                raise ValueError("evidence need dictionary key must equal need_id")
        for key, observation in self.evidence.items():
            if key != observation.evidence_id:
                raise ValueError("evidence dictionary key must equal evidence_id")
        return self

    def register_need(self, need: EvidenceNeed) -> None:
        assert need.need_id is not None
        existing = self.needs.get(need.need_id)
        if existing is not None and existing != need:
            raise ValueError(f"conflicting evidence need ID: {need.need_id}")
        updated = dict(self.needs)
        updated[need.need_id] = need
        self.needs = updated

    def add(self, observation: Evidence) -> None:
        if observation.observation_index == 0:
            payload = observation.model_dump()
            payload.update(
                evidence_id=None,
                observation_index=self.next_observation_index,
                state_version=self.current_state_version,
            )
            observation = Evidence.model_validate(payload)
            self.next_observation_index += 1
        assert observation.evidence_id is not None
        existing = self.evidence.get(observation.evidence_id)
        if existing is not None and existing != observation:
            raise ValueError(f"conflicting evidence ID: {observation.evidence_id}")
        updated = dict(self.evidence)
        updated[observation.evidence_id] = observation
        self.evidence = updated

    def update(self, observations: Evidence | Iterable[Evidence]) -> None:
        if isinstance(observations, Evidence):
            self.add(observations)
            return
        for observation in observations:
            self.add(observation)

    def for_proposition(
        self, proposition: str, *, current_state_only: bool = False
    ) -> list[Evidence]:
        observations = [
            observation
            for observation in self.evidence.values()
            if observation.proposition == proposition
        ]
        if not current_state_only:
            return observations
        return [
            observation
            for observation in observations
            if observation.source_kind
            not in {EvidenceSourceKind.TOOL, EvidenceSourceKind.DERIVED}
            or observation.state_version == self.current_state_version
        ]

    def latest_for_proposition(
        self, proposition: str, *, current_state_only: bool = False
    ) -> list[Evidence]:
        observations = self.for_proposition(proposition)
        if current_state_only:
            observations = [
                item
                for item in observations
                if item.source_kind
                not in {EvidenceSourceKind.TOOL, EvidenceSourceKind.DERIVED}
                or item.state_version == self.current_state_version
            ]
        if not observations:
            return []
        latest = max(observations, key=lambda item: item.observation_index)
        latest_group = [
            item
            for item in observations
            if item.source_turn_id == latest.source_turn_id
            and item.state_version == latest.state_version
        ]
        if latest.status is not EvidenceStatus.KNOWN:
            return latest_group
        known = [
            item
            for item in observations
            if item.status is EvidenceStatus.KNOWN
            and item.state_version == latest.state_version
        ]
        canonical_values = {
            json.dumps(item.value, sort_keys=True, default=str) for item in known
        }
        return known if len(canonical_values) > 1 else latest_group

    def invalidate_tool_state(self) -> int:
        self.current_state_version += 1
        return self.current_state_version

    def satisfying_evidence(self, need: EvidenceNeed | str) -> list[Evidence]:
        requirement = self.needs[need] if isinstance(need, str) else need
        allowed_kinds: set[EvidenceSourceKind] = set()
        for source in requirement.acceptable_sources:
            allowed_kinds.update(SOURCE_KINDS_BY_ACCEPTABLE_SOURCE[source])
        related = self.latest_for_proposition(
            requirement.proposition,
            current_state_only=requirement.freshness == "current_state",
        )
        return [
            observation
            for observation in related
            if observation.status is EvidenceStatus.KNOWN
            and observation.source_kind in allowed_kinds
        ]

    def is_satisfied(self, need: EvidenceNeed | str) -> bool:
        requirement = self.needs[need] if isinstance(need, str) else need
        related = self.latest_for_proposition(
            requirement.proposition,
            current_state_only=requirement.freshness == "current_state",
        )
        if not related:
            return False
        if any(
            observation.status is not EvidenceStatus.KNOWN for observation in related
        ):
            return False

        satisfying = self.satisfying_evidence(requirement)
        if not satisfying:
            return False
        if requirement.cardinality is EvidenceCardinality.UNIQUE:
            canonical_values = {
                json.dumps(
                    observation.value,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                for observation in satisfying
            }
            return len(canonical_values) == 1
        if requirement.cardinality is EvidenceCardinality.ALL:
            return len(satisfying) == len(related)
        return True

    @property
    def pending_needs(self) -> list[EvidenceNeed]:
        return [need for need in self.needs.values() if not self.is_satisfied(need)]


__all__ = [
    "Evidence",
    "EvidenceCardinality",
    "EvidenceNeed",
    "EvidenceSourceKind",
    "EvidenceStatus",
    "EvidenceStore",
]
