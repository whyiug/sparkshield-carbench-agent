"""Pure, named derivations from known evidence values.

Derivations never invent identifiers or silently recover from missing inputs.
Every produced observation contains the exact input evidence IDs and the
registered pure-function name required by the domain provenance contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from ..domain.evidence import (
    Evidence,
    EvidenceSourceKind,
    EvidenceStatus,
)


DerivationFunction: TypeAlias = Callable[[tuple[Any, ...]], Any]


class InsufficientDerivationEvidence(ValueError):
    """Raised when a pure derivation does not have concrete, traceable inputs."""


@dataclass(frozen=True, slots=True)
class DerivationSpec:
    name: str
    function: DerivationFunction

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("derivation name cannot be blank")
        if not callable(self.function):
            raise TypeError("derivation function must be callable")


def _validate_inputs(inputs: Sequence[Evidence]) -> tuple[Evidence, ...]:
    observations = tuple(inputs)
    if not observations:
        raise InsufficientDerivationEvidence(
            "deterministic derivation requires at least one input evidence item"
        )

    evidence_ids: list[str] = []
    for observation in observations:
        if observation.status is not EvidenceStatus.KNOWN:
            raise InsufficientDerivationEvidence(
                f"input evidence is not known: {observation.evidence_id}"
            )
        if observation.evidence_id is None:
            raise InsufficientDerivationEvidence(
                "input evidence must have a stable evidence_id"
            )
        evidence_ids.append(observation.evidence_id)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise InsufficientDerivationEvidence(
            "input evidence IDs must be unique and preserve one provenance edge each"
        )
    return observations


def derive_evidence(
    *,
    proposition: str,
    source_turn_id: str,
    inputs: Sequence[Evidence],
    derivation_name: str,
    function: DerivationFunction,
    confidence: float | None = None,
) -> Evidence:
    """Apply a pure value function and attach complete derivation provenance."""

    if not derivation_name.strip():
        raise ValueError("derivation name cannot be blank")
    observations = _validate_inputs(inputs)
    values = tuple(observation.value for observation in observations)
    value = function(values)
    if value is None or (
        isinstance(value, str)
        and (not value.strip() or value.strip().lower() == "unknown")
    ):
        raise InsufficientDerivationEvidence(
            "derivation did not produce a concrete value"
        )

    derived_confidence = (
        min(observation.confidence for observation in observations)
        if confidence is None
        else confidence
    )
    return Evidence(
        proposition=proposition,
        value=value,
        status=EvidenceStatus.KNOWN,
        source_kind=EvidenceSourceKind.DERIVED,
        source_turn_id=source_turn_id,
        source_tool_call_id=None,
        confidence=derived_confidence,
        derived_from=[
            observation.evidence_id
            for observation in observations
            if observation.evidence_id is not None
        ],
        derivation=derivation_name,
    )


class DeterministicDerivationRegistry:
    """Small immutable-by-name registry for explicitly configured derivations."""

    def __init__(
        self,
        derivations: Mapping[str, DerivationFunction]
        | Sequence[DerivationSpec]
        | None = None,
    ) -> None:
        self._functions: dict[str, DerivationFunction] = {}
        if isinstance(derivations, Mapping):
            for name, function in derivations.items():
                self.register(name, function)
        else:
            for spec in derivations or ():
                self.register(spec.name, spec.function)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._functions)

    def register(self, name: str, function: DerivationFunction) -> None:
        spec = DerivationSpec(name=name, function=function)
        if spec.name in self._functions:
            raise ValueError(f"derivation is already registered: {spec.name}")
        self._functions[spec.name] = spec.function

    def derive(
        self,
        name: str,
        *,
        proposition: str,
        source_turn_id: str,
        inputs: Sequence[Evidence],
        confidence: float | None = None,
    ) -> Evidence:
        try:
            function = self._functions[name]
        except KeyError as exc:
            raise KeyError(f"unknown deterministic derivation: {name}") from exc
        return derive_evidence(
            proposition=proposition,
            source_turn_id=source_turn_id,
            inputs=inputs,
            derivation_name=name,
            function=function,
            confidence=confidence,
        )


__all__ = [
    "DerivationFunction",
    "DerivationSpec",
    "DeterministicDerivationRegistry",
    "InsufficientDerivationEvidence",
    "derive_evidence",
]
