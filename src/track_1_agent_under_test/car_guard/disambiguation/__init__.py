"""Deterministic candidate-elimination API."""

from .candidates import (
    CandidateEliminationRule,
    ClarificationAnswer,
    ClarificationOption,
    ClarificationRequest,
)
from .resolver import (
    RESOLUTION_PRECEDENCE,
    DisambiguationResolution,
    DisambiguationResolver,
    EliminationInput,
)

__all__ = [
    "CandidateEliminationRule",
    "ClarificationAnswer",
    "ClarificationOption",
    "ClarificationRequest",
    "DisambiguationResolution",
    "DisambiguationResolver",
    "EliminationInput",
    "RESOLUTION_PRECEDENCE",
]
