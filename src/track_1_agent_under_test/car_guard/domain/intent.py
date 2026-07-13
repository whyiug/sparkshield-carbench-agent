"""Inventory-blind intent extraction output models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Self

from pydantic import Field, model_validator

from .goals import Goal
from .types import AmbiguitySlot, Confirmation, DomainModel, NonEmptyStr


class IntentKind(str, Enum):
    INFORMATION = "information"
    ACTION = "action"
    CONFIRMATION = "confirmation"
    SELECTION = "selection"
    CONVERSATION = "conversation"


class IntentFrame(DomainModel):
    """Frozen semantic interpretation produced before live tool binding."""

    language: NonEmptyStr
    call_for_action: bool
    goals: list[Goal] = Field(default_factory=list)
    explicit_slots: dict[NonEmptyStr, Any] = Field(default_factory=dict)
    explicit_constraints: dict[NonEmptyStr, Any] = Field(default_factory=dict)
    explicit_confirmations: list[Confirmation] = Field(default_factory=list)
    unresolved_slots: list[AmbiguitySlot] = Field(default_factory=list)
    intent_source_turn_ids: list[NonEmptyStr] = Field(default_factory=list)
    goal_mention_order: list[NonEmptyStr] = Field(default_factory=list)
    intent_kind: IntentKind = IntentKind.CONVERSATION

    @model_validator(mode="after")
    def validate_goal_order(self) -> Self:
        goal_ids = [goal.goal_id for goal in self.goals]
        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError("intent goals must have unique IDs")
        if not self.goal_mention_order:
            object.__setattr__(self, "goal_mention_order", goal_ids)
        elif len(self.goal_mention_order) != len(set(self.goal_mention_order)) or set(
            self.goal_mention_order
        ) != set(goal_ids):
            raise ValueError("goal_mention_order must contain every goal exactly once")

        slot_keys = [(slot.goal_id, slot.name) for slot in self.unresolved_slots]
        if len(slot_keys) != len(set(slot_keys)):
            raise ValueError(
                "unresolved ambiguity slots must be unique per goal and name"
            )
        return self

    @property
    def semantic_goals(self) -> list[Goal]:
        return self.goals


class IntentState(IntentFrame):
    """Session-level intent state; structurally compatible with IntentFrame."""


__all__ = ["IntentFrame", "IntentKind", "IntentState"]
