"""Stage-A semantic intent extraction without live capability input."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain import (
    AmbiguitySlot,
    Candidate,
    Goal,
    GoalSource,
    IntentFrame,
    IntentKind,
)
from ..llm.prompts import INTENT_SYSTEM_PROMPT
from .intent_grounding import (
    focus_explicit_action_request,
    has_explicit_action_request,
    semantic_value_is_explicit,
)


_FOCUSED_RETRY_SUFFIX = (
    "\nThis is a focused correction pass. Classify only the current user "
    "utterance against the invariant semantic contracts. Every desired value "
    "and explicit slot must be literally present in that utterance. Omit any "
    "missing parameter instead of choosing a default, likely value, or example."
)


class IntentClient(Protocol):
    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        critic: bool = False,
    ) -> Any: ...


class DraftGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_operation: str
    desired_outcome: dict[str, Any] = Field(default_factory=dict)
    depends_on_indices: list[int] = Field(default_factory=list)
    atomic_group: str | None = None


class DraftAmbiguity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    candidate_values: list[Any]
    candidate_labels: list[str] = Field(default_factory=list)
    goal_index: int | None = None

    @model_validator(mode="after")
    def labels_match(self) -> "DraftAmbiguity":
        if self.candidate_labels and len(self.candidate_labels) != len(
            self.candidate_values
        ):
            raise ValueError("candidate labels must match candidate values")
        return self


class IntentDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = "en"
    intent_kind: IntentKind
    call_for_action: bool
    goals: list[DraftGoal] = Field(default_factory=list)
    explicit_slots: dict[str, Any] = Field(default_factory=dict)
    explicit_constraints: dict[str, Any] = Field(default_factory=dict)
    ambiguities: list[DraftAmbiguity] = Field(default_factory=list)


def _stable_goal_id(turn_id: str, index: int, goal: DraftGoal) -> str:
    payload = json.dumps(
        {
            "turn": turn_id,
            "index": index,
            "operation": goal.semantic_operation,
            "outcome": goal.desired_outcome,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return f"goal-{hashlib.sha256(payload).hexdigest()[:12]}"


def _conversation_payload(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "turn_id": str(item.get("turn_id", index)),
            "role": item.get("role"),
            "content": item.get("content"),
        }
        for index, item in enumerate(conversation)
        if item.get("role") in {"user", "assistant", "system", "tool", "environment"}
    ]


class IntentExtractor:
    """The absence of a ``live_tools`` argument is an enforced design boundary."""

    def __init__(
        self,
        client: IntentClient,
        *,
        semantic_contracts: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.client = client
        self.semantic_contracts = tuple(dict(item) for item in semantic_contracts)

    def extract(
        self,
        *,
        system_policy: str,
        conversation: list[dict[str, Any]],
    ) -> IntentFrame:
        visible_conversation = _conversation_payload(conversation)
        last_user = next(
            (
                item
                for item in reversed(visible_conversation)
                if item.get("role") == "user"
            ),
            {"turn_id": "0"},
        )
        current_user_text = last_user.get("content")
        focused_user_text = (
            focus_explicit_action_request(current_user_text)
            if isinstance(current_user_text, str)
            else None
        )
        full_messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "system_policy": system_policy,
                        "conversation": visible_conversation,
                        "invariant_semantic_contracts": self.semantic_contracts,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]
        focused_messages = (
            [
                {
                    "role": "system",
                    "content": INTENT_SYSTEM_PROMPT + _FOCUSED_RETRY_SUFFIX,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_user_utterance": focused_user_text,
                            "invariant_semantic_contracts": self.semantic_contracts,
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ]
            if isinstance(current_user_text, str)
            and has_explicit_action_request(current_user_text)
            and focused_user_text
            else None
        )
        response = self.client.generate(
            messages=focused_messages or full_messages,
            response_model=IntentDraft,
        )
        draft = response.value if hasattr(response, "value") else response
        if focused_messages is not None and not draft.goals:
            fallback_response = self.client.generate(
                messages=full_messages,
                response_model=IntentDraft,
            )
            draft = (
                fallback_response.value
                if hasattr(fallback_response, "value")
                else fallback_response
            )
        if focused_messages is not None:
            pruned_goals = [
                goal.model_copy(
                    update={
                        "desired_outcome": {
                            key: value
                            for key, value in goal.desired_outcome.items()
                            if semantic_value_is_explicit(
                                key, value, focused_user_text or ""
                            )
                        }
                    }
                )
                for goal in draft.goals
            ]
            explicit_slots = {
                key: value
                for key, value in draft.explicit_slots.items()
                if semantic_value_is_explicit(key, value, focused_user_text or "")
            }
            draft = draft.model_copy(
                update={"goals": pruned_goals, "explicit_slots": explicit_slots}
            )
        turn_id = str(last_user.get("turn_id", "0"))
        goal_ids = [
            _stable_goal_id(turn_id, index, goal)
            for index, goal in enumerate(draft.goals)
        ]
        goals: list[Goal] = []
        for index, goal in enumerate(draft.goals):
            dependencies = [
                goal_ids[dependency]
                for dependency in goal.depends_on_indices
                if 0 <= dependency < len(goal_ids) and dependency != index
            ]
            goals.append(
                Goal(
                    goal_id=goal_ids[index],
                    semantic_operation=goal.semantic_operation,
                    desired_outcome=goal.desired_outcome,
                    depends_on=dependencies,
                    atomic_group=goal.atomic_group,
                    source=GoalSource.USER,
                )
            )

        ambiguity_slots: list[AmbiguitySlot] = []
        for ambiguity in draft.ambiguities:
            labels = ambiguity.candidate_labels or [None] * len(
                ambiguity.candidate_values
            )
            candidates = [
                Candidate(
                    candidate_id=f"{ambiguity.name}-{index}",
                    value=value,
                    label=labels[index],
                )
                for index, value in enumerate(ambiguity.candidate_values)
            ]
            goal_id = (
                goal_ids[ambiguity.goal_index]
                if ambiguity.goal_index is not None
                and 0 <= ambiguity.goal_index < len(goal_ids)
                else None
            )
            ambiguity_slots.append(
                AmbiguitySlot(
                    name=ambiguity.name,
                    candidates=candidates,
                    goal_id=goal_id,
                    source_turn_ids=[turn_id],
                )
            )

        return IntentFrame(
            language=draft.language,
            call_for_action=draft.call_for_action,
            goals=goals,
            explicit_slots=draft.explicit_slots,
            explicit_constraints=draft.explicit_constraints,
            unresolved_slots=ambiguity_slots,
            intent_source_turn_ids=[turn_id],
            goal_mention_order=goal_ids,
            intent_kind=draft.intent_kind,
        )
