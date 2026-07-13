"""Narrow, typed runtime view supplied to the optional general critic.

The critic never receives raw session dictionaries.  This module projects only
the pieces needed to review the current candidate and rejects candidates that
have not already passed live-schema binding or evidence validation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated

from ..capability.live_tool_registry import LiveToolRegistry
from ..domain import (
    AmbiguitySlot,
    DecisionProposal,
    Evidence,
    EvidenceStatus,
    EvidenceStore,
    Goal,
    ProposalKind,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
JsonObject: TypeAlias = dict[str, JsonValue]


class CriticInputRejected(ValueError):
    """The deterministic runtime boundary is not safe to send to a critic."""


class _CriticDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConversationRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class CriticConversationMessage(_CriticDTO):
    role: ConversationRole
    content: NonEmptyText
    turn_id: NonEmptyText | None = None


class JsonSchemaType(str, Enum):
    ARRAY = "array"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NULL = "null"
    NUMBER = "number"
    OBJECT = "object"
    STRING = "string"


class CriticSchemaNode(_CriticDTO):
    """Allowlisted summary of one live JSON Schema node."""

    json_types: tuple[JsonSchemaType, ...] = ()
    description: str | None = None
    enum: tuple[JsonValue, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    properties: tuple[CriticToolParameter, ...] = ()
    items: CriticSchemaNode | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> CriticSchemaNode:
        if len(self.json_types) != len(set(self.json_types)):
            raise ValueError("schema types must be unique")
        if self.minimum is not None and not math.isfinite(self.minimum):
            raise ValueError("schema minimum must be finite")
        if self.maximum is not None and not math.isfinite(self.maximum):
            raise ValueError("schema maximum must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("schema minimum exceeds maximum")
        names = [item.name for item in self.properties]
        if len(names) != len(set(names)):
            raise ValueError("schema property names must be unique")
        return self


class CriticToolParameter(CriticSchemaNode):
    name: NonEmptyText
    required: bool = False


class CriticToolSchema(_CriticDTO):
    name: NonEmptyText
    description: str | None = None
    parameters: tuple[CriticToolParameter, ...] = ()


class CriticGoal(_CriticDTO):
    goal_id: NonEmptyText
    semantic_operation: NonEmptyText
    desired_outcome: JsonObject = Field(default_factory=dict)
    depends_on: tuple[NonEmptyText, ...] = ()


class CriticCandidateCall(_CriticDTO):
    """Only the fields that could be emitted to the tool boundary."""

    tool_name: NonEmptyText
    arguments: JsonObject = Field(default_factory=dict)
    goal_id: NonEmptyText | None = None


class CriticCandidateAction(_CriticDTO):
    kind: ProposalKind
    goal_ids: tuple[NonEmptyText, ...] = ()
    tool_calls: tuple[CriticCandidateCall, ...] = ()
    user_text: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> CriticCandidateAction:
        has_calls = bool(self.tool_calls)
        has_text = self.user_text is not None
        if has_calls == has_text:
            raise ValueError("candidate must contain calls or text, never both")
        tool_kind = self.kind in {ProposalKind.TOOL_GET, ProposalKind.TOOL_SET}
        if tool_kind != has_calls:
            raise ValueError("candidate kind does not match its public output")
        return self


class CriticKnownEvidence(_CriticDTO):
    """Evidence whose KNOWN status has already been proven by the builder."""

    evidence_id: NonEmptyText
    proposition: NonEmptyText
    value: JsonValue
    source: Literal["system", "user", "tool", "preference", "derived"]
    confidence: float = Field(ge=0.0, le=1.0)


class CriticAmbiguityCandidate(_CriticDTO):
    candidate_id: NonEmptyText
    value: JsonValue
    label: str | None = None


class CriticUnresolvedAmbiguity(_CriticDTO):
    name: NonEmptyText
    goal_id: NonEmptyText | None = None
    candidates: tuple[CriticAmbiguityCandidate, ...]

    @field_validator("candidates")
    @classmethod
    def require_multiple_candidates(
        cls, candidates: tuple[CriticAmbiguityCandidate, ...]
    ) -> tuple[CriticAmbiguityCandidate, ...]:
        if len(candidates) < 2:
            raise ValueError("unresolved ambiguity requires multiple candidates")
        ids = [candidate.candidate_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("ambiguity candidate IDs must be unique")
        return candidates


class CriticInput(_CriticDTO):
    """Complete, allowlisted critic payload; raw runtime objects are excluded."""

    policy: NonEmptyText
    conversation: tuple[CriticConversationMessage, ...] = ()
    live_tools: tuple[CriticToolSchema, ...] = ()
    semantic_goals: tuple[CriticGoal, ...]
    candidate_action: CriticCandidateAction
    known_evidence: tuple[CriticKnownEvidence, ...] = ()
    unresolved_ambiguity: tuple[CriticUnresolvedAmbiguity, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> CriticInput:
        ordered_goal_ids = [goal.goal_id for goal in self.semantic_goals]
        if len(ordered_goal_ids) != len(set(ordered_goal_ids)):
            raise ValueError("semantic goal IDs must be unique")
        goal_ids = set(ordered_goal_ids)
        if any(
            not set(goal.depends_on).issubset(goal_ids) for goal in self.semantic_goals
        ):
            raise ValueError("goal dependency is outside the critic view")
        if not set(self.candidate_action.goal_ids).issubset(goal_ids):
            raise ValueError("candidate references a goal outside the critic view")
        call_goal_ids = {
            call.goal_id
            for call in self.candidate_action.tool_calls
            if call.goal_id is not None
        }
        if not call_goal_ids.issubset(set(self.candidate_action.goal_ids)):
            raise ValueError("candidate call references an unrelated goal")
        tool_names = {tool.name for tool in self.live_tools}
        ordered_call_names = [
            call.tool_name for call in self.candidate_action.tool_calls
        ]
        if len(ordered_call_names) != len(set(ordered_call_names)):
            raise ValueError("candidate tool names must be unique")
        call_names = set(ordered_call_names)
        if call_names != tool_names:
            raise ValueError("live tool summaries must exactly match candidate calls")
        if any(
            ambiguity.goal_id is not None and ambiguity.goal_id not in goal_ids
            for ambiguity in self.unresolved_ambiguity
        ):
            raise ValueError("ambiguity references a goal outside the critic view")
        return self


def build_critic_input(
    *,
    policy: str,
    conversation: Iterable[Mapping[str, Any]],
    live_tools: LiveToolRegistry | Iterable[Mapping[str, Any]],
    semantic_goals: Iterable[Goal],
    candidate_action: DecisionProposal,
    evidence: EvidenceStore | Iterable[Evidence],
    unresolved_ambiguity: Iterable[AmbiguitySlot] = (),
) -> CriticInput:
    """Project validated runtime state into the narrow model-facing DTO.

    A missing live binding, malformed argument, conflicting observation, or
    non-JSON value fails closed here.  Those conditions belong to deterministic
    routing and must never be handed to the model for speculation.
    """

    registry = (
        live_tools
        if isinstance(live_tools, LiveToolRegistry)
        else LiveToolRegistry(live_tools)
    )
    try:
        candidate = _candidate_view(candidate_action, registry)
        goals = _goal_views(semantic_goals, candidate_action.goal_ids)
        return CriticInput(
            policy=policy,
            conversation=_conversation_views(conversation),
            live_tools=_live_tool_views(registry, candidate),
            semantic_goals=goals,
            candidate_action=candidate,
            known_evidence=_known_evidence_views(evidence),
            unresolved_ambiguity=_ambiguity_views(
                unresolved_ambiguity, {goal.goal_id for goal in goals}
            ),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, CriticInputRejected):
            raise
        raise CriticInputRejected(
            f"critic input rejected: {type(exc).__name__}"
        ) from None


def _conversation_views(
    conversation: Iterable[Mapping[str, Any]],
) -> tuple[CriticConversationMessage, ...]:
    visible: list[CriticConversationMessage] = []
    for item in conversation:
        role = item.get("role")
        content = item.get("content")
        # Raw tool/environment turns are represented only by validated evidence.
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        turn_id = item.get("turn_id")
        visible.append(
            CriticConversationMessage(
                role=role,
                content=content,
                turn_id=str(turn_id) if turn_id is not None else None,
            )
        )
    return tuple(visible)


def _candidate_view(
    proposal: DecisionProposal, registry: LiveToolRegistry
) -> CriticCandidateAction:
    calls: list[CriticCandidateCall] = []
    for call in proposal.tool_calls:
        arguments = registry.validate_arguments(call.tool_name, call.arguments)
        calls.append(
            CriticCandidateCall(
                tool_name=call.tool_name,
                arguments=arguments,
                goal_id=call.goal_id,
            )
        )
    return CriticCandidateAction(
        kind=proposal.kind,
        goal_ids=tuple(proposal.goal_ids),
        tool_calls=tuple(calls),
        user_text=proposal.user_text,
    )


def _goal_views(
    goals: Iterable[Goal], candidate_goal_ids: Iterable[str]
) -> tuple[CriticGoal, ...]:
    by_id = {goal.goal_id: goal for goal in goals}
    wanted = set(candidate_goal_ids)
    frontier = list(wanted)
    while frontier:
        goal_id = frontier.pop()
        goal = by_id.get(goal_id)
        if goal is None:
            raise CriticInputRejected("candidate goal is not present")
        for dependency in goal.depends_on:
            if dependency not in wanted:
                wanted.add(dependency)
                frontier.append(dependency)
    return tuple(
        CriticGoal(
            goal_id=goal.goal_id,
            semantic_operation=goal.semantic_operation,
            desired_outcome=goal.desired_outcome,
            depends_on=tuple(goal.depends_on),
        )
        for goal in by_id.values()
        if goal.goal_id in wanted
    )


def _live_tool_views(
    registry: LiveToolRegistry, candidate: CriticCandidateAction
) -> tuple[CriticToolSchema, ...]:
    seen: set[str] = set()
    summaries: list[CriticToolSchema] = []
    for call in candidate.tool_calls:
        if call.tool_name in seen:
            continue
        seen.add(call.tool_name)
        definition = registry.definition(call.tool_name)
        function = definition.get("function", definition)
        parameters = function.get("parameters", {})
        required = set(parameters.get("required", ()))
        properties = parameters.get("properties", {})
        summaries.append(
            CriticToolSchema(
                name=call.tool_name,
                description=_optional_text(function.get("description")),
                parameters=tuple(
                    _parameter_view(name, schema, required=name in required)
                    for name, schema in properties.items()
                ),
            )
        )
    return tuple(summaries)


def _parameter_view(
    name: str, schema: Mapping[str, Any], *, required: bool
) -> CriticToolParameter:
    node = _schema_node_fields(schema)
    return CriticToolParameter(name=name, required=required, **node)


def _schema_node_fields(schema: Mapping[str, Any]) -> dict[str, Any]:
    declared = schema.get("type")
    declared_types: tuple[JsonSchemaType, ...]
    if isinstance(declared, str):
        declared_types = (JsonSchemaType(declared),)
    elif isinstance(declared, list):
        declared_types = tuple(JsonSchemaType(item) for item in declared)
    else:
        declared_types = ()
    required = set(schema.get("required", ()))
    properties = schema.get("properties", {})
    items = schema.get("items")
    enum = schema.get("enum", ())
    return {
        "json_types": declared_types,
        "description": _optional_text(schema.get("description")),
        "enum": tuple(enum) if isinstance(enum, list) else (),
        "minimum": _finite_number(schema.get("minimum")),
        "maximum": _finite_number(schema.get("maximum")),
        "properties": tuple(
            _parameter_view(name, child, required=name in required)
            for name, child in properties.items()
        ),
        "items": CriticSchemaNode(**_schema_node_fields(items))
        if isinstance(items, Mapping)
        else None,
    }


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CriticInputRejected("schema bound is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise CriticInputRejected("schema bound is not finite")
    return numeric


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _known_evidence_views(
    evidence: EvidenceStore | Iterable[Evidence],
) -> tuple[CriticKnownEvidence, ...]:
    if isinstance(evidence, EvidenceStore):
        selected: list[Evidence] = []
        propositions = dict.fromkeys(
            observation.proposition for observation in evidence.evidence.values()
        )
        for proposition in propositions:
            latest = evidence.latest_for_proposition(
                proposition, current_state_only=True
            )
            known = [item for item in latest if item.status is EvidenceStatus.KNOWN]
            if not known:
                continue
            selected.extend(known)
    else:
        selected = [
            observation
            for observation in evidence
            if observation.status is EvidenceStatus.KNOWN
        ]

    views: list[CriticKnownEvidence] = []
    for observation in selected:
        if observation.evidence_id is None:
            raise CriticInputRejected("known evidence has no stable ID")
        views.append(
            CriticKnownEvidence(
                evidence_id=observation.evidence_id,
                proposition=observation.proposition,
                value=observation.value,
                source=observation.source_kind.value,
                confidence=observation.confidence,
            )
        )
    return tuple(views)


def _ambiguity_views(
    ambiguities: Iterable[AmbiguitySlot], visible_goal_ids: set[str]
) -> tuple[CriticUnresolvedAmbiguity, ...]:
    views: list[CriticUnresolvedAmbiguity] = []
    for slot in ambiguities:
        if slot.is_resolved or (
            slot.goal_id is not None and slot.goal_id not in visible_goal_ids
        ):
            continue
        remaining = slot.remaining_candidates
        if len(remaining) < 2:
            continue
        views.append(
            CriticUnresolvedAmbiguity(
                name=slot.name,
                goal_id=slot.goal_id,
                candidates=tuple(
                    CriticAmbiguityCandidate(
                        candidate_id=candidate.candidate_id,
                        value=candidate.value,
                        label=candidate.label,
                    )
                    for candidate in remaining
                ),
            )
        )
    return tuple(views)


__all__ = [
    "ConversationRole",
    "CriticAmbiguityCandidate",
    "CriticCandidateAction",
    "CriticCandidateCall",
    "CriticConversationMessage",
    "CriticGoal",
    "CriticInput",
    "CriticInputRejected",
    "CriticKnownEvidence",
    "CriticSchemaNode",
    "CriticToolParameter",
    "CriticToolSchema",
    "CriticUnresolvedAmbiguity",
    "JsonSchemaType",
    "build_critic_input",
]
