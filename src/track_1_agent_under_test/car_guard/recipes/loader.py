"""Strict loader for the fixed, inventory-independent operation registry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Self

import yaml  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from ..domain import Goal, IntentFrame
from ..domain.types import DomainModel, NonEmptyStr


REQUIRED_RECIPE_DOMAINS = frozenset(
    {
        "calendar",
        "charging_station",
        "climate",
        "contacts",
        "cross_domain",
        "defrost",
        "email",
        "ev_range",
        "exterior_lights",
        "fan",
        "interior_lights",
        "navigation_create",
        "navigation_delete",
        "navigation_edit",
        "phone",
        "poi",
        "route_aware_charging",
        "route_presentation",
        "seat_heating",
        "steering_heating",
        "sunroof",
        "sunshade",
        "trunk",
        "weather",
        "windows",
    }
)


class ConditionOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    TRUTHY = "truthy"
    FALSY = "falsy"


class ConditionSpec(DomainModel):
    """An auditable predicate over one evidence proposition."""

    proposition: NonEmptyStr
    operator: ConditionOperator = ConditionOperator.EQ
    value: Any | None = None

    @model_validator(mode="after")
    def validate_operand(self) -> Self:
        if self.operator in {
            ConditionOperator.IN,
            ConditionOperator.NOT_IN,
        } and not isinstance(self.value, (list, tuple, set, frozenset)):
            raise ValueError("in/not_in conditions require a collection value")
        return self


class OperationSpec(DomainModel):
    """Invariant mapping from a semantic operation to live callable candidates."""

    semantic_operation: NonEmptyStr
    tool_names: list[NonEmptyStr]
    parameter_mapping: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    required_semantic_parameters: list[NonEmptyStr] = Field(default_factory=list)
    fixed_arguments: dict[NonEmptyStr, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_mapping(self) -> Self:
        if not self.tool_names:
            raise ValueError("an operation requires at least one live tool candidate")
        if len(self.tool_names) != len(set(self.tool_names)):
            raise ValueError("operation tool candidates must be unique")
        if len(self.required_semantic_parameters) != len(
            set(self.required_semantic_parameters)
        ):
            raise ValueError("required semantic parameters must be unique")
        unsupported = set(self.required_semantic_parameters).difference(
            self.parameter_mapping
        )
        if unsupported:
            raise ValueError(
                "required semantic parameters need parameter mappings: "
                f"{sorted(unsupported)}"
            )
        collision = set(self.parameter_mapping.values()).intersection(
            self.fixed_arguments
        )
        if collision:
            raise ValueError(
                "mapped and fixed arguments cannot target the same parameter: "
                f"{sorted(collision)}"
            )
        return self


class RecipeEvidenceNeed(DomainModel):
    proposition: NonEmptyStr
    acceptable_sources: set[NonEmptyStr] = Field(
        default_factory=lambda: {"tool_result"}
    )
    read_operation: NonEmptyStr | None = None
    required_before_set: bool = True

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        if not self.acceptable_sources:
            raise ValueError("recipe evidence needs require an acceptable source")
        return self


class ConditionalRequirement(DomainModel):
    """A policy operation instantiated only when observed evidence triggers it."""

    id: NonEmptyStr
    when: ConditionSpec
    operation: OperationSpec
    policy_hook: NonEmptyStr | None = None


class OperationRecipe(DomainModel):
    """One fixed operation workflow; it never contains benchmark task metadata."""

    id: NonEmptyStr
    domain: NonEmptyStr
    intent_patterns: list[NonEmptyStr]
    semantic_preconditions: list[NonEmptyStr] = Field(default_factory=list)
    read_operations: list[OperationSpec] = Field(default_factory=list)
    write_operations: list[OperationSpec] = Field(default_factory=list)
    evidence_needs: list[RecipeEvidenceNeed] = Field(default_factory=list)
    parameter_derivations: list[NonEmptyStr] = Field(default_factory=list)
    policy_hooks: list[NonEmptyStr] = Field(default_factory=list)
    parallel_read_groups: list[list[NonEmptyStr]] = Field(default_factory=list)
    write_order: list[NonEmptyStr] = Field(default_factory=list)
    completion_evidence: list[NonEmptyStr] = Field(default_factory=list)
    conditional_requirements: list[ConditionalRequirement] = Field(default_factory=list)
    primary_operation: NonEmptyStr
    on_unavailable: str = "explain_without_internal_names"
    terminal_after_success: bool = False

    @model_validator(mode="after")
    def validate_recipe(self) -> Self:
        if not self.intent_patterns:
            raise ValueError("a recipe requires at least one intent pattern")
        if len(self.intent_patterns) != len(set(self.intent_patterns)):
            raise ValueError("recipe intent patterns must be unique")

        direct_operations = [*self.read_operations, *self.write_operations]
        direct_names = [operation.semantic_operation for operation in direct_operations]
        if len(direct_names) != len(set(direct_names)):
            raise ValueError(
                "direct semantic operations must be unique within a recipe"
            )
        if self.primary_operation not in direct_names:
            raise ValueError("primary_operation must name a direct operation")

        known_operations = set(direct_names)
        for need in self.evidence_needs:
            if (
                need.read_operation is not None
                and need.read_operation not in known_operations
            ):
                raise ValueError(
                    f"evidence need references unknown read operation: {need.read_operation}"
                )
        for group in self.parallel_read_groups:
            if not group or not set(group).issubset(
                {operation.semantic_operation for operation in self.read_operations}
            ):
                raise ValueError(
                    "parallel read groups must reference direct read operations"
                )
        if not set(self.write_order).issubset(
            {operation.semantic_operation for operation in self.write_operations}
            | {
                branch.operation.semantic_operation
                for branch in self.conditional_requirements
            }
        ):
            raise ValueError("write_order references an unknown write operation")

        branch_ids = [branch.id for branch in self.conditional_requirements]
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("conditional requirement IDs must be unique")
        if self.on_unavailable != "explain_without_internal_names":
            raise ValueError("unsupported on_unavailable behavior")
        return self

    @property
    def direct_operations(self) -> tuple[OperationSpec, ...]:
        return (*self.read_operations, *self.write_operations)

    def matches_semantic_operation(self, semantic_operation: str) -> bool:
        return semantic_operation in {
            self.id,
            *self.intent_patterns,
            *(operation.semantic_operation for operation in self.direct_operations),
        }

    def target_operation(self, semantic_operation: str) -> OperationSpec | None:
        for operation in self.direct_operations:
            if operation.semantic_operation == semantic_operation:
                return operation
        if semantic_operation in {self.id, *self.intent_patterns}:
            return next(
                operation
                for operation in self.direct_operations
                if operation.semantic_operation == self.primary_operation
            )
        return None


class _RecipeDocument(DomainModel):
    schema_version: int = Field(ge=1)
    recipes: list[OperationRecipe]

    @model_validator(mode="after")
    def validate_document(self) -> Self:
        recipe_ids = [recipe.id for recipe in self.recipes]
        if len(recipe_ids) != len(set(recipe_ids)):
            raise ValueError("recipe IDs must be unique")
        return self


class RecipeRegistry:
    """Fully loaded immutable registry with a canonical content hash."""

    DEFAULT_PATH = Path(__file__).with_name("operation_recipes.yaml")

    def __init__(self, path: str | Path | None = None) -> None:
        source = self.DEFAULT_PATH if path is None else Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        document = _RecipeDocument.model_validate(raw)
        if path is None:
            covered = {recipe.domain for recipe in document.recipes}
            absent = REQUIRED_RECIPE_DOMAINS.difference(covered)
            if absent:
                raise ValueError(
                    f"default recipe registry lacks domains: {sorted(absent)}"
                )

        self._recipes = tuple(document.recipes)
        self._by_id = {recipe.id: recipe for recipe in self._recipes}
        canonical = json.dumps(
            document.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        self._registry_hash = hashlib.sha256(canonical).hexdigest()

    @property
    def recipes(self) -> tuple[OperationRecipe, ...]:
        return self._recipes

    @property
    def registry_hash(self) -> str:
        return self._registry_hash

    @property
    def hash(self) -> str:
        return self._registry_hash

    @property
    def covered_domains(self) -> frozenset[str]:
        return frozenset(recipe.domain for recipe in self._recipes)

    def __iter__(self) -> Iterator[OperationRecipe]:
        return iter(self._recipes)

    def __len__(self) -> int:
        return len(self._recipes)

    def get(self, recipe_id: str) -> OperationRecipe:
        try:
            return self._by_id[recipe_id]
        except KeyError as exc:
            raise KeyError(f"unknown recipe: {recipe_id}") from exc

    def select_by_intent_and_policy(
        self,
        intent: IntentFrame | Goal | str | Iterable[Goal | str],
        policy: Iterable[str] | Mapping[str, Any] | object | None = None,
    ) -> tuple[OperationRecipe, ...]:
        """Select in file order without accepting or inspecting a live inventory."""

        operations = _semantic_operations(intent)
        policy_ids = _policy_ids(policy)
        return tuple(
            recipe
            for recipe in self._recipes
            if any(
                recipe.matches_semantic_operation(operation) for operation in operations
            )
            or bool(policy_ids.intersection(recipe.policy_hooks))
        )


def _semantic_operations(
    intent: IntentFrame | Goal | str | Iterable[Goal | str],
) -> frozenset[str]:
    if isinstance(intent, IntentFrame):
        return frozenset(goal.semantic_operation for goal in intent.semantic_goals)
    if isinstance(intent, Goal):
        return frozenset({intent.semantic_operation})
    if isinstance(intent, str):
        return frozenset({intent})
    return frozenset(
        item.semantic_operation if isinstance(item, Goal) else item for item in intent
    )


def _policy_ids(
    policy: Iterable[str] | Mapping[str, Any] | object | None,
) -> frozenset[str]:
    if policy is None:
        return frozenset()
    if isinstance(policy, Mapping):
        value = policy.get("active_rule_ids", policy.get("rule_ids", ()))
    elif isinstance(policy, str):
        value = (policy,)
    elif isinstance(policy, Iterable):
        value = policy
    else:
        value = getattr(policy, "active_rule_ids", getattr(policy, "rule_ids", ()))
    return frozenset(str(item) for item in value)


__all__ = [
    "REQUIRED_RECIPE_DOMAINS",
    "ConditionalRequirement",
    "ConditionOperator",
    "ConditionSpec",
    "OperationRecipe",
    "OperationSpec",
    "RecipeEvidenceNeed",
    "RecipeRegistry",
]
