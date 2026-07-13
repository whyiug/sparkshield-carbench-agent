"""Deterministic semantic-to-live binding for an already-frozen goal."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import Field, field_serializer

from ..domain import Evidence, EvidenceStore, Goal
from ..domain.types import DomainModel, NonEmptyStr
from ..recipes import OperationRecipe, OperationSpec
from .live_tool_registry import LiveToolRegistry, ToolArgumentValidationError


def _enum_identifier(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "_".join(re.findall(r"[^\W_]+", normalized, re.UNICODE))


def _normalize_live_enum(value: Any, domain: Mapping[str, Any] | None) -> Any:
    """Map one semantic string to a unique spelling-only live enum match."""

    if not isinstance(value, str) or domain is None:
        return value
    candidates = domain.get("enum")
    if not isinstance(candidates, list):
        return value
    identifier = _enum_identifier(value)
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, str) and _enum_identifier(candidate) == identifier
    ]
    return matches[0] if len(matches) == 1 else value


class LiveBinding(DomainModel):
    """One goal-local binding derived from an invariant recipe mapping."""

    recipe_id: NonEmptyStr
    semantic_operation: NonEmptyStr
    tool_name: NonEmptyStr
    parameter_mapping: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    arguments: dict[str, Any] = Field(default_factory=dict)
    required_parameters: set[NonEmptyStr] = Field(default_factory=set)
    unbound_parameters: list[NonEmptyStr] = Field(default_factory=list)
    unsupported_parameters: list[NonEmptyStr] = Field(default_factory=list)
    argument_error: str | None = None
    is_read: bool = False

    @field_serializer("required_parameters", when_used="json")
    def serialize_required_parameters(self, value: set[str]) -> list[str]:
        return sorted(value)

    @property
    def is_complete(self) -> bool:
        return not (
            self.unbound_parameters
            or self.unsupported_parameters
            or self.argument_error is not None
        )


class SemanticCapabilityBinder:
    """Bind only requested semantic operations against named live candidates."""

    def bind(
        self,
        goal: Goal,
        relevant_recipes: Iterable[OperationRecipe],
        live_tools: LiveToolRegistry,
    ) -> list[LiveBinding]:
        """Return stable bindings without enumerating or classifying the inventory."""

        bindings: list[LiveBinding] = []
        for recipe in relevant_recipes:
            if not recipe.matches_semantic_operation(goal.semantic_operation):
                continue
            target = recipe.target_operation(goal.semantic_operation)
            if target is None:
                continue

            seen: set[str] = set()
            for operation in (*recipe.read_operations, target):
                if operation.semantic_operation in seen:
                    continue
                seen.add(operation.semantic_operation)
                binding = self.bind_operation(
                    operation,
                    semantic_values=goal.desired_outcome,
                    recipe_id=recipe.id,
                    live_tools=live_tools,
                    is_read=operation in recipe.read_operations,
                )
                if binding is not None:
                    bindings.append(binding)
        return bindings

    def bind_operation(
        self,
        operation: OperationSpec,
        *,
        semantic_values: Mapping[str, Any],
        recipe_id: str,
        live_tools: LiveToolRegistry,
        is_read: bool = False,
    ) -> LiveBinding | None:
        """Choose candidates by recipe order, never by live inventory order."""

        incomplete: LiveBinding | None = None
        for tool_name in operation.tool_names:
            if not live_tools.has_tool(tool_name):
                continue
            required = live_tools.required_parameters(tool_name)
            unsupported = sorted(
                {
                    official_name
                    for official_name in operation.parameter_mapping.values()
                    if live_tools.parameter_domain(tool_name, official_name) is None
                }
                | {
                    official_name
                    for official_name in operation.fixed_arguments
                    if live_tools.parameter_domain(tool_name, official_name) is None
                }
            )
            arguments = dict(operation.fixed_arguments)
            unbound: list[str] = []
            for semantic_name in operation.required_semantic_parameters:
                official_name = operation.parameter_mapping[semantic_name]
                if semantic_name not in semantic_values:
                    unbound.append(official_name)
                elif official_name not in unsupported:
                    arguments[official_name] = _normalize_live_enum(
                        semantic_values[semantic_name],
                        live_tools.parameter_domain(tool_name, official_name),
                    )
            for semantic_name, official_name in operation.parameter_mapping.items():
                if (
                    semantic_name in semantic_values
                    and official_name not in arguments
                    and official_name not in unsupported
                ):
                    arguments[official_name] = _normalize_live_enum(
                        semantic_values[semantic_name],
                        live_tools.parameter_domain(tool_name, official_name),
                    )
            unbound.extend(
                sorted(required.difference(arguments).difference(unsupported))
            )
            unbound = list(dict.fromkeys(unbound))

            argument_error: str | None = None
            if not unsupported and not unbound:
                try:
                    arguments = live_tools.validate_arguments(tool_name, arguments)
                except ToolArgumentValidationError as exc:
                    argument_error = str(exc)
            binding = LiveBinding(
                recipe_id=recipe_id,
                semantic_operation=operation.semantic_operation,
                tool_name=tool_name,
                parameter_mapping=operation.parameter_mapping,
                arguments=arguments,
                required_parameters=set(required),
                unbound_parameters=unbound,
                unsupported_parameters=unsupported,
                argument_error=argument_error,
                is_read=is_read,
            )
            if binding.is_complete:
                return binding
            if incomplete is None:
                incomplete = binding
        return incomplete

    def instantiate_triggered_closure(
        self,
        goal: Goal,
        evidence: EvidenceStore | Iterable[Evidence] | Mapping[str, Any],
        relevant_recipes: Iterable[OperationRecipe],
    ):
        """Convenience delegate preserving the three-stage public API."""

        from .conditional_closure import instantiate_triggered_closure

        return instantiate_triggered_closure(goal, evidence, relevant_recipes)

    def prove_goal_feasibility(self, goal: Goal, closure, live_tools: LiveToolRegistry):
        """Convenience delegate preserving the three-stage public API."""

        from .feasibility import prove_goal_feasibility

        return prove_goal_feasibility(goal, closure, live_tools, binder=self)


__all__ = ["LiveBinding", "SemanticCapabilityBinder"]
