"""Evidence-triggered conditional capability closure."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import Field

from ..domain import Evidence, EvidenceStatus, EvidenceStore, Goal
from ..domain.types import DomainModel, NonEmptyStr
from ..recipes import (
    ConditionOperator,
    ConditionSpec,
    OperationRecipe,
    OperationSpec,
)


class TriggeredRequirement(DomainModel):
    """A final or evidence-triggered operation that must be live-bindable."""

    recipe_id: NonEmptyStr
    operation: OperationSpec
    semantic_values: dict[str, Any] = Field(default_factory=dict)
    trigger: str
    policy_hook: NonEmptyStr | None = None


class CapabilityClosure(DomainModel):
    """Only concrete requirements for one goal and its triggered policy branches."""

    goal_id: NonEmptyStr
    atomic_group: NonEmptyStr | None = None
    recipe_ids: list[NonEmptyStr] = Field(default_factory=list)
    requirements: list[TriggeredRequirement] = Field(default_factory=list)
    triggered_branch_ids: list[NonEmptyStr] = Field(default_factory=list)
    unresolved_evidence: list[NonEmptyStr] = Field(default_factory=list)

    @property
    def required_operations(self) -> tuple[OperationSpec, ...]:
        return tuple(requirement.operation for requirement in self.requirements)

    @property
    def ready_for_feasibility_proof(self) -> bool:
        return not self.unresolved_evidence


class ConditionalClosure:
    def __init__(self, recipes: Iterable[OperationRecipe] = ()) -> None:
        self._recipes = tuple(recipes)

    def instantiate(
        self,
        goal: Goal,
        evidence: EvidenceStore | Iterable[Evidence] | Mapping[str, Any],
        relevant_recipes: Iterable[OperationRecipe] | None = None,
    ) -> CapabilityClosure:
        """Instantiate branch operations only after known evidence satisfies them."""

        recipes = self._recipes if relevant_recipes is None else tuple(relevant_recipes)
        applicable = tuple(
            recipe
            for recipe in recipes
            if recipe.matches_semantic_operation(goal.semantic_operation)
        )
        requirements: list[TriggeredRequirement] = []
        branch_ids: list[str] = []
        unresolved: list[str] = []

        for recipe in applicable:
            target = recipe.target_operation(goal.semantic_operation)
            if target is None:
                continue
            target_requirement = TriggeredRequirement(
                recipe_id=recipe.id,
                operation=target,
                semantic_values=dict(goal.desired_outcome),
                trigger="goal",
            )

            required_propositions = {
                need.proposition
                for need in recipe.evidence_needs
                if need.required_before_set and target in recipe.write_operations
            }
            recipe_requirements: list[TriggeredRequirement] = []
            for branch in recipe.conditional_requirements:
                known, value = _known_value(evidence, branch.when.proposition)
                if not known:
                    if target in recipe.write_operations:
                        required_propositions.add(branch.when.proposition)
                    continue
                if target in recipe.write_operations and _condition_matches(
                    value, branch.when
                ):
                    branch_ids.append(branch.id)
                    recipe_requirements.append(
                        TriggeredRequirement(
                            recipe_id=recipe.id,
                            operation=branch.operation,
                            semantic_values={},
                            trigger=f"evidence:{branch.when.proposition}",
                            policy_hook=branch.policy_hook,
                        )
                    )
            recipe_requirements.append(target_requirement)
            order = {name: index for index, name in enumerate(recipe.write_order)}
            recipe_requirements.sort(
                key=lambda requirement: order.get(
                    requirement.operation.semantic_operation, len(order)
                )
            )
            requirements.extend(recipe_requirements)
            for proposition in required_propositions:
                known, _ = _known_value(evidence, proposition)
                if not known and proposition not in unresolved:
                    unresolved.append(proposition)

        return CapabilityClosure(
            goal_id=goal.goal_id,
            atomic_group=goal.atomic_group,
            recipe_ids=[recipe.id for recipe in applicable],
            requirements=requirements,
            triggered_branch_ids=branch_ids,
            unresolved_evidence=unresolved,
        )


def instantiate_triggered_closure(
    goal: Goal,
    evidence: EvidenceStore | Iterable[Evidence] | Mapping[str, Any],
    relevant_recipes: Iterable[OperationRecipe],
) -> CapabilityClosure:
    return ConditionalClosure(relevant_recipes).instantiate(goal, evidence)


def _known_value(
    evidence: EvidenceStore | Iterable[Evidence] | Mapping[str, Any], proposition: str
) -> tuple[bool, Any]:
    if isinstance(evidence, EvidenceStore):
        observations = evidence.latest_for_proposition(
            proposition, current_state_only=True
        )
    elif isinstance(evidence, Mapping):
        if proposition not in evidence:
            return False, None
        item = evidence[proposition]
        observations = list(item) if isinstance(item, (list, tuple)) else [item]
    else:
        observations = []
        for item in evidence:
            if not isinstance(item, Evidence):
                raise TypeError("iterable evidence must contain Evidence objects")
            if item.proposition == proposition:
                observations.append(item)

    known_values: list[Any] = []
    for observation in observations:
        if isinstance(observation, Evidence):
            if observation.status is not EvidenceStatus.KNOWN:
                return False, None
            known_values.append(observation.value)
        else:
            known_values.append(observation)
    if not known_values:
        return False, None
    canonical = {
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        for value in known_values
    }
    if len(canonical) != 1:
        return False, None
    return True, known_values[-1]


def _condition_matches(actual: Any, condition: ConditionSpec) -> bool:
    operator = condition.operator
    expected = condition.value
    if operator is ConditionOperator.EQ:
        return type(actual) is type(expected) and actual == expected
    if operator is ConditionOperator.NE:
        return type(actual) is not type(expected) or actual != expected
    if operator is ConditionOperator.TRUTHY:
        return bool(actual)
    if operator is ConditionOperator.FALSY:
        return not bool(actual)
    if operator is ConditionOperator.IN:
        return (
            isinstance(expected, (list, tuple, set, frozenset)) and actual in expected
        )
    if operator is ConditionOperator.NOT_IN:
        return (
            isinstance(expected, (list, tuple, set, frozenset))
            and actual not in expected
        )
    if type(actual) not in {int, float} or type(expected) not in {int, float}:
        return False
    return {
        ConditionOperator.GT: actual > expected,
        ConditionOperator.GTE: actual >= expected,
        ConditionOperator.LT: actual < expected,
        ConditionOperator.LTE: actual <= expected,
    }[operator]


__all__ = [
    "CapabilityClosure",
    "ConditionalClosure",
    "TriggeredRequirement",
    "instantiate_triggered_closure",
]
