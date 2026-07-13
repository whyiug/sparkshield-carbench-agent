"""Pre-SET proof that a goal's evidence-triggered closure is expressible."""

from __future__ import annotations

from pydantic import Field

from ..domain import GateOutcome, Goal
from ..domain.types import DomainModel, NonEmptyStr
from .conditional_closure import CapabilityClosure
from .live_tool_registry import LiveToolRegistry
from .semantic_binder import LiveBinding, SemanticCapabilityBinder


class FeasibilityProof(DomainModel):
    goal_id: NonEmptyStr
    feasible: bool
    outcome: GateOutcome
    bindings: list[LiveBinding] = Field(default_factory=list)
    checked_operations: list[NonEmptyStr] = Field(default_factory=list)
    reasons: list[NonEmptyStr] = Field(default_factory=list)

    @property
    def first_set_allowed(self) -> bool:
        return self.feasible and self.outcome is GateOutcome.ALLOW


def prove_goal_feasibility(
    goal: Goal,
    closure: CapabilityClosure,
    live_tools: LiveToolRegistry,
    *,
    binder: SemanticCapabilityBinder | None = None,
) -> FeasibilityProof:
    """Purely prove the final action and every triggered prerequisite."""

    if closure.goal_id != goal.goal_id:
        raise ValueError("capability closure belongs to a different goal")
    semantic_binder = binder or SemanticCapabilityBinder()
    bindings: list[LiveBinding] = []
    checked: list[str] = []

    for requirement in closure.requirements:
        operation = requirement.operation
        checked.append(operation.semantic_operation)
        binding = semantic_binder.bind_operation(
            operation,
            semantic_values=requirement.semantic_values,
            recipe_id=requirement.recipe_id,
            live_tools=live_tools,
            is_read=False,
        )
        if binding is None:
            return FeasibilityProof(
                goal_id=goal.goal_id,
                feasible=False,
                outcome=GateOutcome.UNSUPPORTED_CAPABILITY,
                bindings=bindings,
                checked_operations=checked,
                reasons=[
                    f"The goal-required operation '{operation.semantic_operation}' "
                    "cannot be expressed by a current live callable."
                ],
            )
        bindings.append(binding)
        if binding.unsupported_parameters:
            return FeasibilityProof(
                goal_id=goal.goal_id,
                feasible=False,
                outcome=GateOutcome.UNSUPPORTED_PARAMETER,
                bindings=bindings,
                checked_operations=checked,
                reasons=[
                    f"The live binding for '{operation.semantic_operation}' does not "
                    "support its recipe-defined parameter mapping."
                ],
            )
        if binding.unbound_parameters:
            return FeasibilityProof(
                goal_id=goal.goal_id,
                feasible=False,
                outcome=GateOutcome.UNSUPPORTED_PARAMETER,
                bindings=bindings,
                checked_operations=checked,
                reasons=[
                    f"Required parameters for '{operation.semantic_operation}' lack "
                    "goal-conditioned values."
                ],
            )
        if binding.argument_error is not None:
            return FeasibilityProof(
                goal_id=goal.goal_id,
                feasible=False,
                outcome=GateOutcome.UNSUPPORTED_PARAMETER,
                bindings=bindings,
                checked_operations=checked,
                reasons=[
                    f"Arguments for '{operation.semantic_operation}' do not satisfy "
                    "the current live schema."
                ],
            )

    if closure.unresolved_evidence:
        return FeasibilityProof(
            goal_id=goal.goal_id,
            feasible=False,
            outcome=GateOutcome.NEED_READ,
            bindings=bindings,
            checked_operations=checked,
            reasons=[
                "Goal-required policy evidence must be read before the first SET."
            ],
        )
    if not closure.requirements:
        return FeasibilityProof(
            goal_id=goal.goal_id,
            feasible=False,
            outcome=GateOutcome.UNSUPPORTED_CAPABILITY,
            checked_operations=checked,
            reasons=[
                "No invariant recipe operation is defined for this semantic goal."
            ],
        )
    return FeasibilityProof(
        goal_id=goal.goal_id,
        feasible=True,
        outcome=GateOutcome.ALLOW,
        bindings=bindings,
        checked_operations=checked,
        reasons=["Final and evidence-triggered operations are live-schema feasible."],
    )


__all__ = ["FeasibilityProof", "prove_goal_feasibility"]
