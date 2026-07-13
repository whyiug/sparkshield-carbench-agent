"""Goal and Goal DAG state models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Self

from pydantic import Field, model_validator

from .types import DomainModel, NonEmptyStr


class GoalStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    DONE = "done"
    BLOCKED = "blocked"
    FAILED = "failed"


class GoalSource(str, Enum):
    USER = "user"
    POLICY = "policy"


class Goal(DomainModel):
    goal_id: NonEmptyStr
    semantic_operation: NonEmptyStr
    desired_outcome: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[NonEmptyStr] = Field(default_factory=list)
    atomic_group: NonEmptyStr | None = None
    status: GoalStatus = GoalStatus.PENDING
    block_reason: str | None = None
    source: GoalSource = GoalSource.USER

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.goal_id in self.depends_on:
            raise ValueError("a goal cannot depend on itself")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("goal dependencies must be unique")
        if self.status in {GoalStatus.BLOCKED, GoalStatus.FAILED}:
            if self.block_reason is None or not self.block_reason.strip():
                raise ValueError("blocked or failed goals require a block_reason")
        elif self.block_reason is not None:
            raise ValueError("block_reason is valid only for blocked or failed goals")
        return self


class GoalDAG(DomainModel):
    """A stable-order acyclic graph with local status propagation helpers."""

    goals: list[Goal] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        goal_ids = [goal.goal_id for goal in self.goals]
        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError("goal_id values must be unique")

        known = set(goal_ids)
        for goal in self.goals:
            missing = set(goal.depends_on).difference(known)
            if missing:
                raise ValueError(
                    f"goal {goal.goal_id} has unknown dependencies: {sorted(missing)}"
                )

        visiting: set[str] = set()
        visited: set[str] = set()
        graph = {goal.goal_id: goal.depends_on for goal in self.goals}

        def visit(goal_id: str) -> None:
            if goal_id in visiting:
                raise ValueError("goal dependencies must form an acyclic graph")
            if goal_id in visited:
                return
            visiting.add(goal_id)
            for dependency in graph[goal_id]:
                visit(dependency)
            visiting.remove(goal_id)
            visited.add(goal_id)

        for goal_id in goal_ids:
            visit(goal_id)
        return self

    def get(self, goal_id: str) -> Goal:
        for goal in self.goals:
            if goal.goal_id == goal_id:
                return goal
        raise KeyError(goal_id)

    def topological_order(self) -> list[Goal]:
        """Return dependency order while preserving original order for ties."""

        index = {goal.goal_id: position for position, goal in enumerate(self.goals)}
        indegree = {goal.goal_id: len(goal.depends_on) for goal in self.goals}
        dependents: dict[str, list[str]] = {goal.goal_id: [] for goal in self.goals}
        for goal in self.goals:
            for dependency in goal.depends_on:
                dependents[dependency].append(goal.goal_id)

        ready = sorted(
            (goal_id for goal_id, degree in indegree.items() if degree == 0),
            key=index.__getitem__,
        )
        ordered: list[Goal] = []
        while ready:
            goal_id = ready.pop(0)
            ordered.append(self.get(goal_id))
            for dependent in dependents[goal_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
                    ready.sort(key=index.__getitem__)
        return ordered

    @property
    def ready_goals(self) -> list[Goal]:
        done = {goal.goal_id for goal in self.goals if goal.status is GoalStatus.DONE}
        return [
            goal
            for goal in self.topological_order()
            if goal.status in {GoalStatus.PENDING, GoalStatus.READY}
            and set(goal.depends_on).issubset(done)
        ]

    def set_status(
        self,
        goal_id: str,
        status: GoalStatus,
        *,
        block_reason: str | None = None,
    ) -> Goal:
        current = self.get(goal_id)
        updated = current.model_copy(
            update={"status": status, "block_reason": block_reason}
        )
        updated = Goal.model_validate(updated.model_dump())
        self.goals = [
            updated if goal.goal_id == goal_id else goal for goal in self.goals
        ]
        return updated

    def block_dependents(self, goal_id: str, reason: str) -> list[Goal]:
        """Block only transitive dependents, leaving independent goals untouched."""

        to_block: set[str] = set()
        frontier = [goal_id]
        while frontier:
            dependency = frontier.pop()
            for goal in self.goals:
                if dependency in goal.depends_on and goal.goal_id not in to_block:
                    to_block.add(goal.goal_id)
                    frontier.append(goal.goal_id)

        updated: list[Goal] = []
        for target_id in [
            goal.goal_id for goal in self.goals if goal.goal_id in to_block
        ]:
            goal = self.get(target_id)
            if goal.status not in {GoalStatus.DONE, GoalStatus.FAILED}:
                updated.append(
                    self.set_status(target_id, GoalStatus.BLOCKED, block_reason=reason)
                )
        return updated

    def atomic_group(self, atomic_group: str) -> list[Goal]:
        return [goal for goal in self.goals if goal.atomic_group == atomic_group]


__all__ = ["Goal", "GoalDAG", "GoalSource", "GoalStatus"]
