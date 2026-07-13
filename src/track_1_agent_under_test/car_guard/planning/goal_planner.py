"""Stable Goal DAG updates that preserve completed and blocked local state."""

from __future__ import annotations

from ..domain import Goal, GoalDAG, GoalStatus, IntentFrame


class GoalPlanner:
    def update(self, previous: GoalDAG | None, intent: IntentFrame) -> GoalDAG:
        if previous is None:
            return GoalDAG(goals=[goal.model_copy(deep=True) for goal in intent.goals])

        existing_by_id = {goal.goal_id: goal for goal in previous.goals}
        merged: list[Goal] = []
        current_ids = {goal.goal_id for goal in intent.goals}
        for goal in intent.goals:
            # Goal IDs include the authorizing user turn. A repeated semantic
            # request on a later turn is a new action, not an already-DONE goal.
            old = existing_by_id.get(goal.goal_id)
            if old is None:
                merged.append(goal.model_copy(deep=True))
                continue
            dependencies = [
                dependency
                for dependency in goal.depends_on
                if dependency in current_ids
            ]
            merged.append(
                goal.model_copy(
                    update={
                        "status": old.status,
                        "block_reason": old.block_reason,
                        "depends_on": dependencies,
                    },
                    deep=True,
                )
            )

        # A follow-up confirmation or selection can omit the original goals.
        if not merged and intent.intent_kind.value in {"confirmation", "selection"}:
            merged = [
                goal.model_copy(deep=True)
                for goal in previous.goals
                if goal.status not in {GoalStatus.DONE, GoalStatus.FAILED}
            ]
        return GoalDAG(goals=merged)
