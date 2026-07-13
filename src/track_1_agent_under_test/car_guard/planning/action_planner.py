"""Structured one-step planner over frozen intent and current live evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel

from ..domain import DecisionProposal, GoalDAG, ProposedToolCall
from ..llm.prompts import ACTION_SYSTEM_PROMPT


class ActionClient(Protocol):
    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        critic: bool = False,
    ) -> Any: ...


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, Enum):
        return value.value
    return value


class ActionPlanner:
    def __init__(self, client: ActionClient) -> None:
        self.client = client

    def propose_next(
        self,
        *,
        policy: Any,
        conversation: list[dict[str, Any]],
        live_tools: Any,
        live_bindings: Any,
        conditional_closure: Any,
        goal_dag: GoalDAG,
        evidence: Any,
        ambiguities: Any,
        relevant_recipes: Any,
        budget: Any,
        call_id_factory: Callable[[], str] | None = None,
    ) -> DecisionProposal:
        payload = {
            "policy": _jsonable(policy),
            "conversation": _jsonable(conversation),
            "live_tools": _jsonable(
                live_tools.as_definitions()
                if hasattr(live_tools, "as_definitions")
                else live_tools
            ),
            "live_bindings": _jsonable(live_bindings),
            "conditional_closure": _jsonable(conditional_closure),
            "semantic_goals": _jsonable(goal_dag),
            "evidence": _jsonable(evidence),
            "ambiguities": _jsonable(ambiguities),
            "relevant_recipes": _jsonable(relevant_recipes),
            "budget": _jsonable(budget),
        }
        response = self.client.generate(
            messages=[
                {"role": "system", "content": ACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                },
            ],
            response_model=DecisionProposal,
        )
        proposal = response.value if hasattr(response, "value") else response
        return self._assign_call_ids(proposal, call_id_factory=call_id_factory)

    @staticmethod
    def _assign_call_ids(
        proposal: DecisionProposal,
        *,
        call_id_factory: Callable[[], str] | None = None,
    ) -> DecisionProposal:
        calls: list[ProposedToolCall] = []
        for index, call in enumerate(proposal.tool_calls):
            encoded = json.dumps(
                {
                    "index": index,
                    "tool": call.tool_name,
                    "arguments": call.arguments,
                    "goals": proposal.goal_ids,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
            calls.append(
                call.model_copy(
                    update={
                        "call_id": (
                            call_id_factory()
                            if call_id_factory is not None
                            else f"agent-{hashlib.sha256(encoded).hexdigest()[:16]}"
                        )
                    }
                )
            )
        return proposal.model_copy(update={"tool_calls": calls})
