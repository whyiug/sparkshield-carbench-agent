"""Aggregate measured metrics across planner, critic, and retry calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TurnMetricsAccumulator:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    cost: float = 0.0
    llm_calls: int = 0
    total_llm_time_ms: float = 0.0
    models: list[str] = field(default_factory=list)

    def add(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        thinking_tokens: int = 0,
        cost: float = 0.0,
        elapsed_ms: float = 0.0,
        model: str = "",
    ) -> None:
        self.prompt_tokens += max(0, prompt_tokens)
        self.completion_tokens += max(0, completion_tokens)
        self.thinking_tokens += max(0, thinking_tokens)
        self.cost += max(0.0, cost)
        self.total_llm_time_ms += max(0.0, elapsed_ms)
        self.llm_calls += 1
        if model and model not in self.models:
            self.models.append(model)

    def snapshot(self, *, reset: bool = False) -> dict[str, Any]:
        average = self.total_llm_time_ms / self.llm_calls if self.llm_calls else 0.0
        result = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": self.cost,
            "model": ",".join(self.models),
            "thinking_tokens": self.thinking_tokens,
            "num_llm_calls": self.llm_calls,
            "avg_llm_call_time_ms": round(average, 1),
            "num_passes": 1,
        }
        if reset:
            self.reset()
        return result

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.thinking_tokens = 0
        self.cost = 0.0
        self.llm_calls = 0
        self.total_llm_time_ms = 0.0
        self.models.clear()
