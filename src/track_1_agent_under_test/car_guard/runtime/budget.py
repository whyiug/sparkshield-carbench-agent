"""Bounded step, user-turn, retry, and idempotency accounting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


def call_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"tool_name": tool_name, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class RuntimeBudget:
    soft_max_steps: int = 49
    max_user_turns: int = 12
    max_identical_gets: int = 2
    steps: int = 0
    user_turns: int = 0
    get_counts: dict[str, int] = field(default_factory=dict)
    attempted_sets: set[str] = field(default_factory=set)
    successful_sets: set[str] = field(default_factory=set)

    def consume_step(self) -> None:
        if self.steps >= self.soft_max_steps:
            raise BudgetExceeded("soft step limit reached")
        self.steps += 1

    def consume_user_turn(self) -> None:
        if self.user_turns >= self.max_user_turns:
            raise BudgetExceeded("user turn limit reached")
        self.user_turns += 1

    def allow_call(
        self, tool_name: str, arguments: dict[str, Any], *, state_changing: bool
    ) -> bool:
        fingerprint = call_fingerprint(tool_name, arguments)
        if state_changing:
            return fingerprint not in self.attempted_sets
        return self.get_counts.get(fingerprint, 0) < self.max_identical_gets

    def record_outbound(
        self, tool_name: str, arguments: dict[str, Any], *, state_changing: bool
    ) -> None:
        fingerprint = call_fingerprint(tool_name, arguments)
        if state_changing:
            self.attempted_sets.add(fingerprint)
        else:
            self.get_counts[fingerprint] = self.get_counts.get(fingerprint, 0) + 1

    def record_success(
        self, tool_name: str, arguments: dict[str, Any], *, state_changing: bool
    ) -> None:
        if state_changing:
            fingerprint = call_fingerprint(tool_name, arguments)
            self.attempted_sets.add(fingerprint)
            self.successful_sets.add(fingerprint)

    def reset_get_count(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Start a fresh retry window after a verified intervening state change."""

        self.get_counts.pop(call_fingerprint(tool_name, arguments), None)
