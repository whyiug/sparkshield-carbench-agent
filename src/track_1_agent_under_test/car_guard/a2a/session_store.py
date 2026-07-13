"""Per-context state with TTL, LRU eviction, and explicit lifecycle control."""

from __future__ import annotations

import time
import unicodedata
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Callable

from ..domain import EvidenceStore, GoalDAG, OfficialToolCall
from ..observability.turn_metrics import TurnMetricsAccumulator
from ..runtime.budget import RuntimeBudget
from .result_matching import PendingToolCall


def _new_confirmation_latch() -> Any:
    from ..policy.confirmation_latch import ConfirmationLatch

    return ConfirmationLatch()


class SessionLifecycle(str, Enum):
    TERMINAL = "terminal"
    CANCELLED = "cancelled"


class MissingCapabilityKind(str, Enum):
    CONTROL = "control"
    INFORMATION = "information"


@dataclass(frozen=True, slots=True)
class LimitationContext:
    """One-turn, human-readable context for an immediately adjacent follow-up."""

    missing_kind: MissingCapabilityKind
    missing_description: str
    goal_description: str
    unperformed_description: str
    source_user_turn: int


@dataclass(slots=True)
class SessionTombstone:
    """Minimal closed-session state retained only for idempotent replay."""

    context_id: str
    lifecycle: SessionLifecycle
    closed_at: float
    seen_message_ids: set[str] = field(default_factory=set)
    outbound_by_message_id: dict[str, Any] = field(default_factory=dict)
    last_outbound: Any = None

    @property
    def terminal(self) -> bool:
        return self.lifecycle is SessionLifecycle.TERMINAL

    @property
    def cancelled(self) -> bool:
        return self.lifecycle is SessionLifecycle.CANCELLED

    def replay_for(self, message_id: str | None) -> Any | None:
        return None if not message_id else self.outbound_by_message_id.get(message_id)


class SessionClosedError(RuntimeError):
    def __init__(
        self, context_id: str, tombstone: SessionTombstone | None = None
    ) -> None:
        self.context_id = context_id
        self.tombstone = tombstone
        lifecycle = tombstone.lifecycle.value if tombstone is not None else "closed"
        super().__init__(f"session {context_id!r} is {lifecycle}")


def _safe_tombstone_replay(outbound: Any) -> bool:
    """Only text may be replayed after close; replaying tools repeats effects."""

    if isinstance(outbound, Mapping):
        text = outbound.get("text")
        tool_calls = outbound.get("tool_calls", outbound.get("toolCalls", ()))
    else:
        text = getattr(outbound, "text", None)
        tool_calls = getattr(outbound, "tool_calls", ())
    return text is not None and not tool_calls


@dataclass(slots=True)
class ExecutionBundle:
    goal_ids: tuple[str, ...]
    calls: tuple[OfficialToolCall, ...]
    policy_operations: tuple[str | None, ...] = ()
    call_goal_ids: tuple[str, ...] = ()
    terminal_after_success: bool = False
    next_index: int = 0
    confirmation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.goal_ids or not self.calls:
            raise ValueError("execution bundle requires goals and calls")
        if not self.policy_operations:
            self.policy_operations = (None,) * len(self.calls)
        if len(self.policy_operations) != len(self.calls):
            raise ValueError("policy operation metadata must align with calls")
        if not self.call_goal_ids:
            self.call_goal_ids = (self.goal_ids[0],) * len(self.calls)
        if len(self.call_goal_ids) != len(self.calls):
            raise ValueError("call goal metadata must align with calls")
        if not set(self.call_goal_ids).issubset(self.goal_ids):
            raise ValueError("call goal metadata must reference bundle goals")
        if not 0 <= self.next_index <= len(self.calls):
            raise ValueError("execution bundle index is out of range")

    @property
    def current_call(self) -> OfficialToolCall | None:
        return (
            self.calls[self.next_index] if self.next_index < len(self.calls) else None
        )

    @property
    def current_policy_operation(self) -> str | None:
        return (
            self.policy_operations[self.next_index]
            if self.next_index < len(self.policy_operations)
            else None
        )

    @property
    def current_goal_id(self) -> str | None:
        return (
            self.call_goal_ids[self.next_index]
            if self.next_index < len(self.call_goal_ids)
            else None
        )

    @property
    def complete(self) -> bool:
        return self.next_index >= len(self.calls)

    def advance(self) -> None:
        if not self.complete:
            self.next_index += 1


@dataclass(frozen=True, slots=True)
class ResolvedLocationBinding:
    """Provenance for one exact named-location resolution at one state version."""

    normalized_name: str
    location_id: str
    source_call_id: str
    source_evidence_id: str
    state_version: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.normalized_name, str)
            or not self.normalized_name
            or self.normalized_name != _normalize_location_name(self.normalized_name)
        ):
            raise ValueError("resolved location name must be normalized and non-empty")
        if (
            not isinstance(self.location_id, str)
            or not self.location_id.startswith("loc_")
            or any(character.isspace() for character in self.location_id)
        ):
            raise ValueError("resolved location IDs must be non-empty loc_* values")
        if (
            not isinstance(self.source_call_id, str)
            or not self.source_call_id.strip()
            or not isinstance(self.source_evidence_id, str)
            or not self.source_evidence_id.strip()
        ):
            raise ValueError("resolved locations require call and evidence provenance")
        if (
            not isinstance(self.state_version, int)
            or isinstance(self.state_version, bool)
            or self.state_version < 0
        ):
            raise ValueError("resolved location state version must be non-negative")


@dataclass(frozen=True, slots=True)
class DisplayedPOICandidate:
    """One bounded POI option shown to the user with its exact provenance."""

    normalized_name: str
    display_name: str
    poi_id: str
    phone_number: str
    location_id: str
    source_call_id: str
    source_poi_evidence_id: str
    source_location_evidence_id: str
    displayed_evidence_id: str
    state_version: int
    opening_hours: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.normalized_name
            or self.normalized_name != _normalize_location_name(self.display_name)
            or not self.poi_id.startswith("poi_")
            or not self.location_id.startswith("loc_")
            or not self.phone_number.startswith("+")
            or not self.source_call_id
            or not self.source_poi_evidence_id
            or not self.source_location_evidence_id
            or not self.displayed_evidence_id
            or self.state_version < 0
        ):
            raise ValueError("displayed POI candidate is malformed")
        if self.opening_hours is not None and (
            not isinstance(self.opening_hours, str)
            or not self.opening_hours.strip()
            or self.opening_hours != self.opening_hours.strip()
            or len(self.opening_hours) > 120
            or any(
                unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
                for character in self.opening_hours
            )
        ):
            raise ValueError("displayed POI opening hours are malformed")


def _normalize_location_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


@dataclass(frozen=True, slots=True)
class SuccessfulReadResult:
    goal_ids: tuple[str, ...]
    tool_name: str
    arguments: dict[str, Any]
    semantic_operation: str | None
    call_id: str
    value: Any
    source_turn_id: str
    state_version: int
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class PendingSetReadback:
    """A successful SET whose omitted target fields require a fresh state read."""

    pending_call: PendingToolCall
    expected_state_version: int


@dataclass(frozen=True, slots=True)
class PendingDestinationClarification:
    """One open city name requested by an explicit destination replacement."""

    previous_destination_name: str
    route_choice_alias: str
    source_turn_id: str
    source_user_turn: int

    def __post_init__(self) -> None:
        if (
            not self.previous_destination_name.strip()
            or len(self.previous_destination_name) > 80
            or self.route_choice_alias not in {"fastest", "shortest"}
            or not self.source_turn_id.strip()
            or self.source_user_turn < 1
        ):
            raise ValueError("pending destination clarification is malformed")


@dataclass(slots=True)
class SessionState:
    context_id: str
    policy_text: str | None = None
    compiled_policy: Any = None
    conversation: list[dict[str, Any]] = field(default_factory=list)
    live_tools: Any = None
    intent: Any = None
    authorized_action_goal_ids: set[str] = field(default_factory=set)
    grounded_desired_values_by_goal: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    grounded_value_sources_by_goal: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    derived_value_evidence_by_goal: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    relative_fan_speed_deltas_by_goal: dict[str, int] = field(default_factory=dict)
    relative_climate_temperature_deltas_by_goal: dict[str, int] = field(
        default_factory=dict
    )
    relative_seat_heating_deltas_by_goal: dict[str, int] = field(default_factory=dict)
    occupied_seat_heating_goal_ids: set[str] = field(default_factory=set)
    climate_sync_source_zones_by_goal: dict[str, str] = field(default_factory=dict)
    goal_dag: GoalDAG = field(default_factory=GoalDAG)
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    confirmation_latch: Any = field(default_factory=_new_confirmation_latch)
    pending_calls: list[PendingToolCall] = field(default_factory=list)
    pending_set_readback: PendingSetReadback | None = None
    pending_destination_clarification: PendingDestinationClarification | None = None
    execution_bundle: ExecutionBundle | None = None
    completed_action_calls_by_goal: dict[str, tuple[OfficialToolCall, ...]] = field(
        default_factory=dict
    )
    completed_action_operations_by_goal: dict[str, tuple[str | None, ...]] = field(
        default_factory=dict
    )
    pending_clarifications: dict[str, Any] = field(default_factory=dict)
    completed_policy_operations: set[str] = field(default_factory=set)
    call_sequence: int = 0
    last_tool_results: list[dict[str, Any]] = field(default_factory=list)
    successful_read_results: list[SuccessfulReadResult] = field(default_factory=list)
    resolved_location_ids_by_name: dict[str, ResolvedLocationBinding] = field(
        default_factory=dict
    )
    poisoned_resolved_location_bindings: set[tuple[str, int]] = field(
        default_factory=set
    )
    displayed_destination_pois: tuple[DisplayedPOICandidate, ...] = ()
    seen_message_ids: set[str] = field(default_factory=set)
    seen_result_ids_by_task: dict[str | None, set[str]] = field(default_factory=dict)
    external_call_ids_by_task: dict[str | None, dict[str, str]] = field(
        default_factory=dict
    )
    outbound_by_message_id: dict[str, Any] = field(default_factory=dict)
    last_outbound: Any = None
    last_limitation_context: LimitationContext | None = None
    budget: RuntimeBudget = field(default_factory=RuntimeBudget)
    metrics: TurnMetricsAccumulator = field(default_factory=TurnMetricsAccumulator)
    created_at: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)
    terminal: bool = False
    cancelled: bool = False
    closed: bool = False
    lock: RLock = field(default_factory=RLock, repr=False)

    @contextmanager
    def synchronized(self) -> Iterator[SessionState]:
        """Hold the per-context re-entrant lock for one orchestration step."""

        with self.lock:
            if self.closed:
                raise SessionClosedError(self.context_id)
            yield self

    def append(self, role: str, content: Any, **extra: Any) -> None:
        with self.lock:
            if self.closed:
                raise SessionClosedError(self.context_id)
            self.conversation.append({"role": role, "content": content, **extra})

    def accept_message(self, message_id: str | None) -> bool:
        accepted, _ = self.accept_or_replay(message_id)
        return accepted

    def accept_or_replay(self, message_id: str | None) -> tuple[bool, Any | None]:
        """Atomically accept a new message ID or return its cached response."""

        with self.lock:
            if self.closed:
                raise SessionClosedError(self.context_id)
            if not message_id:
                return True, None
            if message_id in self.seen_message_ids:
                return False, self.outbound_by_message_id.get(message_id)
            self.seen_message_ids.add(message_id)
            return True, None

    def cache_outbound(self, message_id: str | None, outbound: Any) -> None:
        with self.lock:
            if self.closed:
                raise SessionClosedError(self.context_id)
            if message_id:
                existing = self.outbound_by_message_id.get(message_id)
                if existing is not None and existing != outbound:
                    raise ValueError(
                        "a message ID cannot be associated with different outbound data"
                    )
                self.outbound_by_message_id[message_id] = outbound
            self.last_outbound = outbound

    def replay_for(self, message_id: str | None) -> Any | None:
        with self.lock:
            return (
                None if not message_id else self.outbound_by_message_id.get(message_id)
            )

    def next_call_id(self) -> str:
        with self.lock:
            if self.closed:
                raise SessionClosedError(self.context_id)
            self.call_sequence += 1
            return f"agent-call-{self.call_sequence}"

    def remember_resolved_location(
        self,
        *,
        literal_name: str,
        location_id: str,
        source_call_id: str,
        source_evidence_id: str,
        state_version: int,
    ) -> ResolvedLocationBinding:
        """Cache a unique loc_* resolution without weakening its provenance."""

        normalized_name = _normalize_location_name(literal_name)
        binding = ResolvedLocationBinding(
            normalized_name=normalized_name,
            location_id=location_id,
            source_call_id=source_call_id,
            source_evidence_id=source_evidence_id,
            state_version=state_version,
        )
        with self.lock:
            if self.closed:
                raise SessionClosedError(self.context_id)
            existing = self.resolved_location_ids_by_name.get(normalized_name)
            conflict_key = (normalized_name, state_version)
            if existing is not None:
                if existing.state_version > state_version:
                    raise ValueError("cannot replace a newer location resolution")
                if (
                    existing.state_version == state_version
                    and existing.location_id != location_id
                ):
                    self.resolved_location_ids_by_name.pop(normalized_name, None)
                    self.poisoned_resolved_location_bindings.add(conflict_key)
                    raise ValueError(
                        "one literal location name resolved to conflicting IDs"
                    )
            if any(
                name == normalized_name and poisoned_version > state_version
                for name, poisoned_version in self.poisoned_resolved_location_bindings
            ):
                raise ValueError("cannot replace a newer location resolution")
            if conflict_key in self.poisoned_resolved_location_bindings:
                raise ValueError(
                    "location resolution is poisoned for this state version"
                )
            obsolete_conflicts = {
                poisoned_key
                for poisoned_key in self.poisoned_resolved_location_bindings
                if poisoned_key[0] == normalized_name
                and poisoned_key[1] < state_version
            }
            self.poisoned_resolved_location_bindings.difference_update(
                obsolete_conflicts
            )
            self.resolved_location_ids_by_name[normalized_name] = binding
        return binding

    def resolved_location_binding(
        self, literal_name: str, *, state_version: int
    ) -> ResolvedLocationBinding | None:
        """Return a binding only while its observed state version is current."""

        normalized_name = _normalize_location_name(literal_name)
        if not normalized_name:
            return None
        with self.lock:
            if (
                normalized_name,
                state_version,
            ) in self.poisoned_resolved_location_bindings:
                return None
            binding = self.resolved_location_ids_by_name.get(normalized_name)
            if binding is None or binding.state_version != state_version:
                return None
            return binding

    def remember_displayed_destination_pois(
        self, candidates: tuple[DisplayedPOICandidate, ...]
    ) -> None:
        if (
            not 1 <= len(candidates) <= 10
            or len({candidate.normalized_name for candidate in candidates})
            != len(candidates)
            or any(
                candidate.state_version != self.evidence.current_state_version
                for candidate in candidates
            )
        ):
            raise ValueError("displayed POI candidates must be bounded and unique")
        with self.lock:
            if self.closed:
                raise SessionClosedError(self.context_id)
            self.displayed_destination_pois = tuple(candidates)

    def clear_displayed_destination_pois(self) -> None:
        with self.lock:
            if self.closed:
                raise SessionClosedError(self.context_id)
            self.displayed_destination_pois = ()


class SessionStore:
    def __init__(
        self,
        *,
        ttl_seconds: int = 1800,
        max_sessions: int = 256,
        soft_max_steps: int = 49,
        max_user_turns: int = 12,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.soft_max_steps = soft_max_steps
        self.max_user_turns = max_user_turns
        self._clock = clock
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._tombstones: OrderedDict[str, SessionTombstone] = OrderedDict()
        self._lock = RLock()

    def _purge_expired(self, now: float) -> None:
        expired = [
            context_id
            for context_id, session in self._sessions.items()
            if now - session.last_access >= self.ttl_seconds
        ]
        for context_id in expired:
            del self._sessions[context_id]
        expired_tombstones = [
            context_id
            for context_id, tombstone in self._tombstones.items()
            if now - tombstone.closed_at >= self.ttl_seconds
        ]
        for context_id in expired_tombstones:
            del self._tombstones[context_id]

    def get_or_create(self, context_id: str) -> SessionState:
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            tombstone = self._tombstones.get(context_id)
            if tombstone is not None:
                self._tombstones.move_to_end(context_id)
                raise SessionClosedError(context_id, tombstone)
            session = self._sessions.pop(context_id, None)
            if session is None:
                session = SessionState(
                    context_id=context_id,
                    budget=RuntimeBudget(
                        soft_max_steps=self.soft_max_steps,
                        max_user_turns=self.max_user_turns,
                    ),
                    created_at=now,
                )
            elif session.closed:
                raise SessionClosedError(context_id, self._tombstones.get(context_id))
            session.last_access = now
            self._sessions[context_id] = session
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return session

    def reopen(self, context_id: str) -> SessionState:
        """Explicitly discard a tombstone and begin a new context lifecycle."""

        with self._lock:
            self._purge_expired(self._clock())
            if context_id in self._sessions:
                raise ValueError("an active session cannot be reopened")
            self._tombstones.pop(context_id, None)
        return self.get_or_create(context_id)

    def get(self, context_id: str) -> SessionState | None:
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            session = self._sessions.get(context_id)
            if session is not None and not session.closed:
                session.last_access = now
                self._sessions.move_to_end(context_id)
                return session
            return None

    def get_tombstone(self, context_id: str) -> SessionTombstone | None:
        with self._lock:
            self._purge_expired(self._clock())
            tombstone = self._tombstones.get(context_id)
            if tombstone is not None:
                self._tombstones.move_to_end(context_id)
            return tombstone

    def replay_for(self, context_id: str, message_id: str | None) -> Any | None:
        """Replay by exact message ID from either active or closed state."""

        with self._lock:
            self._purge_expired(self._clock())
            session = self._sessions.get(context_id)
            tombstone = self._tombstones.get(context_id)
        if session is not None:
            return session.replay_for(message_id)
        return None if tombstone is None else tombstone.replay_for(message_id)

    @contextmanager
    def locked(self, context_id: str, *, create: bool = True) -> Iterator[SessionState]:
        """Resolve a context and hold its public per-session lock."""

        session = self.get_or_create(context_id) if create else self.get(context_id)
        if session is None:
            raise KeyError(context_id)
        with session.synchronized():
            yield session

    def remove(self, context_id: str) -> None:
        """Hard-delete active state and any replay tombstone."""

        with self._lock:
            self._sessions.pop(context_id, None)
            self._tombstones.pop(context_id, None)

    def cancel(
        self,
        context_id: str,
        *,
        message_id: str | None = None,
        outbound: Any = None,
    ) -> SessionTombstone:
        """Close and erase mutable state, retaining only a safe cancel reply."""

        return self._close(
            context_id,
            lifecycle=SessionLifecycle.CANCELLED,
            message_id=message_id,
            outbound=outbound,
        )

    def mark_terminal(
        self,
        context_id: str,
        *,
        message_id: str | None = None,
        outbound: Any = None,
    ) -> SessionTombstone:
        """Close and retain text replay only; tool calls are never replayed."""

        return self._close(
            context_id,
            lifecycle=SessionLifecycle.TERMINAL,
            message_id=message_id,
            outbound=outbound,
        )

    def _close(
        self,
        context_id: str,
        *,
        lifecycle: SessionLifecycle,
        message_id: str | None,
        outbound: Any,
    ) -> SessionTombstone:
        with self._lock:
            session = self._sessions.get(context_id)
            existing = self._tombstones.get(context_id)
        if session is None and existing is not None:
            return existing

        now = self._clock()
        if session is None:
            seen_message_ids: set[str] = set()
            replay: dict[str, Any] = {}
            last_outbound: Any = None
        else:
            with session.lock:
                with self._lock:
                    current = self._sessions.get(context_id)
                    if current is not session:
                        existing = self._tombstones.get(context_id)
                        if existing is not None:
                            return existing
                        raise SessionClosedError(context_id)
                seen_message_ids = set(session.seen_message_ids)
                replay = (
                    {
                        key: value
                        for key, value in session.outbound_by_message_id.items()
                        if _safe_tombstone_replay(value)
                    }
                    if lifecycle is SessionLifecycle.TERMINAL
                    else {}
                )
                last_outbound = (
                    session.last_outbound
                    if lifecycle is SessionLifecycle.TERMINAL
                    and _safe_tombstone_replay(session.last_outbound)
                    else None
                )
                session.closed = True
                session.terminal = lifecycle is SessionLifecycle.TERMINAL
                session.cancelled = lifecycle is SessionLifecycle.CANCELLED
                if message_id:
                    seen_message_ids.add(message_id)
                    if outbound is not None and _safe_tombstone_replay(outbound):
                        replay[message_id] = outbound
                if outbound is not None and _safe_tombstone_replay(outbound):
                    last_outbound = outbound
                tombstone = SessionTombstone(
                    context_id=context_id,
                    lifecycle=lifecycle,
                    closed_at=now,
                    seen_message_ids=seen_message_ids,
                    outbound_by_message_id=replay,
                    last_outbound=last_outbound,
                )
                with self._lock:
                    if self._sessions.get(context_id) is session:
                        del self._sessions[context_id]
                    self._tombstones[context_id] = tombstone
                    self._tombstones.move_to_end(context_id)
                    while len(self._tombstones) > self.max_sessions:
                        self._tombstones.popitem(last=False)
                return tombstone

        if message_id:
            seen_message_ids.add(message_id)
            if outbound is not None and _safe_tombstone_replay(outbound):
                replay[message_id] = outbound
        if outbound is not None and _safe_tombstone_replay(outbound):
            last_outbound = outbound
        tombstone = SessionTombstone(
            context_id=context_id,
            lifecycle=lifecycle,
            closed_at=now,
            seen_message_ids=seen_message_ids,
            outbound_by_message_id=replay,
            last_outbound=last_outbound,
        )
        with self._lock:
            self._tombstones[context_id] = tombstone
            self._tombstones.move_to_end(context_id)
            while len(self._tombstones) > self.max_sessions:
                self._tombstones.popitem(last=False)
        return tombstone

    @property
    def closed_count(self) -> int:
        with self._lock:
            self._purge_expired(self._clock())
            return len(self._tombstones)

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired(self._clock())
            return len(self._sessions)
