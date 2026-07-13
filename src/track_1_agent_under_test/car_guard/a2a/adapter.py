"""CAR-bench A2A message adapter with a strict tool-XOR-text contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterable

from google.protobuf.json_format import MessageToDict


@dataclass(frozen=True, slots=True)
class InboundEvent:
    message_id: str | None = None
    context_id: str | None = None
    task_id: str | None = None
    system_policy: str | None = None
    user_text: str | None = None
    live_tools: tuple[dict[str, Any], ...] = ()
    tool_results: tuple[dict[str, Any], ...] = ()
    malformed_parts: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    task_metadata: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)
    task_state: str | None = None
    cancel_requested: bool = False
    terminal: bool = False

    @property
    def has_initial_system_policy(self) -> bool:
        return self.system_policy is not None

    @property
    def has_tool_results(self) -> bool:
        return bool(self.tool_results)

    @property
    def should_close_session(self) -> bool:
        return self.cancel_requested or self.terminal


@dataclass(frozen=True, slots=True)
class OutboundEnvelope:
    text: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()
    terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.text is not None) == bool(self.tool_calls):
            raise ValueError("outbound must contain exactly one of text or tool calls")


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        return MessageToDict(value, preserving_proto_field_name=True)
    except TypeError:
        return MessageToDict(value)


def _get(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _object_value(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None and candidate != "":
            return candidate
    return None


def _metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return _mapping(value)
    except Exception:
        return {}


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    value = _get(dict(metadata), *keys)
    if value is not None:
        return value
    lifecycle = metadata.get("lifecycle")
    if isinstance(lifecycle, Mapping):
        return _get(dict(lifecycle), *keys)
    return None


def _truthy_signal(value: Any) -> bool:
    if type(value) is bool:
        return value
    return isinstance(value, str) and value.strip().lower() in {
        "1",
        "true",
        "yes",
        "cancel",
        "canceled",
        "cancelled",
        "terminal",
        "completed",
        "done",
    }


_TASK_STATE_NAMES = {
    0: "unspecified",
    1: "submitted",
    2: "working",
    3: "completed",
    4: "failed",
    5: "canceled",
    6: "input_required",
    7: "rejected",
    8: "auth_required",
}
_TERMINAL_TASK_STATES = frozenset({"completed", "failed", "canceled", "rejected"})
_CANCEL_TASK_STATES = frozenset({"canceled", "cancelled"})


def _normalise_task_state(value: Any) -> str | None:
    if type(value) is int:
        return _TASK_STATE_NAMES.get(value, f"unknown_{value}")
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.removeprefix("task_state_")
    return "canceled" if normalized == "cancelled" else normalized


def _task_state(task: Any, metadata_sources: Iterable[Mapping[str, Any]]) -> str | None:
    status = _object_value(task, "status") if task is not None else None
    state = _object_value(status, "state") if status is not None else None
    normalized = _normalise_task_state(state)
    fallback = normalized
    if normalized is not None and normalized != "unspecified":
        return normalized
    for metadata in metadata_sources:
        normalized = _normalise_task_state(
            _metadata_value(metadata, "task_state", "taskState", "state", "status")
        )
        if normalized is not None:
            return normalized
    return fallback


def _lifecycle_label(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("lifecycle")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _identifier(
    explicit: str | None,
    message: Any,
    task: Any,
    metadata_sources: Iterable[Mapping[str, Any]],
    *names: str,
) -> str | None:
    if explicit:
        return str(explicit)
    value = _object_value(message, *names) or _object_value(task, *names)
    if value:
        return str(value)
    for metadata in metadata_sources:
        value = _metadata_value(metadata, *names)
        if value:
            return str(value)
    return None


class A2AAdapter:
    """Translate protobuf messages without making business decisions."""

    @staticmethod
    def split_initial_text(text: str) -> tuple[str | None, str | None]:
        marker = "\n\nUser:"
        if text.startswith("System:") and marker in text:
            system, user = text.split(marker, 1)
            return system.removeprefix("System:").strip(), user.strip() or None
        normalized = text.strip()
        if not normalized or normalized.lower() == "none":
            return None, None
        return None, normalized

    def parse(
        self,
        message: Any,
        *,
        task: Any | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        request_metadata: Mapping[str, Any] | Any | None = None,
    ) -> InboundEvent:
        policy: str | None = None
        user_text: str | None = None
        tools: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        malformed: list[str] = []

        for index, part in enumerate(getattr(message, "parts", ())):
            try:
                content_type = part.WhichOneof("content")
            except (AttributeError, ValueError):
                malformed.append(f"part[{index}]:unknown-content")
                continue

            if content_type == "text":
                current_policy, current_user = self.split_initial_text(part.text)
                if current_policy is not None:
                    policy = current_policy
                if current_user is not None:
                    user_text = current_user
                continue

            if content_type != "data":
                malformed.append(f"part[{index}]:unsupported-{content_type}")
                continue

            try:
                data = _mapping(part.data)
            except Exception:
                malformed.append(f"part[{index}]:malformed-data")
                continue
            raw_tools = _get(data, "tools")
            raw_results = _get(data, "tool_results", "toolResults")
            if isinstance(raw_tools, list):
                for item_index, item in enumerate(raw_tools):
                    if isinstance(item, dict):
                        tools.append(item)
                    else:
                        malformed.append(
                            f"part[{index}]:tools[{item_index}]-not-object"
                        )
            elif raw_tools is not None:
                malformed.append(f"part[{index}]:tools-not-list")
            if isinstance(raw_results, list):
                for item_index, item in enumerate(raw_results):
                    if isinstance(item, dict):
                        results.append(item)
                    else:
                        malformed.append(
                            f"part[{index}]:results[{item_index}]-not-object"
                        )
            elif raw_results is not None:
                malformed.append(f"part[{index}]:results-not-list")

        message_metadata = _metadata(_object_value(message, "metadata"))
        task_metadata = _metadata(_object_value(task, "metadata"))
        parsed_request_metadata = _metadata(request_metadata)
        metadata_sources = (
            parsed_request_metadata,
            message_metadata,
            task_metadata,
        )
        parsed_state = _task_state(task, metadata_sources)
        cancel_requested = parsed_state in _CANCEL_TASK_STATES or any(
            _lifecycle_label(metadata) in _CANCEL_TASK_STATES
            or _truthy_signal(
                _metadata_value(
                    metadata,
                    "cancel",
                    "canceled",
                    "cancelled",
                    "cancel_requested",
                    "cancelRequested",
                )
            )
            for metadata in metadata_sources
        )
        terminal = (
            cancel_requested
            or parsed_state in _TERMINAL_TASK_STATES
            or any(
                _lifecycle_label(metadata) in _TERMINAL_TASK_STATES
                or _lifecycle_label(metadata) in {"terminal", "done"}
                or _truthy_signal(
                    _metadata_value(
                        metadata,
                        "terminal",
                        "is_terminal",
                        "isTerminal",
                        "done",
                        "end_of_session",
                        "endOfSession",
                    )
                )
                for metadata in metadata_sources
            )
        )
        message_id = _object_value(message, "message_id", "messageId")
        return InboundEvent(
            message_id=str(message_id) if message_id else None,
            context_id=_identifier(
                context_id,
                message,
                task,
                metadata_sources,
                "context_id",
                "contextId",
            ),
            task_id=_identifier(
                task_id,
                message,
                task,
                metadata_sources,
                "task_id",
                "taskId",
                "id",
            ),
            system_policy=policy,
            user_text=user_text,
            live_tools=tuple(tools),
            tool_results=tuple(results),
            malformed_parts=tuple(malformed),
            metadata=message_metadata,
            task_metadata=task_metadata,
            request_metadata=parsed_request_metadata,
            task_state=parsed_state,
            cancel_requested=cancel_requested,
            terminal=terminal,
        )

    @staticmethod
    def text(
        text: str,
        *,
        terminal: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> OutboundEnvelope:
        normalized = " ".join(text.split())
        return OutboundEnvelope(
            text=normalized,
            terminal=terminal,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def calls(
        calls: Iterable[Any],
        *,
        terminal: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> OutboundEnvelope:
        serialized: list[dict[str, Any]] = []
        names: set[str] = set()
        for call in calls:
            if hasattr(call, "model_dump"):
                value = call.model_dump()
            elif isinstance(call, dict):
                value = dict(call)
            else:
                value = {
                    "tool_name": getattr(call, "tool_name"),
                    "arguments": getattr(call, "arguments"),
                }
            name = value.get("tool_name") or value.get("name")
            arguments = value.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("invalid outbound tool call")
            if name in names:
                raise ValueError(
                    "parallel calls with duplicate tool names are ambiguous"
                )
            names.add(name)
            serialized.append({"tool_name": name, "arguments": arguments})
        if not serialized:
            raise ValueError("at least one tool call is required")
        return OutboundEnvelope(
            tool_calls=tuple(serialized),
            terminal=terminal,
            metadata=dict(metadata or {}),
        )

    def serialize(
        self,
        decision: Any,
        *,
        terminal: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> OutboundEnvelope:
        from ..domain import GateOutcome

        if decision.outcome is not GateOutcome.ALLOW:
            raise ValueError("only an allowed commit decision can be serialized")
        if decision.normalized_calls:
            return self.calls(
                decision.normalized_calls,
                terminal=terminal,
                metadata=metadata,
            )
        if decision.user_text is not None:
            return self.text(
                decision.user_text,
                terminal=terminal,
                metadata=metadata,
            )
        raise ValueError("allowed decision has no outbound content")

    @staticmethod
    def to_parts(outbound: OutboundEnvelope) -> list[Any]:
        from a2a.helpers.proto_helpers import new_data_part, new_text_part

        if outbound.text is not None:
            return [new_text_part(outbound.text)]
        return [new_data_part({"tool_calls": list(outbound.tool_calls)})]

    @classmethod
    def to_message(
        cls,
        outbound: OutboundEnvelope,
        *,
        message_id: str | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        role: Any | None = None,
    ) -> Any:
        """Build an official A2A Message and attach metadata at message scope."""

        from a2a.helpers.proto_helpers import new_message

        kwargs: dict[str, Any] = {
            "parts": cls.to_parts(outbound),
            "context_id": context_id,
            "task_id": task_id,
        }
        if role is not None:
            kwargs["role"] = role
        message = new_message(**kwargs)
        if message_id is not None:
            normalized_message_id = message_id.strip()
            if not normalized_message_id:
                raise ValueError("message_id cannot be blank")
            message.message_id = normalized_message_id
        metadata = dict(outbound.metadata)
        if outbound.terminal:
            metadata.setdefault("terminal", True)
        if metadata:
            message.metadata.update(metadata)
        return message
