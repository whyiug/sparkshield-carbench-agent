"""A2A protocol parsing, rendering, result matching, and session state."""

from .adapter import A2AAdapter, InboundEvent, OutboundEnvelope
from .result_matching import MatchedToolResult, PendingToolCall, match_tool_results
from .session_store import (
    ExecutionBundle,
    SessionClosedError,
    SessionLifecycle,
    SessionState,
    SessionStore,
    SessionTombstone,
    SuccessfulReadResult,
)

__all__ = [
    "A2AAdapter",
    "ExecutionBundle",
    "InboundEvent",
    "MatchedToolResult",
    "OutboundEnvelope",
    "PendingToolCall",
    "SessionClosedError",
    "SessionLifecycle",
    "SessionState",
    "SessionStore",
    "SessionTombstone",
    "SuccessfulReadResult",
    "match_tool_results",
]
