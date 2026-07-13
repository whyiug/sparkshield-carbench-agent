"""Thin official A2A executor for the CAR-Guard runtime."""

from __future__ import annotations

import asyncio
import hashlib
import sys
import time
import uuid
from pathlib import Path
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Role
from a2a.utils.errors import TaskNotCancelableError

from logging_utils import configure_logger
from track_1_agent_under_test.car_guard.a2a import A2AAdapter, OutboundEnvelope
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
)


logger = configure_logger(role="agent_under_test", context="-")
_TURN_METRICS_KEY = "turn_metrics"
_SAFE_FAILURE_TEXT = (
    "I couldn't validate a safe next step, so I did not perform an action."
)


class CARBenchAgentExecutor(AgentExecutor):
    """Adapt official A2A requests to one synchronous CAR-Guard orchestrator."""

    def __init__(
        self,
        config: AgentConfig | str | None = None,
        *,
        orchestrator: CARGuardOrchestrator | Any | None = None,
        model: str | None = None,
        temperature: float | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        interleaved_thinking: bool | None = None,
    ) -> None:
        if isinstance(config, str):
            if model is not None:
                raise ValueError("model was supplied twice")
            model = config
            base = AgentConfig.from_env()
        else:
            base = config if config is not None else AgentConfig.from_env()
        values = base.model_dump()
        values.update(
            {
                key: value
                for key, value in {
                    "llm": model,
                    "temperature": temperature,
                    "thinking": thinking,
                    "reasoning_effort": reasoning_effort,
                    "interleaved_thinking": interleaved_thinking,
                }.items()
                if value is not None
            }
        )
        self.config = AgentConfig.model_validate(values)
        # Read-only compatibility attributes for starter integrations.
        self.model = self.config.llm
        self.temperature = self.config.temperature
        self.thinking = self.config.thinking
        self.reasoning_effort = self.config.reasoning_effort
        self.interleaved_thinking = self.config.interleaved_thinking
        self.orchestrator = (
            orchestrator
            if orchestrator is not None
            else CARGuardOrchestrator(self.config)
        )
        self.adapter = A2AAdapter()
        self._emission_locks: dict[str, asyncio.Lock] = {}
        self._cancelled_at: dict[str, float] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run blocking planning/provider work off the event loop and emit once."""

        context_id = context.context_id
        message = context.message
        logical_task_id = context_id
        response_message_id = self._response_message_id(
            context_id=context_id,
            task_id=logical_task_id,
            inbound_message=message,
        )
        try:
            if message is None or not context_id:
                raise ValueError("request is missing its message or context")
            outbound = await asyncio.to_thread(
                self.orchestrator.handle_message,
                message,
                context_id=context_id,
                task=context.current_task,
                task_id=logical_task_id,
                request_metadata=context.metadata,
            )
            outbound = self._sanitize_outbound(outbound)
            response = self.adapter.to_message(
                outbound,
                message_id=response_message_id,
                context_id=context_id,
                role=Role.ROLE_AGENT,
            )
        except Exception as exc:
            logger.bind(
                role="agent_under_test",
                context=self._context_label(context_id),
            ).error("CAR-Guard request failed: {}", type(exc).__name__)
            response = self.adapter.to_message(
                self.adapter.text(_SAFE_FAILURE_TEXT),
                message_id=response_message_id,
                context_id=context_id,
                role=Role.ROLE_AGENT,
            )

        if context_id:
            async with self._emission_lock(context_id):
                if self._is_cancelled(context_id):
                    response = self.adapter.to_message(
                        self.adapter.text(
                            "This request was cancelled, so I did not perform an action.",
                            terminal=True,
                        ),
                        message_id=response_message_id,
                        context_id=context_id,
                        role=Role.ROLE_AGENT,
                    )
                await event_queue.enqueue_event(response)
            return
        await event_queue.enqueue_event(response)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel one context and publish the SDK-required terminal status."""

        context_id = context.context_id
        task_id = context.task_id
        if not context_id or not task_id:
            raise TaskNotCancelableError(
                message="cancellation requires both task_id and context_id"
            )
        async with self._emission_lock(context_id):
            self._cancelled_at[context_id] = time.monotonic()
            try:
                await asyncio.to_thread(self.orchestrator.sessions.cancel, context_id)
            except Exception as exc:
                self._cancelled_at.pop(context_id, None)
                logger.bind(
                    role="agent_under_test",
                    context=self._context_label(context_id),
                ).warning("CAR-Guard cancellation failed: {}", type(exc).__name__)
                raise TaskNotCancelableError(
                    message="the CAR-Guard session could not be cancelled"
                ) from exc

            updater = TaskUpdater(event_queue, task_id, context_id)
            await updater.cancel()

    @staticmethod
    def _sanitize_outbound(outbound: OutboundEnvelope) -> OutboundEnvelope:
        """Enforce tool XOR text and keep metrics off intermediate tool turns."""

        if bool(outbound.text is not None) == bool(outbound.tool_calls):
            raise ValueError("CAR-Guard returned an invalid output mode")
        if not outbound.tool_calls or _TURN_METRICS_KEY not in outbound.metadata:
            return outbound
        metadata = dict(outbound.metadata)
        metadata.pop(_TURN_METRICS_KEY, None)
        return OutboundEnvelope(
            tool_calls=outbound.tool_calls,
            terminal=outbound.terminal,
            metadata=metadata,
        )

    @staticmethod
    def _context_label(context_id: str | None) -> str:
        return "ctx:missing" if not context_id else f"ctx:{context_id[:8]}"

    def _emission_lock(self, context_id: str) -> asyncio.Lock:
        lock = self._emission_locks.get(context_id)
        if lock is None:
            lock = asyncio.Lock()
            self._emission_locks[context_id] = lock
        return lock

    def _is_cancelled(self, context_id: str) -> bool:
        cancelled_at = self._cancelled_at.get(context_id)
        if cancelled_at is None:
            return False
        if time.monotonic() - cancelled_at < self.config.session_ttl_seconds:
            return True
        self._cancelled_at.pop(context_id, None)
        self._emission_locks.pop(context_id, None)
        return False

    @staticmethod
    def _response_message_id(
        *,
        context_id: str | None,
        task_id: str | None,
        inbound_message: Any | None,
    ) -> str:
        inbound_id = getattr(inbound_message, "message_id", "") or ""
        if not inbound_id and inbound_message is not None:
            try:
                encoded = inbound_message.SerializeToString(deterministic=True)
            except (AttributeError, TypeError, ValueError):
                encoded = repr(inbound_message).encode("utf-8", errors="replace")
            inbound_id = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        identity = "\x1f".join(
            (
                "car-guard-agent-response-v1",
                context_id or "",
                task_id or "",
                str(inbound_id),
            )
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


__all__ = ["CARBenchAgentExecutor"]
