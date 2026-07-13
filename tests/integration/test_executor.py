from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from a2a.helpers.proto_helpers import new_message, new_text_part
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.types import (
    Role,
    SendMessageRequest,
    Task,
    TaskState,
    TaskStatusUpdateEvent,
)
from fastapi.testclient import TestClient
from google.protobuf.json_format import MessageToDict
from pydantic import SecretStr

from track_1_agent_under_test.car_bench_agent import CARBenchAgentExecutor
from track_1_agent_under_test.car_guard.a2a import A2AAdapter
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.server import (
    apply_cli_overrides,
    build_argument_parser,
    create_app,
)


class RecordingQueue:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


class RecordingSessions:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.thread_ids: list[int] = []
        self.active = {"ctx-executor-1", "ctx-other"}

    def cancel(self, context_id: str) -> None:
        self.cancelled.append(context_id)
        self.thread_ids.append(threading.get_ident())
        self.active.discard(context_id)


class FakeOrchestrator:
    def __init__(self, outbound: Any) -> None:
        self.outbound = outbound
        self.calls: list[dict[str, Any]] = []
        self.thread_ids: list[int] = []
        self.sessions = RecordingSessions()

    def handle_message(self, message: Any, **kwargs: Any) -> Any:
        self.thread_ids.append(threading.get_ident())
        self.calls.append({"message": message, **kwargs})
        return self.outbound


class RaisingOrchestrator(FakeOrchestrator):
    def handle_message(self, message: Any, **kwargs: Any) -> Any:
        del message, kwargs
        raise RuntimeError("provider credential must never reach the response")


class SlowToolOrchestrator(FakeOrchestrator):
    def __init__(self) -> None:
        super().__init__(
            A2AAdapter.calls(
                [{"tool_name": "set_fan_speed", "arguments": {"level": 2}}]
            )
        )
        self.started = threading.Event()
        self.release = threading.Event()

    def handle_message(self, message: Any, **kwargs: Any) -> Any:
        del message, kwargs
        self.started.set()
        assert self.release.wait(timeout=2)
        return self.outbound


def _request_context(
    *,
    context_id: str = "ctx-executor-1",
    task_id: str = "task-executor-1",
    message_id: str | None = None,
) -> RequestContext:
    message = new_message(
        parts=[new_text_part("System: Follow policy.\n\nUser: Set the fan.")],
        context_id=context_id,
        task_id=task_id,
        role=Role.ROLE_USER,
    )
    if message_id is not None:
        message.message_id = message_id
    message.metadata.update({"message_marker": "message-value"})
    request = SendMessageRequest(message=message)
    request.metadata.update({"request_marker": "request-value"})
    task = Task(id=task_id, context_id=context_id)
    task.metadata.update({"task_marker": "task-value"})
    return RequestContext(
        ServerCallContext(),
        request=request,
        task=task,
    )


@pytest.mark.asyncio
async def test_execute_runs_orchestrator_off_loop_and_forwards_request_context() -> (
    None
):
    adapter = A2AAdapter()
    outbound = adapter.text(
        "The fan request is ready.",
        metadata={
            "turn_metrics": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
            }
        },
    )
    orchestrator = FakeOrchestrator(outbound)
    executor = CARBenchAgentExecutor(
        AgentConfig(llm="test/model"), orchestrator=orchestrator
    )
    queue = RecordingQueue()
    context = _request_context()
    event_loop_thread = threading.get_ident()

    await executor.execute(context, queue)  # type: ignore[arg-type]

    assert orchestrator.thread_ids[0] != event_loop_thread
    assert len(orchestrator.calls) == 1
    call = orchestrator.calls[0]
    assert call["message"] is context.message
    assert call["context_id"] == "ctx-executor-1"
    assert call["task_id"] == "ctx-executor-1"
    assert call["task"] is context.current_task
    assert call["request_metadata"] == {"request_marker": "request-value"}

    assert len(queue.events) == 1
    response = queue.events[0]
    assert response.context_id == "ctx-executor-1"
    assert not response.task_id
    assert response.role == Role.ROLE_AGENT
    assert [part.WhichOneof("content") for part in response.parts] == ["text"]
    assert response.parts[0].text == "The fan request is ready."
    metadata = MessageToDict(response.metadata)
    assert metadata["turn_metrics"]["prompt_tokens"] == 12


@pytest.mark.asyncio
async def test_tool_output_is_one_data_part_and_never_carries_turn_metrics() -> None:
    adapter = A2AAdapter()
    outbound = adapter.calls(
        [{"tool_name": "get_fan", "arguments": {}}],
        metadata={"turn_metrics": {"prompt_tokens": 9}, "trace": "safe"},
    )
    orchestrator = FakeOrchestrator(outbound)
    executor = CARBenchAgentExecutor(
        AgentConfig(llm="test/model"), orchestrator=orchestrator
    )
    queue = RecordingQueue()

    await executor.execute(_request_context(), queue)  # type: ignore[arg-type]

    response = queue.events[0]
    assert [part.WhichOneof("content") for part in response.parts] == ["data"]
    assert MessageToDict(response.parts[0].data) == {
        "tool_calls": [{"tool_name": "get_fan", "arguments": {}}]
    }
    assert MessageToDict(response.metadata) == {"trace": "safe"}


def test_conversational_messages_continue_by_context_without_ephemeral_task_id() -> (
    None
):
    orchestrator = FakeOrchestrator(A2AAdapter.text("continue"))
    executor = CARBenchAgentExecutor(
        AgentConfig(llm="test/model"), orchestrator=orchestrator
    )
    app = create_app(
        executor.config,
        card_url="http://testserver/",
        executor=executor,
    )

    def send(client: TestClient, message: Any, request_id: str) -> dict[str, Any]:
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "SendMessage",
                "params": {"message": MessageToDict(message)},
            },
            headers={
                "Content-Type": "application/a2a+json",
                "A2A-Version": "1.0",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert "error" not in payload
        return payload["result"]["message"]

    first = new_message(parts=[new_text_part("first")], role=Role.ROLE_USER)
    first.message_id = "first-message"
    with TestClient(app) as client:
        first_response = send(client, first, "first-request")
        context_id = first_response["contextId"]
        assert "taskId" not in first_response

        second = new_message(
            parts=[new_text_part("second")],
            context_id=context_id,
            role=Role.ROLE_USER,
        )
        second.message_id = "second-message"
        second_response = send(client, second, "second-request")

    assert second_response["contextId"] == context_id
    assert "taskId" not in second_response
    assert [call["task_id"] for call in orchestrator.calls] == [
        context_id,
        context_id,
    ]


@pytest.mark.asyncio
async def test_replayed_inbound_gets_same_agent_message_id_for_tool_deduplication() -> (
    None
):
    outbound = A2AAdapter.calls(
        [{"tool_name": "set_fan_speed", "arguments": {"level": 2}}]
    )
    orchestrator = FakeOrchestrator(outbound)
    executor = CARBenchAgentExecutor(
        AgentConfig(llm="test/model"), orchestrator=orchestrator
    )
    context = _request_context(message_id="stable-inbound-message")
    first_queue = RecordingQueue()
    replay_queue = RecordingQueue()

    await executor.execute(context, first_queue)  # type: ignore[arg-type]
    await executor.execute(context, replay_queue)  # type: ignore[arg-type]

    first = first_queue.events[0]
    replay = replay_queue.events[0]
    assert first.message_id == replay.message_id
    assert first.message_id != context.message.message_id
    assert first.parts[0].WhichOneof("content") == "data"
    assert replay.parts[0].WhichOneof("content") == "data"

    other_queue = RecordingQueue()
    other_context = _request_context(
        context_id="ctx-other",
        task_id="task-other",
        message_id="stable-inbound-message",
    )
    await executor.execute(other_context, other_queue)  # type: ignore[arg-type]
    assert other_queue.events[0].message_id != first.message_id


@pytest.mark.asyncio
async def test_execute_returns_generic_text_without_exception_details() -> None:
    orchestrator = RaisingOrchestrator(A2AAdapter.text("unused"))
    executor = CARBenchAgentExecutor(
        AgentConfig(llm="test/model"), orchestrator=orchestrator
    )
    queue = RecordingQueue()

    await executor.execute(_request_context(), queue)  # type: ignore[arg-type]

    response = queue.events[0]
    assert [part.WhichOneof("content") for part in response.parts] == ["text"]
    assert "credential" not in response.parts[0].text
    assert "RuntimeError" not in response.parts[0].text
    assert MessageToDict(response.metadata) == {}


@pytest.mark.asyncio
async def test_cancel_calls_session_store_for_only_the_current_context() -> None:
    orchestrator = FakeOrchestrator(A2AAdapter.text("unused"))
    executor = CARBenchAgentExecutor(
        AgentConfig(llm="test/model"), orchestrator=orchestrator
    )
    queue = RecordingQueue()
    event_loop_thread = threading.get_ident()

    await executor.cancel(_request_context(), queue)  # type: ignore[arg-type]

    assert orchestrator.sessions.cancelled == ["ctx-executor-1"]
    assert orchestrator.sessions.thread_ids[0] != event_loop_thread
    assert orchestrator.sessions.active == {"ctx-other"}
    assert len(queue.events) == 1
    event = queue.events[0]
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.task_id == "task-executor-1"
    assert event.context_id == "ctx-executor-1"
    assert event.status.state == TaskState.TASK_STATE_CANCELED
    assert event.status.HasField("timestamp")


@pytest.mark.asyncio
async def test_cancel_linearizes_before_a_slow_tool_response_is_emitted() -> None:
    orchestrator = SlowToolOrchestrator()
    executor = CARBenchAgentExecutor(
        AgentConfig(llm="test/model"), orchestrator=orchestrator
    )
    context = _request_context()
    execute_queue = RecordingQueue()
    cancel_queue = RecordingQueue()

    execution = asyncio.create_task(
        executor.execute(context, execute_queue)  # type: ignore[arg-type]
    )
    assert await asyncio.to_thread(orchestrator.started.wait, 2)
    await executor.cancel(context, cancel_queue)  # type: ignore[arg-type]
    orchestrator.release.set()
    await execution

    assert len(cancel_queue.events) == 1
    assert cancel_queue.events[0].status.state == TaskState.TASK_STATE_CANCELED
    assert len(execute_queue.events) == 1
    response = execute_queue.events[0]
    assert [part.WhichOneof("content") for part in response.parts] == ["text"]
    assert "cancelled" in response.parts[0].text


def test_compatibility_kwargs_merge_into_config_without_losing_false_or_zero() -> None:
    base = AgentConfig(
        llm="base/model",
        temperature=0.8,
        thinking=True,
        interleaved_thinking=True,
    )
    orchestrator = FakeOrchestrator(A2AAdapter.text("unused"))

    executor = CARBenchAgentExecutor(
        base,
        orchestrator=orchestrator,
        model="override/model",
        temperature=0.0,
        thinking=False,
        interleaved_thinking=False,
    )

    assert executor.config.llm == "override/model"
    assert executor.config.temperature == 0.0
    assert executor.config.thinking is False
    assert executor.config.interleaved_thinking is False

    positional = CARBenchAgentExecutor("positional/model", orchestrator=orchestrator)
    assert positional.config.llm == "positional/model"
    assert positional.model == "positional/model"


def test_cli_boolean_and_zero_overrides_are_explicit() -> None:
    parser = build_argument_parser()
    base = AgentConfig(
        llm="base/model",
        temperature=0.7,
        thinking=True,
        interleaved_thinking=True,
        enable_critic=True,
    )
    arguments = parser.parse_args(
        [
            "--temperature",
            "0.0",
            "--no-thinking",
            "--no-interleaved-thinking",
            "--no-enable-critic",
        ]
    )

    configured = apply_cli_overrides(base, arguments)

    assert configured.temperature == 0.0
    assert configured.thinking is False
    assert configured.interleaved_thinking is False
    assert configured.enable_critic is False
    assert build_argument_parser().parse_args([]).host == "127.0.0.1"


def test_unspecified_cli_values_preserve_environment_baseline() -> None:
    base = AgentConfig(
        llm="environment/model",
        temperature=0.25,
        thinking=True,
        enable_critic=False,
    )

    configured = apply_cli_overrides(base, build_argument_parser().parse_args([]))

    assert configured == base


def test_cli_api_base_cannot_redirect_existing_provider_secret() -> None:
    base = AgentConfig(
        llm="openai/profile-model",
        provider="openai",
        api_base="https://profile.example/v1",
        api_key=SecretStr("profile-secret"),
    )
    arguments = build_argument_parser().parse_args(
        ["--api-base", "https://other.example/v1"]
    )

    with pytest.raises(ValueError, match="configure.*route atomically"):
        apply_cli_overrides(base, arguments)


def test_create_app_preserves_jsonrpc_and_agent_card_routes() -> None:
    orchestrator = FakeOrchestrator(A2AAdapter.text("unused"))
    executor = CARBenchAgentExecutor(
        AgentConfig(llm="test/model"), orchestrator=orchestrator
    )

    app = create_app(
        executor.config,
        card_url="http://127.0.0.1:19080/",
        executor=executor,
    )
    route_paths = {getattr(route, "path", None) for route in app.routes}

    assert "/" in route_paths
    assert "/.well-known/agent-card.json" in route_paths


def test_invalid_outbound_shape_fails_closed_to_text() -> None:
    malformed = SimpleNamespace(
        text="unsafe mixed output",
        tool_calls=({"tool_name": "set_fan", "arguments": {}},),
        metadata={},
        terminal=False,
    )
    orchestrator = FakeOrchestrator(malformed)
    executor = CARBenchAgentExecutor(
        AgentConfig(llm="test/model"), orchestrator=orchestrator
    )
    queue = RecordingQueue()

    async def run() -> None:
        await executor.execute(_request_context(), queue)  # type: ignore[arg-type]

    import asyncio

    asyncio.run(run())
    assert [part.WhichOneof("content") for part in queue.events[0].parts] == ["text"]
    assert "unsafe mixed output" not in queue.events[0].parts[0].text
