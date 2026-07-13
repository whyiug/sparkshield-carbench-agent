"""HTTP entry point for the CAR-Guard Track 1 agent."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from dotenv import load_dotenv
from pydantic import ValidationError
from starlette.applications import Starlette

from logging_utils import configure_logger
from track_1_agent_under_test.car_bench_agent import CARBenchAgentExecutor
from track_1_agent_under_test.car_guard.config import AgentConfig


logger = configure_logger(role="agent_under_test", context="server")


def prepare_agent_card(url: str) -> AgentCard:
    """Create the official discovery card without changing its public routes."""

    card = AgentCard(
        name="car_bench_agent",
        description="In-car voice assistant agent for CAR-bench evaluation",
        version="1.0.0",
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
    )
    interface = card.supported_interfaces.add()
    interface.url = url
    interface.protocol_binding = "JSONRPC"
    interface.protocol_version = "1.0"
    card.capabilities.streaming = False
    card.capabilities.push_notifications = False
    card.capabilities.extended_agent_card = False

    skill = card.skills.add()
    skill.id = "car_assistant"
    skill.name = "In-Car Voice Assistant"
    skill.description = (
        "Helps drivers with navigation, communication, charging, and in-car tasks"
    )
    skill.tags.extend(["benchmark", "car-bench", "voice-assistant"])
    return card


def create_app(
    config: AgentConfig | None = None,
    *,
    card_url: str = "http://127.0.0.1:8080/",
    executor: AgentExecutor | None = None,
) -> Starlette:
    """Build the Starlette app for production and isolated route tests."""

    runtime_config = config if config is not None else AgentConfig.from_env()
    card = prepare_agent_card(card_url)
    request_handler = DefaultRequestHandler(
        agent_executor=(
            executor if executor is not None else CARBenchAgentExecutor(runtime_config)
        ),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = create_jsonrpc_routes(
        request_handler,
        "/",
        enable_v0_3_compat=True,
    )
    routes.extend(create_agent_card_routes(card))
    return Starlette(routes=routes)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CAR-Guard Track 1 agent.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind; the container explicitly overrides this to 0.0.0.0",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--card-url", help="Externally visible AgentCard URL")
    parser.add_argument("--agent-llm", dest="llm", help="Primary model route")
    parser.add_argument("--provider", help="LiteLLM provider override")
    parser.add_argument("--api-base", help="Preconfigured provider endpoint")
    parser.add_argument("--deployment", help="Provider deployment identifier")
    parser.add_argument("--service-tier", help="Provider service tier")
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--structured-output-mode",
        choices=("auto", "json_schema", "json_object"),
    )
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--critic-model")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--soft-max-steps", type=int)
    parser.add_argument("--max-user-turns", type=int)
    parser.add_argument("--session-ttl-seconds", type=int)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--interleaved-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enable-critic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


_CONFIG_ARGUMENTS = (
    "llm",
    "provider",
    "api_base",
    "deployment",
    "service_tier",
    "temperature",
    "structured_output_mode",
    "reasoning_effort",
    "critic_model",
    "timeout_seconds",
    "max_retries",
    "soft_max_steps",
    "max_user_turns",
    "session_ttl_seconds",
    "max_sessions",
    "thinking",
    "interleaved_thinking",
    "enable_critic",
)


def apply_cli_overrides(
    config: AgentConfig, arguments: argparse.Namespace
) -> AgentConfig:
    """Apply explicit values only, preserving false and numeric zero."""

    return config.with_cli_overrides(
        **{
            name: getattr(arguments, name)
            for name in _CONFIG_ARGUMENTS
            if getattr(arguments, name) is not None
        }
    )


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        config = apply_cli_overrides(AgentConfig.from_env(), arguments)
    except (TypeError, ValueError, ValidationError) as exc:
        parser.error(f"invalid agent configuration: {type(exc).__name__}")

    card_url = (
        arguments.card_url
        if arguments.card_url is not None
        else f"http://{arguments.host}:{arguments.port}/"
    )
    app = create_app(config, card_url=card_url)
    logger.info(
        "Starting CAR-Guard agent",
        model=config.llm,
        temperature=config.temperature,
        thinking=config.thinking,
        enable_critic=config.enable_critic,
        host=arguments.host,
        port=arguments.port,
    )
    uvicorn.run(
        app,
        host=arguments.host,
        port=arguments.port,
        timeout_keep_alive=1000,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "apply_cli_overrides",
    "build_argument_parser",
    "create_app",
    "main",
    "prepare_agent_card",
]
