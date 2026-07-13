#!/usr/bin/env python3
"""Validate the Track 1 hidden-evaluation submission scenario."""

from __future__ import annotations

import argparse
import copy
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = REPOSITORY_ROOT / "submission/scenario.toml"
OFFICIAL_EVALUATOR_IMAGE = "ghcr.io/car-bench/car-bench-evaluator:latest"
IMAGE_PLACEHOLDER = "ghcr.io/REPLACE_OWNER/car-guard@sha256:REPLACE_WITH_64_HEX_DIGEST"

_RESOLVED_IMAGE_RE = re.compile(
    r"^ghcr\.io/"
    r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
    r"(?:/[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?)+"
    r"@sha256:[0-9a-f]{64}$"
)
_ENV_REFERENCE_RE = re.compile(
    r"^\$\{(?P<name>[A-Z][A-Z0-9_]*)(?:(?P<operator>:-|:\?)(?P<value>[^{}]*))?\}$"
)

_EXPECTED_TOP_LEVEL = frozenset({"evaluator", "agent_under_test", "config"})
_EXPECTED_EVALUATOR_KEYS = frozenset({"image", "env"})
_EXPECTED_AGENT_KEYS = frozenset({"image", "env"})
_EXPECTED_CONFIG = {
    "num_trials": 3,
    "task_split": "hidden",
    "tasks_base_num_tasks": -1,
    "tasks_hallucination_num_tasks": -1,
    "tasks_disambiguation_num_tasks": -1,
    "max_steps": 50,
}
_EXPECTED_EVALUATOR_ENV = {
    "GEMINI_API_KEY": "${GEMINI_API_KEY:?Set GEMINI_API_KEY}",
    "LOGURU_LEVEL": "${LOGURU_LEVEL:-INFO}",
}
_EXPECTED_AGENT_ENV = {
    "AGENT_LLM": "${AGENT_LLM:?Set AGENT_LLM}",
    "AGENT_PROVIDER": "${AGENT_PROVIDER:?Set AGENT_PROVIDER}",
    "AGENT_API_BASE": "${AGENT_API_BASE:-}",
    "AGENT_API_KEY": "${AGENT_API_KEY:?Set AGENT_API_KEY}",
    "AGENT_DEPLOYMENT": "${AGENT_DEPLOYMENT:-}",
    "AGENT_SERVICE_TIER": "${AGENT_SERVICE_TIER:-}",
    "AGENT_REASONING_EFFORT": "${AGENT_REASONING_EFFORT:-}",
    "AGENT_TEMPERATURE": "${AGENT_TEMPERATURE:-0}",
    "AGENT_STRUCTURED_OUTPUT_MODE": "${AGENT_STRUCTURED_OUTPUT_MODE:-json_schema}",
    "AGENT_THINKING": "${AGENT_THINKING:-false}",
    "AGENT_INTERLEAVED_THINKING": "${AGENT_INTERLEAVED_THINKING:-false}",
    "AGENT_ENABLE_CRITIC": "${AGENT_ENABLE_CRITIC:-false}",
    "AGENT_CRITIC_MODEL": "${AGENT_CRITIC_MODEL:-}",
    "AGENT_CRITIC_PROVIDER": "${AGENT_CRITIC_PROVIDER:-}",
    "AGENT_CRITIC_API_BASE": "${AGENT_CRITIC_API_BASE:-}",
    "AGENT_CRITIC_API_KEY": "${AGENT_CRITIC_API_KEY:-}",
    "AGENT_TIMEOUT_SECONDS": "${AGENT_TIMEOUT_SECONDS:-90}",
    "AGENT_MAX_RETRIES": "${AGENT_MAX_RETRIES:-2}",
    "AGENT_SOFT_MAX_STEPS": "${AGENT_SOFT_MAX_STEPS:-49}",
    "AGENT_MAX_USER_TURNS": "${AGENT_MAX_USER_TURNS:-24}",
    "AGENT_SESSION_TTL_SECONDS": "${AGENT_SESSION_TTL_SECONDS:-1800}",
    "AGENT_MAX_SESSIONS": "${AGENT_MAX_SESSIONS:-256}",
    "LOGURU_LEVEL": "${LOGURU_LEVEL:-INFO}",
}
_SENSITIVE_ENV_SUFFIXES = ("_API_KEY", "_PASSWORD", "_SECRET", "_TOKEN")
_OPTIONAL_SENSITIVE_ENV = frozenset({"AGENT_CRITIC_API_KEY"})


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def _table(data: Mapping[str, Any], key: str, errors: list[str]) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"[{key}] must be a TOML table")
        return {}
    return value


def _check_exact_keys(
    table: Mapping[str, Any], expected: frozenset[str], label: str, errors: list[str]
) -> None:
    actual = frozenset(table)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        errors.append(f"{label} is missing keys: {', '.join(missing)}")
    if unexpected:
        errors.append(f"{label} has unexpected keys: {', '.join(unexpected)}")


def _check_env(
    value: Any,
    *,
    label: str,
    expected: Mapping[str, str],
    agent: bool,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be a TOML table")
        return

    missing = sorted(frozenset(expected) - frozenset(value))
    if missing:
        errors.append(f"{label} is missing keys: {', '.join(missing)}")

    unexpected = sorted(frozenset(value) - frozenset(expected))
    if unexpected:
        errors.append(f"{label} has unexpected keys: {', '.join(unexpected)}")

    if agent and "GEMINI_API_KEY" in value:
        errors.append(
            "agent_under_test.env must not receive evaluator-only GEMINI_API_KEY"
        )

    for name, raw_value in sorted(value.items()):
        if not isinstance(name, str) or not isinstance(raw_value, str):
            errors.append(f"{label}.{name} must be a string env reference")
            continue
        if agent and name != "LOGURU_LEVEL" and not name.startswith("AGENT_"):
            errors.append(f"{label}.{name} is not an AGENT_* runtime setting")
        match = _ENV_REFERENCE_RE.fullmatch(raw_value)
        if match is None:
            errors.append(
                f"{label}.{name} must use Compose-style interpolation; "
                "literal values are forbidden"
            )
        elif match.group("name") != name:
            errors.append(
                f"{label}.{name} references {match.group('name')} instead of itself"
            )
        elif (
            name.endswith(_SENSITIVE_ENV_SUFFIXES)
            and name not in _OPTIONAL_SENSITIVE_ENV
            and match.group("operator") != ":?"
        ):
            errors.append(
                f"{label}.{name} is sensitive and must use a required-variable "
                "reference without a default"
            )

    for name, expected_value in expected.items():
        actual_value = value.get(name)
        if actual_value is not None and actual_value != expected_value:
            errors.append(f"{label}.{name} must be {expected_value!r}")


def validate_submission(
    data: Mapping[str, Any],
    *,
    require_resolved_image: bool = False,
) -> ValidationReport:
    """Validate parsed TOML data without reading environment variables."""

    errors: list[str] = []
    warnings: list[str] = []

    _check_exact_keys(data, _EXPECTED_TOP_LEVEL, "document", errors)
    evaluator = _table(data, "evaluator", errors)
    agent = _table(data, "agent_under_test", errors)
    config = _table(data, "config", errors)

    _check_exact_keys(evaluator, _EXPECTED_EVALUATOR_KEYS, "[evaluator]", errors)
    _check_exact_keys(agent, _EXPECTED_AGENT_KEYS, "[agent_under_test]", errors)
    _check_exact_keys(config, frozenset(_EXPECTED_CONFIG), "[config]", errors)

    if evaluator.get("image") != OFFICIAL_EVALUATOR_IMAGE:
        errors.append("evaluator.image must remain " + repr(OFFICIAL_EVALUATOR_IMAGE))

    image = agent.get("image")
    if image == IMAGE_PLACEHOLDER:
        message = "agent_under_test.image still contains the GHCR digest placeholder"
        if require_resolved_image:
            errors.append(message)
        else:
            warnings.append(message)
    elif not isinstance(image, str) or _RESOLVED_IMAGE_RE.fullmatch(image) is None:
        errors.append(
            "agent_under_test.image must be a lowercase ghcr.io reference pinned "
            "to @sha256:<64 lowercase hex characters>"
        )

    _check_env(
        evaluator.get("env"),
        label="evaluator.env",
        expected=_EXPECTED_EVALUATOR_ENV,
        agent=False,
        errors=errors,
    )
    _check_env(
        agent.get("env"),
        label="agent_under_test.env",
        expected=_EXPECTED_AGENT_ENV,
        agent=True,
        errors=errors,
    )

    for key, expected in _EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            errors.append(f"config.{key} must be {expected!r}")

    for forbidden in ("user_model", "policy_evaluator_model"):
        if forbidden in config:
            errors.append(f"config.{forbidden} belongs to the official evaluator")

    return ValidationReport(tuple(errors), tuple(warnings))


def load_scenario(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, nargs="?", default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--require-resolved-image",
        action="store_true",
        help="fail if the checked scenario still has the documented image placeholder",
    )
    parser.add_argument(
        "--image-ref",
        help="validate this effective image reference without rewriting the template",
    )
    args = parser.parse_args(argv)

    try:
        data = load_scenario(args.scenario)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"submission validation failed: {exc}", file=sys.stderr)
        return 2

    if args.image_ref:
        data = copy.deepcopy(data)
        agent = data.setdefault("agent_under_test", {})
        if isinstance(agent, dict):
            agent["image"] = args.image_ref

    report = validate_submission(
        data, require_resolved_image=args.require_resolved_image
    )
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if not report.valid:
        print(
            f"submission validation failed with {len(report.errors)} error(s)",
            file=sys.stderr,
        )
        return 1

    print(f"submission scenario is valid: {args.scenario}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
