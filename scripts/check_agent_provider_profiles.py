#!/usr/bin/env python3
"""Validate local CAR-Guard provider profiles without making network requests."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from track_1_agent_under_test.car_guard.config import (  # noqa: E402
    AgentConfig,
    SUPPORTED_PROVIDER_PROFILES,
)


def _profile_values(
    values: Mapping[str, str], profile: str
) -> tuple[dict[str, str], list[str]]:
    prefix = SUPPORTED_PROVIDER_PROFILES[profile]
    names = (
        f"{prefix}_MODEL_NAME",
        f"{prefix}_API_KEY",
        f"{prefix}_BASE_URL",
    )
    missing = [name for name in names if not values.get(name, "").strip()]
    selected = {
        name: values[name]
        for name in names
        if name in values and values[name].strip()
    }
    selected["AGENT_PROVIDER_PROFILE"] = profile
    return selected, missing


def inspect_profiles(values: Mapping[str, str]) -> tuple[list[str], bool]:
    """Return redacted status lines and whether every configured profile is safe."""

    lines: list[str] = []
    valid = True
    for profile in SUPPORTED_PROVIDER_PROFILES:
        selected, missing = _profile_values(values, profile)
        configured_count = 3 - len(missing)
        if configured_count == 0:
            lines.append(f"{profile}: SKIPPED (missing {', '.join(missing)})")
            continue
        if missing:
            valid = False
            lines.append(f"{profile}: BLOCKED (incomplete profile configuration)")
            continue
        prefix = SUPPORTED_PROVIDER_PROFILES[profile]
        base = values[f"{prefix}_BASE_URL"]
        parsed = urlsplit(base)
        remote_http = parsed.scheme == "http" and parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }
        try:
            AgentConfig.from_env(selected)
        except (TypeError, ValueError):
            valid = False
            reason = (
                "remote HTTP endpoint"
                if remote_http
                else "invalid profile configuration"
            )
            lines.append(f"{profile}: BLOCKED ({reason})")
        else:
            lines.append(f"{profile}: READY (secure transport, no network probe)")
    return lines, valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    raw_values = dotenv_values(args.env_file)
    values = {
        name: value for name, value in raw_values.items() if isinstance(value, str)
    }
    lines, valid = inspect_profiles(values)
    for line in lines:
        print(line)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
