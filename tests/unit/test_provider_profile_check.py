import os
import subprocess
import sys
from pathlib import Path

from scripts.check_agent_provider_profiles import inspect_profiles


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_profile_check_never_outputs_secret_or_endpoint() -> None:
    secret = "profile-secret-must-not-appear"
    endpoint = "https://provider.example/v1"
    lines, valid = inspect_profiles(
        {
            "XOPDSV32IN_MODEL_NAME": "profile-model",
            "XOPDSV32IN_API_KEY": secret,
            "XOPDSV32IN_BASE_URL": endpoint,
        }
    )

    output = "\n".join(lines)
    assert valid
    assert "xopdsv32in: READY" in output
    assert secret not in output
    assert endpoint not in output


def test_profile_check_blocks_remote_http_without_echoing_it() -> None:
    endpoint = "http://203.0.113.10:3006/v1"
    lines, valid = inspect_profiles(
        {
            "GEMINI_2_5_PRO_MODEL_NAME": "profile-model",
            "GEMINI_2_5_PRO_API_KEY": "profile-secret",
            "GEMINI_2_5_PRO_BASE_URL": endpoint,
        }
    )

    output = "\n".join(lines)
    assert not valid
    assert "gemini-2.5-pro: BLOCKED (remote HTTP endpoint)" in output
    assert endpoint not in output


def test_profile_check_blocks_partially_configured_profile() -> None:
    lines, valid = inspect_profiles(
        {"XOPDSV32IN_API_KEY": "profile-secret-must-not-appear"}
    )

    output = "\n".join(lines)
    assert not valid
    assert "xopdsv32in: BLOCKED (incomplete profile configuration)" in output
    assert "profile-secret-must-not-appear" not in output


def test_profile_check_script_runs_without_pythonpath(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "XOP3QWEN235B_MODEL_NAME=profile-model\n"
        "XOP3QWEN235B_API_KEY=profile-secret\n"
        "XOP3QWEN235B_BASE_URL=https://provider.example/v1\n"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/check_agent_provider_profiles.py"),
            "--env-file",
            str(env_file),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "xop3qwen235b: READY" in result.stdout
    assert "profile-secret" not in result.stdout + result.stderr
    assert "provider.example" not in result.stdout + result.stderr
