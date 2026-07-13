from __future__ import annotations

import subprocess
import sys


def test_orchestrator_logger_satisfies_configured_formatter_contract() -> None:
    code = """
from track_1_agent_under_test.car_guard.runtime.orchestrator import logger
from logging_utils import configure_logger
configure_logger(role="agent_under_test", context="server")
logger.warning("CAR-Guard safe failure: {}", "LLMFailure")
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "CAR-Guard safe failure: LLMFailure" in completed.stderr
    assert "Logging error" not in completed.stderr
    assert "KeyError" not in completed.stderr
