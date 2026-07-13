from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.validate_submission import (
    IMAGE_PLACEHOLDER,
    load_scenario,
    validate_submission,
)


class SubmissionMaterialsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario = load_scenario(Path("submission/scenario.toml"))

    def test_final_scenario_is_resolved_and_valid(self) -> None:
        report = validate_submission(self.scenario, require_resolved_image=True)

        self.assertTrue(report.valid, report.errors)
        self.assertFalse(report.warnings)
        self.assertNotEqual(
            self.scenario["agent_under_test"]["image"], IMAGE_PLACEHOLDER
        )

    def test_release_mode_still_rejects_placeholder(self) -> None:
        candidate = copy.deepcopy(self.scenario)
        candidate["agent_under_test"]["image"] = IMAGE_PLACEHOLDER
        report = validate_submission(candidate, require_resolved_image=True)

        self.assertFalse(report.valid)
        self.assertTrue(any("placeholder" in item for item in report.errors))

    def test_release_mode_accepts_digest_and_rejects_literal_secret(self) -> None:
        candidate = copy.deepcopy(self.scenario)
        candidate["agent_under_test"]["image"] = (
            "ghcr.io/example-org/car-guard@sha256:" + "a" * 64
        )
        accepted = validate_submission(candidate, require_resolved_image=True)
        self.assertTrue(accepted.valid, accepted.errors)

        candidate["agent_under_test"]["env"]["AGENT_API_KEY"] = "not-a-real-key"
        rejected = validate_submission(candidate, require_resolved_image=True)
        self.assertFalse(rejected.valid)
        self.assertTrue(any("literal values" in item for item in rejected.errors))

    def test_provider_is_required_but_native_api_base_is_optional(self) -> None:
        env = self.scenario["agent_under_test"]["env"]
        self.assertEqual(env["AGENT_API_BASE"], "${AGENT_API_BASE:-}")

        candidate = copy.deepcopy(self.scenario)
        candidate["agent_under_test"]["env"]["AGENT_PROVIDER"] = (
            "${AGENT_PROVIDER:-}"
        )
        report = validate_submission(candidate)

        self.assertFalse(report.valid)
        self.assertTrue(any("AGENT_PROVIDER" in item for item in report.errors))


if __name__ == "__main__":
    unittest.main()
