from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts.check_runtime_compliance import (
    PRODUCTION_PATH_ALLOWLIST,
    scan_paths,
    scan_production_dependencies,
    scan_repository,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class RuntimeComplianceTest(unittest.TestCase):
    def _write(self, root: Path, relative_path: str, content: str) -> Path:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return path

    def test_repository_production_allowlist_passes(self) -> None:
        self.assertEqual(scan_repository(REPOSITORY_ROOT), [])

    def test_banned_runtime_markers_are_rejected(self) -> None:
        markers = (
            "task_type",
            "removed_part",
            "ground_truth_actions",
            "task_split",
            "task_splits",
            "reward",
            "reward_calculator",
            "policy_evaluator",
            "hallucination_17",
            "disambiguation_2",
            "base_0",
            "_BASE86_RANGE_PATTERN",
            "car_bench.envs",
            "missing_component_label",
            "reference_trajectory",
            "evaluator_internal_state",
            "subscore",
            "pass_fail",
            "public_result_lookup",
            "task_phrase_lookup",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._write(root, "src/agent/runtime.py", "VALUE = 'safe'\n")
            for marker in markers:
                with self.subTest(marker=marker):
                    source.write_text(f"VALUE = {marker!r}\n", encoding="utf-8")
                    self.assertTrue(scan_paths(root, [source]))

    def test_common_base_url_identifier_is_not_a_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._write(
                root, "src/agent/runtime.py", "base_url = 'https://example.test'\n"
            )

            self.assertEqual(scan_paths(root, [source]), [])

    def test_production_lock_is_scanned_as_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lock = self._write(root, "src/agent/uv.lock", "version = 1\n")
            self.assertEqual(scan_paths(root, [lock]), [])

            lock.write_text("value = 'reward_calculator'\n", encoding="utf-8")
            self.assertTrue(scan_paths(root, [lock]))

    def test_static_evaluator_and_benchmark_imports_are_rejected(self) -> None:
        sources = (
            "from evaluator.server import app\n",
            "from car_bench.envs.base import Env\n",
            "import src.evaluator.car_bench_evaluator\n",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._write(root, "src/agent/runtime.py", sources[0])
            for content in sources:
                with self.subTest(content=content):
                    source.write_text(content, encoding="utf-8")
                    rule_ids = {
                        finding.rule_id for finding in scan_paths(root, [source])
                    }
                    self.assertIn("banned-runtime-import", rule_ids)

    def test_constant_dynamic_evaluator_import_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._write(
                root,
                "src/agent/runtime.py",
                """
                import importlib
                module = importlib.import_module("src.evaluator.server")
                """,
            )

            rule_ids = {finding.rule_id for finding in scan_paths(root, [source])}
            self.assertIn("banned-runtime-import", rule_ids)

    def test_repository_test_negative_assertions_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            negative_test = self._write(
                root,
                "tests/compliance/test_negative.py",
                "VALUE = 'removed_part'\n",
            )

            self.assertEqual(scan_paths(root, [root]), [])
            self.assertTrue(scan_paths(root, [negative_test], include_tests=True))

    def test_production_nested_test_directory_is_not_an_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            production_test = self._write(
                root,
                "src/agent/tests/test_negative.py",
                "VALUE = 'removed_part'\n",
            )

            rule_ids = {
                finding.rule_id for finding in scan_paths(root, [production_test])
            }
            self.assertIn("forbidden-runtime-path", rule_ids)

    def test_allowlist_has_no_development_or_evaluator_tree(self) -> None:
        self.assertEqual(
            set(PRODUCTION_PATH_ALLOWLIST),
            {
                "src/track_1_agent_under_test",
                "src/logging_utils.py",
                "src/tool_call_types.py",
                "src/turn_metrics.py",
            },
        )
        joined = "\n".join(PRODUCTION_PATH_ALLOWLIST)
        self.assertNotIn("src/evaluator", joined)
        self.assertNotIn("track_2", joined)
        self.assertNotIn("third_party", joined)

    def test_production_dependency_extra_rejects_benchmark_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write(
                root,
                "pyproject.toml",
                """
                [project]
                dependencies = []

                [project.optional-dependencies]
                track-1-agent = ["car-bench>=1"]
                """,
            )

            rule_ids = {
                finding.rule_id for finding in scan_production_dependencies(root)
            }
            self.assertIn("banned-production-dependency", rule_ids)

    def test_development_only_dependency_extra_is_not_installed_for_track_1(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write(
                root,
                "pyproject.toml",
                """
                [project]
                dependencies = []

                [project.optional-dependencies]
                track-1-agent = ["httpx>=0.28"]
                car-bench-evaluator = ["car-bench"]
                """,
            )

            self.assertEqual(scan_production_dependencies(root), [])


if __name__ == "__main__":
    unittest.main()
