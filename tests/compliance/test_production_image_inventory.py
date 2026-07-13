from __future__ import annotations

import io
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import scripts.check_production_image_inventory as inventory
from scripts.check_production_image_inventory import (
    APP_ROOT,
    REQUIRED_APPLICATION_PATHS,
    ImageEntry,
    entries_from_tar,
    main,
    scan_image_entries,
)


def full_path(relative: str | Path) -> str:
    return f"/{APP_ROOT}/{relative}"


def minimal_inventory(*extra: ImageEntry | str) -> list[ImageEntry | str]:
    required = [full_path(path) for path in REQUIRED_APPLICATION_PATHS]
    return [*required, *extra]


def rule_ids(entries: list[ImageEntry | str]) -> set[str]:
    return {finding.rule_id for finding in scan_image_entries(entries)}


class ProductionImageInventoryTest(unittest.TestCase):
    def test_exact_application_allowlist_accepts_runtime_source_and_venv(
        self,
    ) -> None:
        entries = minimal_inventory(
            f"/{APP_ROOT}",
            full_path(".venv/lib/python3.12/site-packages/httpx/__init__.py"),
            full_path(".venv/lib/python3.12/site-packages/a2a/server/tasks/__init__.py"),
            full_path(".venv/lib/python3.12/site-packages/litellm/types/results.py"),
            full_path(".venv/lib/python3.12/site-packages/certifi/cacert.pem"),
            full_path(".venv/lib/python3.12/site-packages/litellm/proxy/.gitignore"),
            full_path(
                ".venv/lib/python3.12/site-packages/litellm/proxy/auth/public_key.pem"
            ),
            full_path("track_1_agent_under_test/car_guard/domain/types.py"),
            "/usr/share/doc/python3.12/README.md",
            "/opt/unrelated/results/public.json",
        )
        self.assertEqual(scan_image_entries(entries), [])

    def test_forbidden_application_content_is_rejected(self) -> None:
        cases = (
            (
                "track_1_agent_under_test/car_guard/car_bench/core.py",
                "forbidden-benchmark-runtime",
            ),
            (
                "track_1_agent_under_test/car_guard/evaluator/server.py",
                "forbidden-evaluator-runtime",
            ),
            (
                "track_1_agent_under_test/car_guard/tasks/base.json",
                "forbidden-task-data",
            ),
            (
                "track_1_agent_under_test/car_guard/results/run.json",
                "forbidden-result-data",
            ),
            (
                "track_1_agent_under_test/car_guard/tests/test_runtime.py",
                "forbidden-test-artifact",
            ),
            (".env", "forbidden-credential-artifact"),
            ("token.json", "forbidden-credential-artifact"),
            (".git/config", "forbidden-vcs-metadata"),
            ("docs/design.md", "forbidden-development-documentation"),
            (
                "track_1_agent_under_test/car_guard/README.txt",
                "forbidden-development-documentation",
            ),
            ("third_party/car-bench/core.py", "forbidden-benchmark-runtime"),
        )
        for path, expected_rule in cases:
            with self.subTest(path=path):
                self.assertIn(
                    expected_rule, rule_ids(minimal_inventory(full_path(path)))
                )

    def test_venv_is_not_a_benchmark_or_evaluator_blind_spot(self) -> None:
        paths = (
            ".venv/lib/python3.12/site-packages/car_bench/__init__.py",
            ".venv/lib/python3.12/site-packages/car_bench-0.1.0.dist-info/METADATA",
            ".venv/lib/python3.12/site-packages/car_bench_evaluator/server.py",
            ".venv/lib/python3.12/site-packages/evaluator/__init__.py",
        )
        for path in paths:
            with self.subTest(path=path):
                findings = scan_image_entries(minimal_inventory(full_path(path)))
                self.assertTrue(findings)
                self.assertIn(
                    findings[0].rule_id,
                    {
                        "forbidden-benchmark-runtime",
                        "forbidden-evaluator-runtime",
                    },
                )

    def test_venv_still_rejects_repository_and_credential_artifacts(self) -> None:
        cases = (
            ".venv/lib/python3.12/site-packages/package/.git/config",
            ".venv/lib/python3.12/site-packages/package/.env",
            ".venv/lib/python3.12/site-packages/package/token.json",
        )
        for path in cases:
            with self.subTest(path=path):
                self.assertTrue(
                    scan_image_entries(minimal_inventory(full_path(path)))
                )

    def test_unexpected_and_missing_application_files_are_rejected(self) -> None:
        entries = minimal_inventory(full_path("pyproject.toml"))
        entries.remove(full_path("track_1_agent_under_test/server.py"))
        self.assertEqual(
            rule_ids(entries),
            {
                "missing-required-application-path",
                "unexpected-application-path",
            },
        )

    def test_non_normalized_application_path_is_rejected_not_ignored(self) -> None:
        findings = scan_image_entries(
            minimal_inventory("/home/carbench/app/../app/tasks/hidden.json")
        )
        self.assertIn("invalid-application-path", {item.rule_id for item in findings})

    def test_source_links_rejected_but_venv_interpreter_link_allowed(self) -> None:
        entries = minimal_inventory(
            ImageEntry(
                full_path("track_1_agent_under_test/car_guard/config_link.py"),
                kind="symlink",
                link_target="/tmp/config.py",
            )
        )
        entries = [
            ImageEntry(path, kind="symlink", link_target="/usr/local/bin/python")
            if path == full_path(".venv/bin/python")
            else path
            for path in entries
        ]
        self.assertEqual(rule_ids(entries), {"unexpected-application-link"})

    def test_tar_reader_preserves_link_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive_path = Path(temp) / "rootfs.tar"
            with tarfile.open(archive_path, mode="w") as archive:
                for path in minimal_inventory():
                    info = tarfile.TarInfo(str(path).lstrip("/"))
                    info.size = 0
                    archive.addfile(info, io.BytesIO())
                link = tarfile.TarInfo(
                    full_path("track_1_agent_under_test/car_guard/linked.py").lstrip(
                        "/"
                    )
                )
                link.type = tarfile.SYMTYPE
                link.linkname = "/tmp/linked.py"
                archive.addfile(link)

            self.assertIn(
                "unexpected-application-link",
                rule_ids(list(entries_from_tar(archive_path))),
            )

    def test_manifest_cli_has_stable_success_and_failure_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "image-paths.txt"
            manifest.write_text(
                "\n".join(str(path) for path in minimal_inventory()),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["--manifest", str(manifest)]), 0)
            self.assertIn("passed", output.getvalue())

            with manifest.open("a", encoding="utf-8") as stream:
                stream.write(f"\n{full_path('tasks/hidden.json')}\n")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["--manifest", str(manifest)]), 1)
            self.assertIn("forbidden-task-data", output.getvalue())
            self.assertIn("rejected", output.getvalue())

    def test_image_mode_only_creates_exports_and_removes_stopped_container(
        self,
    ) -> None:
        commands: list[list[str]] = []

        def fake_run(command: list[str], **kwargs):
            del kwargs
            commands.append(command)
            verb = command[1]
            if verb == "create":
                return subprocess.CompletedProcess(command, 0, stdout="container-id\n")
            if verb == "export":
                archive_path = Path(command[command.index("--output") + 1])
                with tarfile.open(archive_path, mode="w") as archive:
                    for path in minimal_inventory():
                        info = tarfile.TarInfo(str(path).lstrip("/"))
                        info.size = 0
                        archive.addfile(info, io.BytesIO())
                return subprocess.CompletedProcess(command, 0)
            if verb == "rm":
                return subprocess.CompletedProcess(command, 0, stdout="container-id\n")
            raise AssertionError(f"unexpected docker command: {command}")

        with patch.object(inventory.subprocess, "run", side_effect=fake_run):
            self.assertEqual(
                inventory.scan_docker_image("example.test/car-guard@sha256:abc"),
                [],
            )
        self.assertEqual(
            [command[1] for command in commands], ["create", "export", "rm"]
        )
        self.assertTrue(
            all(command[1] not in {"run", "start", "stop"} for command in commands)
        )
        self.assertEqual(commands[-1][-1], "container-id")


if __name__ == "__main__":
    unittest.main()
