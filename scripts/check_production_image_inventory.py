#!/usr/bin/env python3
"""Verify the Track 1 application filesystem in a production image.

Only ``/home/carbench/app`` is owned by this project and inspected. Base-image
paths are deliberately out of scope. The application source is allowlisted,
while ``.venv`` may contain runtime dependencies but never CAR-bench,
evaluator, task/result data, credentials, or VCS metadata.

``--image`` does not run an image. It creates one stopped container, exports
its merged root filesystem, and removes that same container in a ``finally``
block. ``--tar``, ``--rootfs``, and ``--manifest`` are fully offline inputs.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TextIO


APP_ROOT = PurePosixPath("home/carbench/app")

REQUIRED_APPLICATION_PATHS = frozenset(
    {
        PurePosixPath(".venv/bin/python"),
        PurePosixPath("logging_utils.py"),
        PurePosixPath("tool_call_types.py"),
        PurePosixPath("turn_metrics.py"),
        PurePosixPath("track_1_agent_under_test/server.py"),
        PurePosixPath("track_1_agent_under_test/car_bench_agent.py"),
        PurePosixPath("track_1_agent_under_test/car_guard/__init__.py"),
        PurePosixPath("track_1_agent_under_test/car_guard/runtime/orchestrator.py"),
        PurePosixPath(
            "track_1_agent_under_test/car_guard/recipes/operation_recipes.yaml"
        ),
    }
)

_SHARED_RUNTIME_FILES = frozenset(
    {"logging_utils.py", "tool_call_types.py", "turn_metrics.py"}
)
_TRACK_RUNTIME_FILES = frozenset({"server.py", "car_bench_agent.py"})
_LICENSE_NAMES = frozenset(
    {"license", "license.txt", "notice", "notice.txt", "third_party_notices.txt"}
)
_VCS_PARTS = frozenset(
    {".git", ".github", ".gitignore", ".gitattributes", ".gitmodules"}
)
_DEVELOPMENT_PARTS = frozenset(
    {"docs", "doc", "documentation", "devtools", "scripts", "examples"}
)
_TEST_PARTS = frozenset({"test", "tests", "testing"})
_TASK_PARTS = frozenset({"task", "tasks"})
_RESULT_PARTS = frozenset({"result", "results", "output", "outputs"})
_GENERATED_PARTS = frozenset(
    {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)
_CREDENTIAL_NAMES = frozenset(
    {
        ".netrc",
        ".npmrc",
        ".pypirc",
        "api_key",
        "api_key.txt",
        "auth.json",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ed25519",
        "id_rsa",
        "service_account.json",
        "secrets.json",
        "token",
        "token.json",
    }
)
_PRIVATE_KEY_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
_DEPENDENCY_CREDENTIAL_NAMES = frozenset(
    {
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "id_dsa",
        "id_ed25519",
        "id_rsa",
        "private_key",
        "private_key.json",
        "secrets.json",
        "service_account.json",
        "token.json",
    }
)
_DEVELOPMENT_DOCUMENT_SUFFIXES = frozenset({".adoc", ".md", ".rst"})
_DEVELOPMENT_DOCUMENT_PREFIXES = (
    "changelog",
    "contributing",
    "development",
    "readme",
)
_GENERATED_SUFFIXES = frozenset({".log", ".pyc", ".pyo"})


@dataclass(frozen=True, slots=True)
class ImageEntry:
    path: str
    kind: str = "file"
    link_target: str | None = None


@dataclass(frozen=True, order=True, slots=True)
class InventoryFinding:
    path: PurePosixPath
    rule_id: str
    message: str

    def render(self) -> str:
        return f"/{self.path}: [{self.rule_id}] {self.message}"


def _normalize_image_path(value: str) -> PurePosixPath | None:
    raw = value.strip()
    if not raw or "\\" in raw:
        return None
    while raw.startswith("./"):
        raw = raw[2:]
    raw = raw.lstrip("/").rstrip("/")
    if not raw or raw == ".":
        return None
    parts = PurePosixPath(raw).parts
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return PurePosixPath(*parts)


def _under_app_root(path: PurePosixPath) -> PurePosixPath | None:
    root_parts = APP_ROOT.parts
    if path.parts[: len(root_parts)] != root_parts:
        return None
    relative_parts = path.parts[len(root_parts) :]
    return PurePosixPath(*relative_parts) if relative_parts else PurePosixPath(".")


def _is_credential(name: str) -> bool:
    folded = name.casefold()
    return (
        folded == ".env"
        or folded.startswith(".env.")
        or folded in _CREDENTIAL_NAMES
        or PurePosixPath(folded).suffix in _PRIVATE_KEY_SUFFIXES
    )


def _is_dependency_credential(name: str) -> bool:
    folded = name.casefold()
    return (
        folded == ".env"
        or folded.startswith(".env.")
        or folded in _DEPENDENCY_CREDENTIAL_NAMES
        or PurePosixPath(folded).suffix in {".p12", ".pfx"}
    )


def _is_benchmark_component(part: str) -> bool:
    folded = part.casefold()
    normalized = re.sub(r"[-_.]+", "-", folded)
    if folded in {"car_bench", "car-bench", "car_bench_ijcai"}:
        return True
    if normalized.startswith("car-bench-") and folded != "car_bench_agent.py":
        return True
    return normalized in {
        "car-bench-evaluator",
        "policy-evaluator",
        "reward-calculator",
        "user-simulator",
    }


def _is_evaluator_component(part: str) -> bool:
    normalized = re.sub(r"[-_.]+", "-", part.casefold())
    return normalized in {
        "agentbeats",
        "evaluator",
        "car-bench-evaluator",
        "policy-evaluator",
        "reward-calculator",
        "user-simulator",
    } or normalized.startswith("car-bench-evaluator-")


def _forbidden_finding(relative: PurePosixPath) -> InventoryFinding | None:
    if relative == PurePosixPath("."):
        return None
    parts = tuple(part.casefold() for part in relative.parts)
    name = parts[-1]
    in_venv = parts[0] == ".venv"

    if in_venv:
        if any(_is_evaluator_component(part) for part in parts):
            return InventoryFinding(
                APP_ROOT / relative,
                "forbidden-evaluator-runtime",
                "evaluator runtime code is present in the production application",
            )
        if any(_is_benchmark_component(part) for part in parts):
            return InventoryFinding(
                APP_ROOT / relative,
                "forbidden-benchmark-runtime",
                "CAR-bench core or non-production benchmark code is present",
            )
        if ".git" in parts:
            return InventoryFinding(
                APP_ROOT / relative,
                "forbidden-vcs-metadata",
                "Git repository metadata is present in a runtime dependency",
            )
        if any(_is_dependency_credential(part) for part in parts):
            return InventoryFinding(
                APP_ROOT / relative,
                "forbidden-credential-artifact",
                "a credential-bearing dependency artifact is present",
            )
        return None

    if any(part in _VCS_PARTS for part in parts):
        return InventoryFinding(
            APP_ROOT / relative,
            "forbidden-vcs-metadata",
            "Git or repository metadata is present in the production application",
        )
    if any(_is_credential(part) for part in parts):
        return InventoryFinding(
            APP_ROOT / relative,
            "forbidden-credential-artifact",
            "a credential-bearing filename is present in the production application",
        )
    if any(_is_evaluator_component(part) for part in parts):
        return InventoryFinding(
            APP_ROOT / relative,
            "forbidden-evaluator-runtime",
            "evaluator runtime code is present in the production application",
        )
    if any(_is_benchmark_component(part) for part in parts) or any(
        part in {"third_party", "track_2_agent_under_test"} for part in parts
    ):
        return InventoryFinding(
            APP_ROOT / relative,
            "forbidden-benchmark-runtime",
            "CAR-bench core or non-production benchmark code is present",
        )

    if any(part in _TASK_PARTS for part in parts):
        return InventoryFinding(
            APP_ROOT / relative,
            "forbidden-task-data",
            "benchmark task data is present in the production application",
        )
    if any(part in _RESULT_PARTS for part in parts):
        return InventoryFinding(
            APP_ROOT / relative,
            "forbidden-result-data",
            "benchmark result or output data is present in the production application",
        )

    if any(part in _TEST_PARTS for part in parts) or name.startswith("test_"):
        return InventoryFinding(
            APP_ROOT / relative,
            "forbidden-test-artifact",
            "project test code is present in the production application",
        )
    if (
        any(part in _DEVELOPMENT_PARTS for part in parts)
        or (
            PurePosixPath(name).suffix in _DEVELOPMENT_DOCUMENT_SUFFIXES
            and name not in _LICENSE_NAMES
        )
        or name.startswith(_DEVELOPMENT_DOCUMENT_PREFIXES)
    ):
        return InventoryFinding(
            APP_ROOT / relative,
            "forbidden-development-documentation",
            "development documentation is present in the production application",
        )
    if any(part in _GENERATED_PARTS for part in parts) or (
        PurePosixPath(name).suffix in _GENERATED_SUFFIXES
    ):
        return InventoryFinding(
            APP_ROOT / relative,
            "forbidden-generated-artifact",
            "cache, bytecode, or log output is present in the production application",
        )
    return None


def _is_allowlisted_application_path(relative: PurePosixPath) -> bool:
    if relative == PurePosixPath("."):
        return True
    parts = relative.parts
    if parts[0] == ".venv":
        return True
    if len(parts) == 1 and (
        parts[0] in _SHARED_RUNTIME_FILES or parts[0].casefold() in _LICENSE_NAMES
    ):
        return True
    if parts[0] != "track_1_agent_under_test":
        return False
    if len(parts) == 1:
        return True
    if len(parts) == 2 and parts[1] in _TRACK_RUNTIME_FILES:
        return True
    return len(parts) >= 2 and parts[1] == "car_guard"


def scan_image_entries(
    entries: Iterable[ImageEntry | str],
) -> list[InventoryFinding]:
    """Return deterministic findings for application-owned image entries."""

    findings: set[InventoryFinding] = set()
    seen_application_paths: set[PurePosixPath] = set()
    for raw_entry in entries:
        entry = (
            raw_entry if isinstance(raw_entry, ImageEntry) else ImageEntry(raw_entry)
        )
        path = _normalize_image_path(entry.path)
        if path is None:
            raw_path = entry.path.replace("\\", "/").lstrip("./")
            if APP_ROOT.as_posix() in raw_path:
                findings.add(
                    InventoryFinding(
                        APP_ROOT,
                        "invalid-application-path",
                        f"application image entry is not normalized: {entry.path!r}",
                    )
                )
            continue
        relative = _under_app_root(path)
        if relative is None:
            continue
        seen_application_paths.add(relative)

        forbidden = _forbidden_finding(relative)
        if forbidden is not None:
            findings.add(forbidden)
            continue
        if not _is_allowlisted_application_path(relative):
            findings.add(
                InventoryFinding(
                    APP_ROOT / relative,
                    "unexpected-application-path",
                    "path is outside the exact production application allowlist",
                )
            )
            continue
        if (
            entry.kind in {"symlink", "hardlink"}
            and relative.parts
            and relative.parts[0] != ".venv"
        ):
            findings.add(
                InventoryFinding(
                    APP_ROOT / relative,
                    "unexpected-application-link",
                    "application-owned source must not be supplied through a link",
                )
            )

    for required in sorted(REQUIRED_APPLICATION_PATHS):
        if required not in seen_application_paths:
            findings.add(
                InventoryFinding(
                    APP_ROOT / required,
                    "missing-required-application-path",
                    "required production runtime file is absent from the image",
                )
            )
    return sorted(findings)


def entries_from_manifest(stream: TextIO) -> Iterator[ImageEntry]:
    for line in stream:
        value = line.strip()
        if value and not value.startswith("#"):
            yield ImageEntry(value)


def entries_from_tar(path: Path) -> Iterator[ImageEntry]:
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if member.isdir():
                kind = "directory"
            elif member.issym():
                kind = "symlink"
            elif member.islnk():
                kind = "hardlink"
            else:
                kind = "file"
            yield ImageEntry(
                member.name, kind=kind, link_target=member.linkname or None
            )


def entries_from_rootfs(rootfs: Path) -> Iterator[ImageEntry]:
    if not rootfs.is_dir():
        raise ValueError(f"rootfs is not a directory: {rootfs}")
    for path in sorted(rootfs.rglob("*")):
        relative = path.relative_to(rootfs).as_posix()
        if path.is_symlink():
            yield ImageEntry(relative, kind="symlink", link_target=str(path.readlink()))
        elif path.is_dir():
            yield ImageEntry(relative, kind="directory")
        else:
            yield ImageEntry(relative)


def scan_docker_image(
    image: str, *, docker_binary: str = "docker"
) -> list[InventoryFinding]:
    """Export one newly-created stopped container and scan its merged rootfs."""

    created = subprocess.run(
        [docker_binary, "create", image],
        check=True,
        capture_output=True,
        text=True,
    )
    container_id = created.stdout.strip()
    if not container_id:
        raise RuntimeError("docker create did not return a container ID")
    cleanup_error: subprocess.CalledProcessError | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="car-guard-image-inventory-") as temp:
            archive = Path(temp) / "rootfs.tar"
            subprocess.run(
                [docker_binary, "export", "--output", str(archive), container_id],
                check=True,
            )
            return scan_image_entries(entries_from_tar(archive))
    finally:
        try:
            subprocess.run(
                [docker_binary, "rm", container_id],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            cleanup_error = exc
        if cleanup_error is not None and sys.exc_info()[0] is None:
            raise cleanup_error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="local or pullable Docker image reference")
    source.add_argument("--tar", type=Path, help="docker-export rootfs tar")
    source.add_argument("--rootfs", type=Path, help="extracted full rootfs directory")
    source.add_argument(
        "--manifest",
        type=Path,
        help="newline-delimited full image paths; use '-' for stdin",
    )
    parser.add_argument("--docker-binary", default="docker")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.image:
            findings = scan_docker_image(args.image, docker_binary=args.docker_binary)
        elif args.tar:
            findings = scan_image_entries(entries_from_tar(args.tar))
        elif args.rootfs:
            findings = scan_image_entries(entries_from_rootfs(args.rootfs))
        elif args.manifest == Path("-"):
            findings = scan_image_entries(entries_from_manifest(sys.stdin))
        else:
            assert args.manifest is not None
            with args.manifest.open(encoding="utf-8") as stream:
                findings = scan_image_entries(entries_from_manifest(stream))
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        tarfile.TarError,
    ) as exc:
        print(f"image inventory check failed: {exc}", file=sys.stderr)
        return 2

    if findings:
        for finding in findings:
            print(finding.render())
        print(f"production image inventory rejected: {len(findings)} finding(s)")
        return 1
    print("production image inventory passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
