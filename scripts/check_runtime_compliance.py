#!/usr/bin/env python3
"""Static compliance checks for the Track 1 production runtime.

This module intentionally has no project dependencies so it can run before the
agent environment is installed.  The default scan surface is an allowlist; a
new production path must be added deliberately instead of being picked up from
the repository wholesale.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# These are the only repository source paths eligible for the Track 1 runtime.
# Development assets and other benchmark participants are deliberately absent.
PRODUCTION_PATH_ALLOWLIST = (
    "src/track_1_agent_under_test",
    "src/logging_utils.py",
    "src/tool_call_types.py",
    "src/turn_metrics.py",
)
PRODUCTION_DEPENDENCY_EXTRA = "track-1-agent"

_TEXT_FILE_SUFFIXES = frozenset(
    {
        ".cfg",
        ".ini",
        ".j2",
        ".jinja",
        ".json",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_IGNORED_GENERATED_PARTS = frozenset({".mypy_cache", ".pytest_cache", "__pycache__"})
_BANNED_RUNTIME_PATH_PARTS = frozenset(
    {
        "devtools",
        "evaluator",
        "output",
        "outputs",
        "results",
        "tasks",
        "tests",
        "third_party",
    }
)


@dataclass(frozen=True)
class TextRule:
    rule_id: str
    pattern: re.Pattern[str]
    description: str


def _rule(rule_id: str, expression: str, description: str) -> TextRule:
    return TextRule(rule_id, re.compile(expression, re.IGNORECASE), description)


# The first group is the zero-tolerance list from the competition contract.
# The second group covers the remaining benchmark-only metadata named by the
# same contract.  Patterns match identifiers and prompt/config text alike.
_BANNED_TEXT_RULES = (
    _rule("benchmark-task-type", r"\btask_types?\b", "benchmark task type metadata"),
    _rule("removed-component", r"\bremoved_parts?\b", "removed component metadata"),
    _rule(
        "ground-truth-actions",
        r"\bground[\s_-]+truth(?:[\s_-]+actions?)?\b",
        "ground-truth action data",
    ),
    _rule("task-split", r"\btask_splits?\b", "benchmark split metadata"),
    _rule(
        "reward-calculator",
        r"\breward_calculators?\b",
        "benchmark reward calculator",
    ),
    _rule(
        "policy-evaluator",
        r"\bpolicy_evaluators?\b",
        "benchmark policy evaluator",
    ),
    _rule(
        "public-task-id",
        (
            r"\b(?:hallucination|disambiguation)_[-a-z0-9]+\b|"
            r"\bbase_[0-9]+\b|(?<![a-z0-9])_base[0-9]+_"
        ),
        "public benchmark task identifier",
    ),
    _rule("benchmark-env", r"\bcar_bench\.envs\b", "benchmark environment API"),
    _rule(
        "missing-component-label",
        r"\bmissing[\s_-]+component[\s_-]+labels?\b",
        "missing component label",
    ),
    _rule(
        "reference-trajectory",
        r"\breference[\s_-]+trajector(?:y|ies)\b",
        "reference trajectory data",
    ),
    _rule(
        "evaluator-state",
        r"\bevaluator[\s_-]+internal[\s_-]+state\b",
        "evaluator internal state",
    ),
    _rule(
        "score-signal",
        r"\bsubscores?\b|\bpass[\s_-]+fail\b|\brewards?\b",
        "score signal",
    ),
    _rule(
        "public-result-lookup",
        r"\bpublic[\s_-]+result[\s_-]+lookup\b|\btask[\s_-]+phrase[\s_-]+lookup\b",
        "public result or task phrase lookup",
    ),
)

_BANNED_MODULE_ROOTS = frozenset({"car_bench"})
_BANNED_MODULE_SEGMENTS = frozenset(
    {"evaluator", "policy_evaluator", "reward_calculator", "user_simulator"}
)
_BANNED_DISTRIBUTIONS = frozenset(
    {
        "car-bench",
        "car-bench-evaluator",
        "policy-evaluator",
        "reward-calculator",
        "user-simulator",
    }
)


@dataclass(frozen=True, order=True)
class Finding:
    path: Path
    line: int
    column: int
    rule_id: str
    message: str

    def render(self, repository_root: Path) -> str:
        try:
            display_path = self.path.relative_to(repository_root)
        except ValueError:
            display_path = self.path
        return (
            f"{display_path}:{self.line}:{self.column}: [{self.rule_id}] {self.message}"
        )


def production_paths(repository_root: Path) -> tuple[Path, ...]:
    """Resolve the production source allowlist against ``repository_root``."""

    return tuple(repository_root / relative for relative in PRODUCTION_PATH_ALLOWLIST)


def _relative_path(repository_root: Path, path: Path) -> Path | None:
    try:
        return path.absolute().relative_to(repository_root.absolute())
    except ValueError:
        return None


def _is_negative_test_path(relative_path: Path) -> bool:
    # Only the repository-level tests tree receives this exception.  A tests
    # directory smuggled into production sources remains a compliance failure.
    return bool(relative_path.parts) and relative_path.parts[0].casefold() == "tests"


def _iter_candidates(path: Path) -> Iterator[Path]:
    if path.is_symlink() or path.is_file():
        yield path
        return
    if not path.exists():
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.is_symlink():
            yield candidate
            continue
        if candidate.is_dir():
            continue
        yield candidate


def _is_generated_file(relative_path: Path) -> bool:
    return any(part in _IGNORED_GENERATED_PARTS for part in relative_path.parts) or (
        relative_path.suffix in {".pyc", ".pyo"}
    )


def _is_supported_text_file(path: Path) -> bool:
    return (
        path.suffix.casefold() in _TEXT_FILE_SUFFIXES
        or path.name == "uv.lock"
        or path.name.startswith("Dockerfile")
    )


def _line_and_column(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    previous_newline = text.rfind("\n", 0, offset)
    return line, offset - previous_newline


def _module_is_banned(module: str) -> bool:
    parts = tuple(part for part in module.split(".") if part)
    return bool(parts) and (
        parts[0] in _BANNED_MODULE_ROOTS
        or any(part in _BANNED_MODULE_SEGMENTS for part in parts)
    )


def _import_findings(path: Path, tree: ast.AST) -> list[Finding]:
    findings: list[Finding] = []

    def add(node: ast.AST, module: str) -> None:
        findings.append(
            Finding(
                path=path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                rule_id="banned-runtime-import",
                message=f"runtime import crosses the production boundary: {module!r}",
            )
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_is_banned(alias.name):
                    add(node, alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            candidates = [module]
            candidates.extend(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
            )
            for candidate in candidates:
                if candidate and _module_is_banned(candidate):
                    add(node, candidate)
                    break
        elif isinstance(node, ast.Call) and node.args:
            function_name = ""
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            if function_name not in {"__import__", "import_module"}:
                continue
            module_arg = node.args[0]
            if (
                isinstance(module_arg, ast.Constant)
                and isinstance(module_arg.value, str)
                and _module_is_banned(module_arg.value)
            ):
                add(node, module_arg.value)
    return findings


def _scan_text_file(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [Finding(path, 1, 1, "non-utf8-runtime-file", str(exc))]
    except OSError as exc:
        return [Finding(path, 1, 1, "unreadable-runtime-file", str(exc))]

    findings: list[Finding] = []
    for rule in _BANNED_TEXT_RULES:
        for match in rule.pattern.finditer(text):
            line, column = _line_and_column(text, match.start())
            findings.append(
                Finding(
                    path, line, column, rule.rule_id, f"contains {rule.description}"
                )
            )

    if path.suffix.casefold() == ".py":
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            findings.append(
                Finding(
                    path,
                    exc.lineno or 1,
                    exc.offset or 1,
                    "invalid-python",
                    exc.msg,
                )
            )
        else:
            findings.extend(_import_findings(path, tree))
    return findings


def scan_paths(
    repository_root: Path,
    paths: Iterable[Path],
    *,
    include_tests: bool = False,
) -> list[Finding]:
    """Scan selected paths without inspecting the dependency manifest."""

    repository_root = repository_root.absolute()
    findings: list[Finding] = []
    seen: set[Path] = set()

    for selected_path in paths:
        selected_path = (
            selected_path
            if selected_path.is_absolute()
            else repository_root / selected_path
        )
        for candidate in _iter_candidates(selected_path):
            candidate = candidate.absolute()
            if candidate in seen:
                continue
            seen.add(candidate)

            relative_path = _relative_path(repository_root, candidate)
            if relative_path is None:
                findings.append(
                    Finding(
                        candidate,
                        1,
                        1,
                        "outside-repository",
                        "allowlisted path escapes repository",
                    )
                )
                continue
            if _is_generated_file(relative_path):
                continue
            if _is_negative_test_path(relative_path) and not include_tests:
                continue
            if not candidate.exists():
                findings.append(
                    Finding(
                        candidate,
                        1,
                        1,
                        "missing-allowlisted-path",
                        "allowlisted path does not exist",
                    )
                )
                continue
            if candidate.is_symlink():
                findings.append(
                    Finding(
                        candidate,
                        1,
                        1,
                        "runtime-symlink",
                        "production source may not be a symlink",
                    )
                )
                continue

            path_parts = {part.casefold() for part in relative_path.parent.parts}
            banned_parts = sorted(path_parts & _BANNED_RUNTIME_PATH_PARTS)
            if banned_parts:
                findings.append(
                    Finding(
                        candidate,
                        1,
                        1,
                        "forbidden-runtime-path",
                        f"production path contains forbidden component {banned_parts[0]!r}",
                    )
                )
                continue
            if (
                candidate.name == ".env"
                or candidate.name.startswith(".env.")
                or candidate.suffix == ".log"
            ):
                findings.append(
                    Finding(
                        candidate,
                        1,
                        1,
                        "forbidden-runtime-artifact",
                        "secret or log artifact in production source",
                    )
                )
                continue
            if not _is_supported_text_file(candidate):
                findings.append(
                    Finding(
                        candidate,
                        1,
                        1,
                        "unexpected-runtime-file",
                        "file type is not in the production source allowlist",
                    )
                )
                continue
            findings.extend(_scan_text_file(candidate))

    return sorted(set(findings))


def _normalize_distribution_name(dependency: str) -> str | None:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", dependency)
    if match is None:
        return None
    return re.sub(r"[-_.]+", "-", match.group(1)).casefold()


def scan_production_dependencies(repository_root: Path) -> list[Finding]:
    """Check direct dependencies installed for the production Track 1 extra."""

    manifest_path = repository_root / "pyproject.toml"
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = tomllib.loads(manifest_text)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        return [Finding(manifest_path, 1, 1, "invalid-production-manifest", str(exc))]

    project = manifest.get("project", {})
    base_dependencies = project.get("dependencies", [])
    optional_dependencies = project.get("optional-dependencies", {})
    production_dependencies = optional_dependencies.get(PRODUCTION_DEPENDENCY_EXTRA)
    if not isinstance(base_dependencies, list) or not isinstance(
        production_dependencies, list
    ):
        return [
            Finding(
                manifest_path,
                1,
                1,
                "invalid-production-dependency-set",
                f"project dependencies and {PRODUCTION_DEPENDENCY_EXTRA!r} extra must be lists",
            )
        ]

    findings: list[Finding] = []
    for dependency in [*base_dependencies, *production_dependencies]:
        if not isinstance(dependency, str):
            findings.append(
                Finding(
                    manifest_path,
                    1,
                    1,
                    "invalid-production-dependency",
                    repr(dependency),
                )
            )
            continue
        distribution = _normalize_distribution_name(dependency)
        if distribution not in _BANNED_DISTRIBUTIONS:
            continue
        offset = manifest_text.find(dependency)
        line, column = _line_and_column(manifest_text, max(offset, 0))
        findings.append(
            Finding(
                manifest_path,
                line,
                column,
                "banned-production-dependency",
                f"production dependency crosses benchmark boundary: {distribution!r}",
            )
        )
    return findings


def scan_repository(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    paths: Iterable[Path] | None = None,
    include_tests: bool = False,
    check_dependencies: bool = True,
) -> list[Finding]:
    """Run the complete repository compliance scan."""

    selected_paths = (
        tuple(paths) if paths is not None else production_paths(repository_root)
    )
    findings = scan_paths(repository_root, selected_paths, include_tests=include_tests)
    if check_dependencies:
        findings.extend(scan_production_dependencies(repository_root))
    return sorted(set(findings))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="paths to scan (defaults to the production allowlist)",
    )
    parser.add_argument(
        "--root", type=Path, default=REPOSITORY_ROOT, help="repository root"
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="scan repository tests too (negative assertions are excluded by default)",
    )
    parser.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="skip the production dependency manifest check",
    )
    parser.add_argument(
        "--print-allowlist",
        action="store_true",
        help="print the production source allowlist and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repository_root = args.root.resolve()
    if args.print_allowlist:
        for path in PRODUCTION_PATH_ALLOWLIST:
            print(path)
        return 0

    findings = scan_repository(
        repository_root,
        paths=args.paths or None,
        include_tests=args.include_tests,
        check_dependencies=not args.skip_dependencies,
    )
    if findings:
        for finding in findings:
            print(finding.render(repository_root), file=sys.stderr)
        print(
            f"Runtime compliance scan failed with {len(findings)} finding(s).",
            file=sys.stderr,
        )
        return 1

    selected_paths = args.paths or list(PRODUCTION_PATH_ALLOWLIST)
    print(
        f"Runtime compliance scan passed ({len(selected_paths)} allowlisted path(s))."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
