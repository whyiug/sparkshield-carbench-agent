"""Goal-conditioned extraction of evidence from matched tool results.

The validator intentionally has no expected-response schema.  A caller must
declare both the evidence need on the pending call and an extractor for that
need.  This keeps unrelated fields, including unrelated ``unknown`` values,
out of runtime decisions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

from ..a2a.result_matching import MatchedToolResult
from ..domain.evidence import (
    Evidence,
    EvidenceNeed,
    EvidenceSourceKind,
    EvidenceStatus,
    EvidenceStore,
)


JsonPathPart: TypeAlias = str | int
EvidenceTransform: TypeAlias = Callable[[Any], Any]


class ResultExecutionStatus(str, Enum):
    """Coarse execution state; every non-success state is insufficient."""

    SUCCESS = "success"
    NON_SUCCESS = "non_success"
    MALFORMED = "malformed"
    IGNORED = "ignored"


@dataclass(frozen=True, slots=True)
class ExtractorSpec:
    """An explicit, non-recursive path and optional deterministic transform."""

    path: str | Sequence[JsonPathPart] = ()
    transform: EvidenceTransform | None = None
    allowed_tool_names: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if any(not name.strip() for name in self.allowed_tool_names):
            raise ValueError("allowed extractor tool names cannot be blank")

    def compiled_path(self) -> tuple[JsonPathPart, ...]:
        if isinstance(self.path, str):
            return _parse_json_path(self.path)
        parts = tuple(self.path)
        for part in parts:
            if isinstance(part, bool) or not isinstance(part, (str, int)):
                raise TypeError("JSON path parts must be strings or integer indexes")
            if isinstance(part, str) and (not part or "*" in part):
                raise ValueError("JSON paths cannot contain empty or wildcard keys")
            if isinstance(part, int) and part < 0:
                raise ValueError("JSON path indexes must be non-negative")
        return parts


ExtractorInput: TypeAlias = (
    ExtractorSpec | str | Sequence[JsonPathPart] | EvidenceTransform
)


@dataclass(frozen=True, slots=True)
class EvidenceExtractionIssue:
    need_id: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SemanticValidationResult:
    """Evidence and non-sensitive diagnostics produced for one tool result."""

    execution_status: ResultExecutionStatus
    considered_need_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    issues: tuple[EvidenceExtractionIssue, ...] = ()
    insufficient_need_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ParsedResult:
    status: ResultExecutionStatus
    value: Any
    issue_code: str | None = None
    issue_message: str | None = None


_MISSING = object()
_UNKNOWN_STRINGS = {
    "unknown",
    "null",
    "none",
    "n/a",
    "na",
    "unavailable",
    "not available",
}
_FAILURE_STATUSES = {
    "error",
    "failed",
    "failure",
    "non_success",
    "not_successful",
    "denied",
    "timeout",
    "cancelled",
    "canceled",
}
_JSON_SCALAR_RE = re.compile(
    r"^(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)$"
)
_BRACKET_PART_RE = re.compile(r"\[(?:(\d+)|\"([^\"]+)\"|'([^']+)')\]")


def _parse_json_path(path: str) -> tuple[JsonPathPart, ...]:
    """Compile a small JSONPath subset without wildcard or recursive search."""

    if path == "$":
        return ()
    if not path.startswith("$"):
        raise ValueError("JSON paths must start with '$'")

    parts: list[JsonPathPart] = []
    index = 1
    while index < len(path):
        if path[index] == ".":
            index += 1
            end = index
            while end < len(path) and path[end] not in ".[":
                end += 1
            key = path[index:end]
            if not key or "*" in key or key.strip() != key:
                raise ValueError("JSON paths require concrete, non-empty keys")
            parts.append(key)
            index = end
            continue
        if path[index] == "[":
            match = _BRACKET_PART_RE.match(path, index)
            if match is None:
                raise ValueError("JSON paths require an integer or quoted bracket key")
            integer, double_quoted, single_quoted = match.groups()
            if integer is not None:
                parts.append(int(integer))
            else:
                parts.append(double_quoted or single_quoted or "")
            index = match.end()
            continue
        raise ValueError(f"unsupported JSON path syntax at offset {index}")
    return tuple(parts)


def _normalise_extractor(value: ExtractorInput) -> ExtractorSpec:
    if isinstance(value, ExtractorSpec):
        value.compiled_path()
        return value
    if callable(value):
        return ExtractorSpec(transform=value)
    if isinstance(value, str):
        spec = ExtractorSpec(path=value)
        spec.compiled_path()
        return spec
    spec = ExtractorSpec(path=value)
    spec.compiled_path()
    return spec


def _is_empty_or_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in _UNKNOWN_STRINGS
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return not value
    return False


def _has_failure_marker(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("success") is False or value.get("ok") is False:
        return True
    status = value.get("status")
    if isinstance(status, str) and status.strip().lower() in _FAILURE_STATUSES:
        return True
    error = value.get("error", _MISSING)
    return (
        error is not _MISSING
        and error is not None
        and not (isinstance(error, str) and not error.strip())
        and not (isinstance(error, (list, tuple, dict, set, frozenset)) and not error)
    )


def _parse_result(content: Any) -> _ParsedResult:
    if isinstance(content, (bytes, bytearray)):
        try:
            content = bytes(content).decode("utf-8")
        except UnicodeDecodeError:
            return _ParsedResult(
                ResultExecutionStatus.MALFORMED,
                None,
                "malformed_result",
                "tool result is not valid UTF-8",
            )

    parsed = content
    if isinstance(content, str):
        stripped = content.strip()
        if not stripped:
            parsed = ""
        else:
            looks_like_json = stripped[0] in '{["' or bool(
                _JSON_SCALAR_RE.fullmatch(stripped.lower())
            )
            if looks_like_json:
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    return _ParsedResult(
                        ResultExecutionStatus.MALFORMED,
                        None,
                        "malformed_result",
                        "tool result contains malformed JSON",
                    )
            else:
                parsed = stripped

    if _has_failure_marker(parsed):
        return _ParsedResult(
            ResultExecutionStatus.NON_SUCCESS,
            parsed,
            "non_success_result",
            "tool result explicitly reports non-success",
        )
    return _ParsedResult(ResultExecutionStatus.SUCCESS, parsed)


def _extract_path(value: Any, path: tuple[JsonPathPart, ...]) -> Any:
    current = value
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, Sequence) or isinstance(
                current, (str, bytes, bytearray)
            ):
                return _MISSING
            if part >= len(current):
                return _MISSING
            current = current[part]
        else:
            if not isinstance(current, Mapping) or part not in current:
                return _MISSING
            current = current[part]
    return current


class SemanticResultValidator:
    """Extract only declared, currently pending evidence needs."""

    def __init__(self, extractors: Mapping[str, ExtractorInput] | None = None) -> None:
        self._extractors = {
            key: _normalise_extractor(value)
            for key, value in (extractors or {}).items()
        }

    def register_extractor(
        self, need_id_or_proposition: str, extractor: ExtractorInput
    ) -> None:
        if not need_id_or_proposition.strip():
            raise ValueError("extractor key cannot be blank")
        if need_id_or_proposition in self._extractors:
            raise ValueError(
                f"extractor is already registered: {need_id_or_proposition}"
            )
        self._extractors[need_id_or_proposition] = _normalise_extractor(extractor)

    def _extractor_for(self, need: EvidenceNeed) -> ExtractorSpec | None:
        assert need.need_id is not None
        return self._extractors.get(need.need_id) or self._extractors.get(
            need.proposition
        )

    def _pending_declared_needs(
        self,
        result: MatchedToolResult,
        store: EvidenceStore,
    ) -> list[EvidenceNeed]:
        if result.pending_call is None or result.duplicate:
            return []
        pending = {
            need.need_id: need
            for need in store.pending_needs
            if "tool_result" in need.acceptable_sources
        }
        declared: list[EvidenceNeed] = []
        seen: set[str] = set()
        for need_id in result.pending_call.evidence_needs:
            if need_id in seen:
                continue
            seen.add(need_id)
            need = pending.get(need_id)
            if need is not None:
                declared.append(need)
        return declared

    def validate_result(
        self,
        result: MatchedToolResult,
        *,
        evidence_store: EvidenceStore,
        source_turn_id: str,
    ) -> SemanticValidationResult:
        needs = self._pending_declared_needs(result, evidence_store)
        if not needs:
            return SemanticValidationResult(
                execution_status=ResultExecutionStatus.IGNORED,
                considered_need_ids=(),
                evidence=(),
            )

        parsed = _parse_result(result.content)
        call_id = result.provided_call_id or result.pending_call.call_id  # type: ignore[union-attr]
        observations: list[Evidence] = []
        issues: list[EvidenceExtractionIssue] = []
        sufficient_need_ids: set[str] = set()

        for need in needs:
            assert need.need_id is not None
            status = EvidenceStatus.KNOWN
            value: Any = None
            issue_code: str | None = None
            issue_message: str | None = None

            if parsed.status in {
                ResultExecutionStatus.NON_SUCCESS,
                ResultExecutionStatus.MALFORMED,
            }:
                status = EvidenceStatus.ERROR
                issue_code = parsed.issue_code
                issue_message = parsed.issue_message
            else:
                extractor = self._extractor_for(need)
                if extractor is None:
                    issues.append(
                        EvidenceExtractionIssue(
                            need_id=need.need_id,
                            code="extractor_not_registered",
                            message="no explicit extractor is registered for this need",
                        )
                    )
                    continue
                pending_tool = result.pending_call.tool_name  # type: ignore[union-attr]
                if (
                    extractor.allowed_tool_names
                    and pending_tool not in extractor.allowed_tool_names
                ):
                    status = EvidenceStatus.ERROR
                    value = None
                    issue_code = "extractor_tool_mismatch"
                    issue_message = "the registered extractor rejects this tool source"
                else:
                    value = _extract_path(parsed.value, extractor.compiled_path())
                    if value is _MISSING:
                        status = EvidenceStatus.ERROR
                        value = None
                        issue_code = "extractor_path_missing"
                        issue_message = "the registered path is absent from this result"
                    else:
                        try:
                            if extractor.transform is not None:
                                value = extractor.transform(value)
                        except Exception as exc:
                            status = EvidenceStatus.ERROR
                            value = None
                            issue_code = "extractor_transform_failed"
                            issue_message = (
                                "registered transform rejected the value: "
                                f"{type(exc).__name__}"
                            )
                        if status is EvidenceStatus.KNOWN and _is_empty_or_unknown(value):
                            status = EvidenceStatus.UNKNOWN
                            value = None
                            issue_code = "insufficient_value"
                            issue_message = "the extracted value is empty or unknown"

            if issue_code is not None:
                issues.append(
                    EvidenceExtractionIssue(
                        need_id=need.need_id,
                        code=issue_code,
                        message=issue_message or "evidence is insufficient",
                    )
                )
            observations.append(
                Evidence(
                    proposition=need.proposition,
                    value=value,
                    status=status,
                    source_kind=EvidenceSourceKind.TOOL,
                    source_turn_id=source_turn_id,
                    source_tool_call_id=call_id,
                    confidence=1.0 if status is EvidenceStatus.KNOWN else 0.0,
                )
            )
            if status is EvidenceStatus.KNOWN:
                sufficient_need_ids.add(need.need_id)

        evidence_store.update(observations)
        return SemanticValidationResult(
            execution_status=parsed.status,
            considered_need_ids=tuple(need.need_id for need in needs if need.need_id),
            evidence=tuple(observations),
            issues=tuple(issues),
            insufficient_need_ids=tuple(
                need.need_id
                for need in needs
                if need.need_id is not None and need.need_id not in sufficient_need_ids
            ),
        )

    def validate(
        self,
        result: MatchedToolResult,
        *,
        evidence_store: EvidenceStore,
        source_turn_id: str,
    ) -> SemanticValidationResult:
        """Alias kept for callers that treat validators as single-result stages."""

        return self.validate_result(
            result,
            evidence_store=evidence_store,
            source_turn_id=source_turn_id,
        )

    def validate_results(
        self,
        results: Sequence[MatchedToolResult],
        *,
        evidence_store: EvidenceStore,
        source_turn_id: str,
    ) -> list[SemanticValidationResult]:
        return [
            self.validate_result(
                result,
                evidence_store=evidence_store,
                source_turn_id=source_turn_id,
            )
            for result in results
        ]


__all__ = [
    "EvidenceExtractionIssue",
    "ExtractorSpec",
    "ResultExecutionStatus",
    "SemanticResultValidator",
    "SemanticValidationResult",
]
