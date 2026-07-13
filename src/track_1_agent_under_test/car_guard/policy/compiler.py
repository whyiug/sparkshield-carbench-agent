"""Compile the current runtime system policy into deterministic rules."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import Field

from ..domain import DomainModel
from .rules import (
    GENERAL_LIVE_SCHEMA,
    GENERAL_POI_PRESENTATION,
    GENERAL_REACTIVE_ONLY,
    GENERAL_ROUTE_PRESENTATION,
    GENERAL_TOOL_DESCRIPTION,
    GENERAL_TTS,
    NUMBERED_POLICY_IDS,
    CompiledRule,
    PolicyDecision,
    PolicyRequest,
    PolicyRuleCategory,
    PolicyRuleSource,
    category_for_rule,
    evaluate_rules,
    infer_general_rules,
    infer_numbered_rules_from_semantics,
)


_NUMBER_BY_ID = {rule_id[-3:]: rule_id for rule_id in NUMBERED_POLICY_IDS}
_NUMBERED_LABEL = re.compile(
    r"(?:(?:LLM|AUT)-POL:|POL[-:]?)?"
    r"(?P<number>002|004|005|007|008|009|010|011|012|013|014|016|017|018|019|021|022|023|024):",
    re.IGNORECASE,
)

_GENERAL_EXPLANATIONS = {
    GENERAL_TTS: "Responses must be natural, first-person, user-language, and TTS-safe.",
    GENERAL_REACTIVE_ONLY: (
        "State-changing calls require a user action request, confirmation, or selection."
    ),
    GENERAL_LIVE_SCHEMA: (
        "Only live tools and parameters may be emitted; substitutions require user choice."
    ),
    GENERAL_TOOL_DESCRIPTION: (
        "Tool descriptions and parameter schemas are binding runtime constraints."
    ),
}


class CompiledPolicy(DomainModel):
    """Immutable-by-convention snapshot of one turn's system policy."""

    source_digest: str
    source_system_text: str
    numbered_rules: list[CompiledRule] = Field(default_factory=list)
    general_rules: list[CompiledRule] = Field(default_factory=list)
    current_location: dict[str, Any] | None = None
    current_datetime: dict[str, Any] | None = None

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(rule.rule_id for rule in self.numbered_rules)

    @property
    def general_rule_ids(self) -> tuple[str, ...]:
        return tuple(rule.rule_id for rule in self.general_rules)

    def has_rule(self, rule_id: str) -> bool:
        return rule_id in self.rule_ids or rule_id in self.general_rule_ids

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        return evaluate_rules(
            request,
            active_rule_ids=self.rule_ids,
            general_rule_ids=self.general_rule_ids,
            current_location=self.current_location,
            current_datetime=self.current_datetime,
        )


class PolicyCompiler:
    """Compile only what the current system text says.

    Numbered policies may be recognized by their labels or by sufficiently
    specific semantics. This supports formatting changes without turning the
    local reference wiki into a second source of runtime truth.
    """

    def compile(self, system_policy: str) -> CompiledPolicy:
        if not isinstance(system_policy, str):
            raise TypeError("system_policy must be a string")

        labeled_sources = _labeled_rule_sources(system_policy)
        semantic_ids = infer_numbered_rules_from_semantics(system_policy)
        active_ids = set(labeled_sources).union(semantic_ids)
        paragraphs = _paragraphs(system_policy)

        numbered_rules: list[CompiledRule] = []
        for rule_id in NUMBERED_POLICY_IDS:
            if rule_id not in active_ids:
                continue
            source = (
                PolicyRuleSource.NUMBERED_LABEL
                if rule_id in labeled_sources
                else PolicyRuleSource.SEMANTIC_TEXT
            )
            source_text = labeled_sources.get(rule_id)
            if source_text is None:
                source_text = _semantic_source_excerpt(rule_id, paragraphs)
            numbered_rules.append(
                CompiledRule(
                    rule_id=rule_id,
                    category=category_for_rule(rule_id),
                    source=source,
                    source_text=source_text,
                )
            )

        inferred_general = infer_general_rules(system_policy)
        general_rules: list[CompiledRule] = []
        general_order = (
            GENERAL_TTS,
            GENERAL_REACTIVE_ONLY,
            GENERAL_LIVE_SCHEMA,
            GENERAL_TOOL_DESCRIPTION,
            GENERAL_ROUTE_PRESENTATION,
            GENERAL_POI_PRESENTATION,
        )
        for rule_id in general_order:
            if rule_id not in inferred_general:
                continue
            is_invariant = rule_id in _GENERAL_EXPLANATIONS
            general_rules.append(
                CompiledRule(
                    rule_id=rule_id,
                    category=PolicyRuleCategory.GENERAL,
                    source=(
                        PolicyRuleSource.RUNTIME_INVARIANT
                        if is_invariant
                        else PolicyRuleSource.SEMANTIC_TEXT
                    ),
                    source_text=(
                        _GENERAL_EXPLANATIONS[rule_id]
                        if is_invariant
                        else _general_source_excerpt(rule_id, paragraphs)
                    ),
                )
            )

        return CompiledPolicy(
            source_digest=hashlib.sha256(system_policy.encode("utf-8")).hexdigest(),
            source_system_text=system_policy,
            numbered_rules=numbered_rules,
            general_rules=general_rules,
            current_location=_extract_json_assignment(
                system_policy, "CURRENT_LOCATION"
            ),
            current_datetime=_extract_current_datetime(system_policy),
        )


def _paragraphs(text: str) -> list[str]:
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]


def _labeled_rule_sources(text: str) -> dict[str, str]:
    sources: dict[str, str] = {}
    for paragraph in _paragraphs(text):
        matches = list(_NUMBERED_LABEL.finditer(paragraph))
        for match in matches:
            rule_id = _NUMBER_BY_ID[match.group("number")]
            sources.setdefault(rule_id, paragraph)
    return sources


def _semantic_source_excerpt(rule_id: str, paragraphs: list[str]) -> str:
    for paragraph in paragraphs:
        if rule_id in infer_numbered_rules_from_semantics(paragraph):
            return paragraph
    # Some rules span adjacent Markdown paragraphs. Preserve the actual input
    # rather than substituting a bundled/default policy clause.
    return "\n\n".join(paragraphs)


def _general_source_excerpt(rule_id: str, paragraphs: list[str]) -> str:
    for paragraph in paragraphs:
        if rule_id in infer_general_rules(paragraph):
            return paragraph
    return "\n\n".join(paragraphs)


def _extract_json_assignment(text: str, name: str) -> dict[str, Any] | None:
    marker = re.search(rf"\b{re.escape(name)}\s*=\s*", text)
    if marker is None:
        return None
    tail = text[marker.end() :].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def _extract_current_datetime(text: str) -> dict[str, Any] | None:
    current = _extract_json_assignment(text, "CURRENT_DATETIME")
    legacy = _extract_json_assignment(text, "DATETIME")
    if current is not None and legacy is not None and current != legacy:
        return None
    return current if current is not None else legacy


__all__ = ["CompiledPolicy", "PolicyCompiler"]
