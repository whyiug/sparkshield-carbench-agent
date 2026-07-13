"""Deterministic checks for concise, voice-safe outbound text."""

from __future__ import annotations

import re
import unicodedata


class TTSViolation(ValueError):
    pass


# Newlines are rejected separately, so list syntax only needs to be detected at
# the start of the single line. This avoids treating ordinary speech such as
# "level 2. Shall I continue?" as a numbered Markdown list.
_MARKDOWN = re.compile(r"^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)|[`*_<>]")
_INTERNAL = re.compile(
    r"\b(?:get|set|open_close|navigation)_[a-z0-9_]+\b|"
    r"\bPOL-\d+\b|\b(?:tool_call_id|task[_ -]?type)\b|\.[a-z_]+\b",
    re.IGNORECASE,
)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_UNSAFE_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
# Mask only an unambiguous two-label domain. Extra dot suffixes remain visible
# to the internal-identifier scan instead of being laundered as email text.
_ASCII_EMAIL = re.compile(
    r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?![A-Za-z0-9!#$%&'*+/=?^_`{|}~-])"
)


class TTSGuard:
    def __init__(self, *, max_characters: int = 700) -> None:
        self.max_characters = max_characters

    def violations(self, text: str) -> list[str]:
        issues: list[str] = []
        security_scan = unicodedata.normalize("NFKC", text)
        if not text.strip():
            issues.append("empty")
        if "\n" in text or "\r" in text:
            issues.append("multiline")
        if _MARKDOWN.search(security_scan):
            issues.append("markdown")
        internal_scan = _ASCII_EMAIL.sub("email-address", security_scan)
        if _INTERNAL.search(internal_scan):
            issues.append("internal_identifier")
        if _CONTROL.search(text):
            issues.append("control_character")
        if any(
            unicodedata.category(character) in _UNSAFE_UNICODE_CATEGORIES
            for character in text
        ):
            issues.append("unsafe_unicode")
        if len(text) > self.max_characters:
            issues.append("too_long")
        return issues

    def ensure(self, text: str) -> str:
        raw_issues = self.violations(text)
        if raw_issues:
            raise TTSViolation(", ".join(raw_issues))
        normalized = " ".join(text.split())
        issues = self.violations(normalized)
        if issues:
            raise TTSViolation(", ".join(issues))
        return normalized
