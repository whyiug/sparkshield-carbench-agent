"""Exact-scope, one-shot confirmation latch management."""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import Enum
from typing import Literal

from pydantic import Field

from ..domain import (
    Confirmation,
    ConfirmationScope,
    ConfirmationStatus,
    DomainModel,
    OfficialToolCall,
)


class LatchResolutionKind(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    AMBIGUOUS = "ambiguous"
    UNRECOGNIZED = "unrecognized"
    NO_PENDING = "no_pending"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


class LatchResolution(DomainModel):
    kind: LatchResolutionKind
    confirmation: Confirmation | None = None
    affected_confirmation_ids: list[str] = Field(default_factory=list)
    candidate_confirmation_ids: list[str] = Field(default_factory=list)


class ConfirmationLatch:
    """Session-local confirmations bound to complete ordered action bundles.

    Positive replies are accepted only when exactly one scope is identified.
    A generic positive reply therefore cannot authorize one of several pending
    actions. Confirmed entries remain one-shot authorizations until consumed,
    invalidated, cancelled, or expired.
    """

    def __init__(self, confirmations: Iterable[Confirmation] = ()) -> None:
        self._confirmations: dict[str, Confirmation] = {}
        for confirmation in confirmations:
            self._insert(confirmation)

    @property
    def confirmations(self) -> list[Confirmation]:
        return list(self._confirmations.values())

    @property
    def pending(self) -> list[Confirmation]:
        return [
            confirmation
            for confirmation in self._confirmations.values()
            if confirmation.status is ConfirmationStatus.PENDING
        ]

    @property
    def active(self) -> list[Confirmation]:
        return [
            confirmation
            for confirmation in self._confirmations.values()
            if confirmation.status
            in {ConfirmationStatus.PENDING, ConfirmationStatus.CONFIRMED}
        ]

    def get(self, confirmation_id: str) -> Confirmation:
        return self._confirmations[confirmation_id]

    def arm(
        self,
        scope: ConfirmationScope,
        *,
        confirmation_id: str | None = None,
    ) -> Confirmation:
        for existing in self.active:
            if existing.scope == scope:
                return existing
        identifier = confirmation_id or f"confirmation-{scope.fingerprint[:20]}"
        confirmation = Confirmation(confirmation_id=identifier, scope=scope)
        self._insert(confirmation)
        return confirmation

    def expire(self, current_user_turn: int) -> list[Confirmation]:
        expired: list[Confirmation] = []
        for confirmation in list(self.active):
            if current_user_turn <= confirmation.scope.expires_after_user_turn:
                continue
            updated = self._transition(
                confirmation,
                status=ConfirmationStatus.EXPIRED,
                current_user_turn=current_user_turn,
            )
            expired.append(updated)
        return expired

    def resolve(
        self,
        user_response: str,
        *,
        current_user_turn: int,
        source_turn_id: str,
        response_kind_override: Literal["positive", "negative"] | None = None,
        confirmation_id: str | None = None,
        goal_ids: list[str] | None = None,
        ordered_actions: list[OfficialToolCall] | None = None,
        confirmation_prompt: str | None = None,
    ) -> LatchResolution:
        expired = self.expire(current_user_turn)
        response_kind = response_kind_override or _classify_response(user_response)
        if (
            response_kind == "other"
            and ordered_actions is not None
            and _is_safe_frozen_prerequisite_restatement(
                user_response, ordered_actions
            )
        ):
            response_kind = "positive"

        if response_kind == "negative":
            status = (
                ConfirmationStatus.CANCELLED
                if _is_cancel_response(user_response)
                else ConfirmationStatus.REJECTED
            )
            affected = self._clear_active(
                status=status,
                current_user_turn=current_user_turn,
                source_turn_id=source_turn_id,
                user_response=user_response,
            )
            if not affected:
                return LatchResolution(
                    kind=(
                        LatchResolutionKind.EXPIRED
                        if expired
                        else LatchResolutionKind.NO_PENDING
                    ),
                    affected_confirmation_ids=[
                        item.confirmation_id for item in expired
                    ],
                )
            return LatchResolution(
                kind=(
                    LatchResolutionKind.CANCELLED
                    if status is ConfirmationStatus.CANCELLED
                    else LatchResolutionKind.REJECTED
                ),
                affected_confirmation_ids=[item.confirmation_id for item in affected],
            )

        if response_kind != "positive":
            return LatchResolution(
                kind=(
                    LatchResolutionKind.EXPIRED
                    if expired and not self.pending
                    else LatchResolutionKind.UNRECOGNIZED
                ),
                affected_confirmation_ids=[item.confirmation_id for item in expired],
            )

        if ordered_actions is not None and not _response_matches_action_numbers(
            user_response,
            ordered_actions,
            confirmation_prompt=confirmation_prompt,
        ):
            return LatchResolution(kind=LatchResolutionKind.UNRECOGNIZED)
        if ordered_actions is not None and _response_changes_control_scope(
            user_response, ordered_actions
        ):
            return LatchResolution(kind=LatchResolutionKind.UNRECOGNIZED)

        if (
            confirmation_id is not None
            and goal_ids is not None
            and ordered_actions is not None
        ):
            invalidated = self.invalidate_if_scope_changed(
                confirmation_id,
                goal_ids=goal_ids,
                ordered_actions=ordered_actions,
                current_user_turn=current_user_turn,
                source_turn_id=source_turn_id,
            )
            if invalidated is not None:
                return invalidated

        candidates = self._positive_candidates(
            confirmation_id=confirmation_id,
            goal_ids=goal_ids,
            ordered_actions=ordered_actions,
            current_user_turn=current_user_turn,
        )
        if not candidates:
            return LatchResolution(
                kind=(
                    LatchResolutionKind.EXPIRED
                    if expired and not self.pending
                    else LatchResolutionKind.NO_PENDING
                ),
                affected_confirmation_ids=[item.confirmation_id for item in expired],
            )
        if len(candidates) > 1:
            return LatchResolution(
                kind=LatchResolutionKind.AMBIGUOUS,
                candidate_confirmation_ids=[
                    item.confirmation_id for item in candidates
                ],
            )

        updated = self._transition(
            candidates[0],
            status=ConfirmationStatus.CONFIRMED,
            current_user_turn=current_user_turn,
            source_turn_id=source_turn_id,
            user_response=user_response,
        )
        return LatchResolution(
            kind=LatchResolutionKind.CONFIRMED,
            confirmation=updated,
            affected_confirmation_ids=[updated.confirmation_id],
        )

    def invalidate_if_scope_changed(
        self,
        confirmation_id: str,
        *,
        goal_ids: list[str],
        ordered_actions: list[OfficialToolCall],
        current_user_turn: int,
        source_turn_id: str | None = None,
    ) -> LatchResolution | None:
        confirmation = self._confirmations.get(confirmation_id)
        if confirmation is None or confirmation not in self.active:
            return None
        if current_user_turn < confirmation.scope.requested_at_user_turn:
            updated = self._transition(
                confirmation,
                status=ConfirmationStatus.CANCELLED,
                current_user_turn=current_user_turn,
                source_turn_id=source_turn_id,
                user_response="invalid_turn",
            )
            return LatchResolution(
                kind=LatchResolutionKind.INVALIDATED,
                confirmation=updated,
                affected_confirmation_ids=[confirmation_id],
            )
        if current_user_turn > confirmation.scope.expires_after_user_turn:
            updated = self._transition(
                confirmation,
                status=ConfirmationStatus.EXPIRED,
                current_user_turn=current_user_turn,
                source_turn_id=source_turn_id,
            )
            return LatchResolution(
                kind=LatchResolutionKind.EXPIRED,
                confirmation=updated,
                affected_confirmation_ids=[confirmation_id],
            )
        if confirmation.scope.matches(
            goal_ids=goal_ids,
            ordered_actions=ordered_actions,
            current_user_turn=current_user_turn,
        ):
            return None
        updated = self._transition(
            confirmation,
            status=ConfirmationStatus.CANCELLED,
            current_user_turn=current_user_turn,
            source_turn_id=source_turn_id,
            user_response="scope_changed",
        )
        return LatchResolution(
            kind=LatchResolutionKind.INVALIDATED,
            confirmation=updated,
            affected_confirmation_ids=[confirmation_id],
        )

    def authorized_confirmation(
        self,
        *,
        goal_ids: list[str],
        ordered_actions: list[OfficialToolCall],
        current_user_turn: int,
    ) -> Confirmation | None:
        self.expire(current_user_turn)
        confirmed = [
            confirmation
            for confirmation in self._confirmations.values()
            if confirmation.status is ConfirmationStatus.CONFIRMED
        ]
        matches = [
            confirmation
            for confirmation in confirmed
            if current_user_turn >= confirmation.scope.requested_at_user_turn
            and confirmation.authorizes(
                goal_ids=goal_ids,
                ordered_actions=ordered_actions,
                current_user_turn=current_user_turn,
            )
        ]
        if len(matches) != 1:
            for confirmation in confirmed:
                self._transition(
                    confirmation,
                    status=ConfirmationStatus.CANCELLED,
                    current_user_turn=current_user_turn,
                    user_response="scope_changed",
                )
            return None
        return matches[0]

    def consume_authorization(
        self,
        *,
        goal_ids: list[str],
        ordered_actions: list[OfficialToolCall],
        current_user_turn: int,
    ) -> Confirmation | None:
        confirmation = self.authorized_confirmation(
            goal_ids=goal_ids,
            ordered_actions=ordered_actions,
            current_user_turn=current_user_turn,
        )
        if confirmation is None:
            return None
        del self._confirmations[confirmation.confirmation_id]
        return confirmation

    def cancel_all(
        self,
        *,
        current_user_turn: int,
        source_turn_id: str | None = None,
        user_response: str = "cancel",
    ) -> list[Confirmation]:
        return self._clear_active(
            status=ConfirmationStatus.CANCELLED,
            current_user_turn=current_user_turn,
            source_turn_id=source_turn_id,
            user_response=user_response,
        )

    def _positive_candidates(
        self,
        *,
        confirmation_id: str | None,
        goal_ids: list[str] | None,
        ordered_actions: list[OfficialToolCall] | None,
        current_user_turn: int,
    ) -> list[Confirmation]:
        pending = self.pending
        if confirmation_id is not None:
            return [
                item
                for item in pending
                if item.confirmation_id == confirmation_id
                and current_user_turn >= item.scope.requested_at_user_turn
            ]
        if goal_ids is not None and ordered_actions is not None:
            return [
                item
                for item in pending
                if current_user_turn >= item.scope.requested_at_user_turn
                and item.scope.matches(
                    goal_ids=goal_ids,
                    ordered_actions=ordered_actions,
                    current_user_turn=current_user_turn,
                )
            ]
        return [
            item
            for item in pending
            if current_user_turn >= item.scope.requested_at_user_turn
        ]

    def _clear_active(
        self,
        *,
        status: ConfirmationStatus,
        current_user_turn: int,
        source_turn_id: str | None,
        user_response: str,
    ) -> list[Confirmation]:
        changed: list[Confirmation] = []
        for confirmation in list(self.active):
            changed.append(
                self._transition(
                    confirmation,
                    status=status,
                    current_user_turn=current_user_turn,
                    source_turn_id=source_turn_id,
                    user_response=user_response,
                )
            )
        return changed

    def _transition(
        self,
        confirmation: Confirmation,
        *,
        status: ConfirmationStatus,
        current_user_turn: int,
        source_turn_id: str | None = None,
        user_response: str | None = None,
    ) -> Confirmation:
        updated = Confirmation.model_validate(
            confirmation.model_copy(
                update={
                    "status": status,
                    "source_turn_id": source_turn_id,
                    "user_response": user_response,
                    "resolved_at_user_turn": current_user_turn,
                }
            ).model_dump()
        )
        self._confirmations[confirmation.confirmation_id] = updated
        return updated

    def _insert(self, confirmation: Confirmation) -> None:
        existing = self._confirmations.get(confirmation.confirmation_id)
        if existing is not None and existing != confirmation:
            raise ValueError(
                f"conflicting confirmation ID: {confirmation.confirmation_id}"
            )
        self._confirmations[confirmation.confirmation_id] = confirmation


_NEGATIVE_PATTERN = re.compile(
    r"\b(no|cancel|stop|reject|do not|don't|never mind|nevermind)\b",
    re.IGNORECASE,
)
_CANCEL_PATTERN = re.compile(r"\b(cancel|stop|never mind|nevermind)\b", re.IGNORECASE)
_POSITIVE_RESPONSES = {
    "yes",
    "yes please",
    "yes please go ahead",
    "yes go ahead",
    "yes proceed",
    "yes please proceed",
    "yes do it",
    "yes please do it",
    "please proceed",
    "please go ahead",
    "confirm",
    "confirmed",
    "i confirm",
    "go ahead",
    "go for it",
    "do it",
    "do that",
    "let's do it",
    "lets do it",
    "sounds good",
    "that sounds good",
    "ok",
    "okay",
    "ok proceed",
    "okay proceed",
    "sure",
    "sure please",
    "sure go ahead",
    "sure please go ahead",
    "sure proceed",
    "absolutely",
    "absolutely proceed",
    "definitely",
    "yeah",
    "yep",
}
_POSITIVE_LEADERS = {
    "absolutely",
    "definitely",
    "ok",
    "okay",
    "sure",
    "yeah",
    "yep",
    "yes",
}
_POSITIVE_FILLERS = {
    "ahead",
    "do",
    "for",
    "go",
    "good",
    "i",
    "it",
    "let's",
    "lets",
    "now",
    "please",
    "proceed",
    "so",
    "sounds",
    "that",
    "this",
    "to",
    "want",
    "we",
}
_POSITIVE_SIGNAL_PATTERN = re.compile(
    r"\b(?:already\s+(?:confirmed|said\s+yes)|confirmed\s+this|go\s+ahead|"
    r"please\s+proceed|still\s+want|that['\u2019]?s\s+(?:right|what\s+i\s+want))\b",
    re.IGNORECASE,
)
_MODIFICATION_PATTERN = re.compile(
    r"\b(?:but|change|different|except|instead|only|rather\s+than)\b",
    re.IGNORECASE,
)
_DEFERRED_OR_CONDITIONAL_PATTERN = re.compile(
    r"\b(?:assuming|eventually|later|maybe|might|provided|tomorrow|unless|until|"
    r"when|whenever)\b|\b(?:as\s+long\s+as|as\s+soon\s+as|contingent\s+on|"
    r"depending\s+on|in\s+case|on\s+condition\s+that|subject\s+to)\b|\bif\b",
    re.IGNORECASE,
)
_DEFERRED_TEMPORAL_CLAUSE_PATTERN = re.compile(
    r"\b(?:after|before|once)\s+(?:(?:i|we|you|he|she|they|it)\s+\w+|"
    r"the\s+\w+\s+\w+)|\b(?:after|before)\s+\w+ing\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}
_NUMBER_WORD_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
    + r")(?:[\s-]+(?:and[\s-]+)?(?:"
    + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
    + r"))*\b",
    re.IGNORECASE,
)
_TARGET_NUMBER_UNIT_PATTERN = re.compile(
    r"\A\s*(?:%|percent(?:age)?\b|degrees?\b|celsius\b|fahrenheit\b|"
    r"levels?\b|items?\b|times?\b)",
    re.IGNORECASE,
)
_TARGET_NUMBER_PREFIX_PATTERN = re.compile(
    r"\b(?:adjust|change|close|cool|decrease|heat|increase|lower|open|raise|"
    r"set|warm)\b[^.!?]{0,32}\Z",
    re.IGNORECASE,
)
_READ_ONLY_NUMBER_UNIT_PATTERN = re.compile(
    r"\A\s*(?:hours?\b|kilometers?\b|km\b|minutes?\b|roads?\b|routes?\b)",
    re.IGNORECASE,
)
_READ_ONLY_NUMBER_PREFIX_PATTERN = re.compile(
    r"\b(?:distance|duration|route|takes?|travel\s+time|via)\b[^.!?]{0,32}\Z",
    re.IGNORECASE,
)
_ACTION_ANCHOR_STOP_WORDS = frozenset(
    {
        "add",
        "close",
        "create",
        "delete",
        "final",
        "new",
        "one",
        "open",
        "replace",
        "set",
        "turn",
    }
)
_CONTROL_TARGET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:driver\s+rear|rear\s+driver)(?:\s+(?:window|seat))?\b",
            re.IGNORECASE,
        ),
        "driver_rear",
    ),
    (
        re.compile(
            r"\b(?:passenger\s+rear|rear\s+passenger)(?:\s+(?:window|seat))?\b",
            re.IGNORECASE,
        ),
        "passenger_rear",
    ),
    (re.compile(r"\ball\s+(?:cabin\s+)?zones?\b", re.IGNORECASE), "all_zones"),
    (re.compile(r"\ball\s+windows?\b", re.IGNORECASE), "all"),
    (
        re.compile(
            r"\bdriver(?:['\u2019]?s)?\s+"
            r"(?:window|side|area|zone|temperature|seat)\b|"
            r"\b(?:adjust|change|close|heat|open|set)\s+(?:the\s+)?driver\b",
            re.IGNORECASE,
        ),
        "driver",
    ),
    (
        re.compile(
            r"\bpassenger(?:['\u2019]?s)?\s+"
            r"(?:window|side|area|zone|temperature|seat)\b|"
            r"\b(?:adjust|change|close|heat|open|set)\s+(?:the\s+)?passenger\b",
            re.IGNORECASE,
        ),
        "passenger",
    ),
)
_EXPLICIT_CONTROL_ENTITY_PATTERN = re.compile(
    r"\b(?P<entity>sunroof|sunshade|windows?)\b", re.IGNORECASE
)


def _normalized_response(response: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", response.casefold()).strip()


def _is_cancel_response(response: str) -> bool:
    return _CANCEL_PATTERN.search(response) is not None


def _classify_response(response: str) -> str:
    if _NEGATIVE_PATTERN.search(response) is not None:
        return "negative"
    if _has_deferred_or_conditional_qualification(response):
        return "other"
    normalized = _normalized_response(response)
    if normalized in _POSITIVE_RESPONSES:
        return "positive"
    words = normalized.split()
    if (
        words
        and words[0] in _POSITIVE_LEADERS
        and all(word in _POSITIVE_FILLERS for word in words[1:])
    ):
        return "positive"
    if _MODIFICATION_PATTERN.search(response) is not None:
        return "other"
    if (words and words[0] in _POSITIVE_LEADERS) or _POSITIVE_SIGNAL_PATTERN.search(
        response
    ):
        return "positive"
    return "other"


def _has_deferred_or_conditional_qualification(response: str) -> bool:
    without_unconditional_if = re.sub(
        r"\beven\s+if\b", "", response, flags=re.IGNORECASE
    )
    return bool(
        _DEFERRED_OR_CONDITIONAL_PATTERN.search(without_unconditional_if)
        or _DEFERRED_TEMPORAL_CLAUSE_PATTERN.search(without_unconditional_if)
    )


def _parse_number_words(words: list[str]) -> float | None:
    if not words or words.count("hundred") > 1:
        return None
    if "hundred" in words:
        index = words.index("hundred")
        prefix = words[:index]
        suffix = words[index + 1 :]
        if len(prefix) > 1 or (prefix and _NUMBER_WORDS[prefix[0]] > 9):
            return None
        total = (_NUMBER_WORDS[prefix[0]] if prefix else 1) * 100
        if not suffix:
            return float(total)
        remainder = _parse_number_words(suffix)
        if remainder is None or not 0 <= remainder < 100:
            return None
        return float(total) + remainder

    values = [_NUMBER_WORDS[word] for word in words]
    if len(values) == 1:
        return float(values[0])
    if len(values) == 2 and values[0] in range(20, 100, 10) and 0 < values[1] < 10:
        return float(values[0] + values[1])
    return None


def _numeric_mentions(response: str) -> list[tuple[float, int, int]]:
    mentions = [
        (float(match.group(0)), match.start(), match.end())
        for match in re.finditer(
            r"(?<![\w.])\d+(?:\.\d+)?(?!\w|\.\d)", response
        )
    ]
    mentions.extend(
        (float(match.group(1)), match.start(1), match.end(1))
        for match in re.finditer(
            r"\b[A-Za-z]{1,6}(\d+(?:\.\d+)?)\b", response
        )
    )
    for match in _NUMBER_WORD_PATTERN.finditer(response):
        words = [
            word
            for word in re.findall(r"[a-z]+", match.group(0).casefold())
            if word != "and"
        ]
        value = _parse_number_words(words)
        if value is not None:
            mentions.append((value, match.start(), match.end()))
            continue
        for word_match in re.finditer(r"[a-z]+", match.group(0).casefold()):
            word = word_match.group(0)
            if word in _NUMBER_WORDS:
                mentions.append(
                    (
                        float(_NUMBER_WORDS[word]),
                        match.start() + word_match.start(),
                        match.start() + word_match.end(),
                    )
                )
    return sorted(mentions, key=lambda item: (item[1], item[2]))


def _response_matches_control_targets(
    response: str, ordered_actions: list[OfficialToolCall]
) -> bool:
    mentioned: set[str] = set()
    claimed_spans: list[tuple[int, int]] = []
    for pattern, target in _CONTROL_TARGET_PATTERNS:
        for match in pattern.finditer(response):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in claimed_spans):
                continue
            mentioned.add(target)
            claimed_spans.append(span)
    if not mentioned:
        return True
    allowed = {
        re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
        for action in ordered_actions
        for value in action.arguments.values()
        if isinstance(value, str)
    }
    return mentioned.issubset(allowed)


def _action_number_values(ordered_actions: list[OfficialToolCall]) -> set[float]:
    values: set[float] = set()

    def collect(value: object) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            values.add(float(value))
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    for action in ordered_actions:
        collect(action.arguments)
    return values


def _contextual_number_values(prompt: str | None) -> set[float]:
    if not prompt:
        return set()
    values = {value for value, _, _ in _numeric_mentions(prompt)}
    values.update(
        float(match.group(1))
        for match in re.finditer(
            r"\b[A-Za-z]{1,6}(\d+(?:\.\d+)?)\b", prompt
        )
    )
    return values


def _mention_looks_like_action_target(response: str, start: int, end: int) -> bool:
    before = response[max(0, start - 48) : start]
    after = response[end : min(len(response), end + 32)]
    return bool(
        _TARGET_NUMBER_UNIT_PATTERN.search(after)
        or _TARGET_NUMBER_PREFIX_PATTERN.search(before)
    )


def _mention_looks_like_read_only_context(
    response: str, start: int, end: int
) -> bool:
    before = response[max(0, start - 48) : start]
    after = response[end : min(len(response), end + 32)]
    return bool(
        _READ_ONLY_NUMBER_UNIT_PATTERN.search(after)
        or _READ_ONLY_NUMBER_PREFIX_PATTERN.search(before)
    )


def _action_anchor_tokens(action: OfficialToolCall) -> set[str]:
    anchors = {
        token
        for token in re.findall(r"[a-z0-9]+", action.tool_name.casefold())
        if token not in _ACTION_ANCHOR_STOP_WORDS and not token.isdigit()
    }
    for value in action.arguments.values():
        if not isinstance(value, str) or len(value) > 40:
            continue
        if value.startswith(("loc_", "poi_", "route_", "rll_")):
            continue
        anchors.update(
            token
            for token in re.findall(r"[a-z]+", value.casefold())
            if token not in _ACTION_ANCHOR_STOP_WORDS
        )
    return anchors


def _unique_action_anchor_tokens(
    action: OfficialToolCall,
    ordered_actions: list[OfficialToolCall],
) -> set[str]:
    anchors = _action_anchor_tokens(action)
    sibling_anchors: set[str] = set()
    removed_identity = False
    for sibling in ordered_actions:
        if not removed_identity and sibling is action:
            removed_identity = True
            continue
        sibling_anchors.update(_action_anchor_tokens(sibling))
    return anchors - sibling_anchors


def _bound_action_number_values(
    response: str,
    start: int,
    end: int,
    ordered_actions: list[OfficialToolCall],
) -> set[float] | None:
    numeric_actions = [
        (action, _action_number_values([action]))
        for action in ordered_actions
        if _action_number_values([action])
    ]
    if not numeric_actions:
        return None

    center = (start + end) / 2
    separators = list(
        re.finditer(r"[;,!?]|\.(?=\s|$)|\b(?:and|then)\b", response, re.IGNORECASE)
    )
    clause_start = max(
        (match.end() for match in separators if match.end() <= start),
        default=0,
    )
    clause_end = min(
        (match.start() for match in separators if match.start() >= end),
        default=len(response),
    )

    def ranked_anchors(*, local_only: bool) -> list[tuple[float, set[float]]]:
        ranked: list[tuple[float, set[float]]] = []
        for action, values in numeric_actions:
            positions = [
                (match.start() + match.end()) / 2
                for token in _unique_action_anchor_tokens(action, ordered_actions)
                for match in re.finditer(
                    rf"\b{re.escape(token)}\b", response, re.IGNORECASE
                )
                if not local_only
                or (clause_start <= match.start() and match.end() <= clause_end)
            ]
            if positions:
                ranked.append(
                    (min(abs(position - center) for position in positions), values)
                )
        return ranked

    ranked = ranked_anchors(local_only=True) or ranked_anchors(local_only=False)

    if ranked:
        closest = min(distance for distance, _ in ranked)
        if closest <= 80:
            candidates = [values for distance, values in ranked if distance == closest]
            allowed = set(candidates[0])
            for values in candidates[1:]:
                allowed.intersection_update(values)
            return allowed
    if len(numeric_actions) == 1 and _mention_looks_like_action_target(
        response, start, end
    ):
        return numeric_actions[0][1]
    return None


def _is_safe_frozen_prerequisite_restatement(
    response: str, ordered_actions: list[OfficialToolCall]
) -> bool:
    """Accept one exact two-action prerequisite restatement, not a new condition."""

    if (
        len(ordered_actions) != 2
        or _NEGATIVE_PATTERN.search(response) is not None
        or _MODIFICATION_PATTERN.search(response) is not None
    ):
        return False
    match = re.fullmatch(
        r"\s*(?P<prefix>.+?[.!?])\s*(?:and\s+)?if\s+"
        r"(?P<condition>.+?),?\s+then\s+(?P<consequence>.+?)\s*",
        response,
        re.IGNORECASE,
    )
    if match is None:
        return False
    prefix = match.group("prefix")
    condition = match.group("condition")
    consequence = match.group("consequence")
    if (
        _classify_response(prefix) != "positive"
        or re.search(r"\bneeds?\s+to\b", condition, re.IGNORECASE) is None
        or re.search(r"\bfirst\b", condition, re.IGNORECASE) is None
        or _has_deferred_or_conditional_qualification(condition)
        or _has_deferred_or_conditional_qualification(consequence)
    ):
        return False

    prerequisite, primary = ordered_actions
    prerequisite_anchors = _unique_action_anchor_tokens(
        prerequisite, ordered_actions
    )
    primary_anchors = _unique_action_anchor_tokens(primary, ordered_actions)
    condition_words = set(re.findall(r"[a-z0-9]+", condition.casefold()))
    prefix_words = set(re.findall(r"[a-z0-9]+", prefix.casefold()))
    prerequisite_verbs = set(
        re.findall(r"[a-z0-9]+", prerequisite.tool_name.casefold())
    ) & _ACTION_ANCHOR_STOP_WORDS
    if (
        not prerequisite_anchors.intersection(condition_words)
        or not primary_anchors.intersection(prefix_words)
        or not prerequisite_verbs.intersection(condition_words)
    ):
        return False

    prerequisite_values = _action_number_values([prerequisite])
    primary_values = _action_number_values([primary])
    consequence_values = {
        value for value, _, _ in _numeric_mentions(consequence)
    }
    prefix_values = {value for value, _, _ in _numeric_mentions(prefix)}
    return bool(
        prerequisite_values
        and primary_values
        and consequence_values
        and prefix_values
        and consequence_values.issubset(prerequisite_values)
        and prefix_values.issubset(primary_values)
    )


def _response_matches_action_numbers(
    response: str,
    ordered_actions: list[OfficialToolCall],
    *,
    confirmation_prompt: str | None = None,
) -> bool:
    if not _response_matches_control_targets(response, ordered_actions):
        return False
    action_values = _action_number_values(ordered_actions)
    contextual_values = _contextual_number_values(confirmation_prompt)
    numeric_action_count = sum(
        bool(_action_number_values([action])) for action in ordered_actions
    )
    for value, start, end in _numeric_mentions(response):
        bound_values = _bound_action_number_values(
            response, start, end, ordered_actions
        )
        if bound_values is not None:
            if value not in bound_values:
                return False
            continue
        if (
            numeric_action_count > 1
            and _mention_looks_like_action_target(response, start, end)
        ):
            return False
        if value in action_values:
            continue
        if value not in contextual_values:
            return False
        if (
            _mention_looks_like_action_target(response, start, end)
            or not _mention_looks_like_read_only_context(response, start, end)
        ):
            return False
    return True


def _response_changes_control_scope(
    response: str, ordered_actions: list[OfficialToolCall]
) -> bool:
    """Reject an explicit control entity or open/close direction change."""

    mentioned_entities = {
        "window" if match.group("entity").casefold().startswith("window") else match.group("entity").casefold()
        for match in _EXPLICIT_CONTROL_ENTITY_PATTERN.finditer(response)
    }
    frozen_entities = {
        entity
        for action in ordered_actions
        for entity in ("window", "sunroof", "sunshade")
        if entity in action.tool_name.casefold()
    }
    if mentioned_entities.difference(frozen_entities):
        return True

    if len(ordered_actions) != 1:
        return False
    action = ordered_actions[0]
    percentage = action.arguments.get("percentage")
    if (
        not action.tool_name.startswith("open_close_")
        or isinstance(percentage, bool)
        or not isinstance(percentage, (int, float))
    ):
        return False
    return bool(
        (percentage != 0 and re.search(r"\bclose\b", response, re.IGNORECASE))
        or (percentage == 0 and re.search(r"\bopen\b", response, re.IGNORECASE))
    )


__all__ = [
    "ConfirmationLatch",
    "LatchResolution",
    "LatchResolutionKind",
]
