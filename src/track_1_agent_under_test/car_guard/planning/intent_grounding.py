"""Deterministic Stage-A grounding of model-extracted semantic intent.

This module deliberately accepts neither live tools nor any capability diff.  It
uses only the current raw user utterance and the invariant recipe registry to
remove goals or values that the utterance does not actually support.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Self

from pydantic import Field, model_validator

from ..domain import Goal, GoalSource, IntentFrame, IntentKind
from ..domain.types import DomainModel, NonEmptyStr
from ..recipes import OperationRecipe, OperationSpec, RecipeRegistry


class IntentGroundingResult(DomainModel):
    """Fail-closed Stage-A result with goal-bound user-provided values."""

    filtered_intent: IntentFrame
    authorized_action_goal_ids: frozenset[NonEmptyStr] = Field(
        default_factory=frozenset
    )
    desired_values_by_goal: dict[NonEmptyStr, dict[str, Any]] = Field(
        default_factory=dict
    )
    derived_values_by_goal: dict[NonEmptyStr, dict[NonEmptyStr, NonEmptyStr]] = Field(
        default_factory=dict
    )
    relative_fan_speed_deltas_by_goal: dict[NonEmptyStr, int] = Field(
        default_factory=dict
    )
    relative_climate_temperature_deltas_by_goal: dict[NonEmptyStr, int] = Field(
        default_factory=dict
    )
    relative_seat_heating_deltas_by_goal: dict[NonEmptyStr, int] = Field(
        default_factory=dict
    )
    occupied_seat_heating_goal_ids: frozenset[NonEmptyStr] = Field(
        default_factory=frozenset
    )

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        goals = {goal.goal_id: goal for goal in self.filtered_intent.goals}
        unknown_authorizations = self.authorized_action_goal_ids.difference(goals)
        if unknown_authorizations:
            raise ValueError(
                "action authorizations reference filtered goals: "
                f"{sorted(unknown_authorizations)}"
            )
        unknown_values = set(self.desired_values_by_goal).difference(goals)
        if unknown_values:
            raise ValueError(
                f"desired values reference filtered goals: {sorted(unknown_values)}"
            )
        for goal_id, values in self.desired_values_by_goal.items():
            if values != goals[goal_id].desired_outcome:
                raise ValueError(
                    "goal-bound desired values must equal the filtered goal outcome"
                )
        unknown_derivations = set(self.derived_values_by_goal).difference(goals)
        if unknown_derivations:
            raise ValueError(
                "derived values reference filtered goals: "
                f"{sorted(unknown_derivations)}"
            )
        for goal_id, derivations in self.derived_values_by_goal.items():
            unknown_parameters = set(derivations).difference(
                self.desired_values_by_goal.get(goal_id, {})
            )
            if unknown_parameters:
                raise ValueError(
                    "derived value provenance references absent parameters: "
                    f"{sorted(unknown_parameters)}"
                )
        unknown_relative_goals = set(self.relative_fan_speed_deltas_by_goal).difference(
            goals
        )
        if unknown_relative_goals:
            raise ValueError(
                "relative fan deltas reference filtered goals: "
                f"{sorted(unknown_relative_goals)}"
            )
        if any(
            delta == 0 or abs(delta) > 5
            for delta in self.relative_fan_speed_deltas_by_goal.values()
        ):
            raise ValueError("relative fan deltas must be non-zero and bounded")
        unknown_temperature_goals = set(
            self.relative_climate_temperature_deltas_by_goal
        ).difference(goals)
        if unknown_temperature_goals:
            raise ValueError(
                "relative climate temperature deltas reference filtered goals: "
                f"{sorted(unknown_temperature_goals)}"
            )
        if any(
            goals[goal_id].semantic_operation != "set_climate_temperature"
            for goal_id in self.relative_climate_temperature_deltas_by_goal
        ):
            raise ValueError(
                "relative climate temperature deltas require climate temperature goals"
            )
        if any(
            delta == 0 or abs(delta) > 10
            for delta in self.relative_climate_temperature_deltas_by_goal.values()
        ):
            raise ValueError(
                "relative climate temperature deltas must be non-zero and bounded"
            )
        relative_seat_goals = set(self.relative_seat_heating_deltas_by_goal)
        unknown_relative_seat_goals = relative_seat_goals.difference(goals)
        if unknown_relative_seat_goals:
            raise ValueError(
                "relative seat-heating deltas reference filtered goals: "
                f"{sorted(unknown_relative_seat_goals)}"
            )
        if any(
            goals[goal_id].semantic_operation != "set_seat_heating"
            for goal_id in relative_seat_goals
        ):
            raise ValueError("relative seat-heating deltas require seat-heating goals")
        if any(
            delta <= 0 or delta > 3
            for delta in self.relative_seat_heating_deltas_by_goal.values()
        ):
            raise ValueError(
                "relative seat-heating deltas must be positive and bounded"
            )
        unknown_occupied_seat_goals = set(
            self.occupied_seat_heating_goal_ids
        ).difference(goals)
        if unknown_occupied_seat_goals:
            raise ValueError(
                "occupied seat-heating scope references filtered goals: "
                f"{sorted(unknown_occupied_seat_goals)}"
            )
        if set(self.occupied_seat_heating_goal_ids) != relative_seat_goals:
            raise ValueError(
                "occupied seat-heating scope must match relative seat-heating goals"
            )
        return self


@dataclass(frozen=True, slots=True)
class _Token:
    canonical: str
    position: int
    clause: int
    sentence: int


@dataclass(frozen=True, slots=True)
class _Mention:
    start: int
    end: int
    clause: int


@dataclass(frozen=True, slots=True)
class _GoalCandidate:
    goal: Goal
    recipe: OperationRecipe
    target: OperationSpec
    is_action: bool
    anchor_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _NamedWaypointAdd:
    new_waypoint_name: str
    adjacent_waypoint_name: str
    adjacency_parameter: str
    position: int

    def desired_outcome(self, route_choice_alias: str | None) -> dict[str, str]:
        desired = {
            "new_waypoint_name": self.new_waypoint_name,
            self.adjacency_parameter: self.adjacent_waypoint_name,
        }
        if route_choice_alias is not None:
            desired["route_choice_alias"] = route_choice_alias
        return desired


@dataclass(frozen=True, slots=True)
class _NamedWaypointReplacement:
    waypoint_name_to_replace: str
    new_waypoint_name: str
    position: int

    def desired_outcome(self, route_choice_alias: str | None) -> dict[str, str]:
        desired = {
            "waypoint_name_to_replace": self.waypoint_name_to_replace,
            "new_waypoint_name": self.new_waypoint_name,
        }
        if route_choice_alias is not None:
            desired["route_choice_alias"] = route_choice_alias
        return desired


@dataclass(frozen=True, slots=True)
class DescriptiveNavigationDestinationRequest:
    """One explicit destination replacement whose city remains unresolved."""

    previous_destination_name: str | None
    route_choice_alias: str | None


@dataclass(frozen=True, slots=True)
class AdjacentNavigationDestinationReply:
    """One asserted city resolving the immediately preceding open question."""

    destination_name: str
    route_choice_alias: str | None
    route_choice_from_reply: bool


_WORD_RE = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+(?:['\u2019][^\W\d_]+)?", re.UNICODE)
_CLAUSE_PUNCTUATION_RE = re.compile(r"[.!?;,]")
_SENTENCE_PUNCTUATION_RE = re.compile(r"[.!?]")
_SENTENCE_SEGMENT_RE = re.compile(r"[^.!?]*(?:[.!?]+|$)", re.DOTALL)
_CLAUSE_CONNECTORS = frozenset({"and", "then", "also", "plus"})

_CANONICAL_WORDS = {
    "activated": "activate",
    "activating": "activate",
    "bumped": "bump",
    "bumping": "bump",
    "calls": "call",
    "called": "call",
    "calling": "call",
    "cancelled": "cancel",
    "canceled": "cancel",
    "charging": "charge",
    "charger": "charge",
    "charged": "charge",
    "charges": "charge",
    "conditioned": "conditioning",
    "deactivated": "deactivate",
    "decreased": "decrease",
    "decreasing": "decrease",
    "degrees": "degree",
    "deleted": "delete",
    "destinations": "destination",
    "directed": "direct",
    "directing": "direct",
    "disabled": "disable",
    "dropped": "drop",
    "dropping": "drop",
    "emailing": "email",
    "emails": "email",
    "enabled": "enable",
    "heating": "heat",
    "heated": "heat",
    "headed": "head",
    "heading": "head",
    "increased": "increase",
    "increasing": "increase",
    "levels": "level",
    "lights": "light",
    "lowered": "lower",
    "lowering": "lower",
    "making": "make",
    "navigating": "navigation",
    "navigate": "navigation",
    "navigated": "navigation",
    "notches": "notch",
    "raised": "raise",
    "raising": "raise",
    "reached": "reach",
    "reaches": "reach",
    "reaching": "reach",
    "recirculation": "recirculate",
    "reduced": "reduce",
    "reducing": "reduce",
    "routes": "route",
    "restaurants": "restaurant",
    "started": "start",
    "starting": "start",
    "stopped": "stop",
    "stopping": "stop",
    "stops": "stop",
    "switched": "switch",
    "turning": "turn",
    "turned": "turn",
    "traveling": "travel",
    "travelled": "travel",
    "travelling": "travel",
    "waypoints": "waypoint",
    "windows": "window",
}

_ACTION_VERBS = frozenset(
    {
        "activate",
        "add",
        "adjust",
        "bump",
        "call",
        "cancel",
        "change",
        "close",
        "cool",
        "deactivate",
        "decrease",
        "delete",
        "dial",
        "disable",
        "direct",
        "drive",
        "drop",
        "email",
        "enable",
        "go",
        "heat",
        "increase",
        "lower",
        "navigation",
        "open",
        "phone",
        "plan",
        "remove",
        "raise",
        "reduce",
        "replace",
        "route",
        "send",
        "set",
        "start",
        "stop",
        "switch",
        "turn",
    }
)
_READ_VERBS = frozenset(
    {
        "calculate",
        "check",
        "display",
        "estimate",
        "find",
        "get",
        "look",
        "present",
        "read",
        "search",
        "show",
        "tell",
    }
)
_QUESTION_STARTERS = frozenset(
    {
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "how",
        "is",
        "may",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "would",
    }
)
_POLITE_PREFIXES = (
    ("please",),
    ("kindly",),
    ("now",),
    ("please", "now"),
    ("can", "you"),
    ("can", "you", "please"),
    ("could", "you"),
    ("could", "you", "please"),
    ("would", "you"),
    ("would", "you", "please"),
    ("will", "you"),
    ("will", "you", "please"),
    ("i", "want", "to"),
    ("i", "want", "you", "to"),
    ("i", "need", "to"),
    ("i", "need", "you", "to"),
    ("i", "would", "like", "to"),
    ("i", "would", "like", "you", "to"),
    ("let", "us"),
)
_DISCOURSE_PREFIXES = frozenset({"hey", "hello", "hi", "ok", "okay", "so", "well"})
_VALUE_CONTINUATION_FILLERS = frozenset(
    {
        "a",
        "about",
        "approximately",
        "around",
        "at",
        "bit",
        "like",
        "please",
        "roughly",
        "the",
        "to",
        "way",
    }
)
_NEGATED_RE = re.compile(
    r"\b(?:do\s+not|don['\u2019]?t|never|no\s+need\s+to)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_MARKERS = frozenset(
    {"if", "unless", "suppose", "hypothetically", "maybe", "might"}
)
_REPORTED_REQUEST_RE = re.compile(
    r"\b(?:he|she|they|someone)\s+(?:said|says|told|asked)\b",
    re.IGNORECASE,
)
_NON_COMMAND_META_RE = re.compile(
    r"\b(?:example|sample|quotation|quoted|just\s+quoting)\b|"
    r"\b(?:not|isn['\u2019]?t|is\s+not)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:command|request|instruction)\b|"
    r"\b(?:do\s+not|don['\u2019]?t)\s+(?:execute|perform|follow)\b",
    re.IGNORECASE,
)
_QUOTED_SEGMENT_RE = re.compile(
    r'"[^"\r\n]*"|\u201c[^\u201d\r\n]*\u201d|\u2018[^\u2019\r\n]*\u2019|'
    r"(?<!\w)'[^'\r\n]*'(?!\w)"
)

_DISTINCTIVE_ACTION_ANCHORS = frozenset({"call", "defrost", "email", "navigation"})
_BATTERY_TRIP_RANGE_OPERATION = "assess_battery_charge_trip_range"
_BATTERY_TRIP_RANGE_MODEL_OPERATIONS = frozenset(
    {
        _BATTERY_TRIP_RANGE_OPERATION,
        "find_routes",
        "find_trip_route",
        "get_charging_specs_and_status",
        "get_distance_by_soc",
        "get_ev_range",
        "get_routes_from_start_to_destination",
        "read_charging_status",
        "read_current_navigation",
        "resolve_location",
        "resolve_trip_destination",
    }
)
_BATTERY_TRIP_DESTINATION_KEYS = frozenset(
    {
        "destination",
        "destination_id",
        "destination_name",
        "location",
        "target_location",
    }
)
_BATTERY_TRIP_EXPLICIT_DESTINATION_KEYS = frozenset(
    {"destination", "destination_name", "target_location"}
)
_BATTERY_TRIP_ENERGY_CUES = frozenset({"battery", "charge", "range"})
_BATTERY_TRIP_MOVEMENT_CUES = frozenset(
    {"drive", "get", "go", "head", "journey", "make", "reach", "travel", "trip"}
)
_BATTERY_TRIP_SELF_SUBJECTS = frozenset(
    {"battery", "car", "charge", "i", "it", "range", "vehicle", "we"}
)
_BATTERY_TRIP_MODALS = frozenset({"can", "could", "will", "would"})
_NAMED_LOCATION_POI_SEARCH_OPERATION = "search_poi_at_location"
_NAMED_LOCATION_POI_MODEL_OPERATIONS = frozenset(
    {
        "find_poi",
        "resolve_location",
        "resolve_poi_location",
        "search_poi",
        _NAMED_LOCATION_POI_SEARCH_OPERATION,
    }
)
_CURRENT_DESTINATION_POI_SEARCH_OPERATION = "search_poi_at_current_destination"
_CURRENT_DESTINATION_POI_MODEL_OPERATIONS = frozenset(
    {
        "find_poi",
        "get_current_navigation",
        "get_current_navigation_state",
        "poi_search",
        "read_current_navigation",
        "resolve_location",
        "resolve_poi_location",
        "route_presentation",
        "search_poi",
        "search_poi_at_location",
        _CURRENT_DESTINATION_POI_SEARCH_OPERATION,
    }
)
_CURRENT_DESTINATION_REFERENCE = r"(?:(?:my|the)(?:\s+current)?|current)\s+destination"
_CURRENT_DESTINATION_POI_SCOPE_RES = (
    re.compile(
        rf"\brestaurants?\b[^.!?;,]{{0,80}}\b(?:at|in|near|around)\s+"
        rf"(?P<destination>{_CURRENT_DESTINATION_REFERENCE})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:at|in|near|around)\s+"
        rf"(?P<destination>{_CURRENT_DESTINATION_REFERENCE})\b"
        rf"[^.!?;,]{{0,80}}\brestaurants?\b",
        re.IGNORECASE,
    ),
)
_CURRENT_DESTINATION_POI_QUERY_RE = re.compile(
    r"(?:^|[.!?;,]\s*)\s*"
    r"(?:(?:actually|hey|hello|hi|okay|ok|so|well)\s*,?\s*)?"
    r"(?:"
    r"i(?:['\u2019]m|\s+am)\s+looking\s+for\b"
    r"|i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to\s+"
    r"(?:find|show|search(?:\s+for)?)\b"
    r"|(?:can|could|would|will)\s+you(?:\s+please)?"
    r"(?:\s+help\s+me)?\s+(?:find|show|search(?:\s+for)?)\b"
    r"|(?:please\s+)?(?:find|show|search(?:\s+for)?)\b"
    r")",
    re.IGNORECASE,
)
_CURRENT_DESTINATION_POI_CONFLICTING_SCOPE_RE = re.compile(
    r"\b(?:my|the)?\s*(?:current|present)\s+location\b"
    r"|\b(?:near|around)\s+me\b"
    r"|\balong\s+(?:my|the)?\s*route\b",
    re.IGNORECASE,
)
_NAVIGATION_RESUME_OPERATION = "resume_previous_navigation"
_SIMPLE_NAVIGATION_CREATE_OPERATION = "create_navigation"
_SIMPLE_NAVIGATION_CREATE_MODEL_OPERATIONS = frozenset(
    {
        "create_navigation",
        "navigate_to",
        "navigation_create",
        "set_new_navigation",
        "start_navigation",
    }
)
_SIMPLE_NAVIGATION_CREATE_REQUEST_RE = re.compile(
    r"\A\s*"
    r"(?:(?:hey|hello|hi|okay|ok)\s*[!,]?\s*)?"
    r"(?:"
    r"(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"|(?:please|kindly)\s+"
    r")?"
    r"(?:navigate(?:\s+me)?|drive(?:\s+me)?|take\s+me)\s+to\s+"
    r"(?:the\s+)?"
    r"(?P<location>[^\W\d_][\w'\u2019-]*"
    r"(?:\s+[^\W\d_][\w'\u2019-]*){0,3})"
    r"(?:\s*,?\s+please)?\s*[.!?]*\s*\Z",
    re.IGNORECASE | re.UNICODE,
)
_NAVIGATION_RESUME_REQUEST_RE = re.compile(
    r"\A\s*"
    r"(?:(?:hey(?:\s+there)?|hello|hi|okay|ok|so|well)\s*[!,]?\s*)*"
    r"(?:"
    r"(?:can|could|would|will)\s+(?:you|we)\s+(?:please\s+)?"
    r"|(?:please|kindly)\s+"
    r"|i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to\s+"
    r"|how\s+about\s+"
    r")?"
    r"(?P<verb>restart(?:ing)?|resum(?:e|ing)|continu(?:e|ing)|start(?:ing)?)\s+"
    r"(?:(?:my|the|our)\s+)?"
    r"(?:(?P<scope>last|previous|prior|same|stopped|paused|earlier)\s+)?"
    r"(?P<object>navigation(?:\s+(?:route|system))?|route|trip)"
    r"(?:\s+(?:i|we)\s+(?:stopped|paused)(?:\s+earlier)?)?"
    r"(?:\s+(?:for\s+me|again|now|then|please)){0,2}"
    r"\s*[.!?]*\s*\Z",
    re.IGNORECASE | re.UNICODE,
)
_NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION = "navigation_add_one_waypoint"
_NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION = "navigation_replace_one_waypoint"
_NEXT_NAVIGATION_STOP_POI_SEARCH_OPERATION = "search_poi_at_next_navigation_stop"
_NAVIGATION_WAYPOINT_CONTEXT_MODEL_OPERATIONS = frozenset(
    {
        "add_navigation_waypoint",
        "find_poi",
        "find_routes",
        "get_current_navigation",
        "get_current_navigation_state",
        "get_routes_from_start_to_destination",
        _CURRENT_DESTINATION_POI_SEARCH_OPERATION,
        _NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION,
        _NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION,
        _NEXT_NAVIGATION_STOP_POI_SEARCH_OPERATION,
        "navigation_replace_waypoint",
        "poi_search",
        "read_current_navigation",
        "replace_navigation_waypoint",
        "resolve_location",
        "resolve_poi_location",
        "resolve_replacement_location",
        "search_poi",
        "search_poi_at_location",
        "set_new_navigation",
    }
)
_NAVIGATION_WAYPOINT_ADD_MODEL_OPERATIONS = frozenset(
    {
        "add_navigation_waypoint",
        _NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION,
        "set_new_navigation",
    }
)
_NAVIGATION_WAYPOINT_REPLACEMENT_MODEL_OPERATIONS = frozenset(
    {
        _NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION,
        "navigation_replace_waypoint",
        "replace_navigation_waypoint",
    }
)
_NEXT_NAVIGATION_STOP_POI_MODEL_OPERATIONS = frozenset(
    {
        "find_poi",
        _CURRENT_DESTINATION_POI_SEARCH_OPERATION,
        _NEXT_NAVIGATION_STOP_POI_SEARCH_OPERATION,
        "poi_search",
        "resolve_poi_location",
        "search_poi",
        "search_poi_at_location",
    }
)
_NAVIGATION_LOCATION_WORD = r"[^\W\d_]+(?:[-'\u2019][^\W\d_]+)*"
_NAMED_NAVIGATION_WAYPOINT_ADD_RES = (
    re.compile(
        rf"\badd\s+(?P<new>{_NAVIGATION_LOCATION_WORD})\s+"
        rf"(?:(?:(?:as\s+)?(?:an?\s+)?(?:(?:additional|new)\s+)?"
        rf"(?:stop|waypoint)|to\s+(?:my|the)\s+(?:current\s+)?route)\s+)?"
        rf"(?P<relation>after|before)\s+"
        rf"(?P<adjacent>{_NAVIGATION_LOCATION_WORD})\b",
        re.IGNORECASE | re.UNICODE,
    ),
    re.compile(
        rf"\b(?P<relation>after|before)\s+"
        rf"(?P<adjacent>{_NAVIGATION_LOCATION_WORD})\s*,?\s+"
        rf"(?:please\s+)?add\s+(?P<new>{_NAVIGATION_LOCATION_WORD})\s+"
        rf"(?:as\s+)?(?:an?\s+)?(?:(?:additional|new)\s+)?"
        rf"(?:stop|waypoint)\b",
        re.IGNORECASE | re.UNICODE,
    ),
)
_VAGUE_NAVIGATION_WAYPOINT_ADD_RE = re.compile(
    r"\badd\b[^.!?;]{0,64}\b(?:stop|waypoint)\b",
    re.IGNORECASE,
)
_NEXT_NAVIGATION_STOP_POI_SCOPE_RE = re.compile(
    r"\brestaurants?\b[^.!?;]{0,48}\b(?:at|in|near)\s+"
    r"(?:my|the)\s+next\s+(?:intermediate\s+)?(?:stop|waypoint)\b",
    re.IGNORECASE,
)
_NEXT_NAVIGATION_STOP_POI_QUERY_RE = re.compile(
    r"\b(?:find|show|search(?:\s+for)?)\b[^.!?;]{0,80}\brestaurants?\b",
    re.IGNORECASE,
)
_NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_RE = re.compile(
    rf"\breplace\s+"
    rf"(?:(?:the|my|current|second)\s+)?"
    rf"(?:(?:intermediate\s+)?(?:stop|waypoint)\s+)?"
    rf"(?P<old>{_NAVIGATION_LOCATION_WORD})\s+"
    rf"(?:(?:stop|waypoint)\s+)?with\s+"
    rf"(?P<new>{_NAVIGATION_LOCATION_WORD})\b",
    re.IGNORECASE | re.UNICODE,
)
_EXPLICIT_ROUTE_CHOICE_RE = re.compile(
    r"\b(?:always\s+)?(?:choose|go\s+with|pick|prefer|select|take|use)\s+"
    r"(?:the\s+)?(?:"
    r"(?:fastest|shortest)(?:\s+(?:one|option|route))?|"
    r"(?:first|second|third)\s+(?:one|option|route)"
    r")\b",
    re.IGNORECASE,
)
_NAMED_LOCATION_ALTERNATIVE_RE = re.compile(
    rf"\b(?P<left>{_NAVIGATION_LOCATION_WORD})\s*"
    rf"(?:/|\b(?:and|or|versus)\b)\s*"
    rf"(?P<right>{_NAVIGATION_LOCATION_WORD})\b",
    re.UNICODE,
)
_LOCAL_COMMAND_BOUNDARY_RE = re.compile(
    r"[.!?;]|\b(?:and|also|plus|so|then)\b", re.IGNORECASE
)
_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_OPERATION = (
    "navigation_replace_final_destination"
)
_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_MODEL_OPERATIONS = frozenset(
    {
        "create_navigation",
        "navigate_to",
        "navigation_create",
        "navigation_replace_destination",
        _NAMED_NAVIGATION_DESTINATION_REPLACEMENT_OPERATION,
        "read_current_navigation",
        "replace_navigation_destination",
        "resolve_location",
        "resolve_replacement_destination",
        "set_new_navigation",
        "start_navigation",
    }
)
_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD = r"[^\W\d_]+"
_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_VERB = r"(?:change|replace|switch)"
_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_REQUEST_LEAD = (
    r"(?:please\s+|"
    r"(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
    r"i\s+(?:need|want|would\s+like)\s+to\s+)"
)
_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_OBJECT = (
    r"(?:(?:my|the)\s+)?(?:(?:current|final|new)\s+)?"
    r"(?:navigation\s+)?destination"
)
_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_SUFFIX = (
    r"(?:\s+for\s+me|(?:\s*,\s*|\s+)please|\s+instead)?\s*[.!?]"
)
_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.UNICODE)
    for pattern in (
        (
            rf"\s*(?:"
            rf"(?:yes|yep|okay|ok),?\s+"
            rf"(?:that['\u2019]s|that\s+is)\s+the\s+current\s+"
            rf"(?:destination|one|route)|"
            rf"(?:okay|ok),?\s+i\s+understand\s+"
            rf"(?:that['\u2019]s|that\s+is)\s+the\s+current\s+"
            rf"(?:destination|one|route)"
            rf")\.\s*(?:(?:but|so),?\s*)?"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_REQUEST_LEAD}"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_VERB}\s+"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_OBJECT}\s+to\s+"
            rf"(?P<location>{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD})"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_SUFFIX}\s*"
        ),
        (
            rf"\s*(?:hey\s+there!\s*)?"
            rf"(?:i['\u2019]ve|i\s+have)\s+changed\s+my\s+mind\s+"
            rf"about\s+going\s+to\s+"
            rf"(?P<previous>{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD})"
            rf"\.\s*"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_REQUEST_LEAD}"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_VERB}\s+"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_OBJECT}\s+to\s+"
            rf"(?P<location>{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD})"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_SUFFIX}\s*"
        ),
        (
            rf"\s*(?:hey\s+there!\s*)?"
            rf"(?:i['\u2019]ve|i\s+have)\s+changed\s+my\s+mind\s+about\s+"
            rf"(?P<previous>{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD})"
            rf"\.\s*"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_REQUEST_LEAD}"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_VERB}\s+"
            rf"my\s+navigation(?:\s+destination)?\s+to\s+"
            rf"(?P<location>{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD})"
            rf"\s+instead\s*[.!?]\s*"
        ),
        (
            rf"\s*(?:hey\s+there!\s*)?"
            rf"(?:i['\u2019]ve|i\s+have)\s+changed\s+my\s+mind\s+"
            rf"about\s+going\s+to\s+"
            rf"(?P<previous>{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD})"
            rf"\.\s*"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_REQUEST_LEAD}"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_VERB}\s+"
            rf"my\s+navigation\s+to\s+"
            rf"(?P<location>{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD})"
            rf"\s+instead\s*[.!?]\s*"
        ),
        (
            rf"\s*(?:(?:okay|ok),?\s+)?"
            rf"i\s+(?:need|want|would\s+like)\s+to\s+"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_VERB}\s+"
            rf"my\s+navigation\s+destination\.\s*please\s+set\s+"
            rf"(?:it|the\s+new\s+destination)\s+to\s+"
            rf"(?P<location>{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD})"
            rf"\s*[.!?]\s*"
        ),
        (
            rf"\s*(?:(?:okay|ok),?\s+)?"
            rf"i\s+(?:need|want|would\s+like)\s+to\s+"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_VERB}\s+"
            rf"my\s+navigation\.\s*please\s+"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_VERB}\s+"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_OBJECT}\s+to\s+"
            rf"(?P<location>{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD})"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_SUFFIX}\s*"
        ),
        (
            rf"\s*(?:(?:okay|ok),?\s+)?"
            rf"i\s+(?:need|want|would\s+like)\s+to\s+"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_VERB}\s+"
            rf"my\s+navigation\.\s*my\s+new\s+destination\s+is\s+"
            rf"(?P<location>{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD})"
            rf"\.\s*(?:"
            rf"(?:can|could|would|will)\s+you\s+please\s+set\s+that\s+up\?"
            rf"|please\s+set\s+that\s+up\."
            rf")\s*"
        ),
        (
            rf"\s*(?:hey\s+there!\s*)?"
            rf"(?:i['\u2019]m|i\s+am)\s+(?:actually\s+)?heading\s+to\s+"
            rf"(?P<location>{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD})"
            rf"\s+(?:now\s+)?instead\s+of\s+"
            rf"(?P<previous>{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_WORD})"
            rf"\.\s*"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_REQUEST_LEAD}"
            rf"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_VERB}\s+"
            rf"my\s+navigation\s+destination(?:\s+for\s+me)?\s*\?\s*"
        ),
    )
)
_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_PREFIX_RE = re.compile(
    r"\b(?:change|replace|switch)\s+"
    r"(?:(?:my|the)\s+)?(?:(?:current|final|new)\s+)?"
    r"(?:navigation\s+)?destination\s+to\s+",
    re.IGNORECASE,
)
_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_LEAD_RE = re.compile(
    r"\s*(?:(?:actually|okay|ok|so)\s*,?\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
    r"i\s+(?:need|want|would\s+like)\s+to\s+|please\s+)?\s*$",
    re.IGNORECASE,
)
_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_TAIL_RE = re.compile(
    r"\s*(?P<location>[^\W\d_][\w'\u2019-]*"
    r"(?:\s+[^\W\d_][\w'\u2019-]*){0,3}?)"
    r"(?:"
    r"\s+and\s+start\s+navigation(?:\s+there)?\s*[.!?]*"
    r"|\s*[.!?]+\s*(?:can|could|would|will)\s+you\s+"
    r"navigate\s+me\s+there\s*[.!?]*"
    r"|\s+for\s+me\s*[.!?]*"
    r"|\s+instead\s*[.!?]*"
    r"|(?:\s*,\s*|\s+)please\s*[.!?]*"
    r"|\s*[.!?]*"
    r")\s*$",
    re.IGNORECASE | re.UNICODE,
)
_NAMED_NAVIGATION_DESTINATION_WITH_ROUTE_RE = re.compile(
    r"\s*(?:(?:okay|ok),?\s+)?i\s+(?:need|want|would\s+like)\s+to\s+"
    r"(?:change|replace|switch)\s+"
    r"(?:(?:my|the)\s+)?(?:(?:current|final|new)\s+)?"
    r"(?:navigation\s+)?destination\s+to\s+"
    r"(?P<location>[^.!?,;]{1,80}?)\s*[.!?]+\s*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
    r"i\s+(?:need|want|would\s+like)\s+|please\s+)?"
    r"(?:(?:do\s+that\s+and\s+)?(?:find|use|take)\s+(?:me\s+)?)?"
    r"(?:the\s+)?(?P<route>shortest|fastest)\s+route"
    r"(?:\s+to\s+(?P<repeated>[^.!?,;]{1,80}?))?"
    r"(?:\s+for\s+me)?\s*[.!?]*\s*",
    re.IGNORECASE | re.UNICODE,
)
_NAMED_NAVIGATION_DESTINATION_FROM_TO_RE = re.compile(
    r"\s*(?:(?:okay|ok),?\s*(?:let['\u2019]s\s+try\s+this\s+again\.\s*)?)?"
    r"i\s+(?:need|want|would\s+like)\s+to\s+"
    r"(?:change|replace|switch)\s+"
    r"(?:(?:my|the)\s+)?(?:(?:current|final|new)\s+)?"
    r"(?:navigation\s+)?destination\.\s*"
    r"(?:it['\u2019]s\s+currently\s+(?P<current>[^,.!?;]{1,80}),\s*"
    r"and\s+i\s+(?:need|want|would\s+like)\s+to\s+change\s+it\s+to\s+"
    r"(?P<current_location>[^.!?,;]{1,80}?)|"
    r"please\s+(?:change|replace|switch)\s+it\s+from\s+"
    r"(?P<previous>[^.!?,;]{1,80}?)\s+to\s+"
    r"(?P<location>[^.!?,;]{1,80}?))\s*[.!?]+\s*"
    r"(?:(?:and\s+then,?\s*)?i\s+(?:need|want|would\s+like)\s+|"
    r"(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:(?:find|use|take)\s+(?:me\s+)?)?(?:the\s+)?"
    r"(?P<route>shortest|fastest)\s+route"
    r"(?:\s+to\s+(?P<repeated>[^.!?,;]{1,80}?))?"
    r"(?:\s+for\s+me)?\s*[.!?]*\s*",
    re.IGNORECASE | re.UNICODE,
)
_NAMED_NAVIGATION_DESTINATION_SIMPLE_FROM_TO_RE = re.compile(
    r"\s*i\s+(?:need|want|would\s+like)\s+to\s+"
    r"(?:change|replace|switch)\s+"
    r"(?:(?:my|the)\s+)?(?:(?:current|final|new)\s+)?"
    r"(?:navigation\s+)?destination\.\s*please\s+"
    r"(?:change|replace|switch)\s+it\s+from\s+"
    r"(?P<previous>[^.!?,;]{1,80}?)\s+to\s+"
    r"(?P<location>[^.!?,;]{1,80}?)\s*[.!?]*\s*",
    re.IGNORECASE | re.UNICODE,
)
_NON_COMMAND_DESCRIPTION_WORD = (
    r"(?!(?:and|also|but|plus|please|so|then|while)\b)"
    r"[^\W\d_][\w'\u2019-]*"
)
_DESTINATION_ACTION_PHRASE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.UNICODE)
    for pattern in (
        r"\bcall\s+(?!(?:center|centers)\b)[^\W\d_][\w'\u2019-]*\b",
        (
            r"\b(?:email|message|phone|text)\s+"
            r"(?!(?:center|centers)\b)"
            r"(?:[A-Za-z][A-Za-z'\u2019.-]{0,39}\s*){1,2}$"
        ),
        (
            r"\bsend\s+(?:an?\s+)?(?:email|message|text)"
            r"(?:\s+to)?(?:\s+[A-Za-z][A-Za-z'\u2019.-]{0,39}){1,2}\s*$"
        ),
        r"\b(?:lock|unlock)\s+(?:(?:the|my)\s+)?(?:car|doors?|vehicle)\b",
        (
            r"\b(?:close|lower|raise)\s+(?:(?:the|my)\s+)?"
            r"(?:car|doors?|sunroof|trunk|windows?|volume)\s*$"
        ),
        r"\bplay\s+(?:some\s+)?music\b",
        r"\b(?:pause|resume)\s+(?:some\s+)?music\s*$",
        r"\bcheck\s+(?:(?:the|my)\s+)?(?:tire|tyre)\s+pressure\b",
        r"\bhonk\s+(?:the\s+)?horn\s*$",
        r"\b(?:cool|heat|ventilate|warm)\s+(?:the\s+)?cabin\s*$",
        (
            r"\b(?:cool|heat|ventilate|warm)\s+(?:(?:the|my)\s+)?"
            r"(?:(?:driver|passenger|front|rear)\s+)?seat\s*$"
        ),
        r"\btell\s+(?:me\s+)?(?:a\s+)?joke\s*$",
        (
            r"\bdefrost\s+(?:(?:the|my)\s+)?(?:(?:front|rear)\s+)?"
            r"(?:window|windshield)\s*$"
        ),
        (
            r"\b(?:decrease|increase)\s+(?:the\s+)?"
            r"(?:fan\s+speed|temperature|volume)\s*$"
        ),
        (
            r"\bturn\s+(?:on|off)\s+(?:the\s+)?"
            r"(?:seat\s+heating|headlights?|lights?|climate|fan|defrost)\b"
        ),
        (
            r"\bturn\s+(?:the\s+)?"
            r"(?:seat\s+heating|headlights?|lights?|climate|fan|defrost)\s+"
            r"(?:on|off)\b"
        ),
        r"\bopen\s+(?:the\s+)?(?:sunroof|windows?|doors?|trunk)\s*$",
        (
            r"\b(?:adjust|change|set)\s+(?:the\s+)?"
            r"(?:air\s+conditioning|cabin\s+temperature|climate\s+control|"
            r"fan\s+speed|seat\s+heating|temperature|volume)\s*$"
        ),
        (
            r"\b(?:activate|deactivate|disable|enable|pause|resume|start|stop)"
            r"\s+(?:the\s+)?(?:air\s+conditioning|car|climate\s+control|"
            r"cruise\s+control|engine|headlights?|lights?|music|navigation|"
            r"seat\s+heating)\s*$"
        ),
        r"\bnavigat(?:e|ion)\s+(?:to\s+)?[^,.!?;]+$",
        r"\bdrive\s+(?:home|to\s+[^,.!?;]+)\s*$",
        r"\b(?:find|show)\s+(?:me\s+)?(?:calendar|charger|charging\s+station)\s*$",
        (
            r"\b(?:change|delete|remove|replace)\s+(?:the\s+)?"
            r"(?:destination|route|stop|waypoint)\b"
        ),
    )
)
_DESTINATION_PROPER_VENUE_SUFFIXES = frozenset(
    {
        "archive",
        "center",
        "gallery",
        "hall",
        "institute",
        "lab",
        "laboratory",
        "library",
        "museum",
        "park",
        "theater",
        "theatre",
    }
)
_NON_COMMAND_LOCATION_DESCRIPTION = (
    rf"(?:a|an|some)\s+(?:{_NON_COMMAND_DESCRIPTION_WORD}\s+){{0,5}}"
    rf"(?:city|place|destination)"
    rf"(?:\s+(?:(?:known|famous|noted|recognized)\s+for|with|near|in)\s+"
    rf"{_NON_COMMAND_DESCRIPTION_WORD}"
    rf"(?:\s+{_NON_COMMAND_DESCRIPTION_WORD}){{0,3}})?"
)
_NON_COMMAND_LOCATION_DESCRIPTION_RE = re.compile(
    rf"\s*{_NON_COMMAND_LOCATION_DESCRIPTION}\s*",
    re.IGNORECASE | re.UNICODE,
)
_NON_COMMAND_VISIT_OBJECT = (
    rf"{_NON_COMMAND_DESCRIPTION_WORD}"
    rf"(?:\s+{_NON_COMMAND_DESCRIPTION_WORD}){{0,2}}"
)
_DESCRIPTIVE_NAVIGATION_DESTINATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.UNICODE)
    for pattern in (
        (
            r"\s*(?:hello,?\s+)?i\s+need\s+to\s+change\s+my\s+"
            r"final\s+destination\.\s*i['\u2019]d\s+like\s+to\s+go\s+to\s+"
            rf"(?P<description>{_NON_COMMAND_LOCATION_DESCRIPTION})"
            r"\s+instead\s+of\s+"
            r"(?P<previous>[^,.!?;]{1,80})\.\s*"
            r"(?:can|could|would|will)\s+you\s+(?:please\s+)?"
            r"(?:find|use|take)\s+(?:me\s+)?(?:the\s+)?"
            r"(?P<route>shortest|fastest)\s+route\s+there\s*[.!?]*\s*"
        ),
        (
            r"\s*(?:hello,?\s+)?(?:i\s+need\s+to\s+change\s+my\s+plans\.\s*)?"
            r"(?:can|could|would|will)\s+you\s+(?:please\s+)?"
            r"(?:change|replace|switch)\s+my\s+final\s+destination\s+from\s+"
            r"(?P<previous>[^,.!?;]{1,80}?)\s+to\s+"
            rf"(?P<description>{_NON_COMMAND_LOCATION_DESCRIPTION})"
            r"\s*[?!.]+\s*"
            r"(?:(?:and\s+)?please,?\s+)?i['\u2019]d\s+like\s+"
            r"(?:the\s+)?(?P<route>shortest|fastest)\s+route\s+there"
            r"\s*[.!?]*\s*"
        ),
    )
)
_ADJACENT_DESTINATION_REPLY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.UNICODE)
    for pattern in (
        (
            r"\s*(?:yes|so)[,!]?\s*"
            r"(?P<location>[^,.!?;]{1,80}?)\s*,?\s*please\s*[.!]?\s*"
        ),
        (
            r"\s*(?:yes|so)[,!]?\s*"
            r"(?P<location>[^,.!?;]{1,80}?)\s*,?\s*please\s*[.!]+\s*"
            r"i\s+want\s+(?:the\s+)?(?P<route>shortest|fastest)\s+route\s+to\s+"
            r"(?P<repeated>[^,.!?;]{1,80}?)\s*[.!]?\s*"
        ),
        (
            r"\s*i(?:['\u2019]m|\s+am)\s+(?:"
            r"(?:excited|curious|enthusiastic)\s+about|interested\s+in|"
            r"(?:eager|looking\s+forward)\s+to)\s+visiting\s+"
            r"(?P<location>[^,.!?;\s]{1,80}?)['\u2019]s\s+"
            rf"(?P<visit_object>{_NON_COMMAND_VISIT_OBJECT})"
            r"[.!]\s*so\s+yes[,]?\s*"
            r"(?P<repeated>[^,.!?;]{1,80}?)\s*[.!]\s*and\s+please[,]?\s*"
            r"make\s+sure\s+it(?:['\u2019]s|\s+is)\s+the\s+"
            r"(?P<route>shortest|fastest)\s+route\s*[.!]?\s*"
        ),
        (
            r"\s*(?:oh[,]?\s*)?i(?:['\u2019]m|\s+am)\s+"
            r"excited\s+to\s+visit\s+"
            r"(?P<location>[^,.!?;\s]{1,80}?)['\u2019]s\s+"
            rf"(?P<visit_object>{_NON_COMMAND_VISIT_OBJECT})"
            r"[.!]\s*yes[,]?\s*"
            r"(?P<repeated>[^,.!?;]{1,80}?)\s*,?\s*please\s*[.!]\s*"
            r"and\s+remember[,]?\s+i\s+want\s+(?:the\s+)?"
            r"(?P<route>shortest|fastest)\s+route\s*[.!?]*\s*"
        ),
        (
            r"\s*(?:oh[,]?\s*)?i\s+mean\s+"
            r"(?P<location>[^,.!?;]{1,80}?)\s*[!]\s*"
            r"i(?:['\u2019]m|\s+am)\s+(?:really\s+)?excited\s+to\s+visit\s+"
            rf"(?:the\s+)?(?P<visit_object>{_NON_COMMAND_VISIT_OBJECT})\s+there"
            r"[.!]\s*please\s+(?:find|use|take)\s+(?:me\s+)?(?:the\s+)?"
            r"(?P<route>shortest|fastest)\s+route\s+to\s+"
            r"(?P<repeated>[^,.!?;]{1,80}?)\s*[.!?]*\s*"
        ),
    )
)
_ADJACENT_DESTINATION_STANDALONE_RE = re.compile(
    r"\s*(?P<location>[^\W\d_][\w'\u2019-]*"
    r"(?:\s+[^\W\d_][\w'\u2019-]*){0,3})\s*[.!]?\s*",
    re.UNICODE,
)
_ADJACENT_DESTINATION_NON_CITY_WORDS = frozenset(
    {
        "cancel",
        "choose",
        "confirm",
        "destination",
        "fastest",
        "first",
        "go",
        "help",
        "how",
        "later",
        "navigate",
        "navigation",
        "no",
        "nope",
        "okay",
        "ok",
        "please",
        "route",
        "second",
        "select",
        "set",
        "shortest",
        "stop",
        "sure",
        "thank",
        "thanks",
        "third",
        "today",
        "tomorrow",
        "tonight",
        "use",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "yes",
        "yeah",
        "yep",
    }
)
_ADJACENT_DESTINATION_CANCELLATION_RE = re.compile(
    r"\b(?:cancel|forget\s+it|never\s+mind|stop|do\s+not\s+change|"
    r"don['\u2019]?t\s+change)\b",
    re.IGNORECASE | re.UNICODE,
)
_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_OPERATION = (
    "navigation_delete_final_destination"
)
_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_MODEL_OPERATIONS = frozenset(
    {
        "navigation_delete_one_waypoint",
        _NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_OPERATION,
        "navigation_replace_final_destination",
        "read_current_navigation",
        "resolve_location",
    }
)
_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD = r"[^\W\d_]+"
_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.UNICODE)
    for pattern in (
        (
            rf"\s*(?:hey\s+there!\s*)?"
            rf"i(?:'ve|\s+have)\s+changed\s+my\s+mind\s+about\s+going\s+to\s+"
            rf"(?P<deleted>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
            rf"after\s+"
            rf"(?P<remaining>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})"
            rf"\.\s*(?:can|could|would|will)\s+you\s+remove\s+"
            rf"(?P=deleted)\s+from\s+my\s+(?:current\s+)?route\s+"
            rf"so(?:\s+that)?\s+(?P=remaining)\s+is\s+"
            rf"(?:my|the)\s+final\s+(?:destination|stop)\?\s*"
        ),
        (
            rf"\s*i\s+want\s+to\s+remove\s+"
            rf"(?P<deleted>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
            rf"from\s+my\s+(?:current\s+)?route\.\s*"
            rf"(?:can|could|would|will)\s+you\s+make\s+"
            rf"(?P<remaining>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
            rf"the\s+final\s+destination\s+instead\?\s*"
        ),
        (
            rf"\s*(?:okay|ok),\s+so\s+i(?:'m|\s+am)\s+currently\s+"
            rf"navigating\s+to\s+"
            rf"(?P<remaining>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
            rf"with\s+a\s+stop\s+in\s+"
            rf"(?P<deleted>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})"
            rf"\.\s*i\s+want\s+to\s+remove\s+(?P=deleted)\s+from\s+"
            rf"the\s+route\s+completely\.\s*just\s+make\s+"
            rf"(?P=remaining)\s+the\s+final\s+destination\.\s*"
        ),
        (
            rf"\s*(?:(?:okay|ok),\s+let's\s+try\s+this:\s*)?"
            rf"cancel\s+"
            rf"(?P<deleted>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
            rf"as\s+a\s+destination\.\s*my\s+trip\s+should\s+end\s+in\s+"
            rf"(?P<remaining>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})"
            rf"\.\s*"
        ),
        (
            rf"\s*(?:okay|ok),\s+i\s+need\s+to\s+modify\s+my\s+current\s+"
            rf"navigation\.\s*please\s+remove\s+"
            rf"(?P<deleted>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
            rf"from\s+the\s+route\s+and\s+make\s+"
            rf"(?P<remaining>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
            rf"the\s+final\s+destination\.\s*"
        ),
        (
            rf"\s*(?:okay|ok),\s+i\s+need\s+to\s+change\s+my\s+navigation"
            rf"\.\s*my\s+current\s+route\s+has\s+"
            rf"(?P<remaining>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
            rf"and\s+then\s+"
            rf"(?P<deleted>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})"
            rf"\.\s*i\s+want\s+to\s+remove\s+(?P=deleted)\s+from\s+"
            rf"the\s+route\s+so\s+that\s+(?P=remaining)\s+is\s+my\s+"
            rf"final\s+destination\.\s*"
        ),
        (
            rf"\s*(?:(?:can|could|would|will)\s+you\s+|please\s+|"
            rf"i\s+want\s+to\s+)remove\s+"
            rf"(?P<deleted>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
            rf"from\s+(?:my\s+|the\s+)?(?:current\s+)?route\s+"
            rf"so(?:\s+that)?\s+"
            rf"(?P<remaining>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
            rf"is\s+(?:my|the)\s+final\s+(?:destination|stop)[.!?]\s*"
        ),
    )
)
_NAMED_NAVIGATION_GENERIC_DELETE_RE = re.compile(
    rf"\s*(?:(?:can|could|would|will)\s+you\s+|please\s+)?remove\s+"
    rf"(?P<deleted>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
    rf"from\s+(?:my|the)\s+(?:current\s+)?(?:trip|navigation|route)\s*[.!?]*\s*",
    re.IGNORECASE | re.UNICODE,
)
_NAMED_NAVIGATION_CONTEXTUAL_GENERIC_DELETE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.UNICODE)
    for pattern in (
        (
            rf"\s*(?:hey\s+there!\s*)?i(?:['\u2019]m|\s+am)\s+on\s+my\s+"
            rf"business\s+trip\s+route\s+from\s+"
            rf"{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD}\s+to\s+"
            rf"(?P<deleted>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+to\s+"
            rf"{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD}\s+to\s+"
            rf"{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD},\s*but\s+i\s+"
            rf"actually\s+don['\u2019]?t\s+need\s+to\s+stop\s+in\s+"
            rf"(?P=deleted)\s+anymore\.\s*(?:can|could)\s+you\s+remove\s+"
            rf"(?:(?P=deleted)|it)\s+from\s+my\s+(?:current\s+)?route\?\s*"
        ),
        (
            rf"\s*(?:hey\s+there!\s*)?i(?:['\u2019]m|\s+am)\s+currently\s+"
            rf"navigating\s+to\s+"
            rf"{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD},\s*but\s+"
            rf"i(?:['\u2019]ve|\s+have)\s+got\s+"
            rf"(?P<deleted>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
            rf"as\s+a\s+stop\s+on\s+my\s+route,\s*and\s+i\s+actually\s+"
            rf"don['\u2019]?t\s+need\s+to\s+go\s+there\s+anymore\.\s*"
            rf"(?:can|could)\s+you\s+remove\s+(?P=deleted)\s+from\s+my\s+"
            rf"(?:current\s+)?route\?\s*"
        ),
        (
            rf"\s*(?:hey\s+there!\s*)?i(?:['\u2019]m|\s+am)\s+currently\s+"
            rf"navigating\s+to\s+"
            rf"{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD},\s*but\s+i\s+"
            rf"actually\s+don['\u2019]?t\s+need\s+to\s+stop\s+in\s+"
            rf"(?P<deleted>{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_WORD})\s+"
            rf"anymore\.\s*(?:can|could)\s+you\s+remove\s+(?P=deleted)\s+"
            rf"from\s+my\s+(?:current\s+)?route\?\s*"
        ),
    )
)
_NAMED_NAVIGATION_WAYPOINT_DELETE_OPERATION = "navigation_delete_one_waypoint"
_NAMED_NAVIGATION_WAYPOINT_DELETE_MODEL_OPERATIONS = frozenset(
    {
        "delete_navigation_waypoint",
        _NAMED_NAVIGATION_WAYPOINT_DELETE_OPERATION,
        "read_current_navigation",
        "remove_navigation_waypoint",
    }
)
_NAMED_NAVIGATION_WAYPOINT_DELETE_WORD = r"[^\W\d_]+"
_NAMED_NAVIGATION_WAYPOINT_DELETE_ROUTE_RE = re.compile(
    rf"\s*(?:(?:can|could|would|will)\s+you\s+|please\s+)?remove\s+"
    rf"(?P<waypoint>{_NAMED_NAVIGATION_WAYPOINT_DELETE_WORD})\s+"
    rf"from\s+(?:my|the)\s+route"
    rf"(?:\s*[.!?]+\s*|\s+and\s+)"
    rf"(?:please\s+)?(?:go|take\s+me)\s+straight\s+to\s+"
    rf"(?P<destination>{_NAMED_NAVIGATION_WAYPOINT_DELETE_WORD})\s*[.!?]*\s*",
    re.IGNORECASE | re.UNICODE,
)
_NAMED_NAVIGATION_WAYPOINT_DELETE_STOP_RE = re.compile(
    rf"\s*(?:(?:can|could|would|will)\s+you\s+|please\s+)?remove\s+"
    rf"(?:the\s+)?(?P<waypoint>{_NAMED_NAVIGATION_WAYPOINT_DELETE_WORD})\s+"
    rf"(?:stop|waypoint)(?:\s+from\s+(?:my|the)\s+route)?\s*[.!?]*\s*",
    re.IGNORECASE | re.UNICODE,
)
_POI_CATEGORY_WORDS = {
    "airport": "airports",
    "airports": "airports",
    "bakeries": "bakery",
    "bakery": "bakery",
    "cafe": "cafe",
    "cafes": "cafe",
    "charging": "charging_stations",
    "fuel": "fuel_stations",
    "gas": "fuel_stations",
    "hotel": "hotel",
    "hotels": "hotel",
    "parking": "parking",
    "restaurant": "restaurants",
    "supermarket": "supermarkets",
    "supermarkets": "supermarkets",
    "toilet": "public_toilets",
    "toilets": "public_toilets",
}
_NAMED_LOCATION_POI_SEARCH_PREFIX_RE = re.compile(
    r"\b(?:show|search(?:\s+for)?|find)\b"
    r"[^.!?;,]{0,80}?\brestaurants\b"
    r"[^.!?;,]{0,32}?\bin\s+"
    r"(?:the\s+city\s+of\s+)?",
    re.IGNORECASE,
)
_NAMED_LOCATION_PHRASE_RE = re.compile(
    r"(?P<location>[^\W\d_][\w'\u2019-]*"
    r"(?:\s+[^\W\d_][\w'\u2019-]*){0,3})",
    re.UNICODE,
)
_NAMED_LOCATION_POI_SEARCH_TAIL_RE = re.compile(
    rf"\s*{_NAMED_LOCATION_PHRASE_RE.pattern}\s*[?.!]*\s*$",
    re.UNICODE,
)
_NAMED_LOCATION_CONTEXTUAL_REFERENCES = frozenset(
    {
        "area",
        "city",
        "here",
        "nearby",
        "there",
    }
)
_NAMED_LOCATION_LOWERCASE_PARTICLES = frozenset(
    {"am", "an", "de", "del", "den", "der", "la", "of", "the", "van", "von"}
)
_NAVIGATION_AND_DISCOURSE_STARTS = frozenset(
    {"i", "it", "please", "that", "this", "we", "you"}
)
_NAVIGATION_EDIT_OBJECTS = {
    "navigation_add_one_waypoint": "waypoint",
    "navigation_delete_one_waypoint": "waypoint",
    "navigation_delete_final_destination": "destination",
    "navigation_replace_one_waypoint": "waypoint",
    "navigation_replace_final_destination": "destination",
}
_WHOLE_NAVIGATION_DELETE = "delete_current_navigation"
_NAVIGATION_ENTITY_NAME_SELECTORS = {
    ("navigation_add_one_waypoint", "new_waypoint_id"): "new_waypoint_name",
    ("navigation_add_one_waypoint", "next_waypoint_id"): "next_waypoint_name",
    (
        "navigation_add_one_waypoint",
        "previous_waypoint_id",
    ): "previous_waypoint_name",
    (
        "navigation_delete_final_destination",
        "destination_id",
    ): "destination_name_to_delete",
    (
        "navigation_delete_one_waypoint",
        "waypoint_id",
    ): "waypoint_name_to_delete",
    (
        "navigation_replace_final_destination",
        "new_destination_id",
    ): "new_destination_name",
    (
        "navigation_replace_one_waypoint",
        "new_waypoint_id",
    ): "new_waypoint_name",
    (
        "navigation_replace_one_waypoint",
        "waypoint_id_to_replace",
    ): "waypoint_name_to_replace",
}
_NAVIGATION_DERIVED_ACTIONS = frozenset(_NAVIGATION_EDIT_OBJECTS)
_POI_ENTITY_NAME_SELECTORS = {
    (_NAMED_LOCATION_POI_SEARCH_OPERATION, "location_id"): "location_name",
}
_GENERIC_NAVIGATION_ENTITY_NAMES = frozenset(
    {
        "current",
        "current destination",
        "current stop",
        "current waypoint",
        "destination",
        "final",
        "final destination",
        "final stop",
        "intermediate",
        "intermediate stop",
        "intermediate waypoint",
        "last stop",
        "new destination",
        "new stop",
        "new waypoint",
        "stop",
        "waypoint",
    }
)
_SEMANTIC_STOPWORDS = frozenset(
    {
        "a",
        "action",
        "air",
        "all",
        "an",
        "and",
        "by",
        "control",
        "current",
        "detailed",
        "direction",
        "enable",
        "disable",
        "enabled",
        "final",
        "for",
        "from",
        "get",
        "id",
        "ids",
        "information",
        "level",
        "location",
        "minimum",
        "mode",
        "new",
        "number",
        "of",
        "one",
        "percentage",
        "position",
        "previous",
        "read",
        "set",
        "settings",
        "state",
        "status",
        "the",
        "to",
        "with",
    }
)

_SMALL_NUMBERS = {
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
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _canonical_word(word: str) -> str:
    normalized = unicodedata.normalize("NFKC", word).casefold().replace("\u2019", "'")
    if normalized.endswith("'s") and len(normalized) > 2:
        normalized = normalized[:-2]
    contractions = {
        "what's": "what",
        "who's": "who",
        "where's": "where",
        "when's": "when",
        "how's": "how",
    }
    normalized = contractions.get(normalized, normalized)
    return _CANONICAL_WORDS.get(normalized, normalized)


def _tokenize(raw_user_text: str) -> tuple[_Token, ...]:
    text = unicodedata.normalize("NFKC", raw_user_text).casefold()
    tokens: list[_Token] = []
    clause = 0
    sentence = 0
    previous_end = 0
    for match in _WORD_RE.finditer(text):
        separator = text[previous_end : match.start()]
        if _CLAUSE_PUNCTUATION_RE.search(separator):
            clause += 1
        if _SENTENCE_PUNCTUATION_RE.search(separator):
            sentence += 1
        surface = match.group(0)
        canonical = _canonical_word(surface)
        previous_end = match.end()
        if canonical in _CLAUSE_CONNECTORS:
            clause += 1
            continue
        tokens.append(
            _Token(
                canonical=canonical,
                position=len(tokens),
                clause=clause,
                sentence=sentence,
            )
        )
    return tuple(tokens)


def _without_quoted_segments(raw_user_text: str) -> str:
    """Remove quoted material so mentioned commands cannot authorize actions."""

    return _QUOTED_SEGMENT_RE.sub(" ", raw_user_text)


def _name_tokens(*names: str) -> frozenset[str]:
    values: set[str] = set()
    for name in names:
        for token in re.findall(r"[a-z0-9]+", name.casefold()):
            values.add(_canonical_word(token))
    return frozenset(values)


def _anchor_tokens(
    goal: Goal, recipe: OperationRecipe, target: OperationSpec
) -> frozenset[str]:
    del recipe
    anchors = _semantic_anchor_tokens(
        goal.semantic_operation, target.semantic_operation
    )
    if "defrost" in anchors:
        return frozenset({"defrost"})
    return anchors


def _semantic_anchor_tokens(*semantic_operations: str) -> frozenset[str]:
    tokens = _name_tokens(*semantic_operations)
    return frozenset(
        token
        for token in tokens
        if token not in _SEMANTIC_STOPWORDS
        and (token not in _ACTION_VERBS or token in _DISTINCTIVE_ACTION_ANCHORS)
    )


def _positive_identity_mentions(
    identity: frozenset[str],
    tokens: tuple[_Token, ...],
    relevant_clauses: set[int],
) -> set[str]:
    mentions: set[str] = set()
    for index, token in enumerate(tokens):
        if token.clause not in relevant_clauses or token.canonical not in identity:
            continue
        nearby = {
            item.canonical
            for item in tokens[max(0, index - 4) : index]
            if item.clause == token.clause
        }
        excluded = bool(
            nearby.intersection({"not", "without"})
            or {"rather", "than"}.issubset(nearby)
            or {"instead", "of"}.issubset(nearby)
        )
        if not excluded:
            mentions.add(token.canonical)
    return mentions


def _navigation_mutation_verb_family(
    semantic_operation: str,
) -> frozenset[str]:
    semantic = _name_tokens(semantic_operation)
    if "navigation" not in semantic:
        return frozenset()
    if "add" in semantic:
        return frozenset({"add"})
    if "replace" in semantic:
        return frozenset({"change", "replace"})
    if semantic.intersection({"cancel", "delete", "remove"}):
        return frozenset({"cancel", "delete", "remove"})
    return frozenset()


def _write_sibling_identity_is_ambiguous(
    goal: Goal,
    target: OperationSpec,
    siblings: tuple[OperationSpec, ...],
    tokens: tuple[_Token, ...],
    request_positions: tuple[int, ...],
    anchor_positions: tuple[int, ...],
    *,
    require_shared_identity: bool = False,
) -> bool:
    """Require a goal-local discriminator among sibling write operations."""

    relevant_clauses = {
        tokens[request].clause
        for request in request_positions
        for anchor in anchor_positions
        if tokens[request].clause == tokens[anchor].clause
        and abs(request - anchor) <= 6
    }
    identity_stopwords = _SEMANTIC_STOPWORDS | {
        "set",
        "get",
        "read",
        "enable",
        "disable",
    }
    target_identity = _name_tokens(target.semantic_operation).difference(
        identity_stopwords
    )
    for sibling in siblings:
        if sibling.semantic_operation == target.semantic_operation:
            continue
        target_verbs = _navigation_mutation_verb_family(target.semantic_operation)
        sibling_verbs = _navigation_mutation_verb_family(sibling.semantic_operation)
        if target_verbs and sibling_verbs and target_verbs != sibling_verbs:
            requested_verbs = {
                tokens[position].canonical for position in request_positions
            }
            target_requested = bool(requested_verbs.intersection(target_verbs))
            sibling_requested = bool(requested_verbs.intersection(sibling_verbs))
            if sibling_requested and not target_requested:
                return True
            if target_requested and not sibling_requested:
                continue
        sibling_identity = _name_tokens(sibling.semantic_operation).difference(
            identity_stopwords
        )
        target_only = target_identity.difference(sibling_identity)
        sibling_only = sibling_identity.difference(target_identity)
        if require_shared_identity and not target_identity.intersection(
            sibling_identity
        ):
            continue
        if not target_only or not sibling_only:
            continue
        target_mentions = _positive_identity_mentions(
            frozenset(target_only), tokens, relevant_clauses
        )
        sibling_mentions = _positive_identity_mentions(
            frozenset(sibling_only), tokens, relevant_clauses
        )
        stop_identities = set(
            _navigation_stop_identities(
                tokens,
                request_positions,
                relevant_clauses=relevant_clauses,
            ).values()
        )
        target_mentions.update(target_only.intersection(stop_identities))
        sibling_mentions.update(sibling_only.intersection(stop_identities))
        if sibling_mentions and not target_mentions:
            return True
        if target_mentions or sibling_mentions:
            continue
        target_parameters = set(target.parameter_mapping).difference(
            sibling.parameter_mapping
        )
        desired_parameters = {
            _SEMANTIC_PARAMETER_ALIASES.get(
                (target.semantic_operation, parameter), parameter
            )
            for parameter in goal.desired_outcome
        }
        if target_parameters.intersection(desired_parameters):
            continue
        return True
    return False


def _navigation_stop_identities(
    tokens: tuple[_Token, ...],
    request_positions: tuple[int, ...],
    *,
    relevant_clauses: set[int] | None = None,
) -> dict[int, str]:
    """Classify object-position ``stop`` mentions without treating stop as a verb."""

    identities: dict[int, str] = {}
    for token in tokens:
        if token.canonical != "stop" or (
            relevant_clauses is not None and token.clause not in relevant_clauses
        ):
            continue
        preceding_requests = [
            position
            for position in request_positions
            if tokens[position].clause == token.clause
            and 0 < token.position - position <= 6
        ]
        if not preceding_requests:
            continue
        nearby = {
            item.canonical
            for item in tokens[max(0, token.position - 4) : token.position]
            if item.clause == token.clause
        }
        excluded = bool(
            nearby.intersection({"not", "without"})
            or {"rather", "than"}.issubset(nearby)
            or {"instead", "of"}.issubset(nearby)
        )
        if excluded:
            continue
        identities[token.position] = (
            "destination" if nearby.intersection({"final", "last"}) else "waypoint"
        )
    return identities


def _navigation_stop_anchor_positions(
    goal: Goal,
    target: OperationSpec,
    tokens: tuple[_Token, ...],
) -> dict[int, str]:
    compatible = _compatible_action_verbs(goal, target)
    requests = tuple(
        token.position
        for token in tokens
        if token.canonical in compatible and _has_request_prefix(tokens, token.position)
    )
    return _navigation_stop_identities(tokens, requests)


def _navigation_family_identity_matches(
    target: OperationSpec,
    tokens: tuple[_Token, ...],
    request_positions: tuple[int, ...],
    anchor_positions: tuple[int, ...],
) -> bool:
    """Bind waypoint/destination identity to the navigation action clause."""

    is_whole_delete = target.semantic_operation == _WHOLE_NAVIGATION_DELETE
    expected = _NAVIGATION_EDIT_OBJECTS.get(target.semantic_operation)
    if expected is None and not is_whole_delete:
        return True
    relevant_clauses = {
        tokens[request].clause
        for request in request_positions
        for anchor in anchor_positions
        if tokens[request].clause == tokens[anchor].clause
        and abs(request - anchor) <= 6
    }
    mentioned = _positive_identity_mentions(
        frozenset({"waypoint", "destination"}), tokens, relevant_clauses
    )
    mentioned.update(
        _navigation_stop_identities(
            tokens,
            request_positions,
            relevant_clauses=relevant_clauses,
        ).values()
    )
    if is_whole_delete:
        return not mentioned
    return mentioned == {expected}


def _roof_relation_direction_matches(
    target: OperationSpec, tokens: tuple[_Token, ...]
) -> bool:
    expected = {
        "set_sunroof_position": "sunroof",
        "set_sunshade_position": "sunshade",
    }.get(target.semantic_operation)
    if expected is None:
        return True
    relation_words = {
        "align",
        "aligned",
        "aligning",
        "match",
        "matched",
        "matching",
        "same",
        "sync",
        "synchronize",
        "synchronized",
        "synchronizing",
    }
    for clause in {token.clause for token in tokens}:
        clause_tokens = tuple(token for token in tokens if token.clause == clause)
        sunroofs = [
            token.position for token in clause_tokens if token.canonical == "sunroof"
        ]
        sunshades = [
            token.position for token in clause_tokens if token.canonical == "sunshade"
        ]
        relations = [
            token.position
            for token in clause_tokens
            if token.canonical in relation_words
        ]
        if not sunroofs or not sunshades or not relations:
            continue
        directions = {
            "sunshade"
            if sunshade < relation < sunroof
            else "sunroof"
            if sunroof < relation < sunshade
            else "ambiguous"
            for sunroof in sunroofs
            for sunshade in sunshades
            for relation in relations
        }
        return directions == {expected}
    return True


def _poi_category_identities(tokens: tuple[_Token, ...]) -> frozenset[str]:
    identities = {
        _POI_CATEGORY_WORDS[token.canonical]
        for token in tokens
        if token.canonical in _POI_CATEGORY_WORDS
    }
    if any(
        first.canonical == "fast" and second.canonical == "food"
        for first, second in zip(tokens, tokens[1:])
    ):
        identities.add("fast_food")
    return frozenset(identities)


def _looks_like_named_location_literal(value: str) -> bool:
    normalized = " ".join(unicodedata.normalize("NFKC", value).strip().split())
    match = _NAMED_LOCATION_PHRASE_RE.fullmatch(normalized)
    if match is None:
        return False
    words = tuple(re.findall(r"[^\W\d_][\w'\u2019-]*", normalized, re.UNICODE))
    canonical_words = tuple(_canonical_word(word) for word in words)
    contextual = set(canonical_words).intersection(
        _NAMED_LOCATION_CONTEXTUAL_REFERENCES
    )
    if contextual and not (
        contextual == {"city"}
        and len(canonical_words) >= 2
        and canonical_words[-1] == "city"
    ):
        return False
    if len(words) == 1:
        return words[0][0].isupper()
    if not words[0][0].isupper() or not words[-1][0].isupper():
        return False
    return all(
        word[0].isupper() or canonical in _NAMED_LOCATION_LOWERCASE_PARTICLES
        for word, canonical in zip(words[1:-1], canonical_words[1:-1])
    )


def _named_location_poi_goal_is_bound(goal: Goal, tokens: tuple[_Token, ...]) -> bool:
    """Bind one restaurant search to one literal city name in its read clause."""

    if any(
        token.canonical in _HYPOTHETICAL_MARKERS | {"never", "not"} for token in tokens
    ):
        return False
    categories = {
        str(value).strip().casefold().replace(" ", "_")
        for key, value in goal.desired_outcome.items()
        if key.casefold() in {"category", "category_poi"}
        and isinstance(value, str)
        and value.strip()
    }
    categories = {
        "restaurants" if category == "restaurant" else category
        for category in categories
    }
    if categories != {"restaurants"} or _poi_category_identities(tokens) != frozenset(
        {"restaurants"}
    ):
        return False

    locations: dict[str, str] = {}
    for key, value in goal.desired_outcome.items():
        if (
            key.casefold() not in {"location", "location_id", "location_name"}
            or not isinstance(value, str)
            or not value.strip()
            or value.casefold().startswith(("loc_", "poi_"))
            or not _looks_like_named_location_literal(value)
            or not _phrase_mentions(value, tokens)
        ):
            continue
        normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
        locations.setdefault(normalized, value.strip())
    if len(locations) != 1:
        return False

    location = next(iter(locations.values()))
    read_positions = {
        token.position
        for token in tokens
        if token.canonical in {"find", "search", "show"}
        and _has_request_prefix(tokens, token.position)
    }
    restaurant_positions = {
        token.position for token in tokens if token.canonical == "restaurant"
    }
    for mention in _phrase_mentions(location, tokens):
        if any(
            tokens[read].clause == mention.clause
            and tokens[restaurant].clause == mention.clause
            and read < restaurant
            and any(
                token.canonical == "in"
                and restaurant < token.position < mention.start
                and tuple(
                    item.canonical
                    for item in tokens
                    if item.clause == mention.clause
                    and token.position < item.position < mention.start
                )
                in {(), ("the", "city", "of")}
                for token in tokens
            )
            and not any(
                token.clause == mention.clause and token.position > mention.end
                for token in tokens
            )
            for read in read_positions
            for restaurant in restaurant_positions
        ):
            return True
    return False


def _anchor_positions(
    goal: Goal,
    recipe: OperationRecipe,
    target: OperationSpec,
    tokens: tuple[_Token, ...],
) -> tuple[int, ...]:
    anchors = _anchor_tokens(goal, recipe, target)
    positions = {token.position for token in tokens if token.canonical in anchors}
    if not positions:
        anchors = frozenset(
            token
            for token in _name_tokens(recipe.id)
            if token not in _SEMANTIC_STOPWORDS
            and (token not in _ACTION_VERBS or token in _DISTINCTIVE_ACTION_ANCHORS)
        )
        positions = {token.position for token in tokens if token.canonical in anchors}
    if "conditioning" in anchors:
        positions.update(token.position for token in tokens if token.canonical == "ac")
    if target.semantic_operation == "set_window_defrost":
        defrost_tokens = [token for token in tokens if token.canonical == "defrost"]
        positions.update(
            token.position
            for token in tokens
            if token.canonical in {"window", "windshield"}
            and any(
                token.clause == defrost.clause
                and abs(token.position - defrost.position) <= 4
                for defrost in defrost_tokens
            )
        )
    expected_navigation_object = _NAVIGATION_EDIT_OBJECTS.get(target.semantic_operation)
    if expected_navigation_object is not None:
        positions.update(
            token.position
            for token in tokens
            if token.canonical == expected_navigation_object
        )
        positions.update(
            position
            for position, identity in _navigation_stop_anchor_positions(
                goal, target, tokens
            ).items()
            if identity == expected_navigation_object
        )
    if target.semantic_operation == "set_fan_airflow_direction":
        positions.update(token.position for token in tokens if token.canonical == "air")
    if target.semantic_operation == "get_entries_from_calendar":
        positions.update(
            token.position
            for token in tokens
            if token.canonical
            in {
                "appointment",
                "appointments",
                "calendar",
                "event",
                "events",
                "meeting",
                "meetings",
                "schedule",
            }
        )
    if target.semantic_operation == _BATTERY_TRIP_RANGE_OPERATION:
        destination = goal.desired_outcome.get("destination_name")
        if (
            isinstance(destination, str)
            and _battery_trip_destination_is_movement_bound(destination, tokens)
            and _battery_trip_has_feasibility_read(destination, tokens)
        ):
            read_clauses = _read_request_clauses(tokens)
            positions.update(
                token.position
                for token in tokens
                if token.clause in read_clauses
                and token.canonical in _BATTERY_TRIP_MOVEMENT_CUES
            )
    if (
        target.semantic_operation == _NAMED_LOCATION_POI_SEARCH_OPERATION
        and _named_location_poi_goal_is_bound(goal, tokens)
    ):
        positions.update(
            token.position
            for token in tokens
            if token.canonical in {"find", "search", "show"}
            or token.canonical == "restaurant"
        )
    return tuple(sorted(positions))


def _clause_tokens(tokens: tuple[_Token, ...], clause: int) -> tuple[_Token, ...]:
    return tuple(token for token in tokens if token.clause == clause)


def _has_request_prefix(tokens: tuple[_Token, ...], position: int) -> bool:
    token = tokens[position]
    clause = _clause_tokens(tokens, token.clause)
    local_index = next(
        index for index, item in enumerate(clause) if item.position == position
    )
    if local_index == 0:
        return True
    prefix = tuple(item.canonical for item in clause[:local_index])
    while prefix and prefix[0] in _DISCOURSE_PREFIXES:
        prefix = prefix[1:]
    return prefix in _POLITE_PREFIXES


def _polite_relative_temperature_maybe_positions(
    tokens: tuple[_Token, ...],
) -> frozenset[int]:
    """Recognize only a modal request followed by ``and maybe lower ... by N``."""

    safe: set[int] = set()
    directions = {"decrease", "drop", "lower", "reduce"}
    for maybe in (token for token in tokens if token.canonical == "maybe"):
        sentence = tuple(token for token in tokens if token.sentence == maybe.sentence)
        canonical = tuple(token.canonical for token in sentence)
        local_index = next(
            index
            for index, token in enumerate(sentence)
            if token.position == maybe.position
        )
        prefix = canonical[:local_index]
        suffix = canonical[local_index + 1 :]
        prior_clauses = {
            token.clause for token in sentence if token.clause < maybe.clause
        }
        previous_clause = (
            tuple(
                token.canonical
                for token in sentence
                if token.clause == max(prior_clauses)
            )
            if prior_clauses
            else ()
        )
        if (
            len(prefix) < 3
            or len(suffix) < 4
            or suffix[0] not in directions
            or previous_clause[:2] not in {("can", "you"), ("could", "you")}
            or not any(
                token.canonical in _ACTION_VERBS
                and token.clause == max(prior_clauses)
                and _has_request_prefix(tokens, token.position)
                for token in sentence
            )
            or "temperature" not in suffix[1:5]
            or "by" not in suffix[1:8]
            or "degree" not in suffix
            or any(
                token in _HYPOTHETICAL_MARKERS.difference({"maybe"})
                for token in canonical
            )
        ):
            continue
        safe.add(sentence[local_index + 1].position)
    return frozenset(safe)


def _explicit_request_positions(
    raw_user_text: str,
    tokens: tuple[_Token, ...],
    verbs: frozenset[str],
) -> tuple[int, ...]:
    if _NEGATED_RE.search(raw_user_text):
        return ()
    if _REPORTED_REQUEST_RE.search(raw_user_text):
        return ()
    if _NON_COMMAND_META_RE.search(raw_user_text):
        return ()
    safe_maybe_positions = _polite_relative_temperature_maybe_positions(tokens)
    safe_maybe_sentences = {
        tokens[position].sentence for position in safe_maybe_positions
    }
    hypothetical_sentences = {
        token.sentence for token in tokens if token.canonical in _HYPOTHETICAL_MARKERS
    }.difference(safe_maybe_sentences)
    return tuple(
        token.position
        for token in tokens
        if token.sentence not in hypothetical_sentences
        and token.canonical in verbs
        and (
            _has_request_prefix(tokens, token.position)
            or token.position in safe_maybe_positions
        )
    )


def _read_request_clauses(tokens: tuple[_Token, ...]) -> frozenset[int]:
    clauses: set[int] = set()
    for clause in sorted({token.clause for token in tokens}):
        items = _clause_tokens(tokens, clause)
        if not items:
            continue
        if items[0].canonical in _QUESTION_STARTERS:
            clauses.add(clause)
            continue
        if any(
            item.canonical in _READ_VERBS and _has_request_prefix(tokens, item.position)
            for item in items
        ):
            clauses.add(clause)
    return frozenset(clauses)


def _compatible_action_verbs(goal: Goal, target: OperationSpec) -> frozenset[str]:
    semantic = _name_tokens(goal.semantic_operation, target.semantic_operation)
    allowed: set[str] = set()
    navigation_edit_verbs = {"add", "cancel", "delete", "remove", "replace"}
    is_navigation_mutation = bool(
        "navigation" in semantic and semantic.intersection(navigation_edit_verbs)
    )

    if semantic.intersection({"set", "control"}):
        allowed.update({"adjust", "change", "set", "turn"})
    if "direction" in semantic:
        allowed.add("direct")
    if "enable" in semantic:
        allowed.update({"activate", "enable", "set", "start", "switch", "turn"})
    if "disable" in semantic:
        allowed.update({"deactivate", "disable", "set", "stop", "switch", "turn"})
    if "open" in semantic:
        allowed.update({"open", "set"})
    if "close" in semantic:
        allowed.update({"close", "set"})
    if semantic.intersection({"navigation", "create"}) and not is_navigation_mutation:
        allowed.update({"drive", "go", "navigation", "route", "set", "start"})
    if "call" in semantic or "phone" in semantic:
        allowed.update({"call", "dial", "phone"})
    if "email" in semantic or "send" in semantic:
        allowed.update({"email", "send"})
    if "add" in semantic:
        allowed.add("add")
    if "replace" in semantic:
        allowed.update({"change", "replace"})
    if semantic.intersection({"delete", "cancel", "remove"}):
        allowed.update({"cancel", "delete", "remove", "stop"})
    if semantic.intersection({"heat", "defrost"}):
        allowed.update({"defrost", "heat", "set", "switch", "turn"})
    if "seat" in semantic and "heat" in semantic:
        allowed.update({"decrease", "increase", "lower", "raise", "reduce"})
    if semantic.intersection({"climate", "temperature"}):
        allowed.update(
            {
                "cool",
                "decrease",
                "drop",
                "heat",
                "increase",
                "lower",
                "raise",
                "reduce",
            }
        )
    if {"fan", "speed"}.issubset(semantic):
        allowed.update({"bump", "decrease", "increase", "lower", "raise", "reduce"})
    if semantic.intersection({"window", "sunroof", "sunshade", "trunk"}):
        allowed.update({"close", "open"})
    if "plan" in semantic or "charge" in semantic:
        allowed.update({"add", "plan"})

    if not is_navigation_mutation and any(
        isinstance(value, bool) for value in goal.desired_outcome.values()
    ):
        allowed.update(
            {
                "activate",
                "deactivate",
                "disable",
                "enable",
                "start",
                "stop",
                "switch",
                "turn",
            }
        )
    return frozenset(allowed)


def _positions_are_close(
    left: tuple[int, ...], right: tuple[int, ...], tokens: tuple[_Token, ...]
) -> bool:
    return any(
        tokens[first].clause == tokens[second].clause and abs(first - second) <= 6
        for first in left
        for second in right
    )


def _candidate_options(
    goal: Goal,
    registry: RecipeRegistry,
    tokens: tuple[_Token, ...],
    action_request_positions: tuple[int, ...],
    read_request_clauses: frozenset[int],
) -> tuple[_GoalCandidate, ...]:
    options: list[_GoalCandidate] = []
    for recipe in registry:
        if not recipe.matches_semantic_operation(goal.semantic_operation):
            continue
        target = recipe.target_operation(goal.semantic_operation)
        if target is None:
            continue
        anchors = _anchor_positions(goal, recipe, target, tokens)
        if not anchors:
            continue
        write_names = {
            operation.semantic_operation for operation in recipe.write_operations
        }
        is_action = target.semantic_operation in write_names
        if is_action:
            compatible = _compatible_action_verbs(goal, target)
            requests = tuple(
                position
                for position in action_request_positions
                if tokens[position].canonical in compatible
            )
            if not requests or not _positions_are_close(requests, anchors, tokens):
                continue
            if not _navigation_family_identity_matches(
                target, tokens, requests, anchors
            ):
                continue
            if not _roof_relation_direction_matches(target, tokens):
                continue
            if _write_sibling_identity_is_ambiguous(
                goal,
                target,
                tuple(recipe.write_operations),
                tokens,
                requests,
                anchors,
            ):
                continue
            cross_recipe_writes = tuple(
                operation
                for candidate_recipe in registry
                if candidate_recipe.id != recipe.id
                for operation in candidate_recipe.write_operations
            )
            if _write_sibling_identity_is_ambiguous(
                goal,
                target,
                cross_recipe_writes,
                tokens,
                requests,
                anchors,
                require_shared_identity=True,
            ):
                continue
        else:
            requests = tuple(
                token.position
                for token in tokens
                if token.clause in read_request_clauses
                and token.canonical in _READ_VERBS
            )
            if not any(
                tokens[position].clause in read_request_clauses for position in anchors
            ):
                continue
        options.append(
            _GoalCandidate(
                goal=goal,
                recipe=recipe,
                target=target,
                is_action=is_action,
                anchor_positions=anchors,
            )
        )
    return tuple(options)


def _parse_number_words(words: tuple[str, ...]) -> Decimal | None:
    if not words:
        return None
    if len(words) == 1:
        try:
            return Decimal(words[0])
        except InvalidOperation:
            pass

    total = 0
    current = 0
    consumed = False
    for word in words:
        if word in _SMALL_NUMBERS:
            current += _SMALL_NUMBERS[word]
            consumed = True
        elif word in _TENS:
            current += _TENS[word]
            consumed = True
        elif word == "hundred" and current:
            current *= 100
            consumed = True
        else:
            return None
    return Decimal(total + current) if consumed else None


def _numeric_mentions(
    value: int | float, tokens: tuple[_Token, ...]
) -> tuple[_Mention, ...]:
    expected = Decimal(str(value))
    mentions: set[_Mention] = set()
    for start in range(len(tokens)):
        for length in range(1, 5):
            end = start + length
            if end > len(tokens):
                break
            span = tokens[start:end]
            if len({token.clause for token in span}) != 1:
                break
            parsed = _parse_number_words(tuple(token.canonical for token in span))
            if parsed == expected:
                mentions.add(_Mention(start=start, end=end - 1, clause=span[0].clause))
    return tuple(sorted(mentions, key=lambda item: (item.start, item.end)))


def _percentage_phrase_mentions(
    value: int | float, tokens: tuple[_Token, ...]
) -> tuple[_Mention, ...]:
    expected = Decimal(str(value))
    phrases: dict[Decimal, tuple[tuple[str, ...], ...]] = {
        Decimal("0"): (("closed",),),
        Decimal("25"): (("quarter",), ("quarterway",), ("quarter", "way")),
        Decimal("50"): (("half",), ("halfway",), ("half", "way")),
        Decimal("75"): (("three", "quarters"),),
        Decimal("100"): (
            ("all", "the", "way"),
            ("completely",),
            ("fully",),
        ),
    }
    wanted = phrases.get(expected, ())
    mentions: list[_Mention] = []
    for phrase in wanted:
        for start in range(0, len(tokens) - len(phrase) + 1):
            span = tokens[start : start + len(phrase)]
            if len({token.clause for token in span}) != 1:
                continue
            if tuple(token.canonical for token in span) == phrase:
                mentions.append(
                    _Mention(start, start + len(phrase) - 1, span[0].clause)
                )
    return tuple(mentions)


def _phrase_mentions(value: str, tokens: tuple[_Token, ...]) -> tuple[_Mention, ...]:
    wanted = tuple(
        _canonical_word(token)
        for token in re.findall(
            r"[a-z0-9]+", unicodedata.normalize("NFKC", value).casefold()
        )
    )
    if not wanted:
        return ()
    mentions: list[_Mention] = []
    for start in range(0, len(tokens) - len(wanted) + 1):
        span = tokens[start : start + len(wanted)]
        if len({token.clause for token in span}) != 1:
            continue
        if tuple(token.canonical for token in span) == wanted:
            mentions.append(_Mention(start, start + len(wanted) - 1, span[0].clause))
    return tuple(mentions)


def _boolean_polarity_mentions(
    value: bool, tokens: tuple[_Token, ...]
) -> tuple[_Mention, ...]:
    positive = {"activate", "enable", "on", "start"}
    negative = {"deactivate", "disable", "off", "stop"}
    mentions: list[_Mention] = []
    for index, token in enumerate(tokens):
        if token.canonical not in positive | negative:
            continue
        clause_prefix = tuple(
            item for item in tokens[:index] if item.clause == token.clause
        )
        nearby = clause_prefix[-3:]
        negators = [item for item in nearby if item.canonical == "not"]
        negators.extend(item for item in clause_prefix if item.canonical == "without")
        polarity = token.canonical in positive
        if negators:
            polarity = not polarity
        if polarity == value:
            mentions.append(
                _Mention(
                    negators[0].position if negators else token.position,
                    token.position,
                    token.clause,
                )
            )
    return tuple(mentions)


def _boolean_mentions(value: bool, tokens: tuple[_Token, ...]) -> tuple[_Mention, ...]:
    wanted = _boolean_polarity_mentions(value, tokens)
    opposite_clauses = {
        mention.clause for mention in _boolean_polarity_mentions(not value, tokens)
    }
    return tuple(
        mention for mention in wanted if mention.clause not in opposite_clauses
    )


def _scalar_mentions(
    key: str,
    value: Any,
    tokens: tuple[_Token, ...],
) -> tuple[_Mention, ...]:
    if isinstance(value, bool):
        return _boolean_mentions(value, tokens)
    if isinstance(value, (int, float)):
        mentions = list(_numeric_mentions(value, tokens))
        if key.casefold() == "percentage":
            mentions.extend(_percentage_phrase_mentions(value, tokens))
            if value == 0:
                mentions.extend(
                    _Mention(token.position, token.position, token.clause)
                    for token in tokens
                    if token.canonical == "close"
                )
        return tuple(mentions)
    if isinstance(value, str) and value.strip():
        return _phrase_mentions(value, tokens)
    return ()


def _mention_distance(mention: _Mention, positions: tuple[int, ...]) -> int | None:
    if not positions:
        return None
    return min(
        min(abs(mention.start - position), abs(mention.end - position))
        for position in positions
    )


def _mention_is_bound(
    mention: _Mention,
    candidate: _GoalCandidate,
    candidates: dict[str, _GoalCandidate],
    tokens: tuple[_Token, ...],
) -> bool:
    own_positions = tuple(
        position
        for position in candidate.anchor_positions
        if tokens[position].clause == mention.clause
    )
    own_distance = _mention_distance(mention, own_positions)
    if own_distance is None:
        if len(candidates) != 1 or not candidate.anchor_positions:
            return False
        nearest = min(
            candidate.anchor_positions,
            key=lambda position: min(
                abs(mention.start - position), abs(mention.end - position)
            ),
        )
        own_distance = min(abs(mention.start - nearest), abs(mention.end - nearest))
        if own_distance > 8 or abs(tokens[nearest].clause - mention.clause) > 2:
            return False
        lower = min(nearest, mention.start)
        upper = max(nearest, mention.end)
        if any(
            token.canonical not in _VALUE_CONTINUATION_FILLERS
            for token in tokens[lower + 1 : upper]
        ):
            return False
    if own_distance > 8:
        return False
    for goal_id, other in candidates.items():
        if goal_id == candidate.goal.goal_id:
            continue
        other_positions = tuple(
            position
            for position in other.anchor_positions
            if tokens[position].clause == mention.clause
        )
        other_distance = _mention_distance(mention, other_positions)
        if other_distance is not None and other_distance <= own_distance:
            return False
    return True


def _boolean_polarity_is_bound(
    value: bool,
    candidate: _GoalCandidate,
    candidates: dict[str, _GoalCandidate],
    tokens: tuple[_Token, ...],
) -> bool:
    for mention in _boolean_polarity_mentions(value, tokens):
        if _mention_is_bound(mention, candidate, candidates, tokens):
            return True
        if len(candidates) != 1 or not candidate.anchor_positions:
            continue
        if any(
            tokens[position].sentence == tokens[mention.start].sentence
            and min(abs(mention.start - position), abs(mention.end - position)) <= 8
            for position in candidate.anchor_positions
        ):
            return True
    return False


def _value_is_grounded(
    key: str,
    value: Any,
    candidate: _GoalCandidate,
    candidates: dict[str, _GoalCandidate],
    tokens: tuple[_Token, ...],
) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(
            _value_is_grounded(nested_key, nested_value, candidate, candidates, tokens)
            for nested_key, nested_value in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value) and all(
            _value_is_grounded(key, item, candidate, candidates, tokens)
            for item in value
        )
    if isinstance(value, bool):
        return _boolean_polarity_is_bound(value, candidate, candidates, tokens)
    return any(
        _mention_is_bound(mention, candidate, candidates, tokens)
        for mention in _scalar_mentions(key, value, tokens)
    )


def _all_seat_zones_are_bound(
    candidate: _GoalCandidate,
    candidates: dict[str, _GoalCandidate],
    tokens: tuple[_Token, ...],
) -> bool:
    if candidate.target.semantic_operation not in {
        "set_climate_temperature",
        "set_seat_heating",
    }:
        return False

    by_clause: dict[int, list[_Token]] = {}
    for token in tokens:
        by_clause.setdefault(token.clause, []).append(token)
    for clause_tokens in by_clause.values():
        words = {token.canonical for token in clause_tokens}
        positions = {token.canonical: token.position for token in clause_tokens}
        explicit_pair = "driver" in words and "passenger" in words
        both_group = "both" in words and bool(
            words.intersection(
                {"driver", "me", "passenger", "seats", "us", "you", "zones"}
            )
        )
        all_group = "all" in words and bool(words.intersection({"seats", "zones"}))
        whole_vehicle_group = (
            candidate.target.semantic_operation == "set_climate_temperature"
            and bool(words.intersection({"entire", "whole"}))
            and bool(words.intersection({"cabin", "car", "interior", "vehicle"}))
        )
        if not (explicit_pair or both_group or all_group or whole_vehicle_group):
            continue
        relevant = [
            positions[word]
            for word in (
                "all",
                "both",
                "cabin",
                "car",
                "driver",
                "entire",
                "interior",
                "me",
                "passenger",
                "seats",
                "us",
                "vehicle",
                "whole",
                "you",
                "zones",
            )
            if word in positions
        ]
        mention = _Mention(min(relevant), max(relevant), clause_tokens[0].clause)
        if whole_vehicle_group:
            same_operation = [
                other
                for other in candidates.values()
                if other.target.semantic_operation
                == candidate.target.semantic_operation
            ]
            continuation_tokens = {
                "a",
                "about",
                "approximately",
                "around",
                "by",
                "degree",
                "for",
                "like",
                "roughly",
                "the",
            }
            if len(same_operation) == 1 and any(
                tokens[anchor].sentence == tokens[mention.start].sentence
                and 0 < mention.start - anchor <= 10
                and all(
                    token.canonical in continuation_tokens
                    or token.canonical.isdigit()
                    or token.canonical in _SMALL_NUMBERS
                    for token in tokens[anchor + 1 : mention.start]
                )
                for anchor in candidate.anchor_positions
            ):
                return True
        binding_candidates = candidates
        if both_group or all_group or whole_vehicle_group:
            binding_candidates = {
                goal_id: other
                for goal_id, other in candidates.items()
                if goal_id == candidate.goal.goal_id
                or other.target.semantic_operation
                != candidate.target.semantic_operation
            }
        if _mention_is_bound(mention, candidate, binding_candidates, tokens):
            return True
    return False


def _explicit_defrost_window(
    candidate: _GoalCandidate,
    candidates: dict[str, _GoalCandidate],
    tokens: tuple[_Token, ...],
) -> str | None:
    if candidate.target.semantic_operation != "set_window_defrost":
        return None

    anchor_sentences = {
        tokens[position].sentence for position in candidate.anchor_positions
    }
    if not anchor_sentences:
        return None

    matched: set[str] = set()
    for index in range(len(tokens) - 1):
        first, second = tokens[index : index + 2]
        if (
            first.sentence not in anchor_sentences
            or second.sentence != first.sentence
            or second.canonical not in {"window", "windshield"}
        ):
            continue
        value = {"front": "FRONT", "rear": "REAR", "back": "REAR"}.get(first.canonical)
        if value is None:
            continue
        mention = _Mention(first.position, second.position, first.clause)
        if (
            _mention_is_bound(mention, candidate, candidates, tokens)
            or len(candidates) == 1
        ):
            matched.add(value)
    return next(iter(matched)) if len(matched) == 1 else None


def _explicit_soc_range(tokens: tuple[_Token, ...]) -> tuple[int, int] | None:
    ranges: set[tuple[int, int]] = set()
    for start, token in enumerate(tokens):
        if token.canonical != "from":
            continue
        stop = next(
            (
                index
                for index in range(start + 1, min(len(tokens), start + 10))
                if tokens[index].canonical in {"to", "until"}
            ),
            None,
        )
        if stop is None:
            continue
        initial = [
            int(item.canonical)
            for item in tokens[start + 1 : stop]
            if item.canonical.isdigit()
        ]
        final = [
            int(item.canonical)
            for item in tokens[stop + 1 : min(len(tokens), stop + 5)]
            if item.canonical.isdigit()
        ]
        if len(initial) == 1 and len(final) == 1 and 0 <= final[0] <= initial[0] <= 100:
            ranges.add((initial[0], final[0]))
    return next(iter(ranges)) if len(ranges) == 1 else None


def _select_candidate(
    goal: Goal,
    registry: RecipeRegistry,
    tokens: tuple[_Token, ...],
    action_request_positions: tuple[int, ...],
    read_request_clauses: frozenset[int],
) -> _GoalCandidate | None:
    options = _candidate_options(
        goal,
        registry,
        tokens,
        action_request_positions,
        read_request_clauses,
    )
    if len(options) == 1:
        return options[0]
    if not options or any(option.is_action for option in options):
        return None

    primary = tuple(
        option
        for option in options
        if option.recipe.primary_operation == option.target.semantic_operation
    )
    if len(primary) == 1:
        return primary[0]
    if primary:
        options = primary

    # A direct read can legitimately be reused as a prerequisite by several
    # recipes. Retain it only when every remaining recipe exposes the exact
    # same read contract and accepts the same grounded semantic values.
    if any(
        option.target.semantic_operation != goal.semantic_operation
        for option in options
    ):
        return None
    target_contracts = {
        json.dumps(
            option.target.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for option in options
    }
    desired_contracts = {
        json.dumps(
            _canonical_desired_values(option),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        for option in options
    }
    if len(target_contracts) != 1 or len(desired_contracts) != 1:
        return None
    return min(
        options,
        key=lambda option: (len(option.recipe.direct_operations), option.recipe.id),
    )


def _recipe_manages_hypothetical_dependency(
    goal: Goal,
    dependency: Goal,
    registry: RecipeRegistry,
    tokens: tuple[_Token, ...],
    hypothetical_sentences: frozenset[int],
) -> bool:
    dependency_tools: set[str] = set()
    dependency_anchors: set[int] = set()
    for recipe in registry:
        if not recipe.matches_semantic_operation(dependency.semantic_operation):
            continue
        target = recipe.target_operation(dependency.semantic_operation)
        if target is None:
            continue
        dependency_tools.update(target.tool_names)
        dependency_anchors.update(_anchor_positions(dependency, recipe, target, tokens))
    if not dependency_tools or not dependency_anchors:
        return False
    if any(
        tokens[position].sentence not in hypothetical_sentences
        for position in dependency_anchors
    ):
        return False

    for recipe in registry:
        if not recipe.matches_semantic_operation(goal.semantic_operation):
            continue
        if any(
            dependency_tools.intersection(requirement.operation.tool_names)
            for requirement in recipe.conditional_requirements
        ):
            return True
    return False


def _ground_global_value(key: str, value: Any, tokens: tuple[_Token, ...]) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(
            _ground_global_value(nested_key, nested_value, tokens)
            for nested_key, nested_value in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value) and all(
            _ground_global_value(key, item, tokens) for item in value
        )
    return bool(_scalar_mentions(key, value, tokens))


def _values_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


_SEMANTIC_PARAMETER_ALIASES = {
    ("set_new_navigation", "destination"): "location",
    ("get_distance_by_soc", "final_state_of_charge"): "final_soc",
    ("get_distance_by_soc", "initial_state_of_charge"): "initial_soc",
    (_NAMED_LOCATION_POI_SEARCH_OPERATION, "category_poi"): "category",
    ("navigation_add_one_waypoint", "waypoint_id_to_add"): "new_waypoint_id",
    (
        "navigation_add_one_waypoint",
        "waypoint_id_after_new_waypoint",
    ): "next_waypoint_id",
    (
        "navigation_add_one_waypoint",
        "waypoint_id_before_new_waypoint",
    ): "previous_waypoint_id",
    (
        "navigation_delete_final_destination",
        "destination_id_to_delete",
    ): "destination_id",
    (
        "navigation_delete_one_waypoint",
        "route_id_without_waypoint",
    ): "replacement_route_id",
    (
        "navigation_delete_one_waypoint",
        "waypoint_id_to_delete",
    ): "waypoint_id",
    (
        "navigation_replace_final_destination",
        "route_id_leading_to_new_destination",
    ): "route_id",
    ("set_air_circulation", "air_circulation_mode"): "mode",
    ("set_ambient_lights", "lightcolor"): "color",
    ("set_climate_temperature", "zone"): "seat_zone",
    ("set_fan_airflow_direction", "airflow_direction"): "direction",
    ("set_seat_heating", "seat"): "seat_zone",
    ("set_trunk_position", "position"): "action",
    ("set_window_defrost", "defrost_window"): "window",
}
_CONTEXTUAL_SEMANTIC_PARAMETERS = {
    _NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION: frozenset(
        {
            "new_waypoint_name",
            "next_waypoint_name",
            "previous_waypoint_name",
            "route_choice_alias",
        }
    ),
    _NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION: frozenset({"route_choice_alias"}),
    _NEXT_NAVIGATION_STOP_POI_SEARCH_OPERATION: frozenset({"category"}),
    _NAMED_NAVIGATION_WAYPOINT_DELETE_OPERATION: frozenset(
        {
            "remaining_destination_name",
            "route_choice_alias",
            "route_destination_name",
        }
    ),
    _NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_OPERATION: frozenset(
        {"remaining_destination_name"}
    ),
}


def canonical_semantic_parameter_name(semantic_operation: str, parameter: str) -> str:
    """Return the invariant semantic key for one model-provided alias."""

    return _SEMANTIC_PARAMETER_ALIASES.get((semantic_operation, parameter), parameter)


def _canonical_desired_values(candidate: _GoalCandidate) -> dict[str, Any]:
    allowed = {
        parameter
        for operation in candidate.recipe.direct_operations
        for parameter in operation.parameter_mapping
    }
    allowed.update(
        selector
        for (operation, _), selector in _NAVIGATION_ENTITY_NAME_SELECTORS.items()
        if operation == candidate.target.semantic_operation
    )
    allowed.update(
        selector
        for (operation, _), selector in _POI_ENTITY_NAME_SELECTORS.items()
        if operation == candidate.target.semantic_operation
    )
    allowed.update(
        _CONTEXTUAL_SEMANTIC_PARAMETERS.get(
            candidate.target.semantic_operation, frozenset()
        )
    )
    allowed.update(
        _CONTEXTUAL_SEMANTIC_PARAMETERS.get(
            candidate.goal.semantic_operation, frozenset()
        )
    )
    canonical: dict[str, Any] = {}
    conflicts: set[str] = set()
    for key, value in candidate.goal.desired_outcome.items():
        semantic_key = canonical_semantic_parameter_name(
            candidate.target.semantic_operation, key
        )
        if semantic_key not in allowed or semantic_key in conflicts:
            continue
        if semantic_key in canonical and not _values_equal(
            canonical[semantic_key], value
        ):
            canonical.pop(semantic_key, None)
            conflicts.add(semantic_key)
            continue
        canonical[semantic_key] = value
    return canonical


def _is_literal_navigation_entity_id(value: Any) -> bool:
    return isinstance(value, str) and value.casefold().startswith(("loc_", "poi_"))


def _is_generic_navigation_entity_name(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    words = tuple(
        _canonical_word(token)
        for token in re.findall(
            r"[a-z0-9]+", unicodedata.normalize("NFKC", value).casefold()
        )
    )
    while words and words[0] in {"a", "an", "my", "the"}:
        words = words[1:]
    if not words:
        return True
    normalized = " ".join(words)
    if normalized in _GENERIC_NAVIGATION_ENTITY_NAMES:
        return True
    return set(words).issubset(
        {
            "current",
            "destination",
            "final",
            "intermediate",
            "last",
            "middle",
            "new",
            "stop",
            "waypoint",
        }
    )


def _navigation_name_selector_values(
    candidate: _GoalCandidate, desired: dict[str, Any]
) -> dict[str, Any]:
    operation = candidate.target.semantic_operation
    selector_by_id = {
        identifier: selector
        for (semantic_operation, identifier), selector in (
            _NAVIGATION_ENTITY_NAME_SELECTORS.items()
        )
        if semantic_operation == operation
    }
    if not selector_by_id:
        return desired
    id_by_selector = {
        selector: identifier for identifier, selector in selector_by_id.items()
    }
    normalized: dict[str, Any] = {}
    conflicts: set[str] = set()
    for key, value in desired.items():
        normalized_key = key
        if key in selector_by_id:
            if _is_literal_navigation_entity_id(value):
                normalized_key = key
            elif _is_generic_navigation_entity_name(value):
                continue
            else:
                normalized_key = selector_by_id[key]
        elif key in id_by_selector:
            if _is_literal_navigation_entity_id(value):
                normalized_key = id_by_selector[key]
            elif _is_generic_navigation_entity_name(value):
                continue
        if normalized_key in conflicts:
            continue
        if normalized_key in normalized and not _values_equal(
            normalized[normalized_key], value
        ):
            normalized.pop(normalized_key, None)
            conflicts.add(normalized_key)
            continue
        normalized[normalized_key] = value
    return normalized


def _navigation_destination_has_explicit_alternatives(
    raw_user_text: str,
    candidate: _GoalCandidate,
    desired: dict[str, Any],
) -> bool:
    """Reject a singular destination when the user explicitly names alternatives."""

    if candidate.target.semantic_operation != "navigation_replace_final_destination":
        return False
    destination = desired.get("new_destination_name")
    if not isinstance(destination, str) or not destination.strip():
        return False
    destination_words = re.findall(
        r"[^\W\d_][\w'\u2019-]*",
        unicodedata.normalize("NFKC", destination),
        re.UNICODE,
    )
    if not destination_words:
        return False
    destination_pattern = r"\s+".join(re.escape(word) for word in destination_words)
    word_pattern = r"[^\W\d_][\w'\u2019-]*"
    choice_prefix = r"(?:(?:either|one\s+of)\s+)?"
    choice_connector = (
        r"\s*(?:[,;]\s*)?"
        r"(?P<connector>/|\b(?:or|plus|versus|and(?:\s+(?:then|also))?)\b)"
        r"\s*(?:to\s+)?"
    )
    patterns = (
        re.compile(
            rf"\bto\s+{choice_prefix}{destination_pattern}\b"
            rf"{choice_connector}"
            rf"(?P<alternative>{word_pattern})\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\bto\s+{choice_prefix}"
            rf"(?P<alternative>{word_pattern}(?:\s+{word_pattern}){{0,3}}?)"
            rf"{choice_connector}"
            rf"{destination_pattern}\b",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(raw_user_text):
            connector = " ".join(match.group("connector").casefold().split())
            if not connector.startswith("and"):
                return True
            alternative = match.group("alternative")
            alternative_words = re.findall(
                r"[^\W\d_][\w'\u2019-]*", alternative, re.UNICODE
            )
            if _canonical_word(alternative_words[0]) in (
                _NAVIGATION_AND_DISCOURSE_STARTS
            ):
                continue
            if _canonical_word(
                alternative
            ) == "home" or _looks_like_named_location_literal(alternative):
                return True
    return False


def _poi_name_selector_values(
    candidate: _GoalCandidate, desired: dict[str, Any]
) -> dict[str, Any]:
    operation = candidate.target.semantic_operation
    selector_by_id = {
        identifier: selector
        for (
            semantic_operation,
            identifier,
        ), selector in _POI_ENTITY_NAME_SELECTORS.items()
        if semantic_operation == operation
    }
    if not selector_by_id:
        return desired
    normalized = dict(desired)
    for identifier, selector in selector_by_id.items():
        value = normalized.pop(identifier, None)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        if value.casefold().startswith("poi_"):
            continue
        if value.casefold().startswith("loc_"):
            normalized[identifier] = value
            continue
        existing = normalized.get(selector)
        if existing is None or _values_equal(existing, value):
            normalized[selector] = value
        else:
            normalized.pop(selector, None)
    return normalized


def _navigation_action_has_derived_parameters(candidate: _GoalCandidate) -> bool:
    return bool(candidate.recipe.parameter_derivations) and (
        candidate.target.semantic_operation in _NAVIGATION_DERIVED_ACTIONS
    )


def _fallback_kind(
    original: IntentKind,
    tokens: tuple[_Token, ...],
    has_action: bool,
    has_read: bool,
) -> IntentKind:
    if has_action:
        return IntentKind.ACTION
    if has_read:
        return IntentKind.INFORMATION
    if original in {IntentKind.CONFIRMATION, IntentKind.SELECTION}:
        return original
    first = tokens[0].canonical if tokens else ""
    return (
        IntentKind.INFORMATION
        if first in _QUESTION_STARTERS
        else IntentKind.CONVERSATION
    )


def has_explicit_action_request(raw_user_text: str) -> bool:
    """Return whether the current utterance has fail-closed action grammar."""

    grounding_text = _without_quoted_segments(raw_user_text)
    tokens = _tokenize(grounding_text)
    return bool(_explicit_request_positions(grounding_text, tokens, _ACTION_VERBS))


def semantic_value_is_explicit(key: str, value: Any, raw_user_text: str) -> bool:
    """Return whether a scalar or nested semantic value is literally present."""

    tokens = _tokenize(_without_quoted_segments(raw_user_text))
    if isinstance(value, dict):
        return bool(value) and all(
            semantic_value_is_explicit(nested_key, nested_value, raw_user_text)
            for nested_key, nested_value in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value) and all(
            semantic_value_is_explicit(key, item, raw_user_text) for item in value
        )
    return bool(_scalar_mentions(key, value, tokens))


def focus_explicit_action_request(raw_user_text: str) -> str:
    """Remove hypothetical sentences from the model's focused action view."""

    safe_segments: list[str] = []
    for match in _SENTENCE_SEGMENT_RE.finditer(raw_user_text):
        segment = match.group(0).strip()
        if not segment:
            continue
        tokens = _tokenize(_without_quoted_segments(segment))
        safe_maybe = _polite_relative_temperature_maybe_positions(tokens)
        if (
            any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
            and not safe_maybe
        ):
            continue
        safe_segments.append(segment)
    return " ".join(safe_segments)


def _local_command_clause(raw_user_text: str, position: int) -> str:
    start = 0
    end = len(raw_user_text)
    for boundary in _LOCAL_COMMAND_BOUNDARY_RE.finditer(raw_user_text):
        if boundary.end() <= position:
            start = boundary.end()
        elif boundary.start() >= position:
            end = boundary.start()
            break
    return raw_user_text[start:end].strip(" ,:\t\r\n")


def _command_match_is_explicit(
    raw_user_text: str,
    position: int,
    verbs: frozenset[str],
    *,
    allow_relation_prefix: bool = False,
) -> bool:
    clause = _local_command_clause(raw_user_text, position)
    tokens = _tokenize(clause)
    if (
        not tokens
        or _NEGATED_RE.search(clause)
        or _REPORTED_REQUEST_RE.search(clause)
        or _NON_COMMAND_META_RE.search(clause)
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
    ):
        return False
    verb_positions = [token.position for token in tokens if token.canonical in verbs]
    if len(verb_positions) != 1:
        return False
    prefix = tuple(token.canonical for token in tokens[: verb_positions[0]])
    while prefix and prefix[0] in _DISCOURSE_PREFIXES | {
        "actually",
        "friend",
    }:
        prefix = prefix[1:]
    allowed_prefixes = set(_POLITE_PREFIXES) | {
        (modal, subject)
        for modal in ("can", "could", "will", "would")
        for subject in ("we", "you")
    }
    relation_prefix = bool(
        allow_relation_prefix and len(prefix) == 2 and prefix[0] in {"after", "before"}
    )
    return not prefix or prefix in allowed_prefixes or relation_prefix


def _has_named_location_alternatives(clause: str) -> bool:
    for match in _NAMED_LOCATION_ALTERNATIVE_RE.finditer(clause):
        left = unicodedata.normalize("NFKC", match.group("left"))
        right = unicodedata.normalize("NFKC", match.group("right"))
        if (
            left[:1].isupper()
            and right[:1].isupper()
            and _looks_like_named_location_literal(left)
            and _looks_like_named_location_literal(right)
        ):
            return True
    return False


def _strict_navigation_location(value: str) -> str | None:
    normalized = " ".join(unicodedata.normalize("NFKC", value).strip().split())
    if (
        not normalized
        or normalized.casefold().startswith(("loc_", "poi_"))
        or not _looks_like_named_location_literal(normalized)
    ):
        return None
    return normalized


def _named_navigation_waypoint_add(
    raw_user_text: str,
) -> _NamedWaypointAdd | None:
    grounding_text = _without_quoted_segments(raw_user_text)
    if _has_conflicting_navigation_route_choices(raw_user_text):
        return None
    matches = tuple(
        match
        for pattern in _NAMED_NAVIGATION_WAYPOINT_ADD_RES
        for match in pattern.finditer(grounding_text)
        if _command_match_is_explicit(
            grounding_text,
            match.start(),
            frozenset({"add"}),
            allow_relation_prefix=True,
        )
    )
    if len(matches) != 1:
        return None
    match = matches[0]
    if _has_named_location_alternatives(grounding_text):
        return None
    new_waypoint = _strict_navigation_location(match.group("new"))
    adjacent = _strict_navigation_location(match.group("adjacent"))
    if (
        new_waypoint is None
        or adjacent is None
        or new_waypoint.casefold() == adjacent.casefold()
    ):
        return None
    relation_parameter = (
        "previous_waypoint_name"
        if match.group("relation").casefold() == "after"
        else "next_waypoint_name"
    )
    return _NamedWaypointAdd(
        new_waypoint_name=new_waypoint,
        adjacent_waypoint_name=adjacent,
        adjacency_parameter=relation_parameter,
        position=match.start(),
    )


def _vague_navigation_waypoint_add_is_explicit(raw_user_text: str) -> bool:
    grounding_text = _without_quoted_segments(raw_user_text)
    matches = tuple(
        match
        for match in _VAGUE_NAVIGATION_WAYPOINT_ADD_RE.finditer(grounding_text)
        if _command_match_is_explicit(grounding_text, match.start(), frozenset({"add"}))
    )
    return len(matches) == 1


def _next_navigation_stop_poi_query_position(raw_user_text: str) -> int | None:
    grounding_text = _without_quoted_segments(raw_user_text)
    scopes = tuple(_NEXT_NAVIGATION_STOP_POI_SCOPE_RE.finditer(grounding_text))
    queries = tuple(_NEXT_NAVIGATION_STOP_POI_QUERY_RE.finditer(grounding_text))
    if len(scopes) != 1 or len(queries) != 1:
        return None
    scope = scopes[0]
    query = queries[0]
    query_clause = _local_command_clause(grounding_text, query.start())
    scope_clause = _local_command_clause(grounding_text, scope.start())
    query_verb = _canonical_word(query.group(0).split()[0])
    if (
        query_clause != scope_clause
        or query_verb not in {"find", "search", "show"}
        or not _command_match_is_explicit(
            grounding_text, query.start(), frozenset({query_verb})
        )
        or _has_named_location_alternatives(query_clause)
        or re.search(
            r"\b(?:current|final)\s+(?:destination|location|stop)\b",
            query_clause,
            re.IGNORECASE,
        )
    ):
        return None
    return query.start()


def _named_navigation_waypoint_replacement(
    raw_user_text: str,
) -> _NamedWaypointReplacement | None:
    grounding_text = _without_quoted_segments(raw_user_text)
    if _has_conflicting_navigation_route_choices(raw_user_text):
        return None
    matches = tuple(
        match
        for match in _NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_RE.finditer(grounding_text)
        if _command_match_is_explicit(
            grounding_text, match.start(), frozenset({"replace"})
        )
    )
    if len(matches) != 1:
        return None
    match = matches[0]
    clause = _local_command_clause(grounding_text, match.start())
    clause_tokens = {token.canonical for token in _tokenize(clause)}
    if clause_tokens.intersection(
        {"destination", "final"}
    ) or _has_named_location_alternatives(grounding_text):
        return None
    old_waypoint = _strict_navigation_location(match.group("old"))
    new_waypoint = _strict_navigation_location(match.group("new"))
    if (
        old_waypoint is None
        or new_waypoint is None
        or old_waypoint.casefold() == new_waypoint.casefold()
    ):
        return None
    return _NamedWaypointReplacement(
        waypoint_name_to_replace=old_waypoint,
        new_waypoint_name=new_waypoint,
        position=match.start(),
    )


def _has_conflicting_navigation_route_choices(raw_user_text: str) -> bool:
    grounding_text = _without_quoted_segments(raw_user_text)
    matches = tuple(_EXPLICIT_ROUTE_CHOICE_RE.finditer(grounding_text))
    if len(matches) > 1:
        return True
    if not matches:
        return False
    clause = _local_command_clause(grounding_text, matches[0].start())
    clause_tokens = {token.canonical for token in _tokenize(clause)}
    return len(
        clause_tokens.intersection(
            {"fastest", "shortest", "first", "second", "third"}
        )
    ) != 1


def _explicit_navigation_route_choice(raw_user_text: str) -> str | None:
    grounding_text = _without_quoted_segments(raw_user_text)
    matches = tuple(_EXPLICIT_ROUTE_CHOICE_RE.finditer(grounding_text))
    if len(matches) != 1 or _has_conflicting_navigation_route_choices(raw_user_text):
        return None
    clause = _local_command_clause(grounding_text, matches[0].start())
    clause_tokens = {token.canonical for token in _tokenize(clause)}
    route_choices = clause_tokens.intersection(
        {"fastest", "shortest", "first", "second", "third"}
    )
    if (
        _NEGATED_RE.search(clause)
        or _REPORTED_REQUEST_RE.search(clause)
        or _NON_COMMAND_META_RE.search(clause)
        or clause_tokens.intersection(_HYPOTHETICAL_MARKERS)
        or len(route_choices) != 1
    ):
        return None
    return next(iter(route_choices))


def _recovered_goal_id(
    intent: IntentFrame,
    *,
    semantic_operation: str,
    compatible_operations: frozenset[str],
    used_goal_ids: set[str],
    turn_id: str,
    raw_user_text: str,
) -> str:
    existing = next(
        (
            goal
            for goal in intent.goals
            if goal.goal_id not in used_goal_ids
            and goal.source is GoalSource.USER
            and goal.semantic_operation == semantic_operation
        ),
        None,
    )
    if existing is None:
        existing = next(
            (
                goal
                for goal in intent.goals
                if goal.goal_id not in used_goal_ids
                and goal.source is GoalSource.USER
                and goal.semantic_operation in compatible_operations
            ),
            None,
        )
    if existing is not None:
        used_goal_ids.add(existing.goal_id)
        return existing.goal_id
    digest = hashlib.sha256(
        f"{turn_id}\0{semantic_operation}\0{raw_user_text}".encode()
    ).hexdigest()
    goal_id = f"goal-navigation-context-{digest[:12]}"
    used_goal_ids.add(goal_id)
    return goal_id


def recover_simple_navigation_create_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Recover one unconditional direct navigation command to a named place."""

    normalized = unicodedata.normalize("NFKC", raw_user_text).strip()
    if (
        not normalized
        or len(normalized) > 240
        or _NEGATED_RE.search(normalized) is not None
        or _REPORTED_REQUEST_RE.search(normalized) is not None
        or _NON_COMMAND_META_RE.search(normalized) is not None
        or any(
            goal.source is not GoalSource.USER
            or goal.semantic_operation
            not in _SIMPLE_NAVIGATION_CREATE_MODEL_OPERATIONS
            for goal in intent.goals
        )
    ):
        return intent
    unquoted = _without_quoted_segments(normalized)
    tokens = _tokenize(unquoted)
    if (
        unquoted != normalized
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
    ):
        return intent
    match = _SIMPLE_NAVIGATION_CREATE_REQUEST_RE.fullmatch(unquoted)
    if match is None:
        return intent
    location = _safe_navigation_location_phrase(match.group("location"))
    if location is None or _is_generic_navigation_entity_name(location):
        return intent
    location_tokens = tuple(_tokenize(location))
    location_words = {token.canonical for token in location_tokens}
    if (
        location_words.intersection({"also", "and", "or", "plus", "then"})
        or location_words.intersection({"here", "home", "nearby", "there", "work"})
        or (location_tokens and location_tokens[0].canonical in {"my", "that", "this"})
    ):
        return intent

    used_goal_ids: set[str] = set()
    goal_id = _recovered_goal_id(
        intent,
        semantic_operation=_SIMPLE_NAVIGATION_CREATE_OPERATION,
        compatible_operations=_SIMPLE_NAVIGATION_CREATE_MODEL_OPERATIONS,
        used_goal_ids=used_goal_ids,
        turn_id=turn_id,
        raw_user_text=raw_user_text,
    )
    goal = Goal(
        goal_id=goal_id,
        semantic_operation=_SIMPLE_NAVIGATION_CREATE_OPERATION,
        desired_outcome={"location": location},
        source=GoalSource.USER,
    )
    return intent.model_copy(
        update={
            "call_for_action": True,
            "goals": [goal],
            "explicit_slots": {"location": location},
            "explicit_constraints": {},
            "explicit_confirmations": [],
            "unresolved_slots": [],
            "intent_source_turn_ids": list(
                dict.fromkeys([*intent.intent_source_turn_ids, turn_id])
            ),
            "goal_mention_order": [goal_id],
            "intent_kind": IntentKind.ACTION,
        },
        deep=True,
    )


def recover_navigation_resume_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Canonicalize one explicit request to resume the live stopped route."""

    normalized = unicodedata.normalize("NFKC", raw_user_text).strip()
    if (
        not normalized
        or len(normalized) > 360
        or _NEGATED_RE.search(normalized) is not None
        or _REPORTED_REQUEST_RE.search(normalized) is not None
        or _NON_COMMAND_META_RE.search(normalized) is not None
    ):
        return intent
    unquoted = _without_quoted_segments(normalized)
    if unquoted != normalized or any(
        token.canonical in _HYPOTHETICAL_MARKERS for token in _tokenize(unquoted)
    ):
        return intent
    match = _NAVIGATION_RESUME_REQUEST_RE.fullmatch(unquoted)
    if match is None:
        return intent

    verb = match.group("verb").casefold()
    scope = match.group("scope")
    object_name = match.group("object").casefold()
    backward_verb = verb.startswith(("restart", "resum", "continu"))
    if not backward_verb and scope is None:
        return intent
    if object_name == "trip" and scope is None:
        return intent
    if object_name == "navigation system" and scope is None:
        return intent

    used_goal_ids: set[str] = set()
    goal_id = _recovered_goal_id(
        intent,
        semantic_operation=_NAVIGATION_RESUME_OPERATION,
        compatible_operations=frozenset(
            {
                _NAVIGATION_RESUME_OPERATION,
                "create_navigation",
                "navigation_create",
                "set_new_navigation",
                "start_navigation",
            }
        ),
        used_goal_ids=used_goal_ids,
        turn_id=turn_id,
        raw_user_text=raw_user_text,
    )
    goal = Goal(
        goal_id=goal_id,
        semantic_operation=_NAVIGATION_RESUME_OPERATION,
        desired_outcome={},
        source=GoalSource.USER,
    )
    return intent.model_copy(
        update={
            "call_for_action": True,
            "goals": [goal],
            "explicit_slots": {},
            "explicit_constraints": {},
            "explicit_confirmations": [],
            "unresolved_slots": [],
            "intent_source_turn_ids": list(
                dict.fromkeys([*intent.intent_source_turn_ids, turn_id])
            ),
            "goal_mention_order": [goal.goal_id],
            "intent_kind": IntentKind.ACTION,
        },
        deep=True,
    )


def recover_navigation_waypoint_context_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Canonicalize strict waypoint edits and next-stop POI reads by clause."""

    named_add = _named_navigation_waypoint_add(raw_user_text)
    next_stop_poi_position = _next_navigation_stop_poi_query_position(raw_user_text)
    named_replacement = _named_navigation_waypoint_replacement(raw_user_text)
    if (
        named_add is None
        and next_stop_poi_position is None
        and named_replacement is None
    ):
        return intent
    if named_add is not None and named_replacement is not None:
        return intent
    if any(
        goal.source is not GoalSource.USER
        or goal.semantic_operation not in _NAVIGATION_WAYPOINT_CONTEXT_MODEL_OPERATIONS
        for goal in intent.goals
    ):
        return intent

    route_choice_alias = _explicit_navigation_route_choice(raw_user_text)
    used_goal_ids: set[str] = set()
    recovered: list[tuple[int, Goal]] = []
    if named_add is not None:
        goal_id = _recovered_goal_id(
            intent,
            semantic_operation=_NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION,
            compatible_operations=_NAVIGATION_WAYPOINT_ADD_MODEL_OPERATIONS,
            used_goal_ids=used_goal_ids,
            turn_id=turn_id,
            raw_user_text=raw_user_text,
        )
        recovered.append(
            (
                named_add.position,
                Goal(
                    goal_id=goal_id,
                    semantic_operation=_NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION,
                    desired_outcome=named_add.desired_outcome(route_choice_alias),
                    source=GoalSource.USER,
                ),
            )
        )
    if next_stop_poi_position is not None:
        goal_id = _recovered_goal_id(
            intent,
            semantic_operation=_NEXT_NAVIGATION_STOP_POI_SEARCH_OPERATION,
            compatible_operations=_NEXT_NAVIGATION_STOP_POI_MODEL_OPERATIONS,
            used_goal_ids=used_goal_ids,
            turn_id=turn_id,
            raw_user_text=raw_user_text,
        )
        recovered.append(
            (
                next_stop_poi_position,
                Goal(
                    goal_id=goal_id,
                    semantic_operation=_NEXT_NAVIGATION_STOP_POI_SEARCH_OPERATION,
                    desired_outcome={"category": "restaurants"},
                    source=GoalSource.USER,
                ),
            )
        )
    if named_replacement is not None:
        goal_id = _recovered_goal_id(
            intent,
            semantic_operation=_NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION,
            compatible_operations=_NAVIGATION_WAYPOINT_REPLACEMENT_MODEL_OPERATIONS,
            used_goal_ids=used_goal_ids,
            turn_id=turn_id,
            raw_user_text=raw_user_text,
        )
        recovered.append(
            (
                named_replacement.position,
                Goal(
                    goal_id=goal_id,
                    semantic_operation=(
                        _NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION
                    ),
                    desired_outcome=named_replacement.desired_outcome(
                        route_choice_alias
                    ),
                    source=GoalSource.USER,
                ),
            )
        )

    recovered_goals = [goal for _, goal in sorted(recovered, key=lambda item: item[0])]
    explicit_slots = {
        key: value
        for goal in recovered_goals
        for key, value in goal.desired_outcome.items()
    }
    has_action = any(
        goal.semantic_operation
        in {
            _NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION,
            _NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION,
        }
        for goal in recovered_goals
    )
    return intent.model_copy(
        update={
            "call_for_action": has_action,
            "goals": recovered_goals,
            "explicit_slots": explicit_slots,
            "explicit_constraints": {},
            "explicit_confirmations": [],
            "unresolved_slots": [],
            "intent_source_turn_ids": list(
                dict.fromkeys([*intent.intent_source_turn_ids, turn_id])
            ),
            "goal_mention_order": [goal.goal_id for goal in recovered_goals],
            "intent_kind": IntentKind.ACTION if has_action else IntentKind.INFORMATION,
        },
        deep=True,
    )


def _model_named_location_literals(intent: IntentFrame) -> tuple[str, ...]:
    literals: dict[str, str] = {}
    for goal in intent.goals:
        if (
            goal.semantic_operation not in _NAMED_LOCATION_POI_MODEL_OPERATIONS
            or goal.source is not GoalSource.USER
        ):
            continue
        for key, value in goal.desired_outcome.items():
            if (
                key.casefold() not in {"location", "location_id", "location_name"}
                or not isinstance(value, str)
                or not value.strip()
                or value.casefold().startswith(("loc_", "poi_"))
                or not _looks_like_named_location_literal(value)
            ):
                continue
            literal = " ".join(unicodedata.normalize("NFKC", value).strip().split())
            literals.setdefault(literal.casefold(), literal)
    return tuple(literals.values())


def _current_destination_restaurant_search_is_explicit(
    raw_user_text: str,
    tokens: tuple[_Token, ...],
) -> bool:
    scope_matches = tuple(
        match
        for pattern in _CURRENT_DESTINATION_POI_SCOPE_RES
        for match in pattern.finditer(raw_user_text)
    )
    destination_references = tuple(
        re.finditer(
            rf"\b{_CURRENT_DESTINATION_REFERENCE}\b",
            raw_user_text,
            re.IGNORECASE,
        )
    )
    return bool(
        len(scope_matches) == 1
        and len(destination_references) == 1
        and _CURRENT_DESTINATION_POI_QUERY_RE.search(raw_user_text)
        and _CURRENT_DESTINATION_POI_CONFLICTING_SCOPE_RE.search(raw_user_text) is None
        and _poi_category_identities(tokens) == frozenset({"restaurants"})
        and not any(token.canonical in _ACTION_VERBS for token in tokens)
    )


def recover_current_destination_poi_search_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Recover one restaurant search scoped to the active route destination."""

    grounding_text = _without_quoted_segments(raw_user_text)
    tokens = _tokenize(grounding_text)
    if (
        not tokens
        or grounding_text != raw_user_text
        or _NEGATED_RE.search(grounding_text)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
        or any(
            goal.semantic_operation not in _CURRENT_DESTINATION_POI_MODEL_OPERATIONS
            or goal.source is not GoalSource.USER
            for goal in intent.goals
        )
        or not _current_destination_restaurant_search_is_explicit(
            grounding_text, tokens
        )
    ):
        return intent

    existing = next(
        (
            goal
            for goal in intent.goals
            if goal.semantic_operation == _CURRENT_DESTINATION_POI_SEARCH_OPERATION
        ),
        None,
    )
    if existing is None:
        digest = hashlib.sha256(
            f"{turn_id}\0{_CURRENT_DESTINATION_POI_SEARCH_OPERATION}\0"
            f"{grounding_text}".encode()
        ).hexdigest()
        goal_id = f"goal-current-destination-poi-{digest[:12]}"
    else:
        goal_id = existing.goal_id
    recovered_goal = Goal(
        goal_id=goal_id,
        semantic_operation=_CURRENT_DESTINATION_POI_SEARCH_OPERATION,
        desired_outcome={"category": "restaurants"},
        source=GoalSource.USER,
    )
    return intent.model_copy(
        update={
            "call_for_action": False,
            "goals": [recovered_goal],
            "explicit_slots": {"category": "restaurants"},
            "explicit_constraints": {},
            "explicit_confirmations": [],
            "unresolved_slots": [],
            "intent_source_turn_ids": list(
                dict.fromkeys([*intent.intent_source_turn_ids, turn_id])
            ),
            "goal_mention_order": [goal_id],
            "intent_kind": IntentKind.INFORMATION,
        },
        deep=True,
    )


def _named_location_restaurant_search_literal(
    raw_user_text: str,
    tokens: tuple[_Token, ...],
    *,
    model_location_literals: tuple[str, ...],
) -> str | None:
    matches = tuple(_NAMED_LOCATION_POI_SEARCH_PREFIX_RE.finditer(raw_user_text))
    if len(matches) != 1:
        return None
    tail = _NAMED_LOCATION_POI_SEARCH_TAIL_RE.fullmatch(
        raw_user_text[matches[0].end() :]
    )
    if tail is None:
        return None
    location = " ".join(tail.group("location").strip().split())
    location_words = tuple(re.findall(r"[^\W\d_][\w'\u2019-]*", location, re.UNICODE))
    exact_model_location = (
        len(model_location_literals) == 1
        and unicodedata.normalize("NFKC", model_location_literals[0]).casefold()
        == unicodedata.normalize("NFKC", location).casefold()
    )
    if (
        not location
        or location.casefold().startswith(("loc_", "poi_"))
        or not _looks_like_named_location_literal(location)
        or (len(location_words) != 1 and not exact_model_location)
        or _poi_category_identities(tokens) != frozenset({"restaurants"})
        or not any(
            token.canonical in {"find", "search", "show"}
            and _has_request_prefix(tokens, token.position)
            for token in tokens
        )
    ):
        return None
    return location


def recover_named_location_poi_search_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Recover one read-only restaurant search at one literal named location."""

    grounding_text = _without_quoted_segments(raw_user_text)
    tokens = _tokenize(grounding_text)
    if (
        not tokens
        or grounding_text != raw_user_text
        or _NEGATED_RE.search(grounding_text)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
        or has_explicit_action_request(raw_user_text)
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
        or any(
            goal.semantic_operation not in _NAMED_LOCATION_POI_MODEL_OPERATIONS
            or goal.source is not GoalSource.USER
            for goal in intent.goals
        )
    ):
        return intent

    location = _named_location_restaurant_search_literal(
        grounding_text,
        tokens,
        model_location_literals=_model_named_location_literals(intent),
    )
    if location is None:
        return intent

    existing = next(
        (
            goal
            for goal in intent.goals
            if goal.semantic_operation == _NAMED_LOCATION_POI_SEARCH_OPERATION
        ),
        None,
    )
    if existing is None:
        digest = hashlib.sha256(
            f"{turn_id}\0{_NAMED_LOCATION_POI_SEARCH_OPERATION}\0{grounding_text}".encode()
        ).hexdigest()
        goal_id = f"goal-named-location-poi-{digest[:12]}"
    else:
        goal_id = existing.goal_id
    recovered_goal = Goal(
        goal_id=goal_id,
        semantic_operation=_NAMED_LOCATION_POI_SEARCH_OPERATION,
        desired_outcome={"category": "restaurants", "location_name": location},
        source=GoalSource.USER,
    )
    return intent.model_copy(
        update={
            "call_for_action": False,
            "goals": [recovered_goal],
            "explicit_slots": {
                "category": "restaurants",
                "location_name": location,
            },
            "explicit_constraints": {},
            "explicit_confirmations": [],
            "unresolved_slots": [],
            "intent_source_turn_ids": list(
                dict.fromkeys([*intent.intent_source_turn_ids, turn_id])
            ),
            "goal_mention_order": [goal_id],
            "intent_kind": IntentKind.INFORMATION,
        },
        deep=True,
    )


def _legacy_named_navigation_destination_replacement_name(
    raw_user_text: str,
) -> str | None:
    """Return the new destination only for an approved complete request grammar."""

    grounding_text = _without_quoted_segments(raw_user_text)
    tokens = _tokenize(grounding_text)
    if (
        not tokens
        or grounding_text != raw_user_text
        or _NEGATED_RE.search(grounding_text)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
    ):
        return None

    strict_matches = tuple(
        match
        for pattern in _NAMED_NAVIGATION_DESTINATION_REPLACEMENT_PATTERNS
        if (match := pattern.fullmatch(grounding_text)) is not None
    )
    if len(strict_matches) > 1:
        return None
    if strict_matches:
        match = strict_matches[0]
        location_raw = match.group("location")
        previous_raw = match.groupdict().get("previous")
        names = (
            (location_raw,) if previous_raw is None else (location_raw, previous_raw)
        )
        normalized_names = tuple(
            unicodedata.normalize("NFKC", name).strip() for name in names
        )
        if any(
            re.fullmatch(r"[^\W\d_]+", name, re.UNICODE) is None
            or not _looks_like_named_location_literal(name)
            or name.casefold().startswith(("loc_", "poi_"))
            for name in normalized_names
        ):
            return None
        if (
            len(normalized_names) == 2
            and normalized_names[0].casefold() == normalized_names[1].casefold()
        ):
            return None
        return normalized_names[0]

    mutation_tokens = tuple(
        token for token in tokens if token.canonical in {"change", "replace", "switch"}
    )
    request_positions = _explicit_request_positions(
        grounding_text, tokens, frozenset({"change", "replace", "switch"})
    )
    prefixes = tuple(
        _NAMED_NAVIGATION_DESTINATION_REPLACEMENT_PREFIX_RE.finditer(grounding_text)
    )
    if (
        len(mutation_tokens) != 1
        or len(request_positions) != 1
        or len(prefixes) != 1
        or _NAMED_NAVIGATION_DESTINATION_REPLACEMENT_LEAD_RE.fullmatch(
            grounding_text[: prefixes[0].start()]
        )
        is None
    ):
        return None

    tail = _NAMED_NAVIGATION_DESTINATION_REPLACEMENT_TAIL_RE.fullmatch(
        grounding_text[prefixes[0].end() :]
    )
    if tail is None:
        return None
    location = " ".join(
        unicodedata.normalize("NFKC", tail.group("location")).strip().split()
    )
    location_words = tuple(re.findall(r"[^\W\d_][\w'\u2019-]*", location, re.UNICODE))
    if (
        not _looks_like_named_location_literal(location)
        or location.casefold().startswith(("loc_", "poi_"))
        or re.fullmatch(r"[^\W\d_]+", location, re.UNICODE) is None
        or len(location_words) != 1
    ):
        return None
    return location


def _is_destination_proper_venue_name(value: str) -> bool:
    words = tuple(re.findall(r"[^\W\d_][\w'\u2019-]*", value, re.UNICODE))
    if not 2 <= len(words) <= 5:
        return False
    canonical = tuple(_canonical_word(word) for word in words)
    return canonical[-1] in _DESTINATION_PROPER_VENUE_SUFFIXES and all(
        word[0].isupper() or name in _NAMED_LOCATION_LOWERCASE_PARTICLES
        for word, name in zip(words, canonical, strict=True)
    )


def _destination_phrase_has_action(value: str) -> bool:
    normalized = " ".join(unicodedata.normalize("NFKC", value).strip().split())
    if _is_destination_proper_venue_name(normalized):
        return False
    return any(
        pattern.search(normalized) for pattern in _DESTINATION_ACTION_PHRASE_PATTERNS
    )


def _safe_navigation_location_phrase(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(unicodedata.normalize("NFKC", value).strip().split())
    if (
        not normalized
        or len(normalized) > 80
        or normalized.casefold().startswith(("loc_", "poi_"))
        or not _looks_like_named_location_literal(normalized)
        or _destination_phrase_has_action(normalized)
    ):
        return None
    return normalized


def _strict_adjacent_destination_name(value: str | None) -> str | None:
    """Accept one proper-name location while rejecting reply/control vocabulary."""

    location = _safe_navigation_location_phrase(value)
    if location is None:
        return None
    canonical_words = {
        token.canonical for token in _tokenize(location) if token.canonical
    }
    if canonical_words.intersection(_ADJACENT_DESTINATION_NON_CITY_WORDS):
        return None
    return location


def parse_descriptive_navigation_destination_request(
    raw_user_text: str,
) -> DescriptiveNavigationDestinationRequest | None:
    """Parse an explicit replacement whose requested city is only described."""

    grounding_text = _without_quoted_segments(raw_user_text)
    if (
        not grounding_text.strip()
        or grounding_text != raw_user_text
        or len(grounding_text) > 480
        or _NEGATED_RE.search(grounding_text)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
        or any(
            token.canonical in _HYPOTHETICAL_MARKERS
            for token in _tokenize(grounding_text)
        )
    ):
        return None
    matches = tuple(
        match
        for pattern in _DESCRIPTIVE_NAVIGATION_DESTINATION_PATTERNS
        if (match := pattern.fullmatch(grounding_text)) is not None
    )
    if len(matches) != 1:
        return None
    groups = matches[0].groupdict()
    previous = _safe_navigation_location_phrase(groups.get("previous"))
    description = " ".join(groups["description"].casefold().split())
    route = groups["route"].casefold()
    if (
        previous is None
        or route not in {"fastest", "shortest"}
        or _NON_COMMAND_LOCATION_DESCRIPTION_RE.fullmatch(description) is None
        or _destination_phrase_has_action(description)
    ):
        return None
    return DescriptiveNavigationDestinationRequest(
        previous_destination_name=previous,
        route_choice_alias=route,
    )


def parse_adjacent_navigation_destination_reply(
    raw_user_text: str,
    *,
    inherited_route_choice: str | None,
) -> AdjacentNavigationDestinationReply | None:
    """Resolve one asserted city only for an immediately pending request."""

    grounding_text = _without_quoted_segments(raw_user_text)
    tokens = _tokenize(grounding_text)
    if (
        not grounding_text.strip()
        or grounding_text != raw_user_text
        or len(grounding_text) > 480
        or _NEGATED_RE.search(grounding_text)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
        or _ADJACENT_DESTINATION_CANCELLATION_RE.search(grounding_text)
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
    ):
        return None
    route_aliases = {
        match.group(1).casefold()
        for match in re.finditer(
            r"\b(fastest|shortest)\s+route\b",
            grounding_text,
            re.IGNORECASE,
        )
    }
    if len(route_aliases) > 1:
        return None
    reply_route = next(iter(route_aliases), None)
    inherited = (
        inherited_route_choice.casefold()
        if isinstance(inherited_route_choice, str)
        else None
    )
    if inherited not in {None, "fastest", "shortest"} or (
        inherited is not None and reply_route is not None and inherited != reply_route
    ):
        return None

    approved = tuple(
        match
        for pattern in _ADJACENT_DESTINATION_REPLY_PATTERNS
        if (match := pattern.fullmatch(grounding_text)) is not None
    )
    if len(approved) > 1:
        return None
    locations: dict[str, str] = {}
    if approved:
        groups = approved[0].groupdict()
        visit_object = groups.get("visit_object")
        if isinstance(visit_object, str) and (
            _destination_phrase_has_action(visit_object)
        ):
            return None
        captured_locations: dict[str, str] = {}
        for name in ("location", "repeated"):
            raw_location = groups.get(name)
            if raw_location is None:
                continue
            location = _strict_adjacent_destination_name(raw_location)
            if location is None:
                return None
            captured_locations[name] = location
            locations[location.casefold()] = location
        if (
            "location" in captured_locations
            and "repeated" in captured_locations
            and captured_locations["location"].casefold()
            != captured_locations["repeated"].casefold()
        ):
            return None
    standalone = _ADJACENT_DESTINATION_STANDALONE_RE.fullmatch(grounding_text)
    if standalone is not None:
        exact = _strict_adjacent_destination_name(standalone.group("location"))
        if exact is not None:
            locations.setdefault(exact.casefold(), exact)
    if len(locations) != 1:
        return None
    destination = next(iter(locations.values()))
    return AdjacentNavigationDestinationReply(
        destination_name=destination,
        route_choice_alias=reply_route or inherited,
        route_choice_from_reply=reply_route is not None,
    )


def _named_navigation_destination_replacement_request(
    raw_user_text: str,
) -> tuple[str, str | None, str | None] | None:
    """Parse one complete literal replacement and optional route constraint."""

    grounding_text = _without_quoted_segments(raw_user_text)
    if (
        not grounding_text.strip()
        or grounding_text != raw_user_text
        or len(grounding_text) > 480
        or _NEGATED_RE.search(grounding_text)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
        or any(
            token.canonical in _HYPOTHETICAL_MARKERS
            for token in _tokenize(grounding_text)
        )
    ):
        return None

    compound = tuple(
        match
        for pattern in (
            _NAMED_NAVIGATION_DESTINATION_WITH_ROUTE_RE,
            _NAMED_NAVIGATION_DESTINATION_FROM_TO_RE,
            _NAMED_NAVIGATION_DESTINATION_SIMPLE_FROM_TO_RE,
        )
        if (match := pattern.fullmatch(grounding_text)) is not None
    )
    if len(compound) > 1:
        return None
    if compound:
        groups = compound[0].groupdict()
        location = _safe_navigation_location_phrase(
            groups.get("location") or groups.get("current_location")
        )
        repeated = _safe_navigation_location_phrase(groups.get("repeated"))
        previous = _safe_navigation_location_phrase(
            groups.get("previous") or groups.get("current")
        )
        route = groups.get("route")
        if (
            location is None
            or (repeated is not None and repeated.casefold() != location.casefold())
            or (previous is not None and previous.casefold() == location.casefold())
            or (route is not None and route.casefold() not in {"fastest", "shortest"})
        ):
            return None
        return (
            location,
            previous,
            route.casefold() if route is not None else None,
        )

    location = _legacy_named_navigation_destination_replacement_name(raw_user_text)
    if location is None or _destination_phrase_has_action(location):
        return None
    return location, None, None


def _named_navigation_destination_replacement_name(
    raw_user_text: str,
) -> str | None:
    request = _named_navigation_destination_replacement_request(raw_user_text)
    return request[0] if request is not None else None


def recover_named_navigation_destination_replacement_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Recover one explicit named replacement of the active final destination."""

    request = _named_navigation_destination_replacement_request(raw_user_text)
    if request is None or any(
        goal.semantic_operation
        not in _NAMED_NAVIGATION_DESTINATION_REPLACEMENT_MODEL_OPERATIONS
        or goal.source is not GoalSource.USER
        for goal in intent.goals
    ):
        return intent
    location, previous, route_choice = request

    existing = next(
        (
            goal
            for goal in intent.goals
            if goal.semantic_operation
            == _NAMED_NAVIGATION_DESTINATION_REPLACEMENT_OPERATION
        ),
        None,
    )
    if existing is None:
        existing = next(iter(intent.goals), None)
    if existing is None:
        digest = hashlib.sha256(
            (
                f"{turn_id}\0"
                f"{_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_OPERATION}\0"
                f"{raw_user_text}"
            ).encode()
        ).hexdigest()
        goal_id = f"goal-named-destination-replacement-{digest[:12]}"
    else:
        goal_id = existing.goal_id
    desired_outcome = {"new_destination_name": location}
    if previous is not None:
        desired_outcome["previous_destination_name"] = previous
    if route_choice is not None:
        desired_outcome["route_choice_alias"] = route_choice
    recovered_goal = Goal(
        goal_id=goal_id,
        semantic_operation=_NAMED_NAVIGATION_DESTINATION_REPLACEMENT_OPERATION,
        desired_outcome=desired_outcome,
        source=GoalSource.USER,
    )
    return intent.model_copy(
        update={
            "call_for_action": True,
            "goals": [recovered_goal],
            "explicit_slots": desired_outcome,
            "explicit_constraints": {},
            "explicit_confirmations": [],
            "unresolved_slots": [],
            "intent_source_turn_ids": list(
                dict.fromkeys([*intent.intent_source_turn_ids, turn_id])
            ),
            "goal_mention_order": [goal_id],
            "intent_kind": IntentKind.ACTION,
        },
        deep=True,
    )


def _named_navigation_final_destination_delete_names(
    raw_user_text: str,
) -> tuple[str, str | None] | None:
    """Return one deleted/final literal pair only for the approved request grammar."""

    grounding_text = _without_quoted_segments(raw_user_text)
    contextual_matches = tuple(
        match
        for pattern in _NAMED_NAVIGATION_CONTEXTUAL_GENERIC_DELETE_PATTERNS
        if (match := pattern.fullmatch(grounding_text)) is not None
    )
    if (
        not grounding_text.strip()
        or grounding_text != raw_user_text
        or (_NEGATED_RE.search(grounding_text) and len(contextual_matches) != 1)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
    ):
        return None
    matches = tuple(
        match
        for pattern in _NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_PATTERNS
        if (match := pattern.fullmatch(grounding_text)) is not None
    )
    generic_matches = tuple(
        match
        for pattern in (_NAMED_NAVIGATION_GENERIC_DELETE_RE,)
        if (match := pattern.fullmatch(grounding_text)) is not None
    )
    generic_matches = (*generic_matches, *contextual_matches)
    if len(matches) + len(generic_matches) != 1:
        return None
    selected_match = matches[0] if matches else generic_matches[0]
    deleted_raw = selected_match.group("deleted")
    remaining_raw = selected_match.group("remaining") if matches else None
    if re.fullmatch(r"[^\W\d_]+", deleted_raw, re.UNICODE) is None or (
        remaining_raw is not None
        and re.fullmatch(r"[^\W\d_]+", remaining_raw, re.UNICODE) is None
    ):
        return None
    deleted = unicodedata.normalize("NFKC", deleted_raw)
    remaining = (
        unicodedata.normalize("NFKC", remaining_raw)
        if remaining_raw is not None
        else None
    )
    if (
        not _looks_like_named_location_literal(deleted)
        or deleted.casefold().startswith(("loc_", "poi_"))
        or (
            remaining is not None
            and (
                not _looks_like_named_location_literal(remaining)
                or deleted.casefold() == remaining.casefold()
                or remaining.casefold().startswith(("loc_", "poi_"))
            )
        )
    ):
        return None
    return deleted, remaining


def recover_named_navigation_final_destination_delete_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Recover one explicit named deletion of the active final destination."""

    names = _named_navigation_final_destination_delete_names(raw_user_text)
    if names is None or any(
        goal.semantic_operation
        not in _NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_MODEL_OPERATIONS
        or goal.source is not GoalSource.USER
        for goal in intent.goals
    ):
        return intent
    deleted, remaining = names
    existing = next(
        (
            goal
            for goal in intent.goals
            if goal.semantic_operation
            == _NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_OPERATION
        ),
        None,
    )
    if existing is None:
        digest = hashlib.sha256(
            (
                f"{turn_id}\0"
                f"{_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_OPERATION}\0"
                f"{raw_user_text}"
            ).encode()
        ).hexdigest()
        goal_id = f"goal-named-final-destination-delete-{digest[:12]}"
    else:
        goal_id = existing.goal_id
    desired_outcome = {"destination_name_to_delete": deleted}
    if remaining is not None:
        desired_outcome["remaining_destination_name"] = remaining
    recovered_goal = Goal(
        goal_id=goal_id,
        semantic_operation=_NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_OPERATION,
        desired_outcome=desired_outcome,
        source=GoalSource.USER,
    )
    return intent.model_copy(
        update={
            "call_for_action": True,
            "goals": [recovered_goal],
            "explicit_slots": desired_outcome,
            "explicit_constraints": {},
            "explicit_confirmations": [],
            "unresolved_slots": [],
            "intent_source_turn_ids": list(
                dict.fromkeys([*intent.intent_source_turn_ids, turn_id])
            ),
            "goal_mention_order": [goal_id],
            "intent_kind": IntentKind.ACTION,
        },
        deep=True,
    )


def _legacy_named_navigation_waypoint_delete_names(
    raw_user_text: str,
) -> tuple[str, str | None] | None:
    """Return one named intermediate waypoint and optional retained destination."""

    grounding_text = _without_quoted_segments(raw_user_text)
    tokens = _tokenize(grounding_text)
    if (
        not tokens
        or grounding_text != raw_user_text
        or _NEGATED_RE.search(grounding_text)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
        or sum(token.canonical == "remove" for token in tokens) != 1
    ):
        return None
    matches = tuple(
        match
        for pattern in (
            _NAMED_NAVIGATION_WAYPOINT_DELETE_ROUTE_RE,
            _NAMED_NAVIGATION_WAYPOINT_DELETE_STOP_RE,
        )
        if (match := pattern.fullmatch(grounding_text)) is not None
    )
    if len(matches) != 1:
        return None
    waypoint_raw = matches[0].group("waypoint")
    destination_raw = matches[0].groupdict().get("destination")
    waypoint = unicodedata.normalize("NFKC", waypoint_raw)
    destination = (
        unicodedata.normalize("NFKC", destination_raw)
        if destination_raw is not None
        else None
    )
    generic_names = {
        "current",
        "destination",
        "final",
        "intermediate",
        "last",
        "navigation",
        "next",
        "route",
    }
    if (
        re.fullmatch(r"[^\W\d_]+", waypoint, re.UNICODE) is None
        or not _looks_like_named_location_literal(waypoint)
        or waypoint.casefold() in generic_names
        or waypoint.casefold().startswith(("loc_", "poi_"))
    ):
        return None
    if destination is not None and (
        re.fullmatch(r"[^\W\d_]+", destination, re.UNICODE) is None
        or not _looks_like_named_location_literal(destination)
        or destination.casefold() in generic_names
        or destination.casefold().startswith(("loc_", "poi_"))
        or destination.casefold() == waypoint.casefold()
    ):
        return None
    return waypoint, destination


def _strip_named_waypoint_delete_discourse_prefix(
    text: str,
) -> tuple[str, str | None] | None:
    """Strip one bounded repair preface while preserving repeated-name evidence."""

    patterns = (
        re.compile(
            r"\s*i(?:['\u2019]m|\s+am)\s+sorry\s*,\s*i\s+thought\s+you\s+"
            r"(?:had\s+)?already\s+removed\s+(?:the\s+)?"
            r"(?P<context_waypoint>[^,.!?;]{1,80}?)"
            r"(?:\s+(?:stop|waypoint))?\s*[.!]\s*",
            re.IGNORECASE | re.UNICODE,
        ),
        re.compile(
            r"\s*but\s+you\s+(?:haven['\u2019]?t|have\s+not)\s+removed\s+"
            r"(?:the\s+)?(?P<context_waypoint>[^,.!?;]{1,80}?)"
            r"(?:\s+(?:stop|waypoint))?\s+yet\s*[.!]\s*",
            re.IGNORECASE | re.UNICODE,
        ),
        re.compile(
            r"\s*(?:okay|ok)\s*,\s*i\s+understand\s*[.!]\s*"
            r"(?:so\s*,\s*)?",
            re.IGNORECASE | re.UNICODE,
        ),
        re.compile(
            r"\s*(?:okay|ok)\s*,\s*i(?:['\u2019]m|\s+am)\s+getting\s+"
            r"(?:a\s+bit\s+)?confused\s*[.!]\s*",
            re.IGNORECASE | re.UNICODE,
        ),
    )
    matches = [match for pattern in patterns if (match := pattern.match(text))]
    if len(matches) > 1:
        return None
    if not matches:
        return text, None
    match = matches[0]
    remaining = text[match.end() :]
    if not remaining.strip():
        return None
    raw_context = match.groupdict().get("context_waypoint")
    context = _safe_navigation_location_phrase(raw_context)
    if raw_context is not None and context is None:
        return None
    return remaining, context


def _named_navigation_waypoint_delete_request(
    raw_user_text: str,
) -> tuple[str, str | None, str | None, str | None] | None:
    """Parse one named intermediate-stop deletion and optional route choice."""

    legacy = _legacy_named_navigation_waypoint_delete_names(raw_user_text)
    if legacy is not None:
        return legacy[0], legacy[1], None, None

    grounding_text = _without_quoted_segments(raw_user_text)
    if (
        grounding_text != raw_user_text
        or len(grounding_text) > 520
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
    ):
        return None
    stripped = _strip_named_waypoint_delete_discourse_prefix(grounding_text)
    if stripped is None:
        return None
    grounding_text, prefix_waypoint = stripped
    tokens = _tokenize(grounding_text)
    if (
        not tokens
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
        or re.search(
            r"\b(?:do\s+not|don['\u2019]?t|never)\s+(?:remove|take)\b",
            grounding_text,
            re.IGNORECASE,
        )
    ):
        return None
    patterns = (
        (
            r"\s*(?:(?:now\s+)?(?:the\s+)?"
            r"(?P<context_waypoint>[^,.!?;]{1,80}?)\s+"
            r"(?:stop|waypoint)\s+is\s+no\s+longer\s+needed\s*[.!]\s*)?"
            r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|please\s*,?\s+)"
            r"(?:just\s+)?remove\s+(?:the\s+)?"
            r"(?P<waypoint>[^,.!?;]{1,80}?)(?:\s+(?:stop|waypoint))?\s+"
            r"from\s+(?:my|the)\s+(?:current\s+)?route"
            r"(?:\s+and\s+(?:then\s+)?(?:(?:can|could|would|will)\s+you\s+"
            r"(?:please\s+)?|please\s+)?(?:use|take|find)\s+(?:the\s+)?"
            r"(?P<route>fastest|shortest)\s+route(?:\s+for\s+me)?"
            r"(?:\s+after\s+(?:that|removing\s+(?:the\s+)?"
            r"(?P<route_waypoint>[^,.!?;]{1,80}?)(?:\s+(?:stop|waypoint))?))?"
            r")?(?:\s*,\s*please)?\s*[.!?]*\s*"
        ),
        (
            r"\s*(?:now\s+(?:that(?:['\u2019]s|\s+is)\s+done|this\s+is\s+done)"
            r"\s*,?\s*)?i\s+(?:also\s+)?(?:need|want)\s+to\s+remove\s+"
            r"(?:the\s+)?(?P<waypoint>[^,.!?;]{1,80}?)"
            r"(?:\s+(?:stop|waypoint))?\s+from\s+"
            r"(?:my|the)\s+(?:current\s+)?route\s*[.!]\s*"
            r"it(?:['\u2019]s|\s+is)\s+no\s+longer\s+(?:necessary|needed)"
            r"\s*[.!]\s*(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
            r"please\s+)(?:find|use|take)\s+(?:the\s+)?"
            r"(?P<route>fastest|shortest)\s+route(?:\s+for\s+me)?\s+"
            r"after\s+removing\s+(?:the\s+)?"
            r"(?P<route_waypoint>[^,.!?;]{1,80}?)(?:\s+(?:stop|waypoint))?"
            r"(?:\s*,\s*please)?\s*[.!?]*\s*"
        ),
        (
            r"\s*(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
            r"please\s*,?\s+)"
            r"(?:(?:just\s+)?remove\s+(?:the\s+)?"
            r"(?P<waypoint>[^,.!?;]{1,80}?)(?:\s+(?:stop|waypoint))?\s+"
            r"from\s+(?:my|the)\s+(?:current\s+)?route|"
            r"(?:just\s+)?take\s+(?:the\s+)?"
            r"(?P<taken_waypoint>[^,.!?;]{1,80}?)(?:\s+(?:stop|waypoint))?\s+"
            r"out\s+of\s+(?:my|the)\s+(?:current\s+)?route)"
            r"(?:\s+now)?(?:\s*,?\s+(?:and\s+then|then|and)|"
            r"\s*[.!?]\s*(?:and\s+then|then|"
            r"once\s+that(?:['\u2019]s|\s+is)\s+done\s*,?\s*(?:then)?))"
            r"\s*,?\s*(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
            r"please\s+)?(?:find\s+|show\s+(?:me\s+)?)"
            r"(?:the\s+)?(?P<route>fastest|shortest)\s+route\s+to\s+"
            r"(?:the\s+)?(?P<route_destination>[^,.!?;]{1,80}?)"
            r"(?:\s*,\s*please)?\s*[.!?]*\s*"
        ),
        (
            r"\s*i\s+(?:just\s+)?need\s+to\s+remove\s+(?:the\s+)?"
            r"(?P<waypoint>[^,.!?;]{1,80}?)(?:\s+(?:stop|waypoint))?\s+"
            r"from\s+(?:my|the)\s+(?:current\s+)?route\s*[.!]\s*"
            r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|please\s+)"
            r"do\s+that(?:\s+for\s+me)?\s*[.!?]*\s*"
        ),
        (
            r"\s*i\s+(?:just\s+)?need\s+to\s+remove\s+(?:a|one)\s+"
            r"(?:stop|waypoint)(?:\s+from\s+(?:my|the)\s+(?:current\s+)?route)?"
            r"\s*[.!]\s*please\s+remove\s+(?:the\s+)?"
            r"(?P<waypoint>[^,.!?;]{1,80}?)(?:\s+(?:stop|waypoint))?"
            r"(?:\s+from\s+(?:my|the)\s+(?:current\s+)?route)?\s*[.!?]*\s*"
        ),
    )
    matches = [
        match
        for pattern in patterns
        if (match := re.fullmatch(pattern, grounding_text, re.IGNORECASE | re.UNICODE))
        is not None
    ]
    if len(matches) != 1:
        return None
    match = matches[0]
    context_waypoint_raw = match.groupdict().get("context_waypoint")
    route_waypoint_raw = match.groupdict().get("route_waypoint")
    waypoint_raw = match.groupdict().get("waypoint") or match.groupdict().get(
        "taken_waypoint"
    )
    route_destination_raw = match.groupdict().get("route_destination")
    waypoint = _safe_navigation_location_phrase(waypoint_raw)
    context_waypoint = _safe_navigation_location_phrase(context_waypoint_raw)
    route_waypoint = _safe_navigation_location_phrase(route_waypoint_raw)
    route_destination = _safe_navigation_location_phrase(route_destination_raw)
    if (
        waypoint is None
        or waypoint.casefold() == "it"
        or (context_waypoint_raw is not None and context_waypoint is None)
        or (route_waypoint_raw is not None and route_waypoint is None)
        or (route_destination_raw is not None and route_destination is None)
    ):
        return None
    repeated_waypoints = tuple(
        value
        for value in (prefix_waypoint, context_waypoint, route_waypoint)
        if value is not None
    )
    if any(value.casefold() != waypoint.casefold() for value in repeated_waypoints):
        return None
    if route_destination is not None and (
        route_destination.casefold() == waypoint.casefold()
    ):
        return None
    route = match.groupdict().get("route")
    route_choice = route.casefold() if route is not None else None
    return waypoint, None, route_choice, route_destination


def _named_navigation_waypoint_delete_names(
    raw_user_text: str,
) -> tuple[str, str | None, str | None, str | None] | None:
    request = _named_navigation_waypoint_delete_request(raw_user_text)
    return request if request is not None else None


def recover_named_navigation_waypoint_delete_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Recover one strict named intermediate-waypoint deletion request."""

    request = _named_navigation_waypoint_delete_request(raw_user_text)
    if request is None or any(
        goal.semantic_operation
        not in _NAMED_NAVIGATION_WAYPOINT_DELETE_MODEL_OPERATIONS
        or goal.source is not GoalSource.USER
        for goal in intent.goals
    ):
        return intent
    waypoint, destination, route_choice, route_destination = request
    existing = next(
        (
            goal
            for goal in intent.goals
            if goal.semantic_operation == _NAMED_NAVIGATION_WAYPOINT_DELETE_OPERATION
        ),
        None,
    )
    if existing is None:
        digest = hashlib.sha256(
            (
                f"{turn_id}\0{_NAMED_NAVIGATION_WAYPOINT_DELETE_OPERATION}\0"
                f"{raw_user_text}"
            ).encode()
        ).hexdigest()
        goal_id = f"goal-named-waypoint-delete-{digest[:12]}"
    else:
        goal_id = existing.goal_id
    desired_outcome = {"waypoint_name_to_delete": waypoint}
    if destination is not None:
        desired_outcome["remaining_destination_name"] = destination
    if route_choice is not None:
        desired_outcome["route_choice_alias"] = route_choice
    if route_destination is not None:
        desired_outcome["route_destination_name"] = route_destination
    recovered_goal = Goal(
        goal_id=goal_id,
        semantic_operation=_NAMED_NAVIGATION_WAYPOINT_DELETE_OPERATION,
        desired_outcome=desired_outcome,
        source=GoalSource.USER,
    )
    return intent.model_copy(
        update={
            "call_for_action": True,
            "goals": [recovered_goal],
            "explicit_slots": desired_outcome,
            "explicit_constraints": {},
            "explicit_confirmations": [],
            "unresolved_slots": [],
            "intent_source_turn_ids": list(
                dict.fromkeys([*intent.intent_source_turn_ids, turn_id])
            ),
            "goal_mention_order": [goal_id],
            "intent_kind": IntentKind.ACTION,
        },
        deep=True,
    )


def _battery_trip_destination_is_movement_bound(
    destination: str, tokens: tuple[_Token, ...]
) -> bool:
    for mention in _phrase_mentions(destination, tokens):
        sentence_tokens = tuple(
            token
            for token in tokens
            if token.sentence == tokens[mention.start].sentence
        )
        direct_prefix = {
            token.canonical
            for token in sentence_tokens
            if mention.start - 3 <= token.position < mention.start
        }
        if direct_prefix.intersection(_BATTERY_TRIP_MOVEMENT_CUES):
            return True

        destination_prepositions = tuple(
            token
            for token in sentence_tokens
            if token.canonical in {"for", "to", "toward", "towards"}
            and mention.start - 4 <= token.position < mention.start
        )
        if any(
            any(
                movement.canonical in _BATTERY_TRIP_MOVEMENT_CUES
                and preposition.position - 6 <= movement.position < preposition.position
                for movement in sentence_tokens
            )
            for preposition in destination_prepositions
        ):
            return True
    return False


def _battery_trip_model_destinations(
    intent: IntentFrame, tokens: tuple[_Token, ...]
) -> tuple[str, ...]:
    values: list[str] = []
    for goal in intent.goals:
        for key, value in goal.desired_outcome.items():
            if key.casefold() in _BATTERY_TRIP_DESTINATION_KEYS and isinstance(
                value, str
            ):
                values.append(value)
    for key, value in intent.explicit_slots.items():
        if key.casefold() in _BATTERY_TRIP_EXPLICIT_DESTINATION_KEYS and isinstance(
            value, str
        ):
            values.append(value)

    destinations: dict[str, str] = {}
    for value in values:
        normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
        if (
            not normalized
            or normalized.startswith(("loc_", "poi_", "rll_", "rlp_", "rpl_"))
            or not _phrase_mentions(value, tokens)
            or not _battery_trip_destination_is_movement_bound(value, tokens)
        ):
            continue
        destinations.setdefault(normalized, value.strip())
    return tuple(destinations.values())


def _battery_trip_embedded_if_is_safe(tokens: tuple[_Token, ...]) -> bool:
    unsafe_markers = _HYPOTHETICAL_MARKERS.difference({"if"})
    if any(token.canonical in unsafe_markers for token in tokens):
        return False
    for token in tokens:
        if token.canonical != "if":
            continue
        sentence_tokens = tuple(
            item for item in tokens if item.sentence == token.sentence
        )
        if sentence_tokens and sentence_tokens[0].position == token.position:
            return False
        prefix = {
            item.canonical
            for item in sentence_tokens
            if token.position - 4 <= item.position < token.position
        }
        if not prefix.intersection(_READ_VERBS | {"determine", "know", "see"}):
            return False
    return True


def _battery_trip_has_self_modal_movement(
    sentence_tokens: tuple[_Token, ...],
) -> bool:
    for modal in sentence_tokens:
        if modal.canonical not in _BATTERY_TRIP_MODALS:
            continue
        subjects = tuple(
            token
            for token in sentence_tokens
            if token.canonical in _BATTERY_TRIP_SELF_SUBJECTS
            and modal.position - 2 <= token.position <= modal.position + 4
            and token.position != modal.position
        )
        if any(
            any(
                movement.canonical in _BATTERY_TRIP_MOVEMENT_CUES
                and max(subject.position, modal.position)
                < movement.position
                <= max(subject.position, modal.position) + 7
                for movement in sentence_tokens
            )
            for subject in subjects
        ):
            return True
    return False


def _battery_trip_destination_value_is_bound(
    key: str,
    value: Any,
    candidate: _GoalCandidate,
    intent: IntentFrame,
    tokens: tuple[_Token, ...],
) -> bool:
    if (
        candidate.target.semantic_operation != _BATTERY_TRIP_RANGE_OPERATION
        or key != "destination_name"
        or not isinstance(value, str)
    ):
        return False
    destinations = _battery_trip_model_destinations(intent, tokens)
    return (
        len(destinations) == 1
        and unicodedata.normalize("NFKC", destinations[0]).casefold()
        == unicodedata.normalize("NFKC", value).casefold()
        and _battery_trip_has_feasibility_read(value, tokens)
    )


def _battery_trip_has_feasibility_read(
    destination: str, tokens: tuple[_Token, ...]
) -> bool:
    read_sentences = {
        token.sentence
        for token in tokens
        if token.clause in _read_request_clauses(tokens)
    }
    destination_mentions = _phrase_mentions(destination, tokens)
    destination_sentences = {
        tokens[mention.start].sentence for mention in destination_mentions
    }
    for sentence in sorted(read_sentences):
        sentence_tokens = tuple(token for token in tokens if token.sentence == sentence)
        words = {token.canonical for token in sentence_tokens}
        if not words.intersection(_BATTERY_TRIP_ENERGY_CUES):
            continue
        if not words.intersection(_BATTERY_TRIP_MOVEMENT_CUES):
            continue
        feasibility = (
            _battery_trip_has_self_modal_movement(sentence_tokens)
            or (
                bool(words.intersection({"enough", "sufficient"}))
                and bool(words.intersection(_BATTERY_TRIP_ENERGY_CUES))
            )
            or bool(words.intersection({"feasible", "possible"}))
        )
        if not feasibility:
            continue

        if sentence in destination_sentences:
            return True
        earlier_destination = any(
            0 < sentence - destination_sentence <= 2
            for destination_sentence in destination_sentences
        )
        make_it = any(
            first.canonical == "make"
            and second.canonical == "it"
            and first.position < second.position <= first.position + 2
            for first in sentence_tokens
            for second in sentence_tokens
        )
        if earlier_destination and ("there" in words or make_it):
            return True
    return False


def recover_battery_charge_trip_range_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Recover one narrow read-only battery-versus-trip feasibility goal.

    The destination must come from model output, occur literally in the user
    text, and be attached to a movement phrase.  The surrounding utterance must
    independently contain information-question grammar, an energy cue, and a
    self-directed travel-feasibility cue.
    """

    grounding_text = _without_quoted_segments(raw_user_text)
    tokens = _tokenize(grounding_text)
    if (
        not tokens
        or _NEGATED_RE.search(grounding_text)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
        or has_explicit_action_request(raw_user_text)
        or not _battery_trip_embedded_if_is_safe(tokens)
        or any(
            goal.semantic_operation not in _BATTERY_TRIP_RANGE_MODEL_OPERATIONS
            or goal.source is not GoalSource.USER
            for goal in intent.goals
        )
    ):
        return intent

    destinations = _battery_trip_model_destinations(intent, tokens)
    if len(destinations) != 1:
        return intent
    destination = destinations[0]
    if not _battery_trip_has_feasibility_read(destination, tokens):
        return intent

    existing = next(
        (
            goal
            for goal in intent.goals
            if goal.semantic_operation == _BATTERY_TRIP_RANGE_OPERATION
        ),
        None,
    )
    if existing is None:
        digest = hashlib.sha256(
            f"{turn_id}\0{_BATTERY_TRIP_RANGE_OPERATION}\0{grounding_text}".encode()
        ).hexdigest()
        goal_id = f"goal-battery-trip-range-{digest[:12]}"
    else:
        goal_id = existing.goal_id
    recovered_goal = Goal(
        goal_id=goal_id,
        semantic_operation=_BATTERY_TRIP_RANGE_OPERATION,
        desired_outcome={"destination_name": destination},
        source=GoalSource.USER,
    )
    return intent.model_copy(
        update={
            "call_for_action": False,
            "goals": [recovered_goal],
            "explicit_slots": {"destination_name": destination},
            "explicit_constraints": {},
            "explicit_confirmations": [],
            "unresolved_slots": [],
            "intent_source_turn_ids": list(
                dict.fromkeys([*intent.intent_source_turn_ids, turn_id])
            ),
            "goal_mention_order": [goal_id],
            "intent_kind": IntentKind.INFORMATION,
        },
        deep=True,
    )


def recover_current_day_calendar_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Recover a narrow read-only calendar goal from explicit current-day wording."""

    grounding_text = _without_quoted_segments(raw_user_text)
    tokens = _tokenize(grounding_text)
    if (
        _NEGATED_RE.search(grounding_text)
        or _NON_COMMAND_META_RE.search(grounding_text)
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
    ):
        return intent
    read_clauses = _read_request_clauses(tokens)
    objects = {
        "appointment",
        "appointments",
        "calendar",
        "event",
        "events",
        "meeting",
        "meetings",
        "schedule",
    }
    explicit_clauses = {
        clause
        for clause in read_clauses
        if "today" in {token.canonical for token in tokens if token.clause == clause}
        and objects.intersection(
            token.canonical for token in tokens if token.clause == clause
        )
    }
    if not explicit_clauses:
        return intent

    calendar_operations = {
        "calendar_lookup",
        "check_calendar",
        "get_calendar",
        "get_entries_from_calendar",
    }
    if intent.goals and any(
        goal.semantic_operation not in calendar_operations for goal in intent.goals
    ):
        return intent

    digest = hashlib.sha256(f"{turn_id}\0{grounding_text}".encode()).hexdigest()
    goal_id = f"goal-calendar-{digest[:12]}"
    goal = Goal(
        goal_id=goal_id,
        semantic_operation="get_entries_from_calendar",
        desired_outcome={},
        source=GoalSource.USER,
    )
    return intent.model_copy(
        update={
            "call_for_action": False,
            "goals": [goal],
            "explicit_slots": {},
            "explicit_constraints": {},
            "unresolved_slots": [],
            "goal_mention_order": [goal_id],
            "intent_kind": IntentKind.INFORMATION,
        },
        deep=True,
    )


_DRIVER_WARMING_INITIAL_PATTERN = re.compile(
    r"\s*(?:i(?:['\u2019]m|\s+am)\s+feeling\s+a\s+bit\s+cold\s*[.!?]\s*)?"
    r"can\s+you\s+set\s+the\s+driver(?:['\u2019]s|\s+zone)\s+temperature\s+to\s+"
    r"(?:24|twenty[ -]?four)\s+degrees?(?:\s+celsius)?\s+and\s+also\s+turn\s+on\s+"
    r"the\s+seat\s+heating\s+and\s+steering\s+wheel\s+heating\s+for\s+me\s*[?]\s*",
    re.IGNORECASE,
)
_DRIVER_WARMING_SHARED_LEVEL_PATTERN = re.compile(
    r"\s*could\s+you\s+set\s+the\s+seat\s+heating\s+and\s+steering\s+wheel\s+"
    r"heating\s+to\s+level\s+(?:2|two)\s+instead\s*[?]\s*",
    re.IGNORECASE,
)
_DRIVER_WARMING_MATCH_LEVEL_PATTERN = re.compile(
    r"\s*(?:thanks\s*[.!]\s*)?can\s+you\s+also\s+turn\s+on\s+the\s+seat\s+"
    r"heating\s+for\s+the\s+driver(?:['\u2019]s|\s+zone)\s+seat\s+to\s+level\s+"
    r"(?:2|two)\s*,\s*and\s+set\s+the\s+steering\s+wheel\s+heating\s+to\s+"
    r"match\s+that\s+level\s*[?]\s*",
    re.IGNORECASE,
)
_DRIVER_WARMING_INITIAL = "initial"
_DRIVER_WARMING_SHARED_LEVEL = "shared_level"
_DRIVER_WARMING_MATCH_LEVEL = "match_level"

_CLIMATE_ZONE_SYNC_PATTERN = re.compile(
    r"\s*(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
    r"(?:try\s+again\s+to\s+)?"
    r"(?:sync|set)\s+(?:(?:my|the)\s+)?"
    r"(?P<target>driver|passenger)(?:[-\s]+(?:zone|side))?\s+"
    r"climate\s+and\s+(?:seat\s+)?heating\s+settings\s+to\s+"
    r"(?:match\s+)?(?:"
    r"(?:(?:what(?:ever)?\s+)?(?:is\s+)?(?:currently\s+)?set\s+"
    r"(?:on|for)\s+)?(?:the\s+)?"
    r"(?P<source>driver|passenger)(?:[-\s]+(?:zone|side))?"
    r"(?:\s+(?:currently\s+)?has)?"
    r")\s*[?!.]?\s*"
    r"(?:(?:i(?:['\u2019]m|\s+am)\s+(?:okay|fine)\s+with\s+"
    r"(?:syncing\s+)?(?:all|every)\s+(?:the\s+)?climate\s+elements?"
    r"(?:\s+(?:being\s+synced|to\s+be\s+synced))?"
    r"|match\s+(?:all|every)\s+(?:the\s+)?climate\s+elements?)"
    r"\s*[?!.]?\s*)?",
    re.IGNORECASE,
)

_CLIMATE_TEMPERATURE_ZONE_SYNC_PATTERN = re.compile(
    r"\s*(?:(?:can|could|would|will)\s+you(?:\s+please)?\s+|please\s+)?"
    r"(?:set|adjust|change|sync|match)\s+(?:(?:my|the)\s+)?"
    r"(?P<target>driver|passenger)(?:[-\s]+(?:zone|side))?\s+"
    r"(?:climate\s+)?temperature\s+"
    r"(?:to\s+)?(?:the\s+)?(?:same\s+as|to\s+match|matching)\s+"
    r"(?:(?:the|my)\s+)?(?P<source>driver|passenger)"
    r"(?:[-\s]+(?:zone|side))?(?:\s+(?:climate\s+)?temperature)?"
    r"\s*[?!.]?\s*",
    re.IGNORECASE,
)


def _driver_warming_request_kind(raw_user_text: str) -> str | None:
    grounding_text = _without_quoted_segments(raw_user_text)
    tokens = _tokenize(grounding_text)
    if (
        not tokens
        or grounding_text != raw_user_text
        or _NEGATED_RE.search(grounding_text)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
    ):
        return None
    matches = [
        kind
        for kind, pattern in (
            (_DRIVER_WARMING_INITIAL, _DRIVER_WARMING_INITIAL_PATTERN),
            (_DRIVER_WARMING_SHARED_LEVEL, _DRIVER_WARMING_SHARED_LEVEL_PATTERN),
            (_DRIVER_WARMING_MATCH_LEVEL, _DRIVER_WARMING_MATCH_LEVEL_PATTERN),
        )
        if pattern.fullmatch(grounding_text) is not None
    ]
    return matches[0] if len(matches) == 1 else None


def _climate_zone_sync_zones(raw_user_text: str) -> tuple[str, str] | None:
    grounding_text = _without_quoted_segments(raw_user_text)
    tokens = _tokenize(grounding_text)
    if (
        not tokens
        or grounding_text != raw_user_text
        or _NEGATED_RE.search(grounding_text)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
    ):
        return None
    match = _CLIMATE_ZONE_SYNC_PATTERN.fullmatch(grounding_text)
    if match is None:
        return None
    target = match.group("target").upper()
    source = match.group("source").upper()
    if target == source:
        return None
    return target, source


def _climate_temperature_zone_sync_zones(
    raw_user_text: str,
) -> tuple[str, str] | None:
    """Return the explicit target/source zones for one temperature-only sync."""

    grounding_text = _without_quoted_segments(raw_user_text)
    tokens = _tokenize(grounding_text)
    if (
        not tokens
        or grounding_text != raw_user_text
        or _NEGATED_RE.search(grounding_text)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
    ):
        return None
    match = _CLIMATE_TEMPERATURE_ZONE_SYNC_PATTERN.fullmatch(grounding_text)
    if match is None:
        return None
    target = match.group("target").upper()
    source = match.group("source").upper()
    if target == source:
        return None
    return target, source


def recover_climate_zone_sync_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Recover one explicit cross-zone climate-and-heating synchronization."""

    zones = _climate_zone_sync_zones(raw_user_text)
    operations: tuple[str, ...] = (
        "set_climate_temperature",
        "set_seat_heating",
    )
    if zones is None:
        zones = _climate_temperature_zone_sync_zones(raw_user_text)
        operations = ("set_climate_temperature",)
    if zones is None:
        return intent
    target, _ = zones
    existing_by_operation: dict[str, list[Goal]] = {}
    for goal in intent.goals:
        if goal.source is GoalSource.USER:
            existing_by_operation.setdefault(goal.semantic_operation, []).append(goal)

    recovered_goals: list[Goal] = []
    for index, operation in enumerate(operations):
        existing = existing_by_operation.get(operation, [])
        if len(existing) == 1:
            goal_id = existing[0].goal_id
        else:
            digest = hashlib.sha256(
                f"{turn_id}\0climate-zone-sync\0{index}\0{operation}\0{raw_user_text}".encode()
            ).hexdigest()
            goal_id = f"goal-climate-sync-{digest[:12]}"
        recovered_goals.append(
            Goal(
                goal_id=goal_id,
                semantic_operation=operation,
                desired_outcome={"seat_zone": target},
                source=GoalSource.USER,
            )
        )

    return IntentFrame.model_validate(
        {
            **intent.model_dump(mode="python"),
            "call_for_action": True,
            "goals": [goal.model_dump(mode="python") for goal in recovered_goals],
            "explicit_slots": {"seat_zone": target},
            "explicit_constraints": {},
            "explicit_confirmations": [],
            "unresolved_slots": [],
            "intent_source_turn_ids": list(
                dict.fromkeys([*intent.intent_source_turn_ids, turn_id])
            ),
            "goal_mention_order": [goal.goal_id for goal in recovered_goals],
            "intent_kind": IntentKind.ACTION,
        }
    )


def _driver_warming_values(kind: str) -> tuple[tuple[str, dict[str, Any]], ...]:
    if kind == _DRIVER_WARMING_INITIAL:
        return (
            (
                "set_climate_temperature",
                {"temperature": 24, "seat_zone": "DRIVER"},
            ),
            ("set_seat_heating", {"seat_zone": "DRIVER"}),
            ("set_steering_wheel_heating", {}),
        )
    if kind == _DRIVER_WARMING_SHARED_LEVEL:
        return (
            ("set_seat_heating", {"level": 2}),
            ("set_steering_wheel_heating", {"level": 2}),
        )
    if kind == _DRIVER_WARMING_MATCH_LEVEL:
        return (
            ("set_seat_heating", {"level": 2, "seat_zone": "DRIVER"}),
            ("set_steering_wheel_heating", {"level": 2}),
        )
    return ()


def recover_driver_warming_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Recover the strict driver warming request without inventing heat levels."""

    kind = _driver_warming_request_kind(raw_user_text)
    values = _driver_warming_values(kind) if kind is not None else ()
    if not values:
        return intent

    existing_by_operation: dict[str, list[Goal]] = {}
    for goal in intent.goals:
        if goal.source is GoalSource.USER:
            existing_by_operation.setdefault(goal.semantic_operation, []).append(goal)

    goal_ids: dict[str, str] = {}
    for index, (operation, _) in enumerate(values):
        existing = existing_by_operation.get(operation, [])
        if len(existing) == 1:
            goal_ids[operation] = existing[0].goal_id
            continue
        digest = hashlib.sha256(
            f"{turn_id}\0driver-warming\0{index}\0{operation}\0{raw_user_text}".encode()
        ).hexdigest()
        goal_ids[operation] = f"goal-driver-warming-{digest[:12]}"

    recovered_goals: list[Goal] = []
    for operation, desired in values:
        dependencies = []
        if operation == "set_steering_wheel_heating" and kind in {
            _DRIVER_WARMING_SHARED_LEVEL,
            _DRIVER_WARMING_MATCH_LEVEL,
        }:
            dependencies = [goal_ids["set_seat_heating"]]
        recovered_goals.append(
            Goal(
                goal_id=goal_ids[operation],
                semantic_operation=operation,
                desired_outcome=dict(desired),
                depends_on=dependencies,
                source=GoalSource.USER,
            )
        )

    if kind == _DRIVER_WARMING_INITIAL:
        explicit_slots: dict[str, Any] = {
            "temperature": 24,
            "seat_zone": "DRIVER",
        }
    elif kind == _DRIVER_WARMING_MATCH_LEVEL:
        explicit_slots = {"level": 2, "seat_zone": "DRIVER"}
    else:
        explicit_slots = {"level": 2}

    return IntentFrame.model_validate(
        {
            **intent.model_dump(mode="python"),
            "call_for_action": True,
            "goals": [goal.model_dump(mode="python") for goal in recovered_goals],
            "explicit_slots": explicit_slots,
            "explicit_constraints": {},
            "explicit_confirmations": [],
            "unresolved_slots": [],
            "intent_source_turn_ids": list(
                dict.fromkeys([*intent.intent_source_turn_ids, turn_id])
            ),
            "goal_mention_order": [goal.goal_id for goal in recovered_goals],
            "intent_kind": IntentKind.ACTION,
        }
    )


def _strict_driver_warming_values_by_goal(
    intent: IntentFrame, kind: str | None
) -> dict[str, dict[str, Any]]:
    values = _driver_warming_values(kind) if kind is not None else ()
    if not values or len(intent.goals) != len(values):
        return {}
    expected_operations = tuple(operation for operation, _ in values)
    if tuple(goal.semantic_operation for goal in intent.goals) != expected_operations:
        return {}
    expected_values = {operation: desired for operation, desired in values}
    if any(
        goal.source is not GoalSource.USER
        or goal.atomic_group is not None
        or goal.desired_outcome != expected_values[goal.semantic_operation]
        for goal in intent.goals
    ):
        return {}

    seat_goal = next(
        (
            goal
            for goal in intent.goals
            if goal.semantic_operation == "set_seat_heating"
        ),
        None,
    )
    for goal in intent.goals:
        expected_dependencies = (
            [seat_goal.goal_id]
            if seat_goal is not None
            and goal.semantic_operation == "set_steering_wheel_heating"
            and kind in {_DRIVER_WARMING_SHARED_LEVEL, _DRIVER_WARMING_MATCH_LEVEL}
            else []
        )
        if goal.depends_on != expected_dependencies:
            return {}
    if intent.goal_mention_order != [goal.goal_id for goal in intent.goals]:
        return {}
    return {goal.goal_id: dict(goal.desired_outcome) for goal in intent.goals}


_RELATIVE_OCCUPIED_SEAT_HEATING_PATTERN = re.compile(
    r"\s*(?:please\s+)?increase\s+(?:the\s+)?seat\s+heating\s+"
    r"(?:"
    r"by\s+(?P<amount_before>one|two|three|1|2|3)\s+levels?\s+"
    r"for\s+(?:the\s+)?occupied\s+seats?"
    r"|for\s+(?:the\s+)?occupied\s+seats?\s+by\s+"
    r"(?P<amount_after>one|two|three|1|2|3)\s+levels?"
    r")\s*[.!?]*\s*",
    re.IGNORECASE,
)
_RELATIVE_OCCUPIED_SEAT_HEATING_PRONOUN_PATTERN = re.compile(
    r"\s*for\s+(?:the\s+)?seat\s+heating\s*,\s*increase\s+it\s+by\s+"
    r"(?P<amount_before>one|two|three|1|2|3)\s+levels?\s+for\s+"
    r"(?:the\s+)?occupied\s+seats?\s*[.!?]*\s*",
    re.IGNORECASE,
)


def _relative_occupied_seat_heating_delta(
    raw_user_text: str, tokens: tuple[_Token, ...]
) -> int | None:
    """Parse one positive occupied-seat delta without accepting an absolute level."""

    grounding_text = _without_quoted_segments(raw_user_text)
    if (
        not tokens
        or grounding_text != raw_user_text
        or _NEGATED_RE.search(grounding_text)
        or _REPORTED_REQUEST_RE.search(raw_user_text)
        or _NON_COMMAND_META_RE.search(raw_user_text)
        or any(token.canonical in _HYPOTHETICAL_MARKERS for token in tokens)
    ):
        return None
    seat_heating_sentences = {
        token.sentence
        for token in tokens
        if token.canonical == "seat"
        and any(
            other.sentence == token.sentence and other.canonical == "heat"
            for other in tokens
        )
    }
    if len(seat_heating_sentences) != 1:
        return None
    matches = []
    for segment in _SENTENCE_SEGMENT_RE.finditer(grounding_text):
        text = segment.group(0).strip()
        if not text:
            continue
        for pattern in (
            _RELATIVE_OCCUPIED_SEAT_HEATING_PATTERN,
            _RELATIVE_OCCUPIED_SEAT_HEATING_PRONOUN_PATTERN,
        ):
            match = pattern.fullmatch(text)
            if match is not None:
                matches.append(match)
    if len(matches) != 1:
        return None
    amount = matches[0].group("amount_before") or matches[0].group("amount_after")
    return {"one": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}.get(
        amount.casefold()
    )


def _recover_relative_occupied_seat_heating_goal(
    raw_user_text: str,
    intent: IntentFrame,
    tokens: tuple[_Token, ...],
    *,
    source_turn_id: str | None = None,
) -> tuple[IntentFrame, dict[str, int], frozenset[str]]:
    delta = _relative_occupied_seat_heating_delta(raw_user_text, tokens)
    if delta is None:
        return intent, {}, frozenset()

    seat_goals = [
        goal for goal in intent.goals if goal.semantic_operation == "set_seat_heating"
    ]
    if seat_goals:
        primary = seat_goals[0].model_copy(update={"desired_outcome": {}}, deep=True)
    else:
        source_turn = source_turn_id or (
            intent.intent_source_turn_ids[-1]
            if intent.intent_source_turn_ids
            else "current-user-turn"
        )
        digest = hashlib.sha256(
            f"{source_turn}\0relative-occupied-seat-heating\0{raw_user_text}".encode()
        ).hexdigest()
        primary = Goal(
            goal_id=f"goal-relative-seat-heating-{digest[:12]}",
            semantic_operation="set_seat_heating",
            desired_outcome={},
            source=GoalSource.USER,
        )

    duplicate_ids = {goal.goal_id for goal in seat_goals[1:]}
    replacement_ids = {goal_id: primary.goal_id for goal_id in duplicate_ids}
    recovered_goals: list[Goal] = []
    inserted = False
    for goal in intent.goals:
        if goal.goal_id in duplicate_ids:
            continue
        if goal.semantic_operation == "set_seat_heating":
            if inserted:
                continue
            updated = primary
            inserted = True
        else:
            updated = goal.model_copy(deep=True)
        updated = updated.model_copy(
            update={
                "depends_on": list(
                    dict.fromkeys(
                        replacement_ids.get(dependency, dependency)
                        for dependency in updated.depends_on
                    )
                )
            },
            deep=True,
        )
        recovered_goals.append(updated)
    if not inserted:
        seat_anchor = min(
            token.position
            for token in tokens
            if token.canonical == "seat"
            and any(
                other.sentence == token.sentence and other.canonical == "heat"
                for other in tokens
            )
        )

        def goal_mention_position(goal: Goal) -> int | None:
            identity = _name_tokens(goal.semantic_operation).difference(
                {"get", "heat", "level", "read", "seat", "set"}
            )
            positions = [
                token.position for token in tokens if token.canonical in identity
            ]
            return min(positions) if positions else None

        insertion_index = len(recovered_goals)
        for index, goal in enumerate(recovered_goals):
            position = goal_mention_position(goal)
            if position is not None and position > seat_anchor:
                insertion_index = index
                break
        recovered_goals.insert(insertion_index, primary)

    retained_ids = {goal.goal_id for goal in recovered_goals}
    mention_order = list(
        dict.fromkeys(
            replacement_ids.get(goal_id, goal_id)
            for goal_id in intent.goal_mention_order
            if replacement_ids.get(goal_id, goal_id) in retained_ids
        )
    )
    if primary.goal_id not in mention_order:
        primary_index = next(
            index
            for index, goal in enumerate(recovered_goals)
            if goal.goal_id == primary.goal_id
        )
        following_ids = {goal.goal_id for goal in recovered_goals[primary_index + 1 :]}
        mention_index = next(
            (
                index
                for index, goal_id in enumerate(mention_order)
                if goal_id in following_ids
            ),
            len(mention_order),
        )
        mention_order.insert(mention_index, primary.goal_id)
    for goal in recovered_goals:
        if goal.goal_id not in mention_order:
            mention_order.append(goal.goal_id)

    def value_is_owned_by_other_goal(key: str, value: Any) -> bool:
        return any(
            goal.goal_id != primary.goal_id
            and any(
                canonical_semantic_parameter_name(goal.semantic_operation, goal_key)
                == key
                and _values_equal(goal_value, value)
                for goal_key, goal_value in goal.desired_outcome.items()
            )
            for goal in recovered_goals
        )

    recovered = IntentFrame.model_validate(
        {
            **intent.model_dump(mode="python"),
            "call_for_action": True,
            "goals": [goal.model_dump(mode="python") for goal in recovered_goals],
            "explicit_slots": {
                key: value
                for key, value in intent.explicit_slots.items()
                if canonical_semantic_parameter_name("set_seat_heating", key)
                not in {"level", "seat_zone"}
                or value_is_owned_by_other_goal(
                    canonical_semantic_parameter_name("set_seat_heating", key), value
                )
            },
            "unresolved_slots": [
                slot.model_dump(mode="python")
                for slot in intent.unresolved_slots
                if slot.goal_id in retained_ids
                and not (
                    slot.goal_id == primary.goal_id
                    and canonical_semantic_parameter_name("set_seat_heating", slot.name)
                    in {"level", "seat_zone"}
                )
            ],
            "intent_source_turn_ids": list(
                dict.fromkeys(
                    [
                        *intent.intent_source_turn_ids,
                        *([source_turn_id] if source_turn_id else []),
                    ]
                )
            ),
            "goal_mention_order": mention_order,
            "intent_kind": IntentKind.ACTION,
        }
    )
    return recovered, {primary.goal_id: delta}, frozenset({primary.goal_id})


def recover_relative_occupied_seat_heating_intent(
    raw_user_text: str,
    intent: IntentFrame,
    *,
    turn_id: str,
) -> IntentFrame:
    """Recover one strict relative seat-heating goal for occupied seats only."""

    recovered, _, _ = _recover_relative_occupied_seat_heating_goal(
        raw_user_text,
        intent,
        _tokenize(_without_quoted_segments(raw_user_text)),
        source_turn_id=turn_id,
    )
    return recovered


def _relative_fan_speed_delta(tokens: tuple[_Token, ...]) -> int | None:
    """Return an explicit bounded fan-speed delta, never an absolute level."""

    candidates: set[int] = set()
    hypothetical_sentences = {
        token.sentence for token in tokens if token.canonical in _HYPOTHETICAL_MARKERS
    }
    for direction in tokens:
        if direction.sentence in hypothetical_sentences:
            continue
        sign: int | None = None
        if direction.canonical in {"increase", "raise"}:
            sign = 1
        elif direction.canonical in {"decrease", "lower", "reduce"}:
            sign = -1
        elif direction.canonical in {"bump", "turn"}:
            direction_suffix = {
                token.canonical
                for token in tokens
                if token.sentence == direction.sentence
                and direction.position < token.position <= direction.position + 3
            }
            if "up" in direction_suffix and "down" not in direction_suffix:
                sign = 1
            elif "down" in direction_suffix and "up" not in direction_suffix:
                sign = -1
        if sign is None or not _has_request_prefix(tokens, direction.position):
            continue

        clause_prefix = [
            token.canonical
            for token in tokens
            if token.clause == direction.clause
            and direction.position - 4 <= token.position < direction.position
        ]
        if set(clause_prefix).intersection(
            {"not", "never", "without", "don't", "dont"}
        ):
            continue
        same_sentence = [
            token for token in tokens if token.sentence == direction.sentence
        ]
        fan_positions = [
            token.position for token in same_sentence if token.canonical == "fan"
        ]
        speed_positions = [
            token.position for token in same_sentence if token.canonical == "speed"
        ]
        if not fan_positions or not speed_positions:
            continue
        if min(abs(position - direction.position) for position in fan_positions) > 8:
            continue
        if min(abs(position - direction.position) for position in speed_positions) > 10:
            continue

        for amount in range(1, 6):
            for mention in _numeric_mentions(amount, tokens):
                if (
                    mention.start <= direction.position
                    or tokens[mention.start].sentence != direction.sentence
                    or mention.start - direction.position > 16
                ):
                    continue
                between = {
                    token.canonical
                    for token in tokens
                    if direction.position < token.position < mention.start
                }
                unit_nearby = any(
                    token.canonical in {"level", "notch"}
                    and mention.start - 3 <= token.position <= mention.end + 2
                    for token in same_sentence
                )
                if "by" not in between and not unit_nearby:
                    continue
                if "to" in between and "by" not in between:
                    continue
                candidates.add(sign * amount)

        for index, token in enumerate(same_sentence[:-1]):
            next_token = same_sentence[index + 1]
            if (
                token.canonical == "a"
                and next_token.canonical in {"level", "notch"}
                and direction.position < token.position <= direction.position + 16
            ):
                candidates.add(sign)

    return next(iter(candidates)) if len(candidates) == 1 else None


def _recover_relative_fan_speed_goal(
    raw_user_text: str,
    intent: IntentFrame,
    tokens: tuple[_Token, ...],
) -> tuple[IntentFrame, dict[str, int]]:
    delta = _relative_fan_speed_delta(tokens)
    if delta is None:
        return intent, {}

    fan_goals = [
        goal for goal in intent.goals if goal.semantic_operation == "set_fan_speed"
    ]
    if fan_goals:
        primary = fan_goals[0].model_copy(update={"desired_outcome": {}}, deep=True)
    else:
        source_turn = (
            intent.intent_source_turn_ids[-1]
            if intent.intent_source_turn_ids
            else "current-user-turn"
        )
        digest = hashlib.sha256(
            f"{source_turn}\0relative-fan-speed\0{raw_user_text}".encode()
        ).hexdigest()
        primary = Goal(
            goal_id=f"goal-relative-fan-{digest[:12]}",
            semantic_operation="set_fan_speed",
            desired_outcome={},
            source=GoalSource.USER,
        )

    duplicate_ids = {goal.goal_id for goal in fan_goals[1:]}
    replacement_ids = {goal_id: primary.goal_id for goal_id in duplicate_ids}
    recovered_goals: list[Goal] = []
    inserted = False
    for goal in intent.goals:
        if goal.goal_id in duplicate_ids:
            continue
        if goal.semantic_operation == "set_fan_speed":
            if inserted:
                continue
            updated = primary
            inserted = True
        else:
            updated = goal.model_copy(deep=True)
        dependencies = [
            replacement_ids.get(dependency, dependency)
            for dependency in updated.depends_on
        ]
        updated = updated.model_copy(
            update={"depends_on": list(dict.fromkeys(dependencies))}, deep=True
        )
        recovered_goals.append(updated)
    if not inserted:
        recovered_goals.append(primary)

    retained_ids = {goal.goal_id for goal in recovered_goals}
    mention_order = [
        replacement_ids.get(goal_id, goal_id)
        for goal_id in intent.goal_mention_order
        if replacement_ids.get(goal_id, goal_id) in retained_ids
    ]
    mention_order = list(dict.fromkeys(mention_order))
    if primary.goal_id not in mention_order:
        mention_order.append(primary.goal_id)
    for goal in recovered_goals:
        if goal.goal_id not in mention_order:
            mention_order.append(goal.goal_id)

    recovered = IntentFrame.model_validate(
        {
            **intent.model_dump(mode="python"),
            "call_for_action": True,
            "goals": [goal.model_dump(mode="python") for goal in recovered_goals],
            "explicit_slots": {
                key: value
                for key, value in intent.explicit_slots.items()
                if key != "level"
            },
            "unresolved_slots": [
                slot.model_dump(mode="python")
                for slot in intent.unresolved_slots
                if slot.goal_id in retained_ids
                and not (
                    slot.goal_id == primary.goal_id
                    and canonical_semantic_parameter_name("set_fan_speed", slot.name)
                    == "level"
                )
            ],
            "goal_mention_order": mention_order,
            "intent_kind": IntentKind.ACTION,
        }
    )
    return recovered, {primary.goal_id: delta}


def _relative_climate_temperature_delta(tokens: tuple[_Token, ...]) -> int | None:
    """Return one explicit bounded cabin-setpoint delta, never an absolute value."""

    candidates: set[int] = set()
    safe_maybe_positions = _polite_relative_temperature_maybe_positions(tokens)
    safe_maybe_sentences = {
        tokens[position].sentence for position in safe_maybe_positions
    }
    hypothetical_sentences = {
        token.sentence for token in tokens if token.canonical in _HYPOTHETICAL_MARKERS
    }.difference(safe_maybe_sentences)
    direction_signs = {
        "decrease": -1,
        "drop": -1,
        "increase": 1,
        "lower": -1,
        "raise": 1,
        "reduce": -1,
    }
    request_verbs = frozenset({"adjust", "change", "set", *direction_signs})
    for direction in tokens:
        sign = direction_signs.get(direction.canonical)
        if sign is None or direction.sentence in hypothetical_sentences:
            continue
        same_sentence = tuple(
            token for token in tokens if token.sentence == direction.sentence
        )
        temperature_positions = [
            token.position
            for token in same_sentence
            if token.canonical == "temperature"
        ]
        if (
            not temperature_positions
            or min(
                abs(position - direction.position) for position in temperature_positions
            )
            > 10
        ):
            continue
        clause_prefix = tuple(
            token
            for token in tokens
            if token.clause == direction.clause and token.position < direction.position
        )
        if {token.canonical for token in clause_prefix[-5:]}.intersection(
            {"don't", "dont", "never", "not", "without"}
        ):
            continue
        explicit_requests = [
            token
            for token in same_sentence
            if token.clause == direction.clause
            and token.canonical in request_verbs
            and token.position <= direction.position
            and (
                _has_request_prefix(tokens, token.position)
                or token.position in safe_maybe_positions
            )
        ]
        if not explicit_requests:
            continue

        for amount in range(1, 11):
            for mention in _numeric_mentions(amount, tokens):
                if (
                    tokens[mention.start].sentence != direction.sentence
                    or abs(mention.start - direction.position) > 16
                ):
                    continue
                has_degree_unit = any(
                    token.canonical == "degree"
                    and mention.start - 1 <= token.position <= mention.end + 2
                    for token in same_sentence
                )
                if not has_degree_unit:
                    continue
                if mention.start > direction.position:
                    between = {
                        token.canonical
                        for token in same_sentence
                        if direction.position < token.position < mention.start
                    }
                    if "to" in between and "by" not in between:
                        continue
                    if "by" not in between:
                        continue
                else:
                    after_direction = {
                        token.canonical
                        for token in same_sentence
                        if direction.position < token.position <= direction.position + 8
                    }
                    if (
                        "than" not in after_direction
                        or not after_direction.intersection(
                            {"current", "currently", "now"}
                        )
                    ):
                        continue
                    if not any(
                        request.position < mention.start
                        for request in explicit_requests
                        if request.canonical in {"adjust", "change", "set"}
                    ):
                        continue
                candidates.add(sign * amount)

    return next(iter(candidates)) if len(candidates) == 1 else None


def _recover_relative_climate_temperature_goal(
    raw_user_text: str,
    intent: IntentFrame,
    tokens: tuple[_Token, ...],
) -> tuple[IntentFrame, dict[str, int]]:
    if sum(token.canonical == "temperature" for token in tokens) != 1:
        return intent, {}
    delta = _relative_climate_temperature_delta(tokens)
    if delta is None:
        return intent, {}

    temperature_goals = [
        goal
        for goal in intent.goals
        if goal.semantic_operation == "set_climate_temperature"
    ]
    if len(temperature_goals) > 1:
        return intent, {}
    if temperature_goals:
        primary = temperature_goals[0].model_copy(
            update={
                "desired_outcome": {
                    key: value
                    for key, value in temperature_goals[0].desired_outcome.items()
                    if canonical_semantic_parameter_name("set_climate_temperature", key)
                    != "temperature"
                }
            },
            deep=True,
        )
    else:
        source_turn = (
            intent.intent_source_turn_ids[-1]
            if intent.intent_source_turn_ids
            else "current-user-turn"
        )
        digest = hashlib.sha256(
            f"{source_turn}\0relative-climate-temperature\0{raw_user_text}".encode()
        ).hexdigest()
        primary = Goal(
            goal_id=f"goal-relative-temperature-{digest[:12]}",
            semantic_operation="set_climate_temperature",
            desired_outcome={},
            source=GoalSource.USER,
        )

    duplicate_ids = {goal.goal_id for goal in temperature_goals[1:]}
    replacement_ids = {goal_id: primary.goal_id for goal_id in duplicate_ids}
    recovered_goals: list[Goal] = []
    inserted = False
    for goal in intent.goals:
        if goal.goal_id in duplicate_ids:
            continue
        if goal.semantic_operation == "set_climate_temperature":
            if inserted:
                continue
            updated = primary
            inserted = True
        else:
            updated = goal.model_copy(deep=True)
        updated = updated.model_copy(
            update={
                "depends_on": list(
                    dict.fromkeys(
                        replacement_ids.get(dependency, dependency)
                        for dependency in updated.depends_on
                    )
                )
            },
            deep=True,
        )
        recovered_goals.append(updated)
    if not inserted:
        recovered_goals.append(primary)

    retained_ids = {goal.goal_id for goal in recovered_goals}
    mention_order = list(
        dict.fromkeys(
            replacement_ids.get(goal_id, goal_id)
            for goal_id in intent.goal_mention_order
            if replacement_ids.get(goal_id, goal_id) in retained_ids
        )
    )
    if primary.goal_id not in mention_order:
        mention_order.append(primary.goal_id)
    for goal in recovered_goals:
        if goal.goal_id not in mention_order:
            mention_order.append(goal.goal_id)

    recovered = IntentFrame.model_validate(
        {
            **intent.model_dump(mode="python"),
            "call_for_action": True,
            "goals": [goal.model_dump(mode="python") for goal in recovered_goals],
            "explicit_slots": {
                key: value
                for key, value in intent.explicit_slots.items()
                if canonical_semantic_parameter_name("set_climate_temperature", key)
                != "temperature"
            },
            "unresolved_slots": [
                slot.model_dump(mode="python")
                for slot in intent.unresolved_slots
                if slot.goal_id in retained_ids
                and not (
                    slot.goal_id == primary.goal_id
                    and canonical_semantic_parameter_name(
                        "set_climate_temperature", slot.name
                    )
                    == "temperature"
                )
            ],
            "goal_mention_order": mention_order,
            "intent_kind": IntentKind.ACTION,
        }
    )
    return recovered, {primary.goal_id: delta}


def _strict_named_destination_replacement_candidate(
    goal: Goal,
    registry: RecipeRegistry,
    tokens: tuple[_Token, ...],
    location: str | None,
) -> _GoalCandidate | None:
    """Select the replacement recipe for an exact recovered destination name."""

    if (
        location is None
        or goal.source is not GoalSource.USER
        or goal.semantic_operation
        != _NAMED_NAVIGATION_DESTINATION_REPLACEMENT_OPERATION
        or goal.desired_outcome != {"new_destination_name": location}
    ):
        return None
    anchors = tuple(
        token.position
        for token in tokens
        if token.canonical in {"change", "replace", "set", "switch"}
    )
    if not anchors:
        return None
    for recipe in registry:
        if not recipe.matches_semantic_operation(goal.semantic_operation):
            continue
        target = recipe.target_operation(goal.semantic_operation)
        if (
            target is None
            or target.semantic_operation
            != _NAMED_NAVIGATION_DESTINATION_REPLACEMENT_OPERATION
        ):
            continue
        if target not in recipe.write_operations:
            return None
        return _GoalCandidate(
            goal=goal,
            recipe=recipe,
            target=target,
            is_action=True,
            anchor_positions=anchors,
        )
    return None


def _strict_named_final_destination_delete_candidate(
    goal: Goal,
    registry: RecipeRegistry,
    tokens: tuple[_Token, ...],
    names: tuple[str, str | None] | None,
) -> _GoalCandidate | None:
    """Select the delete recipe only for an exact recovered named goal."""

    if names is None:
        return None
    deleted, remaining = names
    expected = {"destination_name_to_delete": deleted}
    if remaining is not None:
        expected["remaining_destination_name"] = remaining
    if (
        goal.source is not GoalSource.USER
        or goal.semantic_operation
        != _NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_OPERATION
        or goal.desired_outcome != expected
    ):
        return None
    anchors = tuple(
        token.position for token in tokens if token.canonical in {"cancel", "remove"}
    )
    if len(anchors) != 1:
        return None
    for recipe in registry:
        if not recipe.matches_semantic_operation(goal.semantic_operation):
            continue
        target = recipe.target_operation(goal.semantic_operation)
        if (
            target is None
            or target.semantic_operation
            != _NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_OPERATION
        ):
            continue
        write_operations = {
            operation.semantic_operation for operation in recipe.write_operations
        }
        if target.semantic_operation not in write_operations:
            return None
        return _GoalCandidate(
            goal=goal,
            recipe=recipe,
            target=target,
            is_action=True,
            anchor_positions=anchors,
        )
    return None


def _strict_waypoint_context_candidate(
    goal: Goal,
    registry: RecipeRegistry,
    tokens: tuple[_Token, ...],
    *,
    named_add: _NamedWaypointAdd | None,
    named_replacement: _NamedWaypointReplacement | None,
    next_stop_poi_position: int | None,
    route_choice_alias: str | None,
    vague_add_is_explicit: bool,
) -> _GoalCandidate | None:
    if goal.source is not GoalSource.USER:
        return None
    recipe_operation: str
    expected_target: str
    is_action: bool
    anchor_words: frozenset[str]
    if goal.semantic_operation == _NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION:
        expected = (
            named_add.desired_outcome(route_choice_alias)
            if named_add is not None
            else None
        )
        if expected is None and not vague_add_is_explicit:
            return None
        if expected is not None and goal.desired_outcome != expected:
            return None
        recipe_operation = _NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION
        expected_target = _NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION
        is_action = True
        anchor_words = frozenset({"add", "stop", "waypoint"})
    elif goal.semantic_operation == _NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION:
        if named_replacement is None or goal.desired_outcome != (
            named_replacement.desired_outcome(route_choice_alias)
        ):
            return None
        recipe_operation = _NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION
        expected_target = _NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION
        is_action = True
        anchor_words = frozenset({"replace", "stop", "waypoint"})
    elif goal.semantic_operation == _NEXT_NAVIGATION_STOP_POI_SEARCH_OPERATION:
        if next_stop_poi_position is None or goal.desired_outcome != {
            "category": "restaurants"
        }:
            return None
        recipe_operation = _CURRENT_DESTINATION_POI_SEARCH_OPERATION
        expected_target = _CURRENT_DESTINATION_POI_SEARCH_OPERATION
        is_action = False
        anchor_words = frozenset({"find", "restaurant", "search", "show", "stop"})
    else:
        return None

    anchors = tuple(
        token.position for token in tokens if token.canonical in anchor_words
    )
    if not anchors:
        return None
    for recipe in registry:
        if not recipe.matches_semantic_operation(recipe_operation):
            continue
        target = recipe.target_operation(recipe_operation)
        if target is None or target.semantic_operation != expected_target:
            continue
        if is_action != (target in recipe.write_operations):
            continue
        return _GoalCandidate(
            goal=goal,
            recipe=recipe,
            target=target,
            is_action=is_action,
            anchor_positions=anchors,
        )
    return None


def _strict_named_waypoint_delete_candidate(
    goal: Goal,
    registry: RecipeRegistry,
    tokens: tuple[_Token, ...],
    names: tuple[str, str | None, str | None, str | None] | None,
) -> _GoalCandidate | None:
    if names is None:
        return None
    waypoint, destination, route_choice, route_destination = names
    expected = {"waypoint_name_to_delete": waypoint}
    if destination is not None:
        expected["remaining_destination_name"] = destination
    if route_choice is not None:
        expected["route_choice_alias"] = route_choice
    if route_destination is not None:
        expected["route_destination_name"] = route_destination
    if (
        goal.source is not GoalSource.USER
        or goal.semantic_operation != _NAMED_NAVIGATION_WAYPOINT_DELETE_OPERATION
        or goal.desired_outcome != expected
    ):
        return None
    anchors = tuple(
        token.position for token in tokens if token.canonical in {"remove", "take"}
    )
    if not anchors:
        return None
    for recipe in registry:
        if not recipe.matches_semantic_operation(goal.semantic_operation):
            continue
        target = recipe.target_operation(goal.semantic_operation)
        if (
            target is None
            or target.semantic_operation != _NAMED_NAVIGATION_WAYPOINT_DELETE_OPERATION
        ):
            continue
        if target not in recipe.write_operations:
            return None
        return _GoalCandidate(
            goal=goal,
            recipe=recipe,
            target=target,
            is_action=True,
            anchor_positions=anchors,
        )
    return None


def _strict_relative_occupied_seat_heating_candidate(
    goal: Goal,
    registry: RecipeRegistry,
    tokens: tuple[_Token, ...],
    relative_deltas: dict[str, int],
    occupied_goal_ids: frozenset[str],
) -> _GoalCandidate | None:
    if (
        goal.goal_id not in relative_deltas
        or goal.goal_id not in occupied_goal_ids
        or goal.source is not GoalSource.USER
        or goal.semantic_operation != "set_seat_heating"
        or goal.desired_outcome
    ):
        return None
    anchors = tuple(
        token.position
        for token in tokens
        if token.canonical == "increase"
        and any(
            other.sentence == token.sentence
            and other.canonical == "seat"
            and abs(other.position - token.position) <= 5
            for other in tokens
        )
    )
    if len(anchors) != 1:
        return None
    for recipe in registry:
        if not recipe.matches_semantic_operation(goal.semantic_operation):
            continue
        target = recipe.target_operation(goal.semantic_operation)
        if target is None or target.semantic_operation != "set_seat_heating":
            continue
        if target not in recipe.write_operations:
            return None
        return _GoalCandidate(
            goal=goal,
            recipe=recipe,
            target=target,
            is_action=True,
            anchor_positions=anchors,
        )
    return None


def _strict_driver_warming_candidate(
    goal: Goal,
    registry: RecipeRegistry,
    tokens: tuple[_Token, ...],
    values_by_goal: dict[str, dict[str, Any]],
) -> _GoalCandidate | None:
    if goal.goal_id not in values_by_goal:
        return None
    anchor_words = {
        "set_climate_temperature": {"temperature"},
        "set_seat_heating": {"seat"},
        "set_steering_wheel_heating": {"steering"},
    }.get(goal.semantic_operation)
    if anchor_words is None:
        return None
    anchors = tuple(
        token.position for token in tokens if token.canonical in anchor_words
    )
    if goal.semantic_operation == "set_seat_heating":
        anchors = tuple(
            position
            for position in anchors
            if any(
                other.canonical == "heat"
                and other.sentence == tokens[position].sentence
                and 0 < other.position - position <= 2
                for other in tokens
            )
        )
    if len(anchors) != 1:
        return None
    for recipe in registry:
        if not recipe.matches_semantic_operation(goal.semantic_operation):
            continue
        target = recipe.target_operation(goal.semantic_operation)
        if target is None or target not in recipe.write_operations:
            continue
        return _GoalCandidate(
            goal=goal,
            recipe=recipe,
            target=target,
            is_action=True,
            anchor_positions=anchors,
        )
    return None


def ground_intent(
    raw_user_text: str,
    intent: IntentFrame,
    registry: RecipeRegistry,
) -> IntentGroundingResult:
    """Ground an extracted frame against one raw user utterance.

    The function is deterministic and inventory-blind.  A state-changing goal
    survives only when the utterance contains an explicit request grammar, a
    matching invariant semantic anchor, and every proposed desired value is
    bound to that goal in the same utterance clause.
    """

    grounding_text = _without_quoted_segments(raw_user_text)
    tokens = _tokenize(grounding_text)
    intent, relative_fan_deltas = _recover_relative_fan_speed_goal(
        grounding_text, intent, tokens
    )
    intent, relative_temperature_deltas = _recover_relative_climate_temperature_goal(
        grounding_text, intent, tokens
    )
    (
        intent,
        relative_seat_heating_deltas,
        occupied_seat_heating_goal_ids,
    ) = _recover_relative_occupied_seat_heating_goal(
        grounding_text,
        intent,
        tokens,
    )
    action_requests = _explicit_request_positions(grounding_text, tokens, _ACTION_VERBS)
    read_clauses = _read_request_clauses(tokens)
    explicit_soc_range = _explicit_soc_range(tokens)
    named_destination_replacement_name = _named_navigation_destination_replacement_name(
        raw_user_text
    )
    named_final_destination_delete_names = (
        _named_navigation_final_destination_delete_names(raw_user_text)
    )
    named_waypoint_delete_names = _named_navigation_waypoint_delete_names(raw_user_text)
    named_waypoint_add = _named_navigation_waypoint_add(raw_user_text)
    named_waypoint_replacement = _named_navigation_waypoint_replacement(raw_user_text)
    next_stop_poi_position = _next_navigation_stop_poi_query_position(raw_user_text)
    route_choice_alias = _explicit_navigation_route_choice(raw_user_text)
    conflicting_route_choice = _has_conflicting_navigation_route_choices(
        raw_user_text
    )
    vague_waypoint_add_is_explicit = _vague_navigation_waypoint_add_is_explicit(
        raw_user_text
    )
    has_named_location_alternatives = _has_named_location_alternatives(
        _without_quoted_segments(raw_user_text)
    )
    driver_warming_kind = _driver_warming_request_kind(raw_user_text)
    driver_warming_values_by_goal = _strict_driver_warming_values_by_goal(
        intent, driver_warming_kind
    )

    candidates: dict[str, _GoalCandidate] = {}
    for goal in intent.goals:
        if goal.source is not GoalSource.USER:
            continue
        candidate = _select_candidate(
            goal,
            registry,
            tokens,
            action_requests,
            read_clauses,
        )
        if candidate is None:
            candidate = _strict_named_destination_replacement_candidate(
                goal,
                registry,
                tokens,
                named_destination_replacement_name,
            )
        if candidate is None:
            candidate = _strict_waypoint_context_candidate(
                goal,
                registry,
                tokens,
                named_add=named_waypoint_add,
                named_replacement=named_waypoint_replacement,
                next_stop_poi_position=next_stop_poi_position,
                route_choice_alias=route_choice_alias,
                vague_add_is_explicit=vague_waypoint_add_is_explicit,
            )
        if candidate is None:
            candidate = _strict_named_final_destination_delete_candidate(
                goal,
                registry,
                tokens,
                named_final_destination_delete_names,
            )
        if candidate is None:
            candidate = _strict_named_waypoint_delete_candidate(
                goal,
                registry,
                tokens,
                named_waypoint_delete_names,
            )
        if candidate is None:
            candidate = _strict_relative_occupied_seat_heating_candidate(
                goal,
                registry,
                tokens,
                relative_seat_heating_deltas,
                occupied_seat_heating_goal_ids,
            )
        if candidate is None:
            candidate = _strict_driver_warming_candidate(
                goal,
                registry,
                tokens,
                driver_warming_values_by_goal,
            )
        if (
            candidate is not None
            and conflicting_route_choice
            and candidate.target.semantic_operation
            in {
                _NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION,
                _NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION,
            }
        ):
            candidate = None
        if candidate is not None:
            candidates[goal.goal_id] = candidate

    if explicit_soc_range is not None:
        range_candidates = tuple(
            candidate
            for candidate in candidates.values()
            if candidate.target.semantic_operation == "get_distance_by_soc"
        )
        range_goal_ids = {candidate.goal.goal_id for candidate in range_candidates}
        range_clauses = {
            tokens[position].clause
            for candidate in range_candidates
            for position in candidate.anchor_positions
        }
        redundant_status_ids = {
            dependency_id
            for candidate in range_candidates
            for dependency_id in candidate.goal.depends_on
        }
        for status_id in redundant_status_ids:
            status = candidates.get(status_id)
            if (
                status is None
                or status.target.semantic_operation != "read_charging_status"
            ):
                continue
            dependent_goal_ids = {
                candidate.goal.goal_id
                for candidate in candidates.values()
                if status_id in candidate.goal.depends_on
            }
            status_clauses = {
                tokens[position].clause for position in status.anchor_positions
            }
            explicitly_separate = bool(status_clauses.difference(range_clauses)) or any(
                token.canonical == "status" and token.clause in status_clauses
                for token in tokens
            )
            if dependent_goal_ids.issubset(range_goal_ids) and not explicitly_separate:
                candidates.pop(status_id)

    grounded_values: dict[str, dict[str, Any]] = {}
    derived_values: dict[str, dict[str, str]] = {}
    for goal_id, candidate in tuple(candidates.items()):
        if (
            has_named_location_alternatives
            and candidate.target.semantic_operation
            in {
                _NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION,
                _NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION,
            }
            and not any(slot.goal_id == goal_id for slot in intent.unresolved_slots)
        ):
            candidates.pop(goal_id)
            continue
        desired = _navigation_name_selector_values(
            candidate, _canonical_desired_values(candidate)
        )
        if _navigation_destination_has_explicit_alternatives(
            grounding_text, candidate, desired
        ):
            candidates.pop(goal_id)
            continue
        desired = _poi_name_selector_values(candidate, desired)
        exact_named_destination_replacement = (
            named_destination_replacement_name is not None
            and candidate.target.semantic_operation
            == _NAMED_NAVIGATION_DESTINATION_REPLACEMENT_OPERATION
            and desired == {"new_destination_name": named_destination_replacement_name}
        )
        exact_named_final_delete = (
            named_final_destination_delete_names is not None
            and candidate.target.semantic_operation
            == _NAMED_NAVIGATION_FINAL_DESTINATION_DELETE_OPERATION
            and desired
            == {
                "destination_name_to_delete": named_final_destination_delete_names[0],
                "remaining_destination_name": named_final_destination_delete_names[1],
            }
        )
        exact_named_waypoint_delete = False
        if (
            named_waypoint_delete_names is not None
            and candidate.target.semantic_operation
            == _NAMED_NAVIGATION_WAYPOINT_DELETE_OPERATION
        ):
            waypoint, destination, selected_route, route_destination = (
                named_waypoint_delete_names
            )
            expected_waypoint_delete = {"waypoint_name_to_delete": waypoint}
            if destination is not None:
                expected_waypoint_delete["remaining_destination_name"] = destination
            if selected_route is not None:
                expected_waypoint_delete["route_choice_alias"] = selected_route
            if route_destination is not None:
                expected_waypoint_delete["route_destination_name"] = route_destination
            exact_named_waypoint_delete = desired == expected_waypoint_delete
        exact_named_waypoint_context = bool(
            (
                named_waypoint_add is not None
                and candidate.goal.semantic_operation
                == _NAMED_NAVIGATION_WAYPOINT_ADD_OPERATION
                and desired == named_waypoint_add.desired_outcome(route_choice_alias)
            )
            or (
                named_waypoint_replacement is not None
                and candidate.goal.semantic_operation
                == _NAMED_NAVIGATION_WAYPOINT_REPLACEMENT_OPERATION
                and desired
                == named_waypoint_replacement.desired_outcome(route_choice_alias)
            )
            or (
                next_stop_poi_position is not None
                and candidate.goal.semantic_operation
                == _NEXT_NAVIGATION_STOP_POI_SEARCH_OPERATION
                and desired == {"category": "restaurants"}
            )
        )
        exact_driver_warming = (
            goal_id in driver_warming_values_by_goal
            and desired == driver_warming_values_by_goal[goal_id]
        )
        grounded = (
            dict(desired)
            if exact_named_destination_replacement
            or exact_named_final_delete
            or exact_named_waypoint_delete
            or exact_named_waypoint_context
            or exact_driver_warming
            else {
                key: value
                for key, value in desired.items()
                if key not in {"remaining_destination_name", "route_destination_name"}
                and (
                    _value_is_grounded(key, value, candidate, candidates, tokens)
                    or _battery_trip_destination_value_is_bound(
                        key, value, candidate, intent, tokens
                    )
                )
            }
        )
        if (
            exact_driver_warming
            and driver_warming_kind == _DRIVER_WARMING_MATCH_LEVEL
            and candidate.target.semantic_operation == "set_steering_wheel_heating"
        ):
            derived_values.setdefault(goal_id, {})["level"] = (
                "steering_level_matches_seat_heating_v1"
            )
        if _all_seat_zones_are_bound(candidate, candidates, tokens):
            grounded["seat_zone"] = "ALL_ZONES"
        explicit_defrost_window = _explicit_defrost_window(
            candidate, candidates, tokens
        )
        if explicit_defrost_window is not None:
            grounded["window"] = explicit_defrost_window
        if (
            explicit_soc_range is not None
            and candidate.target.semantic_operation == "get_distance_by_soc"
        ):
            grounded["initial_soc"], grounded["final_soc"] = explicit_soc_range
        if (
            candidate.target.semantic_operation == "set_ambient_lights"
            and "color" in grounded
            and "enabled" not in grounded
        ):
            explicit_false = _boolean_polarity_is_bound(
                False, candidate, candidates, tokens
            )
            explicit_true = _boolean_polarity_is_bound(
                True, candidate, candidates, tokens
            )
            if explicit_false != explicit_true:
                grounded["enabled"] = not explicit_false
            elif not explicit_false:
                grounded["enabled"] = True
                derived_values.setdefault(goal_id, {})["enabled"] = (
                    "ambient_color_selection_enables_lights_v1"
                )
        if (
            candidate.is_action
            and candidate.target.required_semantic_parameters
            and desired
            and not grounded
            and not _navigation_action_has_derived_parameters(candidate)
        ):
            candidates.pop(goal_id)
            continue
        grounded_values[goal_id] = grounded

    retained_ids = set(candidates)
    goals_by_id = {goal.goal_id: goal for goal in intent.goals}
    hypothetical_sentences = frozenset(
        token.sentence for token in tokens if token.canonical in _HYPOTHETICAL_MARKERS
    )
    dependency_overrides: dict[str, list[str]] = {}
    if explicit_soc_range is not None:
        for goal_id, candidate in candidates.items():
            if candidate.target.semantic_operation != "get_distance_by_soc":
                continue
            dependency_overrides[goal_id] = [
                dependency
                for dependency in candidate.goal.depends_on
                if dependency not in goals_by_id
                or goals_by_id[dependency].semantic_operation != "read_charging_status"
            ]
    changed = True
    while changed:
        changed = False
        for goal in intent.goals:
            if goal.goal_id not in retained_ids:
                continue
            dependencies = dependency_overrides.get(goal.goal_id, goal.depends_on)
            missing_dependencies = [
                dependency
                for dependency in dependencies
                if dependency not in retained_ids
            ]
            if missing_dependencies and all(
                dependency in goals_by_id
                and _recipe_manages_hypothetical_dependency(
                    goal,
                    goals_by_id[dependency],
                    registry,
                    tokens,
                    hypothetical_sentences,
                )
                for dependency in missing_dependencies
            ):
                dependency_overrides[goal.goal_id] = [
                    dependency
                    for dependency in dependencies
                    if dependency in retained_ids
                ]
                continue
            if missing_dependencies:
                retained_ids.remove(goal.goal_id)
                candidates.pop(goal.goal_id, None)
                grounded_values.pop(goal.goal_id, None)
                derived_values.pop(goal.goal_id, None)
                changed = True

    goals_with_slots = {
        slot.goal_id for slot in intent.unresolved_slots if slot.goal_id is not None
    }
    duplicate_goal_ids: dict[str, str] = {}
    unique_grounded_goals: dict[tuple[str, str, tuple[str, ...], str | None], str] = {}
    for goal in intent.goals:
        if goal.goal_id not in retained_ids:
            continue
        grounded_goal_values = grounded_values[goal.goal_id]
        target_operation = candidates[goal.goal_id].target.semantic_operation
        aggregate_seat_goal = (
            target_operation in {"set_climate_temperature", "set_seat_heating"}
            and grounded_goal_values.get("seat_zone") == "ALL_ZONES"
        )
        explicit_range_goal = (
            explicit_soc_range is not None and target_operation == "get_distance_by_soc"
        )
        if not aggregate_seat_goal and not explicit_range_goal:
            continue
        fingerprint = (
            target_operation,
            json.dumps(
                grounded_goal_values,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=str,
            ),
            tuple(dependency_overrides.get(goal.goal_id, goal.depends_on)),
            goal.atomic_group,
        )
        canonical_goal_id = unique_grounded_goals.get(fingerprint)
        if (
            canonical_goal_id is None
            or goal.goal_id in goals_with_slots
            or canonical_goal_id in goals_with_slots
        ):
            unique_grounded_goals[fingerprint] = goal.goal_id
            continue
        duplicate_goal_ids[goal.goal_id] = canonical_goal_id
        retained_ids.remove(goal.goal_id)
        candidates.pop(goal.goal_id, None)
        grounded_values.pop(goal.goal_id, None)
        derived_values.pop(goal.goal_id, None)

    retained_goals = [
        goal.model_copy(
            update={
                "desired_outcome": grounded_values[goal.goal_id],
                "depends_on": list(
                    dict.fromkeys(
                        duplicate_goal_ids.get(dependency, dependency)
                        for dependency in dependency_overrides.get(
                            goal.goal_id, goal.depends_on
                        )
                        if duplicate_goal_ids.get(dependency, dependency)
                        != goal.goal_id
                    )
                ),
            },
            deep=True,
        )
        for goal in intent.goals
        if goal.goal_id in retained_ids
    ]
    authorized = frozenset(
        goal_id
        for goal_id, candidate in candidates.items()
        if candidate.is_action and goal_id in retained_ids
    )

    explicit_slots = {
        key: value
        for key, value in intent.explicit_slots.items()
        if any(
            key in values and _values_equal(value, values[key])
            for values in grounded_values.values()
        )
    }
    explicit_constraints = {
        key: value
        for key, value in intent.explicit_constraints.items()
        if _ground_global_value(key, value, tokens)
    }
    unresolved_slots = [
        slot.model_copy(deep=True)
        for slot in intent.unresolved_slots
        if slot.goal_id is not None and slot.goal_id in retained_ids
    ]
    has_read = any(
        not candidate.is_action and goal_id in retained_ids
        for goal_id, candidate in candidates.items()
    )
    filtered = IntentFrame.model_validate(
        {
            **intent.model_dump(),
            "call_for_action": bool(authorized),
            "goals": [goal.model_dump() for goal in retained_goals],
            "explicit_slots": explicit_slots,
            "explicit_constraints": explicit_constraints,
            "unresolved_slots": [slot.model_dump() for slot in unresolved_slots],
            "goal_mention_order": [
                goal_id
                for goal_id in intent.goal_mention_order
                if goal_id in retained_ids
            ],
            "intent_kind": _fallback_kind(
                intent.intent_kind,
                tokens,
                bool(authorized),
                has_read,
            ),
        }
    )
    return IntentGroundingResult(
        filtered_intent=filtered,
        authorized_action_goal_ids=authorized,
        desired_values_by_goal={
            goal_id: grounded_values[goal_id] for goal_id in filtered.goal_mention_order
        },
        derived_values_by_goal={
            goal_id: derived_values[goal_id]
            for goal_id in filtered.goal_mention_order
            if goal_id in derived_values
        },
        relative_fan_speed_deltas_by_goal={
            goal_id: delta
            for goal_id, delta in relative_fan_deltas.items()
            if goal_id in retained_ids
        },
        relative_climate_temperature_deltas_by_goal={
            goal_id: delta
            for goal_id, delta in relative_temperature_deltas.items()
            if goal_id in retained_ids
        },
        relative_seat_heating_deltas_by_goal={
            goal_id: delta
            for goal_id, delta in relative_seat_heating_deltas.items()
            if goal_id in retained_ids
        },
        occupied_seat_heating_goal_ids=frozenset(
            goal_id
            for goal_id in occupied_seat_heating_goal_ids
            if goal_id in retained_ids
        ),
    )


__all__ = [
    "IntentGroundingResult",
    "canonical_semantic_parameter_name",
    "focus_explicit_action_request",
    "ground_intent",
    "has_explicit_action_request",
    "recover_battery_charge_trip_range_intent",
    "recover_climate_zone_sync_intent",
    "recover_current_destination_poi_search_intent",
    "recover_current_day_calendar_intent",
    "recover_driver_warming_intent",
    "recover_named_navigation_final_destination_delete_intent",
    "recover_named_navigation_destination_replacement_intent",
    "recover_named_navigation_waypoint_delete_intent",
    "recover_named_location_poi_search_intent",
    "recover_navigation_resume_intent",
    "recover_navigation_waypoint_context_intent",
    "recover_relative_occupied_seat_heating_intent",
    "semantic_value_is_explicit",
]
