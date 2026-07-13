"""Deterministic policy rule contracts and evaluation.

The rule engine deliberately operates on semantic operations instead of a
catalog of benchmark tools. A later live-capability binder is responsible for
mapping these requirements to tools that are actually available in the current
turn. This keeps policy compilation independent from evaluator internals and
from any canonical or full tool inventory.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, Final

from pydantic import Field, model_validator

from ..domain import DomainModel, OfficialToolCall


NUMBERED_POLICY_IDS: Final[tuple[str, ...]] = (
    "POL-002",
    "POL-004",
    "POL-005",
    "POL-007",
    "POL-008",
    "POL-009",
    "POL-010",
    "POL-011",
    "POL-012",
    "POL-013",
    "POL-014",
    "POL-016",
    "POL-017",
    "POL-018",
    "POL-019",
    "POL-021",
    "POL-022",
    "POL-023",
    "POL-024",
)

GENERAL_TTS = "GEN-TTS"
GENERAL_REACTIVE_ONLY = "GEN-REACTIVE-ONLY"
GENERAL_LIVE_SCHEMA = "GEN-LIVE-SCHEMA"
GENERAL_TOOL_DESCRIPTION = "GEN-TOOL-DESCRIPTION"
GENERAL_ROUTE_PRESENTATION = "GEN-ROUTE-PRESENTATION"
GENERAL_POI_PRESENTATION = "GEN-POI-PRESENTATION"


class PolicyRuleSource(str, Enum):
    NUMBERED_LABEL = "numbered_label"
    SEMANTIC_TEXT = "semantic_text"
    RUNTIME_INVARIANT = "runtime_invariant"


class PolicyRuleCategory(str, Enum):
    GENERAL = "general"
    FORMAT = "format"
    CONFIRMATION = "confirmation"
    VEHICLE = "vehicle"
    NAVIGATION = "navigation"
    INFORMATION = "information"


class ActionAuthorization(str, Enum):
    NONE = "none"
    USER_REQUEST = "user_request"
    USER_CONFIRMATION = "user_confirmation"
    USER_SELECTION = "user_selection"


class RequirementOrder(str, Enum):
    BEFORE_ACTION = "before_action"
    AFTER_ACTION = "after_action"


class ExecutionMode(str, Enum):
    DEFAULT = "default"
    SERIAL = "serial"


class CompiledRule(DomainModel):
    rule_id: str
    category: PolicyRuleCategory
    source: PolicyRuleSource
    source_text: str


class SemanticPolicyCall(DomainModel):
    """A policy-required operation, not a claim that a live tool exists."""

    semantic_operation: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rule_ids: list[str] = Field(default_factory=list)
    order: RequirementOrder = RequirementOrder.BEFORE_ACTION
    required_facts: list[str] = Field(default_factory=list)


class PolicyFormatDecision(DomainModel):
    code: str
    rule_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class PolicyArgumentDecision(DomainModel):
    name: str
    rule_ids: list[str] = Field(default_factory=list)
    presence_required: bool = True
    has_fixed_value: bool = False
    fixed_value: Any | None = None
    source: str | None = None


class PolicyConflict(DomainModel):
    code: str
    message: str
    rule_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False
    resolvable_with_confirmation: bool = False


class ConfirmationRequirement(DomainModel):
    explicit_yes: bool = True
    describe_action_and_parameters: bool = True
    include_full_bundle: bool = True
    rule_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    bundle_operations: list[SemanticPolicyCall] = Field(default_factory=list)


class PolicyDecision(DomainModel):
    applied_rule_ids: list[str] = Field(default_factory=list)
    required_reads: list[SemanticPolicyCall] = Field(default_factory=list)
    prerequisite_calls: list[SemanticPolicyCall] = Field(default_factory=list)
    confirmation: ConfirmationRequirement | None = None
    format_decisions: list[PolicyFormatDecision] = Field(default_factory=list)
    argument_decisions: list[PolicyArgumentDecision] = Field(default_factory=list)
    conflicts: list[PolicyConflict] = Field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.DEFAULT

    @property
    def has_blocking_conflict(self) -> bool:
        return bool(self.conflicts)


class PolicyRequest(DomainModel):
    """Facts needed to evaluate one candidate action or response.

    ``state`` contains observed vehicle/tool state. ``facts`` contains
    provenance-checked planning facts such as selected route metadata or
    presentation metadata. Missing facts cause reads or conflicts; they are
    never guessed by the rule engine.
    """

    goal_ids: list[str] = Field(default_factory=list)
    action: OfficialToolCall | None = None
    semantic_operation: str | None = None
    ordered_actions: list[OfficialToolCall] = Field(default_factory=list)
    tool_description: str | None = None
    state_changing: bool | None = None
    authorization: ActionAuthorization = ActionAuthorization.NONE
    state: dict[str, Any] = Field(default_factory=dict)
    facts: dict[str, Any] = Field(default_factory=dict)
    requested_tool_available: bool = True
    requested_parameters_available: bool = True
    substitution_authorized_by_user: bool = False

    @model_validator(mode="after")
    def infer_operation_metadata(self) -> "PolicyRequest":
        if self.semantic_operation is None and self.action is not None:
            object.__setattr__(self, "semantic_operation", self.action.tool_name)
        if self.state_changing is None:
            object.__setattr__(
                self,
                "state_changing",
                _operation_is_state_changing(self.semantic_operation or ""),
            )
        return self

    @property
    def arguments(self) -> dict[str, Any]:
        return {} if self.action is None else self.action.arguments


_CATEGORY_BY_RULE: Final[dict[str, PolicyRuleCategory]] = {
    "POL-002": PolicyRuleCategory.FORMAT,
    "POL-004": PolicyRuleCategory.CONFIRMATION,
    "POL-005": PolicyRuleCategory.VEHICLE,
    "POL-007": PolicyRuleCategory.CONFIRMATION,
    "POL-008": PolicyRuleCategory.CONFIRMATION,
    "POL-009": PolicyRuleCategory.VEHICLE,
    "POL-010": PolicyRuleCategory.VEHICLE,
    "POL-011": PolicyRuleCategory.VEHICLE,
    "POL-012": PolicyRuleCategory.FORMAT,
    "POL-013": PolicyRuleCategory.VEHICLE,
    "POL-014": PolicyRuleCategory.VEHICLE,
    "POL-016": PolicyRuleCategory.NAVIGATION,
    "POL-017": PolicyRuleCategory.NAVIGATION,
    "POL-018": PolicyRuleCategory.NAVIGATION,
    "POL-019": PolicyRuleCategory.NAVIGATION,
    "POL-021": PolicyRuleCategory.FORMAT,
    "POL-022": PolicyRuleCategory.NAVIGATION,
    "POL-023": PolicyRuleCategory.INFORMATION,
    "POL-024": PolicyRuleCategory.INFORMATION,
}


def category_for_rule(rule_id: str) -> PolicyRuleCategory:
    return _CATEGORY_BY_RULE[rule_id]


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _has_all(text: str, *groups: tuple[str, ...]) -> bool:
    return all(any(term in text for term in group) for group in groups)


def infer_numbered_rules_from_semantics(system_text: str) -> set[str]:
    """Recognize rules even when policy labels have been stripped."""

    text = _normalized_text(system_text)
    checks: dict[str, bool] = {
        "POL-002": _has_all(
            text,
            ("metric system", "kilometers and meters"),
            ("degree celsius", "degrees celsius"),
            ("24h format", "24-hour format"),
        ),
        "POL-004": _has_all(
            text,
            ("requires_confirmation",),
            ("tool description",),
            ("explicit",),
            ("confirmation",),
        ),
        "POL-005": _has_all(
            text,
            ("sunroof",),
            ("sunshade",),
            ("fully opened", "fully open", "100%"),
        ),
        "POL-007": _has_all(
            text,
            ("windows", "window"),
            ("more than 25%", ">25%"),
            ("ac is on", "air conditioning is on"),
            ("energy inefficiency", "energy inefficient"),
        ),
        "POL-008": _has_all(
            text,
            ("weather",),
            ("sunroof",),
            ("fog lights",),
            ("explicit expressive user confirmation", "explicit confirmation"),
            ("partly_cloudy",),
        ),
        "POL-009": _has_all(
            text,
            ("weather",),
            ("checked manually", "checked before", "check the weather"),
            ("action is performed", "before the action", "before opening"),
        ),
        "POL-010": _has_all(
            text,
            ("defrost",),
            ("fan speed",),
            ("level 2",),
            ("windshield",),
            ("air conditioning",),
        ),
        "POL-011": _has_all(
            text,
            ("air conditioning to on", "air conditioning on"),
            ("close all windows",),
            ("more than 20%", ">20%"),
            ("fan speed",),
            ("level 1",),
        ),
        "POL-012": _has_all(
            text,
            ("single seat zone",),
            ("temperature difference",),
            ("more than 3 degrees", ">3 degrees"),
            ("informed",),
        ),
        "POL-013": _has_all(
            text,
            ("activating fog lights", "activate fog lights"),
            ("low beam",),
            ("high beam",),
        ),
        "POL-014": _has_all(
            text,
            ("high beam",),
            ("cannot be activated", "must not be activated"),
            ("fog lights",),
        ),
        "POL-016": _has_all(
            text,
            ("start of the overall route", "new route start"),
            ("current car location", "current vehicle location"),
        ),
        "POL-017": _has_all(
            text,
            ("delete, replace, or add", "add, delete, or replace"),
            ("waypoint",),
            ("navigation system is already active", "navigation active"),
            ("route is set", "existing route"),
        ),
        "POL-018": _has_all(
            text,
            ("navigation system is already active", "active route"),
            ("multiple waypoints", "multiple navigation edits"),
            ("in sequence", "serial"),
            ("new navigation",),
            ("inactive",),
        ),
        "POL-019": _has_all(
            text,
            ("at least a start and a destination", "at least start and destination"),
            ("destination cannot be deleted", "cannot delete the destination"),
        ),
        "POL-021": _has_all(
            text,
            ("route is presented in detail", "detailed route"),
            ("toll roads", "toll road"),
            ("informed", "disclose"),
        ),
        "POL-022": _has_all(
            text,
            ("multi-stop route", "multistop route"),
            ("fastest route",),
            ("per route segment", "each segment"),
            ("alternative routes", "route alternatives"),
        ),
        "POL-023": _has_all(
            text,
            ("calendar",),
            ("only entries for the current day", "only current-day entries"),
        ),
        "POL-024": _has_all(
            text,
            ("weather",),
            ("only be requested for the current day", "only the current day"),
            ("specified time",),
        ),
    }
    return {rule_id for rule_id, matches in checks.items() if matches}


def infer_general_rules(system_text: str) -> set[str]:
    text = _normalized_text(system_text)
    found: set[str] = {
        GENERAL_TTS,
        GENERAL_REACTIVE_ONLY,
        GENERAL_LIVE_SCHEMA,
        GENERAL_TOOL_DESCRIPTION,
    }
    if _has_all(
        text,
        ("multiple alternative routes", "route alternatives"),
        ("fastest route",),
        ("shortest route",),
        ("further route alternatives", "further routes"),
    ):
        found.add(GENERAL_ROUTE_PRESENTATION)
    if _has_all(
        text,
        ("points of interest", "poi"),
        ("at least by name",),
        ("wants directions", "navigation", "directions"),
    ):
        found.add(GENERAL_POI_PRESENTATION)
    return found


_STATE_CHANGING_NAMES: Final[set[str]] = {
    "open_close_sunroof",
    "open_close_sunshade",
    "open_close_window",
    "open_close_trunk_door",
    "set_air_conditioning",
    "set_window_defrost",
    "set_fan_speed",
    "set_fan_airflow_direction",
    "set_fog_lights",
    "set_head_lights_low_beams",
    "set_head_lights_high_beams",
    "set_climate_temperature",
    "set_new_navigation",
    "navigation_add_one_waypoint",
    "navigation_delete_destination",
    "navigation_delete_final_destination",
    "navigation_delete_waypoint",
    "navigation_replace_final_destination",
    "navigation_replace_one_waypoint",
    "delete_current_navigation",
    "send_email",
    "call_phone_by_number",
}


def _canonical_operation(operation: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", operation.casefold()).strip("_")


def _operation_is_state_changing(operation: str) -> bool:
    canonical = _canonical_operation(operation)
    operation_tokens = set(canonical.split("_"))
    return (
        canonical in _STATE_CHANGING_NAMES
        or canonical.startswith(("set_", "open_", "close_", "delete_"))
        or canonical.endswith(("_set", "_delete", "_replace", "_add"))
        or bool(
            operation_tokens.intersection(
                {"set", "open", "close", "delete", "replace", "add", "send"}
            )
        )
    )


_ALIASES: Final[dict[str, set[str]]] = {
    "sunroof_set": {
        "open_close_sunroof",
        "set_sunroof_position",
        "vehicle_sunroof_set_position",
        "sunroof_set",
    },
    "sunshade_set": {
        "open_close_sunshade",
        "set_sunshade_position",
        "vehicle_sunshade_set_position",
        "sunshade_set",
    },
    "window_set": {
        "open_close_window",
        "set_window_position",
        "vehicle_window_set_position",
        "window_set",
    },
    "ac_set": {
        "set_air_conditioning",
        "vehicle_ac_set",
        "climate_air_conditioning_set",
    },
    "defrost_set": {
        "set_window_defrost",
        "vehicle_defrost_set",
        "window_defrost_set",
    },
    "fog_set": {
        "set_fog_lights",
        "vehicle_fog_lights_set",
        "fog_lights_set",
    },
    "high_beam_set": {
        "set_head_lights_high_beams",
        "set_high_beams",
        "vehicle_high_beam_set",
        "high_beam_set",
    },
    "low_beam_set": {
        "set_head_lights_low_beams",
        "set_low_beams",
        "vehicle_low_beam_set",
        "low_beam_set",
    },
    "temperature_set": {
        "set_climate_temperature",
        "vehicle_climate_temperature_set",
        "climate_temperature_set",
    },
    "navigation_set": {
        "set_new_navigation",
        "navigation_set_new",
        "navigation_new_set",
    },
    "navigation_edit": {
        "navigation_add_one_waypoint",
        "navigation_delete_destination",
        "navigation_delete_final_destination",
        "navigation_delete_waypoint",
        "navigation_delete_one_waypoint",
        "navigation_replace_final_destination",
        "navigation_replace_one_waypoint",
        "navigation_waypoint_add",
        "navigation_waypoint_delete",
        "navigation_waypoint_replace",
        "navigation_destination_delete",
        "navigation_destination_replace",
    },
    "navigation_delete_destination": {
        "navigation_delete_destination",
        "navigation_delete_final_destination",
        "navigation_destination_delete",
    },
    "calendar_get": {
        "get_entries_from_calendar",
        "calendar_entries_get",
        "calendar_get",
    },
    "weather_get": {"get_weather", "weather_current_read", "weather_get"},
}


def _is_operation(request: PolicyRequest, operation: str) -> bool:
    canonical = _canonical_operation(request.semantic_operation or "")
    return canonical in _ALIASES[operation]


_MISSING = object()


def _mapping_path(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _fact(request: PolicyRequest, *paths: str) -> Any:
    for mapping in (request.state, request.facts):
        for path in paths:
            value = _mapping_path(mapping, path)
            if value is not _MISSING and value is not None:
                return value
    return _MISSING


def _argument(request: PolicyRequest, *names: str) -> Any:
    for name in names:
        if name in request.arguments:
            return request.arguments[name]
    return _MISSING


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "on", "yes", "1", "active"}:
            return True
        if normalized in {"false", "off", "no", "0", "inactive"}:
            return False
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stable_key(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


class _DecisionBuilder:
    def __init__(self) -> None:
        self.applied: list[str] = []
        self.reads: list[SemanticPolicyCall] = []
        self.prerequisites: list[SemanticPolicyCall] = []
        self.formats: list[PolicyFormatDecision] = []
        self.arguments: list[PolicyArgumentDecision] = []
        self.conflicts: list[PolicyConflict] = []
        self.confirmation_rule_ids: list[str] = []
        self.confirmation_reasons: list[str] = []
        self.confirmation_warnings: list[str] = []
        self.execution_mode = ExecutionMode.DEFAULT

    def apply(self, rule_id: str) -> None:
        if rule_id not in self.applied:
            self.applied.append(rule_id)

    def read(
        self,
        semantic_operation: str,
        *,
        rule_id: str,
        arguments: dict[str, Any] | None = None,
        required_facts: Iterable[str] = (),
    ) -> None:
        self.apply(rule_id)
        self.reads.append(
            SemanticPolicyCall(
                semantic_operation=semantic_operation,
                arguments=arguments or {},
                rule_ids=[rule_id],
                required_facts=list(required_facts),
            )
        )

    def prerequisite(
        self,
        semantic_operation: str,
        arguments: dict[str, Any],
        *,
        rule_id: str,
        order: RequirementOrder = RequirementOrder.BEFORE_ACTION,
    ) -> None:
        self.apply(rule_id)
        self.prerequisites.append(
            SemanticPolicyCall(
                semantic_operation=semantic_operation,
                arguments=arguments,
                rule_ids=[rule_id],
                order=order,
            )
        )

    def require_confirmation(
        self,
        rule_id: str,
        reason: str,
        *,
        warning: str | None = None,
    ) -> None:
        self.apply(rule_id)
        if rule_id not in self.confirmation_rule_ids:
            self.confirmation_rule_ids.append(rule_id)
        if reason not in self.confirmation_reasons:
            self.confirmation_reasons.append(reason)
        if warning is not None and warning not in self.confirmation_warnings:
            self.confirmation_warnings.append(warning)

    def format(
        self,
        code: str,
        *,
        rule_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.apply(rule_id)
        self.formats.append(
            PolicyFormatDecision(
                code=code,
                rule_ids=[rule_id],
                details=details or {},
            )
        )

    def argument(
        self,
        name: str,
        *,
        rule_id: str,
        fixed_value: Any = _MISSING,
        source: str | None = None,
    ) -> None:
        self.apply(rule_id)
        self.arguments.append(
            PolicyArgumentDecision(
                name=name,
                rule_ids=[rule_id],
                has_fixed_value=fixed_value is not _MISSING,
                fixed_value=None if fixed_value is _MISSING else fixed_value,
                source=source,
            )
        )

    def conflict(
        self,
        code: str,
        message: str,
        *,
        rule_id: str,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
        resolvable_with_confirmation: bool = False,
    ) -> None:
        self.apply(rule_id)
        self.conflicts.append(
            PolicyConflict(
                code=code,
                message=message,
                rule_ids=[rule_id],
                details=details or {},
                retryable=retryable,
                resolvable_with_confirmation=resolvable_with_confirmation,
            )
        )

    @staticmethod
    def _merge_calls(calls: list[SemanticPolicyCall]) -> list[SemanticPolicyCall]:
        merged: dict[str, SemanticPolicyCall] = {}
        for call in calls:
            key = _stable_key(
                {
                    "operation": call.semantic_operation,
                    "arguments": call.arguments,
                    "order": call.order.value,
                    "facts": call.required_facts,
                }
            )
            if key not in merged:
                merged[key] = call
                continue
            prior = merged[key]
            merged[key] = prior.model_copy(
                update={"rule_ids": _ordered_unique(prior.rule_ids + call.rule_ids)}
            )
        return list(merged.values())

    @staticmethod
    def _merge_formats(
        formats: list[PolicyFormatDecision],
    ) -> list[PolicyFormatDecision]:
        merged: dict[str, PolicyFormatDecision] = {}
        for decision in formats:
            key = _stable_key({"code": decision.code, "details": decision.details})
            if key not in merged:
                merged[key] = decision
                continue
            prior = merged[key]
            merged[key] = prior.model_copy(
                update={"rule_ids": _ordered_unique(prior.rule_ids + decision.rule_ids)}
            )
        return list(merged.values())

    @staticmethod
    def _merge_arguments(
        arguments: list[PolicyArgumentDecision],
    ) -> list[PolicyArgumentDecision]:
        merged: dict[str, PolicyArgumentDecision] = {}
        for decision in arguments:
            key = decision.name
            if key not in merged:
                merged[key] = decision
                continue
            prior = merged[key]
            if (
                prior.has_fixed_value
                and decision.has_fixed_value
                and prior.fixed_value != decision.fixed_value
            ):
                raise ValueError(f"conflicting policy values for argument {key}")
            merged[key] = prior.model_copy(
                update={
                    "rule_ids": _ordered_unique(prior.rule_ids + decision.rule_ids),
                    "has_fixed_value": (
                        prior.has_fixed_value or decision.has_fixed_value
                    ),
                    "fixed_value": (
                        prior.fixed_value
                        if prior.has_fixed_value
                        else decision.fixed_value
                    ),
                    "source": prior.source or decision.source,
                }
            )
        return list(merged.values())

    def build(self, request: PolicyRequest) -> PolicyDecision:
        reads = self._merge_calls(self.reads)
        prerequisites = self._merge_calls(self.prerequisites)
        formats = self._merge_formats(self.formats)
        arguments = self._merge_arguments(self.arguments)
        confirmation: ConfirmationRequirement | None = None
        if self.confirmation_rule_ids:
            bundle = list(prerequisites)
            if request.semantic_operation is not None:
                bundle.append(
                    SemanticPolicyCall(
                        semantic_operation=request.semantic_operation,
                        arguments=request.arguments,
                        rule_ids=list(self.confirmation_rule_ids),
                    )
                )
            confirmation = ConfirmationRequirement(
                rule_ids=list(self.confirmation_rule_ids),
                reason_codes=list(self.confirmation_reasons),
                warnings=list(self.confirmation_warnings),
                bundle_operations=self._merge_calls(bundle),
            )
        return PolicyDecision(
            applied_rule_ids=self.applied,
            required_reads=reads,
            prerequisite_calls=prerequisites,
            confirmation=confirmation,
            format_decisions=formats,
            argument_decisions=arguments,
            conflicts=self.conflicts,
            execution_mode=self.execution_mode,
        )


def _apply_general_constraints(
    request: PolicyRequest,
    builder: _DecisionBuilder,
    general_rule_ids: set[str],
) -> None:
    builder.format("natural_first_person", rule_id=GENERAL_TTS)
    builder.format("respond_in_user_language", rule_id=GENERAL_TTS)
    builder.format("tts_no_markdown_or_visual_lists", rule_id=GENERAL_TTS)

    if request.state_changing and request.authorization is ActionAuthorization.NONE:
        builder.conflict(
            "reactive_action_not_authorized",
            "A state-changing action requires a user request, confirmation, or selection.",
            rule_id=GENERAL_REACTIVE_ONLY,
        )

    if not request.requested_tool_available:
        builder.conflict(
            "requested_tool_unavailable",
            "The explicitly requested action is not available in the live tools.",
            rule_id=GENERAL_LIVE_SCHEMA,
            details={
                "must_not_substitute": not request.substitution_authorized_by_user
            },
        )
        builder.format("transparently_report_unavailable", rule_id=GENERAL_LIVE_SCHEMA)

    if not request.requested_parameters_available:
        builder.conflict(
            "requested_parameter_unavailable",
            "An explicitly requested parameter is not available in the live schema.",
            rule_id=GENERAL_LIVE_SCHEMA,
            details={
                "must_not_substitute": not request.substitution_authorized_by_user
            },
        )
        builder.format("ask_before_any_substitution", rule_id=GENERAL_LIVE_SCHEMA)

    description = (request.tool_description or "").lstrip()
    if description.casefold().startswith("requires_confirmation"):
        builder.require_confirmation(
            GENERAL_TOOL_DESCRIPTION,
            "tool_description_requires_confirmation",
        )
        builder.format(
            "describe_action_and_all_parameters_before_confirmation",
            rule_id=GENERAL_TOOL_DESCRIPTION,
            details={"arguments": request.arguments},
        )

    if GENERAL_ROUTE_PRESENTATION in general_rule_ids:
        _apply_route_presentation(request, builder)
    if GENERAL_POI_PRESENTATION in general_rule_ids:
        _apply_poi_presentation(request, builder)


def _apply_route_presentation(
    request: PolicyRequest, builder: _DecisionBuilder
) -> None:
    route_count = _as_float(
        _fact(
            request, "route_alternative_count", "presentation.route_alternative_count"
        )
    )
    if route_count is None or route_count < 2:
        return
    fastest_id = _fact(request, "fastest_route_id", "presentation.fastest_route_id")
    shortest_id = _fact(request, "shortest_route_id", "presentation.shortest_route_id")
    same_route = (
        fastest_id is not _MISSING
        and shortest_id is not _MISSING
        and fastest_id == shortest_id
    )
    builder.format(
        "route_alternatives_fastest_and_shortest",
        rule_id=GENERAL_ROUTE_PRESENTATION,
        details={
            "collapse_if_same": True,
            "same_route": same_route,
            "other_routes_count_only": True,
            "ask_for_more_or_navigation": True,
        },
    )


def _apply_poi_presentation(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    poi_names = _fact(request, "poi_names", "presentation.poi_names")
    if poi_names is _MISSING:
        return
    builder.format(
        "present_all_poi_names",
        rule_id=GENERAL_POI_PRESENTATION,
        details={
            "names": list(poi_names) if isinstance(poi_names, list) else poi_names
        },
    )
    builder.format(
        "ask_which_poi_to_navigate_to",
        rule_id=GENERAL_POI_PRESENTATION,
    )


def _apply_pol_002(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    del request
    builder.format("metric_distance_km_and_m", rule_id="POL-002")
    builder.format("temperature_celsius", rule_id="POL-002")
    builder.format("datetime_24h", rule_id="POL-002")


def _apply_pol_004(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    description = (request.tool_description or "").lstrip()
    if not description.casefold().startswith("requires_confirmation"):
        return
    builder.require_confirmation("POL-004", "tool_description_requires_confirmation")
    builder.format(
        "describe_action_and_all_parameters_before_confirmation",
        rule_id="POL-004",
        details={"arguments": request.arguments},
    )


def _apply_pol_005(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    if not _is_operation(request, "sunroof_set"):
        return
    percentage = _as_float(_argument(request, "percentage", "position"))
    if percentage is None or percentage <= 0:
        return
    shade = _fact(
        request,
        "sunshade_open_percentage",
        "sunshade_position",
        "roof.sunshade_position",
        "vehicle.sunshade_position",
    )
    if shade is _MISSING:
        builder.read(
            "vehicle.roof_state.read",
            rule_id="POL-005",
            required_facts=["sunshade_open_percentage"],
        )
        return
    if (_as_float(shade) or 0.0) < 100.0:
        builder.prerequisite(
            "vehicle.sunshade.set_position",
            {"percentage": 100},
            rule_id="POL-005",
        )


def _apply_pol_007(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    if not _is_operation(request, "window_set"):
        return
    percentage = _as_float(_argument(request, "percentage", "position"))
    if percentage is None or percentage <= 25:
        return
    ac_state = _fact(
        request,
        "air_conditioning",
        "ac_on",
        "climate.air_conditioning",
    )
    if ac_state is _MISSING:
        builder.read(
            "vehicle.climate_state.read",
            rule_id="POL-007",
            required_facts=["air_conditioning"],
        )
    elif _as_bool(ac_state) is True:
        builder.require_confirmation(
            "POL-007",
            "window_open_with_ac_on",
            warning="Opening the window this far while AC is on is energy inefficient.",
        )
        builder.format("warn_energy_inefficiency", rule_id="POL-007")


def _weather_action_kind(request: PolicyRequest) -> str | None:
    if _is_operation(request, "sunroof_set"):
        percentage = _as_float(_argument(request, "percentage", "position"))
        return "sunroof" if percentage is not None and percentage > 0 else None
    if _is_operation(request, "fog_set"):
        return (
            "fog_lights"
            if _as_bool(_argument(request, "on", "enabled")) is True
            else None
        )
    return None


def _weather_read_arguments(
    *, current_location: dict[str, Any] | None, current_datetime: dict[str, Any] | None
) -> dict[str, Any] | None:
    location_id = None if current_location is None else current_location.get("id")
    if not location_id or current_datetime is None:
        return None
    needed = ("month", "day", "hour")
    if any(key not in current_datetime for key in needed):
        return None
    return {
        "location_or_poi_id": location_id,
        "month": current_datetime["month"],
        "day": current_datetime["day"],
        "time_hour_24hformat": current_datetime["hour"],
        "time_minutes": current_datetime.get("minute", 0),
    }


def _weather_condition(request: PolicyRequest) -> Any:
    return _fact(
        request,
        "weather_condition",
        "weather.condition",
        "weather.current_slot.condition",
        "current_slot.condition",
    )


def _apply_pol_008(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    action_kind = _weather_action_kind(request)
    if action_kind is None:
        return
    condition = _weather_condition(request)
    if condition is _MISSING:
        return
    normalized = str(condition).strip().casefold()
    allowed = {
        "sunroof": {"sunny", "cloudy", "partly_cloudy"},
        "fog_lights": {"cloudy_and_thunderstorm", "cloudy_and_hail"},
    }[action_kind]
    if normalized not in allowed:
        builder.require_confirmation(
            "POL-008",
            f"{action_kind}_outside_policy_weather",
            warning=f"Current weather is {normalized}.",
        )
        builder.format(
            "state_weather_warning_before_confirmation",
            rule_id="POL-008",
            details={"condition": normalized, "action": action_kind},
        )


def _apply_pol_009(
    request: PolicyRequest,
    builder: _DecisionBuilder,
    *,
    current_location: dict[str, Any] | None,
    current_datetime: dict[str, Any] | None,
) -> None:
    action_kind = _weather_action_kind(request)
    if action_kind is None or _weather_condition(request) is not _MISSING:
        return
    arguments = _weather_read_arguments(
        current_location=current_location,
        current_datetime=current_datetime,
    )
    if arguments is None:
        builder.conflict(
            "current_weather_context_unavailable",
            "Current location and datetime are required before checking weather.",
            rule_id="POL-009",
            retryable=True,
        )
        return
    builder.read(
        "weather.current.read",
        rule_id="POL-009",
        arguments=arguments,
        required_facts=["weather_condition"],
    )


def _climate_fact(request: PolicyRequest, key: str) -> Any:
    aliases: dict[str, tuple[str, ...]] = {
        "fan_speed": ("fan_speed", "climate.fan_speed"),
        "fan_airflow_direction": (
            "fan_airflow_direction",
            "airflow_direction",
            "climate.fan_airflow_direction",
        ),
        "air_conditioning": (
            "air_conditioning",
            "ac_on",
            "climate.air_conditioning",
        ),
    }
    return _fact(request, *aliases[key])


def _apply_pol_010(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    if not _is_operation(request, "defrost_set"):
        return
    if _as_bool(_argument(request, "on", "enabled")) is not True:
        return
    target = str(_argument(request, "defrost_window", "window")).upper()
    if target not in {"FRONT", "ALL"}:
        return
    facts = {
        key: _climate_fact(request, key)
        for key in ("fan_speed", "fan_airflow_direction", "air_conditioning")
    }
    missing = [key for key, value in facts.items() if value is _MISSING]
    if missing:
        builder.read(
            "vehicle.climate_state.read",
            rule_id="POL-010",
            required_facts=missing,
        )
        return
    fan_speed = _as_float(facts["fan_speed"])
    if fan_speed is not None and fan_speed < 2:
        builder.prerequisite("vehicle.fan.set_speed", {"level": 2}, rule_id="POL-010")
    airflow = str(facts["fan_airflow_direction"]).upper()
    if "WINDSHIELD" not in airflow:
        builder.prerequisite(
            "vehicle.fan.set_airflow",
            {"direction": "WINDSHIELD"},
            rule_id="POL-010",
        )
    if _as_bool(facts["air_conditioning"]) is not True:
        builder.prerequisite("vehicle.ac.set", {"on": True}, rule_id="POL-010")


_WINDOW_FACTS: Final[tuple[tuple[str, str], ...]] = (
    ("window_driver_position", "DRIVER"),
    ("window_passenger_position", "PASSENGER"),
    ("window_driver_rear_position", "DRIVER_REAR"),
    ("window_passenger_rear_position", "PASSENGER_REAR"),
)


def _window_fact(request: PolicyRequest, key: str) -> Any:
    short = key.removeprefix("window_").removesuffix("_position")
    return _fact(
        request,
        key,
        f"window_positions.{key}",
        f"window_positions.{short}",
        f"windows.{short}",
    )


def _ordered_actions_set_fan_at_least(
    request: PolicyRequest, minimum_level: float
) -> bool:
    for action in request.ordered_actions:
        if _canonical_operation(action.tool_name) != "set_fan_speed":
            continue
        level = _as_float(action.arguments.get("level"))
        if level is not None and level >= minimum_level:
            return True
    return False


def _ordered_window_close_before_current_ac(
    request: PolicyRequest, target: str
) -> OfficialToolCall | None:
    current = request.action
    if current is None:
        return None
    for action in request.ordered_actions:
        if (
            action.tool_name == current.tool_name
            and action.arguments == current.arguments
        ):
            break
        if _canonical_operation(action.tool_name) not in _ALIASES["window_set"]:
            continue
        window = str(action.arguments.get("window", "")).upper()
        percentage = _as_float(action.arguments.get("percentage"))
        if window in {"ALL", target} and percentage is not None and percentage <= 20:
            return action
    return None


def _apply_pol_011(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    if not _is_operation(request, "ac_set"):
        return
    if _as_bool(_argument(request, "on", "enabled")) is not True:
        return
    missing_windows = [
        key for key, _ in _WINDOW_FACTS if _window_fact(request, key) is _MISSING
    ]
    if missing_windows:
        builder.read(
            "vehicle.window_state.read",
            rule_id="POL-011",
            required_facts=missing_windows,
        )
    else:
        retained_ordered_closes: set[str] = set()
        for key, target in _WINDOW_FACTS:
            position = _as_float(_window_fact(request, key))
            if position is None or position <= 20:
                continue
            ordered_close = _ordered_window_close_before_current_ac(request, target)
            if ordered_close is None:
                builder.prerequisite(
                    "vehicle.window.set_position",
                    {"window": target, "percentage": 0},
                    rule_id="POL-011",
                )
                continue
            fingerprint = _stable_key(ordered_close.arguments)
            if fingerprint in retained_ordered_closes:
                continue
            retained_ordered_closes.add(fingerprint)
            builder.prerequisite(
                "vehicle.window.set_position",
                dict(ordered_close.arguments),
                rule_id="POL-011",
            )
    fan_speed = _climate_fact(request, "fan_speed")
    if fan_speed is _MISSING:
        builder.read(
            "vehicle.climate_state.read",
            rule_id="POL-011",
            required_facts=["fan_speed"],
        )
    elif _as_float(fan_speed) == 0 and not _ordered_actions_set_fan_at_least(
        request, 1
    ):
        builder.prerequisite("vehicle.fan.set_speed", {"level": 1}, rule_id="POL-011")


def _temperature_facts(request: PolicyRequest) -> tuple[Any, Any]:
    driver = _fact(
        request,
        "climate_temperature_driver",
        "temperatures.driver",
        "climate.temperatures.driver",
    )
    passenger = _fact(
        request,
        "climate_temperature_passenger",
        "temperatures.passenger",
        "climate.temperatures.passenger",
    )
    return driver, passenger


def _apply_pol_012(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    if not _is_operation(request, "temperature_set"):
        return
    zone = str(_argument(request, "seat_zone", "zone")).upper()
    if zone not in {"DRIVER", "PASSENGER"}:
        return
    requested_temperature = _as_float(_argument(request, "temperature"))
    if requested_temperature is None:
        return
    driver, passenger = _temperature_facts(request)
    if driver is _MISSING or passenger is _MISSING:
        builder.read(
            "vehicle.cabin_temperatures.read",
            rule_id="POL-012",
            required_facts=[
                "climate_temperature_driver",
                "climate_temperature_passenger",
            ],
        )
        return
    other = passenger if zone == "DRIVER" else driver
    other_temperature = _as_float(other)
    if other_temperature is None:
        return
    difference = abs(requested_temperature - other_temperature)
    if difference > 3:
        builder.format(
            "inform_temperature_zone_difference",
            rule_id="POL-012",
            details={
                "changed_zone": zone,
                "other_zone": "PASSENGER" if zone == "DRIVER" else "DRIVER",
                "resulting_difference_celsius": difference,
            },
        )


def _light_fact(request: PolicyRequest, light: str) -> Any:
    aliases: dict[str, tuple[str, ...]] = {
        "fog": ("fog_lights", "fog_lights_on", "lights.fog_lights"),
        "low": (
            "head_lights_low_beams",
            "low_beam_on",
            "lights.head_lights_low_beams",
        ),
        "high": (
            "head_lights_high_beams",
            "high_beam_on",
            "lights.head_lights_high_beams",
        ),
    }
    return _fact(request, *aliases[light])


def _apply_pol_013(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    if not _is_operation(request, "fog_set"):
        return
    if _as_bool(_argument(request, "on", "enabled")) is not True:
        return
    low = _light_fact(request, "low")
    high = _light_fact(request, "high")
    missing = []
    if low is _MISSING:
        missing.append("head_lights_low_beams")
    if high is _MISSING:
        missing.append("head_lights_high_beams")
    if missing:
        builder.read(
            "vehicle.exterior_lights.read",
            rule_id="POL-013",
            required_facts=missing,
        )
        return
    if _as_bool(low) is not True:
        builder.prerequisite("vehicle.low_beam.set", {"on": True}, rule_id="POL-013")
    if _as_bool(high) is True:
        builder.prerequisite("vehicle.high_beam.set", {"on": False}, rule_id="POL-013")


def _apply_pol_014(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    if not _is_operation(request, "high_beam_set"):
        return
    if _as_bool(_argument(request, "on", "enabled")) is not True:
        return
    fog = _light_fact(request, "fog")
    if fog is _MISSING:
        builder.read(
            "vehicle.exterior_lights.read",
            rule_id="POL-014",
            required_facts=["fog_lights"],
        )
        return
    if _as_bool(fog) is True:
        builder.prerequisite("vehicle.fog_lights.set", {"on": False}, rule_id="POL-014")
        builder.require_confirmation(
            "POL-014",
            "turn_off_fog_lights_before_high_beam",
            warning="Fog lights must be turned off before high beam can be activated.",
        )
        builder.format(
            "describe_conflicting_lights_bundle",
            rule_id="POL-014",
            details={"ordered_actions": ["fog_lights_off", "high_beam_on"]},
        )


def _navigation_active(request: PolicyRequest) -> Any:
    return _fact(
        request,
        "navigation_active",
        "navigation.active",
        "navigation_is_active",
    )


def _navigation_waypoints(request: PolicyRequest) -> Any:
    return _fact(
        request,
        "waypoints",
        "waypoints_id",
        "navigation.waypoints",
        "navigation.waypoints_id",
    )


def _route_start_id(request: PolicyRequest) -> Any:
    start = _fact(
        request,
        "route_start_id",
        "selected_route.start_id",
        "route.start_id",
    )
    if start is not _MISSING:
        return start
    segments = _fact(request, "route_segments", "selected_route_segments")
    if isinstance(segments, list) and segments and isinstance(segments[0], Mapping):
        return segments[0].get("start_id", _MISSING)
    return _MISSING


def _apply_pol_016(
    request: PolicyRequest,
    builder: _DecisionBuilder,
    *,
    current_location: dict[str, Any] | None,
) -> None:
    if not _is_operation(request, "navigation_set"):
        return
    expected = None if current_location is None else current_location.get("id")
    actual = _route_start_id(request)
    if expected is None:
        builder.conflict(
            "current_location_unavailable",
            "The current vehicle location is required to validate a new route.",
            rule_id="POL-016",
            retryable=True,
        )
    elif actual is _MISSING:
        builder.conflict(
            "route_start_unverified",
            "Selected route metadata must prove its start is the vehicle location.",
            rule_id="POL-016",
            details={"expected_start_id": expected},
            retryable=True,
        )
    elif actual != expected:
        builder.conflict(
            "route_start_not_current_location",
            "A new route must start at the current vehicle location.",
            rule_id="POL-016",
            details={"expected_start_id": expected, "actual_start_id": actual},
        )


def _require_navigation_state(
    request: PolicyRequest,
    builder: _DecisionBuilder,
    *,
    rule_id: str,
) -> tuple[bool | None, Any]:
    active_raw = _navigation_active(request)
    waypoints = _navigation_waypoints(request)
    missing = []
    if active_raw is _MISSING:
        missing.append("navigation_active")
    if waypoints is _MISSING:
        missing.append("waypoints")
    if missing:
        builder.read(
            "navigation.state.read",
            rule_id=rule_id,
            required_facts=missing,
        )
    active = None if active_raw is _MISSING else _as_bool(active_raw)
    return active, waypoints


def _apply_pol_017(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    if not _is_operation(request, "navigation_edit"):
        return
    active, waypoints = _require_navigation_state(request, builder, rule_id="POL-017")
    if active is False:
        builder.conflict(
            "navigation_edit_requires_active_route",
            "Waypoint and destination edits require active navigation.",
            rule_id="POL-017",
        )
    elif waypoints is not _MISSING and (
        not isinstance(waypoints, list) or len(waypoints) < 2
    ):
        builder.conflict(
            "navigation_edit_requires_existing_route",
            "Waypoint and destination edits require an existing route.",
            rule_id="POL-017",
        )


def _apply_pol_018(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    if not (
        _is_operation(request, "navigation_edit")
        or _is_operation(request, "navigation_set")
    ):
        return
    active_raw = _navigation_active(request)
    if active_raw is _MISSING:
        builder.read(
            "navigation.state.read",
            rule_id="POL-018",
            required_facts=["navigation_active"],
        )
        return
    active = _as_bool(active_raw)
    if _is_operation(request, "navigation_set") and active is True:
        builder.conflict(
            "active_navigation_requires_edit_operation",
            "An active route must be modified with navigation edit operations.",
            rule_id="POL-018",
        )
    if _is_operation(request, "navigation_edit"):
        builder.apply("POL-018")
        builder.execution_mode = ExecutionMode.SERIAL
        if active is False:
            builder.conflict(
                "inactive_navigation_requires_new_route",
                "A new navigation may be set only while navigation is inactive.",
                rule_id="POL-018",
            )


def _apply_pol_019(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    if not _is_operation(request, "navigation_delete_destination"):
        return
    _, waypoints = _require_navigation_state(request, builder, rule_id="POL-019")
    if waypoints is _MISSING:
        return
    if not isinstance(waypoints, list) or len(waypoints) <= 2:
        builder.conflict(
            "cannot_delete_only_destination",
            "A route must retain a start and destination.",
            rule_id="POL-019",
            details={
                "waypoint_count": len(waypoints) if isinstance(waypoints, list) else 0
            },
        )


def _apply_pol_021(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    detailed = _as_bool(
        _fact(request, "route_presented_in_detail", "presentation.route_detailed")
    )
    toll = _as_bool(
        _fact(request, "route_includes_toll", "presentation.route_includes_toll")
    )
    if detailed is True and toll is True:
        builder.format("disclose_route_toll", rule_id="POL-021")


def _apply_pol_022(request: PolicyRequest, builder: _DecisionBuilder) -> None:
    segment_count = _as_float(
        _fact(request, "route_segment_count", "route.segment_count")
    )
    specified = _as_bool(
        _fact(
            request,
            "route_selection_specified",
            "route.selection_specified",
        )
    )
    if segment_count is None or segment_count <= 1 or specified is not False:
        return
    builder.format(
        "select_fastest_route_per_segment",
        rule_id="POL-022",
        details={"selection": "fastest", "segment_count": int(segment_count)},
    )
    builder.format(
        "announce_fastest_default_and_offer_alternatives",
        rule_id="POL-022",
    )
    toll = _as_bool(
        _fact(request, "selected_segments_include_toll", "route.includes_toll")
    )
    if toll is True:
        builder.format("disclose_route_toll", rule_id="POL-022")


def _current_date_arguments(
    current_datetime: dict[str, Any] | None,
) -> tuple[Any, Any] | None:
    if current_datetime is None:
        return None
    if "month" not in current_datetime or "day" not in current_datetime:
        return None
    return current_datetime["month"], current_datetime["day"]


def _apply_current_day_rule(
    request: PolicyRequest,
    builder: _DecisionBuilder,
    *,
    current_datetime: dict[str, Any] | None,
    rule_id: str,
) -> None:
    current = _current_date_arguments(current_datetime)
    if current is None:
        builder.conflict(
            "current_date_unavailable",
            "The system current date is required by this policy.",
            rule_id=rule_id,
            retryable=True,
        )
        return
    month, day = current
    builder.argument(
        "month", rule_id=rule_id, fixed_value=month, source="system_datetime"
    )
    builder.argument("day", rule_id=rule_id, fixed_value=day, source="system_datetime")
    actual_month = _argument(request, "month")
    actual_day = _argument(request, "day")
    if actual_month is not _MISSING and actual_month != month:
        builder.conflict(
            "non_current_day_request",
            "This information can only be requested for the current day.",
            rule_id=rule_id,
            details={"argument": "month", "expected": month, "actual": actual_month},
        )
    if actual_day is not _MISSING and actual_day != day:
        builder.conflict(
            "non_current_day_request",
            "This information can only be requested for the current day.",
            rule_id=rule_id,
            details={"argument": "day", "expected": day, "actual": actual_day},
        )


def _apply_pol_023(
    request: PolicyRequest,
    builder: _DecisionBuilder,
    *,
    current_datetime: dict[str, Any] | None,
) -> None:
    if _is_operation(request, "calendar_get"):
        _apply_current_day_rule(
            request,
            builder,
            current_datetime=current_datetime,
            rule_id="POL-023",
        )


def _apply_pol_024(
    request: PolicyRequest,
    builder: _DecisionBuilder,
    *,
    current_datetime: dict[str, Any] | None,
) -> None:
    if not _is_operation(request, "weather_get"):
        return
    _apply_current_day_rule(
        request,
        builder,
        current_datetime=current_datetime,
        rule_id="POL-024",
    )
    builder.argument("time_hour_24hformat", rule_id="POL-024")
    if _argument(request, "time_hour_24hformat", "hour") is _MISSING:
        builder.conflict(
            "weather_time_required",
            "Weather can only be requested for a specified time.",
            rule_id="POL-024",
            retryable=True,
        )


_SIMPLE_HANDLERS = {
    "POL-002": _apply_pol_002,
    "POL-004": _apply_pol_004,
    "POL-005": _apply_pol_005,
    "POL-007": _apply_pol_007,
    "POL-008": _apply_pol_008,
    "POL-010": _apply_pol_010,
    "POL-011": _apply_pol_011,
    "POL-012": _apply_pol_012,
    "POL-013": _apply_pol_013,
    "POL-014": _apply_pol_014,
    "POL-017": _apply_pol_017,
    "POL-018": _apply_pol_018,
    "POL-019": _apply_pol_019,
    "POL-021": _apply_pol_021,
    "POL-022": _apply_pol_022,
}


def evaluate_rules(
    request: PolicyRequest,
    *,
    active_rule_ids: Iterable[str],
    general_rule_ids: Iterable[str],
    current_location: dict[str, Any] | None,
    current_datetime: dict[str, Any] | None,
) -> PolicyDecision:
    """Evaluate a request without I/O, model calls, or mutable global state."""

    active = set(active_rule_ids)
    general = set(general_rule_ids)
    builder = _DecisionBuilder()
    _apply_general_constraints(request, builder, general)

    for rule_id in NUMBERED_POLICY_IDS:
        if rule_id not in active:
            continue
        if rule_id in _SIMPLE_HANDLERS:
            _SIMPLE_HANDLERS[rule_id](request, builder)
        elif rule_id == "POL-009":
            _apply_pol_009(
                request,
                builder,
                current_location=current_location,
                current_datetime=current_datetime,
            )
        elif rule_id == "POL-016":
            _apply_pol_016(
                request,
                builder,
                current_location=current_location,
            )
        elif rule_id == "POL-023":
            _apply_pol_023(
                request,
                builder,
                current_datetime=current_datetime,
            )
        elif rule_id == "POL-024":
            _apply_pol_024(
                request,
                builder,
                current_datetime=current_datetime,
            )
    return builder.build(request)


__all__ = [
    "ActionAuthorization",
    "CompiledRule",
    "ConfirmationRequirement",
    "ExecutionMode",
    "GENERAL_LIVE_SCHEMA",
    "GENERAL_POI_PRESENTATION",
    "GENERAL_REACTIVE_ONLY",
    "GENERAL_ROUTE_PRESENTATION",
    "GENERAL_TOOL_DESCRIPTION",
    "GENERAL_TTS",
    "NUMBERED_POLICY_IDS",
    "PolicyArgumentDecision",
    "PolicyConflict",
    "PolicyDecision",
    "PolicyFormatDecision",
    "PolicyRequest",
    "PolicyRuleCategory",
    "PolicyRuleSource",
    "RequirementOrder",
    "SemanticPolicyCall",
    "category_for_rule",
    "evaluate_rules",
    "infer_general_rules",
    "infer_numbered_rules_from_semantics",
]
