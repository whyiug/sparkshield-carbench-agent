"""Goal-conditioned evidence registration and explicit result extractors."""

from __future__ import annotations

import re
from math import isfinite
from collections.abc import Iterable, Mapping
from typing import Any

from ..capability.semantic_result_validator import (
    ExtractorSpec,
    SemanticResultValidator,
)
from ..domain import (
    Evidence,
    EvidenceNeed,
    EvidenceSourceKind,
    EvidenceStatus,
    EvidenceStore,
    Goal,
)
from ..recipes import OperationRecipe


def _maximum_window_position(value: Any) -> int | float | None:
    if not isinstance(value, Mapping):
        return None
    positions = [
        item
        for key, item in value.items()
        if key.startswith("window_")
        and key.endswith("_position")
        and type(item) in {int, float}
    ]
    return max(positions) if positions else None


def _window_position_snapshot(value: Any) -> dict[str, int | float] | None:
    """Accept one complete, bounded snapshot of the four controllable windows."""

    if not isinstance(value, Mapping):
        return None
    names = (
        "window_driver_position",
        "window_passenger_position",
        "window_driver_rear_position",
        "window_passenger_rear_position",
    )
    snapshot: dict[str, int | float] = {}
    for name in names:
        position = value.get(name)
        if (
            isinstance(position, bool)
            or not isinstance(position, (int, float))
            or not isfinite(float(position))
            or not 0 <= position <= 100
        ):
            return None
        snapshot[name] = position
    return snapshot


def _exterior_lights_snapshot(value: Any) -> dict[str, bool] | None:
    if not isinstance(value, Mapping):
        return None
    names = {
        "fog_lights",
        "head_lights_low_beams",
        "head_lights_high_beams",
    }
    if set(value) != names or any(type(value.get(name)) is not bool for name in names):
        return None
    return {name: value[name] for name in names}


def _sunroof_preference_percentage(value: Any) -> int | float | None:
    if isinstance(value, str):
        statements = [value]
    elif isinstance(value, (list, tuple)):
        statements = [item for item in value if isinstance(item, str)]
    else:
        return None
    candidates: set[int] = set()
    positive = re.compile(r"\b(?:default|prefer(?:s|red)?|usually|want(?:s|ed)?)\b")
    negative = re.compile(
        r"\b(?:avoid|never|not|without|don['\u2019]?t|doesn['\u2019]?t)\b"
    )
    for statement in statements:
        for raw_clause in re.split(r"[.!?;,]+", statement):
            clause = " ".join(raw_clause.casefold().split())
            if (
                "sunroof" not in clause
                or not re.search(r"\b(?:open|opening|position)\b", clause)
                or not positive.search(clause)
                or negative.search(clause)
            ):
                continue
            for match in re.finditer(r"(?<![\w.])(100|[1-9]?\d)\s*%", clause):
                candidates.add(int(match.group(1)))
    return next(iter(candidates)) if len(candidates) == 1 else None


def _preference_clauses(value: Any) -> tuple[str, ...]:
    statements: tuple[str, ...]
    if isinstance(value, str):
        statements = (value,)
    elif isinstance(value, (list, tuple)):
        statements = tuple(item for item in value if isinstance(item, str))
    else:
        return ()
    return tuple(
        clause
        for statement in statements
        for raw_clause in re.split(r"[.!?;,]+", statement)
        if (clause := " ".join(raw_clause.casefold().split()))
    )


_PREFERENCE_POSITIVE = re.compile(
    r"\b(?:default|comfort(?:able)?|has\s+preference|prefer(?:s|red)?|usually)\b"
)
_PREFERENCE_NEGATIVE = re.compile(
    r"\b(?:avoid|never|not|without|don['\u2019]?t|doesn['\u2019]?t)\b"
)


def _air_circulation_preference(value: Any) -> str | None:
    candidates: set[str] = set()
    for clause in _preference_clauses(value):
        if not _PREFERENCE_POSITIVE.search(clause) or _PREFERENCE_NEGATIVE.search(
            clause
        ):
            continue
        if not re.search(r"\b(?:air\s+circulation|air\s+mode|fresh\s+air|recirculat)", clause):
            continue
        if re.search(r"\bfresh\s+air(?:\s+mode)?\b", clause):
            candidates.add("FRESH_AIR")
        if re.search(r"\brecirculat(?:e|ed|ing|ion)(?:\s+mode)?\b", clause):
            candidates.add("RECIRCULATION")
        if re.search(r"\b(?:auto|automatic)(?:\s+mode)?\b", clause):
            candidates.add("AUTO")
    return next(iter(candidates)) if len(candidates) == 1 else None


def _bounded_level_preference(
    value: Any, *, anchor: re.Pattern[str], maximum: int
) -> int | None:
    candidates: set[int] = set()
    for clause in _preference_clauses(value):
        if (
            not anchor.search(clause)
            or not _PREFERENCE_POSITIVE.search(clause)
            or _PREFERENCE_NEGATIVE.search(clause)
        ):
            continue
        for match in re.finditer(r"\blevel\s+([0-9]+)\b", clause):
            candidate = int(match.group(1))
            if 0 <= candidate <= maximum:
                candidates.add(candidate)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _fan_speed_preference(value: Any) -> int | None:
    return _bounded_level_preference(
        value,
        anchor=re.compile(r"\bfan(?:\s+speed)?\b"),
        maximum=5,
    )


def _steering_wheel_heating_preference(value: Any) -> int | None:
    return _bounded_level_preference(
        value,
        anchor=re.compile(r"\bsteering\s+wheel\s+heat(?:ing)?\b"),
        maximum=3,
    )


def _climate_temperature_preference(value: Any) -> int | float | None:
    candidates: set[float] = set()
    for clause in _preference_clauses(value):
        if (
            not re.search(r"\b(?:climate|comfort|temperature)\b", clause)
            or not _PREFERENCE_POSITIVE.search(clause)
            or _PREFERENCE_NEGATIVE.search(clause)
        ):
            continue
        for match in re.finditer(
            r"(?<![\w.])([0-9]{2}(?:\.5)?)\s*(?:degree|degrees|celsius|c)\b",
            clause,
        ):
            candidate = float(match.group(1))
            if 16 <= candidate <= 28:
                candidates.add(candidate)
    if len(candidates) != 1:
        return None
    selected = next(iter(candidates))
    return int(selected) if selected.is_integer() else selected


def _cabin_temperature_snapshot(value: Any) -> dict[str, int | float | str] | None:
    """Accept only one complete Celsius snapshot from the cabin sensor."""

    if not isinstance(value, Mapping):
        return None
    driver = value.get("climate_temperature_driver")
    passenger = value.get("climate_temperature_passenger")
    unit = value.get("temperature_unit")
    if (
        isinstance(driver, bool)
        or isinstance(passenger, bool)
        or not isinstance(driver, (int, float))
        or not isinstance(passenger, (int, float))
        or not isfinite(float(driver))
        or not isfinite(float(passenger))
        or not isinstance(unit, str)
        or unit.strip().casefold()
        not in {"celsius", "degree celsius", "degrees celsius"}
    ):
        return None
    return {
        "climate_temperature_driver": driver,
        "climate_temperature_passenger": passenger,
        "temperature_unit": "Celsius",
    }


def _front_seat_heating_snapshot(value: Any) -> dict[str, int] | None:
    """Accept one complete, bounded front-seat heating snapshot."""

    if not isinstance(value, Mapping):
        return None
    driver = value.get("seat_heating_driver")
    passenger = value.get("seat_heating_passenger")
    if (
        type(driver) is not int
        or type(passenger) is not int
        or not 0 <= driver <= 3
        or not 0 <= passenger <= 3
    ):
        return None
    return {
        "seat_heating_driver": driver,
        "seat_heating_passenger": passenger,
    }


def _seat_occupancy_snapshot(value: Any) -> dict[str, bool] | None:
    """Accept one complete occupancy snapshot for every reported seat."""

    if not isinstance(value, Mapping):
        return None
    names = ("driver", "passenger", "driver_rear", "passenger_rear")
    if any(type(value.get(name)) is not bool for name in names):
        return None
    return {name: value[name] for name in names}


def _reading_lights_snapshot(value: Any) -> dict[str, bool] | None:
    if not isinstance(value, Mapping):
        return None
    names = (
        "reading_light_driver",
        "reading_light_passenger",
        "reading_light_driver_rear",
        "reading_light_passenger_rear",
    )
    if any(type(value.get(name)) is not bool for name in names):
        return None
    return {name: value[name] for name in names}


RESULT_EXTRACTORS: dict[str, str | ExtractorSpec] = {
    "fan_speed": "$.result.fan_speed",
    "fan_airflow_direction": "$.result.fan_airflow_direction",
    "air_conditioning": "$.result.air_conditioning",
    "maximum_window_open_percentage": ExtractorSpec(
        "$.result", transform=_maximum_window_position
    ),
    "contextual_window_position_snapshot": ExtractorSpec(
        "$.result",
        transform=_window_position_snapshot,
        allowed_tool_names=frozenset({"get_vehicle_window_positions"}),
    ),
    "window_driver_position": "$.result.window_driver_position",
    "window_passenger_position": "$.result.window_passenger_position",
    "window_driver_rear_position": "$.result.window_driver_rear_position",
    "window_passenger_rear_position": "$.result.window_passenger_rear_position",
    "sunroof_position": ExtractorSpec(
        "$.result.sunroof_position",
        allowed_tool_names=frozenset({"get_sunroof_and_sunshade_position"}),
    ),
    "sunshade_open_percentage": "$.result.sunshade_position",
    "sunshade_position": "$.result.sunshade_position",
    "weather_condition": "$.result.current_slot.condition",
    "seat_occupancy_state": "$.result.seats_occupied",
    "reading_light_seat_occupancy": ExtractorSpec(
        "$.result.seats_occupied",
        transform=_seat_occupancy_snapshot,
        allowed_tool_names=frozenset({"get_seats_occupancy"}),
    ),
    "reading_lights_state": ExtractorSpec(
        "$.result",
        transform=_reading_lights_snapshot,
        allowed_tool_names=frozenset({"get_reading_lights_status"}),
    ),
    "ambient_car_exterior_color": ExtractorSpec(
        "$.result.car_color",
        allowed_tool_names=frozenset({"get_car_color"}),
    ),
    "route_options": "$.result.routes",
    "conditional_arrival_route_options": ExtractorSpec(
        "$.result.routes",
        allowed_tool_names=frozenset({"get_routes_from_start_to_destination"}),
    ),
    "conditional_fallback_location_id": ExtractorSpec(
        "$.result.id",
        allowed_tool_names=frozenset({"get_location_id_by_location_name"}),
    ),
    "conditional_arrival_weather_snapshot": ExtractorSpec(
        "$.result",
        allowed_tool_names=frozenset({"get_weather"}),
    ),
    "resolved_navigation_location_id": ExtractorSpec(
        "$.result.id",
        allowed_tool_names=frozenset({"get_location_id_by_location_name"}),
    ),
    "route_to_replacement_options": ExtractorSpec(
        "$.result.routes",
        allowed_tool_names=frozenset({"get_routes_from_start_to_destination"}),
    ),
    "route_from_replacement_options": ExtractorSpec(
        "$.result.routes",
        allowed_tool_names=frozenset({"get_routes_from_start_to_destination"}),
    ),
    "waypoint_delete_route_options": ExtractorSpec(
        "$.result.routes",
        allowed_tool_names=frozenset({"get_routes_from_start_to_destination"}),
    ),
    "poi_location_id": ExtractorSpec(
        "$.result.id",
        allowed_tool_names=frozenset({"get_location_id_by_location_name"}),
    ),
    "poi_candidates": ExtractorSpec(
        "$.result.pois_found",
        allowed_tool_names=frozenset({"search_poi_at_location"}),
    ),
    "current_destination_poi_candidates": ExtractorSpec(
        "$.result.pois_found",
        allowed_tool_names=frozenset({"search_poi_at_location"}),
    ),
    "replacement_destination_location_id": ExtractorSpec(
        "$.result.id",
        allowed_tool_names=frozenset({"get_location_id_by_location_name"}),
    ),
    "replacement_destination_route_options": ExtractorSpec(
        "$.result.routes",
        allowed_tool_names=frozenset({"get_routes_from_start_to_destination"}),
    ),
    "replacement_navigation_reverification": ExtractorSpec(
        "$.result",
        allowed_tool_names=frozenset({"get_current_navigation_state"}),
    ),
    "trip_charging_status": ExtractorSpec(
        "$.result",
        allowed_tool_names=frozenset({"get_charging_specs_and_status"}),
    ),
    "trip_destination_location_id": ExtractorSpec(
        "$.result.id",
        allowed_tool_names=frozenset({"get_location_id_by_location_name"}),
    ),
    "trip_route_options": ExtractorSpec(
        "$.result.routes",
        allowed_tool_names=frozenset({"get_routes_from_start_to_destination"}),
    ),
    "local_charging_station_candidates": ExtractorSpec(
        "$.result.pois_found",
        allowed_tool_names=frozenset({"search_poi_at_location"}),
    ),
    "local_charging_first_segment_routes": ExtractorSpec(
        "$.result.routes",
        allowed_tool_names=frozenset({"get_routes_from_start_to_destination"}),
    ),
    "local_charging_time_result": ExtractorSpec(
        "$.result",
        allowed_tool_names=frozenset({"calculate_charging_time_by_soc"}),
    ),
    "local_charging_second_segment_routes": ExtractorSpec(
        "$.result.routes",
        allowed_tool_names=frozenset({"get_routes_from_start_to_destination"}),
    ),
    "active_route_charging_status": ExtractorSpec(
        "$.result",
        allowed_tool_names=frozenset({"get_charging_specs_and_status"}),
    ),
    "active_route_charging_navigation": ExtractorSpec(
        "$.result",
        allowed_tool_names=frozenset({"get_current_navigation_state"}),
    ),
    "active_route_buffer_distance": ExtractorSpec(
        "$.result",
        allowed_tool_names=frozenset({"get_distance_by_soc"}),
    ),
    "active_route_charging_station_candidates": ExtractorSpec(
        "$.result",
        allowed_tool_names=frozenset({"search_poi_along_the_route"}),
    ),
    "relative_climate_temperature_snapshot": ExtractorSpec(
        "$.result",
        transform=_cabin_temperature_snapshot,
        allowed_tool_names=frozenset({"get_temperature_inside_car"}),
    ),
    "contextual_climate_temperature_snapshot": ExtractorSpec(
        "$.result",
        transform=_cabin_temperature_snapshot,
        allowed_tool_names=frozenset({"get_temperature_inside_car"}),
    ),
    "relative_seat_heating_snapshot": ExtractorSpec(
        "$.result",
        transform=_front_seat_heating_snapshot,
        allowed_tool_names=frozenset({"get_seat_heating_level"}),
    ),
    "relative_seat_occupancy_snapshot": ExtractorSpec(
        "$.result.seats_occupied",
        transform=_seat_occupancy_snapshot,
        allowed_tool_names=frozenset({"get_seats_occupancy"}),
    ),
    "current_navigation_state": "$.result",
    "navigation_active": "$.result.navigation_active",
    "waypoints": "$.result.waypoints_id",
    "waypoints_id": "$.result.waypoints_id",
    "route_segments": "$.result.details.routes",
    "contact_information": "$.result",
    "current_state_of_charge": "$.result.state_of_charge",
    "charging_station_candidates": "$.result",
    "fog_lights": "$.result.fog_lights",
    "head_lights_low_beams": "$.result.head_lights_low_beams",
    "head_lights_high_beams": "$.result.head_lights_high_beams",
    "broad_headlights_snapshot": ExtractorSpec(
        "$.result",
        transform=_exterior_lights_snapshot,
        allowed_tool_names=frozenset({"get_exterior_lights_status"}),
    ),
    "climate_temperature_driver": ExtractorSpec(
        "$.result.climate_temperature_driver",
        allowed_tool_names=frozenset({"get_temperature_inside_car"}),
    ),
    "climate_temperature_passenger": ExtractorSpec(
        "$.result.climate_temperature_passenger",
        allowed_tool_names=frozenset({"get_temperature_inside_car"}),
    ),
    "preferred_sunroof_percentage": ExtractorSpec(
        "$.result.vehicle_settings.vehicle_settings",
        transform=_sunroof_preference_percentage,
        allowed_tool_names=frozenset({"get_user_preferences"}),
    ),
    "preferred_air_circulation_mode": ExtractorSpec(
        "$.result.vehicle_settings.climate_control",
        transform=_air_circulation_preference,
        allowed_tool_names=frozenset({"get_user_preferences"}),
    ),
    "preferred_fan_speed_level": ExtractorSpec(
        "$.result.vehicle_settings.climate_control",
        transform=_fan_speed_preference,
        allowed_tool_names=frozenset({"get_user_preferences"}),
    ),
    "preferred_steering_wheel_heating_level": ExtractorSpec(
        "$.result.vehicle_settings.vehicle_settings",
        transform=_steering_wheel_heating_preference,
        allowed_tool_names=frozenset({"get_user_preferences"}),
    ),
    "preferred_climate_temperature": ExtractorSpec(
        "$.result.vehicle_settings.climate_control",
        transform=_climate_temperature_preference,
        allowed_tool_names=frozenset({"get_user_preferences"}),
    ),
}


POLICY_READ_TO_TOOL: dict[str, str] = {
    "vehicle.roof_state.read": "get_sunroof_and_sunshade_position",
    "vehicle.climate_state.read": "get_climate_settings",
    "vehicle.cabin_temperatures.read": "get_temperature_inside_car",
    "vehicle.window_state.read": "get_vehicle_window_positions",
    "vehicle.exterior_lights.read": "get_exterior_lights_status",
    "navigation.state.read": "get_current_navigation_state",
    "weather.current.read": "get_weather",
}


def make_result_validator() -> SemanticResultValidator:
    return SemanticResultValidator(RESULT_EXTRACTORS)


def register_recipe_needs(
    store: EvidenceStore,
    goal: Goal,
    recipes: Iterable[OperationRecipe],
    *,
    excluded_read_operations: frozenset[str] = frozenset(),
) -> dict[str, list[str]]:
    """Register only needs declared by recipes relevant to ``goal``."""

    by_read_operation: dict[str, list[str]] = {}
    for recipe in recipes:
        if not recipe.matches_semantic_operation(goal.semantic_operation):
            continue
        for declared in recipe.evidence_needs:
            if declared.read_operation in excluded_read_operations:
                continue
            need = EvidenceNeed(
                proposition=declared.proposition,
                required_for_goal_id=goal.goal_id,
                acceptable_sources=set(declared.acceptable_sources),
                freshness="current_state",
                required_before_set=declared.required_before_set,
            )
            store.register_need(need)
            if declared.read_operation and need.need_id:
                by_read_operation.setdefault(declared.read_operation, []).append(
                    need.need_id
                )
    return by_read_operation


def register_policy_read_needs(
    store: EvidenceStore,
    *,
    goal_id: str,
    required_facts: Iterable[str],
) -> list[str]:
    need_ids: list[str] = []
    for fact in required_facts:
        need = EvidenceNeed(
            proposition=fact,
            required_for_goal_id=goal_id,
            acceptable_sources={"tool_result"},
            freshness="current_state",
        )
        store.register_need(need)
        if need.need_id:
            need_ids.append(need.need_id)
    return need_ids


def register_preference_need(
    store: EvidenceStore, *, goal_id: str, proposition: str
) -> str | None:
    need = EvidenceNeed(
        proposition=proposition,
        required_for_goal_id=goal_id,
        acceptable_sources={"tool_result"},
        freshness="current_state",
    )
    store.register_need(need)
    return need.need_id


def known_facts(store: EvidenceStore) -> dict[str, Any]:
    """Expose only unique, currently known propositions to policy rules."""

    facts: dict[str, Any] = {}
    propositions = {item.proposition for item in store.evidence.values()}
    for proposition in propositions:
        observations = store.latest_for_proposition(
            proposition, current_state_only=True
        )
        if not observations:
            continue
        latest = observations[-1]
        if latest.status is not EvidenceStatus.KNOWN:
            continue
        known_values = [
            item.value for item in observations if item.status is EvidenceStatus.KNOWN
        ]
        if known_values and all(value == known_values[-1] for value in known_values):
            facts[proposition] = known_values[-1]
    return facts


def materialize_recipe_derivations(
    store: EvidenceStore, *, source_turn_id: str
) -> list[Evidence]:
    """Create only registered pure facts from current, known observations."""

    created: list[Evidence] = []
    weather = store.latest_for_proposition("weather_condition", current_state_only=True)
    if weather and all(item.status is EvidenceStatus.KNOWN for item in weather):
        canonical = {str(item.value).strip().casefold() for item in weather}
        existing = store.latest_for_proposition(
            "weather_allows_sunroof", current_state_only=True
        )
        if len(canonical) == 1 and not existing:
            parent = weather[-1]
            assert parent.evidence_id is not None
            observation = Evidence(
                proposition="weather_allows_sunroof",
                value=next(iter(canonical)) in {"sunny", "cloudy", "partly_cloudy"},
                status=EvidenceStatus.KNOWN,
                source_kind=EvidenceSourceKind.DERIVED,
                source_turn_id=source_turn_id,
                confidence=parent.confidence,
                derived_from=[parent.evidence_id],
                derivation="weather_allows_sunroof_v1",
            )
            store.add(observation)
            created.append(
                store.latest_for_proposition(
                    "weather_allows_sunroof", current_state_only=True
                )[-1]
            )
    return created
