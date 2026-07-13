"""Bind policy-triggered semantic operations to the current live schema."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain import GateOutcome, OfficialToolCall
from ..policy import SemanticPolicyCall
from .live_tool_registry import LiveToolRegistry, ToolArgumentValidationError


POLICY_OPERATION_TOOLS: dict[str, tuple[str, ...]] = {
    "vehicle.roof_state.read": ("get_sunroof_and_sunshade_position",),
    "vehicle.sunshade.set_position": ("open_close_sunshade",),
    "vehicle.climate_state.read": ("get_climate_settings",),
    "vehicle.cabin_temperatures.read": ("get_temperature_inside_car",),
    "vehicle.window_state.read": ("get_vehicle_window_positions",),
    "vehicle.exterior_lights.read": ("get_exterior_lights_status",),
    "navigation.state.read": ("get_current_navigation_state",),
    "weather.current.read": ("get_weather",),
    "vehicle.window.set_position": ("open_close_window",),
    "vehicle.fan.set_speed": ("set_fan_speed",),
    "vehicle.fan.set_airflow": ("set_fan_airflow_direction",),
    "vehicle.ac.set": ("set_air_conditioning",),
    "vehicle.low_beam.set": ("set_head_lights_low_beams",),
    "vehicle.high_beam.set": ("set_head_lights_high_beams",),
    "vehicle.fog_lights.set": ("set_fog_lights",),
}


@dataclass(frozen=True, slots=True)
class PolicyBindingResult:
    outcome: GateOutcome
    call: OfficialToolCall | None = None
    reason: str | None = None


class PolicyCapabilityBinder:
    def bind(
        self, requirement: SemanticPolicyCall, live_tools: LiveToolRegistry
    ) -> PolicyBindingResult:
        candidates = POLICY_OPERATION_TOOLS.get(requirement.semantic_operation, ())
        for tool_name in candidates:
            if not live_tools.has_tool(tool_name):
                continue
            try:
                arguments = live_tools.validate_arguments(
                    tool_name, requirement.arguments
                )
            except ToolArgumentValidationError as exc:
                return PolicyBindingResult(
                    outcome=GateOutcome.UNSUPPORTED_PARAMETER,
                    reason=str(exc),
                )
            return PolicyBindingResult(
                outcome=GateOutcome.ALLOW,
                call=OfficialToolCall(tool_name=tool_name, arguments=arguments),
            )
        return PolicyBindingResult(
            outcome=GateOutcome.UNSUPPORTED_CAPABILITY,
            reason=(
                "A policy-required operation cannot be expressed with the current "
                "controls."
            ),
        )
