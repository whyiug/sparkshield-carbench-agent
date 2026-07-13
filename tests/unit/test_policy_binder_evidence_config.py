from track_1_agent_under_test.car_guard.capability import (
    LiveToolRegistry,
    PolicyCapabilityBinder,
)
from track_1_agent_under_test.car_guard.domain import (
    Evidence,
    EvidenceSourceKind,
    EvidenceStatus,
    EvidenceStore,
)
from track_1_agent_under_test.car_guard.policy import SemanticPolicyCall
from track_1_agent_under_test.car_guard.runtime.evidence_config import (
    known_facts,
    materialize_recipe_derivations,
)


def tool(name: str, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "current control",
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


def test_policy_binding_checks_only_triggered_current_operation() -> None:
    registry = LiveToolRegistry(
        [
            tool(
                "set_fan_speed",
                {"level": {"type": "integer", "minimum": 0, "maximum": 5}},
                ["level"],
            )
        ]
    )
    requirement = SemanticPolicyCall(
        semantic_operation="vehicle.fan.set_speed", arguments={"level": 2}
    )

    result = PolicyCapabilityBinder().bind(requirement, registry)

    assert result.call is not None
    assert result.call.tool_name == "set_fan_speed"


def test_latest_unknown_fact_is_not_exposed_as_known() -> None:
    store = EvidenceStore()
    store.add(
        Evidence(
            proposition="fan_speed",
            value=1,
            status=EvidenceStatus.KNOWN,
            source_kind=EvidenceSourceKind.TOOL,
            source_turn_id="one",
            source_tool_call_id="call-one",
            confidence=1,
        )
    )
    store.add(
        Evidence(
            proposition="fan_speed",
            value=None,
            status=EvidenceStatus.UNKNOWN,
            source_kind=EvidenceSourceKind.TOOL,
            source_turn_id="two",
            source_tool_call_id="call-two",
            confidence=0,
        )
    )

    assert "fan_speed" not in known_facts(store)


def test_weather_permission_is_derived_with_provenance_and_becomes_stale() -> None:
    store = EvidenceStore()
    store.add(
        Evidence(
            proposition="weather_condition",
            value="sunny",
            status=EvidenceStatus.KNOWN,
            source_kind=EvidenceSourceKind.TOOL,
            source_turn_id="weather-result",
            source_tool_call_id="weather-call",
            confidence=1,
        )
    )

    created = materialize_recipe_derivations(store, source_turn_id="derived-turn")

    assert len(created) == 1
    assert created[0].value is True
    assert created[0].derivation == "weather_allows_sunroof_v1"
    assert created[0].derived_from
    store.invalidate_tool_state()
    assert not store.latest_for_proposition(
        "weather_allows_sunroof", current_state_only=True
    )
