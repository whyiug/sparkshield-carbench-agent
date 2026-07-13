from __future__ import annotations

from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
)


def test_direct_operation_contracts_expose_their_own_semantic_parameters() -> None:
    runtime = CARGuardOrchestrator(AgentConfig(llm="test/model"))
    contracts = {
        contract["primary_operation"]: contract
        for contract in runtime.intent_semantic_contracts
    }

    assert contracts["set_ambient_lights"]["required_semantic_parameters"] == [
        "enabled",
        "color",
    ]
    assert contracts["set_fan_airflow_direction"][
        "required_semantic_parameters"
    ] == ["direction"]
    assert contracts["set_air_circulation"]["required_semantic_parameters"] == [
        "mode"
    ]
