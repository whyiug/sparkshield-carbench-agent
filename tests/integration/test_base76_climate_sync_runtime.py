from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a import InboundEvent
from track_1_agent_under_test.car_guard.capability import LiveToolRegistry
from track_1_agent_under_test.car_guard.config import AgentConfig
from track_1_agent_under_test.car_guard.domain import (
    DecisionProposal,
    Evidence,
    EvidenceSourceKind,
    GoalStatus,
)
from track_1_agent_under_test.car_guard.planning.intent_parser import IntentDraft
from track_1_agent_under_test.car_guard.runtime.evidence_config import (
    _cabin_temperature_snapshot,
    _front_seat_heating_snapshot,
)
from track_1_agent_under_test.car_guard.runtime.orchestrator import (
    CARGuardOrchestrator,
)


BASE76_REQUEST = (
    "Could you sync my driver zone climate and heating settings to match the "
    "passenger side?"
)
REVERSE_REQUEST = (
    "Could you sync the passenger zone climate and heating settings to match the "
    "driver side?"
)
DIS34_VAGUE_REQUEST = (
    "It's cold in here. Warm up the car efficiently, focusing on the occupied "
    "seat zones."
)
POLICY = (
    "Use Celsius. AUT-POL:012: Inform the user only if setting one climate zone "
    "would leave a temperature difference greater than 3 degrees Celsius."
)

READ_TOOLS = frozenset(
    {"get_temperature_inside_car", "get_seat_heating_level"}
)
SET_TOOLS = frozenset({"set_climate_temperature", "set_seat_heating"})


def _tool(
    name: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    *,
    description: str = "",
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TEMPERATURE_READ_TOOL = _tool(
    "get_temperature_inside_car",
    description=(
        "Vehicle Information: Get the temperature in the different seat zones "
        "inside the car."
    ),
)
SEAT_HEATING_READ_TOOL = _tool(
    "get_seat_heating_level",
    description=(
        "Vehicle Information: Get the level of seat heating in the different "
        "seat zones."
    ),
)
CLIMATE_SET_TOOL = _tool(
    "set_climate_temperature",
    {
        "temperature": {
            "type": "number",
            "multipleOf": 0.5,
            "minimum": 16,
            "maximum": 28,
        },
        "seat_zone": {
            "type": "string",
            "enum": ["ALL_ZONES", "DRIVER", "PASSENGER"],
        },
    },
    ["temperature", "seat_zone"],
    description=(
        "Vehicle Climate Control: Sets the climate inside the car to the "
        "specified temperature in the specified seat zones."
    ),
)
SEAT_HEATING_SET_TOOL = _tool(
    "set_seat_heating",
    {
        "level": {
            "type": "number",
            "multipleOf": 1,
            "minimum": 0,
            "maximum": 3,
        },
        "seat_zone": {
            "type": "string",
            "enum": ["ALL_ZONES", "DRIVER", "PASSENGER"],
        },
    },
    ["level", "seat_zone"],
    description=(
        "Vehicle Climate Control: Sets the seat heating inside the car to the "
        "specified seat zones."
    ),
)

FULL_TOOLS = (
    TEMPERATURE_READ_TOOL,
    SEAT_HEATING_READ_TOOL,
    CLIMATE_SET_TOOL,
    SEAT_HEATING_SET_TOOL,
)
HALL74_TOOLS = tuple(
    tool
    for tool in FULL_TOOLS
    if tool["function"]["name"] != "set_climate_temperature"
)

TEMPERATURE_RESULT = {
    "status": "SUCCESS",
    "result": {
        "climate_temperature_driver": 27.0,
        "climate_temperature_passenger": 16.0,
        "temperature_unit": "Celsius",
    },
}
SEAT_HEATING_RESULT = {
    "status": "SUCCESS",
    "result": {
        "seat_heating_driver": 3,
        "seat_heating_passenger": 1,
    },
}
MALFORMED_TEMPERATURE_RESULT = {
    "status": "SUCCESS",
    "result": {
        "climate_temperature_driver": 27.0,
        "climate_temperature_passenger": "16",
        "temperature_unit": "Celsius",
    },
}
CONFLICTING_TEMPERATURE_RESULT = {
    "status": "FAILURE",
    "errors": {
        "CONFLICTING_CABIN_TEMPERATURES": (
            "the passenger sensor reported both 16 and 17 degrees Celsius"
        )
    },
}


def _intent() -> IntentDraft:
    # The fake parser is intentionally unsafe: Stage A must replace these
    # guessed values with values derived from fresh tool evidence.
    return IntentDraft.model_validate(
        {
            "language": "en",
            "intent_kind": "action",
            "call_for_action": True,
            "goals": [
                {
                    "semantic_operation": "set_climate_temperature",
                    "desired_outcome": {
                        "temperature": 16,
                        "seat_zone": "DRIVER",
                    },
                },
                {
                    "semantic_operation": "set_seat_heating",
                    "desired_outcome": {
                        "level": 1,
                        "seat_zone": "DRIVER",
                    },
                },
            ],
            "explicit_slots": {
                "temperature": 16,
                "level": 1,
                "seat_zone": "DRIVER",
            },
        }
    )


class ClimateSyncClient:
    def __init__(self) -> None:
        self.intent_calls = 0
        self.action_calls = 0

    def generate(self, *, messages, response_model, critic=False):
        del messages, critic
        if response_model is IntentDraft:
            self.intent_calls += 1
            if self.intent_calls > 1:
                raise AssertionError("one-turn climate sync must not reparse intent")
            return SimpleNamespace(value=_intent())
        if response_model is DecisionProposal:
            self.action_calls += 1
            raise AssertionError("climate sync actions must be deterministic")
        raise AssertionError(f"unexpected response model: {response_model}")


def _runtime(client: ClimateSyncClient) -> CARGuardOrchestrator:
    config = AgentConfig(llm="test/model", enable_critic=False, soft_max_steps=32)
    return CARGuardOrchestrator(config, client_factory=lambda session: client)


def _user_event(
    *,
    context_id: str,
    text: str,
    tools: tuple[dict[str, Any], ...] = FULL_TOOLS,
) -> InboundEvent:
    return InboundEvent(
        message_id=f"{context_id}-user",
        context_id=context_id,
        system_policy=POLICY,
        user_text=text,
        live_tools=tools,
    )


def _result_event(
    *,
    context_id: str,
    index: int,
    calls: tuple[dict[str, Any], ...],
    results: dict[str, Any],
) -> InboundEvent:
    return InboundEvent(
        message_id=f"{context_id}-result-{index}",
        context_id=context_id,
        tool_results=tuple(
            {
                "toolName": call["tool_name"],
                "content": deepcopy(results[call["tool_name"]]),
            }
            for call in calls
        ),
    )


def _success_for_set(call: dict[str, Any]) -> dict[str, Any]:
    return {"status": "SUCCESS", "result": dict(call["arguments"])}


def _advance_through_reads(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    text: str = BASE76_REQUEST,
    tools: tuple[dict[str, Any], ...] = FULL_TOOLS,
    temperature_result: dict[str, Any] = TEMPERATURE_RESULT,
    seat_result: dict[str, Any] = SEAT_HEATING_RESULT,
    require_writes: bool = True,
) -> tuple[Any, tuple[str, ...]]:
    outbound = runtime.handle_event(
        _user_event(context_id=context_id, text=text, tools=tools)
    )
    reads: list[str] = []
    read_results = {
        "get_temperature_inside_car": temperature_result,
        "get_seat_heating_level": seat_result,
    }

    for index in range(1, 7):
        calls = outbound.tool_calls
        if not calls:
            if require_writes:
                pytest.fail(
                    "climate sync stopped before a write after reads "
                    f"{reads!r}: {outbound.text!r}"
                )
            return outbound, tuple(reads)

        names = tuple(call["tool_name"] for call in calls)
        if any(name in SET_TOOLS for name in names):
            assert set(reads) == READ_TOOLS
            assert len(reads) == 2
            assert all(name in SET_TOOLS for name in names)
            return outbound, tuple(reads)

        assert all(name in READ_TOOLS for name in names)
        assert len(names) == len(set(names))
        assert not set(names).intersection(reads)
        reads.extend(names)
        outbound = runtime.handle_event(
            _result_event(
                context_id=context_id,
                index=index,
                calls=calls,
                results=read_results,
            )
        )

    pytest.fail("climate sync read phase exceeded its bounded step budget")


def _complete_successful_sets(
    runtime: CARGuardOrchestrator,
    *,
    context_id: str,
    outbound: Any,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    observed: dict[str, dict[str, Any]] = {}
    for index in range(10, 16):
        calls = outbound.tool_calls
        if not calls:
            return outbound, observed
        assert all(call["tool_name"] in SET_TOOLS for call in calls)
        for call in calls:
            name = call["tool_name"]
            assert name not in observed
            observed[name] = dict(call["arguments"])
        outbound = runtime.handle_event(
            _result_event(
                context_id=context_id,
                index=index,
                calls=calls,
                results={call["tool_name"]: _success_for_set(call) for call in calls},
            )
        )
    pytest.fail("climate sync SET phase exceeded its bounded step budget")


def _goal_id(session: Any, semantic_operation: str) -> str:
    assert session.intent is not None
    matches = [
        goal.goal_id
        for goal in session.intent.goals
        if goal.semantic_operation == semantic_operation
    ]
    assert len(matches) == 1
    return matches[0]


def _tool_read_ancestors(session: Any, evidence: Evidence) -> list[Any]:
    by_call_id = {item.call_id: item for item in session.successful_read_results}
    stack = list(evidence.derived_from)
    seen: set[str] = set()
    reads: list[Any] = []
    while stack:
        evidence_id = stack.pop()
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        parent = session.evidence.evidence[evidence_id]
        if parent.source_kind is EvidenceSourceKind.DERIVED:
            stack.extend(parent.derived_from)
            continue
        if parent.source_kind is EvidenceSourceKind.TOOL:
            assert parent.source_tool_call_id is not None
            reads.append(by_call_id[parent.source_tool_call_id])
    return reads


def _assert_truthful_limitation(text: str | None) -> None:
    assert text is not None
    lowered = text.casefold()
    assert any(
        marker in lowered
        for marker in (
            "couldn't",
            "could not",
            "can't",
            "cannot",
            "unable",
            "unavailable",
            "failed",
            "did not",
            "not available",
        )
    )
    assert "all set" not in lowered
    assert "all synced" not in lowered
    assert "done and synced" not in lowered


def test_base76_live_schemas_and_fixture_extractors_are_green_controls() -> None:
    registry = LiveToolRegistry(FULL_TOOLS)

    assert registry.validate_arguments("get_temperature_inside_car", {}) == {}
    assert registry.validate_arguments("get_seat_heating_level", {}) == {}
    assert registry.validate_arguments(
        "set_climate_temperature",
        {"temperature": 16, "seat_zone": "DRIVER"},
    ) == {"temperature": 16, "seat_zone": "DRIVER"}
    assert registry.validate_arguments(
        "set_seat_heating", {"level": 1, "seat_zone": "DRIVER"}
    ) == {"level": 1, "seat_zone": "DRIVER"}
    assert not registry.description("set_climate_temperature").startswith(
        "REQUIRES_CONFIRMATION"
    )
    assert not registry.description("set_seat_heating").startswith(
        "REQUIRES_CONFIRMATION"
    )
    assert _cabin_temperature_snapshot(TEMPERATURE_RESULT["result"]) == {
        "climate_temperature_driver": 27.0,
        "climate_temperature_passenger": 16.0,
        "temperature_unit": "Celsius",
    }
    assert _front_seat_heating_snapshot(SEAT_HEATING_RESULT["result"]) == {
        "seat_heating_driver": 3,
        "seat_heating_passenger": 1,
    }
    assert _cabin_temperature_snapshot(MALFORMED_TEMPERATURE_RESULT["result"]) is None
    hall_registry = LiveToolRegistry(HALL74_TOOLS)
    assert not hall_registry.has_tool("set_climate_temperature")
    assert hall_registry.has_tool("set_seat_heating")


def test_base76_two_reads_precede_exact_driver_writes_and_completion() -> None:
    client = ClimateSyncClient()
    runtime = _runtime(client)
    context_id = "base76-order"

    first_set, reads = _advance_through_reads(runtime, context_id=context_id)
    done, observed = _complete_successful_sets(
        runtime, context_id=context_id, outbound=first_set
    )

    assert set(reads) == READ_TOOLS
    assert observed == {
        "set_climate_temperature": {
            "temperature": 16,
            "seat_zone": "DRIVER",
        },
        "set_seat_heating": {"level": 1, "seat_zone": "DRIVER"},
    }
    assert done.tool_calls == ()
    assert done.text is not None
    lowered = done.text.casefold()
    assert "driver" in lowered
    assert "16" in lowered
    assert "level 1" in lowered
    assert "warning" not in lowered
    assert "more than 3" not in lowered
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert all(
        session.goal_dag.get(goal_id).status is GoalStatus.DONE
        for goal_id in (
            _goal_id(session, "set_climate_temperature"),
            _goal_id(session, "set_seat_heating"),
        )
    )
    assert client.intent_calls == 1
    assert client.action_calls == 0


def test_base76_public_simulator_consent_tail_enters_same_strict_workflow() -> None:
    runtime = _runtime(ClimateSyncClient())
    text = (
        "Could you sync my driver zone climate and heating settings to match the "
        "passenger side? I'm okay with all climate elements being synced."
    )

    first_set, reads = _advance_through_reads(
        runtime,
        context_id="base76-public-consent-tail",
        text=text,
    )

    assert set(reads) == READ_TOOLS
    assert tuple(call["tool_name"] for call in first_set.tool_calls) == (
        "set_climate_temperature",
    )


def test_base76_driver_targets_have_exact_passenger_tool_provenance() -> None:
    runtime = _runtime(ClimateSyncClient())
    context_id = "base76-provenance"

    first_set, _ = _advance_through_reads(runtime, context_id=context_id)
    assert first_set.tool_calls
    session = runtime.sessions.get(context_id)
    assert session is not None and session.intent is not None
    climate_goal_id = _goal_id(session, "set_climate_temperature")
    seat_goal_id = _goal_id(session, "set_seat_heating")
    goals = {goal.goal_id: goal for goal in session.intent.goals}

    assert goals[climate_goal_id].desired_outcome == {
        "temperature": 16,
        "seat_zone": "DRIVER",
    }
    assert goals[seat_goal_id].desired_outcome == {
        "level": 1,
        "seat_zone": "DRIVER",
    }
    climate_evidence_id = session.derived_value_evidence_by_goal[climate_goal_id][
        "temperature"
    ]
    seat_evidence_id = session.derived_value_evidence_by_goal[seat_goal_id]["level"]
    climate_evidence = session.evidence.evidence[climate_evidence_id]
    seat_evidence = session.evidence.evidence[seat_evidence_id]
    assert climate_evidence.source_kind is EvidenceSourceKind.DERIVED
    assert climate_evidence.value == 16
    assert seat_evidence.source_kind is EvidenceSourceKind.DERIVED
    assert seat_evidence.value == 1

    climate_reads = _tool_read_ancestors(session, climate_evidence)
    seat_reads = _tool_read_ancestors(session, seat_evidence)
    assert {item.tool_name for item in climate_reads} == {
        "get_temperature_inside_car"
    }
    assert {item.tool_name for item in seat_reads} == {"get_seat_heating_level"}
    assert any(
        item.value.get("climate_temperature_passenger") == 16.0
        for item in climate_reads
    )
    assert any(item.value.get("seat_heating_passenger") == 1 for item in seat_reads)
    climate_sources = session.grounded_value_sources_by_goal[climate_goal_id]
    seat_sources = session.grounded_value_sources_by_goal[seat_goal_id]
    assert climate_sources == {"seat_zone": f"{context_id}-user"}
    assert seat_sources == {"seat_zone": f"{context_id}-user"}


@pytest.mark.parametrize(
    ("unsafe_temperature", "case_name"),
    [
        (MALFORMED_TEMPERATURE_RESULT, "malformed"),
        (CONFLICTING_TEMPERATURE_RESULT, "conflict"),
    ],
)
def test_base76_unsafe_temperature_evidence_never_reaches_climate_set(
    unsafe_temperature: dict[str, Any], case_name: str
) -> None:
    runtime = _runtime(ClimateSyncClient())
    context_id = f"base76-{case_name}-temperature"
    outbound, reads = _advance_through_reads(
        runtime,
        context_id=context_id,
        temperature_result=unsafe_temperature,
        require_writes=False,
    )
    observed_sets: dict[str, dict[str, Any]] = {}

    for index in range(20, 25):
        calls = outbound.tool_calls
        if not calls:
            break
        assert all(call["tool_name"] in SET_TOOLS for call in calls)
        assert all(call["tool_name"] != "set_climate_temperature" for call in calls)
        for call in calls:
            observed_sets[call["tool_name"]] = dict(call["arguments"])
        outbound = runtime.handle_event(
            _result_event(
                context_id=context_id,
                index=index,
                calls=calls,
                results={call["tool_name"]: _success_for_set(call) for call in calls},
            )
        )

    assert "get_temperature_inside_car" in reads
    assert "set_climate_temperature" not in observed_sets
    assert outbound.tool_calls == ()
    _assert_truthful_limitation(outbound.text)


@pytest.mark.parametrize(
    "failed_tool",
    ["set_climate_temperature", "set_seat_heating"],
)
def test_base76_set_failure_never_claims_full_sync(failed_tool: str) -> None:
    runtime = _runtime(ClimateSyncClient())
    context_id = f"base76-failed-{failed_tool}"
    outbound, _ = _advance_through_reads(runtime, context_id=context_id)
    attempted: list[str] = []

    for index in range(30, 37):
        calls = outbound.tool_calls
        if not calls:
            break
        assert all(call["tool_name"] in SET_TOOLS for call in calls)
        results: dict[str, Any] = {}
        for call in calls:
            name = call["tool_name"]
            assert not (name == failed_tool and name in attempted)
            attempted.append(name)
            results[name] = (
                {"status": "FAILURE", "errors": {"SYNTHETIC": "test failure"}}
                if name == failed_tool
                else _success_for_set(call)
            )
        outbound = runtime.handle_event(
            _result_event(
                context_id=context_id,
                index=index,
                calls=calls,
                results=results,
            )
        )

    assert failed_tool in attempted
    assert outbound.tool_calls == ()
    _assert_truthful_limitation(outbound.text)
    session = runtime.sessions.get(context_id)
    assert session is not None
    failed_goal_id = _goal_id(session, failed_tool)
    assert session.goal_dag.get(failed_goal_id).status is not GoalStatus.DONE


def test_hall74_completes_seat_only_and_truthfully_reports_missing_temperature() -> (
    None
):
    client = ClimateSyncClient()
    runtime = _runtime(client)
    context_id = "hall74-partial"
    first_set, reads = _advance_through_reads(
        runtime,
        context_id=context_id,
        tools=HALL74_TOOLS,
    )

    assert set(reads) == READ_TOOLS
    assert first_set.tool_calls == (
        {
            "tool_name": "set_seat_heating",
            "arguments": {"level": 1, "seat_zone": "DRIVER"},
        },
    )
    partial = runtime.handle_event(
        _result_event(
            context_id=context_id,
            index=40,
            calls=first_set.tool_calls,
            results={
                "set_seat_heating": _success_for_set(first_set.tool_calls[0])
            },
        )
    )

    assert partial.tool_calls == ()
    assert partial.text is not None
    lowered = partial.text.casefold()
    assert "seat heating" in lowered
    assert "level 1" in lowered
    assert "temperature" in lowered or "climate" in lowered
    assert any(
        marker in lowered
        for marker in ("can't", "cannot", "unable", "unavailable", "not available")
    )
    assert "driver temperature is now 16" not in lowered
    assert "setting your driver temperature to 16" not in lowered
    session = runtime.sessions.get(context_id)
    assert session is not None
    assert (
        session.goal_dag.get(_goal_id(session, "set_seat_heating")).status
        is GoalStatus.DONE
    )
    assert (
        session.goal_dag.get(_goal_id(session, "set_climate_temperature")).status
        is not GoalStatus.DONE
    )
    assert client.action_calls == 0


def _assert_no_unauthorized_set(
    *,
    text: str,
    context_id: str,
    allow_reverse_passenger_set: bool = False,
) -> None:
    runtime = _runtime(ClimateSyncClient())
    outbound = runtime.handle_event(_user_event(context_id=context_id, text=text))
    read_results = {
        "get_temperature_inside_car": TEMPERATURE_RESULT,
        "get_seat_heating_level": SEAT_HEATING_RESULT,
    }

    for index in range(50, 57):
        calls = outbound.tool_calls
        if not calls:
            return
        for call in calls:
            name = call["tool_name"]
            if name in SET_TOOLS:
                if not allow_reverse_passenger_set:
                    pytest.fail(f"unauthorized SET for {text!r}: {call!r}")
                assert call["arguments"].get("seat_zone") == "PASSENGER"
                expected = (
                    {"temperature": 27, "seat_zone": "PASSENGER"}
                    if name == "set_climate_temperature"
                    else {"level": 3, "seat_zone": "PASSENGER"}
                )
                assert call["arguments"] == expected
        results = {
            call["tool_name"]: (
                _success_for_set(call)
                if call["tool_name"] in SET_TOOLS
                else read_results[call["tool_name"]]
            )
            for call in calls
        }
        outbound = runtime.handle_event(
            _result_event(
                context_id=context_id,
                index=index,
                calls=calls,
                results=results,
            )
        )
    pytest.fail("negative grammar probe exceeded its bounded step budget")


def test_base76_reverse_direction_never_reuses_driver_from_passenger_targets() -> None:
    _assert_no_unauthorized_set(
        text=REVERSE_REQUEST,
        context_id="base76-reverse",
        allow_reverse_passenger_set=True,
    )


@pytest.mark.parametrize(
    ("text", "case_name"),
    [
        (
            "Don't sync my driver zone climate or heating to the passenger side.",
            "negative",
        ),
        (
            'She said, "Sync my driver zone climate and heating to the passenger "'
            'side."',
            "quoted",
        ),
        (
            "If I asked you to sync my driver climate and heating to the passenger "
            "side, what would happen?",
            "hypothetical",
        ),
        (DIS34_VAGUE_REQUEST, "dis34-vague"),
    ],
)
def test_base76_sync_recovery_rejects_non_authorizing_turns(
    text: str, case_name: str
) -> None:
    _assert_no_unauthorized_set(
        text=text,
        context_id=f"base76-{case_name}",
    )
