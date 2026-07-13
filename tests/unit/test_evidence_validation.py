from __future__ import annotations

from typing import Any

import pytest

from track_1_agent_under_test.car_guard.a2a.result_matching import (
    MatchedToolResult,
    PendingToolCall,
)
from track_1_agent_under_test.car_guard.capability.semantic_result_validator import (
    ExtractorSpec,
    ResultExecutionStatus,
    SemanticResultValidator,
)
from track_1_agent_under_test.car_guard.domain import (
    Evidence,
    EvidenceNeed,
    EvidenceSourceKind,
    EvidenceStatus,
    EvidenceStore,
)
from track_1_agent_under_test.car_guard.planning.deterministic_derivations import (
    DeterministicDerivationRegistry,
    InsufficientDerivationEvidence,
)
from track_1_agent_under_test.car_guard.runtime.evidence_config import (
    make_result_validator,
)


def _need(need_id: str, proposition: str) -> EvidenceNeed:
    return EvidenceNeed(
        need_id=need_id,
        proposition=proposition,
        required_for_goal_id="goal-1",
        acceptable_sources={"tool_result"},
    )


def _matched_result(
    content: Any,
    *need_ids: str,
    provided_call_id: str | None = None,
    tool_name: str = "get_state",
) -> MatchedToolResult:
    pending = PendingToolCall(
        call_id="call-state-1",
        tool_name=tool_name,
        arguments={},
        outbound_index=0,
        evidence_needs=tuple(need_ids),
    )
    return MatchedToolResult(
        pending_call=pending,
        tool_name=pending.tool_name,
        content=content,
        provided_call_id=provided_call_id,
        matched_by="call_id" if provided_call_id else "tool_name_order",
    )


def test_validator_extracts_only_declared_pending_needs() -> None:
    state = _need("need-state", "window state is known")
    unrelated = _need("need-diagnostic", "diagnostic state is known")
    store = EvidenceStore()
    store.register_need(state)
    store.register_need(unrelated)
    validator = SemanticResultValidator(
        {
            state.need_id: "$.data.state",
            unrelated.need_id: "$.data.diagnostic",
        }
    )

    result = validator.validate_result(
        _matched_result(
            '{"data":{"state":"closed","diagnostic":"unknown"}}',
            state.need_id,
        ),
        evidence_store=store,
        source_turn_id="turn-2",
    )

    assert result.execution_status is ResultExecutionStatus.SUCCESS
    assert result.considered_need_ids == (state.need_id,)
    assert len(result.evidence) == 1
    assert result.evidence[0].value == "closed"
    assert result.evidence[0].status is EvidenceStatus.KNOWN
    assert store.is_satisfied(state)
    assert store.for_proposition(unrelated.proposition) == []


def test_validator_does_not_reextract_a_satisfied_need() -> None:
    need = _need("need-state", "window state is known")
    store = EvidenceStore()
    store.register_need(need)
    store.add(
        Evidence(
            proposition=need.proposition,
            value="closed",
            status=EvidenceStatus.KNOWN,
            source_kind=EvidenceSourceKind.TOOL,
            source_turn_id="turn-1",
            source_tool_call_id="call-state-0",
            confidence=1.0,
        )
    )

    result = SemanticResultValidator({need.need_id: "$.state"}).validate_result(
        _matched_result({"state": "unknown"}, need.need_id),
        evidence_store=store,
        source_turn_id="turn-2",
    )

    assert result.execution_status is ResultExecutionStatus.IGNORED
    assert result.evidence == ()
    assert len(store.for_proposition(need.proposition)) == 1


@pytest.mark.parametrize(
    ("content", "expected_execution", "expected_evidence"),
    [
        (None, ResultExecutionStatus.SUCCESS, EvidenceStatus.UNKNOWN),
        ("", ResultExecutionStatus.SUCCESS, EvidenceStatus.UNKNOWN),
        ("unknown", ResultExecutionStatus.SUCCESS, EvidenceStatus.UNKNOWN),
        ("null", ResultExecutionStatus.SUCCESS, EvidenceStatus.UNKNOWN),
        ("[]", ResultExecutionStatus.SUCCESS, EvidenceStatus.UNKNOWN),
        ('{"state":', ResultExecutionStatus.MALFORMED, EvidenceStatus.ERROR),
        (
            {"success": False, "state": "open"},
            ResultExecutionStatus.NON_SUCCESS,
            EvidenceStatus.ERROR,
        ),
        (
            {"status": "failed", "state": "open"},
            ResultExecutionStatus.NON_SUCCESS,
            EvidenceStatus.ERROR,
        ),
    ],
)
def test_unavailable_forms_are_all_insufficient_evidence(
    content: Any,
    expected_execution: ResultExecutionStatus,
    expected_evidence: EvidenceStatus,
) -> None:
    need = _need("need-state", "window state is known")
    store = EvidenceStore()
    store.register_need(need)

    result = SemanticResultValidator({need.need_id: ExtractorSpec("$")}).validate_result(
        _matched_result(content, need.need_id),
        evidence_store=store,
        source_turn_id="turn-2",
    )

    assert result.execution_status is expected_execution
    assert result.evidence[0].status is expected_evidence
    assert result.evidence[0].value is None
    assert result.insufficient_need_ids == (need.need_id,)
    assert not store.is_satisfied(need)


def test_registered_json_path_and_transform_preserve_tool_call_provenance() -> None:
    need = _need("need-temp", "cabin temperature celsius")
    store = EvidenceStore()
    store.register_need(need)
    validator = SemanticResultValidator(
        {
            need.proposition: ExtractorSpec(
                '$["measurements"][0].value',
                transform=float,
            )
        }
    )

    result = validator.validate_result(
        _matched_result(
            '{"measurements":[{"value":"21.5"}]}',
            need.need_id,
            provided_call_id="call-state-1",
        ),
        evidence_store=store,
        source_turn_id="turn-7",
    )

    observation = result.evidence[0]
    assert observation.value == 21.5
    assert observation.source_tool_call_id == "call-state-1"
    assert observation.source_turn_id == "turn-7"
    assert result.insufficient_need_ids == ()


def test_missing_explicit_path_is_insufficient_without_scanning_other_fields() -> None:
    need = _need("need-state", "window state is known")
    store = EvidenceStore()
    store.register_need(need)

    result = SemanticResultValidator({need.need_id: "$.state"}).validate_result(
        _matched_result(
            {"different_state": "closed", "unknown_field": "unknown"},
            need.need_id,
        ),
        evidence_store=store,
        source_turn_id="turn-2",
    )

    assert result.evidence[0].status is EvidenceStatus.ERROR
    assert result.evidence[0].value is None
    assert [issue.code for issue in result.issues] == ["extractor_path_missing"]


@pytest.mark.parametrize(
    ("preference", "expected"),
    [
        (
            "Default value to open the sunroof is 50%, never wants to open "
            "the sunroof fully",
            50,
        ),
        ("Never wants the sunroof at 50%.", None),
        ("Default sunroof values are 25% and 50%.", None),
        ("Default window opening is 50%.", None),
        ("Default window opening is 75%; sunroof position is available.", None),
        ("Default value to open the sunroof is 50.5%.", None),
    ],
)
def test_sunroof_preference_extractor_is_target_bound_and_unambiguous(
    preference: str, expected: int | None
) -> None:
    need = _need("need-sunroof-pref", "preferred_sunroof_percentage")
    store = EvidenceStore()
    store.register_need(need)

    result = make_result_validator().validate_result(
        _matched_result(
            {
                "status": "SUCCESS",
                "result": {
                    "vehicle_settings": {"vehicle_settings": [preference]}
                },
            },
            need.need_id,
            tool_name="get_user_preferences",
        ),
        evidence_store=store,
        source_turn_id="turn-pref",
    )

    if expected is None:
        assert result.insufficient_need_ids == (need.need_id,)
        assert not store.is_satisfied(need)
    else:
        assert result.evidence[0].value == expected
        assert result.insufficient_need_ids == ()
        assert store.is_satisfied(need)


def test_sunroof_preference_extractor_rejects_same_shape_from_other_tool() -> None:
    need = _need("need-sunroof-pref", "preferred_sunroof_percentage")
    store = EvidenceStore()
    store.register_need(need)

    result = make_result_validator().validate_result(
        _matched_result(
            {
                "status": "SUCCESS",
                "result": {
                    "vehicle_settings": {
                        "vehicle_settings": [
                            "Default value to open the sunroof is 50%."
                        ]
                    }
                },
            },
            need.need_id,
            tool_name="get_state",
        ),
        evidence_store=store,
        source_turn_id="turn-wrong-tool",
    )

    assert result.insufficient_need_ids == (need.need_id,)
    assert [issue.code for issue in result.issues] == ["extractor_tool_mismatch"]
    assert not store.is_satisfied(need)


def test_result_without_registered_extractor_stays_pending() -> None:
    need = _need("need-state", "window state is known")
    store = EvidenceStore()
    store.register_need(need)

    result = SemanticResultValidator().validate_result(
        _matched_result({"state": "closed"}, need.need_id),
        evidence_store=store,
        source_turn_id="turn-2",
    )

    assert result.evidence == ()
    assert result.insufficient_need_ids == (need.need_id,)
    assert [issue.code for issue in result.issues] == ["extractor_not_registered"]
    assert store.pending_needs == [need]


def test_deterministic_derivation_records_all_input_provenance() -> None:
    temperature = Evidence(
        evidence_id="ev-temperature",
        proposition="cabin temperature celsius",
        value=24,
        status=EvidenceStatus.KNOWN,
        source_kind=EvidenceSourceKind.TOOL,
        source_turn_id="turn-2",
        source_tool_call_id="call-temperature",
        confidence=0.9,
    )
    threshold = Evidence(
        evidence_id="ev-threshold",
        proposition="configured high temperature threshold",
        value=22,
        status=EvidenceStatus.KNOWN,
        source_kind=EvidenceSourceKind.PREFERENCE,
        source_turn_id="turn-1",
        confidence=0.8,
    )
    registry = DeterministicDerivationRegistry(
        {"greater_than_v1": lambda values: values[0] > values[1]}
    )

    derived = registry.derive(
        "greater_than_v1",
        proposition="cabin is above configured threshold",
        source_turn_id="turn-2",
        inputs=[temperature, threshold],
    )

    assert derived.value is True
    assert derived.source_kind is EvidenceSourceKind.DERIVED
    assert derived.source_tool_call_id is None
    assert derived.derived_from == ["ev-temperature", "ev-threshold"]
    assert derived.derivation == "greater_than_v1"
    assert derived.confidence == 0.8


def test_derivation_refuses_unknown_or_untraceable_inputs() -> None:
    unknown = Evidence(
        proposition="current state",
        value=None,
        status=EvidenceStatus.UNKNOWN,
        source_kind=EvidenceSourceKind.TOOL,
        source_turn_id="turn-2",
        source_tool_call_id="call-state",
        confidence=0.0,
    )
    registry = DeterministicDerivationRegistry({"identity_v1": lambda values: values[0]})

    with pytest.raises(InsufficientDerivationEvidence, match="not known"):
        registry.derive(
            "identity_v1",
            proposition="derived state",
            source_turn_id="turn-2",
            inputs=[unknown],
        )
