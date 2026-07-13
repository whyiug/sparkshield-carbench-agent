import inspect
from types import SimpleNamespace

from track_1_agent_under_test.car_guard.domain import (
    Goal,
    GoalDAG,
    GoalStatus,
    IntentFrame,
    IntentKind,
)
from track_1_agent_under_test.car_guard.planning.goal_planner import GoalPlanner
from track_1_agent_under_test.car_guard.planning.intent_grounding import ground_intent
from track_1_agent_under_test.car_guard.planning.intent_parser import (
    IntentDraft,
    IntentExtractor,
)
from track_1_agent_under_test.car_guard.recipes import RecipeRegistry


class FakeClient:
    def __init__(self, draft: dict) -> None:
        self.draft = draft
        self.messages = None

    def generate(self, *, messages, response_model, critic=False):
        self.messages = messages
        assert response_model is IntentDraft
        return SimpleNamespace(value=IntentDraft.model_validate(self.draft))


class SequencedFakeClient:
    def __init__(self, drafts: list[dict]) -> None:
        self.drafts = list(drafts)
        self.calls: list[list[dict]] = []

    def generate(self, *, messages, response_model, critic=False):
        del critic
        self.calls.append(messages)
        assert response_model is IntentDraft
        return SimpleNamespace(
            value=IntentDraft.model_validate(self.drafts[len(self.calls) - 1])
        )


def action_draft() -> dict:
    return {
        "language": "en",
        "intent_kind": "action",
        "call_for_action": True,
        "goals": [
            {
                "semantic_operation": "open_trunk",
                "desired_outcome": {"position": "open"},
            },
            {
                "semantic_operation": "start_navigation",
                "desired_outcome": {"destination": "airport"},
            },
        ],
        "explicit_slots": {"destination": "airport"},
        "explicit_constraints": {},
        "ambiguities": [],
    }


def test_intent_extractor_has_no_live_inventory_parameter() -> None:
    signature = inspect.signature(IntentExtractor.extract)
    assert "live_tools" not in signature.parameters


def test_intent_prompt_contains_only_policy_and_conversation() -> None:
    client = FakeClient(action_draft())
    frame = IntentExtractor(client).extract(
        system_policy="Follow current policy.",
        conversation=[
            {
                "turn_id": "u1",
                "role": "user",
                "content": "request",
            }
        ],
    )

    assert [goal.semantic_operation for goal in frame.goals] == [
        "open_trunk",
        "start_navigation",
    ]
    serialized = str(client.messages)
    assert "live_tools" not in serialized
    assert "Follow current policy" in serialized


def test_static_semantic_contract_does_not_change_inventory_blind_api() -> None:
    client = FakeClient(action_draft())
    IntentExtractor(
        client,
        semantic_contracts=[
            {
                "primary_operation": "set_fan_speed",
                "required_semantic_parameters": ["level"],
            }
        ],
    ).extract(
        system_policy="policy",
        conversation=[{"turn_id": "u1", "role": "user", "content": "fan two"}],
    )

    serialized = str(client.messages)
    assert "set_fan_speed" in serialized
    assert "live_tools" not in serialized


def test_same_policy_and_conversation_produce_same_frame() -> None:
    first = IntentExtractor(FakeClient(action_draft())).extract(
        system_policy="policy",
        conversation=[{"turn_id": "1", "role": "user", "content": "request"}],
    )
    second = IntentExtractor(FakeClient(action_draft())).extract(
        system_policy="policy",
        conversation=[{"turn_id": "1", "role": "user", "content": "request"}],
    )
    assert first == second


def test_explicit_action_with_empty_focused_draft_falls_back_once() -> None:
    empty = {
        "language": "en",
        "intent_kind": "conversation",
        "call_for_action": False,
        "goals": [],
        "explicit_slots": {},
        "explicit_constraints": {},
        "ambiguities": [],
    }
    corrected = {
        "language": "en",
        "intent_kind": "action",
        "call_for_action": True,
        "goals": [
            {
                "semantic_operation": "set_sunroof_position",
                "desired_outcome": {"percentage": 50},
            }
        ],
        "explicit_slots": {"percentage": 50},
        "explicit_constraints": {},
        "ambiguities": [],
    }
    client = SequencedFakeClient([empty, corrected])

    intent = IntentExtractor(
        client,
        semantic_contracts=[
            {
                "primary_operation": "set_sunroof_position",
                "required_semantic_parameters": ["percentage"],
            }
        ],
    ).extract(
        system_policy="long policy",
        conversation=[
            {
                "turn_id": "u1",
                "role": "user",
                "content": "Hey, can you open the sunroof halfway?",
            }
        ],
    )

    assert len(client.calls) == 2
    assert intent.call_for_action
    assert [goal.semantic_operation for goal in intent.goals] == [
        "set_sunroof_position"
    ]
    focused_payload = str(client.calls[0])
    assert "current_user_utterance" in focused_payload
    assert "long policy" not in focused_payload
    assert "long policy" in str(client.calls[1])


def test_explicit_action_uses_only_the_successful_focused_pass() -> None:
    corrected = {
        "language": "en",
        "intent_kind": "action",
        "call_for_action": True,
        "goals": [
            {
                "semantic_operation": "set_sunroof_position",
                "desired_outcome": {"percentage": 50},
            }
        ],
        "explicit_slots": {"percentage": 50},
        "explicit_constraints": {},
        "ambiguities": [],
    }
    client = SequencedFakeClient([corrected])

    intent = IntentExtractor(client).extract(
        system_policy="long policy",
        conversation=[
            {
                "turn_id": "u1",
                "role": "user",
                "content": "Hey, can you open the sunroof halfway?",
            }
        ],
    )

    assert len(client.calls) == 1
    assert intent.goals[0].semantic_operation == "set_sunroof_position"
    serialized = str(client.calls[0])
    assert "current_user_utterance" in serialized
    assert "long policy" not in serialized


def test_focused_action_prunes_model_inferred_full_open_percentage() -> None:
    inferred = {
        "language": "en",
        "intent_kind": "action",
        "call_for_action": True,
        "goals": [
            {
                "semantic_operation": "set_sunroof_position",
                "desired_outcome": {"percentage": 100},
            }
        ],
        "explicit_slots": {"percentage": 100},
        "explicit_constraints": {},
        "ambiguities": [],
    }

    intent = IntentExtractor(FakeClient(inferred)).extract(
        system_policy="policy",
        conversation=[
            {"turn_id": "u1", "role": "user", "content": "Open the sunroof."}
        ],
    )

    assert intent.goals[0].desired_outcome == {}
    assert intent.explicit_slots == {}


def test_focused_official_conditional_draft_retains_only_unbound_roof_goal() -> None:
    draft = {
        "language": "en",
        "intent_kind": "action",
        "call_for_action": True,
        "goals": [
            {
                "semantic_operation": "set_sunshade_position",
                "desired_outcome": {"percentage": 100},
            },
            {
                "semantic_operation": "set_sunroof_position",
                "desired_outcome": {"percentage": 50},
                "depends_on_indices": [0],
            },
        ],
        "explicit_slots": {"percentage": 50},
        "explicit_constraints": {},
        "ambiguities": [],
    }
    utterance = (
        "Can you open the sunroof? I'd like to get some fresh air. If the "
        "sunshade needs to open first, please open it all the way."
    )

    extracted = IntentExtractor(FakeClient(draft)).extract(
        system_policy="policy",
        conversation=[{"turn_id": "u1", "role": "user", "content": utterance}],
    )
    grounded = ground_intent(utterance, extracted, RecipeRegistry())

    assert [goal.semantic_operation for goal in grounded.filtered_intent.goals] == [
        "set_sunroof_position"
    ]
    assert grounded.filtered_intent.goals[0].desired_outcome == {}
    assert grounded.filtered_intent.goals[0].depends_on == []
    assert grounded.authorized_action_goal_ids == frozenset(
        {grounded.filtered_intent.goals[0].goal_id}
    )


def test_non_action_empty_draft_does_not_retry() -> None:
    empty = {
        "language": "en",
        "intent_kind": "conversation",
        "call_for_action": False,
        "goals": [],
        "explicit_slots": {},
        "explicit_constraints": {},
        "ambiguities": [],
    }
    client = SequencedFakeClient([empty])

    intent = IntentExtractor(client).extract(
        system_policy="policy",
        conversation=[
            {"turn_id": "u1", "role": "user", "content": "Hello there"}
        ],
    )

    assert len(client.calls) == 1
    assert intent.goals == []


def test_goal_planner_preserves_done_independent_goal() -> None:
    extractor = IntentExtractor(FakeClient(action_draft()))
    intent = extractor.extract(
        system_policy="policy",
        conversation=[{"turn_id": "1", "role": "user", "content": "request"}],
    )
    previous = GoalDAG(goals=[goal.model_copy(deep=True) for goal in intent.goals])
    previous.set_status(intent.goals[0].goal_id, GoalStatus.DONE)

    updated = GoalPlanner().update(previous, intent)

    assert updated.goals[0].status is GoalStatus.DONE
    assert updated.goals[1].status is GoalStatus.PENDING


def test_repeated_semantic_request_gets_fresh_goal_state() -> None:
    previous = GoalDAG(
        goals=[
            Goal(
                goal_id="goal-old",
                semantic_operation="set_fan_speed",
                desired_outcome={"level": 2},
                status=GoalStatus.DONE,
            )
        ]
    )
    intent = IntentFrame(
        language="en",
        call_for_action=True,
        goals=[
            Goal(
                goal_id="goal-new",
                semantic_operation="set_fan_speed",
                desired_outcome={"level": 2},
            )
        ],
        intent_kind=IntentKind.ACTION,
    )

    updated = GoalPlanner().update(previous, intent)

    assert updated.get("goal-new").status is GoalStatus.PENDING


def test_goal_dag_blocks_dependents_not_independent_goals() -> None:
    draft = action_draft()
    draft["goals"].append(
        {
            "semantic_operation": "announce_arrival",
            "desired_outcome": {},
            "depends_on_indices": [1],
        }
    )
    intent = IntentExtractor(FakeClient(draft)).extract(
        system_policy="policy",
        conversation=[{"turn_id": "1", "role": "user", "content": "request"}],
    )
    dag = GoalDAG(goals=intent.goals)
    dag.set_status(
        intent.goals[1].goal_id,
        GoalStatus.BLOCKED,
        block_reason="route evidence unavailable",
    )
    dag.block_dependents(intent.goals[1].goal_id, "dependency blocked")

    assert dag.goals[0].status is GoalStatus.PENDING
    assert dag.goals[2].status is GoalStatus.BLOCKED
