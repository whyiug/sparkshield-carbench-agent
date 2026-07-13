import json
import unittest

from pydantic import ValidationError

from track_1_agent_under_test.car_guard.domain import (
    AmbiguitySlot,
    Candidate,
    CommitDecision,
    Confirmation,
    ConfirmationScope,
    ConfirmationStatus,
    DecisionProposal,
    Elimination,
    Evidence,
    EvidenceCardinality,
    EvidenceNeed,
    EvidenceSourceKind,
    EvidenceStatus,
    EvidenceStore,
    GateOutcome,
    GateReason,
    Goal,
    GoalDAG,
    GoalStatus,
    IntentFrame,
    IntentKind,
    IntentState,
    OfficialToolCall,
    ProposalKind,
    ProposedToolCall,
    ResolutionLevel,
)


class DecisionProposalTest(unittest.TestCase):
    def test_tool_and_text_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValidationError, "exactly one"):
            DecisionProposal(
                kind="respond",
                user_text="hello",
                tool_calls=[ProposedToolCall(tool_name="get_weather")],
            )

        with self.assertRaisesRegex(ValidationError, "exactly one"):
            DecisionProposal(kind="respond")

    def test_kind_must_match_output_and_sets_are_serial(self) -> None:
        with self.assertRaisesRegex(ValidationError, "output mode"):
            DecisionProposal(
                kind="tool_get",
                user_text="I cannot do that.",
            )
        with self.assertRaisesRegex(ValidationError, "exactly one call"):
            DecisionProposal(
                kind="tool_set",
                tool_calls=[
                    ProposedToolCall(tool_name="set_ac"),
                    ProposedToolCall(tool_name="set_fan"),
                ],
            )

    def test_get_proposal_round_trips_through_json(self) -> None:
        proposal = DecisionProposal(
            kind=ProposalKind.TOOL_GET,
            goal_ids=["weather-check"],
            tool_calls=[
                ProposedToolCall(
                    call_id="call-1",
                    tool_name="get_weather",
                    arguments={"date": "today"},
                    argument_sources={"date": "system-current-date"},
                )
            ],
            evidence_used=["ev-location"],
        )

        restored = DecisionProposal.model_validate_json(proposal.model_dump_json())
        self.assertEqual(restored, proposal)
        self.assertEqual(restored.kind, ProposalKind.TOOL_GET)

    def test_parallel_get_names_must_be_distinct(self) -> None:
        with self.assertRaisesRegex(ValidationError, "duplicate tool names"):
            DecisionProposal(
                kind="tool_get",
                tool_calls=[
                    ProposedToolCall(tool_name="get_weather", arguments={"hour": 9}),
                    ProposedToolCall(tool_name="get_weather", arguments={"hour": 10}),
                ],
            )


class GateContractTest(unittest.TestCase):
    def test_official_call_has_minimal_a2a_serialization(self) -> None:
        call = OfficialToolCall(
            tool_call_id="internal-call-id",
            tool_name="set_ac",
            arguments={"enabled": True},
        )
        self.assertEqual(
            call.to_a2a_payload(),
            {"tool_name": "set_ac", "arguments": {"enabled": True}},
        )

    def test_allow_decision_requires_one_output_mode(self) -> None:
        with self.assertRaisesRegex(ValidationError, "requires calls or user text"):
            CommitDecision(outcome="allow")
        with self.assertRaisesRegex(ValidationError, "calls and text"):
            CommitDecision(
                outcome="allow",
                normalized_calls=[OfficialToolCall(tool_name="get_weather")],
                user_text="Checking.",
            )

        decision = CommitDecision(
            outcome=GateOutcome.ALLOW,
            user_text="  Please choose a contact.  ",
            reasons=[
                GateReason(code="ambiguity.contact", message="Two contacts remain")
            ],
        )
        self.assertEqual(
            decision.to_outbound_payload(), {"text": "Please choose a contact."}
        )
        self.assertEqual(
            CommitDecision.model_validate_json(decision.model_dump_json()), decision
        )


class AmbiguityContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.alex_home = Candidate(candidate_id="alex-home", value="contact-1")
        self.alex_work = Candidate(candidate_id="alex-work", value="contact-2")

    def test_elimination_is_auditable_and_choice_must_remain(self) -> None:
        slot = AmbiguitySlot(
            name="contact",
            candidates=[self.alex_home, self.alex_work],
            eliminated=[
                Elimination(
                    candidate_id="alex-work",
                    eliminated_by="explicit_user",
                    reason="User requested the home contact",
                    evidence_ids=["ev-user-choice"],
                )
            ],
        )
        resolved = slot.with_choice("alex-home", ResolutionLevel.EXPLICIT_USER)
        self.assertTrue(resolved.is_resolved)
        self.assertEqual(resolved.remaining_candidates, [self.alex_home])

        with self.assertRaisesRegex(ValidationError, "eliminated candidate"):
            AmbiguitySlot(
                name="contact",
                candidates=[self.alex_home, self.alex_work],
                eliminated=[
                    Elimination(
                        candidate_id="alex-work",
                        eliminated_by="context",
                        reason="Not available",
                    )
                ],
                chosen=self.alex_work,
                chosen_by="context",
            )

    def test_unknown_elimination_is_invalid(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unknown candidates"):
            AmbiguitySlot(
                name="contact",
                candidates=[self.alex_home],
                eliminated=[
                    Elimination(
                        candidate_id="absent",
                        eliminated_by="preference",
                        reason="Not preferred",
                    )
                ],
            )


class GoalDAGTest(unittest.TestCase):
    def test_cycle_and_unknown_dependency_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unknown dependencies"):
            GoalDAG(
                goals=[Goal(goal_id="a", semantic_operation="op", depends_on=["b"])]
            )

        with self.assertRaisesRegex(ValidationError, "acyclic"):
            GoalDAG(
                goals=[
                    Goal(goal_id="a", semantic_operation="one", depends_on=["b"]),
                    Goal(goal_id="b", semantic_operation="two", depends_on=["a"]),
                ]
            )

    def test_blocking_is_local_and_independent_goal_stays_ready(self) -> None:
        dag = GoalDAG(
            goals=[
                Goal(goal_id="read", semantic_operation="read_state"),
                Goal(
                    goal_id="set", semantic_operation="set_state", depends_on=["read"]
                ),
                Goal(goal_id="independent", semantic_operation="get_weather"),
            ]
        )
        dag.set_status("read", GoalStatus.BLOCKED, block_reason="No reliable evidence")
        blocked = dag.block_dependents("read", "Dependency is blocked")

        self.assertEqual([goal.goal_id for goal in blocked], ["set"])
        self.assertEqual(dag.get("set").status, GoalStatus.BLOCKED)
        self.assertEqual(
            [goal.goal_id for goal in dag.ready_goals],
            ["independent"],
        )

    def test_topological_order_is_stable_for_independent_goals(self) -> None:
        dag = GoalDAG(
            goals=[
                Goal(goal_id="second", semantic_operation="second"),
                Goal(goal_id="first", semantic_operation="first"),
                Goal(
                    goal_id="dependent",
                    semantic_operation="dependent",
                    depends_on=["first"],
                ),
            ]
        )
        self.assertEqual(
            [goal.goal_id for goal in dag.topological_order()],
            ["second", "first", "dependent"],
        )


class EvidenceContractTest(unittest.TestCase):
    def _need(self, **updates) -> EvidenceNeed:
        data = {
            "proposition": "weather permits sunroof action",
            "required_for_goal_id": "open-sunroof",
            "acceptable_sources": {"tool_result"},
            "cardinality": EvidenceCardinality.UNIQUE,
        }
        data.update(updates)
        return EvidenceNeed(**data)

    def _evidence(self, **updates) -> Evidence:
        data = {
            "proposition": "weather permits sunroof action",
            "value": True,
            "status": EvidenceStatus.KNOWN,
            "source_kind": EvidenceSourceKind.TOOL,
            "source_turn_id": "turn-2",
            "source_tool_call_id": "call-weather",
            "confidence": 1.0,
        }
        data.update(updates)
        return Evidence(**data)

    def test_unknown_is_not_known_and_does_not_satisfy_need(self) -> None:
        need = self._need()
        store = EvidenceStore()
        store.register_need(need)
        store.add(
            self._evidence(
                value=None,
                status=EvidenceStatus.UNKNOWN,
            )
        )
        self.assertFalse(store.is_satisfied(need))
        self.assertEqual(store.pending_needs, [need])

        with self.assertRaisesRegex(ValidationError, "concrete value"):
            self._evidence(value="unknown", status=EvidenceStatus.KNOWN)

    def test_only_goal_conditioned_matching_evidence_satisfies_need(self) -> None:
        need = self._need()
        unrelated = self._evidence(
            proposition="cabin temperature is known",
            value=21,
        )
        store = EvidenceStore()
        store.register_need(need)
        store.update([unrelated, self._evidence()])

        self.assertTrue(store.is_satisfied(need))
        self.assertEqual(len(store.satisfying_evidence(need)), 1)

    def test_unique_cardinality_rejects_conflicting_values(self) -> None:
        need = self._need()
        store = EvidenceStore()
        store.register_need(need)
        store.update(
            [
                self._evidence(value=True, source_turn_id="turn-1"),
                self._evidence(value=False, source_turn_id="turn-2"),
            ]
        )
        self.assertFalse(store.is_satisfied(need))

    def test_derived_evidence_requires_complete_provenance(self) -> None:
        with self.assertRaisesRegex(ValidationError, "input evidence IDs"):
            self._evidence(
                source_kind=EvidenceSourceKind.DERIVED,
                source_tool_call_id=None,
            )

        derived = self._evidence(
            source_kind=EvidenceSourceKind.DERIVED,
            source_tool_call_id=None,
            derived_from=["evidence-weather"],
            derivation="weather_allows_sunroof_v1",
        )
        self.assertTrue(derived.evidence_id.startswith("evidence-"))

    def test_tool_evidence_requires_a_real_call_id(self) -> None:
        with self.assertRaisesRegex(ValidationError, "source_tool_call_id"):
            self._evidence(source_tool_call_id=None)

    def test_store_and_set_fields_round_trip_deterministically(self) -> None:
        need = self._need(acceptable_sources={"tool_result", "system_context"})
        store = EvidenceStore()
        store.register_need(need)
        store.add(self._evidence())

        payload = store.model_dump_json()
        self.assertEqual(
            json.loads(payload)["needs"][need.need_id]["acceptable_sources"],
            ["system_context", "tool_result"],
        )
        self.assertEqual(EvidenceStore.model_validate_json(payload), store)

    def test_latest_observation_and_state_version_control_freshness(self) -> None:
        need = self._need(freshness="current_state")
        store = EvidenceStore()
        store.register_need(need)
        store.add(self._evidence(source_turn_id="turn-1"))
        self.assertTrue(store.is_satisfied(need))

        store.add(
            self._evidence(
                source_turn_id="turn-2",
                value=None,
                status=EvidenceStatus.UNKNOWN,
            )
        )
        self.assertFalse(store.is_satisfied(need))

        store.invalidate_tool_state()
        store.add(self._evidence(source_turn_id="turn-3"))
        self.assertTrue(store.is_satisfied(need))
        store.invalidate_tool_state()
        self.assertFalse(store.is_satisfied(need))

    def test_repeated_observation_gets_a_fresh_stable_id(self) -> None:
        store = EvidenceStore()
        first = self._evidence(source_turn_id="result-1", source_tool_call_id="call-1")
        second = self._evidence(source_turn_id="result-2", source_tool_call_id="call-2")

        store.update([first, second])

        observations = list(store.evidence.values())
        self.assertNotEqual(observations[0].evidence_id, observations[1].evidence_id)
        self.assertEqual([item.observation_index for item in observations], [1, 2])


class ConfirmationContractTest(unittest.TestCase):
    def test_scope_binds_goal_action_arguments_order_and_expiration(self) -> None:
        actions = [
            OfficialToolCall(tool_name="set_fog_lights", arguments={"enabled": False}),
            OfficialToolCall(tool_name="set_high_beam", arguments={"enabled": True}),
        ]
        scope = ConfirmationScope(
            goal_ids=["high-beam"],
            ordered_actions=actions,
            requested_at_user_turn=3,
            expires_after_user_turn=4,
        )
        confirmation = Confirmation(
            confirmation_id="confirm-1",
            scope=scope,
            status=ConfirmationStatus.CONFIRMED,
            source_turn_id="turn-4",
            user_response="yes",
            resolved_at_user_turn=4,
        )

        self.assertTrue(
            confirmation.authorizes(
                goal_ids=["high-beam"],
                ordered_actions=actions,
                current_user_turn=4,
            )
        )
        self.assertFalse(
            confirmation.authorizes(
                goal_ids=["high-beam"],
                ordered_actions=list(reversed(actions)),
                current_user_turn=4,
            )
        )
        same_actions_with_new_internal_ids = [
            action.model_copy(update={"tool_call_id": f"new-{index}"})
            for index, action in enumerate(actions)
        ]
        self.assertTrue(
            confirmation.authorizes(
                goal_ids=["high-beam"],
                ordered_actions=same_actions_with_new_internal_ids,
                current_user_turn=4,
            )
        )
        self.assertFalse(
            confirmation.authorizes(
                goal_ids=["high-beam"],
                ordered_actions=actions,
                current_user_turn=5,
            )
        )
        self.assertEqual(
            Confirmation.model_validate_json(confirmation.model_dump_json()),
            confirmation,
        )

    def test_scope_rejects_argument_mismatch(self) -> None:
        with self.assertRaisesRegex(ValidationError, "exactly match"):
            ConfirmationScope(
                goal_ids=["ac"],
                ordered_actions=[
                    OfficialToolCall(tool_name="set_ac", arguments={"enabled": True})
                ],
                normalized_arguments=[{"enabled": False}],
                requested_at_user_turn=1,
                expires_after_user_turn=2,
            )


class IntentContractTest(unittest.TestCase):
    def test_intent_state_and_frame_preserve_goal_mention_order(self) -> None:
        goals = [
            Goal(goal_id="window", semantic_operation="open_window"),
            Goal(goal_id="weather", semantic_operation="get_weather"),
        ]
        frame = IntentFrame(
            language="en",
            call_for_action=True,
            intent_kind=IntentKind.ACTION,
            goals=goals,
            explicit_slots={"window": "front_left"},
            intent_source_turn_ids=["turn-1"],
        )

        self.assertEqual(frame.goal_mention_order, ["window", "weather"])
        self.assertEqual(frame.semantic_goals, goals)
        restored = IntentState.model_validate_json(frame.model_dump_json())
        self.assertEqual(restored.goals, frame.goals)
        self.assertEqual(restored.intent_kind, IntentKind.ACTION)

    def test_goal_mention_order_must_be_complete(self) -> None:
        with self.assertRaisesRegex(ValidationError, "every goal"):
            IntentFrame(
                language="en",
                call_for_action=True,
                goals=[Goal(goal_id="one", semantic_operation="one")],
                goal_mention_order=["different"],
            )


if __name__ == "__main__":
    unittest.main()
