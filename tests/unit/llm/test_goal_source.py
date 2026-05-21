"""
tests/unit/llm/test_goal_source.py

Unit tests for llm/goal_source.py.

All tests run without a live Factorio instance, no RCON, no game state.

Coverage:
  GoalQueueEntry:
    - to_goal() sets correct fields and dynamic attributes
    - to_goal() sets bounding_box and clear_mode when provided
    - from_goal() roundtrips all standard fields
    - to_dict() / from_dict() JSON roundtrip preserves all fields
    - from_dict() tolerates missing optional fields
  GoalQueue:
    - next_goal() dispenses in order
    - next_goal() returns None when exhausted
    - next_goal() loops when loop_forever=True
    - handle_stuck() returns []
    - record_outcome() records to outcomes list
    - remaining() counts correctly
    - append() adds to end of queue
    - save() / from_file() roundtrip preserves goals and outcomes
    - load_with_outcomes() restores outcomes
    - len() returns entry count
    - repr() contains useful info
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm.goal_source import GoalQueue, GoalQueueEntry
from planning.goal import GoalStatus, Priority


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(
    description: str = "Collect 10 iron ore",
    success_condition: str = "inventory('iron-ore') >= 10",
    failure_condition: str = "tick > 3600",
    goal_type: str = "collection",
    **kwargs,
) -> GoalQueueEntry:
    return GoalQueueEntry(
        description=description,
        success_condition=success_condition,
        failure_condition=failure_condition,
        goal_type=goal_type,
        **kwargs,
    )


def _stub_stuck_context():
    """Minimal stand-in for a StuckContext — GoalQueue only logs it."""
    class _Stub:
        goal = type("Goal", (), {"id": "test-goal-id"})()
        immediate_failure = None
    return _Stub()


# ---------------------------------------------------------------------------
# GoalQueueEntry.to_goal()
# ---------------------------------------------------------------------------

class TestGoalQueueEntryToGoal(unittest.TestCase):

    def test_description_passed_through(self):
        e = _entry(description="Test goal")
        goal = e.to_goal()
        self.assertEqual(goal.description, "Test goal")

    def test_success_condition_passed_through(self):
        e = _entry(success_condition="inventory('iron-ore') >= 5")
        goal = e.to_goal()
        self.assertEqual(goal.success_condition, "inventory('iron-ore') >= 5")

    def test_failure_condition_passed_through(self):
        e = _entry(failure_condition="tick > 1800")
        goal = e.to_goal()
        self.assertEqual(goal.failure_condition, "tick > 1800")

    def test_goal_type_set_as_dynamic_attribute(self):
        e = _entry(goal_type="exploration")
        goal = e.to_goal()
        self.assertEqual(getattr(goal, "type", None), "exploration")

    def test_priority_normal_by_default(self):
        e = _entry()
        goal = e.to_goal()
        self.assertEqual(goal.priority, Priority.NORMAL)

    def test_priority_urgent_respected(self):
        e = _entry()
        e.priority = "URGENT"
        goal = e.to_goal()
        self.assertEqual(goal.priority, Priority.URGENT)

    def test_reward_spec_values_passed(self):
        e = _entry()
        e.success_reward = 2.5
        e.failure_penalty = 0.3
        goal = e.to_goal()
        self.assertAlmostEqual(goal.reward_spec.success_reward, 2.5)
        self.assertAlmostEqual(goal.reward_spec.failure_penalty, 0.3)

    def test_bounding_box_set_when_provided(self):
        bbox = {"x_min": -10.0, "y_min": -10.0, "x_max": 10.0, "y_max": 10.0}
        e = _entry(goal_type="clear_region", bounding_box=bbox)
        goal = e.to_goal()
        self.assertEqual(getattr(goal, "bounding_box", None), bbox)

    def test_bounding_box_not_set_when_none(self):
        e = _entry()
        goal = e.to_goal()
        self.assertFalse(hasattr(goal, "bounding_box"))

    def test_clear_mode_set_when_provided(self):
        e = _entry(goal_type="clear_region", clear_mode="clear_natural")
        goal = e.to_goal()
        self.assertEqual(getattr(goal, "clear_mode", None), "clear_natural")

    def test_clear_mode_not_set_when_none(self):
        e = _entry()
        goal = e.to_goal()
        self.assertFalse(hasattr(goal, "clear_mode"))

    def test_to_goal_returns_new_object_each_call(self):
        e = _entry()
        self.assertIsNot(e.to_goal(), e.to_goal())


# ---------------------------------------------------------------------------
# GoalQueueEntry.from_goal()
# ---------------------------------------------------------------------------

class TestGoalQueueEntryFromGoal(unittest.TestCase):

    def test_from_goal_roundtrip_description(self):
        e = _entry(description="Roundtrip test")
        goal = e.to_goal()
        restored = GoalQueueEntry.from_goal(goal)
        self.assertEqual(restored.description, "Roundtrip test")

    def test_from_goal_roundtrip_conditions(self):
        e = _entry(
            success_condition="charted_chunks >= 5",
            failure_condition="tick > 9000",
        )
        goal = e.to_goal()
        restored = GoalQueueEntry.from_goal(goal)
        self.assertEqual(restored.success_condition, "charted_chunks >= 5")
        self.assertEqual(restored.failure_condition, "tick > 9000")

    def test_from_goal_roundtrip_goal_type(self):
        e = _entry(goal_type="exploration")
        goal = e.to_goal()
        restored = GoalQueueEntry.from_goal(goal)
        self.assertEqual(restored.goal_type, "exploration")

    def test_from_goal_roundtrip_priority(self):
        e = _entry()
        e.priority = "URGENT"
        goal = e.to_goal()
        restored = GoalQueueEntry.from_goal(goal)
        self.assertEqual(restored.priority, "URGENT")


# ---------------------------------------------------------------------------
# GoalQueueEntry serialisation (to_dict / from_dict)
# ---------------------------------------------------------------------------

class TestGoalQueueEntrySerialisation(unittest.TestCase):

    def test_to_dict_contains_required_keys(self):
        e = _entry()
        d = e.to_dict()
        for key in ("description", "success_condition", "failure_condition",
                    "goal_type", "priority", "success_reward"):
            self.assertIn(key, d)

    def test_from_dict_roundtrip(self):
        e = _entry(description="Serialise me", goal_type="exploration")
        e.success_reward = 3.0
        restored = GoalQueueEntry.from_dict(e.to_dict())
        self.assertEqual(restored.description, "Serialise me")
        self.assertEqual(restored.goal_type, "exploration")
        self.assertAlmostEqual(restored.success_reward, 3.0)

    def test_from_dict_ignores_unknown_keys(self):
        d = _entry().to_dict()
        d["future_field_not_yet_in_dataclass"] = "ignored"
        # Should not raise.
        GoalQueueEntry.from_dict(d)

    def test_from_dict_handles_bounding_box(self):
        bbox = {"x_min": 0.0, "y_min": 0.0, "x_max": 5.0, "y_max": 5.0}
        e = _entry(goal_type="clear_region", bounding_box=bbox)
        restored = GoalQueueEntry.from_dict(e.to_dict())
        self.assertEqual(restored.bounding_box, bbox)

    def test_json_roundtrip(self):
        e = _entry(description="JSON roundtrip")
        raw = json.dumps(e.to_dict())
        restored = GoalQueueEntry.from_dict(json.loads(raw))
        self.assertEqual(restored.description, "JSON roundtrip")


# ---------------------------------------------------------------------------
# GoalQueue.next_goal()
# ---------------------------------------------------------------------------

class TestGoalQueueNextGoal(unittest.TestCase):

    def test_dispenses_goals_in_order(self):
        q = GoalQueue([_entry(description="A"), _entry(description="B")])
        g1 = q.next_goal({})
        g2 = q.next_goal({})
        self.assertEqual(g1.description, "A")
        self.assertEqual(g2.description, "B")

    def test_returns_none_when_exhausted(self):
        q = GoalQueue([_entry()])
        q.next_goal({})
        self.assertIsNone(q.next_goal({}))

    def test_empty_queue_returns_none_immediately(self):
        q = GoalQueue([])
        self.assertIsNone(q.next_goal({}))

    def test_loop_forever_restarts_when_exhausted(self):
        q = GoalQueue([_entry(description="loop")], loop_forever=True)
        q.next_goal({})
        # After exhaustion should restart.
        g = q.next_goal({})
        self.assertIsNotNone(g)
        self.assertEqual(g.description, "loop")

    def test_goal_type_attribute_set_on_returned_goal(self):
        q = GoalQueue([_entry(goal_type="exploration")])
        goal = q.next_goal({})
        self.assertEqual(getattr(goal, "type", None), "exploration")

    def test_returned_goal_is_pending(self):
        """next_goal() constructs a Goal but does not activate it — activation
        is the loop's responsibility (it needs the current tick). The goal
        should be in PENDING status when returned."""
        q = GoalQueue([_entry()])
        goal = q.next_goal({})
        self.assertEqual(goal.status, GoalStatus.PENDING)

    def test_context_argument_ignored_by_goal_queue(self):
        """GoalQueue ignores context — should not raise with any dict."""
        q = GoalQueue([_entry()])
        q.next_goal({"tick": 999, "inventory": {"iron-ore": 3}, "unexpected": True})


# ---------------------------------------------------------------------------
# GoalQueue.handle_stuck()
# ---------------------------------------------------------------------------

class TestGoalQueueHandleStuck(unittest.TestCase):

    def test_handle_stuck_returns_empty_list(self):
        q = GoalQueue([_entry()])
        result = q.handle_stuck(_stub_stuck_context())
        self.assertEqual(result, [])

    def test_handle_stuck_does_not_raise(self):
        q = GoalQueue([_entry()])
        q.handle_stuck(_stub_stuck_context())


# ---------------------------------------------------------------------------
# GoalQueue.record_outcome()
# ---------------------------------------------------------------------------

class TestGoalQueueRecordOutcome(unittest.TestCase):

    def test_record_outcome_appends_to_outcomes(self):
        q = GoalQueue([_entry()])
        goal = q.next_goal({})
        q.record_outcome(goal, reward=1.0)
        self.assertEqual(len(q.outcomes()), 1)

    def test_record_outcome_stores_reward(self):
        q = GoalQueue([_entry()])
        goal = q.next_goal({})
        q.record_outcome(goal, reward=0.75)
        self.assertAlmostEqual(q.outcomes()[0]["reward"], 0.75)

    def test_record_outcome_stores_goal_id(self):
        q = GoalQueue([_entry()])
        goal = q.next_goal({})
        q.record_outcome(goal, reward=1.0)
        self.assertEqual(q.outcomes()[0]["goal_id"], goal.id)

    def test_multiple_outcomes_all_recorded(self):
        q = GoalQueue([_entry(), _entry(description="B")])
        g1 = q.next_goal({})
        g2 = q.next_goal({})
        q.record_outcome(g1, 1.0)
        q.record_outcome(g2, 0.5)
        self.assertEqual(len(q.outcomes()), 2)


# ---------------------------------------------------------------------------
# GoalQueue.remaining() / append() / len() / repr()
# ---------------------------------------------------------------------------

class TestGoalQueueMisc(unittest.TestCase):

    def test_remaining_decrements_on_next_goal(self):
        q = GoalQueue([_entry(), _entry()])
        self.assertEqual(q.remaining(), 2)
        q.next_goal({})
        self.assertEqual(q.remaining(), 1)
        q.next_goal({})
        self.assertEqual(q.remaining(), 0)

    def test_remaining_never_negative(self):
        q = GoalQueue([_entry()])
        q.next_goal({})
        q.next_goal({})   # already exhausted
        self.assertEqual(q.remaining(), 0)

    def test_append_adds_to_end(self):
        q = GoalQueue([_entry(description="first")])
        q.append(_entry(description="appended"))
        self.assertEqual(len(q), 2)
        q.next_goal({})
        g = q.next_goal({})
        self.assertEqual(g.description, "appended")

    def test_len_returns_total_entries(self):
        q = GoalQueue([_entry(), _entry(), _entry()])
        self.assertEqual(len(q), 3)

    def test_repr_contains_count(self):
        q = GoalQueue([_entry(), _entry()])
        r = repr(q)
        self.assertIn("GoalQueue", r)
        self.assertIn("2", r)


# ---------------------------------------------------------------------------
# GoalQueue.save() / from_file() / load_with_outcomes()
# ---------------------------------------------------------------------------

class TestGoalQueuePersistence(unittest.TestCase):

    def test_save_and_from_file_roundtrip(self):
        q = GoalQueue([
            _entry(description="goal A"),
            _entry(description="goal B", goal_type="exploration"),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "goals.json"
            q.save(path)
            q2 = GoalQueue.from_file(path)

        self.assertEqual(len(q2), 2)
        g1 = q2.next_goal({})
        g2 = q2.next_goal({})
        self.assertEqual(g1.description, "goal A")
        self.assertEqual(g2.description, "goal B")
        self.assertEqual(getattr(g2, "type", None), "exploration")

    def test_save_includes_outcomes(self):
        q = GoalQueue([_entry()])
        goal = q.next_goal({})
        q.record_outcome(goal, reward=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "goals.json"
            q.save(path)
            data = json.loads(path.read_text())
        self.assertIn("outcomes", data)
        self.assertEqual(len(data["outcomes"]), 1)

    def test_load_with_outcomes_restores_outcomes(self):
        q = GoalQueue([_entry()])
        goal = q.next_goal({})
        q.record_outcome(goal, reward=0.8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "goals.json"
            q.save(path)
            q2 = GoalQueue.load_with_outcomes(path)
        self.assertEqual(len(q2.outcomes()), 1)
        self.assertAlmostEqual(q2.outcomes()[0]["reward"], 0.8)

    def test_from_file_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            GoalQueue.from_file("/nonexistent/path/goals.json")

    def test_save_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "goals.json"
            GoalQueue([_entry()]).save(path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()