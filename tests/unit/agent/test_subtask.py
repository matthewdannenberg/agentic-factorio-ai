"""
tests/unit/agent/test_subtask.py

Tests for agent/subtask.py — Subtask, SubtaskRecord, SubtaskLedger.

Run with:  python -m unittest tests.unit.agent.test_subtask
"""

from __future__ import annotations

import unittest

from agent.subtask import (
    Subtask,
    SubtaskLedger,
    SubtaskRecord,
    SubtaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subtask(
    description: str = "mine 10 iron ore",
    parent_goal_id: str = "goal-1",
    parent_subtask_id: str | None = None,
    derived_locally: bool = True,
    created_at: int = 0,
) -> Subtask:
    return Subtask(
        description=description,
        success_condition="inventory('iron-ore') >= 10",
        failure_condition="tick > 500",
        parent_goal_id=parent_goal_id,
        created_at=created_at,
        derived_locally=derived_locally,
        parent_subtask_id=parent_subtask_id,
    )


# ---------------------------------------------------------------------------
# Subtask lifecycle
# ---------------------------------------------------------------------------

class TestSubtaskLifecycle(unittest.TestCase):
    def test_initial_status_is_pending(self):
        self.assertEqual(_make_subtask().status, SubtaskStatus.PENDING)

    def test_activate_pending_to_active(self):
        t = _make_subtask()
        t.activate()
        self.assertEqual(t.status, SubtaskStatus.ACTIVE)

    def test_activate_non_pending_raises(self):
        t = _make_subtask()
        t.activate()
        with self.assertRaises(RuntimeError):
            t.activate()

    def test_complete_active_to_complete(self):
        t = _make_subtask()
        t.activate()
        t.complete(tick=100)
        self.assertEqual(t.status, SubtaskStatus.COMPLETE)
        self.assertEqual(t.resolved_at, 100)

    def test_complete_non_active_raises(self):
        t = _make_subtask()
        with self.assertRaises(RuntimeError):
            t.complete(tick=0)

    def test_fail_active_to_failed(self):
        t = _make_subtask()
        t.activate()
        t.fail(tick=200)
        self.assertEqual(t.status, SubtaskStatus.FAILED)
        self.assertEqual(t.resolved_at, 200)

    def test_fail_non_active_raises(self):
        t = _make_subtask()
        with self.assertRaises(RuntimeError):
            t.fail(tick=0)

    def test_escalate_active_to_escalated(self):
        t = _make_subtask()
        t.activate()
        t.escalate(tick=300)
        self.assertEqual(t.status, SubtaskStatus.ESCALATED)
        self.assertEqual(t.resolved_at, 300)

    def test_escalate_non_active_raises(self):
        t = _make_subtask()
        with self.assertRaises(RuntimeError):
            t.escalate(tick=0)

    def test_is_terminal_complete(self):
        t = _make_subtask(); t.activate(); t.complete(1)
        self.assertTrue(t.is_terminal)

    def test_is_terminal_failed(self):
        t = _make_subtask(); t.activate(); t.fail(1)
        self.assertTrue(t.is_terminal)

    def test_is_terminal_escalated(self):
        t = _make_subtask(); t.activate(); t.escalate(1)
        self.assertTrue(t.is_terminal)

    def test_not_terminal_pending(self):
        self.assertFalse(_make_subtask().is_terminal)

    def test_not_terminal_active(self):
        t = _make_subtask(); t.activate()
        self.assertFalse(t.is_terminal)

    def test_is_active_property(self):
        t = _make_subtask()
        self.assertFalse(t.is_active)
        t.activate()
        self.assertTrue(t.is_active)
        t.complete(1)
        self.assertFalse(t.is_active)


class TestSubtaskFields(unittest.TestCase):
    def test_derived_locally_preserved(self):
        self.assertTrue(_make_subtask(derived_locally=True).derived_locally)
        self.assertFalse(_make_subtask(derived_locally=False).derived_locally)

    def test_parent_goal_id_preserved(self):
        self.assertEqual(_make_subtask(parent_goal_id="g-99").parent_goal_id, "g-99")

    def test_parent_subtask_id_none_by_default(self):
        self.assertIsNone(_make_subtask().parent_subtask_id)

    def test_parent_id_returns_subtask_id_when_set(self):
        parent = _make_subtask()
        child = _make_subtask(parent_subtask_id=parent.id)
        self.assertEqual(child.parent_id, parent.id)

    def test_parent_id_returns_goal_id_when_top_level(self):
        t = _make_subtask(parent_goal_id="goal-42")
        self.assertEqual(t.parent_id, "goal-42")

    def test_unique_ids(self):
        ids = {_make_subtask().id for _ in range(10)}
        self.assertEqual(len(ids), 10)

    def test_created_at_preserved(self):
        self.assertEqual(_make_subtask(created_at=77).created_at, 77)


# ---------------------------------------------------------------------------
# SubtaskRecord
# ---------------------------------------------------------------------------

class TestSubtaskRecord(unittest.TestCase):
    def _make_record(self, outcome="complete") -> SubtaskRecord:
        t = _make_subtask("buy milk")
        t.activate()
        t.complete(tick=10)
        return SubtaskRecord(subtask=t, outcome=outcome, children_ids=["c1", "c2"])

    def test_to_dict_fields(self):
        rec = self._make_record("complete")
        d = rec.to_dict()
        self.assertEqual(d["description"], "buy milk")
        self.assertEqual(d["outcome"], "complete")
        self.assertTrue(d["derived_locally"])
        self.assertEqual(d["resolved_at"], 10)
        self.assertEqual(d["children_ids"], ["c1", "c2"])

    def test_to_dict_children_ids_is_copy(self):
        rec = self._make_record()
        d = rec.to_dict()
        d["children_ids"].append("extra")
        self.assertEqual(rec.children_ids, ["c1", "c2"])


# ---------------------------------------------------------------------------
# SubtaskLedger — basic stack operations
# ---------------------------------------------------------------------------

class TestSubtaskLedgerStack(unittest.TestCase):
    def setUp(self):
        self.ledger = SubtaskLedger()

    def test_empty_len_zero(self):
        self.assertEqual(len(self.ledger), 0)
        self.assertFalse(self.ledger)

    def test_push_single_activates_immediately(self):
        t = _make_subtask()
        self.ledger.push(t)
        self.assertEqual(t.status, SubtaskStatus.ACTIVE)
        self.assertEqual(len(self.ledger), 1)

    def test_peek_returns_top(self):
        t = _make_subtask()
        self.ledger.push(t)
        self.assertIs(self.ledger.peek(), t)

    def test_peek_empty_returns_none(self):
        self.assertIsNone(self.ledger.peek())

    def test_second_push_activates_new_top_and_suspends_bottom(self):
        """Pushing a prerequisite activates it and suspends the parent."""
        t1 = _make_subtask("parent")
        t2 = _make_subtask("prerequisite")
        self.ledger.push(t1)
        self.assertEqual(t1.status, SubtaskStatus.ACTIVE)
        self.ledger.push(t2)
        # t2 (top) is the prerequisite — active now
        self.assertEqual(t2.status, SubtaskStatus.ACTIVE)
        # t1 (bottom/parent) is suspended while t2 runs
        self.assertEqual(t1.status, SubtaskStatus.PENDING)
        self.assertIs(self.ledger.peek(), t2)

    def test_pop_requires_terminal_status(self):
        t = _make_subtask()
        self.ledger.push(t)
        with self.assertRaises(RuntimeError):
            self.ledger.pop()   # still ACTIVE

    def test_pop_after_complete_succeeds(self):
        t = _make_subtask()
        self.ledger.push(t)
        t.complete(tick=10)
        popped = self.ledger.pop()
        self.assertIs(popped, t)
        self.assertEqual(len(self.ledger), 0)

    def test_pop_reactivates_suspended_parent(self):
        """Completing a prerequisite resumes the parent."""
        t1 = _make_subtask("parent")
        t2 = _make_subtask("prerequisite")
        self.ledger.push(t1)
        self.ledger.push(t2)
        t2.complete(tick=5)
        self.ledger.pop()
        self.assertEqual(t1.status, SubtaskStatus.ACTIVE)

    def test_pop_empty_returns_none(self):
        self.assertIsNone(self.ledger.pop())

    def test_push_non_pending_raises(self):
        t = _make_subtask()
        t.activate()
        with self.assertRaises(ValueError):
            self.ledger.push(t)

    def test_bool_true_when_non_empty(self):
        self.ledger.push(_make_subtask())
        self.assertTrue(self.ledger)

    def test_clear_resets_everything(self):
        t = _make_subtask()
        self.ledger.push(t)
        self.ledger.clear()
        self.assertEqual(len(self.ledger), 0)
        self.assertIsNone(self.ledger.peek())


# ---------------------------------------------------------------------------
# SubtaskLedger — history recording
# ---------------------------------------------------------------------------

class TestSubtaskLedgerHistory(unittest.TestCase):
    def setUp(self):
        self.ledger = SubtaskLedger()
        self.goal_id = "goal-1"

    def test_completed_subtask_recorded_in_history(self):
        t = _make_subtask("task-a", parent_goal_id=self.goal_id)
        self.ledger.push(t)
        t.complete(tick=10)
        self.ledger.pop()
        history = self.ledger.history_for(self.goal_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].outcome, "complete")
        self.assertIs(history[0].subtask, t)

    def test_failed_subtask_recorded_in_history(self):
        t = _make_subtask(parent_goal_id=self.goal_id)
        self.ledger.push(t)
        t.fail(tick=20)
        self.ledger.pop()
        history = self.ledger.history_for(self.goal_id)
        self.assertEqual(history[0].outcome, "failed")

    def test_escalated_subtask_recorded_in_history(self):
        t = _make_subtask(parent_goal_id=self.goal_id)
        self.ledger.push(t)
        t.escalate(tick=30)
        self.ledger.pop()
        history = self.ledger.history_for(self.goal_id)
        self.assertEqual(history[0].outcome, "escalated")

    def test_multiple_siblings_recorded_in_order(self):
        for desc in ["alpha", "beta", "gamma"]:
            t = _make_subtask(desc, parent_goal_id=self.goal_id)
            self.ledger.push(t)
            t.complete(tick=1)
            self.ledger.pop()
        history = self.ledger.history_for(self.goal_id)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0].subtask.description, "alpha")
        self.assertEqual(history[2].subtask.description, "gamma")

    def test_nested_subtask_recorded_under_parent_subtask(self):
        parent = _make_subtask("parent", parent_goal_id=self.goal_id)
        child = _make_subtask("child", parent_goal_id=self.goal_id,
                               parent_subtask_id=parent.id)
        self.ledger.push(parent)
        self.ledger.push(child)
        child.complete(tick=5)
        self.ledger.pop()
        # child should be in history under parent.id, not goal_id
        self.assertEqual(len(self.ledger.history_for(parent.id)), 1)
        self.assertEqual(len(self.ledger.history_for(self.goal_id)), 0)

    def test_children_ids_populated_in_parent_record(self):
        parent = _make_subtask("parent", parent_goal_id=self.goal_id)
        child1 = _make_subtask("c1", parent_goal_id=self.goal_id,
                                parent_subtask_id=parent.id)
        child2 = _make_subtask("c2", parent_goal_id=self.goal_id,
                                parent_subtask_id=parent.id)
        self.ledger.push(parent)
        self.ledger.push(child1)
        child1.complete(tick=1)
        self.ledger.pop()
        self.ledger.push(child2)
        child2.complete(tick=2)
        self.ledger.pop()
        # Now pop the parent
        parent.complete(tick=3)
        self.ledger.pop()
        parent_record = self.ledger.history_for(self.goal_id)[0]
        self.assertIn(child1.id, parent_record.children_ids)
        self.assertIn(child2.id, parent_record.children_ids)

    def test_history_cleared_on_clear(self):
        t = _make_subtask(parent_goal_id=self.goal_id)
        self.ledger.push(t)
        t.complete(tick=1)
        self.ledger.pop()
        self.ledger.clear()
        self.assertEqual(self.ledger.history_for(self.goal_id), [])

    def test_history_for_unknown_id_returns_empty(self):
        self.assertEqual(self.ledger.history_for("nonexistent"), [])


# ---------------------------------------------------------------------------
# SubtaskLedger — escalation context queries
# ---------------------------------------------------------------------------

class TestSubtaskLedgerEscalationContext(unittest.TestCase):
    """
    Simulate the exact scenario from the design discussion:

        Goal1
        ├── Task1  ✓
        └── Task2  ✗ (escalated)
            ├── TaskA  ✓
            └── Task3  ✗ (escalated) ← stuck here
    """

    def setUp(self):
        self.ledger = SubtaskLedger()
        self.goal_id = "goal-1"

        # Task1 completes
        self.task1 = _make_subtask("mine iron ore", parent_goal_id=self.goal_id)
        self.ledger.push(self.task1)
        self.task1.complete(tick=100)
        self.ledger.pop()

        # Task2 starts (direct child of goal)
        self.task2 = _make_subtask("build smelter", parent_goal_id=self.goal_id)
        self.ledger.push(self.task2)

        # TaskA (child of Task2) completes
        self.taskA = _make_subtask("craft furnaces", parent_goal_id=self.goal_id,
                                    parent_subtask_id=self.task2.id)
        self.ledger.push(self.taskA)
        self.taskA.complete(tick=200)
        self.ledger.pop()

        # Task3 (child of Task2) gets stuck — escalated but not yet popped
        self.task3 = _make_subtask("place furnaces", parent_goal_id=self.goal_id,
                                    parent_subtask_id=self.task2.id)
        self.ledger.push(self.task3)
        # Task3 is ACTIVE (the live stack is now [Task2, Task3])

    def test_failure_chain_is_task2_then_task3(self):
        chain = self.ledger.failure_chain()
        self.assertEqual(len(chain), 2)
        self.assertIs(chain[0], self.task2)
        self.assertIs(chain[1], self.task3)

    def test_failure_chain_is_empty_when_stack_empty(self):
        ledger = SubtaskLedger()
        self.assertEqual(ledger.failure_chain(), [])

    def test_sibling_history_has_task1_under_goal(self):
        chain = self.ledger.failure_chain()
        siblings = self.ledger.sibling_history(chain, self.goal_id)
        self.assertIn(self.goal_id, siblings)
        self.assertEqual(len(siblings[self.goal_id]), 1)
        self.assertEqual(siblings[self.goal_id][0].subtask.description, "mine iron ore")
        self.assertEqual(siblings[self.goal_id][0].outcome, "complete")

    def test_sibling_history_has_taskA_under_task2(self):
        chain = self.ledger.failure_chain()
        siblings = self.ledger.sibling_history(chain, self.goal_id)
        self.assertIn(self.task2.id, siblings)
        self.assertEqual(len(siblings[self.task2.id]), 1)
        self.assertEqual(siblings[self.task2.id][0].subtask.description, "craft furnaces")

    def test_sibling_history_omits_levels_with_no_resolved_siblings(self):
        """Task3 has no resolved children yet, so task3.id should not appear."""
        chain = self.ledger.failure_chain()
        siblings = self.ledger.sibling_history(chain, self.goal_id)
        self.assertNotIn(self.task3.id, siblings)

    def test_stuck_at_goal_level_gives_empty_chain_and_empty_siblings(self):
        ledger = SubtaskLedger()
        chain = ledger.failure_chain()
        siblings = ledger.sibling_history(chain, self.goal_id)
        self.assertEqual(chain, [])
        self.assertEqual(siblings, {})

    def test_sibling_history_returns_copies(self):
        chain = self.ledger.failure_chain()
        siblings = self.ledger.sibling_history(chain, self.goal_id)
        siblings[self.goal_id].append("garbage")
        # Original history should be unchanged
        self.assertEqual(
            len(self.ledger.history_for(self.goal_id)), 1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)