"""
tests/unit/planning/test_task.py

Tests for planning/tasks/task.py and planning/tasks/task_ledger.py

Organised in sections:
  1. TaskStatus — enum values and ordering
  2. Task — construction, lifecycle transitions, properties, repr
  3. TaskRecord — to_dict() correctness
  4. TaskLedger — push/pop/peek, history, escalation queries

Run with:  python -m pytest tests/unit/planning/test_task.py -v
"""

from __future__ import annotations

import unittest

from planning import Task, TaskStatus, TaskRecord, TaskLedger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task(
    description: str = "navigate to patch",
    success_condition: str = "is_at(target)",
    failure_condition: str = "elapsed_ticks > 1200",
    parent_goal_id: str = "goal-001",
    created_at: int = 0,
    derived_locally: bool = True,
    parent_task_id: str | None = None,
) -> Task:
    return Task(
        description=description,
        success_condition=success_condition,
        failure_condition=failure_condition,
        parent_goal_id=parent_goal_id,
        created_at=created_at,
        derived_locally=derived_locally,
        parent_task_id=parent_task_id,
    )


def _push_and_complete(ledger: TaskLedger, task: Task, tick: int = 10) -> Task:
    """Helper: push a task, complete it, pop it, return the popped task."""
    ledger.push(task)
    task.complete(tick)
    return ledger.pop()


def _push_and_fail(ledger: TaskLedger, task: Task, tick: int = 10) -> Task:
    ledger.push(task)
    task.fail(tick)
    return ledger.pop()


# ===========================================================================
# Section 1 — TaskStatus
# ===========================================================================

class TestTaskStatus(unittest.TestCase):

    def test_all_statuses_exist(self):
        for name in ("PENDING", "ACTIVE", "COMPLETE", "FAILED", "ESCALATED"):
            self.assertIsNotNone(TaskStatus[name])

    def test_pending_is_not_terminal(self):
        t = _task()
        self.assertFalse(t.is_terminal)

    def test_complete_is_terminal(self):
        t = _task()
        t.activate()
        t.complete(tick=10)
        self.assertTrue(t.is_terminal)

    def test_failed_is_terminal(self):
        t = _task()
        t.activate()
        t.fail(tick=10)
        self.assertTrue(t.is_terminal)

    def test_escalated_is_terminal(self):
        t = _task()
        t.activate()
        t.escalate(tick=10)
        self.assertTrue(t.is_terminal)


# ===========================================================================
# Section 2 — Task lifecycle
# ===========================================================================

class TestTaskConstruction(unittest.TestCase):

    def test_auto_id_generated(self):
        t = _task()
        self.assertEqual(len(t.id), 36)  # UUID4

    def test_unique_ids(self):
        self.assertNotEqual(_task().id, _task().id)

    def test_default_status_pending(self):
        self.assertEqual(_task().status, TaskStatus.PENDING)

    def test_resolved_at_none_initially(self):
        self.assertIsNone(_task().resolved_at)

    def test_parent_id_is_goal_id_when_no_parent_task(self):
        t = _task(parent_goal_id="goal-123")
        self.assertEqual(t.parent_id, "goal-123")

    def test_parent_id_is_task_id_when_nested(self):
        t = _task(parent_goal_id="goal-123", parent_task_id="task-456")
        self.assertEqual(t.parent_id, "task-456")


class TestTaskTransitions(unittest.TestCase):

    def test_pending_to_active(self):
        t = _task()
        t.activate()
        self.assertEqual(t.status, TaskStatus.ACTIVE)
        self.assertTrue(t.is_active)

    def test_active_to_complete(self):
        t = _task()
        t.activate()
        t.complete(tick=100)
        self.assertEqual(t.status, TaskStatus.COMPLETE)
        self.assertEqual(t.resolved_at, 100)

    def test_active_to_failed(self):
        t = _task()
        t.activate()
        t.fail(tick=200)
        self.assertEqual(t.status, TaskStatus.FAILED)
        self.assertEqual(t.resolved_at, 200)

    def test_active_to_escalated(self):
        t = _task()
        t.activate()
        t.escalate(tick=300)
        self.assertEqual(t.status, TaskStatus.ESCALATED)
        self.assertEqual(t.resolved_at, 300)

    def test_activate_non_pending_raises(self):
        t = _task()
        t.activate()
        with self.assertRaises(RuntimeError):
            t.activate()

    def test_complete_non_active_raises(self):
        t = _task()
        with self.assertRaises(RuntimeError):
            t.complete(tick=10)

    def test_fail_non_active_raises(self):
        t = _task()
        with self.assertRaises(RuntimeError):
            t.fail(tick=10)

    def test_escalate_non_active_raises(self):
        t = _task()
        with self.assertRaises(RuntimeError):
            t.escalate(tick=10)

    def test_complete_already_complete_raises(self):
        t = _task()
        t.activate()
        t.complete(10)
        with self.assertRaises(RuntimeError):
            t.complete(20)

    def test_fail_already_failed_raises(self):
        t = _task()
        t.activate()
        t.fail(10)
        with self.assertRaises(RuntimeError):
            t.fail(20)


class TestTaskProperties(unittest.TestCase):

    def test_is_active_false_when_pending(self):
        self.assertFalse(_task().is_active)

    def test_is_active_true_when_active(self):
        t = _task()
        t.activate()
        self.assertTrue(t.is_active)

    def test_is_active_false_after_complete(self):
        t = _task()
        t.activate()
        t.complete(10)
        self.assertFalse(t.is_active)

    def test_derived_locally_stored(self):
        self.assertTrue(_task(derived_locally=True).derived_locally)
        self.assertFalse(_task(derived_locally=False).derived_locally)

    def test_repr_contains_status_and_desc(self):
        t = _task(description="go to ore patch")
        r = repr(t)
        self.assertIn("PENDING", r)
        self.assertIn("go to ore patch", r)

    def test_repr_shows_local_origin(self):
        self.assertIn("local", repr(_task(derived_locally=True)))

    def test_repr_shows_injected_origin(self):
        self.assertIn("injected", repr(_task(derived_locally=False)))


# ===========================================================================
# Section 3 — TaskRecord
# ===========================================================================

class TestTaskRecord(unittest.TestCase):

    def _completed_record(self) -> TaskRecord:
        t = _task(description="mine coal", parent_goal_id="g-1")
        t.activate()
        t.complete(tick=500)
        return TaskRecord(task=t, outcome="complete")

    def test_to_dict_contains_id(self):
        r = self._completed_record()
        d = r.to_dict()
        self.assertEqual(d["id"], r.task.id)

    def test_to_dict_contains_description(self):
        d = self._completed_record().to_dict()
        self.assertEqual(d["description"], "mine coal")

    def test_to_dict_outcome(self):
        self.assertEqual(self._completed_record().to_dict()["outcome"], "complete")

    def test_to_dict_resolved_at(self):
        self.assertEqual(self._completed_record().to_dict()["resolved_at"], 500)

    def test_to_dict_derived_locally(self):
        self.assertTrue(self._completed_record().to_dict()["derived_locally"])

    def test_to_dict_children_ids_empty_by_default(self):
        self.assertEqual(self._completed_record().to_dict()["children_ids"], [])

    def test_to_dict_children_ids_populated(self):
        t = _task()
        t.activate()
        t.complete(10)
        r = TaskRecord(task=t, outcome="complete", children_ids=["child-1", "child-2"])
        self.assertEqual(r.to_dict()["children_ids"], ["child-1", "child-2"])

    def test_failed_outcome_string(self):
        t = _task()
        t.activate()
        t.fail(10)
        r = TaskRecord(task=t, outcome="failed")
        self.assertEqual(r.to_dict()["outcome"], "failed")

    def test_escalated_outcome_string(self):
        t = _task()
        t.activate()
        t.escalate(10)
        r = TaskRecord(task=t, outcome="escalated")
        self.assertEqual(r.to_dict()["outcome"], "escalated")


# ===========================================================================
# Section 4 — TaskLedger
# ===========================================================================

class TestTaskLedgerEmpty(unittest.TestCase):

    def setUp(self):
        self.ledger = TaskLedger()

    def test_empty_by_default(self):
        self.assertEqual(len(self.ledger), 0)
        self.assertFalse(bool(self.ledger))

    def test_peek_returns_none_when_empty(self):
        self.assertIsNone(self.ledger.peek())

    def test_pop_returns_none_when_empty(self):
        self.assertIsNone(self.ledger.pop())

    def test_failure_chain_empty_when_empty(self):
        self.assertEqual(self.ledger.failure_chain(), [])

    def test_history_for_empty_when_nothing_resolved(self):
        self.assertEqual(self.ledger.history_for("goal-001"), [])

    def test_repr_shows_zero_depth(self):
        self.assertIn("0", repr(self.ledger))


class TestTaskLedgerPush(unittest.TestCase):

    def setUp(self):
        self.ledger = TaskLedger()

    def test_push_activates_task(self):
        t = _task()
        self.ledger.push(t)
        self.assertEqual(t.status, TaskStatus.ACTIVE)

    def test_pushed_task_is_peeked(self):
        t = _task()
        self.ledger.push(t)
        self.assertIs(self.ledger.peek(), t)

    def test_push_non_pending_raises(self):
        t = _task()
        t.activate()
        with self.assertRaises(ValueError):
            self.ledger.push(t)

    def test_len_increments_on_push(self):
        self.ledger.push(_task())
        self.assertEqual(len(self.ledger), 1)
        self.ledger.push(_task())
        self.assertEqual(len(self.ledger), 2)

    def test_second_push_suspends_first(self):
        """Pushing a prerequisite suspends the current top."""
        t1 = _task(description="outer")
        t2 = _task(description="prereq")
        self.ledger.push(t1)
        self.ledger.push(t2)
        # t2 is ACTIVE (top); t1 is back to PENDING (suspended)
        self.assertEqual(t2.status, TaskStatus.ACTIVE)
        self.assertEqual(t1.status, TaskStatus.PENDING)

    def test_third_push_stacks_correctly(self):
        t1, t2, t3 = _task(), _task(), _task()
        self.ledger.push(t1)
        self.ledger.push(t2)
        self.ledger.push(t3)
        self.assertEqual(len(self.ledger), 3)
        self.assertIs(self.ledger.peek(), t3)


class TestTaskLedgerPop(unittest.TestCase):

    def setUp(self):
        self.ledger = TaskLedger()

    def test_pop_non_terminal_raises(self):
        t = _task()
        self.ledger.push(t)
        # t is ACTIVE but not terminal
        with self.assertRaises(RuntimeError):
            self.ledger.pop()

    def test_pop_returns_completed_task(self):
        t = _task()
        self.ledger.push(t)
        t.complete(tick=10)
        popped = self.ledger.pop()
        self.assertIs(popped, t)

    def test_pop_decrements_len(self):
        t = _task()
        self.ledger.push(t)
        t.complete(10)
        self.ledger.pop()
        self.assertEqual(len(self.ledger), 0)

    def test_pop_resumes_suspended_task(self):
        """Popping a prerequisite should re-activate the task below it."""
        t1 = _task(description="outer")
        t2 = _task(description="prereq")
        self.ledger.push(t1)
        self.ledger.push(t2)
        t2.complete(tick=10)
        self.ledger.pop()
        # t1 should be ACTIVE again
        self.assertEqual(t1.status, TaskStatus.ACTIVE)
        self.assertIs(self.ledger.peek(), t1)

    def test_pop_records_in_history(self):
        t = _task(parent_goal_id="goal-001")
        self.ledger.push(t)
        t.complete(10)
        self.ledger.pop()
        history = self.ledger.history_for("goal-001")
        self.assertEqual(len(history), 1)
        self.assertIs(history[0].task, t)
        self.assertEqual(history[0].outcome, "complete")

    def test_pop_failed_records_outcome(self):
        t = _task(parent_goal_id="goal-001")
        self.ledger.push(t)
        t.fail(10)
        self.ledger.pop()
        self.assertEqual(self.ledger.history_for("goal-001")[0].outcome, "failed")

    def test_pop_escalated_records_outcome(self):
        t = _task(parent_goal_id="goal-001")
        self.ledger.push(t)
        t.escalate(10)
        self.ledger.pop()
        self.assertEqual(self.ledger.history_for("goal-001")[0].outcome, "escalated")


class TestTaskLedgerHistory(unittest.TestCase):

    def setUp(self):
        self.ledger = TaskLedger()
        self.goal_id = "goal-001"

    def test_multiple_resolved_tasks_all_recorded(self):
        for i in range(3):
            t = _task(description=f"task-{i}", parent_goal_id=self.goal_id)
            _push_and_complete(self.ledger, t, tick=i * 10)
        self.assertEqual(len(self.ledger.history_for(self.goal_id)), 3)

    def test_history_ordered_by_resolution(self):
        descriptions = ["first", "second", "third"]
        for desc in descriptions:
            t = _task(description=desc, parent_goal_id=self.goal_id)
            _push_and_complete(self.ledger, t)
        history = self.ledger.history_for(self.goal_id)
        self.assertEqual(
            [r.task.description for r in history], descriptions
        )

    def test_history_returns_copy(self):
        """Mutating the returned list does not affect the ledger."""
        t = _task(parent_goal_id=self.goal_id)
        _push_and_complete(self.ledger, t)
        history = self.ledger.history_for(self.goal_id)
        history.clear()
        self.assertEqual(len(self.ledger.history_for(self.goal_id)), 1)

    def test_history_for_unknown_parent_is_empty(self):
        self.assertEqual(self.ledger.history_for("nonexistent"), [])

    def test_nested_task_recorded_under_parent_task_id(self):
        """A task whose parent is another task is recorded under that task's id."""
        outer = _task(description="outer", parent_goal_id=self.goal_id)
        self.ledger.push(outer)

        inner = _task(
            description="inner",
            parent_goal_id=self.goal_id,
            parent_task_id=outer.id,
        )
        _push_and_complete(self.ledger, inner)

        # inner is recorded under outer.id, not goal_id
        self.assertEqual(len(self.ledger.history_for(outer.id)), 1)
        self.assertEqual(len(self.ledger.history_for(self.goal_id)), 0)

    def test_children_ids_populated_when_child_resolved_first(self):
        """If a child resolves before the parent is popped, its id appears
        in the parent's TaskRecord.children_ids."""
        outer = _task(parent_goal_id=self.goal_id)
        self.ledger.push(outer)

        inner = _task(parent_goal_id=self.goal_id, parent_task_id=outer.id)
        inner_id = inner.id
        _push_and_complete(self.ledger, inner)

        # Now complete and pop the outer task
        outer.complete(tick=20)
        self.ledger.pop()

        outer_record = self.ledger.history_for(self.goal_id)[0]
        self.assertIn(inner_id, outer_record.children_ids)


class TestTaskLedgerClear(unittest.TestCase):

    def test_clear_empties_stack(self):
        ledger = TaskLedger()
        ledger.push(_task())
        ledger.clear()
        self.assertEqual(len(ledger), 0)

    def test_clear_empties_history(self):
        ledger = TaskLedger()
        t = _task(parent_goal_id="goal-001")
        _push_and_complete(ledger, t)
        ledger.clear()
        self.assertEqual(ledger.history_for("goal-001"), [])

    def test_clear_on_empty_does_not_raise(self):
        TaskLedger().clear()


class TestTaskLedgerEscalationQueries(unittest.TestCase):
    """
    Simulate the structure the coordinator uses when building a StuckContext:

        Goal1
        ├── Task1  ✓  (history under goal_id)
        └── Task2  ✗  (live stack, bottom)
            ├── TaskA  ✓  (history under task2.id)
            └── Task3  ✗  (live stack, top — stuck here)
    """

    def setUp(self):
        self.ledger = TaskLedger()
        self.goal_id = "goal-001"

        # Task1 resolves first (sibling of Task2)
        self.task1 = _task("Task1", parent_goal_id=self.goal_id)
        _push_and_complete(self.ledger, self.task1)

        # Task2 is pushed and stays active (stuck later)
        self.task2 = _task("Task2", parent_goal_id=self.goal_id)
        self.ledger.push(self.task2)

        # TaskA resolves (child of Task2)
        self.taskA = _task("TaskA", parent_goal_id=self.goal_id,
                           parent_task_id=self.task2.id)
        _push_and_complete(self.ledger, self.taskA)

        # Task3 is pushed and stays active (stuck)
        self.task3 = _task("Task3", parent_goal_id=self.goal_id,
                           parent_task_id=self.task2.id)
        self.ledger.push(self.task3)

    def test_failure_chain_is_outermost_to_innermost(self):
        chain = self.ledger.failure_chain()
        self.assertEqual(len(chain), 2)
        self.assertIs(chain[0], self.task2)
        self.assertIs(chain[1], self.task3)

    def test_failure_chain_top_is_active(self):
        chain = self.ledger.failure_chain()
        self.assertEqual(chain[-1].status, TaskStatus.ACTIVE)

    def test_sibling_history_at_goal_level(self):
        chain = self.ledger.failure_chain()
        siblings = self.ledger.sibling_history(chain, self.goal_id)
        # Goal level: Task1 completed before Task2 was pushed
        self.assertIn(self.goal_id, siblings)
        self.assertEqual(len(siblings[self.goal_id]), 1)
        self.assertIs(siblings[self.goal_id][0].task, self.task1)

    def test_sibling_history_at_task2_level(self):
        chain = self.ledger.failure_chain()
        siblings = self.ledger.sibling_history(chain, self.goal_id)
        # Task2 level: TaskA completed before Task3 was pushed
        self.assertIn(self.task2.id, siblings)
        self.assertEqual(len(siblings[self.task2.id]), 1)
        self.assertIs(siblings[self.task2.id][0].task, self.taskA)

    def test_sibling_history_leaf_not_included(self):
        """The leaf (stuck) task has no resolved siblings — its level is omitted."""
        chain = self.ledger.failure_chain()
        siblings = self.ledger.sibling_history(chain, self.goal_id)
        self.assertNotIn(self.task3.id, siblings)

    def test_sibling_history_empty_when_no_siblings(self):
        """If nothing resolved before the stuck task, sibling_history is empty."""
        ledger = TaskLedger()
        t = _task(parent_goal_id="g-new")
        ledger.push(t)
        chain = ledger.failure_chain()
        self.assertEqual(ledger.sibling_history(chain, "g-new"), {})

    def test_len_reflects_live_stack_only(self):
        # Stack has Task2 (bottom) and Task3 (top) = 2
        self.assertEqual(len(self.ledger), 2)

    def test_repr_contains_stack_depth_and_history_count(self):
        r = repr(self.ledger)
        self.assertIn("TaskLedger", r)
        self.assertIn("2", r)  # stack depth


if __name__ == "__main__":
    unittest.main(verbosity=2)
