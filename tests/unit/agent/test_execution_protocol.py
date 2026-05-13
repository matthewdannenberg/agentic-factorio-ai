"""
tests/unit/agent/test_execution_protocol.py

Tests for agent/execution_protocol.py

Run with:  python -m unittest tests.unit.agent.test_execution_protocol
"""

from __future__ import annotations

import unittest

from bridge.actions import NoOp, Wait
from planning.goal import make_goal
from agent.execution_protocol import (
    ExecutionLayerProtocol,
    ExecutionResult,
    ExecutionStatus,
    StuckContext,
)
from agent.subtask import Subtask, SubtaskRecord, SubtaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_goal(desc: str = "test goal"):
    return make_goal(
        description=desc,
        success_condition="True",
        failure_condition="False",
    )


def _make_subtask(
    description: str = "do something",
    parent_goal_id: str = "goal-1",
    parent_subtask_id: str | None = None,
) -> Subtask:
    return Subtask(
        description=description,
        success_condition="True",
        failure_condition="False",
        parent_goal_id=parent_goal_id,
        created_at=0,
        derived_locally=True,
        parent_subtask_id=parent_subtask_id,
    )


def _active_subtask(**kwargs) -> Subtask:
    t = _make_subtask(**kwargs)
    t.activate()
    return t


def _make_record(subtask: Subtask, outcome="complete") -> SubtaskRecord:
    return SubtaskRecord(subtask=subtask, outcome=outcome)


# ---------------------------------------------------------------------------
# ExecutionStatus
# ---------------------------------------------------------------------------

class TestExecutionStatus(unittest.TestCase):
    def test_four_statuses(self):
        names = {s.name for s in ExecutionStatus}
        self.assertSetEqual(names, {"PROGRESSING", "WAITING", "STUCK", "COMPLETE"})

    def test_all_distinct(self):
        vals = list(ExecutionStatus)
        self.assertEqual(len(vals), len(set(vals)))


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

class TestExecutionResult(unittest.TestCase):
    def _stuck_ctx(self, goal):
        return StuckContext(
            goal=goal,
            failure_chain=[],
            sibling_history={},
            blackboard_snapshot={},
        )

    def test_progressing_with_actions(self):
        result = ExecutionResult(
            actions=[NoOp(), Wait(ticks=10)],
            status=ExecutionStatus.PROGRESSING,
        )
        self.assertEqual(len(result.actions), 2)
        self.assertIsNone(result.stuck_context)

    def test_waiting_empty_actions(self):
        result = ExecutionResult(actions=[], status=ExecutionStatus.WAITING)
        self.assertEqual(result.actions, [])
        self.assertIsNone(result.stuck_context)

    def test_stuck_requires_context(self):
        with self.assertRaises(ValueError):
            ExecutionResult(actions=[], status=ExecutionStatus.STUCK)

    def test_non_stuck_rejects_context(self):
        goal = _make_goal()
        ctx = self._stuck_ctx(goal)
        with self.assertRaises(ValueError):
            ExecutionResult(actions=[], status=ExecutionStatus.WAITING,
                            stuck_context=ctx)

    def test_stuck_with_valid_context(self):
        goal = _make_goal()
        ctx = self._stuck_ctx(goal)
        result = ExecutionResult(actions=[], status=ExecutionStatus.STUCK,
                                 stuck_context=ctx)
        self.assertIsNotNone(result.stuck_context)
        self.assertEqual(result.stuck_context.goal.id, goal.id)


# ---------------------------------------------------------------------------
# StuckContext
# ---------------------------------------------------------------------------

class TestStuckContextGoalLevel(unittest.TestCase):
    """Stuck at goal level — no subtasks derived yet."""

    def setUp(self):
        self.goal = _make_goal("establish iron production")
        self.ctx = StuckContext(
            goal=self.goal,
            failure_chain=[],
            sibling_history={},
            blackboard_snapshot={"obs": {"k": "v"}},
        )

    def test_stuck_at_goal_level_true(self):
        self.assertTrue(self.ctx.stuck_at_goal_level)

    def test_immediate_failure_is_none(self):
        self.assertIsNone(self.ctx.immediate_failure)

    def test_to_dict_goal_fields(self):
        d = self.ctx.to_dict()
        self.assertEqual(d["goal_id"], self.goal.id)
        self.assertEqual(d["goal_description"], "establish iron production")
        self.assertEqual(d["failure_chain"], [])
        self.assertEqual(d["sibling_history"], {})
        self.assertEqual(d["blackboard_snapshot"]["obs"]["k"], "v")


class TestStuckContextWithChain(unittest.TestCase):
    """
    Simulate: Goal1 → [Task1✓, Task2(stuck)] / Task2 → [TaskA✓, Task3(stuck)]
    """

    def setUp(self):
        self.goal = _make_goal("Goal1")
        gid = self.goal.id

        task1 = _make_subtask("mine iron", parent_goal_id=gid)
        task1.activate(); task1.complete(tick=50)

        self.task2 = _active_subtask(description="build smelter", parent_goal_id=gid)

        taskA = _make_subtask("craft furnaces", parent_goal_id=gid,
                               parent_subtask_id=self.task2.id)
        taskA.activate(); taskA.complete(tick=150)

        self.task3 = _active_subtask(description="place furnaces",
                                     parent_goal_id=gid,
                                     parent_subtask_id=self.task2.id)

        self.ctx = StuckContext(
            goal=self.goal,
            failure_chain=[self.task2, self.task3],
            sibling_history={
                gid: [_make_record(task1, "complete")],
                self.task2.id: [_make_record(taskA, "complete")],
            },
            blackboard_snapshot={},
        )

    def test_stuck_at_goal_level_false(self):
        self.assertFalse(self.ctx.stuck_at_goal_level)

    def test_immediate_failure_is_task3(self):
        self.assertIs(self.ctx.immediate_failure, self.task3)

    def test_failure_chain_length(self):
        self.assertEqual(len(self.ctx.failure_chain), 2)
        self.assertIs(self.ctx.failure_chain[0], self.task2)
        self.assertIs(self.ctx.failure_chain[1], self.task3)

    def test_to_dict_failure_chain(self):
        d = self.ctx.to_dict()
        chain = d["failure_chain"]
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0]["description"], "build smelter")
        self.assertEqual(chain[1]["description"], "place furnaces")
        self.assertEqual(chain[0]["status"], "ACTIVE")

    def test_to_dict_sibling_history_has_task1(self):
        d = self.ctx.to_dict()
        goal_siblings = d["sibling_history"][self.goal.id]
        self.assertEqual(len(goal_siblings), 1)
        self.assertEqual(goal_siblings[0]["description"], "mine iron")
        self.assertEqual(goal_siblings[0]["outcome"], "complete")

    def test_to_dict_sibling_history_has_taskA(self):
        d = self.ctx.to_dict()
        task2_siblings = d["sibling_history"][self.task2.id]
        self.assertEqual(len(task2_siblings), 1)
        self.assertEqual(task2_siblings[0]["description"], "craft furnaces")

    def test_to_dict_is_serialisable(self):
        import json
        d = self.ctx.to_dict()
        # Should not raise — all values must be JSON-compatible types
        json.dumps(d)


# ---------------------------------------------------------------------------
# ExecutionLayerProtocol
# ---------------------------------------------------------------------------

class TestExecutionLayerProtocol(unittest.TestCase):
    def test_base_methods_raise(self):
        proto = ExecutionLayerProtocol()
        goal = _make_goal()
        with self.assertRaises(NotImplementedError):
            proto.reset(goal=goal, wq=None)
        with self.assertRaises(NotImplementedError):
            proto.tick(goal=goal, wq=None, ww=None, tick=0)
        with self.assertRaises(NotImplementedError):
            proto.progress(goal=goal, wq=None)
        with self.assertRaises(NotImplementedError):
            proto.observe(wq=None)

    def test_reset_accepts_seed_subtasks(self):
        """reset() should accept the optional seed_subtasks parameter."""
        class ConcreteExecution(ExecutionLayerProtocol):
            def reset(self, goal, wq, seed_subtasks=None):
                self.last_seeds = seed_subtasks
            def tick(self, goal, wq, ww, tick):
                return ExecutionResult(actions=[], status=ExecutionStatus.WAITING)
            def progress(self, goal, wq): return 0.0
            def observe(self, wq): return {}

        impl = ConcreteExecution()
        goal = _make_goal()
        seed = [_make_subtask("injected task")]
        impl.reset(goal=goal, wq=None, seed_subtasks=seed)
        self.assertEqual(impl.last_seeds, seed)

    def test_concrete_subclass_satisfies_protocol(self):
        class ConcreteExecution(ExecutionLayerProtocol):
            def reset(self, goal, wq, seed_subtasks=None): pass
            def tick(self, goal, wq, ww, tick):
                return ExecutionResult(actions=[], status=ExecutionStatus.WAITING)
            def progress(self, goal, wq): return 0.5
            def observe(self, wq): return {"x": 1}

        impl = ConcreteExecution()
        self.assertIsInstance(impl, ExecutionLayerProtocol)
        result = impl.tick(goal=_make_goal(), wq=None, ww=None, tick=0)
        self.assertEqual(result.status, ExecutionStatus.WAITING)
        self.assertAlmostEqual(impl.progress(_make_goal(), None), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)