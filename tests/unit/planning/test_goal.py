"""
tests/planning/test_goal.py

Tests for planning/goal.py 

Run with:  python tests/planning/test_goal.py
"""

from __future__ import annotations

import unittest

from planning.goal import Priority, GoalStatus, RewardSpec, Goal, make_goal


class TestPriority(unittest.TestCase):
    def test_ordering(self):
        self.assertLess(Priority.BACKGROUND, Priority.NORMAL)
        self.assertLess(Priority.NORMAL, Priority.URGENT)
        self.assertLess(Priority.URGENT, Priority.EMERGENCY)

    def test_int_values(self):
        self.assertEqual(Priority.NORMAL, 1)
        self.assertEqual(Priority.EMERGENCY, 3)


class TestRewardSpec(unittest.TestCase):
    def test_defaults(self):
        s = RewardSpec()
        self.assertEqual(s.success_reward, 1.0)
        self.assertEqual(s.time_discount, 1.0)

    def test_invalid_time_discount_zero(self):
        with self.assertRaises(ValueError):
            RewardSpec(time_discount=0.0)

    def test_invalid_time_discount_over_one(self):
        with self.assertRaises(ValueError):
            RewardSpec(time_discount=1.1)

    def test_invalid_negative_penalty(self):
        with self.assertRaises(ValueError):
            RewardSpec(failure_penalty=-1.0)

    def test_no_decay_at_one(self):
        s = RewardSpec(success_reward=1.0, time_discount=1.0)
        self.assertEqual(s.discounted_success_reward(99999), 1.0)

    def test_decay_reduces_reward(self):
        s = RewardSpec(time_discount=0.999)
        self.assertLess(s.discounted_success_reward(1000), 1.0)
        self.assertEqual(s.discounted_success_reward(0), 1.0)


class TestGoal(unittest.TestCase):
    def _g(self, priority=Priority.NORMAL) -> Goal:
        return make_goal(
            description="Build 50 iron gears",
            success_condition="inventory_count('iron-gear-wheel') >= 50",
            failure_condition="game_time_seconds > 600",
            priority=priority,
        )

    def test_auto_id(self):
        g = self._g()
        self.assertEqual(len(g.id), 36)

    def test_unique_ids(self):
        self.assertNotEqual(self._g().id, self._g().id)

    def test_default_status_pending(self):
        self.assertEqual(self._g().status, GoalStatus.PENDING)

    def test_activate(self):
        g = self._g()
        g.activate(tick=100)
        self.assertEqual(g.status, GoalStatus.ACTIVE)
        self.assertEqual(g.created_at, 100)

    def test_created_at_preserved_on_resume(self):
        g = self._g()
        g.activate(tick=100)
        g.suspend()
        g.activate(tick=200)
        self.assertEqual(g.created_at, 100)

    def test_suspend_requires_active(self):
        with self.assertRaises(RuntimeError):
            self._g().suspend()

    def test_complete(self):
        g = self._g()
        g.activate(0)
        g.complete(300)
        self.assertEqual(g.status, GoalStatus.COMPLETE)
        self.assertEqual(g.resolved_at, 300)
        self.assertTrue(g.is_terminal)

    def test_fail(self):
        g = self._g()
        g.activate(0)
        g.fail(400)
        self.assertEqual(g.status, GoalStatus.FAILED)
        self.assertTrue(g.is_terminal)

    def test_double_activate_raises(self):
        g = self._g()
        g.activate(0)
        with self.assertRaises(RuntimeError):
            g.activate(10)

    def test_priority_ordering(self):
        self.assertGreater(
            self._g(Priority.EMERGENCY).priority,
            self._g(Priority.BACKGROUND).priority,
        )

    def test_is_active(self):
        g = self._g()
        self.assertFalse(g.is_active)
        g.activate(0)
        self.assertTrue(g.is_active)

    def test_make_goal_with_milestone(self):
        g = make_goal(
            "test", "True", "False",
            milestone_rewards={"coal >= 10": 0.1},
            parent_id="abc",
        )
        self.assertEqual(len(g.reward_spec.milestone_rewards), 1)
        self.assertEqual(g.parent_id, "abc")


if __name__ == "__main__":
    unittest.main(verbosity=2)