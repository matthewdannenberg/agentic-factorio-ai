"""
tests/unit/planning/test_goal_tree.py

Tests for planning/goal_tree.py

Run with:  python -m pytest tests/unit/planning/test_goal_tree.py -v
       or: python tests/unit/planning/test_goal_tree.py
"""

from __future__ import annotations

import unittest

from planning.goal import GoalStatus, Priority, make_goal
from planning.goal_tree import GoalTree


def _goal(desc="test goal", priority=Priority.NORMAL, parent_id=None):
    return make_goal(
        description=desc,
        success_condition="False",
        failure_condition="False",
        priority=priority,
        parent_id=parent_id,
    )


class TestGoalTreeEmpty(unittest.TestCase):
    def test_active_goal_none_when_empty(self):
        tree = GoalTree()
        self.assertIsNone(tree.active_goal())

    def test_complete_active_on_empty_does_not_raise(self):
        tree = GoalTree()
        result = tree.complete_active(tick=100)
        self.assertIsNone(result)

    def test_fail_active_on_empty_does_not_raise(self):
        tree = GoalTree()
        result = tree.fail_active(tick=100)
        self.assertIsNone(result)

    def test_all_goals_empty(self):
        self.assertEqual(GoalTree().all_goals(), [])

    def test_pending_goals_empty(self):
        self.assertEqual(GoalTree().pending_goals(), [])


class TestGoalTreeSingleGoal(unittest.TestCase):
    def test_add_goal_makes_it_active(self):
        tree = GoalTree()
        g = _goal()
        tree.add_goal(g)
        self.assertIs(tree.active_goal(), g)

    def test_added_goal_status_is_active(self):
        tree = GoalTree()
        g = _goal()
        tree.add_goal(g)
        self.assertEqual(g.status, GoalStatus.ACTIVE)

    def test_complete_active_returns_none_when_no_next(self):
        tree = GoalTree()
        g = _goal()
        tree.add_goal(g)
        result = tree.complete_active(tick=100)
        self.assertIsNone(result)

    def test_completed_goal_status(self):
        tree = GoalTree()
        g = _goal()
        tree.add_goal(g)
        tree.complete_active(tick=100)
        self.assertEqual(g.status, GoalStatus.COMPLETE)

    def test_fail_active_marks_failed(self):
        tree = GoalTree()
        g = _goal()
        tree.add_goal(g)
        tree.fail_active(tick=200, reason="timeout")
        self.assertEqual(g.status, GoalStatus.FAILED)

    def test_after_completion_tree_is_idle(self):
        tree = GoalTree()
        g = _goal()
        tree.add_goal(g)
        tree.complete_active(tick=10)
        self.assertIsNone(tree.active_goal())


class TestGoalTreePreemption(unittest.TestCase):
    def test_lower_priority_does_not_preempt(self):
        tree = GoalTree()
        g_normal = _goal("normal", Priority.NORMAL)
        g_bg = _goal("background", Priority.BACKGROUND)
        tree.add_goal(g_normal)
        tree.add_goal(g_bg)
        self.assertIs(tree.active_goal(), g_normal)

    def test_lower_priority_stays_pending(self):
        tree = GoalTree()
        g_normal = _goal("normal", Priority.NORMAL)
        g_bg = _goal("background", Priority.BACKGROUND)
        tree.add_goal(g_normal)
        tree.add_goal(g_bg)
        self.assertEqual(g_bg.status, GoalStatus.PENDING)

    def test_higher_priority_preempts(self):
        tree = GoalTree()
        g_normal = _goal("normal", Priority.NORMAL)
        g_urgent = _goal("urgent", Priority.URGENT)
        tree.add_goal(g_normal)
        tree.add_goal(g_urgent)
        self.assertIs(tree.active_goal(), g_urgent)

    def test_preempted_goal_is_suspended(self):
        tree = GoalTree()
        g_normal = _goal("normal", Priority.NORMAL)
        g_urgent = _goal("urgent", Priority.URGENT)
        tree.add_goal(g_normal)
        tree.add_goal(g_urgent)
        self.assertEqual(g_normal.status, GoalStatus.SUSPENDED)

    def test_completing_preemptor_resumes_suspended(self):
        tree = GoalTree()
        g_normal = _goal("normal", Priority.NORMAL)
        g_urgent = _goal("urgent", Priority.URGENT)
        tree.add_goal(g_normal)
        tree.add_goal(g_urgent)
        tree.complete_active(tick=100)
        self.assertIs(tree.active_goal(), g_normal)
        self.assertEqual(g_normal.status, GoalStatus.ACTIVE)

    def test_failing_preemptor_resumes_suspended(self):
        tree = GoalTree()
        g_normal = _goal("normal", Priority.NORMAL)
        g_urgent = _goal("urgent", Priority.URGENT)
        tree.add_goal(g_normal)
        tree.add_goal(g_urgent)
        tree.fail_active(tick=100)
        self.assertIs(tree.active_goal(), g_normal)

    def test_lifo_resume_order(self):
        """Three goals: normal → urgent preempts → emergency preempts.
        On emergency complete, urgent resumes. On urgent complete, normal resumes."""
        tree = GoalTree()
        g_normal = _goal("normal", Priority.NORMAL)
        g_urgent = _goal("urgent", Priority.URGENT)
        g_emergency = _goal("emergency", Priority.EMERGENCY)
        tree.add_goal(g_normal)
        tree.add_goal(g_urgent)
        tree.add_goal(g_emergency)

        self.assertIs(tree.active_goal(), g_emergency)
        tree.complete_active(tick=10)
        self.assertIs(tree.active_goal(), g_urgent)
        tree.complete_active(tick=20)
        self.assertIs(tree.active_goal(), g_normal)

    def test_equal_priority_does_not_preempt(self):
        tree = GoalTree()
        g1 = _goal("first", Priority.NORMAL)
        g2 = _goal("second", Priority.NORMAL)
        tree.add_goal(g1)
        tree.add_goal(g2)
        self.assertIs(tree.active_goal(), g1)
        self.assertEqual(g2.status, GoalStatus.PENDING)


class TestGoalTreePendingPromotion(unittest.TestCase):
    def test_pending_goal_activated_when_active_completes(self):
        tree = GoalTree()
        g1 = _goal("first", Priority.NORMAL)
        g2 = _goal("second", Priority.NORMAL)
        tree.add_goal(g1)
        tree.add_goal(g2)
        tree.complete_active(tick=10)
        self.assertIs(tree.active_goal(), g2)

    def test_highest_priority_pending_activated_first(self):
        tree = GoalTree()
        g_main = _goal("main", Priority.EMERGENCY)
        g_low = _goal("low", Priority.BACKGROUND)
        g_high = _goal("high", Priority.URGENT)
        tree.add_goal(g_main)
        tree.add_goal(g_low)
        tree.add_goal(g_high)
        tree.complete_active(tick=10)
        # g_high should activate before g_low
        self.assertIs(tree.active_goal(), g_high)


class TestGoalTreeSubgoals(unittest.TestCase):
    def test_parent_not_completed_while_children_pending(self):
        """
        A parent is not auto-completed until ALL its children are complete.
        We sequence: parent active → child1 preempts (emergency) → child2 preempts child1
        → child2 completes → child1 resumes → child1 completes → parent resumes.
        After child1 alone completes, child2 is still pending so parent must not complete.
        """
        tree = GoalTree()
        parent = _goal("parent", Priority.NORMAL)
        tree.add_goal(parent)

        # Two children both at EMERGENCY to preempt; child2 added second so it preempts child1
        child1 = _goal("child1", Priority.EMERGENCY, parent_id=parent.id)
        child2 = _goal("child2", Priority.EMERGENCY, parent_id=parent.id)
        tree.add_goal(child1)  # preempts parent
        tree.add_goal(child2)  # same priority — does NOT preempt child1 (equal, not greater)

        # child1 is active (child2 pending, parent suspended)
        self.assertIs(tree.active_goal(), child1)
        tree.complete_active(tick=10)  # child1 done → parent resumes (LIFO, child2 still pending)
        # parent resumes but child2 is still pending — parent must not be COMPLETE
        self.assertNotEqual(parent.status, GoalStatus.COMPLETE)

    def test_parent_auto_completes_when_all_children_done(self):
        tree = GoalTree()
        parent = _goal("parent", Priority.URGENT)
        tree.add_goal(parent)
        # Immediately suspend parent by adding higher-priority children
        child1 = _goal("child1", Priority.EMERGENCY, parent_id=parent.id)
        tree.add_goal(child1)
        tree.complete_active(tick=10)   # completes child1, resumes parent

        # Now complete the parent itself
        tree.complete_active(tick=20)
        self.assertEqual(parent.status, GoalStatus.COMPLETE)

    def test_goal_by_id_finds_child(self):
        tree = GoalTree()
        parent = _goal("parent")
        child = _goal("child", parent_id=parent.id)
        tree.add_goal(parent)
        tree.add_goal(child)
        found = tree.goal_by_id(child.id)
        self.assertIs(found, child)

    def test_goal_by_id_returns_none_for_unknown(self):
        tree = GoalTree()
        self.assertIsNone(tree.goal_by_id("nonexistent-uuid"))


class TestGoalTreeQueries(unittest.TestCase):
    def test_all_goals_includes_all_statuses(self):
        tree = GoalTree()
        g1 = _goal("g1", Priority.NORMAL)
        g2 = _goal("g2", Priority.URGENT)
        tree.add_goal(g1)
        tree.add_goal(g2)
        tree.complete_active(tick=10)  # g2 completes, g1 resumes
        all_g = tree.all_goals()
        self.assertIn(g1, all_g)
        self.assertIn(g2, all_g)

    def test_pending_goals_sorted_priority_descending(self):
        tree = GoalTree()
        g_active = _goal("active", Priority.EMERGENCY)
        g_bg = _goal("bg", Priority.BACKGROUND)
        g_normal = _goal("normal", Priority.NORMAL)
        g_urgent = _goal("urgent", Priority.URGENT)
        tree.add_goal(g_active)
        tree.add_goal(g_bg)
        tree.add_goal(g_normal)
        tree.add_goal(g_urgent)
        pending = tree.pending_goals()
        priorities = [p.priority for p in pending]
        self.assertEqual(priorities, sorted(priorities, reverse=True))

    def test_pending_goals_excludes_active(self):
        tree = GoalTree()
        g = _goal()
        tree.add_goal(g)
        self.assertEqual(tree.pending_goals(), [])

    def test_goal_by_id_finds_completed(self):
        tree = GoalTree()
        g = _goal()
        tree.add_goal(g)
        tree.complete_active(tick=5)
        found = tree.goal_by_id(g.id)
        self.assertIs(found, g)
        self.assertEqual(found.status, GoalStatus.COMPLETE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
