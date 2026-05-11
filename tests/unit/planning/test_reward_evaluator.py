"""
tests/unit/planning/test_reward_evaluator.py

Tests for planning/reward_evaluator.py

WorldState/WorldQuery construction is handled via fixtures helpers so that a
WorldState signature change does not require edits here.

Run with:  python tests/unit/planning/test_reward_evaluator.py
"""

from __future__ import annotations

import unittest

from planning.goal import RewardSpec, make_goal, Priority
from planning.reward_evaluator import EvaluationResult, RewardEvaluator
from world.query import WorldQuery
from world.state import Inventory, InventorySlot, PlayerState, Position, WorldState
from tests.fixtures import make_world_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec(milestones=None, success_reward=1.0, failure_penalty=0.5,
          time_discount=1.0) -> RewardSpec:
    return RewardSpec(
        success_reward=success_reward,
        failure_penalty=failure_penalty,
        milestone_rewards=milestones or {},
        time_discount=time_discount,
    )


def _eval(ev, success="False", failure="False", spec=None, wq=None,
          tick=0, start_tick=0):
    if spec is None:
        spec = _spec()
    if wq is None:
        wq = make_world_query()
    return ev.evaluate_conditions(
        success_condition=success,
        failure_condition=failure,
        spec=spec,
        wq=wq,
        tick=tick,
        start_tick=start_tick,
    )


class TestRewardEvaluatorSuccess(unittest.TestCase):
    def setUp(self):
        self.ev = RewardEvaluator()

    def test_true_condition_returns_success(self):
        result = _eval(self.ev, success="True")
        self.assertTrue(result.success)
        self.assertFalse(result.failure)

    def test_false_condition_no_success(self):
        self.assertFalse(_eval(self.ev, success="False").success)

    def test_success_adds_reward(self):
        result = _eval(self.ev, success="True", spec=_spec(success_reward=2.0))
        self.assertAlmostEqual(result.reward, 2.0)

    def test_no_success_no_reward(self):
        self.assertAlmostEqual(_eval(self.ev, success="False").reward, 0.0)


class TestRewardEvaluatorFailure(unittest.TestCase):
    def setUp(self):
        self.ev = RewardEvaluator()

    def test_true_failure_condition(self):
        result = _eval(self.ev, failure="True")
        self.assertTrue(result.failure)
        self.assertFalse(result.success)

    def test_failure_subtracts_penalty(self):
        spec = _spec(success_reward=1.0, failure_penalty=0.5, time_discount=1.0)
        result = _eval(self.ev, failure="True", spec=spec, tick=0, start_tick=0)
        self.assertAlmostEqual(result.reward, -0.5)

    def test_syntax_error_in_failure_condition_no_raise(self):
        try:
            result = _eval(self.ev, failure="this is not valid python !!!")
        except Exception as exc:
            self.fail(f"evaluate_conditions() raised unexpectedly: {exc}")
        self.assertFalse(result.failure)

    def test_name_error_in_failure_condition_no_raise(self):
        self.assertFalse(_eval(self.ev, failure="undefined_variable > 10").failure)

    def test_exception_in_success_condition_no_raise(self):
        self.assertFalse(_eval(self.ev, success="1/0 > 0").success)


class TestRewardEvaluatorMilestones(unittest.TestCase):
    def setUp(self):
        self.ev = RewardEvaluator()

    def test_milestone_that_triggers_adds_reward(self):
        spec = _spec(milestones={"True": 0.25})
        result = _eval(self.ev, spec=spec)
        self.assertIn("True", result.milestones_hit)
        self.assertAlmostEqual(result.reward, 0.25)

    def test_milestone_that_does_not_trigger(self):
        spec = _spec(milestones={"False": 0.25})
        result = _eval(self.ev, spec=spec)
        self.assertEqual(result.milestones_hit, [])
        self.assertAlmostEqual(result.reward, 0.0)

    def test_multiple_milestones_accumulate(self):
        spec = _spec(milestones={"True": 0.1, "1 == 1": 0.2})
        result = _eval(self.ev, spec=spec)
        self.assertAlmostEqual(result.reward, 0.3)
        self.assertEqual(len(result.milestones_hit), 2)

    def test_milestones_fire_independent_of_success(self):
        spec = _spec(milestones={"True": 0.5})
        result = _eval(self.ev, success="False", spec=spec)
        self.assertFalse(result.success)
        self.assertAlmostEqual(result.reward, 0.5)

    def test_milestone_exception_does_not_raise(self):
        spec = _spec(milestones={"bad syntax !!!": 0.1})
        try:
            result = _eval(self.ev, spec=spec)
        except Exception as exc:
            self.fail(f"evaluate_conditions() raised with bad milestone: {exc}")
        self.assertEqual(result.milestones_hit, [])


class TestRewardEvaluatorTimeDiscount(unittest.TestCase):
    def setUp(self):
        self.ev = RewardEvaluator()

    def test_no_discount_at_one(self):
        spec = _spec(success_reward=1.0, time_discount=1.0)
        result = _eval(self.ev, success="True", spec=spec, tick=99999, start_tick=0)
        self.assertAlmostEqual(result.reward, 1.0)

    def test_discount_below_one_reduces_reward(self):
        spec = _spec(success_reward=1.0, time_discount=0.999)
        result = _eval(self.ev, success="True", spec=spec, tick=1000, start_tick=0)
        self.assertLess(result.reward, 1.0)
        self.assertGreater(result.reward, 0.0)

    def test_discount_zero_elapsed_no_decay(self):
        spec = _spec(success_reward=1.0, time_discount=0.5)
        result = _eval(self.ev, success="True", spec=spec, tick=100, start_tick=100)
        self.assertAlmostEqual(result.reward, 1.0)

    def test_elapsed_ticks_computed_correctly(self):
        self.assertEqual(_eval(self.ev, tick=500, start_tick=100).elapsed_ticks, 400)

    def test_elapsed_ticks_negative_start_clamped_to_zero(self):
        self.assertEqual(_eval(self.ev, tick=50, start_tick=100).elapsed_ticks, 0)

    def test_milestones_not_discounted(self):
        spec = RewardSpec(milestone_rewards={"True": 0.5}, time_discount=0.9)
        self.assertAlmostEqual(_eval(self.ev, spec=spec, tick=0, start_tick=0).reward, 0.5)
        self.assertAlmostEqual(_eval(self.ev, spec=spec, tick=10000, start_tick=0).reward, 0.5)


class TestRewardEvaluatorViaGoal(unittest.TestCase):
    def setUp(self):
        self.ev = RewardEvaluator()

    def test_goal_success_condition_evaluated(self):
        goal = make_goal("test", "True", "False", Priority.NORMAL)
        result = self.ev.evaluate(goal, make_world_query(), tick=0, start_tick=0)
        self.assertTrue(result.success)

    def test_goal_failure_condition_evaluated(self):
        goal = make_goal("test", "False", "True", Priority.NORMAL)
        result = self.ev.evaluate(goal, make_world_query(), tick=0, start_tick=0)
        self.assertTrue(result.failure)

    def test_goal_milestone_evaluated(self):
        goal = make_goal("test", "False", "False", milestone_rewards={"True": 0.3})
        result = self.ev.evaluate(goal, make_world_query(), tick=0, start_tick=0)
        self.assertIn("True", result.milestones_hit)


class TestRewardEvaluatorNamespace(unittest.TestCase):
    def setUp(self):
        self.ev = RewardEvaluator()

    def test_inventory_shorthand(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot("iron-plate", 50)]),
        ))
        result = _eval(self.ev, success="inventory('iron-plate') >= 50",
                       wq=WorldQuery(ws))
        self.assertTrue(result.success)

    def test_inventory_shorthand_below_threshold(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot("iron-plate", 10)]),
        ))
        result = _eval(self.ev, success="inventory('iron-plate') >= 50",
                       wq=WorldQuery(ws))
        self.assertFalse(result.success)

    def test_tick_available_in_namespace(self):
        self.assertTrue(_eval(self.ev, success="tick >= 500", tick=600).success)

    def test_state_available_in_namespace(self):
        ws = WorldState(tick=1234)
        result = _eval(self.ev, success="state.tick == 1234", wq=WorldQuery(ws))
        self.assertTrue(result.success)

    def test_empty_worldstate_does_not_raise(self):
        try:
            result = _eval(self.ev, success="inventory('coal') > 0",
                           failure="tick > 9999")
        except Exception as exc:
            self.fail(f"evaluate_conditions() raised on empty WorldState: {exc}")
        self.assertFalse(result.success)
        self.assertFalse(result.failure)

    def test_empty_condition_strings_do_not_trigger(self):
        result = _eval(self.ev, success="", failure="   ")
        self.assertFalse(result.success)
        self.assertFalse(result.failure)

    def test_wq_available_in_namespace(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot("iron-plate", 50)]),
        ))
        result = _eval(self.ev, success="wq.inventory_count('iron-plate') >= 50",
                       wq=WorldQuery(ws))
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main(verbosity=2)