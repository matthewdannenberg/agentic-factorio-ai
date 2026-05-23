"""
tests_in_game/02_collection/test_collect_iron.py

Verifies the full collection pipeline: coordinator derives approach + gather
subtasks, NavigationAgent walks to the patch, MiningAgent mines the ore.

Uses new.inventory() for tests that require active collection regardless of
prior inventory state, and inventory() for tests that only care about
having a total amount.

Notes on inventory persistence
--------------------------------
Inventory persists across goals within a session and across test runs. The
new.inventory() delta conditions are preferred because they require genuine
collection work regardless of what was already in the player's inventory.
"""

from llm.goal_source import GoalQueueEntry


def test_collect_5_new_iron_ore(run_goal):
    """
    Collect 5 new iron ore — requires active mining regardless of prior
    inventory. Uses new.inventory() so the test is meaningful even if the
    player already has iron ore.
    """
    entry = GoalQueueEntry(
        description="Collect 5 new iron ore",
        success_condition="new.inventory('iron-ore') >= 5",
        failure_condition="new.tick > 10800",   # 3 minutes
        goal_type="collection",
    )
    stats, wq = run_goal(entry)

    assert stats.goals_completed == 1, (
        f"Goal did not complete (completed={stats.goals_completed}, "
        f"failed={stats.goals_failed}, stuck_events={stats.stuck_events})"
    )


def test_collect_5_iron_ore_total(run_goal):
    """
    Have at least 5 iron ore total in inventory. Uses the absolute
    inventory() condition as a sanity check that the non-delta form
    still works correctly alongside the new framework.
    """
    entry = GoalQueueEntry(
        description="Have 5 iron ore total",
        success_condition="inventory('iron-ore') >= 5",
        failure_condition="new.tick > 10800",
        goal_type="collection",
    )
    stats, wq = run_goal(entry)

    assert stats.goals_completed == 1, (
        f"Goal did not complete (completed={stats.goals_completed}, "
        f"failed={stats.goals_failed}, stuck_events={stats.stuck_events})"
    )
    assert wq.inventory_count("iron-ore") >= 5


def test_collect_new_iron_then_new_copper(run_goals):
    """
    Collect 5 new iron ore then 5 new copper ore — both goals require
    active collection regardless of prior inventory.
    """
    iron_entry = GoalQueueEntry(
        description="Collect 5 new iron ore",
        success_condition="new.inventory('iron-ore') >= 5",
        failure_condition="new.tick > 10800",
        goal_type="collection",
    )
    copper_entry = GoalQueueEntry(
        description="Collect 5 new copper ore",
        success_condition="new.inventory('copper-ore') >= 5",
        failure_condition="new.tick > 18000",   # 5 minutes — patch may be far
        goal_type="collection",
    )
    stats, wq = run_goals([iron_entry, copper_entry])

    assert stats.goals_completed == 2, (
        f"Expected 2 goals completed, got {stats.goals_completed} "
        f"(failed={stats.goals_failed})"
    )