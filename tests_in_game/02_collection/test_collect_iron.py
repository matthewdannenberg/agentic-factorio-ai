"""
tests_in_game/02_collection/test_collect_iron.py

Verifies the full collection pipeline: coordinator derives approach + gather
subtasks, NavigationAgent walks to the patch, MiningAgent mines the ore.
"""

from llm.goal_source import GoalQueueEntry


def test_collect_10_iron_ore(run_goal):
    """
    Happy path: collect 10 iron ore from a patch in scan radius.

    Failure condition is 3 minutes — generous for 10 ore but short enough
    that a broken agent fails fast.
    """
    entry = GoalQueueEntry(
        description="Collect 10 iron ore",
        success_condition="inventory('iron-ore') >= 10",
        failure_condition="tick > 10800",   # 3 minutes
        goal_type="collection",
    )
    stats, wq = run_goal(entry)

    assert stats.goals_completed == 1, (
        f"Goal did not complete (completed={stats.goals_completed}, "
        f"failed={stats.goals_failed}, stuck_events={stats.stuck_events})"
    )
    assert wq.inventory_count("iron-ore") >= 10


def test_collect_iron_then_copper(run_goals):
    """
    Two sequential collection goals sharing KB state.
    Verifies that the agent can complete consecutive goals cleanly —
    the second goal starts with a fresh coordinator but the KB and
    inventory from the first goal persist.
    """
    iron_entry = GoalQueueEntry(
        description="Collect 5 iron ore",
        success_condition="inventory('iron-ore') >= 5",
        failure_condition="tick > 7200",
        goal_type="collection",
    )
    copper_entry = GoalQueueEntry(
        description="Collect 5 copper ore",
        success_condition="inventory('copper-ore') >= 5",
        failure_condition="tick > 7200",
        goal_type="collection",
    )
    stats, wq = run_goals([iron_entry, copper_entry])

    assert stats.goals_completed == 2, (
        f"Expected 2 goals completed, got {stats.goals_completed} "
        f"(failed={stats.goals_failed})"
    )
    assert wq.inventory_count("iron-ore") >= 5
    assert wq.inventory_count("copper-ore") >= 5
