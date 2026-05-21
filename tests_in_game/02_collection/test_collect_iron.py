"""
tests_in_game/02_collection/test_collect_iron.py

Verifies the full collection pipeline: coordinator derives approach + gather
subtasks, NavigationAgent walks to the patch, MiningAgent mines the ore.

Notes on inventory persistence
--------------------------------
Inventory persists across goals within a session and across test runs (it lives
in the Factorio save). Thresholds must be set high enough that they cannot be
trivially satisfied by ore already in the player's inventory from earlier runs.
The single-goal test uses 50 iron ore for this reason. The sequential test uses
an absolute threshold (30 iron) that is deliberately higher than what a single
prior run would have collected, plus a long timeout for the copper patch which
may be far from spawn.
"""

from llm.goal_source import GoalQueueEntry


def test_collect_5_iron_ore(run_goal):
    """
    Happy path: collect 5 iron ore from a patch in scan radius.

    Failure condition is 3 minutes — generous for 10 ore but short enough
    that a broken agent fails fast.
    """
    entry = GoalQueueEntry(
        description="Collect 5 iron ore",
        success_condition="inventory('iron-ore') >= 5",
        failure_condition="tick > 10800",   # 3 minutes
        goal_type="collection",
    )
    stats, wq = run_goal(entry)

    assert stats.goals_completed == 1, (
        f"Goal did not complete (completed={stats.goals_completed}, "
        f"failed={stats.goals_failed}, stuck_events={stats.stuck_events})"
    )
    assert wq.inventory_count("iron-ore") >= 5


def test_collect_iron_then_copper(run_goals):
    """
    Two sequential collection goals sharing KB state.
    Verifies that the agent can complete consecutive goals cleanly —
    the second goal starts with a fresh coordinator but the KB and
    inventory from the first goal persist.

    Iron threshold is set to 10 so the goal requires active mining even if
    the player already has some iron from a previous test. Copper threshold
    is 5 (copper patches may be further from spawn, so keep the count low
    but the timeout generous).
    """
    iron_entry = GoalQueueEntry(
        description="Collect 10 iron ore",
        success_condition="inventory('iron-ore') >= 10",
        failure_condition="tick > 10800",   # 3 minutes
        goal_type="collection",
    )
    copper_entry = GoalQueueEntry(
        description="Collect 5 copper ore",
        success_condition="inventory('copper-ore') >= 5",
        failure_condition="tick > 18000",   # 5 minutes — patch may be far
        goal_type="collection",
    )
    stats, wq = run_goals([iron_entry, copper_entry])

    assert stats.goals_completed == 2, (
        f"Expected 2 goals completed, got {stats.goals_completed} "
        f"(failed={stats.goals_failed})"
    )
    assert wq.inventory_count("iron-ore") >= 10
    assert wq.inventory_count("copper-ore") >= 5