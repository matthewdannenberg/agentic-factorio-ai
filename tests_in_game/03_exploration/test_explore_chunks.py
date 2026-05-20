"""
tests_in_game/03_exploration/test_explore_chunks.py

Verifies that the exploration goal type drives the NavigationAgent to chart
new chunks until the target is met.
"""

from llm.goal_source import GoalQueueEntry


def test_explore_5_chunks(run_goal):
    """
    Chart at least 5 chunks. A fresh map starts with ~1 chunk charted around
    spawn; the agent needs to walk outward to chart more.

    5 chunks is intentionally modest — enough to verify the exploration
    subtask derivation and navigation work, without requiring a long run.
    """
    entry = GoalQueueEntry(
        description="Explore until 5 chunks charted",
        success_condition="charted_chunks >= 5",
        failure_condition="tick > 10800",   # 3 minutes
        goal_type="exploration",
    )
    stats, wq = run_goal(entry)

    assert stats.goals_completed == 1, (
        f"Exploration goal did not complete "
        f"(completed={stats.goals_completed}, failed={stats.goals_failed})"
    )
    assert wq.charted_chunks >= 5


def test_explore_then_collect(run_goals):
    """
    Explore first, then collect — verifies that after exploration the
    resource_map is populated enough for a collection goal to succeed.

    This is the closest we currently get to testing the "find then mine"
    scenario deferred from Phase 6 planning.
    """
    explore_entry = GoalQueueEntry(
        description="Explore until 5 chunks charted",
        success_condition="charted_chunks >= 5",
        failure_condition="tick > 10800",
        goal_type="exploration",
    )
    collect_entry = GoalQueueEntry(
        description="Collect 5 iron ore",
        success_condition="inventory('iron-ore') >= 5",
        failure_condition="tick > 10800",
        goal_type="collection",
    )
    stats, wq = run_goals([explore_entry, collect_entry])

    assert stats.goals_completed == 2, (
        f"Expected 2 goals completed, got {stats.goals_completed} "
        f"(failed={stats.goals_failed}, stuck={stats.stuck_events})"
    )
    assert wq.charted_chunks >= 5
    assert wq.inventory_count("iron-ore") >= 5
