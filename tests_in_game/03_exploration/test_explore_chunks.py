"""
tests_in_game/03_exploration/test_explore_chunks.py

Verifies that the exploration goal type drives the NavigationAgent to chart
new chunks until the target is met.

Uses new_charted_chunks (chunks revealed since goal activation) rather than
charted_chunks (absolute total) so tests pass regardless of how many chunks
were already charted before the test runs.
"""

from llm.goal_source import GoalQueueEntry


def test_explore_5_new_chunks(run_goal):
    """
    Chart at least 5 new chunks. Uses new_charted_chunks so the test
    requires genuine exploration regardless of prior map state.
    """
    entry = GoalQueueEntry(
        description="Explore until 5 new chunks charted",
        success_condition="new.charted_chunks >= 5",
        failure_condition="elapsed_ticks > 10800",   # 3 minutes
        goal_type="exploration",
    )
    stats, wq = run_goal(entry)

    assert stats.goals_completed == 1, (
        f"Exploration goal did not complete "
        f"(completed={stats.goals_completed}, failed={stats.goals_failed})"
    )


def test_explore_then_collect(run_goals):
    """
    Explore first (chart 5 new chunks), then collect — verifies that after
    exploration the resource_map is populated enough for a collection goal.
    """
    explore_entry = GoalQueueEntry(
        description="Explore until 5 new chunks charted",
        success_condition="new.charted_chunks >= 5",
        failure_condition="elapsed_ticks > 10800",
        goal_type="exploration",
    )
    collect_entry = GoalQueueEntry(
        description="Collect 5 iron ore",
        success_condition="inventory('iron-ore') >= 5",
        failure_condition="elapsed_ticks > 10800",
        goal_type="collection",
    )
    stats, wq = run_goals([explore_entry, collect_entry])

    assert stats.goals_completed == 2, (
        f"Expected 2 goals completed, got {stats.goals_completed} "
        f"(failed={stats.goals_failed}, stuck={stats.stuck_events})"
    )
    assert wq.inventory_count("iron-ore") >= 5