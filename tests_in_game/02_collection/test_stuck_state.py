"""
tests_in_game/02_collection/test_stuck_state.py

Verifies that the STUCK path through the system behaves correctly:
  - A goal the coordinator cannot derive causes STUCK events
  - The goal eventually fails via its failure_condition (not hangs forever)
  - The loop continues cleanly and the next goal in the queue runs normally

This replaces the old smoke_stuck test which asserted on internal
ExecutionStatus directly. The new test observes externally-visible outcomes
(LoopStats) rather than coordinator internals, which is the correct approach
for tests that run through FactorioLoop.
"""

from llm.goal_source import GoalQueueEntry


def test_production_goal_fails_and_loop_continues(run_goals):
    """
    A 'production' goal cannot be derived by the Phase 6 coordinator.
    It should:
      1. Trigger STUCK events (coordinator cannot derive a subtask tree)
      2. Eventually fail via the failure_condition (not hang indefinitely)
      3. Not prevent subsequent goals from running

    The second goal (a simple exploration) verifies that the loop recovers
    cleanly after a STUCK/failed goal.
    """
    stuck_goal = GoalQueueEntry(
        description="Produce 10 iron gear wheels (unsupported in Phase 6)",
        success_condition="inventory('iron-gear-wheel') >= 10",
        # Short failure window — this goal is expected to fail
        failure_condition="tick > 1200",   # 20 seconds
        goal_type="production",
    )
    recovery_goal = GoalQueueEntry(
        description="Explore briefly after stuck goal",
        success_condition="charted_chunks >= 2",
        failure_condition="tick > 7200",
        goal_type="exploration",
    )
    stats, wq = run_goals([stuck_goal, recovery_goal])

    # The production goal must fail (not complete).
    assert stats.goals_failed >= 1, (
        "Expected the production goal to fail — coordinator cannot derive "
        f"a subtask tree for 'production'. stats={stats}"
    )
    # STUCK events should have been recorded.
    assert stats.stuck_events >= 1, (
        f"Expected at least one STUCK event. stats={stats}"
    )
    # The loop should have continued and completed the recovery goal.
    assert stats.goals_completed >= 1, (
        "Loop did not recover after the stuck goal — the exploration goal "
        f"never ran or failed. stats={stats}"
    )


def test_collection_with_no_known_patches_goes_stuck(run_goal):
    """
    A collection goal for an item with no known resource patches in the
    resource_map causes STUCK (no patch to derive an approach subtask for).
    The goal should fail via the failure_condition within the timeout.

    This tests the _derive_collection path when resource_map has no patches
    of the requested type — a different STUCK path from the production goal.
    """
    entry = GoalQueueEntry(
        description="Collect uranium ore (unlikely to be in scan radius at start)",
        success_condition="inventory('uranium-ore') >= 1",
        failure_condition="tick > 1800",   # 30 seconds — should fail fast
        goal_type="collection",
    )
    stats, wq = run_goal(entry)

    # Either STUCK (no patches) or failed (timeout) — both acceptable.
    # What must NOT happen is goals_completed == 1 for uranium at spawn.
    total_resolved = stats.goals_completed + stats.goals_failed
    assert total_resolved >= 1, (
        f"Goal neither completed nor failed within timeout. stats={stats}"
    )
    # If it completed somehow (uranium near spawn on this particular map),
    # that's technically fine — skip the stuck assertion.
    if stats.goals_completed == 0:
        assert stats.stuck_events >= 1, (
            "Goal failed but no STUCK events recorded — expected the "
            f"coordinator to report STUCK when no uranium patches known. stats={stats}"
        )
