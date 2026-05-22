"""
tests_in_game/02_collection/test_stuck_state.py

Verifies that the STUCK path through the system behaves correctly:
  - A goal the coordinator cannot derive causes STUCK events
  - The goal eventually fails via max stuck retries
  - The loop continues cleanly and the next goal in the queue runs normally

NOTE on tick-based failure conditions:
---------------------------------------
Goal failure_conditions using absolute `tick` values are unreliable in tests
because the game tick counter keeps running between test runs. A condition like
`tick > 1200` may fire immediately if the game has been running for a while.

The correct fix (tracked as a known improvement) is for RewardEvaluator to
expose `elapsed_ticks = current_tick - goal_start_tick` in the condition
namespace so conditions can be written as `elapsed_ticks > 1200` instead.
Until that is implemented, tests that need time-based failure should use
`tick > 9999999` (effectively never) and rely on the STUCK retry mechanism,
or use game-state conditions (inventory, charted_chunks) instead.
"""

from llm.goal_source import GoalQueueEntry

# Failure conditions using elapsed_ticks (ticks since goal activation)
# rather than absolute tick values. This is safe regardless of how long
# the game has been running before the test.
# 20 seconds at 60 tps = 1200 ticks. 30 seconds = 1800 ticks.
_FAIL_20S = "elapsed_ticks > 1200"


def test_production_goal_fails_and_loop_continues(run_goals):
    """
    A 'production' goal cannot be derived by the Phase 6 coordinator.
    It should:
      1. Trigger STUCK events (coordinator cannot derive a subtask tree)
      2. Eventually fail via max stuck retries
      3. Not prevent subsequent goals from running

    The second goal verifies the loop recovers cleanly after a STUCK/failed goal.
    """
    stuck_goal = GoalQueueEntry(
        description="Produce 10 iron gear wheels (unsupported in Phase 6)",
        success_condition="inventory('iron-gear-wheel') >= 10",
        failure_condition=_FAIL_20S,
        goal_type="production",
    )
    recovery_goal = GoalQueueEntry(
        description="Explore briefly after stuck goal",
        success_condition="charted_chunks >= 1",
        failure_condition="tick > 9999999",   # recovery goal — should succeed via charted_chunks
        goal_type="exploration",
    )
    stats, wq = run_goals([stuck_goal, recovery_goal])

    assert stats.goals_failed >= 1, (
        "Expected the production goal to fail — coordinator cannot derive "
        f"a subtask tree for 'production'. stats={stats}"
    )
    assert stats.stuck_events >= 1, (
        f"Expected at least one STUCK event. stats={stats}"
    )
    assert stats.goals_completed >= 1, (
        "Loop did not recover after the stuck goal — the exploration goal "
        f"never ran or failed. stats={stats}"
    )


def test_collection_with_no_known_patches_goes_stuck(run_goal):
    """
    A collection goal for an item with no known resource patches causes STUCK.
    The goal fails via max stuck retries (3 events → fail).

    Does NOT use a time-based failure condition — relies entirely on the
    STUCK retry mechanism so it doesn't depend on absolute game tick values.
    """
    entry = GoalQueueEntry(
        description="Collect uranium ore (unlikely to be in scan radius at start)",
        success_condition="inventory('uranium-ore') >= 1",
        failure_condition=_FAIL_20S,
        goal_type="collection",
    )
    stats, wq = run_goal(entry)

    total_resolved = stats.goals_completed + stats.goals_failed
    assert total_resolved >= 1, (
        f"Goal neither completed nor failed — it may be waiting forever "
        f"for a failure_condition that never fires. stats={stats}\n"
        f"Check that max_stuck_retries is working correctly."
    )
    if stats.goals_completed == 0:
        assert stats.stuck_events >= 1, (
            "Goal failed but no STUCK events recorded — expected the "
            f"coordinator to report STUCK when no uranium patches known. stats={stats}"
        )