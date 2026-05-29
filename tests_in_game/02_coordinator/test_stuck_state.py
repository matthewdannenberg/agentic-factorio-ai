"""
tests_in_game/02_coordinator/test_stuck_state.py

Verifies that the STUCK path through the system behaves correctly:
  - A goal the coordinator cannot handle causes STUCK events
  - The goal eventually fails via max stuck retries
  - The loop continues cleanly and the next goal in the queue runs normally

STUCK behaviour by goal type (Phase 6)
----------------------------------------
  production  — _handle_production marks frame.failed immediately (Phase 8
                stub). After max_stuck_retries the loop fails the goal.

  collection  — _handle_collection returns STUCK when no resource patches
                are known for the requested item (wq.resources_of_type is
                empty). The loop retries up to max_stuck_retries, then fails.

Note: the loop now extracts coordinator params (item, count) from the
success_condition expression, so collection goals work end-to-end when the
resource is known — these tests deliberately use items unlikely to be in the
resource map to exercise the STUCK path.

Failure condition note
-----------------------
All failure_conditions use elapsed_ticks (ticks since goal activation) rather
than absolute tick values.
"""

from planning import GoalQueueEntry

# 20 seconds at 60 tps = 1200 ticks. 30 seconds = 1800 ticks.
_FAIL_20S = "elapsed_ticks > 1200"
_FAIL_30S = "elapsed_ticks > 1800"


def test_production_goal_fails_and_loop_continues(run_goals):
    """
    A 'production' goal cannot be derived by the Phase 6 coordinator.
    It should:
      1. Trigger STUCK events immediately (_handle_production marks frame.failed)
      2. The goal fails after max_stuck_retries
      3. The subsequent goal runs normally — loop recovered cleanly

    The recovery goal uses charted_chunks >= 1, which is true on any loaded
    map. It tests that the loop is still alive, not that new exploration occurs.
    """
    stuck_goal = GoalQueueEntry(
        description="Produce 10 iron gear wheels (unsupported in Phase 6)",
        goal_type="production",
        success_condition="inventory('iron-gear-wheel') >= 10",
        failure_condition=_FAIL_20S,
    )
    recovery_goal = GoalQueueEntry(
        description="Confirm loop is alive after stuck goal",
        goal_type="exploration",
        success_condition="charted_chunks >= 1",
        failure_condition="elapsed_ticks > 9999999",
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

    The loop extracts {"item": "uranium-ore", "count": 1} from the
    success_condition and passes it to the coordinator. _handle_collection
    then checks wq.resources_of_type('uranium-ore') — empty on a fresh spawn
    — and returns STUCK. After max_stuck_retries the loop fails the goal.

    Note: if uranium patches happen to be in the resource map from prior
    exploration, the coordinator will attempt mining and the goal may complete
    via the success_condition. That is a valid outcome and the test accepts it.
    """
    entry = GoalQueueEntry(
        description="Collect uranium ore (unlikely to have known patches)",
        goal_type="collection",
        success_condition="inventory('uranium-ore') >= 1",
        failure_condition=_FAIL_30S,
    )
    stats, wq = run_goal(entry)

    total_resolved = stats.goals_completed + stats.goals_failed
    assert total_resolved >= 1, (
        f"Goal neither completed nor failed — loop may be hanging. "
        f"stats={stats}\n"
        f"Check that max_stuck_retries is working correctly."
    )
    # Accept completion only if uranium genuinely in inventory.
    if stats.goals_completed == 1:
        assert wq.inventory_count("uranium-ore") >= 1, (
            "Goal completed but uranium-ore not in inventory — evaluator issue."
        )
        return
    assert stats.stuck_events >= 1, (
        f"Goal failed but no STUCK events recorded. stats={stats}"
    )
    assert stats.goals_failed >= 1, (
        f"Expected goals_failed >= 1 via STUCK retry path. stats={stats}"
    )