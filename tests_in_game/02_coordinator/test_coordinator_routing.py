"""
tests_in_game/02_coordinator/test_coordinator_routing.py

Verifies that the coordinator correctly routes goal types to their handlers,
returns STUCK for stub/unimplemented types, and recovers cleanly.

Key constraint: GoalQueueEntry has no params field. The coordinator always
receives frame.params={} (empty dict). Handlers that require params (item,
tech, bbox, etc.) now safe-fail with STUCK immediately via .get() defaults.
This means goals like collection, crafting, research, and clear_region will
always STUCK at the coordinator level — completion can only come from the
RewardEvaluator firing the success_condition independently.

What is tested here
--------------------
  - Unknown goal type → STUCK immediately, goal fails
  - Stub goal types (production, logistics, byproduct) → STUCK, goal fails
  - research goal → STUCK (missing 'tech' param), goal fails
  - prep_region → STUCK (missing 'bbox' param), goal fails
  - exploration → completes normally (no required params, uses .get() default)
  - Loop recovery: STUCK goal followed by exploration goal runs cleanly
"""

import pytest
from planning import GoalQueueEntry


# ---------------------------------------------------------------------------
# Unknown goal type
# ---------------------------------------------------------------------------

def test_unknown_goal_type_goes_stuck(run_goal):
    """
    A goal_type with no coordinator handler returns STUCK immediately.
    After max_stuck_retries the goal fails. Loop does not hang.
    """
    entry = GoalQueueEntry(
        description="Unknown goal type test",
        goal_type="totally_unknown_goal_type_xyz",
        success_condition="False",
        failure_condition="elapsed_ticks > 1200",
    )
    stats, wq = run_goal(entry)

    assert stats.goals_failed >= 1, (
        f"Unknown goal type should have caused a failure via STUCK. stats={stats}"
    )
    assert stats.stuck_events >= 1, (
        f"Expected at least one STUCK event for unknown goal type. stats={stats}"
    )


# ---------------------------------------------------------------------------
# Stub goal types — immediate STUCK from handlers
# ---------------------------------------------------------------------------

def test_production_goal_stubs_to_stuck(run_goal):
    """
    GOAL_PRODUCTION is a Phase 8 stub. Handler marks frame.failed immediately.
    Goal fails after max_stuck_retries. Loop does not hang.
    """
    entry = GoalQueueEntry(
        description="Production goal (Phase 8 stub)",
        goal_type="production",
        success_condition="False",
        failure_condition="elapsed_ticks > 1200",
    )
    stats, wq = run_goal(entry)

    resolved = stats.goals_completed + stats.goals_failed
    assert resolved == 1, (
        f"Production stub did not resolve within time limit. stats={stats}"
    )
    assert stats.goals_failed >= 1, (
        f"Production stub should fail — not implemented. stats={stats}"
    )


def test_logistics_goal_stubs_to_stuck(run_goal):
    """
    GOAL_LOGISTICS is a Phase 9 stub. Handler marks frame.failed immediately.
    Goal fails after max_stuck_retries. Loop does not hang.
    """
    entry = GoalQueueEntry(
        description="Logistics goal (Phase 9 stub)",
        goal_type="logistics",
        success_condition="False",
        failure_condition="elapsed_ticks > 1200",
    )
    stats, wq = run_goal(entry)

    resolved = stats.goals_completed + stats.goals_failed
    assert resolved == 1, (
        f"Logistics stub did not resolve within time limit. stats={stats}"
    )
    assert stats.goals_failed >= 1, (
        f"Logistics stub should fail — not implemented. stats={stats}"
    )


def test_byproduct_goal_stubs_to_stuck(run_goal):
    """
    GOAL_BYPRODUCT is a Phase 8 stub. Same expectation as logistics.
    """
    entry = GoalQueueEntry(
        description="Byproduct goal (Phase 8 stub)",
        goal_type="byproduct",
        success_condition="False",
        failure_condition="elapsed_ticks > 1200",
    )
    stats, wq = run_goal(entry)

    resolved = stats.goals_completed + stats.goals_failed
    assert resolved == 1, (
        f"Byproduct stub did not resolve within time limit. stats={stats}"
    )
    assert stats.goals_failed >= 1, (
        f"Byproduct stub should fail — not implemented. stats={stats}"
    )


# ---------------------------------------------------------------------------
# Goals that STUCK due to missing params
# ---------------------------------------------------------------------------

def test_research_goal_runs_stub_handler(run_goal):
    """
    The research handler is partially implemented: it skips the stub steps
    (science production, lab count, logistics checks) and pushes a
    set_research_queue task at step 4.

    The loop extracts {'tech': 'automation'} from the success_condition, so
    the handler receives the tech name correctly. The set_research_queue task
    routes to the crafting agent (placeholder — no research agent yet), which
    logs an error about missing targets but does not crash.

    The goal resolves in one of two ways:
      - Immediately: automation is already unlocked → evaluator fires success
      - Via failure_condition: the task silently no-ops, nothing researches,
        the elapsed_ticks cap fires and the goal fails

    Either resolution is acceptable. The test asserts only that the loop
    does not crash and that the goal resolves within the time limit.
    """
    entry = GoalQueueEntry(
        description="Research automation (stub handler test)",
        goal_type="research",
        success_condition="tech_unlocked('automation')",
        failure_condition="elapsed_ticks > 60",
    )
    stats, wq = run_goal(entry)

    resolved = stats.goals_completed + stats.goals_failed
    assert resolved == 1, (
        f"Research goal did not resolve within time limit. "
        f"Loop may be hanging on the set_research_queue task. stats={stats}"
    )
    if stats.goals_completed == 1:
        assert wq.tech_unlocked("automation"), (
            "Research goal completed but automation not unlocked — "
            "success_condition evaluator inconsistency."
        )


def test_prep_region_stubs_to_stuck_without_params(run_goal):
    """
    The prep_region handler reads frame.params.get('bbox'). With no params,
    bbox=None triggers the safe-fail path and returns STUCK.
    """
    entry = GoalQueueEntry(
        description="Prep region at spawn (STUCK: no bbox param)",
        goal_type="prep_region",
        success_condition="False",
        failure_condition="elapsed_ticks > 3600",
    )
    stats, wq = run_goal(entry)

    resolved = stats.goals_completed + stats.goals_failed
    assert resolved == 1, (
        f"prep_region goal did not resolve. Loop may be hanging. stats={stats}"
    )
    assert stats.goals_failed >= 1, (
        f"Expected goals_failed >= 1 (missing bbox param). stats={stats}"
    )


# ---------------------------------------------------------------------------
# Exploration — works without params (uses .get() default of 0)
# ---------------------------------------------------------------------------

def test_exploration_goal_completes(run_goal):
    """
    Exploration is the one goal type whose handler uses .get() with a default
    (target_chunks=0 when absent). With target=0, charted_chunks >= 0 is
    immediately true and the goal completes in one tick.

    This test verifies that exploration is the reliable baseline goal type
    for all other tests that need a warm-up run.
    """
    entry = GoalQueueEntry(
        description="Explore (should complete immediately with no target)",
        goal_type="exploration",
        success_condition="charted_chunks >= 1",
        failure_condition="elapsed_ticks > 3600",
    )
    stats, wq = run_goal(entry)

    resolved = stats.goals_completed + stats.goals_failed
    assert resolved == 1, (
        f"Exploration goal did not resolve. stats={stats}"
    )
    # Either completed (charted_chunks >= 1) or failed (extreme edge case).
    # On any loaded map, charted_chunks >= 1 is always already true.
    assert stats.goals_completed == 1, (
        f"Expected exploration to complete (charted_chunks >= 1 on any loaded map). "
        f"stats={stats}"
    )


# ---------------------------------------------------------------------------
# Loop recovery: STUCK goal followed by working goal
# ---------------------------------------------------------------------------

def test_stuck_goal_followed_by_exploration_recovers(run_goals):
    """
    After a STUCK/failed goal, the loop must continue and process the next
    goal in the queue correctly. Tests the _request_next_goal path after
    _on_goal_failed is called.
    """
    stuck_goal = GoalQueueEntry(
        description="Logistics stub (will STUCK and fail)",
        goal_type="logistics",
        success_condition="False",
        failure_condition="elapsed_ticks > 1200",
    )
    recovery_goal = GoalQueueEntry(
        description="Exploration after STUCK (loop recovery check)",
        goal_type="exploration",
        success_condition="charted_chunks >= 1",
        failure_condition="elapsed_ticks > 3600",
    )
    stats, wq = run_goals([stuck_goal, recovery_goal])

    assert stats.goals_failed >= 1, (
        f"Stuck goal should have failed. stats={stats}"
    )
    assert stats.goals_completed >= 1, (
        f"Recovery exploration goal should have completed. stats={stats}"
    )
    assert stats.stuck_events >= 1, (
        f"Expected at least one STUCK event. stats={stats}"
    )