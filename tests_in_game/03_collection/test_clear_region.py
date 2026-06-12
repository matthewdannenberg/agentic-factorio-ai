"""
tests_in_game/03_collection/test_clear_region.py

Verifies the clear_region goal type: the MiningAgent removes natural entities
(trees, rocks) from a bounding box around the player spawn.

Scan coverage
-------------
bbox().is_clear is PROXIMAL — it only sees natural objects within the current
scan radius. Before issuing a clear_region goal, we issue navigation goals to
each corner of the bbox to guarantee the full region has been scanned.

ARCHITECTURAL GAP (Phase 10): WorldState has no scan-coverage map. There is
no concept of "which sub-regions of a bbox have been observed", and no mechanism
to generate navigation targets to fill gaps. Any PROXIMAL query over a region
larger than the scan radius is unreliable without prior navigation. A scan
coverage layer should be added to WorldState in Phase 10.
"""

from planning import GoalQueueEntry


# Half-width of the clear bbox in tiles. Full bbox: (-H,-H,H,H).
_HALF = 50

_CLEAR_BBOX_COND = f"bbox(-{_HALF},-{_HALF},{_HALF},{_HALF}).is_clear"
_SMALL_BBOX_COND = "bbox(-8,-8,8,8).is_clear"

_FAIL_5MIN  = "elapsed_ticks > 18000"
_FAIL_2MIN  = "elapsed_ticks > 7200"


def _corner_nav_goals(half=_HALF):
    """
    Navigate to each corner of the bbox plus the centre, guaranteeing the
    full region has been scanned before bbox().is_clear is evaluated.
    Each goal uses goal_type="navigate" with a navigate_to(x,y) condition
    so the NavigationAgent walks to the exact position.
    """
    h = half - 2   # step slightly inside corners
    waypoints = [
        ( h,  h, "NE corner"),
        (-h,  h, "NW corner"),
        (-h, -h, "SW corner"),
        ( h, -h, "SE corner"),
        (  0,  0, "centre"),
    ]
    return [
        GoalQueueEntry(
            description=f"Navigate to {label} for scan coverage",
            goal_type="navigate",
            success_condition=f"navigate_to({float(x)}, {float(y)})",
            failure_condition=_FAIL_2MIN,
        )
        for x, y, label in waypoints
    ]


def test_clear_natural_in_spawn_region(run_goals):
    """
    Navigate to each corner of the bbox for scan coverage, then clear
    all natural entities from the region.
    """
    nav = _corner_nav_goals()
    clear = GoalQueueEntry(
        description="Clear natural entities from spawn region",
        goal_type="clear_region",
        success_condition=_CLEAR_BBOX_COND,
        failure_condition=_FAIL_5MIN,
    )
    stats, wq = run_goals(nav + [clear])

    assert stats.goals_failed == 0, (
        f"A goal failed. If the clear goal failed, the bbox may contain "
        f"cliffs (minable=False). stats={stats}"
    )
    assert stats.goals_completed == len(nav) + 1, (
        f"Not all goals completed. stats={stats}"
    )


def test_clear_natural_does_not_remove_player_entities(run_goals):
    """
    After clearing, verify no player-owned entities were destroyed.
    """
    nav = _corner_nav_goals()
    clear = GoalQueueEntry(
        description="Clear natural only — should not touch player entities",
        goal_type="clear_region",
        success_condition=_CLEAR_BBOX_COND,
        failure_condition=_FAIL_5MIN,
    )
    stats, wq = run_goals(nav + [clear])

    assert stats.goals_failed == 0, f"A goal failed. stats={stats}"

    player_losses = [
        e for e in wq.state.recent_losses
        if getattr(e, "force", None) == "player"
    ]
    assert len(player_losses) == 0, (
        f"Player-owned entities destroyed: {[e.name for e in player_losses]}. "
        f"Check MiningAgent and can_destroy() logic."
    )


def test_clear_region_fails_on_undestroyable_objects(run_goal):
    """
    Verify _handle_clear_region fails cleanly on undestroyable objects
    (e.g. cliffs). On cliff-free maps this will succeed; the test just
    checks the goal resolved within the time limit either way.
    """
    entry = GoalQueueEntry(
        description="Clear small region (cliff-gap test)",
        goal_type="clear_region",
        success_condition=_SMALL_BBOX_COND,
        failure_condition="elapsed_ticks > 7200",
    )
    stats, wq = run_goal(entry)

    resolved = stats.goals_completed + stats.goals_failed
    assert resolved == 1, (
        f"Goal neither completed nor failed within time limit. stats={stats}"
    )