"""
tests_in_game/03_exploration/test_clear_region.py

Verifies the clear_region goal type: the MiningAgent removes natural entities
(trees, rocks) from a bounding box around the player spawn.

The bounding box is deliberately generous (32×32 tiles around spawn) so it
almost certainly contains some trees or rocks on a default Factorio map.
The test succeeds if no natural entities remain in the box after the run.

Note: this test runs in 03_exploration/ because it depends on the player
being somewhere sensible — after exploration has oriented the agent — and
because clearing is a prerequisite for later construction tests.
"""

import pytest
from llm.goal_source import GoalQueueEntry


# A 32×32 box centred on spawn (0, 0). Generous enough that a default
# Factorio map will almost certainly contain trees or rocks within it.
_CLEAR_BBOX = {
    "x_min": -16.0,
    "y_min": -16.0,
    "x_max":  16.0,
    "y_max":  16.0,
}

# Natural entity name substrings — mirrors MiningAgent._is_natural()
_NATURAL_KEYWORDS = ("tree", "rock", "boulder", "cliff", "fish")


def _count_natural_entities(wq) -> int:
    """Count entities in the scan that look like natural world objects."""
    count = 0
    for entity in wq.state.entities:
        name = entity.name.lower()
        if any(k in name for k in _NATURAL_KEYWORDS):
            pos = entity.position
            if (
                _CLEAR_BBOX["x_min"] <= pos.x <= _CLEAR_BBOX["x_max"]
                and _CLEAR_BBOX["y_min"] <= pos.y <= _CLEAR_BBOX["y_max"]
            ):
                count += 1
    return count


def test_clear_natural_in_spawn_region(run_goal):
    """
    Clear all natural entities from a 32×32 region around spawn.

    Success condition is time-based: we allow 5 minutes for the agent to
    walk through the box and mine all visible trees/rocks. The post-run
    assertion checks that none remain in the scan.
    """
    entry = GoalQueueEntry(
        description="Clear natural entities from spawn region",
        success_condition="tick > 0",   # Always true after first poll —
                                        # the mining agent drives to completion;
                                        # the test asserts on final world state.
        failure_condition="tick > 18000",   # 5 minutes hard cap
        goal_type="clear_region",
        bounding_box=_CLEAR_BBOX,
        clear_mode="clear_natural",
    )
    stats, wq = run_goal(entry)

    # The goal should not have failed via the hard cap.
    assert stats.goals_failed == 0, (
        f"Clear goal hit the failure timeout. "
        f"The agent may be stuck or the bounding box is too large. stats={stats}"
    )

    remaining = _count_natural_entities(wq)
    assert remaining == 0, (
        f"{remaining} natural entities still present in the bounding box "
        "after the clear_natural run. The MiningAgent may not have processed "
        "all targets, or the scan radius didn't cover the full box."
    )


def test_clear_natural_removes_only_natural_entities(run_goal):
    """
    After a clear_natural run, verify that player-built entities (if any
    are in the box) were not removed. On a fresh spawn this is trivially
    satisfied since there are no player buildings — the test documents the
    expected behaviour for future environments with placed entities.

    This test is conservative: it simply checks that goals_failed == 0 and
    trusts that if a player entity were destroyed it would show up in
    state.recent_losses.
    """
    entry = GoalQueueEntry(
        description="Clear natural only — should not touch player entities",
        success_condition="tick > 0",
        failure_condition="tick > 18000",
        goal_type="clear_region",
        bounding_box=_CLEAR_BBOX,
        clear_mode="clear_natural",
    )
    stats, wq = run_goal(entry)

    assert stats.goals_failed == 0, (
        f"Clear goal failed. stats={stats}"
    )
    # Check no player-force entities were destroyed.
    player_losses = [
        e for e in wq.state.recent_losses
        if getattr(e, "force", "player") == "player"
    ]
    assert len(player_losses) == 0, (
        f"Player-owned entities were destroyed during clear_natural: "
        f"{[e.name for e in player_losses]}"
    )
