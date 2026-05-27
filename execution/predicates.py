"""
execution/predicates.py

Pure world-state observation predicates.

These functions answer "is the world currently in state X?" — they observe
WorldQuery and return a bool. They are called during execution (by agents,
skills, and the coordinator) to check current conditions, not to decide
whether to start an action.

Distinction from preconditions.py
----------------------------------
preconditions.py answers "can we start action X given current state?" —
richer checks that may involve KnowledgeBase, inventory arithmetic, or
recipe lookups. Those are called before committing to a task or action.

predicates.py answers "is the world in state X right now?" — pure,
cheap observation over WorldQuery only, no KnowledgeBase. These are
called during tick loops to detect arrival, check positioning, and gate
immediate action dispatch.

Rules
-----
- No LLM calls. No RCON. No mutations. No KnowledgeBase.
- All functions take WorldQuery as their primary argument.
- All functions return bool.
- No side effects — safe to call any number of times per tick.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from world import Position

if TYPE_CHECKING:
    from world import WorldQuery


def is_at(
    target: Position,
    wq: "WorldQuery",
    tolerance: float = 1.0,
) -> bool:
    """
    True if the player is within *tolerance* tiles of *target*.

    The default tolerance of 1.0 tile is sufficient for most interaction
    purposes without requiring pixel-perfect positioning.

    Used by agents to detect arrival at a bare position target — for example,
    after navigating to a frontier chunk centre during exploration.
    """
    player_pos = wq.player_position()
    dx = player_pos.x - target.x
    dy = player_pos.y - target.y
    return math.sqrt(dx * dx + dy * dy) <= tolerance


def is_reachable(entity_id: int, wq: "WorldQuery") -> bool:
    """
    True if *entity_id* is in the player's current reachable set.

    The reachable set is populated by the bridge from Factorio's reach radius
    as reported by the Lua mod — it reflects the actual game-engine reach,
    not a hardcoded Python constant.

    Used by agents to decide whether to interact with an entity immediately
    or navigate closer first, and as a cut-off condition for NavigateSkill
    when navigating toward entity targets.

    Returns False if the entity is not present in the scan or not reachable.
    """
    return entity_id in wq.state.player.reachable


def can_mine(entity_id: int, wq: "WorldQuery") -> bool:
    """
    True if *entity_id* is present in the current scan and reachable.

    A convenience wrapper: an entity that is not in the scan cannot be mined
    even if it appears reachable by id. This can happen when the player has
    moved since the last scan and the entity is now outside scan radius.

    Used by agents before issuing MineEntity actions.
    """
    if wq.entity_by_id(entity_id) is None:
        return False
    return is_reachable(entity_id, wq)


def player_has_item(item: str, count: int, wq: "WorldQuery") -> bool:
    """
    True if the player currently holds at least *count* units of *item*.

    Thin wrapper over WorldQuery.inventory_count for use in tick-loop
    completion checks without importing WorldQuery directly in agents.
    """
    return wq.inventory_count(item) >= count
