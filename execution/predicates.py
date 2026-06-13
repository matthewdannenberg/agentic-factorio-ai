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
    from world import WorldQuery, KnowledgeBase, EntityState, Position


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
    True if *entity_id* (a sys_id) is in the player's current reachable set.

    The reachable set is populated by WorldWriter each poll: placed entities
    are included if their Factorio unit_number was in the bridge-reported reach
    set; natural objects are included if their position is within
    player.reach_distance tiles of the player. All entries are sys_ids.

    Used by agents to decide whether to interact with an entity immediately
    or navigate closer first, and as a gate for immediate action dispatch.
    """
    return entity_id in wq.state.player.reachable


def can_mine(entity_id: int, wq: "WorldQuery") -> bool:
    """
    True if *entity_id* is present in the accumulated entity list and reachable.

    Both conditions are checked as a defensive guard: reachability implies the
    entity was within reach distance on the most recent poll, but checking
    entity presence additionally guards against sys_ids that have since been
    removed from the entity list (confirmed gone by the writer) but whose
    sys_id might theoretically linger in a stale reachable set.

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

def is_present(sys_id: int, wq: "WorldQuery") -> bool:
    """
    True if the entity with the given sys_id currently exists in the world.

    Works uniformly for both placed entities and natural objects — all entities
    now carry stable sys_ids assigned by WorldWriter and live in the unified
    wq.state.entities list. The old two-case logic (entity_id=0 proximity scan
    for natural objects) is no longer needed.

    Used by DestroySkill to detect completion and by MiningAgent to detect
    that a harvest target has been cleared.
    """
    return wq.entity_by_id(sys_id) is not None


def can_destroy(obj: "EntityState", kb: "KnowledgeBase") -> bool:
    """
    True if the player can destroy this natural object with a plain MineEntity
    action (no special item or technology required).

    Takes an EntityState with is_natural=True. The check is made against the
    KnowledgeBase's EntityRecord, which stores prototype mineable_properties.minable
    learned at runtime from fa.get_entity_prototype. This works correctly for
    any total-conversion mod — nothing is hardcoded about entity names.

    The KB record is the sole authority. Unknown entities (no record or
    placeholder) are conservatively treated as not destroyable until the
    KB learns the prototype via ensure_entity().

    Returns False when:
    - KB returns None or a placeholder: prototype not yet learned.
    - record.minable=False: entity requires special handling (Phase 7),
      e.g. cliffs require cliff explosives via UseItemOnEntity.
    """
    record = kb.get_entity(obj.name)
    if record is None or record.is_placeholder:
        return False   # unknown entity — conservative
    return record.minable