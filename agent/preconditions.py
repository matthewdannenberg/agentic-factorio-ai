"""
agent/preconditions.py

Precondition helpers — check action validity against the current WorldQuery.

Used by the coordinator and navigation agent to filter candidate actions before
dispatch, avoiding invalid or impossible RCON commands.

Rules
-----
- Pure predicate functions. No LLM calls. No RCON. No mutations.
- All functions return bool or a filtered list; none raise on missing data.
- Reach distance is derived exclusively from WorldQuery (player.reachable list),
  never from a hardcoded constant. is_reachable() is the canonical arrival
  criterion for the navigation agent.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from bridge.actions import (
    Action,
    CraftItem,
    EquipArmor,
    FlipEntity,
    MineEntity,
    MineResource,
    MoveTo,
    NoOp,
    PlaceEntity,
    RotateEntity,
    SetFilter,
    SetRecipe,
    SetResearchQueue,
    SetSplitterPriority,
    StopMovement,
    TransferItems,
    UseItem,
    Wait,
)
from world.state import Position

if TYPE_CHECKING:
    from world.query import WorldQuery
    from world.knowledge import KnowledgeBase, RecipeRecord

# The Factorio prototype category string for recipes that can be hand-crafted
# by the player character. All other categories require a specific machine.
_HAND_CRAFT_CATEGORY = "crafting"

# The entity name Factorio uses for the player character in made_in lists.
_CHARACTER_ENTITY = "character"


# ---------------------------------------------------------------------------
# Position / proximity
# ---------------------------------------------------------------------------

def is_at(target: Position, wq: "WorldQuery", tolerance: float = 1.0) -> bool:
    """
    True if the player is within *tolerance* tiles of *target*.

    Used by the coordinator to check whether a position-targeted movement
    subtask is complete. Tolerance defaults to 1.0 tile — sufficient for
    most interaction purposes without requiring pixel-perfect positioning.
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

    This is the canonical arrival criterion for the navigation agent. When
    is_reachable() returns True for the target entity, the navigator signals
    waypoint completion.

    Returns False if the entity is not present in the scan or not reachable.
    """
    return entity_id in wq.state.player.reachable


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def can_mine(entity_id: int, wq: "WorldQuery") -> bool:
    """
    True if the entity is present in the current scan and reachable.

    Delegates the reach check to is_reachable(), which derives reach from
    WorldQuery rather than a hardcoded constant.
    """
    entity = wq.entity_by_id(entity_id)
    if entity is None:
        return False
    return is_reachable(entity_id, wq)


# ---------------------------------------------------------------------------
# Crafting
# ---------------------------------------------------------------------------

def can_reach_count(
    item: str,
    target_count: int,
    wq: "WorldQuery",
    kb: "KnowledgeBase",
) -> bool:
    """
    True if the player already has *target_count* of *item* in inventory, or
    can reach that count by hand-crafting the deficit using items currently in
    inventory (recursively).

    This asks "can we get to target_count?" not "can we craft one?". If the
    player already has 40 of 50 needed, this checks whether the remaining 10
    can be hand-crafted.

    Hand-craftable means:
      - The recipe's category is "crafting" (the player character's category),
        AND "character" appears in the recipe's made_in list.
      - Recipes requiring machines (furnaces, assemblers, centrifuges, etc.)
        are never considered hand-craftable.

    Implementation: depth-first search over the recipe graph.

    A mutable budget dict (copy of player inventory) and a mutable visited set
    (items currently on the active DFS path) are passed through the recursion.
    Both are restored on backtrack, so each independent path through the recipe
    graph sees a clean slate for items it didn't personally consume or visit.
    This prevents the same resource being double-counted across sibling branches
    (e.g. iron-plate consumed both directly and via gears) while still cutting
    off genuine cycles within a single path.

    Parameters
    ----------
    item         : Factorio item name.
    target_count : Number of items to reach in inventory.
    wq           : Current WorldQuery (for inventory counts).
    kb           : KnowledgeBase (for recipe lookups).
    """
    budget: dict[str, int] = {}
    for slot in wq.state.player.inventory.slots:
        budget[slot.item] = budget.get(slot.item, 0) + slot.count

    visited: set[str] = set()
    return _can_reach_recursive(item, target_count, kb, budget, visited)


def _hand_craftable_recipes(
    item: str,
    kb: "KnowledgeBase",
) -> list["RecipeRecord"]:
    """
    Return all known hand-craftable recipes that produce *item*.

    A recipe is hand-craftable if and only if:
      1. Its category is "crafting" (the character crafting category).
      2. "character" appears in its made_in list.
      3. It is not a placeholder.
      4. It produces *item* as a non-fluid product.
    """
    candidates = kb.recipes_for_product(item)
    result = []
    for recipe in candidates:
        if recipe.is_placeholder:
            continue
        if recipe.category != _HAND_CRAFT_CATEGORY:
            continue
        if _CHARACTER_ENTITY not in recipe.made_in:
            continue
        if not any(p.name == item and not p.is_fluid for p in recipe.products):
            continue
        result.append(recipe)
    return result


def _can_reach_recursive(
    item: str,
    target_count: int,
    kb: "KnowledgeBase",
    budget: dict[str, int],
    visited: set[str],
) -> bool:
    """
    Recursive DFS core of can_reach_count.

    budget  : Mutable dict of available items. Decremented as ingredients are
              committed on the current path; restored on backtrack.
    visited : Mutable set of items currently on the active DFS path. An item
              in visited means we are already trying to produce it higher up
              the call stack — recursing into it again would be a cycle.
              Restored on backtrack so sibling paths are not affected.
    """
    have = budget.get(item, 0)

    if have >= target_count:
        # Consume from budget so siblings can't reuse these units.
        budget[item] = have - target_count
        return True

    # This item is already being resolved on the current path — a cycle.
    if item in visited:
        return False

    still_needed = target_count - have
    recipes = _hand_craftable_recipes(item, kb)

    if not recipes:
        return False

    # Commit whatever we already have; only the deficit needs crafting.
    budget[item] = 0
    visited.add(item)

    for recipe in recipes:
        budget_snapshot = dict(budget)
        if _recipe_can_cover(recipe, item, still_needed, kb, budget, visited):
            visited.discard(item)
            return True
        # This recipe failed — roll back budget and try the next.
        budget.clear()
        budget.update(budget_snapshot)

    # No recipe worked — restore the pre-committed inventory and visited state.
    budget[item] = have
    visited.discard(item)
    return False


def _recipe_can_cover(
    recipe: "RecipeRecord",
    item: str,
    still_needed: int,
    kb: "KnowledgeBase",
    budget: dict[str, int],
    visited: set[str],
) -> bool:
    """
    True if *recipe* can be run enough times to produce *still_needed* units
    of *item* given the current budget, committing ingredients as it goes.

    Returns False immediately if any ingredient cannot be satisfied. The caller
    is responsible for rolling back the budget if False is returned.
    """
    yield_per_run = 0.0
    for product in recipe.products:
        if product.name == item and not product.is_fluid:
            yield_per_run += product.amount * product.probability

    if yield_per_run <= 0.0:
        return False

    runs_needed = math.ceil(still_needed / yield_per_run)

    for ingredient in recipe.ingredients:
        if ingredient.is_fluid:
            return False
        total_needed = math.ceil(ingredient.amount * runs_needed)
        if not _can_reach_recursive(ingredient.name, total_needed, kb, budget, visited):
            return False

    return True


# ---------------------------------------------------------------------------
# Building / placement
# ---------------------------------------------------------------------------

def can_place(item: str, position: Position, wq: "WorldQuery") -> bool:
    """
    True if *item* is in the player's inventory and *position* is unoccupied.

    Position is considered unoccupied if no currently-scanned entity has its
    centre within 1 tile of the target. This is a conservative 1×1 check;
    large-entity collision detection is the spatial-logistics agent's job
    (Phase 9).
    """
    if wq.inventory_count(item) < 1:
        return False

    for entity in wq.state.entities:
        dx = abs(entity.position.x - position.x)
        dy = abs(entity.position.y - position.y)
        if dx < 1.0 and dy < 1.0:
            return False

    return True


# ---------------------------------------------------------------------------
# Action list filtering
# ---------------------------------------------------------------------------

def valid_actions(
    wq: "WorldQuery",
    kb: "KnowledgeBase",
    candidate_actions: list[Action],
) -> list[Action]:
    """
    Filter *candidate_actions* to those currently valid given the current
    WorldQuery and KnowledgeBase.

    Each action type is checked against its appropriate precondition. The
    intent of each entry is explicit:

      Checked here (non-trivial precondition worth enforcing):
        MineEntity        — entity must be present and reachable
        MineResource      — player must be within interaction range
        CraftItem         — player must have (or be able to craft) all ingredients
        PlaceEntity       — item must be in inventory, position must be clear
        RotateEntity      — entity must be reachable
        FlipEntity        — entity must be reachable
        SetRecipe         — entity must be reachable
        SetFilter         — entity must be reachable
        SetSplitterPriority — entity must be reachable
        TransferItems     — entity must be reachable
        EquipArmor        — item must be in player inventory
        UseItem           — item must be in player inventory

      Always valid (no useful precondition to check here):
        MoveTo            — pathfinder handles obstacles; always emittable
        StopMovement      — always valid
        SetResearchQueue  — validity is KB/tech-tree-dependent; checked by
                            the coordinator before construction, not here
        Wait              — always valid
        NoOp              — always valid

      Passed through (vehicle/combat stubs, unknown future types):
        Everything else   — bridge validates and returns ok=false on failure;
                            the execution layer gates these by ActionCategory
                            via actions_for_context() before they reach here

    Parameters
    ----------
    wq                : Current WorldQuery.
    kb                : KnowledgeBase for crafting checks.
    candidate_actions : Actions proposed by agents this tick.

    Returns
    -------
    Subset of *candidate_actions* that pass their precondition, in original order.
    """
    result: list[Action] = []
    for action in candidate_actions:

        # --- MINING ---
        if isinstance(action, MineEntity):
            if can_mine(action.entity_id, wq):
                result.append(action)

        elif isinstance(action, MineResource):
            # Resource tiles have no entity_id; check positional proximity.
            if is_at(action.position, wq, tolerance=2.0):
                result.append(action)

        # --- CRAFTING ---
        elif isinstance(action, CraftItem):
            if can_reach_count(action.recipe, action.count, wq, kb):
                result.append(action)

        # --- BUILDING ---
        elif isinstance(action, PlaceEntity):
            if can_place(action.item, action.position, wq):
                result.append(action)

        elif isinstance(action, RotateEntity):
            if is_reachable(action.entity_id, wq):
                result.append(action)

        elif isinstance(action, FlipEntity):
            if is_reachable(action.entity_id, wq):
                result.append(action)

        elif isinstance(action, SetRecipe):
            if is_reachable(action.entity_id, wq):
                result.append(action)

        elif isinstance(action, SetFilter):
            if is_reachable(action.entity_id, wq):
                result.append(action)

        elif isinstance(action, SetSplitterPriority):
            if is_reachable(action.entity_id, wq):
                result.append(action)

        # --- INVENTORY ---
        elif isinstance(action, TransferItems):
            if is_reachable(action.entity_id, wq):
                result.append(action)

        # --- PLAYER ---
        elif isinstance(action, EquipArmor):
            if wq.inventory_count(action.item) >= 1:
                result.append(action)

        elif isinstance(action, UseItem):
            if wq.inventory_count(action.item) >= 1:
                result.append(action)

        # --- ALWAYS VALID ---
        elif isinstance(action, (MoveTo, StopMovement, SetResearchQueue, Wait, NoOp)):
            result.append(action)

        # --- PASS THROUGH ---
        # Vehicle stubs, combat stubs, and any future action types not yet
        # known to this function. The bridge handles validation gracefully;
        # the execution layer gates by ActionCategory before reaching here.
        else:
            result.append(action)

    return result