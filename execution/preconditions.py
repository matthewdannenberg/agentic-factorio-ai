"""
execution/preconditions.py

Precondition helpers — check action feasibility against the current WorldQuery
and KnowledgeBase before committing to a task or dispatching an action.

These functions answer "can we start action X given current state?" — they
involve inventory arithmetic, recipe lookups, and KB queries. They are called
by the coordinator and agents before deriving tasks or dispatching actions.

For pure world-state observation predicates (is_at, is_reachable, can_mine),
see execution/predicates.py.

Rules
-----
- No LLM calls. No RCON. No mutations.
- Functions return bool, tuple, or dict; none raise on missing data.
- Reach distance is derived exclusively from WorldQuery (player.reachable list),
  never from a hardcoded constant.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from bridge import (
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
from world import Position
from execution.predicates import is_at, is_reachable, can_mine

if TYPE_CHECKING:
    from world import WorldQuery
    from world import KnowledgeBase, RecipeRecord

# The Factorio prototype category string for recipes that can be hand-crafted
# by the player character. All other categories require a specific machine.
_HAND_CRAFT_CATEGORY = "crafting"

# The entity name Factorio uses for the player character in made_in lists.
_CHARACTER_ENTITY = "character"


# ---------------------------------------------------------------------------
# Crafting
# ---------------------------------------------------------------------------

def can_reach_count(
    item: str,
    target_count: int,
    wq: "WorldQuery",
    kb: "KnowledgeBase",
) -> tuple[bool, int]:
    """
    Return ``(reachable, achievable)`` where *reachable* is True if the player
    already has *target_count* of *item* in inventory or can reach that count by
    hand-crafting the deficit, and *achievable* is the maximum count the player
    can actually produce.

    On success: ``(True, target_count)``.
    On failure: ``(False, N)`` where N < target_count is the highest count
    reachable given current inventory (0 if nothing at all is possible).

    The achievable count is found by binary search over [0, target_count - 1]
    so the cost is O(log(target_count)) recursive DFS calls rather than one.
    Each call gets a fresh budget copy so budget mutations don't leak between
    iterations.

    Hand-craftable means:
      - The recipe's category is "crafting" (the player character's category),
        AND "character" appears in the recipe's made_in list.
      - Recipes requiring machines (furnaces, assemblers, centrifuges, etc.)
        are never considered hand-craftable.

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

    def _try(count: int) -> bool:
        return _can_reach_recursive(item, count, kb, dict(budget), set())

    if _try(target_count):
        return True, target_count

    # Binary search for the maximum achievable count in [0, target_count - 1].
    lo, hi, best = 0, target_count - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid == 0 or _try(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return False, best


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


def _cached_stack_size(item: str, kb: "KnowledgeBase") -> int:
    """
    Return the stack size for *item* from the KnowledgeBase.

    Uses ``ensure_item()`` so the value is fetched from Factorio on first
    access and cached thereafter. Returns 1 (conservative) only if the item
    is genuinely unknown to the game — i.e. ``ensure_item`` produced a
    placeholder because the prototype query returned nothing.
    """
    return kb.ensure_item(item).stack_size


def has_inventory_space(
    item: str,
    count: int,
    inventory: dict[str, int],
    total_slots: int,
    kb: "KnowledgeBase",
) -> bool:
    """
    True if *count* units of *item* will fit in the player's inventory after
    ingredients have already been consumed.

    Parameters
    ----------
    item        : The output item name (e.g. "iron-gear-wheel").
    count       : Number of units expected to arrive in inventory.
    inventory   : Current inventory as ``{item_name: count, ...}``, typically
                  *after* ingredient costs have been subtracted so that freed
                  slots are already accounted for. Items with count <= 0 are
                  treated as absent. Negative values are treated as 0.
    total_slots : Total inventory slot count for the player
                  (from ``PlayerState.inventory_size``).
    kb          : KnowledgeBase for stack-size lookups.

    Returns False (conservatively) if *total_slots* is 0, which indicates the
    bridge hasn't yet reported inventory size for this player.

    Slot accounting
    ---------------
    Factorio inventory slots are type-homogeneous: each slot holds up to
    ``stack_size`` units of one item type.

    1. Count currently occupied slots from *post_consumption_inventory*,
       including the slots already holding existing units of *item*.
       Unknown items default to stack_size=1 — worst case, one slot each.
    2. Compute free slots = total_slots - occupied.
    3. Compute how many of *count* new units absorb into the partial stack
       of *item* already present (if any). This does NOT free a slot — the
       slot is already counted as occupied; it just holds more units.
    4. Compute how many additional fresh slots the overflow beyond the
       partial stack requires.
    5. Return True iff slots_needed_for_overflow <= free_slots.
    """
    if total_slots <= 0:
        return False

    stack_size = _cached_stack_size(item, kb)
    existing = max(0, inventory.get(item, 0))

    # Occupied slots across all items, including existing units of the output
    # item. The partial stack is NOT subtracted — it is already "taken".
    occupied = 0
    for inv_item, inv_count in inventory.items():
        if inv_count <= 0:
            continue
        ss = _cached_stack_size(inv_item, kb)
        occupied += math.ceil(inv_count / ss)

    free_slots = total_slots - occupied

    # Space available in the last partial stack of *item* already present.
    # (stack_size - existing % stack_size) % stack_size gives 0 when the
    # existing count is 0 or exactly fills a stack — both correctly signal
    # that a fresh slot is needed.
    partial_space = (stack_size - existing % stack_size) % stack_size

    # Units that need fresh slots beyond the partial stack.
    overflow = max(0, count - partial_space)
    slots_needed = math.ceil(overflow / stack_size) if overflow > 0 else 0

    return slots_needed <= free_slots


def post_crafting_inventory(
    item: str,
    count: int,
    inventory: dict[str, int],
    kb: "KnowledgeBase",
) -> dict[str, int]:
    """
    Return the player inventory after queuing *count* units of *item* for crafting.

    Runs the same DFS that ``can_reach_count`` uses, so existing higher-tier
    items in inventory are consumed directly rather than re-crafted. For example,
    if the player has 3 gears and needs 5 total, only 2 are crafted (consuming
    4 plates), not 5 (10 plates).

    Parameters
    ----------
    item      : Output item name (e.g. "iron-gear-wheel").
    count     : Number of units to craft.
    inventory : Current inventory as ``{item_name: count}``. Not mutated.
    kb        : KnowledgeBase for recipe lookups.

    Returns a new dict representing the inventory after all ingredient
    consumption. Items not consumed are included unchanged. If no
    hand-craftable recipe is found, returns a copy of *inventory* unmodified.
    """
    # Compute post-consumption inventory by subtracting ingredient costs.
    # Use the first hand-craftable recipe — the same one can_reach_count used.
    post = dict(inventory)
    recipes = _hand_craftable_recipes(item, kb)
    if recipes:
        recipe = recipes[0]
        yield_per_run = sum(
            p.amount * p.probability
            for p in recipe.products
            if p.name == item and not p.is_fluid
        )
        if yield_per_run > 0:
            have = post.get(item, 0)
            still_needed = max(0, count - have)
            runs_needed = math.ceil(still_needed / yield_per_run)
            for ingredient in recipe.ingredients:
                if not ingredient.is_fluid:
                    cost = math.ceil(ingredient.amount * runs_needed)
                    post[ingredient.name] = max(0, post.get(ingredient.name, 0) - cost)
    return post


def check_crafting_preconditions(
    item: str,
    count: int,
    wq: "WorldQuery",
    kb: "KnowledgeBase",
) -> tuple[bool, bool]:
    """
    Return ``(can_craft, has_space)`` for crafting *count* units of *item*.

    Combines ``can_reach_count`` and ``has_inventory_space`` in a single call,
    using ``post_crafting_inventory`` to compute the post-consumption state.

    Parameters
    ----------
    item  : Output item name (e.g. "iron-gear-wheel").
    count : Number of units to craft.
    wq    : Current WorldQuery (inventory, slot count).
    kb    : KnowledgeBase (recipes, stack sizes).

    Returns
    -------
    can_craft : True if the player has (or can hand-craft) sufficient
                ingredients for *count* units of *item*.
    has_space : True if the player's inventory will have room for the output
                once ingredients are consumed. Always False when can_craft is
                False — no point checking space if we can't craft.

    Note: ``has_space`` uses ``PlayerState.inventory_size`` from *wq*. If the
    bridge has not yet reported this value (it will be 0), ``has_space`` returns
    False conservatively.
    """
    craftable, _ = can_reach_count(item, count, wq, kb)
    if not craftable:
        return False, False

    # Build current inventory dict.
    current: dict[str, int] = {}
    for slot in wq.state.player.inventory.slots:
        current[slot.item] = current.get(slot.item, 0) + slot.count

    post = post_crafting_inventory(item,count,current,kb)

    total_slots = wq.state.player.inventory_size
    space = has_inventory_space(item, count, post, total_slots, kb)
    return True, space


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
            reachable, _ = can_reach_count(action.recipe, action.count, wq, kb)
            if reachable:
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