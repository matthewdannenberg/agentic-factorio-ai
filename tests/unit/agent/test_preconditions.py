"""
tests/unit/agent/test_preconditions.py  (updated for Phase 6 corrections)

Changes from original:
  - can_craft renamed to can_reach_count throughout
  - New tests for hand-craftable filtering (non-crafting category rejected,
    machine-only recipes rejected, fluid ingredients rejected)
  - Multiple-recipe tests (hand-craftable wins over machine-only)
  - Circular recipe guard test
  - FlipEntity added to valid_actions tests
"""

import unittest

from agent.preconditions import (
    can_mine,
    can_place,
    can_reach_count,
    is_at,
    is_reachable,
    valid_actions,
)
from bridge.actions import (
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
    SetSplitterPriority,
    StopMovement,
    TransferItems,
    UseItem,
    Wait,
)
from world.state import (
    EntityState,
    Inventory,
    InventorySlot,
    Position,
    WorldState,
)
from world.query import WorldQuery


# ---------------------------------------------------------------------------
# Minimal KB stubs
# ---------------------------------------------------------------------------

class _IngredientStub:
    def __init__(self, name, amount, is_fluid=False):
        self.name = name
        self.amount = amount
        self.is_fluid = is_fluid


class _ProductStub:
    def __init__(self, name, amount, probability=1.0, is_fluid=False):
        self.name = name
        self.amount = amount
        self.probability = probability
        self.is_fluid = is_fluid


class _RecipeStub:
    def __init__(
        self,
        name,
        ingredients,
        products,
        category="crafting",
        made_in=None,
        is_placeholder=False,
    ):
        self.name = name
        self.ingredients = ingredients
        self.products = products
        self.category = category
        self.made_in = made_in if made_in is not None else ["character"]
        self.is_placeholder = is_placeholder


class _KBStub:
    """Minimal KnowledgeBase stub for precondition tests."""

    def __init__(self, recipes_by_product: dict):
        # recipes_by_product: item_name -> list[_RecipeStub]
        self._by_product = recipes_by_product

    def recipes_for_product(self, item: str):
        return list(self._by_product.get(item, []))

    def get_recipe(self, name: str):
        # Legacy single-recipe accessor — not used by updated can_reach_count.
        for recipes in self._by_product.values():
            for r in recipes:
                if r.name == name:
                    return r
        return None


# ---------------------------------------------------------------------------
# WorldQuery helpers
# ---------------------------------------------------------------------------

def _make_wq(
    player_pos: Position = None,
    reachable: list = None,
    entities: list = None,
    inventory_items: dict = None,
    tick: int = 100,
) -> WorldQuery:
    state = WorldState(tick=tick, entities=entities or [])
    state.player.position = player_pos or Position(0.0, 0.0)
    state.player.reachable = reachable or []
    if inventory_items:
        state.player.inventory = Inventory(
            slots=[InventorySlot(item=k, count=v) for k, v in inventory_items.items()]
        )
    state._rebuild_entity_indices()
    return WorldQuery(state)


def _make_entity(entity_id, pos, name="iron-ore"):
    return EntityState(entity_id=entity_id, name=name, position=pos)


# ---------------------------------------------------------------------------
# is_at
# ---------------------------------------------------------------------------

class TestIsAt(unittest.TestCase):

    def test_exactly_at_target(self):
        wq = _make_wq(player_pos=Position(0.0, 0.0))
        self.assertTrue(is_at(Position(0.0, 0.0), wq))

    def test_within_default_tolerance(self):
        wq = _make_wq(player_pos=Position(0.5, 0.5))
        self.assertTrue(is_at(Position(0.0, 0.0), wq, tolerance=1.0))

    def test_just_outside_default_tolerance(self):
        wq = _make_wq(player_pos=Position(1.1, 0.0))
        self.assertFalse(is_at(Position(0.0, 0.0), wq, tolerance=1.0))

    def test_custom_tolerance(self):
        wq = _make_wq(player_pos=Position(4.0, 0.0))
        self.assertTrue(is_at(Position(0.0, 0.0), wq, tolerance=5.0))
        self.assertFalse(is_at(Position(0.0, 0.0), wq, tolerance=3.0))

    def test_diagonal_distance(self):
        wq = _make_wq(player_pos=Position(3.0, 4.0))
        self.assertTrue(is_at(Position(0.0, 0.0), wq, tolerance=5.0))
        self.assertFalse(is_at(Position(0.0, 0.0), wq, tolerance=4.9))


# ---------------------------------------------------------------------------
# is_reachable
# ---------------------------------------------------------------------------

class TestIsReachable(unittest.TestCase):

    def test_entity_in_reachable_list(self):
        wq = _make_wq(reachable=[1, 2, 3])
        self.assertTrue(is_reachable(1, wq))
        self.assertTrue(is_reachable(3, wq))

    def test_entity_not_in_reachable_list(self):
        wq = _make_wq(reachable=[1, 2])
        self.assertFalse(is_reachable(99, wq))

    def test_empty_reachable_list(self):
        wq = _make_wq(reachable=[])
        self.assertFalse(is_reachable(1, wq))

    def test_reach_from_worldquery_not_hardcoded(self):
        entity = _make_entity(42, Position(0.1, 0.1))
        wq_unreachable = _make_wq(reachable=[], entities=[entity])
        self.assertFalse(is_reachable(42, wq_unreachable))

        wq_reachable = _make_wq(reachable=[42], entities=[entity])
        self.assertTrue(is_reachable(42, wq_reachable))


# ---------------------------------------------------------------------------
# can_mine
# ---------------------------------------------------------------------------

class TestCanMine(unittest.TestCase):

    def test_entity_present_and_reachable(self):
        entity = _make_entity(10, Position(1.0, 1.0))
        wq = _make_wq(reachable=[10], entities=[entity])
        self.assertTrue(can_mine(10, wq))

    def test_entity_present_but_not_reachable(self):
        entity = _make_entity(10, Position(1.0, 1.0))
        wq = _make_wq(reachable=[], entities=[entity])
        self.assertFalse(can_mine(10, wq))

    def test_entity_absent(self):
        wq = _make_wq(reachable=[10], entities=[])
        self.assertFalse(can_mine(10, wq))

    def test_delegates_to_is_reachable_not_distance(self):
        entity = _make_entity(5, Position(100.0, 100.0))
        wq = _make_wq(reachable=[5], entities=[entity])
        self.assertTrue(can_mine(5, wq))


# ---------------------------------------------------------------------------
# can_reach_count (renamed from can_craft)
# ---------------------------------------------------------------------------

class TestCanReachCount(unittest.TestCase):

    def _iron_plate_recipe(self):
        """Simple hand-craftable recipe: 1 iron-ore → 1 iron-plate."""
        return _RecipeStub(
            name="iron-plate",
            ingredients=[_IngredientStub("iron-ore", 1)],
            products=[_ProductStub("iron-plate", 1)],
            category="crafting",
            made_in=["character", "stone-furnace"],
        )

    def _gear_recipe(self):
        """2 iron-plate → 1 iron-gear-wheel."""
        return _RecipeStub(
            name="iron-gear-wheel",
            ingredients=[_IngredientStub("iron-plate", 2)],
            products=[_ProductStub("iron-gear-wheel", 1)],
            category="crafting",
            made_in=["character", "assembling-machine-1"],
        )

    def test_already_have_enough(self):
        wq = _make_wq(inventory_items={"iron-ore": 50})
        kb = _KBStub({})
        self.assertTrue(can_reach_count("iron-ore", 50, wq, kb))

    def test_already_have_more_than_enough(self):
        wq = _make_wq(inventory_items={"iron-ore": 100})
        kb = _KBStub({})
        self.assertTrue(can_reach_count("iron-ore", 50, wq, kb))

    def test_have_none_no_recipe(self):
        wq = _make_wq(inventory_items={})
        kb = _KBStub({})
        self.assertFalse(can_reach_count("iron-ore", 10, wq, kb))

    def test_can_craft_deficit_from_ingredients(self):
        # Have 0 iron-plate, have 10 iron-ore → can craft 10 iron-plate.
        recipe = self._iron_plate_recipe()
        wq = _make_wq(inventory_items={"iron-ore": 10})
        kb = _KBStub({"iron-plate": [recipe]})
        self.assertTrue(can_reach_count("iron-plate", 10, wq, kb))

    def test_cannot_craft_missing_ingredients(self):
        recipe = self._iron_plate_recipe()
        wq = _make_wq(inventory_items={})  # no iron-ore
        kb = _KBStub({"iron-plate": [recipe]})
        self.assertFalse(can_reach_count("iron-plate", 10, wq, kb))

    def test_partial_inventory_counts_toward_target(self):
        # Have 8 iron-plate already; need 2 more from 2 iron-ore.
        recipe = self._iron_plate_recipe()
        wq = _make_wq(inventory_items={"iron-plate": 8, "iron-ore": 2})
        kb = _KBStub({"iron-plate": [recipe]})
        self.assertTrue(can_reach_count("iron-plate", 10, wq, kb))

    def test_partial_inventory_insufficient_ingredients(self):
        # Have 8 iron-plate; need 2 more but only 1 iron-ore.
        recipe = self._iron_plate_recipe()
        wq = _make_wq(inventory_items={"iron-plate": 8, "iron-ore": 1})
        kb = _KBStub({"iron-plate": [recipe]})
        self.assertFalse(can_reach_count("iron-plate", 10, wq, kb))

    def test_recursive_ingredient_chain(self):
        # Gear needs iron-plate; iron-plate needs iron-ore.
        gear_recipe = self._gear_recipe()
        plate_recipe = self._iron_plate_recipe()
        wq = _make_wq(inventory_items={"iron-ore": 4})  # can make 4 plates, 2 gears
        kb = _KBStub({
            "iron-gear-wheel": [gear_recipe],
            "iron-plate": [plate_recipe],
        })
        self.assertTrue(can_reach_count("iron-gear-wheel", 2, wq, kb))
        self.assertFalse(can_reach_count("iron-gear-wheel", 3, wq, kb))

    # --- Hand-craftable filtering ---

    def test_machine_only_category_rejected(self):
        # Category "smelting" is not hand-craftable.
        recipe = _RecipeStub(
            name="iron-plate",
            ingredients=[_IngredientStub("iron-ore", 1)],
            products=[_ProductStub("iron-plate", 1)],
            category="smelting",
            made_in=["stone-furnace"],
        )
        wq = _make_wq(inventory_items={"iron-ore": 10})
        kb = _KBStub({"iron-plate": [recipe]})
        self.assertFalse(can_reach_count("iron-plate", 5, wq, kb))

    def test_character_not_in_made_in_rejected(self):
        # Category is "crafting" but made_in has only machines, not "character".
        recipe = _RecipeStub(
            name="iron-plate",
            ingredients=[_IngredientStub("iron-ore", 1)],
            products=[_ProductStub("iron-plate", 1)],
            category="crafting",
            made_in=["assembling-machine-1"],  # no "character"
        )
        wq = _make_wq(inventory_items={"iron-ore": 10})
        kb = _KBStub({"iron-plate": [recipe]})
        self.assertFalse(can_reach_count("iron-plate", 5, wq, kb))

    def test_fluid_ingredient_rejected(self):
        # Recipe requires a fluid ingredient — not hand-craftable in practice.
        recipe = _RecipeStub(
            name="some-item",
            ingredients=[_IngredientStub("water", 10, is_fluid=True)],
            products=[_ProductStub("some-item", 1)],
            category="crafting",
            made_in=["character"],
        )
        wq = _make_wq(inventory_items={})
        kb = _KBStub({"some-item": [recipe]})
        self.assertFalse(can_reach_count("some-item", 1, wq, kb))

    def test_placeholder_recipe_rejected(self):
        recipe = _RecipeStub(
            name="iron-plate",
            ingredients=[],
            products=[_ProductStub("iron-plate", 1)],
            category="crafting",
            made_in=["character"],
            is_placeholder=True,
        )
        wq = _make_wq(inventory_items={"iron-ore": 10})
        kb = _KBStub({"iron-plate": [recipe]})
        self.assertFalse(can_reach_count("iron-plate", 5, wq, kb))

    def test_multiple_recipes_machine_only_and_hand_craftable(self):
        # One machine-only recipe and one hand-craftable for the same item.
        # The hand-craftable one should be found and used.
        machine_recipe = _RecipeStub(
            name="iron-plate-machine",
            ingredients=[_IngredientStub("iron-ore", 1)],
            products=[_ProductStub("iron-plate", 1)],
            category="smelting",
            made_in=["stone-furnace"],
        )
        hand_recipe = _RecipeStub(
            name="iron-plate-hand",
            ingredients=[_IngredientStub("iron-ore", 1)],
            products=[_ProductStub("iron-plate", 1)],
            category="crafting",
            made_in=["character"],
        )
        wq = _make_wq(inventory_items={"iron-ore": 10})
        kb = _KBStub({"iron-plate": [machine_recipe, hand_recipe]})
        self.assertTrue(can_reach_count("iron-plate", 5, wq, kb))

    def test_circular_recipe_guard(self):
        # A recipe that consumes the item it produces (circular).
        # Player has 0 of the item → cannot bootstrap, so False.
        circular = _RecipeStub(
            name="circular-item",
            ingredients=[_IngredientStub("circular-item", 1)],
            products=[_ProductStub("circular-item", 2)],
            category="crafting",
            made_in=["character"],
        )
        wq = _make_wq(inventory_items={})
        kb = _KBStub({"circular-item": [circular]})
        self.assertFalse(can_reach_count("circular-item", 1, wq, kb))

    def test_circular_recipe_with_existing_stock(self):
        # If we already have enough, circular doesn't matter.
        circular = _RecipeStub(
            name="circular-item",
            ingredients=[_IngredientStub("circular-item", 1)],
            products=[_ProductStub("circular-item", 2)],
            category="crafting",
            made_in=["character"],
        )
        wq = _make_wq(inventory_items={"circular-item": 5})
        kb = _KBStub({"circular-item": [circular]})
        self.assertTrue(can_reach_count("circular-item", 5, wq, kb))


class TestCanReachCountProductionChains(unittest.TestCase):
    """
    Realistic multi-step production chain tests for can_reach_count.

    Each scenario uses a KB stub that mirrors the actual Factorio recipe graph
    for that chain. Expected outcomes are hand-verified against the arithmetic
    in _recipe_can_cover and _can_reach_recursive.

    Naming convention: _r_<item> returns the RecipeStub for that item.
    """

    # ------------------------------------------------------------------
    # Recipe library (all hand-craftable, using character + assembler)
    # ------------------------------------------------------------------

    def _r_iron_plate(self):
        # 1 iron-ore → 1 iron-plate (smelting, but marked craftable for tests
        # where we want it available; use a machine-only stub when testing the
        # smelting-only gate)
        return _RecipeStub(
            name="iron-plate",
            ingredients=[_IngredientStub("iron-ore", 1)],
            products=[_ProductStub("iron-plate", 1)],
            category="crafting",
            made_in=["character", "stone-furnace"],
        )

    def _r_copper_plate(self):
        return _RecipeStub(
            name="copper-plate",
            ingredients=[_IngredientStub("copper-ore", 1)],
            products=[_ProductStub("copper-plate", 1)],
            category="crafting",
            made_in=["character", "stone-furnace"],
        )

    def _r_iron_gear_wheel(self):
        # 2 iron-plate → 1 iron-gear-wheel
        return _RecipeStub(
            name="iron-gear-wheel",
            ingredients=[_IngredientStub("iron-plate", 2)],
            products=[_ProductStub("iron-gear-wheel", 1)],
            category="crafting",
            made_in=["character", "assembling-machine-1"],
        )

    def _r_copper_cable(self):
        # 1 copper-plate → 2 copper-cable  (yield > 1 per run)
        return _RecipeStub(
            name="copper-cable",
            ingredients=[_IngredientStub("copper-plate", 1)],
            products=[_ProductStub("copper-cable", 2)],
            category="crafting",
            made_in=["character", "assembling-machine-1"],
        )

    def _r_electronic_circuit(self):
        # 1 iron-plate + 3 copper-cable → 1 electronic-circuit
        return _RecipeStub(
            name="electronic-circuit",
            ingredients=[
                _IngredientStub("iron-plate", 1),
                _IngredientStub("copper-cable", 3),
            ],
            products=[_ProductStub("electronic-circuit", 1)],
            category="crafting",
            made_in=["character", "assembling-machine-1"],
        )

    def _r_iron_stick(self):
        # 1 iron-plate → 2 iron-stick  (yield > 1 per run)
        return _RecipeStub(
            name="iron-stick",
            ingredients=[_IngredientStub("iron-plate", 1)],
            products=[_ProductStub("iron-stick", 2)],
            category="crafting",
            made_in=["character", "assembling-machine-1"],
        )

    def _r_stone_furnace(self):
        # 5 stone → 1 stone-furnace
        return _RecipeStub(
            name="stone-furnace",
            ingredients=[_IngredientStub("stone", 5)],
            products=[_ProductStub("stone-furnace", 1)],
            category="crafting",
            made_in=["character"],
        )

    def _r_burner_mining_drill(self):
        # 3 iron-plate + 3 iron-gear-wheel + 1 stone-furnace → 1 burner-mining-drill
        return _RecipeStub(
            name="burner-mining-drill",
            ingredients=[
                _IngredientStub("iron-plate", 3),
                _IngredientStub("iron-gear-wheel", 3),
                _IngredientStub("stone-furnace", 1),
            ],
            products=[_ProductStub("burner-mining-drill", 1)],
            category="crafting",
            made_in=["character", "assembling-machine-1"],
        )

    def _r_plastic_machine_only(self):
        # plastic-bar requires petroleum gas — not hand-craftable
        return _RecipeStub(
            name="plastic-bar",
            ingredients=[
                _IngredientStub("petroleum-gas", 20, is_fluid=True),
                _IngredientStub("coal", 1),
            ],
            products=[_ProductStub("plastic-bar", 2)],
            category="crafting",
            made_in=["chemical-plant"],   # no "character"
        )

    def _r_advanced_circuit(self):
        # 2 electronic-circuit + 2 plastic-bar + 4 copper-cable → 1 advanced-circuit
        return _RecipeStub(
            name="advanced-circuit",
            ingredients=[
                _IngredientStub("electronic-circuit", 2),
                _IngredientStub("plastic-bar", 2),
                _IngredientStub("copper-cable", 4),
            ],
            products=[_ProductStub("advanced-circuit", 1)],
            category="crafting",
            made_in=["character", "assembling-machine-2"],
        )

    # ------------------------------------------------------------------
    # Helper: build a KB from a set of recipe stubs
    # ------------------------------------------------------------------

    def _kb(self, *recipes) -> _KBStub:
        by_product = {}
        for recipe in recipes:
            for product in recipe.products:
                by_product.setdefault(product.name, []).append(recipe)
        return _KBStub(by_product)

    # ------------------------------------------------------------------
    # Iron gear wheel — two-level chain with ingredient multiplication
    # ------------------------------------------------------------------

    def test_gear_wheel_exact_ingredients(self):
        # 1 gear needs 2 iron-plate, each needing 1 iron-ore → 2 iron-ore total.
        kb = self._kb(self._r_iron_gear_wheel(), self._r_iron_plate())
        wq = _make_wq(inventory_items={"iron-ore": 2})
        self.assertTrue(can_reach_count("iron-gear-wheel", 1, wq, kb))

    def test_gear_wheel_insufficient_ore_by_one(self):
        kb = self._kb(self._r_iron_gear_wheel(), self._r_iron_plate())
        wq = _make_wq(inventory_items={"iron-ore": 1})
        self.assertFalse(can_reach_count("iron-gear-wheel", 1, wq, kb))

    def test_gear_wheel_multiple_count(self):
        # 5 gears need 10 iron-plate → 10 iron-ore.
        kb = self._kb(self._r_iron_gear_wheel(), self._r_iron_plate())
        wq = _make_wq(inventory_items={"iron-ore": 10})
        self.assertTrue(can_reach_count("iron-gear-wheel", 5, wq, kb))

    def test_gear_wheel_multiple_count_one_short(self):
        kb = self._kb(self._r_iron_gear_wheel(), self._r_iron_plate())
        wq = _make_wq(inventory_items={"iron-ore": 9})
        self.assertFalse(can_reach_count("iron-gear-wheel", 5, wq, kb))

    def test_gear_wheel_partial_plates_in_inventory(self):
        # 2 gears = 4 plates. Have 3 plates + 2 ore → can craft 1 more plate → 4 total.
        kb = self._kb(self._r_iron_gear_wheel(), self._r_iron_plate())
        wq = _make_wq(inventory_items={"iron-plate": 3, "iron-ore": 1})
        self.assertTrue(can_reach_count("iron-gear-wheel", 2, wq, kb))

    def test_gear_wheel_partial_plates_still_short(self):
        # 3 gears = 6 plates. Have 3 plates + 2 ore (→ 2 more = 5 total). Short by 1.
        kb = self._kb(self._r_iron_gear_wheel(), self._r_iron_plate())
        wq = _make_wq(inventory_items={"iron-plate": 3, "iron-ore": 2})
        self.assertFalse(can_reach_count("iron-gear-wheel", 3, wq, kb))

    # ------------------------------------------------------------------
    # Copper cable — yield > 1 per run (1 copper-plate → 2 cables)
    # ------------------------------------------------------------------

    def test_copper_cable_yield_two_per_run(self):
        # 4 cables = 2 runs = 2 copper-plate = 2 copper-ore.
        kb = self._kb(self._r_copper_cable(), self._r_copper_plate())
        wq = _make_wq(inventory_items={"copper-ore": 2})
        self.assertTrue(can_reach_count("copper-cable", 4, wq, kb))

    def test_copper_cable_odd_count_rounds_up(self):
        # 3 cables: ceil(3/2)=2 runs → 2 copper-plate → 2 copper-ore.
        kb = self._kb(self._r_copper_cable(), self._r_copper_plate())
        wq = _make_wq(inventory_items={"copper-ore": 2})
        self.assertTrue(can_reach_count("copper-cable", 3, wq, kb))

    def test_copper_cable_one_short_of_even_run(self):
        # 5 cables: ceil(5/2)=3 runs → 3 copper-ore. 2 ore only covers 4 cables.
        kb = self._kb(self._r_copper_cable(), self._r_copper_plate())
        wq = _make_wq(inventory_items={"copper-ore": 2})
        self.assertFalse(can_reach_count("copper-cable", 5, wq, kb))

    def test_copper_cable_existing_cables_reduce_runs_needed(self):
        # Have 2 cables; need 5 → deficit 3 → ceil(3/2)=2 runs → 2 copper-ore.
        kb = self._kb(self._r_copper_cable(), self._r_copper_plate())
        wq = _make_wq(inventory_items={"copper-cable": 2, "copper-ore": 2})
        self.assertTrue(can_reach_count("copper-cable", 5, wq, kb))

    # ------------------------------------------------------------------
    # Electronic circuit — branching chain with non-unit cable yield
    # 1 circuit: 1 iron-plate + 3 copper-cable
    # 3 cable: ceil(3/2)=2 copper-plate runs → 2 copper-ore
    # So per circuit: 1 iron-ore + 2 copper-ore
    # ------------------------------------------------------------------

    def test_electronic_circuit_exact_raw_materials(self):
        kb = self._kb(
            self._r_electronic_circuit(),
            self._r_copper_cable(),
            self._r_iron_plate(),
            self._r_copper_plate(),
        )
        wq = _make_wq(inventory_items={"iron-ore": 1, "copper-ore": 2})
        self.assertTrue(can_reach_count("electronic-circuit", 1, wq, kb))

    def test_electronic_circuit_missing_copper(self):
        kb = self._kb(
            self._r_electronic_circuit(),
            self._r_copper_cable(),
            self._r_iron_plate(),
            self._r_copper_plate(),
        )
        wq = _make_wq(inventory_items={"iron-ore": 1})
        self.assertFalse(can_reach_count("electronic-circuit", 1, wq, kb))

    def test_electronic_circuit_missing_iron(self):
        kb = self._kb(
            self._r_electronic_circuit(),
            self._r_copper_cable(),
            self._r_iron_plate(),
            self._r_copper_plate(),
        )
        wq = _make_wq(inventory_items={"copper-ore": 2})
        self.assertFalse(can_reach_count("electronic-circuit", 1, wq, kb))

    def test_electronic_circuit_four_circuits(self):
        # 4 circuits: 4 iron-ore + 6 copper-ore
        # (4 iron-plate; 12 cable = ceil(12/2)=6 copper-plate = 6 copper-ore)
        kb = self._kb(
            self._r_electronic_circuit(),
            self._r_copper_cable(),
            self._r_iron_plate(),
            self._r_copper_plate(),
        )
        wq = _make_wq(inventory_items={"iron-ore": 4, "copper-ore": 6})
        self.assertTrue(can_reach_count("electronic-circuit", 4, wq, kb))

    def test_electronic_circuit_four_circuits_one_copper_short(self):
        kb = self._kb(
            self._r_electronic_circuit(),
            self._r_copper_cable(),
            self._r_iron_plate(),
            self._r_copper_plate(),
        )
        wq = _make_wq(inventory_items={"iron-ore": 4, "copper-ore": 5})
        self.assertFalse(can_reach_count("electronic-circuit", 4, wq, kb))

    def test_electronic_circuit_partial_intermediates_in_inventory(self):
        # Have 2 copper-cable already; need 3 for 1 circuit → deficit 1.
        # ceil(1/2)=1 run → 1 copper-plate → 1 copper-ore. Plus 1 iron-ore.
        kb = self._kb(
            self._r_electronic_circuit(),
            self._r_copper_cable(),
            self._r_iron_plate(),
            self._r_copper_plate(),
        )
        wq = _make_wq(inventory_items={
            "copper-cable": 2,
            "copper-ore": 1,
            "iron-ore": 1,
        })
        self.assertTrue(can_reach_count("electronic-circuit", 1, wq, kb))

    # ------------------------------------------------------------------
    # Burner mining drill — three levels, diamond dependency on iron-plate
    # Total iron-plate needed: 3 direct + 3 gears × 2 plate = 9
    # Plus 5 stone for the stone-furnace ingredient
    # ------------------------------------------------------------------

    def test_burner_mining_drill_exact_raw(self):
        kb = self._kb(
            self._r_burner_mining_drill(),
            self._r_iron_gear_wheel(),
            self._r_stone_furnace(),
            self._r_iron_plate(),
        )
        wq = _make_wq(inventory_items={"iron-ore": 9, "stone": 5})
        self.assertTrue(can_reach_count("burner-mining-drill", 1, wq, kb))

    def test_burner_mining_drill_one_iron_short(self):
        kb = self._kb(
            self._r_burner_mining_drill(),
            self._r_iron_gear_wheel(),
            self._r_stone_furnace(),
            self._r_iron_plate(),
        )
        wq = _make_wq(inventory_items={"iron-ore": 8, "stone": 5})
        self.assertFalse(can_reach_count("burner-mining-drill", 1, wq, kb))

    def test_burner_mining_drill_no_stone(self):
        kb = self._kb(
            self._r_burner_mining_drill(),
            self._r_iron_gear_wheel(),
            self._r_stone_furnace(),
            self._r_iron_plate(),
        )
        wq = _make_wq(inventory_items={"iron-ore": 9})
        self.assertFalse(can_reach_count("burner-mining-drill", 1, wq, kb))

    def test_burner_mining_drill_diamond_dependency_iron_plate(self):
        # Iron-plate is consumed both directly (3) and via gears (6).
        # The visited set guards against infinite loops on the item being
        # *produced*, not items being *consumed* — iron-plate here is only
        # consumed, so the second branch must not be blocked by visited.
        kb = self._kb(
            self._r_burner_mining_drill(),
            self._r_iron_gear_wheel(),
            self._r_stone_furnace(),
            self._r_iron_plate(),
        )
        # Provide iron-plate directly, skipping ore, to isolate the diamond.
        wq = _make_wq(inventory_items={"iron-plate": 9, "stone": 5})
        self.assertTrue(can_reach_count("burner-mining-drill", 1, wq, kb))

    def test_burner_mining_drill_furnace_prefabbed(self):
        # Stone-furnace already in inventory — no stone needed.
        kb = self._kb(
            self._r_burner_mining_drill(),
            self._r_iron_gear_wheel(),
            self._r_stone_furnace(),
            self._r_iron_plate(),
        )
        wq = _make_wq(inventory_items={"iron-ore": 9, "stone-furnace": 1})
        self.assertTrue(can_reach_count("burner-mining-drill", 1, wq, kb))

    def test_burner_mining_drill_gears_prefabbed(self):
        # All 3 gears already in inventory — only need 3 plate + 1 furnace.
        kb = self._kb(
            self._r_burner_mining_drill(),
            self._r_iron_gear_wheel(),
            self._r_stone_furnace(),
            self._r_iron_plate(),
        )
        wq = _make_wq(inventory_items={
            "iron-gear-wheel": 3,
            "iron-ore": 3,
            "stone": 5,
        })
        self.assertTrue(can_reach_count("burner-mining-drill", 1, wq, kb))

    # ------------------------------------------------------------------
    # Advanced circuit — deep chain that fails on machine-only plastic
    # 1 advanced-circuit: 2 electronic-circuit + 2 plastic-bar + 4 copper-cable
    # plastic-bar is not hand-craftable → whole chain fails
    # ------------------------------------------------------------------

    def test_advanced_circuit_fails_on_machine_only_plastic(self):
        kb = self._kb(
            self._r_advanced_circuit(),
            self._r_electronic_circuit(),
            self._r_copper_cable(),
            self._r_iron_plate(),
            self._r_copper_plate(),
            self._r_plastic_machine_only(),
        )
        wq = _make_wq(inventory_items={"iron-ore": 100, "copper-ore": 100})
        self.assertFalse(can_reach_count("advanced-circuit", 1, wq, kb))

    def test_advanced_circuit_succeeds_when_plastic_in_inventory(self):
        # With plastic already in inventory, the chain can complete.
        # 1 advanced-circuit: 2 electronic-circuits + 2 plastic + 4 cable.
        # 2 circuits: 2 iron-plate + 6 cable.
        # Total cable: 6 + 4 = 10; ceil(10/2)=5 copper-plate = 5 copper-ore.
        # Iron: 2 iron-ore.
        kb = self._kb(
            self._r_advanced_circuit(),
            self._r_electronic_circuit(),
            self._r_copper_cable(),
            self._r_iron_plate(),
            self._r_copper_plate(),
            self._r_plastic_machine_only(),
        )
        wq = _make_wq(inventory_items={
            "plastic-bar": 2,
            "iron-ore": 2,
            "copper-ore": 5,
        })
        self.assertTrue(can_reach_count("advanced-circuit", 1, wq, kb))

    # ------------------------------------------------------------------
    # Iron stick — yield > 1, ceiling arithmetic at boundary conditions
    # 1 iron-plate → 2 iron-stick
    # ------------------------------------------------------------------

    def test_iron_stick_exact_even_target(self):
        # 4 sticks = 2 runs = 2 iron-plate = 2 iron-ore.
        kb = self._kb(self._r_iron_stick(), self._r_iron_plate())
        wq = _make_wq(inventory_items={"iron-ore": 2})
        self.assertTrue(can_reach_count("iron-stick", 4, wq, kb))

    def test_iron_stick_odd_target_rounds_up(self):
        # 5 sticks: ceil(5/2)=3 runs → 3 iron-plate → 3 iron-ore.
        kb = self._kb(self._r_iron_stick(), self._r_iron_plate())
        wq = _make_wq(inventory_items={"iron-ore": 3})
        self.assertTrue(can_reach_count("iron-stick", 5, wq, kb))

    def test_iron_stick_odd_target_one_ore_short(self):
        # 5 sticks needs 3 ore; 2 ore only gives 4 sticks.
        kb = self._kb(self._r_iron_stick(), self._r_iron_plate())
        wq = _make_wq(inventory_items={"iron-ore": 2})
        self.assertFalse(can_reach_count("iron-stick", 5, wq, kb))

    def test_iron_stick_existing_sticks_reduce_deficit(self):
        # Have 3 sticks; need 5 → deficit 2 → ceil(2/2)=1 run → 1 iron-ore.
        kb = self._kb(self._r_iron_stick(), self._r_iron_plate())
        wq = _make_wq(inventory_items={"iron-stick": 3, "iron-ore": 1})
        self.assertTrue(can_reach_count("iron-stick", 5, wq, kb))

    # ------------------------------------------------------------------
    # Probabilistic yield — product.probability < 1.0
    # Expected yield per run = amount × probability; runs = ceil(deficit / yield)
    # ------------------------------------------------------------------

    def test_probabilistic_yield_enough_expected(self):
        # 0.5 probability per run; need 3 → ceil(3/0.5)=6 runs → 6 ingredients.
        recipe = _RecipeStub(
            name="rare-item",
            ingredients=[_IngredientStub("base-material", 1)],
            products=[_ProductStub("rare-item", 1, probability=0.5)],
            category="crafting",
            made_in=["character"],
        )
        kb = _KBStub({"rare-item": [recipe]})
        wq = _make_wq(inventory_items={"base-material": 6})
        self.assertTrue(can_reach_count("rare-item", 3, wq, kb))

    def test_probabilistic_yield_insufficient(self):
        recipe = _RecipeStub(
            name="rare-item",
            ingredients=[_IngredientStub("base-material", 1)],
            products=[_ProductStub("rare-item", 1, probability=0.5)],
            category="crafting",
            made_in=["character"],
        )
        kb = _KBStub({"rare-item": [recipe]})
        wq = _make_wq(inventory_items={"base-material": 5})
        self.assertFalse(can_reach_count("rare-item", 3, wq, kb))

    # ------------------------------------------------------------------
    # Multi-product recipe — only the target product's yield matters
    # ------------------------------------------------------------------

    def test_multi_product_recipe_target_yield_used(self):
        # Recipe yields 2 of item-a and 1 of item-b per run.
        # Requesting item-a should use yield=2; requesting item-b should use yield=1.
        recipe = _RecipeStub(
            name="combo-recipe",
            ingredients=[_IngredientStub("raw", 1)],
            products=[
                _ProductStub("item-a", 2),
                _ProductStub("item-b", 1),
            ],
            category="crafting",
            made_in=["character"],
        )
        kb = _KBStub({"item-a": [recipe], "item-b": [recipe]})

        # 4 of item-a: ceil(4/2)=2 runs → 2 raw.
        wq_a = _make_wq(inventory_items={"raw": 2})
        self.assertTrue(can_reach_count("item-a", 4, wq_a, kb))

        # 4 of item-b: ceil(4/1)=4 runs → 4 raw.
        wq_b_ok = _make_wq(inventory_items={"raw": 4})
        self.assertTrue(can_reach_count("item-b", 4, wq_b_ok, kb))

        wq_b_short = _make_wq(inventory_items={"raw": 3})
        self.assertFalse(can_reach_count("item-b", 4, wq_b_short, kb))


# ---------------------------------------------------------------------------
# can_place
# ---------------------------------------------------------------------------

class TestCanPlace(unittest.TestCase):

    def test_item_in_inventory_position_clear(self):
        wq = _make_wq(inventory_items={"iron-plate": 5}, entities=[])
        self.assertTrue(can_place("iron-plate", Position(5.0, 5.0), wq))

    def test_item_not_in_inventory(self):
        wq = _make_wq(inventory_items={})
        self.assertFalse(can_place("stone-furnace", Position(5.0, 5.0), wq))

    def test_position_occupied(self):
        entity = _make_entity(1, Position(5.0, 5.0))
        wq = _make_wq(inventory_items={"stone-furnace": 1}, entities=[entity])
        self.assertFalse(can_place("stone-furnace", Position(5.0, 5.0), wq))

    def test_position_clear_when_entity_two_tiles_away(self):
        entity = _make_entity(1, Position(5.0, 5.0))
        wq = _make_wq(inventory_items={"stone-furnace": 1}, entities=[entity])
        self.assertTrue(can_place("stone-furnace", Position(7.0, 5.0), wq))


# ---------------------------------------------------------------------------
# valid_actions
# ---------------------------------------------------------------------------

class TestValidActions(unittest.TestCase):

    def setUp(self):
        self.kb = _KBStub({})

    # --- MINING ---

    def test_mine_entity_passes_when_reachable(self):
        entity = _make_entity(7, Position(1.0, 0.0))
        wq = _make_wq(reachable=[7], entities=[entity])
        result = valid_actions(wq, self.kb, [MineEntity(entity_id=7)])
        self.assertEqual(len(result), 1)

    def test_mine_entity_filtered_when_not_reachable(self):
        entity = _make_entity(7, Position(1.0, 0.0))
        wq = _make_wq(reachable=[], entities=[entity])
        result = valid_actions(wq, self.kb, [MineEntity(entity_id=7)])
        self.assertEqual(len(result), 0)

    def test_mine_resource_passes_when_close(self):
        wq = _make_wq()
        result = valid_actions(
            wq, self.kb,
            [MineResource(position=Position(0.5, 0.5), resource="iron-ore")],
        )
        self.assertEqual(len(result), 1)

    def test_mine_resource_filtered_when_far(self):
        wq = _make_wq()
        result = valid_actions(
            wq, self.kb,
            [MineResource(position=Position(100.0, 100.0), resource="iron-ore")],
        )
        self.assertEqual(len(result), 0)

    # --- CRAFTING ---

    def test_craft_item_passes_with_ingredients(self):
        recipe = _RecipeStub(
            name="iron-plate",
            ingredients=[_IngredientStub("iron-ore", 1)],
            products=[_ProductStub("iron-plate", 1)],
            category="crafting",
            made_in=["character"],
        )
        wq = _make_wq(inventory_items={"iron-ore": 5})
        kb = _KBStub({"iron-plate": [recipe]})
        result = valid_actions(wq, kb, [CraftItem(recipe="iron-plate", count=3)])
        self.assertEqual(len(result), 1)

    def test_craft_item_filtered_when_no_ingredients(self):
        recipe = _RecipeStub(
            name="iron-plate",
            ingredients=[_IngredientStub("iron-ore", 1)],
            products=[_ProductStub("iron-plate", 1)],
            category="crafting",
            made_in=["character"],
        )
        wq = _make_wq(inventory_items={})
        kb = _KBStub({"iron-plate": [recipe]})
        result = valid_actions(wq, kb, [CraftItem(recipe="iron-plate", count=3)])
        self.assertEqual(len(result), 0)

    # --- BUILDING ---

    def test_place_entity_passes_when_valid(self):
        wq = _make_wq(inventory_items={"stone-furnace": 1}, entities=[])
        result = valid_actions(
            wq, self.kb,
            [PlaceEntity(item="stone-furnace", position=Position(5.0, 5.0))],
        )
        self.assertEqual(len(result), 1)

    def test_place_entity_filtered_when_no_item(self):
        wq = _make_wq(inventory_items={})
        result = valid_actions(
            wq, self.kb,
            [PlaceEntity(item="stone-furnace", position=Position(5.0, 5.0))],
        )
        self.assertEqual(len(result), 0)

    def test_rotate_entity_passes_when_reachable(self):
        entity = _make_entity(5, Position(1.0, 0.0))
        wq = _make_wq(reachable=[5], entities=[entity])
        result = valid_actions(wq, self.kb, [RotateEntity(entity_id=5)])
        self.assertEqual(len(result), 1)

    def test_rotate_entity_filtered_when_not_reachable(self):
        entity = _make_entity(5, Position(1.0, 0.0))
        wq = _make_wq(reachable=[], entities=[entity])
        result = valid_actions(wq, self.kb, [RotateEntity(entity_id=5)])
        self.assertEqual(len(result), 0)

    def test_flip_entity_passes_when_reachable(self):
        entity = _make_entity(9, Position(1.0, 0.0))
        wq = _make_wq(reachable=[9], entities=[entity])
        result = valid_actions(wq, self.kb, [FlipEntity(entity_id=9)])
        self.assertEqual(len(result), 1)

    def test_flip_entity_filtered_when_not_reachable(self):
        entity = _make_entity(9, Position(100.0, 100.0))
        wq = _make_wq(reachable=[], entities=[entity])
        result = valid_actions(wq, self.kb, [FlipEntity(entity_id=9)])
        self.assertEqual(len(result), 0)

    def test_set_recipe_passes_when_reachable(self):
        entity = _make_entity(3, Position(1.0, 0.0))
        wq = _make_wq(reachable=[3], entities=[entity])
        result = valid_actions(
            wq, self.kb, [SetRecipe(entity_id=3, recipe="iron-gear-wheel")]
        )
        self.assertEqual(len(result), 1)

    def test_set_recipe_filtered_when_not_reachable(self):
        entity = _make_entity(3, Position(100.0, 0.0))
        wq = _make_wq(reachable=[], entities=[entity])
        result = valid_actions(
            wq, self.kb, [SetRecipe(entity_id=3, recipe="iron-gear-wheel")]
        )
        self.assertEqual(len(result), 0)

    def test_set_filter_passes_when_reachable(self):
        entity = _make_entity(4, Position(1.0, 0.0))
        wq = _make_wq(reachable=[4], entities=[entity])
        result = valid_actions(
            wq, self.kb, [SetFilter(entity_id=4, slot=0, item="iron-plate")]
        )
        self.assertEqual(len(result), 1)

    def test_set_filter_filtered_when_not_reachable(self):
        entity = _make_entity(4, Position(100.0, 0.0))
        wq = _make_wq(reachable=[], entities=[entity])
        result = valid_actions(
            wq, self.kb, [SetFilter(entity_id=4, slot=0, item="iron-plate")]
        )
        self.assertEqual(len(result), 0)

    def test_set_splitter_priority_passes_when_reachable(self):
        entity = _make_entity(6, Position(1.0, 0.0))
        wq = _make_wq(reachable=[6], entities=[entity])
        result = valid_actions(
            wq, self.kb,
            [SetSplitterPriority(entity_id=6, output_priority="left")],
        )
        self.assertEqual(len(result), 1)

    def test_set_splitter_priority_filtered_when_not_reachable(self):
        entity = _make_entity(6, Position(100.0, 0.0))
        wq = _make_wq(reachable=[], entities=[entity])
        result = valid_actions(
            wq, self.kb,
            [SetSplitterPriority(entity_id=6, output_priority="left")],
        )
        self.assertEqual(len(result), 0)

    # --- INVENTORY ---

    def test_transfer_items_passes_when_reachable(self):
        entity = _make_entity(10, Position(1.0, 0.0))
        wq = _make_wq(reachable=[10], entities=[entity])
        result = valid_actions(
            wq, self.kb,
            [TransferItems(entity_id=10, item="iron-ore", count=10)],
        )
        self.assertEqual(len(result), 1)

    def test_transfer_items_filtered_when_not_reachable(self):
        entity = _make_entity(10, Position(100.0, 0.0))
        wq = _make_wq(reachable=[], entities=[entity])
        result = valid_actions(
            wq, self.kb,
            [TransferItems(entity_id=10, item="iron-ore", count=10)],
        )
        self.assertEqual(len(result), 0)

    # --- PLAYER ---

    def test_equip_armor_passes_when_in_inventory(self):
        wq = _make_wq(inventory_items={"heavy-armor": 1})
        result = valid_actions(wq, self.kb, [EquipArmor(item="heavy-armor")])
        self.assertEqual(len(result), 1)

    def test_equip_armor_filtered_when_not_in_inventory(self):
        wq = _make_wq(inventory_items={})
        result = valid_actions(wq, self.kb, [EquipArmor(item="heavy-armor")])
        self.assertEqual(len(result), 0)

    def test_use_item_passes_when_in_inventory(self):
        wq = _make_wq(inventory_items={"raw-fish": 1})
        result = valid_actions(wq, self.kb, [UseItem(item="raw-fish")])
        self.assertEqual(len(result), 1)

    def test_use_item_filtered_when_not_in_inventory(self):
        wq = _make_wq(inventory_items={})
        result = valid_actions(wq, self.kb, [UseItem(item="raw-fish")])
        self.assertEqual(len(result), 0)

    # --- ALWAYS VALID ---

    def test_moveto_always_passes(self):
        wq = _make_wq()
        result = valid_actions(wq, self.kb, [MoveTo(position=Position(10.0, 10.0))])
        self.assertEqual(len(result), 1)

    def test_stop_movement_always_passes(self):
        wq = _make_wq()
        result = valid_actions(wq, self.kb, [StopMovement()])
        self.assertEqual(len(result), 1)

    def test_wait_always_passes(self):
        wq = _make_wq()
        result = valid_actions(wq, self.kb, [Wait(ticks=60)])
        self.assertEqual(len(result), 1)

    def test_noop_always_passes(self):
        wq = _make_wq()
        result = valid_actions(wq, self.kb, [NoOp()])
        self.assertEqual(len(result), 1)

    # --- PASS THROUGH and mixed ---

    def test_unknown_action_passes_through(self):
        # Actions not explicitly handled (e.g. vehicle/combat stubs) pass through.
        from bridge.actions import EnterVehicle
        wq = _make_wq()
        result = valid_actions(wq, self.kb, [EnterVehicle(entity_id=1)])
        self.assertEqual(len(result), 1)

    def test_mixed_candidate_list(self):
        entity = _make_entity(3, Position(0.5, 0.0))
        wq = _make_wq(
            reachable=[3],
            entities=[entity],
            inventory_items={"stone-furnace": 1, "heavy-armor": 1},
        )
        result = valid_actions(wq, self.kb, [
            MoveTo(position=Position(10.0, 10.0)),            # always valid
            MineEntity(entity_id=3),                          # reachable → pass
            MineEntity(entity_id=99),                         # absent → filter
            PlaceEntity(item="stone-furnace", position=Position(5.0, 5.0)),  # pass
            PlaceEntity(item="missing-item", position=Position(6.0, 6.0)),   # filter
            RotateEntity(entity_id=3),                        # reachable → pass
            RotateEntity(entity_id=99),                       # absent → filter
            EquipArmor(item="heavy-armor"),                   # in inventory → pass
            EquipArmor(item="power-armor"),                   # missing → filter
            NoOp(),                                           # always valid
        ])
        passing_kinds = [a.kind for a in result]
        self.assertIn("MoveTo", passing_kinds)
        self.assertIn("NoOp", passing_kinds)
        self.assertEqual(passing_kinds.count("MineEntity"), 1)
        self.assertEqual(passing_kinds.count("PlaceEntity"), 1)
        self.assertEqual(passing_kinds.count("RotateEntity"), 1)
        self.assertEqual(passing_kinds.count("EquipArmor"), 1)
        self.assertEqual(len(result), 6)


if __name__ == "__main__":
    unittest.main()