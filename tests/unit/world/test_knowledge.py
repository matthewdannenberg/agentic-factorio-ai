"""
tests/unit/world/test_knowledge.py

Unit tests for world/knowledge.py — KnowledgeBase and all record types.

All tests run without a live Factorio instance. query_fn is either None
(offline mode) or a controlled stub that returns canned JSON.

Run with:  python -m unittest tests.unit.world.test_knowledge
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Optional

from world.knowledge import (
    EntityCategory,
    EntityRecord,
    FluidRecord,
    IngredientRecord,
    KnowledgeBase,
    ProductRecord,
    RecipeRecord,
    ResourceRecord,
    TechRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kb(query_fn=None, tmp_dir=None) -> KnowledgeBase:
    """Create a KnowledgeBase backed by a temp directory."""
    d = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp())
    return KnowledgeBase(data_dir=d, query_fn=query_fn)


def _entity_json(name="assembling-machine-2", proto_type="assembling-machine",
                 tile_width=3, tile_height=3, has_recipe_slot=True,
                 ingredient_slots=4, output_slots=1) -> str:
    return json.dumps({
        "name": name, "type": proto_type,
        "tile_width": tile_width, "tile_height": tile_height,
        "has_recipe_slot": has_recipe_slot,
        "ingredient_slots": ingredient_slots,
        "output_slots": output_slots,
    })


def _resource_json(name="iron-ore", is_fluid=False, is_infinite=False,
                   display_name="Iron Ore") -> str:
    return json.dumps({
        "name": name, "is_fluid": is_fluid,
        "is_infinite": is_infinite, "display_name": display_name,
    })


def _fluid_json(name="steam", default_temperature=165, max_temperature=500,
                fuel_value=0.0, emissions_multiplier=1.0) -> str:
    return json.dumps({
        "name": name,
        "default_temperature": default_temperature,
        "max_temperature": max_temperature,
        "fuel_value": fuel_value,
        "emissions_multiplier": emissions_multiplier,
    })


def _recipe_json(name="iron-gear-wheel") -> str:
    return json.dumps({
        "name": name,
        "category": "crafting",
        "energy_required": 0.5,
        "ingredients": [{"name": "iron-plate", "amount": 2, "type": "item"}],
        "products": [{"name": "iron-gear-wheel", "amount": 1,
                      "probability": 1.0, "type": "item"}],
        "made_in": ["assembling-machine-1", "assembling-machine-2"],
        "enabled": True,
    })


def _tech_json(name="automation") -> str:
    return json.dumps({
        "name": name,
        "prerequisites": [],
        "effects": [
            {"type": "unlock-recipe", "recipe": "assembling-machine-1"},
            {"type": "unlock-recipe", "recipe": "long-handed-inserter"},
        ],
        "researched": False,
        "enabled": True,
    })


def _tech_logistics_json() -> str:
    return json.dumps({
        "name": "logistics",
        "prerequisites": ["automation"],
        "effects": [
            {"type": "unlock-recipe", "recipe": "transport-belt"},
        ],
        "researched": False,
        "enabled": True,
    })


# ---------------------------------------------------------------------------
# KnowledgeBase construction and file handling
# ---------------------------------------------------------------------------

class TestKnowledgeBaseConstruction(unittest.TestCase):
    def test_creates_data_dir_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb_dir = Path(tmp) / "deep" / "nested" / "knowledge"
            kb = KnowledgeBase(data_dir=kb_dir)
            self.assertTrue(kb_dir.exists())

    def test_starts_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            self.assertEqual(len(kb.all_entities()), 0)
            self.assertEqual(len(kb.all_resources()), 0)
            self.assertEqual(len(kb.all_fluids()), 0)
            self.assertEqual(len(kb.all_recipes()), 0)
            self.assertEqual(len(kb.all_techs()), 0)

    def test_summary_keys_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            s = kb.summary()
            for key in ("entities", "resources", "fluids", "recipes", "techs", "data_dir"):
                self.assertIn(key, s)

    def test_loads_existing_csv_on_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            # First KB: create an entity
            kb1 = _kb(tmp_dir=tmp)
            kb1._entities["iron-chest"] = EntityRecord(
                name="iron-chest", proto_type="container",
                category=EntityCategory.STORAGE.value,
                tile_width=1, tile_height=1,
                has_recipe_slot=False, ingredient_slots=0,
                output_slots=0, is_placeholder=False,
            )
            kb1._append_csv("entities.csv", kb1._entities["iron-chest"],
                            EntityRecord._FIELDS)
            # Second KB: should load the record
            kb2 = _kb(tmp_dir=tmp)
            self.assertIn("iron-chest", kb2.all_entities())

    def test_loads_existing_json_on_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb1 = _kb(tmp_dir=tmp)
            rec = TechRecord(
                name="automation", prerequisites=[],
                unlocks_recipes=["assembling-machine-1"],
                unlocks_entities=["assembling-machine-1"],
                requires_dlc=False, is_placeholder=False,
            )
            kb1._techs["automation"] = rec
            kb1._rewrite_json("tech_tree.json", kb1._techs,
                              lambda r: r.to_json_obj())
            kb2 = _kb(tmp_dir=tmp)
            self.assertIn("automation", kb2.all_techs())


# ---------------------------------------------------------------------------
# Entity registry
# ---------------------------------------------------------------------------

class TestEntityRegistry(unittest.TestCase):
    def test_ensure_unknown_no_query_fn_returns_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            rec = kb.ensure_entity("some-mod-machine")
            self.assertIsInstance(rec, EntityRecord)
            self.assertTrue(rec.is_placeholder)
            self.assertEqual(rec.name, "some-mod-machine")

    def test_ensure_with_query_fn_parses_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def qfn(expr):
                calls.append(expr)
                return _entity_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            rec = kb.ensure_entity("assembling-machine-2")
            self.assertFalse(rec.is_placeholder)
            self.assertEqual(rec.tile_width, 3)
            self.assertEqual(rec.ingredient_slots, 4)
            self.assertEqual(rec.category, EntityCategory.ASSEMBLY.value)

    def test_ensure_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            call_count = [0]
            def qfn(expr):
                call_count[0] += 1
                return _entity_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_entity("assembling-machine-2")
            kb.ensure_entity("assembling-machine-2")
            self.assertEqual(call_count[0], 1)

    def test_ensure_persists_to_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            kb.ensure_entity("wooden-chest")
            self.assertTrue((Path(tmp) / "entities.csv").exists())

    def test_get_returns_none_for_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            self.assertIsNone(kb.get_entity("nonexistent"))

    def test_get_returns_record_after_ensure(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            kb.ensure_entity("wooden-chest")
            self.assertIsNotNone(kb.get_entity("wooden-chest"))

    def test_proto_type_maps_to_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _entity_json(proto_type="furnace")
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            rec = kb.ensure_entity("electric-furnace")
            self.assertEqual(rec.category, EntityCategory.SMELTING.value)

    def test_unknown_proto_type_maps_to_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _entity_json(proto_type="totally-modded-type")
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            rec = kb.ensure_entity("mystery-building")
            self.assertEqual(rec.category, EntityCategory.OTHER.value)

    def test_bad_json_from_query_returns_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return "this is not json {"
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            rec = kb.ensure_entity("broken-entity")
            self.assertTrue(rec.is_placeholder)

    def test_entity_record_category_enum_property(self):
        rec = EntityRecord(
            name="test", proto_type="furnace",
            category=EntityCategory.SMELTING.value,
            tile_width=2, tile_height=2,
            has_recipe_slot=True, ingredient_slots=1,
            output_slots=1, is_placeholder=False,
        )
        self.assertEqual(rec.category_enum, EntityCategory.SMELTING)


# ---------------------------------------------------------------------------
# Resource registry
# ---------------------------------------------------------------------------

class TestResourceRegistry(unittest.TestCase):
    def test_ensure_unknown_no_query_fn_returns_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            rec = kb.ensure_resource("mystery-ore")
            self.assertTrue(rec.is_placeholder)
            self.assertFalse(rec.is_fluid)
            self.assertFalse(rec.is_infinite)

    def test_ensure_with_query_fn_parses_iron_ore(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _resource_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            rec = kb.ensure_resource("iron-ore")
            self.assertFalse(rec.is_placeholder)
            self.assertFalse(rec.is_fluid)
            self.assertEqual(rec.display_name, "Iron Ore")

    def test_ensure_with_query_fn_parses_crude_oil(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _resource_json(name="crude-oil", is_fluid=True, is_infinite=True,
                                      display_name="Crude Oil")
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            rec = kb.ensure_resource("crude-oil")
            self.assertTrue(rec.is_fluid)
            self.assertTrue(rec.is_infinite)

    def test_ensure_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            count = [0]
            def qfn(expr):
                count[0] += 1
                return _resource_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_resource("iron-ore")
            kb.ensure_resource("iron-ore")
            self.assertEqual(count[0], 1)

    def test_persists_to_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            kb.ensure_resource("copper-ore")
            self.assertTrue((Path(tmp) / "resources.csv").exists())

    def test_get_returns_none_for_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            self.assertIsNone(kb.get_resource("nonexistent"))


# ---------------------------------------------------------------------------
# Fluid registry
# ---------------------------------------------------------------------------

class TestFluidRegistry(unittest.TestCase):
    def test_ensure_base_fluid_no_query_fn(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            rec = kb.ensure_fluid("water")
            self.assertIsNone(rec.temperature)
            self.assertTrue(rec.is_placeholder)

    def test_ensure_fluid_with_query_fn(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _fluid_json(name="steam", default_temperature=165,
                                   max_temperature=500)
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            rec = kb.ensure_fluid("steam")
            self.assertFalse(rec.is_placeholder)
            self.assertEqual(rec.default_temperature, 165)
            self.assertEqual(rec.max_temperature, 500)

    def test_temperature_variant_stored_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _fluid_json(name="steam", default_temperature=165,
                                   max_temperature=500)
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            base = kb.ensure_fluid("steam")
            hot = kb.ensure_fluid("steam", temperature=500)
            self.assertIsNone(base.temperature)
            self.assertEqual(hot.temperature, 500)
            self.assertEqual(hot.registry_key, "steam@500")
            self.assertEqual(base.registry_key, "steam")
            # They are different registry entries
            self.assertIsNot(base, hot)

    def test_is_fuel_true_when_fuel_value_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _fluid_json(name="rocket-fuel", fuel_value=5_000_000.0)
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            rec = kb.ensure_fluid("rocket-fuel")
            self.assertTrue(rec.is_fuel)
            self.assertAlmostEqual(rec.fuel_value_mj, 5.0)

    def test_is_fuel_false_for_water(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _fluid_json(name="water", fuel_value=0.0)
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            rec = kb.ensure_fluid("water")
            self.assertFalse(rec.is_fuel)

    def test_ensure_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            count = [0]
            def qfn(expr):
                count[0] += 1
                return _fluid_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_fluid("steam")
            kb.ensure_fluid("steam")
            self.assertEqual(count[0], 1)

    def test_persists_to_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            kb.ensure_fluid("water")
            self.assertTrue((Path(tmp) / "fluids.csv").exists())

    def test_get_fluid_with_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _fluid_json(name="steam")
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_fluid("steam", temperature=165)
            self.assertIsNotNone(kb.get_fluid("steam", temperature=165))
            self.assertIsNone(kb.get_fluid("steam"))  # base not yet loaded


# ---------------------------------------------------------------------------
# Recipe registry
# ---------------------------------------------------------------------------

class TestRecipeRegistry(unittest.TestCase):
    def test_ensure_unknown_no_query_fn_returns_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            rec = kb.ensure_recipe("iron-gear-wheel")
            self.assertTrue(rec.is_placeholder)
            self.assertEqual(rec.name, "iron-gear-wheel")

    def test_ensure_with_query_fn_parses_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _recipe_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            rec = kb.ensure_recipe("iron-gear-wheel")
            self.assertFalse(rec.is_placeholder)
            self.assertEqual(len(rec.ingredients), 1)
            self.assertEqual(rec.ingredients[0].name, "iron-plate")
            self.assertEqual(rec.ingredients[0].amount, 2)
            self.assertEqual(len(rec.products), 1)
            self.assertEqual(rec.products[0].name, "iron-gear-wheel")
            self.assertEqual(rec.crafting_time, 0.5)
            self.assertIn("assembling-machine-1", rec.made_in)

    def test_ensure_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            count = [0]
            def qfn(expr):
                count[0] += 1
                return _recipe_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_recipe("iron-gear-wheel")
            kb.ensure_recipe("iron-gear-wheel")
            self.assertEqual(count[0], 1)

    def test_persists_to_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            kb.ensure_recipe("iron-gear-wheel")
            self.assertTrue((Path(tmp) / "recipes.json").exists())

    def test_recipes_for_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _recipe_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_recipe("iron-gear-wheel")
            results = kb.recipes_for_product("iron-gear-wheel")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].name, "iron-gear-wheel")

    def test_recipes_for_ingredient(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _recipe_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_recipe("iron-gear-wheel")
            results = kb.recipes_for_ingredient("iron-plate")
            self.assertEqual(len(results), 1)

    def test_recipes_for_product_empty_when_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            self.assertEqual(kb.recipes_for_product("unicorn-dust"), [])

    def test_ingredient_is_fluid_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return json.dumps({
                    "name": "plastic-bar",
                    "category": "chemistry",
                    "energy_required": 1.0,
                    "ingredients": [
                        {"name": "petroleum-gas", "amount": 20,
                         "type": "fluid"},
                        {"name": "coal", "amount": 1, "type": "item"},
                    ],
                    "products": [{"name": "plastic-bar", "amount": 2,
                                  "probability": 1.0, "type": "item"}],
                    "made_in": ["chemical-plant"],
                    "enabled": True,
                })
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            rec = kb.ensure_recipe("plastic-bar")
            fluid_ing = next(i for i in rec.ingredients if i.name == "petroleum-gas")
            self.assertTrue(fluid_ing.is_fluid)


# ---------------------------------------------------------------------------
# Tech tree registry
# ---------------------------------------------------------------------------

class TestTechRegistry(unittest.TestCase):
    def test_ensure_unknown_no_query_fn_returns_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            rec = kb.ensure_tech("automation")
            self.assertTrue(rec.is_placeholder)

    def test_ensure_with_query_fn_parses_tech(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _tech_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            rec = kb.ensure_tech("automation")
            self.assertFalse(rec.is_placeholder)
            self.assertEqual(rec.prerequisites, [])
            self.assertIn("assembling-machine-1", rec.unlocks_recipes)

    def test_ensure_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            count = [0]
            def qfn(expr):
                count[0] += 1
                return _tech_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_tech("automation")
            kb.ensure_tech("automation")
            self.assertEqual(count[0], 1)

    def test_persists_to_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            kb.ensure_tech("automation")
            self.assertTrue((Path(tmp) / "tech_tree.json").exists())

    def test_prerequisites_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                if "logistics" in expr:
                    return _tech_logistics_json()
                return _tech_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_tech("automation")
            kb.ensure_tech("logistics")
            self.assertEqual(kb.prerequisites("logistics"), ["automation"])

    def test_all_prerequisites_transitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                if "logistics" in expr:
                    return _tech_logistics_json()
                return _tech_json()
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_tech("automation")
            kb.ensure_tech("logistics")
            all_p = kb.all_prerequisites("logistics")
            self.assertIn("automation", all_p)

    def test_all_prerequisites_unknown_tech_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            self.assertEqual(kb.all_prerequisites("ghost-tech"), set())


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistenceRoundTrip(unittest.TestCase):
    def test_entity_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _entity_json(name="assembling-machine-2",
                                    proto_type="assembling-machine",
                                    tile_width=3, tile_height=3,
                                    has_recipe_slot=True,
                                    ingredient_slots=4, output_slots=1)
            kb1 = _kb(query_fn=qfn, tmp_dir=tmp)
            kb1.ensure_entity("assembling-machine-2")
            # Simulate restart
            kb2 = _kb(tmp_dir=tmp)
            rec = kb2.get_entity("assembling-machine-2")
            self.assertIsNotNone(rec)
            self.assertEqual(rec.tile_width, 3)
            self.assertEqual(rec.ingredient_slots, 4)
            self.assertFalse(rec.is_placeholder)

    def test_resource_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _resource_json(name="iron-ore", is_fluid=False,
                                      is_infinite=False, display_name="Iron Ore")
            kb1 = _kb(query_fn=qfn, tmp_dir=tmp)
            kb1.ensure_resource("iron-ore")
            kb2 = _kb(tmp_dir=tmp)
            rec = kb2.get_resource("iron-ore")
            self.assertIsNotNone(rec)
            self.assertFalse(rec.is_fluid)
            self.assertEqual(rec.display_name, "Iron Ore")

    def test_fluid_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _fluid_json(name="steam", default_temperature=165,
                                   max_temperature=500)
            kb1 = _kb(query_fn=qfn, tmp_dir=tmp)
            kb1.ensure_fluid("steam", temperature=165)
            kb2 = _kb(tmp_dir=tmp)
            rec = kb2.get_fluid("steam", temperature=165)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.temperature, 165)
            self.assertEqual(rec.default_temperature, 165)

    def test_recipe_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _recipe_json()
            kb1 = _kb(query_fn=qfn, tmp_dir=tmp)
            kb1.ensure_recipe("iron-gear-wheel")
            kb2 = _kb(tmp_dir=tmp)
            rec = kb2.get_recipe("iron-gear-wheel")
            self.assertIsNotNone(rec)
            self.assertEqual(len(rec.ingredients), 1)
            self.assertEqual(rec.ingredients[0].name, "iron-plate")

    def test_tech_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                return _tech_json()
            kb1 = _kb(query_fn=qfn, tmp_dir=tmp)
            kb1.ensure_tech("automation")
            kb2 = _kb(tmp_dir=tmp)
            rec = kb2.get_tech("automation")
            self.assertIsNotNone(rec)
            self.assertIn("assembling-machine-1", rec.unlocks_recipes)
            self.assertFalse(rec.is_placeholder)

    def test_multiple_entities_accumulate_in_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(expr):
                if "iron-chest" in expr:
                    return _entity_json(name="iron-chest", proto_type="container",
                                        tile_width=1, tile_height=1,
                                        has_recipe_slot=False,
                                        ingredient_slots=0, output_slots=0)
                return _entity_json()
            kb1 = _kb(query_fn=qfn, tmp_dir=tmp)
            kb1.ensure_entity("assembling-machine-2")
            kb1.ensure_entity("iron-chest")
            kb2 = _kb(tmp_dir=tmp)
            self.assertIn("assembling-machine-2", kb2.all_entities())
            self.assertIn("iron-chest", kb2.all_entities())


# ---------------------------------------------------------------------------
# TechTree facade
# ---------------------------------------------------------------------------

class TestTechTreeFacade(unittest.TestCase):
    """Smoke tests for world/tech_tree.py (the TechTree class)."""

    def _tech_tree_with_automation_and_logistics(self):
        from world.tech_tree import TechTree
        from world.state import ResearchState

        def qfn(expr):
            if "logistics" in expr:
                return _tech_logistics_json()
            return _tech_json()

        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_tech("automation")
            kb.ensure_tech("logistics")
            tree = TechTree(kb)
            return tree, kb, tmp

    def test_is_unlocked_true(self):
        from world.tech_tree import TechTree
        from world.state import ResearchState
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            tree = TechTree(kb)
            research = ResearchState(unlocked=["automation"])
            self.assertTrue(tree.is_unlocked("automation", research))

    def test_is_unlocked_false(self):
        from world.tech_tree import TechTree
        from world.state import ResearchState
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            tree = TechTree(kb)
            research = ResearchState(unlocked=[])
            self.assertFalse(tree.is_unlocked("automation", research))

    def test_is_reachable_placeholder_returns_false(self):
        from world.tech_tree import TechTree
        from world.state import ResearchState
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)   # no query_fn → placeholders
            tree = TechTree(kb)
            research = ResearchState(unlocked=["automation"])
            # placeholder has no prereq data → is_reachable returns False
            self.assertFalse(tree.is_reachable("logistics", research))

    def test_is_reachable_true_when_prereqs_met(self):
        from world.tech_tree import TechTree
        from world.state import ResearchState

        def qfn(expr):
            if "logistics" in expr:
                return _tech_logistics_json()
            return _tech_json()

        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_tech("automation")
            kb.ensure_tech("logistics")
            tree = TechTree(kb)
            research = ResearchState(unlocked=["automation"])
            self.assertTrue(tree.is_reachable("logistics", research))

    def test_prerequisites_empty_for_automation(self):
        from world.tech_tree import TechTree

        def qfn(expr):
            return _tech_json()

        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_tech("automation")
            tree = TechTree(kb)
            self.assertEqual(tree.prerequisites("automation"), [])

    def test_prerequisites_logistics(self):
        from world.tech_tree import TechTree

        def qfn(expr):
            if "logistics" in expr:
                return _tech_logistics_json()
            return _tech_json()

        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_tech("automation")
            kb.ensure_tech("logistics")
            tree = TechTree(kb)
            self.assertEqual(tree.prerequisites("logistics"), ["automation"])

    def test_path_to_raises_for_completely_unknown_tech(self):
        from world.tech_tree import TechTree
        from world.state import ResearchState
        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(tmp_dir=tmp)
            tree = TechTree(kb)
            research = ResearchState(unlocked=[])
            with self.assertRaises(ValueError):
                tree.path_to("ghost-tech-xyzzy", research)

    def test_path_to_empty_if_already_unlocked(self):
        from world.tech_tree import TechTree
        from world.state import ResearchState

        def qfn(expr):
            return _tech_json()

        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            kb.ensure_tech("automation")
            tree = TechTree(kb)
            research = ResearchState(unlocked=["automation"])
            self.assertEqual(tree.path_to("automation", research), [])

    def test_absorb_research_state_learns_techs(self):
        from world.tech_tree import TechTree
        from world.state import ResearchState

        def qfn(expr):
            if "logistics" in expr:
                return _tech_logistics_json()
            return _tech_json()

        with tempfile.TemporaryDirectory() as tmp:
            kb = _kb(query_fn=qfn, tmp_dir=tmp)
            tree = TechTree(kb)
            research = ResearchState(unlocked=["automation"],
                                     queued=["logistics"])
            tree.absorb_research_state(research)
            self.assertIsNotNone(kb.get_tech("automation"))
            self.assertIsNotNone(kb.get_tech("logistics"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
