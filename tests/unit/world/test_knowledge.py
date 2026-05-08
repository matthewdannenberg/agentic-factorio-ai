"""
tests/unit/world/test_knowledge.py

Unit tests for world/knowledge.py — SQLite-backed KnowledgeBase.

Windows compatibility note
--------------------------
SQLite holds a file lock for the lifetime of the connection. On Windows,
TemporaryDirectory.cleanup() will fail with PermissionError if any
KnowledgeBase connection is still open. Every test therefore uses KnowledgeBase
as a context manager (``with KnowledgeBase(...) as kb:``) so the connection is
guaranteed closed before the temp dir is deleted.

Run with:  python -m unittest tests.unit.world.test_knowledge
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
    _DB_NAME,
)


# ---------------------------------------------------------------------------
# Canned query responses
# ---------------------------------------------------------------------------

def _entity_json(name="assembling-machine-2", proto_type="assembling-machine",
                 tile_width=3, tile_height=3, has_recipe_slot=True,
                 ingredient_slots=4, output_slots=1) -> str:
    return json.dumps({"name": name, "type": proto_type,
                       "tile_width": tile_width, "tile_height": tile_height,
                       "has_recipe_slot": has_recipe_slot,
                       "ingredient_slots": ingredient_slots,
                       "output_slots": output_slots})


def _resource_json(name="iron-ore", is_fluid=False, is_infinite=False,
                   display_name="Iron Ore") -> str:
    return json.dumps({"name": name, "is_fluid": is_fluid,
                       "is_infinite": is_infinite, "display_name": display_name})


def _fluid_json(name="steam", default_temperature=165, max_temperature=500,
                fuel_value=0.0, emissions_multiplier=1.0) -> str:
    return json.dumps({"name": name, "default_temperature": default_temperature,
                       "max_temperature": max_temperature, "fuel_value": fuel_value,
                       "emissions_multiplier": emissions_multiplier})


def _recipe_json(name="iron-gear-wheel",
                 ingredients=None, products=None, made_in=None,
                 category="crafting", energy_required=0.5) -> str:
    return json.dumps({
        "name": name,
        "category": category,
        "energy_required": energy_required,
        "ingredients": ingredients or [{"name": "iron-plate", "amount": 2, "type": "item"}],
        "products":    products    or [{"name": name, "amount": 1,
                                        "probability": 1.0, "type": "item"}],
        "made_in":     made_in     or ["assembling-machine-1", "assembling-machine-2"],
        "enabled": True,
    })


def _tech_json(name="automation", prerequisites=None, recipe_effects=None) -> str:
    effects = [{"type": "unlock-recipe", "recipe": r}
               for r in (recipe_effects or ["assembling-machine-1",
                                            "long-handed-inserter"])]
    return json.dumps({"name": name,
                       "prerequisites": prerequisites or [],
                       "effects": effects,
                       "researched": False, "enabled": True})


# ---------------------------------------------------------------------------
# Helper — always used as a context manager in tests
# ---------------------------------------------------------------------------

def _kb(query_fn=None, tmp_dir=None) -> KnowledgeBase:
    """Return an open KnowledgeBase. Callers must use it as a context manager."""
    d = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp())
    return KnowledgeBase(data_dir=d, query_fn=query_fn)


# ===========================================================================
# Construction
# ===========================================================================

class TestConstruction(unittest.TestCase):
    def test_creates_data_dir_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb_dir = Path(tmp) / "deep" / "path"
            with KnowledgeBase(data_dir=kb_dir):
                self.assertTrue(kb_dir.exists())

    def test_creates_sqlite_db_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)):
                self.assertTrue((Path(tmp) / _DB_NAME).exists())

    def test_no_csv_or_json_files_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                kb.ensure_entity("iron-chest")
                kb.ensure_recipe("iron-gear-wheel")
                extensions = {f.suffix for f in Path(tmp).iterdir()}
            self.assertNotIn(".csv", extensions)
            self.assertNotIn(".json", extensions)

    def test_starts_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertEqual(len(kb.all_entities()), 0)
                self.assertEqual(len(kb.all_resources()), 0)
                self.assertEqual(len(kb.all_fluids()), 0)
                self.assertEqual(len(kb.all_recipes()), 0)
                self.assertEqual(len(kb.all_techs()), 0)

    def test_summary_contains_db_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                s = kb.summary()
            self.assertIn("db_path", s)
            self.assertTrue(s["db_path"].endswith(_DB_NAME))

    def test_summary_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                s = kb.summary()
            for key in ("entities", "resources", "fluids", "recipes", "techs",
                        "data_dir", "db_path"):
                self.assertIn(key, s)

    def test_context_manager_closes_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                pass
            # If connection is still open on Windows this would PermissionError;
            # the fact that cleanup succeeds proves it's closed.


# ===========================================================================
# Entity registry
# ===========================================================================

class TestEntityRegistry(unittest.TestCase):
    def test_ensure_placeholder_when_no_query_fn(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                rec = kb.ensure_entity("mystery-machine")
                self.assertTrue(rec.is_placeholder)
                self.assertEqual(rec.name, "mystery-machine")

    def test_ensure_with_query_fn_parses_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _entity_json()) as kb:
                rec = kb.ensure_entity("assembling-machine-2")
                self.assertFalse(rec.is_placeholder)
                self.assertEqual(rec.tile_width, 3)
                self.assertEqual(rec.ingredient_slots, 4)
                self.assertEqual(rec.category, EntityCategory.ASSEMBLY.value)

    def test_furnace_maps_to_smelting(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _entity_json(
                                   proto_type="furnace")) as kb:
                rec = kb.ensure_entity("electric-furnace")
                self.assertEqual(rec.category, EntityCategory.SMELTING.value)

    def test_unknown_proto_type_maps_to_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _entity_json(
                                   proto_type="space-wizard-device")) as kb:
                rec = kb.ensure_entity("mod-thing")
                self.assertEqual(rec.category, EntityCategory.OTHER.value)

    def test_ensure_idempotent_no_extra_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [0]
            def qfn(e): calls[0] += 1; return _entity_json()
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_entity("assembling-machine-2")
                kb.ensure_entity("assembling-machine-2")
                self.assertEqual(calls[0], 1)

    def test_bad_json_yields_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: "not json {{{") as kb:
                rec = kb.ensure_entity("broken-thing")
                self.assertTrue(rec.is_placeholder)

    def test_get_returns_none_for_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertIsNone(kb.get_entity("ghost"))

    def test_get_returns_record_after_ensure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                kb.ensure_entity("iron-chest")
                self.assertIsNotNone(kb.get_entity("iron-chest"))

    def test_category_enum_property(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _entity_json(
                                   proto_type="furnace")) as kb:
                rec = kb.ensure_entity("electric-furnace")
                self.assertEqual(rec.category_enum, EntityCategory.SMELTING)


# ===========================================================================
# Resource registry
# ===========================================================================

class TestResourceRegistry(unittest.TestCase):
    def test_placeholder_when_no_query_fn(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                rec = kb.ensure_resource("mystery-ore")
                self.assertTrue(rec.is_placeholder)
                self.assertFalse(rec.is_fluid)

    def test_parses_solid_resource(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _resource_json()) as kb:
                rec = kb.ensure_resource("iron-ore")
                self.assertFalse(rec.is_placeholder)
                self.assertFalse(rec.is_fluid)
                self.assertEqual(rec.display_name, "Iron Ore")

    def test_parses_fluid_infinite_resource(self):
        with tempfile.TemporaryDirectory() as tmp:
            qfn = lambda e: _resource_json(name="crude-oil", is_fluid=True,
                                           is_infinite=True, display_name="Crude Oil")
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                rec = kb.ensure_resource("crude-oil")
                self.assertTrue(rec.is_fluid)
                self.assertTrue(rec.is_infinite)

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [0]
            def qfn(e): calls[0] += 1; return _resource_json()
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_resource("iron-ore")
                kb.ensure_resource("iron-ore")
                self.assertEqual(calls[0], 1)

    def test_get_none_for_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertIsNone(kb.get_resource("nope"))


# ===========================================================================
# Fluid registry
# ===========================================================================

class TestFluidRegistry(unittest.TestCase):
    def test_placeholder_base_fluid(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                rec = kb.ensure_fluid("water")
                self.assertIsNone(rec.temperature)
                self.assertTrue(rec.is_placeholder)

    def test_parses_fluid(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _fluid_json()) as kb:
                rec = kb.ensure_fluid("steam")
                self.assertFalse(rec.is_placeholder)
                self.assertEqual(rec.default_temperature, 165)
                self.assertEqual(rec.max_temperature, 500)

    def test_temperature_variants_stored_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _fluid_json()) as kb:
                base = kb.ensure_fluid("steam")
                hot  = kb.ensure_fluid("steam", temperature=500)
                self.assertIsNone(base.temperature)
                self.assertEqual(hot.temperature, 500)
                self.assertEqual(hot.registry_key, "steam@500")
                self.assertIsNot(base, hot)

    def test_fuel_fluid(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _fluid_json(
                                   fuel_value=5_000_000.0)) as kb:
                rec = kb.ensure_fluid("rocket-fuel")
                self.assertTrue(rec.is_fuel)
                self.assertAlmostEqual(rec.fuel_value_mj, 5.0)

    def test_non_fuel_fluid(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _fluid_json(
                                   fuel_value=0.0)) as kb:
                self.assertFalse(kb.ensure_fluid("water").is_fuel)

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [0]
            def qfn(e): calls[0] += 1; return _fluid_json()
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_fluid("steam")
                kb.ensure_fluid("steam")
                self.assertEqual(calls[0], 1)

    def test_get_with_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _fluid_json()) as kb:
                kb.ensure_fluid("steam", temperature=165)
                self.assertIsNotNone(kb.get_fluid("steam", temperature=165))
                self.assertIsNone(kb.get_fluid("steam"))   # base not yet loaded


# ===========================================================================
# Recipe registry
# ===========================================================================

class TestRecipeRegistry(unittest.TestCase):
    def test_placeholder_when_no_query_fn(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                rec = kb.ensure_recipe("iron-gear-wheel")
                self.assertTrue(rec.is_placeholder)

    def test_parses_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _recipe_json()) as kb:
                rec = kb.ensure_recipe("iron-gear-wheel")
                self.assertFalse(rec.is_placeholder)
                self.assertEqual(len(rec.ingredients), 1)
                self.assertEqual(rec.ingredients[0].name, "iron-plate")
                self.assertEqual(rec.ingredients[0].amount, 2)
                self.assertEqual(len(rec.products), 1)
                self.assertEqual(rec.products[0].name, "iron-gear-wheel")
                self.assertEqual(rec.crafting_time, 0.5)
                self.assertIn("assembling-machine-1", rec.made_in)

    def test_fluid_ingredient_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(e):
                return json.dumps({
                    "name": "plastic-bar", "category": "chemistry",
                    "energy_required": 1.0,
                    "ingredients": [
                        {"name": "petroleum-gas", "amount": 20, "type": "fluid"},
                        {"name": "coal", "amount": 1, "type": "item"},
                    ],
                    "products": [{"name": "plastic-bar", "amount": 2,
                                  "probability": 1.0, "type": "item"}],
                    "made_in": ["chemical-plant"], "enabled": True,
                })
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                rec = kb.ensure_recipe("plastic-bar")
                fluid_ing = next(i for i in rec.ingredients
                                 if i.name == "petroleum-gas")
                self.assertTrue(fluid_ing.is_fluid)
                item_ing = next(i for i in rec.ingredients if i.name == "coal")
                self.assertFalse(item_ing.is_fluid)

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [0]
            def qfn(e): calls[0] += 1; return _recipe_json()
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_recipe("iron-gear-wheel")
                kb.ensure_recipe("iron-gear-wheel")
                self.assertEqual(calls[0], 1)

    def test_get_none_for_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertIsNone(kb.get_recipe("nope"))

    def test_recipes_for_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _recipe_json()) as kb:
                kb.ensure_recipe("iron-gear-wheel")
                results = kb.recipes_for_product("iron-gear-wheel")
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].name, "iron-gear-wheel")

    def test_recipes_for_product_empty_for_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertEqual(kb.recipes_for_product("unicorn-dust"), [])

    def test_recipes_for_ingredient(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _recipe_json()) as kb:
                kb.ensure_recipe("iron-gear-wheel")
                self.assertEqual(len(kb.recipes_for_ingredient("iron-plate")), 1)

    def test_recipes_for_ingredient_empty_for_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertEqual(kb.recipes_for_ingredient("nope"), [])

    def test_recipes_made_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _recipe_json(
                                   made_in=["assembling-machine-1",
                                            "assembling-machine-2"])) as kb:
                kb.ensure_recipe("iron-gear-wheel")
                results = kb.recipes_made_in("assembling-machine-1")
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].name, "iron-gear-wheel")

    def test_recipes_made_in_multiple(self):
        with tempfile.TemporaryDirectory() as tmp:
            call_count = [0]
            def qfn(e):
                call_count[0] += 1
                if call_count[0] == 1:
                    return _recipe_json(name="iron-gear-wheel",
                                        made_in=["assembling-machine-1"])
                return _recipe_json(
                    name="copper-cable",
                    products=[{"name": "copper-cable", "amount": 2,
                               "probability": 1.0, "type": "item"}],
                    ingredients=[{"name": "copper-plate", "amount": 1,
                                  "type": "item"}],
                    made_in=["assembling-machine-1"])
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_recipe("iron-gear-wheel")
                kb.ensure_recipe("copper-cable")
                results = kb.recipes_made_in("assembling-machine-1")
                names = {r.name for r in results}
                self.assertIn("iron-gear-wheel", names)
                self.assertIn("copper-cable", names)

    def test_recipes_made_in_empty_for_unknown_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertEqual(kb.recipes_made_in("ghost-machine"), [])


# ===========================================================================
# Tech registry
# ===========================================================================

class TestTechRegistry(unittest.TestCase):
    def test_placeholder_when_no_query_fn(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                rec = kb.ensure_tech("automation")
                self.assertTrue(rec.is_placeholder)

    def test_parses_tech(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _tech_json()) as kb:
                rec = kb.ensure_tech("automation")
                self.assertFalse(rec.is_placeholder)
                self.assertEqual(rec.prerequisites, [])
                self.assertIn("assembling-machine-1", rec.unlocks_recipes)

    def test_parses_prerequisites(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(e):
                if "logistics" in e:
                    return _tech_json(name="logistics",
                                     prerequisites=["automation"])
                return _tech_json()
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_tech("automation")
                kb.ensure_tech("logistics")
                self.assertEqual(kb.prerequisites("logistics"), ["automation"])

    def test_all_prerequisites_transitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(e):
                if "logistics" in e:
                    return _tech_json(name="logistics",
                                     prerequisites=["automation"])
                return _tech_json()
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_tech("automation")
                kb.ensure_tech("logistics")
                self.assertIn("automation", kb.all_prerequisites("logistics"))

    def test_prerequisites_empty_for_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                kb.ensure_tech("automation")
                self.assertEqual(kb.prerequisites("automation"), [])

    def test_all_prerequisites_empty_for_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertEqual(kb.all_prerequisites("ghost"), set())

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [0]
            def qfn(e): calls[0] += 1; return _tech_json()
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_tech("automation")
                kb.ensure_tech("automation")
                self.assertEqual(calls[0], 1)

    def test_get_none_for_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertIsNone(kb.get_tech("nope"))

    def test_techs_unlocking_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _tech_json(
                                   recipe_effects=["assembling-machine-1"])) as kb:
                kb.ensure_tech("automation")
                results = kb.techs_unlocking_recipe("assembling-machine-1")
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].name, "automation")

    def test_techs_unlocking_recipe_empty_for_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertEqual(kb.techs_unlocking_recipe("ghost-recipe"), [])


# ===========================================================================
# production_chain (recursive CTE)
# ===========================================================================

class TestProductionChain(unittest.TestCase):
    def _build_kb(self, kb: KnowledgeBase) -> None:
        """
        Teach the KB three recipes:
          electronic-circuit ← iron-plate + copper-cable
          copper-cable       ← copper-plate
          iron-gear-wheel    ← iron-plate
        """
        recipes = {
            "electronic-circuit": json.dumps({
                "name": "electronic-circuit", "category": "crafting",
                "energy_required": 0.5,
                "ingredients": [
                    {"name": "iron-plate",   "amount": 1, "type": "item"},
                    {"name": "copper-cable", "amount": 3, "type": "item"},
                ],
                "products": [{"name": "electronic-circuit", "amount": 1,
                              "probability": 1.0, "type": "item"}],
                "made_in": ["assembling-machine-1"], "enabled": True,
            }),
            "copper-cable": json.dumps({
                "name": "copper-cable", "category": "crafting",
                "energy_required": 0.5,
                "ingredients": [{"name": "copper-plate", "amount": 1, "type": "item"}],
                "products": [{"name": "copper-cable", "amount": 2,
                              "probability": 1.0, "type": "item"}],
                "made_in": ["assembling-machine-1"], "enabled": True,
            }),
            "iron-gear-wheel": json.dumps({
                "name": "iron-gear-wheel", "category": "crafting",
                "energy_required": 0.5,
                "ingredients": [{"name": "iron-plate", "amount": 2, "type": "item"}],
                "products": [{"name": "iron-gear-wheel", "amount": 1,
                              "probability": 1.0, "type": "item"}],
                "made_in": ["assembling-machine-1"], "enabled": True,
            }),
        }

        # Patch the KB's query_fn for this setup call
        def qfn(expr):
            for name, j in recipes.items():
                if f'"{name}"' in expr:
                    return j
            return json.dumps({"ok": False, "reason": "unknown"})

        kb._query_fn = qfn
        for name in recipes:
            kb.ensure_recipe(name)
        kb._query_fn = None

    def test_direct_ingredients_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self._build_kb(kb)
                self.assertIn("iron-plate", kb.production_chain("iron-gear-wheel"))

    def test_target_item_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self._build_kb(kb)
                self.assertNotIn("iron-gear-wheel",
                                 kb.production_chain("iron-gear-wheel"))

    def test_transitive_ingredients_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self._build_kb(kb)
                chain = kb.production_chain("electronic-circuit")
                self.assertIn("iron-plate",   chain)
                self.assertIn("copper-cable", chain)
                self.assertIn("copper-plate", chain)   # transitive

    def test_unknown_item_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self._build_kb(kb)
                self.assertEqual(kb.production_chain("unicorn-dust"), set())

    def test_raw_material_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self._build_kb(kb)
                # iron-plate has no recipe in our KB → no chain
                self.assertEqual(kb.production_chain("iron-plate"), set())


# ===========================================================================
# Persistence (simulated restart)
# ===========================================================================

class TestPersistence(unittest.TestCase):
    def test_entity_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _entity_json(
                                   tile_width=3, ingredient_slots=4)) as kb:
                kb.ensure_entity("assembling-machine-2")

            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                rec = kb.get_entity("assembling-machine-2")
                self.assertIsNotNone(rec)
                self.assertEqual(rec.tile_width, 3)
                self.assertEqual(rec.ingredient_slots, 4)
                self.assertFalse(rec.is_placeholder)

    def test_resource_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _resource_json()) as kb:
                kb.ensure_resource("iron-ore")

            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                rec = kb.get_resource("iron-ore")
                self.assertIsNotNone(rec)
                self.assertFalse(rec.is_fluid)
                self.assertEqual(rec.display_name, "Iron Ore")

    def test_fluid_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _fluid_json(
                                   default_temperature=165)) as kb:
                kb.ensure_fluid("steam", temperature=165)

            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                rec = kb.get_fluid("steam", temperature=165)
                self.assertIsNotNone(rec)
                self.assertEqual(rec.temperature, 165)
                self.assertEqual(rec.default_temperature, 165)

    def test_recipe_with_ingredients_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _recipe_json()) as kb:
                kb.ensure_recipe("iron-gear-wheel")

            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                rec = kb.get_recipe("iron-gear-wheel")
                self.assertIsNotNone(rec)
                self.assertEqual(len(rec.ingredients), 1)
                self.assertEqual(rec.ingredients[0].name, "iron-plate")
                self.assertEqual(rec.ingredients[0].amount, 2)
                self.assertIn("assembling-machine-1", rec.made_in)

    def test_recipe_ingredient_order_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(e):
                return json.dumps({
                    "name": "processing-unit", "category": "crafting",
                    "energy_required": 10.0,
                    "ingredients": [
                        {"name": "electronic-circuit", "amount": 20, "type": "item"},
                        {"name": "advanced-circuit",   "amount": 2,  "type": "item"},
                        {"name": "sulfuric-acid",      "amount": 5,  "type": "fluid"},
                    ],
                    "products": [{"name": "processing-unit", "amount": 1,
                                  "probability": 1.0, "type": "item"}],
                    "made_in": ["assembling-machine-3"], "enabled": True,
                })
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_recipe("processing-unit")

            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                rec = kb.get_recipe("processing-unit")
                names = [i.name for i in rec.ingredients]
                self.assertEqual(names, ["electronic-circuit", "advanced-circuit",
                                         "sulfuric-acid"])

    def test_tech_with_prereqs_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            def qfn(e):
                if "logistics" in e:
                    return _tech_json(name="logistics",
                                     prerequisites=["automation"],
                                     recipe_effects=["transport-belt"])
                return _tech_json()
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_tech("automation")
                kb.ensure_tech("logistics")

            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertIn("automation", kb.all_techs())
                self.assertIn("logistics", kb.all_techs())
                self.assertEqual(kb.prerequisites("logistics"), ["automation"])
                self.assertIn("transport-belt",
                              kb.get_tech("logistics").unlocks_recipes)

    def test_multiple_entities_accumulate(self):
        with tempfile.TemporaryDirectory() as tmp:
            call_count = [0]
            def qfn(e):
                call_count[0] += 1
                if "iron-chest" in e:
                    return _entity_json(name="iron-chest", proto_type="container",
                                        tile_width=1, tile_height=1,
                                        has_recipe_slot=False,
                                        ingredient_slots=0, output_slots=0)
                return _entity_json()
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_entity("assembling-machine-2")
                kb.ensure_entity("iron-chest")

            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertIn("assembling-machine-2", kb.all_entities())
                self.assertIn("iron-chest", kb.all_entities())
                self.assertEqual(kb.get_entity("iron-chest").category,
                                 EntityCategory.STORAGE.value)

    def test_sql_queries_work_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _recipe_json()) as kb:
                kb.ensure_recipe("iron-gear-wheel")

            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                self.assertEqual(len(kb.recipes_for_product("iron-gear-wheel")), 1)
                self.assertEqual(len(kb.recipes_for_ingredient("iron-plate")), 1)
                self.assertEqual(len(kb.recipes_made_in("assembling-machine-1")), 1)




# ===========================================================================
# Placeholder enrichment
# ===========================================================================

class TestPlaceholderEnrichment(unittest.TestCase):
    """
    Tests for the two-phase lifecycle:
      1. Placeholder created offline (no query_fn)
      2. query_fn becomes available → placeholder is replaced with real data
    """

    def test_ensure_re_queries_placeholder_when_query_fn_available(self):
        # Phase 1: learn tech offline → placeholder
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                rec = kb.ensure_tech("automation")
                self.assertTrue(rec.is_placeholder)

                # Phase 2: query_fn becomes available mid-session
                kb._query_fn = lambda e: _tech_json()
                rec2 = kb.ensure_tech("automation")
                self.assertFalse(rec2.is_placeholder)
                self.assertIn("assembling-machine-1", rec2.unlocks_recipes)

    def test_ensure_does_not_re_query_non_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [0]
            def qfn(e): calls[0] += 1; return _tech_json()
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_tech("automation")
                first_count = calls[0]
                kb.ensure_tech("automation")   # already real → no re-query
                self.assertEqual(calls[0], first_count)

    def test_ensure_placeholder_not_re_queried_without_query_fn(self):
        # Offline session: placeholder stays placeholder even on repeated calls
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                kb.ensure_tech("automation")
                rec = kb.ensure_tech("automation")
                self.assertTrue(rec.is_placeholder)

    def test_enrich_placeholders_resolves_tech(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Session 1: offline → placeholder stored in DB
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                kb.ensure_tech("automation")
                self.assertTrue(kb.get_tech("automation").is_placeholder)

            # Session 2: now online
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _tech_json()) as kb:
                counts = kb.enrich_placeholders()
                self.assertEqual(counts["techs"], 1)
                self.assertFalse(kb.get_tech("automation").is_placeholder)
                self.assertIn("assembling-machine-1",
                              kb.get_tech("automation").unlocks_recipes)

    def test_enrich_placeholders_resolves_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                kb.ensure_entity("assembling-machine-2")
                self.assertTrue(kb.get_entity("assembling-machine-2").is_placeholder)

            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _entity_json()) as kb:
                counts = kb.enrich_placeholders()
                self.assertEqual(counts["entities"], 1)
                self.assertFalse(
                    kb.get_entity("assembling-machine-2").is_placeholder)

    def test_enrich_placeholders_resolves_resource(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                kb.ensure_resource("iron-ore")

            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _resource_json()) as kb:
                counts = kb.enrich_placeholders()
                self.assertEqual(counts["resources"], 1)
                self.assertFalse(kb.get_resource("iron-ore").is_placeholder)

    def test_enrich_placeholders_resolves_fluid(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                kb.ensure_fluid("steam", temperature=165)

            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _fluid_json()) as kb:
                counts = kb.enrich_placeholders()
                self.assertEqual(counts["fluids"], 1)
                self.assertFalse(
                    kb.get_fluid("steam", temperature=165).is_placeholder)

    def test_enrich_placeholders_resolves_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                kb.ensure_recipe("iron-gear-wheel")

            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _recipe_json()) as kb:
                counts = kb.enrich_placeholders()
                self.assertEqual(counts["recipes"], 1)
                self.assertFalse(
                    kb.get_recipe("iron-gear-wheel").is_placeholder)

    def test_enrich_placeholders_skips_non_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [0]
            def qfn(e): calls[0] += 1; return _tech_json()
            with KnowledgeBase(data_dir=Path(tmp), query_fn=qfn) as kb:
                kb.ensure_tech("automation")   # real record, 1 query
                call_count_after_ensure = calls[0]
                counts = kb.enrich_placeholders()
                self.assertEqual(counts["techs"], 0)
                # enrich should not have triggered another query
                self.assertEqual(calls[0], call_count_after_ensure)

    def test_enrich_placeholders_noop_without_query_fn(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                kb.ensure_tech("automation")
                counts = kb.enrich_placeholders()
                self.assertEqual(counts["techs"], 0)
                # placeholder unchanged
                self.assertTrue(kb.get_tech("automation").is_placeholder)

    def test_enrich_placeholders_returns_zero_counts_when_no_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _tech_json()) as kb:
                kb.ensure_tech("automation")   # real record
                counts = kb.enrich_placeholders()
                self.assertEqual(sum(counts.values()), 0)

    def test_enriched_data_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Session 1: offline placeholder
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                kb.ensure_tech("automation")

            # Session 2: enrich
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=lambda e: _tech_json()) as kb:
                kb.enrich_placeholders()

            # Session 3: no query_fn — enriched data should still be present
            with KnowledgeBase(data_dir=Path(tmp)) as kb:
                rec = kb.get_tech("automation")
                self.assertIsNotNone(rec)
                self.assertFalse(rec.is_placeholder)
                self.assertIn("assembling-machine-1", rec.unlocks_recipes)
if __name__ == "__main__":
    unittest.main(verbosity=2)