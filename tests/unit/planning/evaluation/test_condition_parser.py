"""
tests/unit/planning/test_condition_parser.py

Unit tests for planning/evaluation/condition_parser.py.

Each test class covers one goal type. The module is a pure function with no
external dependencies, so no stubs or mocks are needed.

Run with:  python -m pytest tests/unit/planning/test_condition_parser.py -v
"""

from __future__ import annotations

import unittest

from planning.evaluation.condition_parser import params_from_condition


class TestCollectionParams(unittest.TestCase):

    def test_new_inventory_form(self):
        p = params_from_condition("collection", "new.inventory('iron-ore') >= 5")
        self.assertEqual(p, {"item": "iron-ore", "count": 5})

    def test_bare_inventory_form(self):
        p = params_from_condition("collection", "inventory('iron-ore') >= 5")
        self.assertEqual(p, {"item": "iron-ore", "count": 5})

    def test_copper_ore(self):
        p = params_from_condition("collection", "new.inventory('copper-ore') >= 10")
        self.assertEqual(p, {"item": "copper-ore", "count": 10})

    def test_large_count(self):
        p = params_from_condition("collection", "inventory('coal') >= 200")
        self.assertEqual(p, {"item": "coal", "count": 200})

    def test_double_quoted_item(self):
        p = params_from_condition("collection", 'new.inventory("iron-ore") >= 5')
        self.assertEqual(p, {"item": "iron-ore", "count": 5})

    def test_no_match_returns_empty(self):
        p = params_from_condition("collection", "elapsed_ticks > 10800")
        self.assertEqual(p, {})

    def test_empty_condition_returns_empty(self):
        self.assertEqual(params_from_condition("collection", ""), {})

    def test_none_condition_returns_empty(self):
        self.assertEqual(params_from_condition("collection", None), {})


class TestAcquireParams(unittest.TestCase):
    """acquire uses the same patterns as collection."""

    def test_new_inventory(self):
        p = params_from_condition("acquire", "new.inventory('iron-ore') >= 10")
        self.assertEqual(p, {"item": "iron-ore", "count": 10})

    def test_bare_inventory(self):
        p = params_from_condition("acquire", "inventory('copper-ore') >= 3")
        self.assertEqual(p, {"item": "copper-ore", "count": 3})


class TestCraftingParams(unittest.TestCase):
    """crafting uses the same inventory pattern."""

    def test_new_inventory(self):
        p = params_from_condition("crafting", "new.inventory('iron-gear-wheel') >= 2")
        self.assertEqual(p, {"item": "iron-gear-wheel", "count": 2})

    def test_bare_inventory(self):
        p = params_from_condition("crafting", "inventory('electronic-circuit') >= 10")
        self.assertEqual(p, {"item": "electronic-circuit", "count": 10})


class TestExplorationChunkParams(unittest.TestCase):

    def test_new_charted_chunks(self):
        p = params_from_condition("exploration", "new.charted_chunks >= 5")
        self.assertEqual(p, {"target_chunks": 5})

    def test_bare_charted_chunks(self):
        p = params_from_condition("exploration", "charted_chunks >= 50")
        self.assertEqual(p, {"target_chunks": 50})

    def test_large_target(self):
        p = params_from_condition("exploration", "new.charted_chunks >= 100")
        self.assertEqual(p, {"target_chunks": 100})


class TestExplorationResourceDiscoveryParams(unittest.TestCase):

    def test_resources_of_type_form(self):
        sc = "len(resources_of_type('copper-ore')) >= 1"
        p = params_from_condition("exploration", sc)
        self.assertEqual(p["target_chunks"], 0)
        self.assertEqual(p["success_condition"], sc)
        self.assertIn("copper-ore", p["description"])

    def test_resources_of_type_count_two(self):
        sc = "len(resources_of_type('crude-oil')) >= 2"
        p = params_from_condition("exploration", sc)
        self.assertEqual(p["target_chunks"], 0)
        self.assertIn("2", p["description"])

    def test_new_resource_patches_form(self):
        sc = "new.resource_patches('iron-ore') >= 1"
        p = params_from_condition("exploration", sc)
        self.assertEqual(p["target_chunks"], 0)
        self.assertEqual(p["success_condition"], sc)

    def test_double_quoted_resource(self):
        sc = 'len(resources_of_type("coal")) >= 1'
        p = params_from_condition("exploration", sc)
        self.assertEqual(p["target_chunks"], 0)

    def test_no_match_returns_empty(self):
        p = params_from_condition("exploration", "elapsed_ticks > 18000")
        self.assertEqual(p, {})


class TestResearchParams(unittest.TestCase):

    def test_tech_unlocked(self):
        p = params_from_condition("research", "tech_unlocked('automation')")
        self.assertEqual(p, {"tech": "automation"})

    def test_multi_word_tech(self):
        p = params_from_condition("research", "tech_unlocked('steel-processing')")
        self.assertEqual(p, {"tech": "steel-processing"})

    def test_double_quoted(self):
        p = params_from_condition("research", 'tech_unlocked("logistics")')
        self.assertEqual(p, {"tech": "logistics"})

    def test_no_match(self):
        p = params_from_condition("research", "elapsed_ticks > 7200")
        self.assertEqual(p, {})


class TestProductionParams(unittest.TestCase):

    def test_production_rate_integer(self):
        p = params_from_condition("production", "production_rate('iron-plate') >= 60")
        self.assertEqual(p, {"item": "iron-plate", "rate_per_min": 60.0})

    def test_production_rate_float(self):
        p = params_from_condition("production", "production_rate('copper-plate') >= 30.5")
        self.assertEqual(p["rate_per_min"], 30.5)
        self.assertEqual(p["item"], "copper-plate")

    def test_no_match(self):
        p = params_from_condition("production", "inventory('iron-plate') >= 100")
        self.assertEqual(p, {})


class TestUnknownGoalType(unittest.TestCase):

    def test_unknown_type_returns_empty(self):
        p = params_from_condition("totally_unknown_type", "new.inventory('iron-ore') >= 5")
        self.assertEqual(p, {})

    def test_clear_region_returns_empty(self):
        # clear_region params (bbox) cannot be extracted from condition strings
        p = params_from_condition("clear_region", "elapsed_ticks > 10800")
        self.assertEqual(p, {})

    def test_noop_returns_empty(self):
        p = params_from_condition("noop", "False")
        self.assertEqual(p, {})


class TestGoalTypeIsolation(unittest.TestCase):
    """Wrong goal_type with a matching pattern should not extract incorrectly."""

    def test_inventory_pattern_not_matched_for_exploration(self):
        # exploration handler doesn't care about inventory counts
        p = params_from_condition("exploration", "new.inventory('iron-ore') >= 5")
        self.assertEqual(p, {})

    def test_tech_pattern_not_matched_for_collection(self):
        p = params_from_condition("collection", "tech_unlocked('automation')")
        self.assertEqual(p, {})

    def test_chunks_pattern_not_matched_for_research(self):
        p = params_from_condition("research", "charted_chunks >= 10")
        self.assertEqual(p, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
