"""
tests/unit/world/test_tech_tree.py

Unit tests for world/tech_tree.py — TechTree backed by KnowledgeBase.

Run with:  python -m unittest tests.unit.world.test_tech_tree
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from world.knowledge import KnowledgeBase, TechRecord
from world.state import ResearchState
from world.tech_tree import TechTree


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_TECH_DEFS = {
    "automation": {
        "name": "automation",
        "prerequisites": [],
        "effects": [
            {"type": "unlock-recipe", "recipe": "assembling-machine-1"},
            {"type": "unlock-recipe", "recipe": "long-handed-inserter"},
        ],
        "researched": False, "enabled": True,
    },
    "electronics": {
        "name": "electronics",
        "prerequisites": [],
        "effects": [
            {"type": "unlock-recipe", "recipe": "electronic-circuit"},
            {"type": "unlock-recipe", "recipe": "small-electric-pole"},
        ],
        "researched": False, "enabled": True,
    },
    "logistics": {
        "name": "logistics",
        "prerequisites": ["automation"],
        "effects": [
            {"type": "unlock-recipe", "recipe": "transport-belt"},
            {"type": "unlock-recipe", "recipe": "underground-belt"},
            {"type": "unlock-recipe", "recipe": "splitter"},
        ],
        "researched": False, "enabled": True,
    },
    "automation-2": {
        "name": "automation-2",
        "prerequisites": ["automation", "electronics"],
        "effects": [
            {"type": "unlock-recipe", "recipe": "assembling-machine-2"},
        ],
        "researched": False, "enabled": True,
    },
    "logistics-2": {
        "name": "logistics-2",
        "prerequisites": ["logistics", "automation-2"],
        "effects": [
            {"type": "unlock-recipe", "recipe": "fast-transport-belt"},
            {"type": "unlock-recipe", "recipe": "fast-inserter"},
        ],
        "researched": False, "enabled": True,
    },
    "steel-processing": {
        "name": "steel-processing",
        "prerequisites": ["automation"],
        "effects": [{"type": "unlock-recipe", "recipe": "steel-plate"}],
        "researched": False, "enabled": True,
    },
    "fluid-handling": {
        "name": "fluid-handling",
        "prerequisites": ["steel-processing"],
        "effects": [{"type": "unlock-recipe", "recipe": "pipe"}],
        "researched": False, "enabled": True,
    },
    "advanced-material-processing": {
        "name": "advanced-material-processing",
        "prerequisites": ["steel-processing", "fluid-handling"],
        "effects": [{"type": "unlock-recipe", "recipe": "oil-refinery"}],
        "researched": False, "enabled": True,
    },
    "space-science": {
        "name": "space-science",
        "prerequisites": [],
        "effects": [{"type": "unlock-recipe", "recipe": "space-science-pack"}],
        "researched": False, "enabled": True,
        "requires_dlc": True,
    },
}


def _make_query_fn(*known_techs: str):
    def qfn(expr: str) -> str:
        for name in known_techs:
            if f'"{name}"' in expr:
                return json.dumps(_TECH_DEFS[name])
        return json.dumps({"ok": False, "reason": "unknown_technology"})
    return qfn


def _kb_with(*tech_names: str, tmp_dir: str = None) -> tuple[KnowledgeBase, str]:
    d = tmp_dir or tempfile.mkdtemp()
    qfn = _make_query_fn(*tech_names)
    kb = KnowledgeBase(data_dir=Path(d), query_fn=qfn)
    for name in tech_names:
        kb.ensure_tech(name)
    # Close the DB connection after loading — tech_tree tests only use the
    # in-memory cache (no SQL query methods), so this is safe cross-platform.
    kb.close()
    return kb, d


def _tree_with(*tech_names: str) -> tuple[TechTree, KnowledgeBase, str]:
    kb, d = _kb_with(*tech_names)
    return TechTree(kb), kb, d


def _research(*unlocked: str, queued=(), in_progress=None) -> ResearchState:
    return ResearchState(
        unlocked=list(unlocked),
        queued=list(queued),
        in_progress=in_progress,
    )


# ---------------------------------------------------------------------------
# Baseline / empty-KB behaviour
# ---------------------------------------------------------------------------

class TestTechTreeEmptyKB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.kb = KnowledgeBase(data_dir=Path(self.tmp), query_fn=None)
        self.tree = TechTree(self.kb)

    def test_known_returns_false_for_unlearned_tech(self):
        self.assertFalse(self.tree.known("automation"))

    def test_known_returns_true_after_ensure(self):
        self.tree.ensure("automation")
        self.assertTrue(self.tree.known("automation"))

    def test_ensure_returns_placeholder_without_query_fn(self):
        rec = self.tree.ensure("automation")
        self.assertTrue(rec.is_placeholder)

    def test_prerequisites_empty_for_placeholder(self):
        self.tree.ensure("logistics")
        self.assertEqual(self.tree.prerequisites("logistics"), [])

    def test_prerequisites_empty_for_unknown(self):
        self.assertEqual(self.tree.prerequisites("automation"), [])

    def test_all_prerequisites_empty_for_placeholder(self):
        self.tree.ensure("logistics")
        self.assertEqual(self.tree.all_prerequisites("logistics"), set())

    def test_unlocks_entity_empty_for_placeholder(self):
        self.tree.ensure("automation")
        self.assertEqual(self.tree.unlocks_entity("automation"), [])

    def test_unlocks_recipe_empty_for_placeholder(self):
        self.tree.ensure("automation")
        self.assertEqual(self.tree.unlocks_recipe("automation"), [])

    def test_is_reachable_false_for_placeholder(self):
        self.tree.ensure("logistics")
        self.assertFalse(self.tree.is_reachable("logistics", _research("automation")))

    def test_is_unlocked_checks_research_state_not_kb(self):
        research = _research("automation", "logistics")
        self.assertTrue(self.tree.is_unlocked("automation", research))
        self.assertFalse(self.tree.is_unlocked("steel-processing", research))

    def test_path_to_raises_for_tech_not_in_kb(self):
        with self.assertRaises(ValueError):
            self.tree.path_to("ghost-tech", _research())

    def test_next_researchable_empty_when_all_placeholders(self):
        self.tree.ensure("logistics")
        self.tree.ensure("automation")
        self.assertEqual(self.tree.next_researchable(_research()), [])


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

class TestPrerequisites(unittest.TestCase):
    def test_automation_has_no_prerequisites(self):
        tree, kb, _ = _tree_with("automation")
        self.assertEqual(tree.prerequisites("automation"), [])

    def test_logistics_has_automation_as_prerequisite(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        self.assertEqual(tree.prerequisites("logistics"), ["automation"])

    def test_automation_2_has_two_prerequisites(self):
        tree, kb, _ = _tree_with("automation", "electronics", "automation-2")
        prereqs = tree.prerequisites("automation-2")
        self.assertIn("automation", prereqs)
        self.assertIn("electronics", prereqs)
        self.assertEqual(len(prereqs), 2)

    def test_prerequisites_returns_copy_not_reference(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        prereqs = tree.prerequisites("logistics")
        prereqs.clear()
        self.assertEqual(tree.prerequisites("logistics"), ["automation"])

    def test_prerequisites_for_unknown_tech_returns_empty(self):
        tree, kb, _ = _tree_with("automation")
        self.assertEqual(tree.prerequisites("completely-unknown-tech"), [])

    def test_all_prerequisites_automation_empty(self):
        tree, kb, _ = _tree_with("automation")
        self.assertEqual(tree.all_prerequisites("automation"), set())

    def test_all_prerequisites_logistics_contains_automation(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        self.assertIn("automation", tree.all_prerequisites("logistics"))

    def test_all_prerequisites_logistics_2_transitive(self):
        tree, kb, _ = _tree_with(
            "automation", "electronics", "logistics", "automation-2", "logistics-2"
        )
        all_p = tree.all_prerequisites("logistics-2")
        self.assertIn("logistics", all_p)
        self.assertIn("automation-2", all_p)
        self.assertIn("automation", all_p)
        self.assertIn("electronics", all_p)

    def test_all_prerequisites_does_not_include_tech_itself(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        self.assertNotIn("logistics", tree.all_prerequisites("logistics"))

    def test_all_prerequisites_unknown_returns_empty_set(self):
        tree, kb, _ = _tree_with("automation")
        self.assertEqual(tree.all_prerequisites("nonexistent-tech"), set())

    def test_all_prerequisites_deep_chain(self):
        tree, kb, _ = _tree_with(
            "automation", "steel-processing", "fluid-handling",
            "advanced-material-processing"
        )
        all_p = tree.all_prerequisites("advanced-material-processing")
        self.assertIn("automation", all_p)
        self.assertIn("steel-processing", all_p)
        self.assertIn("fluid-handling", all_p)


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

class TestReachability(unittest.TestCase):
    def test_no_prereq_tech_always_reachable(self):
        tree, kb, _ = _tree_with("automation")
        self.assertTrue(tree.is_reachable("automation", _research()))

    def test_reachable_when_prereqs_met(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        self.assertTrue(tree.is_reachable("logistics", _research("automation")))

    def test_not_reachable_when_prereq_missing(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        self.assertFalse(tree.is_reachable("logistics", _research()))

    def test_reachable_multi_prereq_all_met(self):
        tree, kb, _ = _tree_with("automation", "electronics", "automation-2")
        self.assertTrue(tree.is_reachable("automation-2", _research("automation", "electronics")))

    def test_not_reachable_multi_prereq_one_missing(self):
        tree, kb, _ = _tree_with("automation", "electronics", "automation-2")
        self.assertFalse(tree.is_reachable("automation-2", _research("automation")))

    def test_is_unlocked_independent_of_kb(self):
        tree, kb, _ = _tree_with("automation")
        self.assertTrue(tree.is_unlocked("automation", _research("automation")))
        self.assertFalse(tree.is_unlocked("automation", _research()))


# ---------------------------------------------------------------------------
# Unlock queries
# ---------------------------------------------------------------------------

class TestUnlockQueries(unittest.TestCase):
    def test_unlocks_recipe_automation(self):
        tree, kb, _ = _tree_with("automation")
        recipes = tree.unlocks_recipe("automation")
        self.assertIn("assembling-machine-1", recipes)
        self.assertIn("long-handed-inserter", recipes)

    def test_unlocks_recipe_logistics(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        recipes = tree.unlocks_recipe("logistics")
        self.assertIn("transport-belt", recipes)
        self.assertIn("underground-belt", recipes)
        self.assertIn("splitter", recipes)

    def test_unlocks_recipe_empty_for_unknown(self):
        tree, kb, _ = _tree_with("automation")
        self.assertEqual(tree.unlocks_recipe("ghost-tech"), [])

    def test_unlocks_entity_empty_for_unknown(self):
        tree, kb, _ = _tree_with("automation")
        self.assertEqual(tree.unlocks_entity("ghost-tech"), [])

    def test_unlocks_recipe_returns_copy(self):
        tree, kb, _ = _tree_with("automation")
        recipes = tree.unlocks_recipe("automation")
        original_len = len(recipes)
        recipes.clear()
        self.assertEqual(len(tree.unlocks_recipe("automation")), original_len)


# ---------------------------------------------------------------------------
# Path planning
# ---------------------------------------------------------------------------

class TestPathTo(unittest.TestCase):
    def test_path_to_already_unlocked_returns_empty(self):
        tree, kb, _ = _tree_with("automation")
        self.assertEqual(tree.path_to("automation", _research("automation")), [])

    def test_path_to_no_prereq_tech(self):
        tree, kb, _ = _tree_with("automation")
        path = tree.path_to("automation", _research())
        self.assertEqual(path, ["automation"])

    def test_path_to_single_prereq_ordered(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        path = tree.path_to("logistics", _research())
        self.assertIn("automation", path)
        self.assertIn("logistics", path)
        self.assertLess(path.index("automation"), path.index("logistics"))

    def test_path_to_skips_already_unlocked_prereqs(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        path = tree.path_to("logistics", _research("automation"))
        self.assertIn("logistics", path)
        self.assertNotIn("automation", path)

    def test_path_to_deep_chain_ordered(self):
        tree, kb, _ = _tree_with(
            "automation", "steel-processing", "fluid-handling",
            "advanced-material-processing"
        )
        path = tree.path_to("advanced-material-processing", _research())
        self.assertIn("automation", path)
        self.assertIn("steel-processing", path)
        self.assertIn("fluid-handling", path)
        self.assertIn("advanced-material-processing", path)
        self.assertLess(path.index("automation"), path.index("steel-processing"))
        self.assertLess(path.index("steel-processing"), path.index("fluid-handling"))
        self.assertLess(path.index("fluid-handling"),
                        path.index("advanced-material-processing"))

    def test_path_to_multi_prereq_both_before_target(self):
        tree, kb, _ = _tree_with("automation", "electronics", "automation-2")
        path = tree.path_to("automation-2", _research())
        self.assertIn("automation", path)
        self.assertIn("electronics", path)
        self.assertIn("automation-2", path)
        self.assertLess(path.index("automation"), path.index("automation-2"))
        self.assertLess(path.index("electronics"), path.index("automation-2"))

    def test_path_to_raises_for_tech_not_in_kb(self):
        tree, kb, _ = _tree_with("automation")
        with self.assertRaises(ValueError):
            tree.path_to("completely-unknown-tech", _research())

    def test_path_to_placeholder_tech_is_included(self):
        tmp = tempfile.mkdtemp()
        kb = KnowledgeBase(data_dir=Path(tmp), query_fn=None)
        kb.ensure_tech("mystery-tech")
        tree = TechTree(kb)
        path = tree.path_to("mystery-tech", _research())
        self.assertIn("mystery-tech", path)


# ---------------------------------------------------------------------------
# Next researchable
# ---------------------------------------------------------------------------

class TestNextResearchable(unittest.TestCase):
    def test_no_research_automation_available(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        self.assertIn("automation", tree.next_researchable(_research()))

    def test_no_research_logistics_not_available(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        self.assertNotIn("logistics", tree.next_researchable(_research()))

    def test_after_automation_logistics_available(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        self.assertIn("logistics", tree.next_researchable(_research("automation")))

    def test_unlocked_techs_excluded(self):
        tree, kb, _ = _tree_with("automation", "logistics")
        self.assertNotIn("automation", tree.next_researchable(_research("automation")))

    def test_placeholders_excluded(self):
        tmp = tempfile.mkdtemp()
        kb = KnowledgeBase(data_dir=Path(tmp), query_fn=None)
        kb.ensure_tech("automation")
        tree = TechTree(kb)
        self.assertNotIn("automation", tree.next_researchable(_research()))

    def test_sorted_by_prerequisite_count_ascending(self):
        tree, kb, _ = _tree_with(
            "automation", "electronics", "logistics", "automation-2"
        )
        result = tree.next_researchable(_research())
        counts = [len(tree.all_prerequisites(t)) for t in result]
        self.assertEqual(counts, sorted(counts))

    def test_multi_prereq_tech_requires_all_met(self):
        tree, kb, _ = _tree_with("automation", "electronics", "automation-2")
        self.assertNotIn("automation-2", tree.next_researchable(_research("automation")))
        self.assertIn("automation-2",
                      tree.next_researchable(_research("automation", "electronics")))

    def test_returns_list(self):
        tree, kb, _ = _tree_with("automation")
        self.assertIsInstance(tree.next_researchable(_research()), list)


# ---------------------------------------------------------------------------
# absorb_research_state
# ---------------------------------------------------------------------------

class TestAbsorbResearchState(unittest.TestCase):
    def test_absorb_learns_unlocked_techs(self):
        tmp = tempfile.mkdtemp()
        kb = KnowledgeBase(data_dir=Path(tmp),
                           query_fn=_make_query_fn("automation", "electronics"))
        tree = TechTree(kb)
        self.assertFalse(tree.known("automation"))
        tree.absorb_research_state(_research("automation", "electronics"))
        self.assertTrue(tree.known("automation"))
        self.assertTrue(tree.known("electronics"))

    def test_absorb_learns_queued_techs(self):
        tmp = tempfile.mkdtemp()
        kb = KnowledgeBase(data_dir=Path(tmp),
                           query_fn=_make_query_fn("automation", "logistics"))
        tree = TechTree(kb)
        tree.absorb_research_state(_research("automation", queued=["logistics"]))
        self.assertTrue(tree.known("logistics"))

    def test_absorb_learns_in_progress_tech(self):
        tmp = tempfile.mkdtemp()
        kb = KnowledgeBase(data_dir=Path(tmp),
                           query_fn=_make_query_fn("automation", "logistics"))
        tree = TechTree(kb)
        tree.absorb_research_state(
            ResearchState(unlocked=["automation"], queued=[], in_progress="logistics")
        )
        self.assertTrue(tree.known("logistics"))

    def test_absorb_safe_with_empty_research_state(self):
        tmp = tempfile.mkdtemp()
        kb = KnowledgeBase(data_dir=Path(tmp), query_fn=None)
        tree = TechTree(kb)
        tree.absorb_research_state(_research())  # must not raise

    def test_absorb_idempotent(self):
        tmp = tempfile.mkdtemp()
        call_count = [0]
        def counting_qfn(expr):
            call_count[0] += 1
            return json.dumps(_TECH_DEFS["automation"])
        kb = KnowledgeBase(data_dir=Path(tmp), query_fn=counting_qfn)
        tree = TechTree(kb)
        tree.absorb_research_state(_research("automation"))
        first = call_count[0]
        tree.absorb_research_state(_research("automation"))
        self.assertEqual(call_count[0], first)


# ---------------------------------------------------------------------------
# Persistence across KB restarts
# ---------------------------------------------------------------------------

class TestTechTreePersistence(unittest.TestCase):
    def test_learned_tech_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=_make_query_fn("automation", "logistics")) as kb1:
                TechTree(kb1).absorb_research_state(
                    _research("automation", queued=["logistics"])
                )
            with KnowledgeBase(data_dir=Path(tmp), query_fn=None) as kb2:
                tree2 = TechTree(kb2)
                self.assertTrue(tree2.known("automation"))
                self.assertTrue(tree2.known("logistics"))

    def test_prerequisites_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=_make_query_fn("automation", "logistics")) as kb1:
                kb1.ensure_tech("automation")
                kb1.ensure_tech("logistics")
            with KnowledgeBase(data_dir=Path(tmp), query_fn=None) as kb2:
                tree2 = TechTree(kb2)
                self.assertEqual(tree2.prerequisites("logistics"), ["automation"])

    def test_unlock_recipes_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=_make_query_fn("automation")) as kb1:
                kb1.ensure_tech("automation")
            with KnowledgeBase(data_dir=Path(tmp), query_fn=None) as kb2:
                tree2 = TechTree(kb2)
                self.assertIn("assembling-machine-1",
                              tree2.unlocks_recipe("automation"))

    def test_is_reachable_works_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=_make_query_fn("automation", "logistics")) as kb1:
                kb1.ensure_tech("automation")
                kb1.ensure_tech("logistics")
            with KnowledgeBase(data_dir=Path(tmp), query_fn=None) as kb2:
                tree2 = TechTree(kb2)
                self.assertTrue(tree2.is_reachable("logistics", _research("automation")))

    def test_next_researchable_works_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with KnowledgeBase(data_dir=Path(tmp),
                               query_fn=_make_query_fn("automation", "logistics")) as kb1:
                kb1.ensure_tech("automation")
                kb1.ensure_tech("logistics")
            with KnowledgeBase(data_dir=Path(tmp), query_fn=None) as kb2:
                tree2 = TechTree(kb2)
                self.assertIn("automation", tree2.next_researchable(_research()))


# ---------------------------------------------------------------------------
# Mod compatibility
# ---------------------------------------------------------------------------

class TestModCompatibility(unittest.TestCase):
    def test_mod_tech_learned_via_query_fn(self):
        tmp = tempfile.mkdtemp()
        def qfn(expr):
            return json.dumps({
                "name": "se-interstellar-travel",
                "prerequisites": ["space-science"],
                "effects": [{"type": "unlock-recipe",
                             "recipe": "se-interstellar-rocket"}],
                "researched": False, "enabled": True,
            })
        kb = KnowledgeBase(data_dir=Path(tmp), query_fn=qfn)
        tree = TechTree(kb)
        rec = tree.ensure("se-interstellar-travel")
        self.assertFalse(rec.is_placeholder)
        self.assertIn("se-interstellar-rocket",
                      tree.unlocks_recipe("se-interstellar-travel"))

    def test_error_response_from_query_yields_placeholder(self):
        tmp = tempfile.mkdtemp()
        def qfn(expr):
            return json.dumps({"ok": False, "reason": "unknown_technology"})
        kb = KnowledgeBase(data_dir=Path(tmp), query_fn=qfn)
        tree = TechTree(kb)
        rec = tree.ensure("ghost-tech")
        self.assertTrue(rec.is_placeholder)
        self.assertEqual(tree.prerequisites("ghost-tech"), [])
        self.assertEqual(tree.unlocks_recipe("ghost-tech"), [])

    def test_requires_dlc_flag_preserved(self):
        tmp = tempfile.mkdtemp()
        kb = KnowledgeBase(data_dir=Path(tmp),
                           query_fn=_make_query_fn("space-science"))
        tree = TechTree(kb)
        rec = tree.ensure("space-science")
        self.assertTrue(rec.requires_dlc)

    def test_mixed_vanilla_and_mod_techs_transitive_prereqs(self):
        tmp = tempfile.mkdtemp()
        mod_def = {
            "name": "mod-custom-assembler",
            "prerequisites": ["automation-2"],
            "effects": [{"type": "unlock-recipe", "recipe": "mod-assembler-3"}],
            "researched": False, "enabled": True,
        }
        def qfn(expr):
            if "mod-custom-assembler" in expr:
                return json.dumps(mod_def)
            for name, data in _TECH_DEFS.items():
                if f'"{name}"' in expr:
                    return json.dumps(data)
            return json.dumps({"ok": False, "reason": "unknown"})

        kb = KnowledgeBase(data_dir=Path(tmp), query_fn=qfn)
        for t in ("automation", "electronics", "automation-2", "mod-custom-assembler"):
            kb.ensure_tech(t)
        tree = TechTree(kb)

        all_p = tree.all_prerequisites("mod-custom-assembler")
        self.assertIn("automation-2", all_p)
        self.assertIn("automation", all_p)
        self.assertIn("electronics", all_p)


if __name__ == "__main__":
    unittest.main(verbosity=2)