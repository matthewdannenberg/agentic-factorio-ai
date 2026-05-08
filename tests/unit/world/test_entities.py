"""
tests/unit/world/test_entities.py

Unit tests for world/entities.py

entities.py is a thin facade over KnowledgeBase. Tests here verify that the
facade delegates correctly and upholds its public contracts (never raises,
correct return types, idempotency). They use a real KnowledgeBase with
query_fn=None (offline/placeholder mode) and a temporary data directory, so
no Factorio connection is required.

Specific field values for vanilla resources (is_fluid, is_infinite, etc.) are
not tested here — those come from Factorio prototype data and belong in
test_knowledge.py. In placeholder mode every newly-discovered record uses safe
defaults (is_fluid=False, is_infinite=False, category=OTHER, etc.), which IS
testable and meaningful here.

Run with:  python tests/unit/world/test_entities.py
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from world.entities import (
    EntityCategory,
    EntityRecord,
    ResourceRecord,
    ResourceRegistry,
    get_entity_metadata,
)
from world.knowledge import KnowledgeBase


def _make_kb(tmp_path: Path) -> KnowledgeBase:
    """Real KnowledgeBase in offline/placeholder mode (query_fn=None)."""
    return KnowledgeBase(data_dir=tmp_path, query_fn=None)


# ---------------------------------------------------------------------------
# ResourceRegistry — delegation and return types
# ---------------------------------------------------------------------------

class TestResourceRegistryReturnTypes(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.kb = _make_kb(Path(self._tmp.name))
        self.reg = ResourceRegistry(self.kb)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ensure_returns_resource_record(self):
        result = self.reg.ensure("iron-ore")
        self.assertIsInstance(result, ResourceRecord)

    def test_ensure_result_has_correct_name(self):
        result = self.reg.ensure("coal")
        self.assertEqual(result.name, "coal")

    def test_get_returns_none_for_unknown(self):
        self.assertIsNone(self.reg.get("does-not-exist"))

    def test_get_returns_record_after_ensure(self):
        self.reg.ensure("coal")
        result = self.reg.get("coal")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ResourceRecord)
        self.assertEqual(result.name, "coal")

    def test_all_known_returns_list_of_strings(self):
        self.reg.ensure("iron-ore")
        known = self.reg.all_known()
        self.assertIsInstance(known, list)
        self.assertTrue(all(isinstance(k, str) for k in known))

    def test_all_known_contains_ensured_resource(self):
        self.reg.ensure("iron-ore")
        self.assertIn("iron-ore", self.reg.all_known())


# ---------------------------------------------------------------------------
# ResourceRegistry — placeholder defaults (offline mode)
# ---------------------------------------------------------------------------

class TestResourceRegistryPlaceholderDefaults(unittest.TestCase):
    """
    In offline mode (query_fn=None) every new resource gets a placeholder.
    These are the only field values we can assert without a live game.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.kb = _make_kb(Path(self._tmp.name))
        self.reg = ResourceRegistry(self.kb)

    def tearDown(self):
        self._tmp.cleanup()

    def test_placeholder_is_fluid_false(self):
        meta = self.reg.ensure("any-new-resource")
        self.assertFalse(meta.is_fluid)

    def test_placeholder_is_infinite_false(self):
        meta = self.reg.ensure("any-new-resource")
        self.assertFalse(meta.is_infinite)

    def test_placeholder_display_name_falls_back_to_name(self):
        meta = self.reg.ensure("se-cryonite")
        self.assertEqual(meta.display_name, "se-cryonite")

    def test_placeholder_is_marked_as_placeholder(self):
        meta = self.reg.ensure("unknown-mod-ore")
        self.assertTrue(meta.is_placeholder)


# ---------------------------------------------------------------------------
# ResourceRegistry — is_fluid / is_infinite helpers
# ---------------------------------------------------------------------------

class TestResourceRegistryHelpers(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.kb = _make_kb(Path(self._tmp.name))
        self.reg = ResourceRegistry(self.kb)

    def tearDown(self):
        self._tmp.cleanup()

    def test_is_fluid_false_for_placeholder(self):
        self.reg.ensure("iron-ore")
        self.assertFalse(self.reg.is_fluid("iron-ore"))

    def test_is_fluid_false_for_never_seen_resource(self):
        # get() returns None — facade must return False, not raise
        self.assertFalse(self.reg.is_fluid("completely-unknown"))

    def test_is_infinite_false_for_placeholder(self):
        self.reg.ensure("iron-ore")
        self.assertFalse(self.reg.is_infinite("iron-ore"))

    def test_is_infinite_false_for_never_seen_resource(self):
        self.assertFalse(self.reg.is_infinite("completely-unknown"))

    def test_is_fluid_reflects_record_value(self):
        # Inject a non-placeholder record directly to test facade passthrough
        from world.knowledge import ResourceRecord as KBResourceRecord
        fluid_record = KBResourceRecord(
            name="crude-oil", is_fluid=True, is_infinite=True,
            display_name="Crude Oil", is_placeholder=False,
        )
        self.kb._resources["crude-oil"] = fluid_record
        self.assertTrue(self.reg.is_fluid("crude-oil"))

    def test_is_infinite_reflects_record_value(self):
        from world.knowledge import ResourceRecord as KBResourceRecord
        fluid_record = KBResourceRecord(
            name="crude-oil", is_fluid=True, is_infinite=True,
            display_name="Crude Oil", is_placeholder=False,
        )
        self.kb._resources["crude-oil"] = fluid_record
        self.assertTrue(self.reg.is_infinite("crude-oil"))


# ---------------------------------------------------------------------------
# ResourceRegistry — safety guarantees
# ---------------------------------------------------------------------------

class TestResourceRegistrySafety(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.kb = _make_kb(Path(self._tmp.name))
        self.reg = ResourceRegistry(self.kb)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ensure_does_not_raise_for_unknown_mod_resource(self):
        try:
            self.reg.ensure("se-cryonite")
        except Exception as exc:
            self.fail(f"ensure() raised unexpectedly: {exc}")

    def test_ensure_does_not_raise_for_empty_string(self):
        try:
            self.reg.ensure("")
        except Exception as exc:
            self.fail(f"ensure('') raised unexpectedly: {exc}")

    def test_ensure_is_idempotent(self):
        r1 = self.reg.ensure("unknown-mod-ore")
        r2 = self.reg.ensure("unknown-mod-ore")
        self.assertIs(r1, r2)

    def test_ensure_registers_new_resource_in_all_known(self):
        self.reg.ensure("modded-crystal")
        self.assertIn("modded-crystal", self.reg.all_known())

    def test_multiple_unknown_resources_registered_independently(self):
        r1 = self.reg.ensure("mod-ore-a")
        r2 = self.reg.ensure("mod-ore-b")
        self.assertIsNot(r1, r2)
        self.assertEqual(r1.name, "mod-ore-a")
        self.assertEqual(r2.name, "mod-ore-b")
        self.assertIn("mod-ore-a", self.reg.all_known())
        self.assertIn("mod-ore-b", self.reg.all_known())

    def test_is_fluid_does_not_raise_for_never_seen_resource(self):
        try:
            self.reg.is_fluid("brand-new-resource")
        except Exception as exc:
            self.fail(f"is_fluid() raised unexpectedly: {exc}")

    def test_is_infinite_does_not_raise_for_never_seen_resource(self):
        try:
            self.reg.is_infinite("brand-new-resource")
        except Exception as exc:
            self.fail(f"is_infinite() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# get_entity_metadata — return types and delegation
# ---------------------------------------------------------------------------

class TestGetEntityMetadata(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.kb = _make_kb(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_entity_record(self):
        result = get_entity_metadata("assembling-machine-2", self.kb)
        self.assertIsInstance(result, EntityRecord)

    def test_result_name_matches_input(self):
        result = get_entity_metadata("electric-furnace", self.kb)
        self.assertEqual(result.name, "electric-furnace")

    def test_category_is_string(self):
        # EntityRecord.category is stored as a string for CSV compatibility;
        # access the enum via .category_enum
        result = get_entity_metadata("assembling-machine-1", self.kb)
        self.assertIsInstance(result.category, str)

    def test_category_enum_is_entity_category(self):
        result = get_entity_metadata("any-entity", self.kb)
        self.assertIsInstance(result.category_enum, EntityCategory)


# ---------------------------------------------------------------------------
# get_entity_metadata — placeholder defaults (offline mode)
# ---------------------------------------------------------------------------

class TestGetEntityMetadataPlaceholderDefaults(unittest.TestCase):
    """
    In offline mode all new entities get placeholders with OTHER category,
    no recipe slot, and zero ingredient/output slots.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.kb = _make_kb(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_placeholder_category_is_other(self):
        result = get_entity_metadata("modded-super-machine", self.kb)
        self.assertEqual(result.category_enum, EntityCategory.OTHER)

    def test_placeholder_has_no_recipe_slot(self):
        result = get_entity_metadata("modded-super-machine", self.kb)
        self.assertFalse(result.has_recipe_slot)

    def test_placeholder_ingredient_slots_zero(self):
        result = get_entity_metadata("modded-super-machine", self.kb)
        self.assertEqual(result.ingredient_slots, 0)

    def test_placeholder_output_slots_zero(self):
        result = get_entity_metadata("modded-super-machine", self.kb)
        self.assertEqual(result.output_slots, 0)

    def test_placeholder_is_marked_as_placeholder(self):
        result = get_entity_metadata("modded-super-machine", self.kb)
        self.assertTrue(result.is_placeholder)

    def test_category_enum_valid_for_placeholder(self):
        result = get_entity_metadata("modded-super-machine", self.kb)
        # category_enum must not raise even for unknown stored strings
        self.assertIsInstance(result.category_enum, EntityCategory)


# ---------------------------------------------------------------------------
# get_entity_metadata — safety guarantees
# ---------------------------------------------------------------------------

class TestGetEntityMetadataSafety(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.kb = _make_kb(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_does_not_raise_for_unknown_entity(self):
        for name in ["", "x", "some-mod-entity-123"]:
            try:
                get_entity_metadata(name, self.kb)
            except Exception as exc:
                self.fail(f"get_entity_metadata({name!r}) raised unexpectedly: {exc}")

    def test_is_idempotent(self):
        r1 = get_entity_metadata("assembling-machine-2", self.kb)
        r2 = get_entity_metadata("assembling-machine-2", self.kb)
        self.assertIs(r1, r2)

    def test_prototype_category_passthrough(self):
        # Inject a fully-populated EntityRecord to verify the facade returns
        # whatever the KB gives it, including non-OTHER categories.
        from world.knowledge import EntityRecord as KBEntityRecord
        assembly_record = KBEntityRecord(
            name="assembling-machine-2",
            proto_type="assembling-machine",
            category=EntityCategory.ASSEMBLY.value,
            tile_width=3,
            tile_height=3,
            has_recipe_slot=True,
            ingredient_slots=4,
            output_slots=1,
            is_placeholder=False,
        )
        self.kb._entities["assembling-machine-2"] = assembly_record
        result = get_entity_metadata("assembling-machine-2", self.kb)
        self.assertEqual(result.category_enum, EntityCategory.ASSEMBLY)
        self.assertTrue(result.has_recipe_slot)
        self.assertEqual(result.ingredient_slots, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)