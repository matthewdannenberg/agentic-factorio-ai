"""
tests/unit/planning/test_condition_namespace.py

Unit tests for planning/condition_namespace.py.

Covers:
  - build_core_namespace: presence and correctness of every entry
  - _DeltaView: scalar __getattr__ delegation, explicit methods,
    tick property, clamping, same-state (no delta) behaviour
  - safe_builtins: blocked names absent, expected names present
  - Eval integration: conditions evaluated against the namespace

Run with:
    pytest tests/unit/planning/test_condition_namespace.py -v
"""

from __future__ import annotations

import unittest

from planning import build_core_namespace, safe_builtins, BLOCKED_NAMES
from planning.evaluation.condition_namespace import _DeltaView
from world import WorldQuery
from world import (
    ExplorationState,
    Inventory,
    InventorySlot,
    PlayerState,
    Position,
    ResourcePatch,
)
from world.observable.state import WorldState  # white-box: constructs WorldState directly
from tests.fixtures import make_world_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wq(
    charted_chunks: int = 0,
    iron_ore: int = 0,
    copper_ore: int = 0,
    resource_patches: list = None,
) -> WorldQuery:
    ws = WorldState(
        player=PlayerState(
            position=Position(0.0, 0.0),
            inventory=Inventory(slots=[
                *([] if iron_ore == 0 else [InventorySlot("iron-ore", iron_ore)]),
                *([] if copper_ore == 0 else [InventorySlot("copper-ore", copper_ore)]),
            ]),
            exploration=ExplorationState(charted_chunks=charted_chunks),
        ),
        resource_map=resource_patches or [],
    )
    ws._rebuild_entity_indices()
    return WorldQuery(ws)


def _ns(wq=None, tick=100, start_tick=0, start_wq=None):
    if wq is None:
        wq = make_world_query()
    return build_core_namespace(wq, tick, start_tick, start_wq)


def _eval(condition, wq=None, tick=100, start_tick=0, start_wq=None):
    ns = _ns(wq, tick, start_tick, start_wq)
    ns["__builtins__"] = safe_builtins()
    return bool(eval(condition, ns))


# ===========================================================================
# TestBuildCoreNamespaceEntries
# ===========================================================================

class TestBuildCoreNamespaceEntries(unittest.TestCase):

    def setUp(self):
        self.ns = _ns()

    def test_wq_present(self):
        self.assertIn("wq", self.ns)

    def test_state_present(self):
        self.assertIn("state", self.ns)

    def test_tick_present_and_correct(self):
        self.assertEqual(_ns(tick=500)["tick"], 500)

    def test_elapsed_ticks_present_and_correct(self):
        self.assertEqual(_ns(tick=500, start_tick=100)["elapsed_ticks"], 400)

    def test_elapsed_ticks_clamped_to_zero(self):
        self.assertEqual(_ns(tick=50, start_tick=100)["elapsed_ticks"], 0)

    def test_new_present_and_is_deltaview(self):
        self.assertIsInstance(self.ns["new"], _DeltaView)

    def test_inventory_callable(self):
        self.assertTrue(callable(self.ns["inventory"]))

    def test_charted_chunks_present(self):
        self.assertIn("charted_chunks", self.ns)

    def test_charted_tiles_present(self):
        self.assertIn("charted_tiles", self.ns)

    def test_charted_area_km2_present(self):
        self.assertIn("charted_area_km2", self.ns)

    def test_research_present(self):
        self.assertIn("research", self.ns)

    def test_tech_unlocked_callable(self):
        self.assertTrue(callable(self.ns["tech_unlocked"]))

    def test_resources_of_type_callable(self):
        self.assertTrue(callable(self.ns["resources_of_type"]))

    def test_entities_callable(self):
        self.assertTrue(callable(self.ns["entities"]))

    def test_safe_builtins_True_False(self):
        self.assertIs(self.ns["True"], True)
        self.assertIs(self.ns["False"], False)

    def test_safe_builtins_functions(self):
        for name in ("len", "any", "all", "sum", "min", "max", "abs",
                     "int", "float", "str", "bool"):
            self.assertIn(name, self.ns, f"Missing builtin: {name}")


# ===========================================================================
# TestBuildCoreNamespaceValues
# ===========================================================================

class TestBuildCoreNamespaceValues(unittest.TestCase):

    def test_inventory_returns_correct_count(self):
        wq = _make_wq(iron_ore=25)
        self.assertEqual(_ns(wq)["inventory"]("iron-ore"), 25)

    def test_inventory_returns_zero_for_missing(self):
        self.assertEqual(_ns(_make_wq())["inventory"]("coal"), 0)

    def test_charted_chunks_value(self):
        wq = _make_wq(charted_chunks=42)
        self.assertEqual(_ns(wq)["charted_chunks"], 42)

    def test_charted_tiles_value(self):
        wq = _make_wq(charted_chunks=10)
        self.assertEqual(_ns(wq)["charted_tiles"], 10 * 1024)

    def test_resources_of_type_count(self):
        patches = [
            ResourcePatch("iron-ore", Position(10, 10), 5000, 20),
            ResourcePatch("iron-ore", Position(20, 20), 3000, 15),
            ResourcePatch("coal", Position(30, 30), 2000, 10),
        ]
        wq = _make_wq(resource_patches=patches)
        ns = _ns(wq)
        self.assertEqual(len(ns["resources_of_type"]("iron-ore")), 2)
        self.assertEqual(len(ns["resources_of_type"]("coal")), 1)


# ===========================================================================
# TestDeltaView
# ===========================================================================

class TestDeltaViewGetattr(unittest.TestCase):

    def test_charted_chunks_delta(self):
        start_wq = _make_wq(charted_chunks=10)
        wq       = _make_wq(charted_chunks=15)
        dv = _DeltaView(wq, start_wq, 0)
        self.assertEqual(dv.charted_chunks, 5)

    def test_charted_tiles_auto_via_getattr(self):
        start_wq = _make_wq(charted_chunks=10)
        wq       = _make_wq(charted_chunks=15)
        dv = _DeltaView(wq, start_wq, 0)
        self.assertEqual(dv.charted_tiles, 5 * 1024)

    def test_clamped_to_zero(self):
        start_wq = _make_wq(charted_chunks=10)
        wq       = _make_wq(charted_chunks=5)
        dv = _DeltaView(wq, start_wq, 0)
        self.assertEqual(dv.charted_chunks, 0)

    def test_no_delta_when_same_state(self):
        wq = _make_wq(charted_chunks=100)
        dv = _DeltaView(wq, wq, 0)
        self.assertEqual(dv.charted_chunks, 0)

    def test_unknown_property_raises_attribute_error(self):
        wq = make_world_query()
        dv = _DeltaView(wq, wq, 0)
        with self.assertRaises(AttributeError):
            _ = dv.this_property_does_not_exist

    def test_callable_property_raises_attribute_error(self):
        wq = make_world_query()
        dv = _DeltaView(wq, wq, 0)
        with self.assertRaises(AttributeError):
            _ = dv.inventory_count


class TestDeltaViewExplicitMethods(unittest.TestCase):

    def test_inventory_delta(self):
        start_wq = _make_wq(iron_ore=5)
        wq       = _make_wq(iron_ore=15)
        dv = _DeltaView(wq, start_wq, 0)
        self.assertEqual(dv.inventory("iron-ore"), 10)

    def test_inventory_no_gain(self):
        wq = _make_wq(iron_ore=5)
        dv = _DeltaView(wq, wq, 0)
        self.assertEqual(dv.inventory("iron-ore"), 0)

    def test_inventory_clamped_to_zero(self):
        start_wq = _make_wq(iron_ore=10)
        wq       = _make_wq(iron_ore=3)
        dv = _DeltaView(wq, start_wq, 0)
        self.assertEqual(dv.inventory("iron-ore"), 0)

    def test_inventory_item_not_at_start(self):
        """Item not in inventory at goal start — baseline is 0."""
        start_wq = _make_wq(iron_ore=0)
        wq       = _make_wq(iron_ore=10)
        dv = _DeltaView(wq, start_wq, 0)
        self.assertEqual(dv.inventory("iron-ore"), 10)

    def test_resource_patches_delta(self):
        start_patches = [ResourcePatch("iron-ore", Position(10, 10), 5000, 20)]
        end_patches   = [
            ResourcePatch("iron-ore", Position(10, 10), 5000, 20),
            ResourcePatch("iron-ore", Position(20, 20), 3000, 15),
        ]
        dv = _DeltaView(_make_wq(resource_patches=end_patches),
                        _make_wq(resource_patches=start_patches), 0)
        self.assertEqual(dv.resource_patches("iron-ore"), 1)

    def test_resource_patches_no_new(self):
        patches = [ResourcePatch("iron-ore", Position(10, 10), 5000, 20)]
        wq = _make_wq(resource_patches=patches)
        dv = _DeltaView(wq, wq, 0)
        self.assertEqual(dv.resource_patches("iron-ore"), 0)


class TestDeltaViewTick(unittest.TestCase):

    def test_tick_property(self):
        wq = make_world_query()
        dv = _DeltaView(wq, wq, 400)
        self.assertEqual(dv.tick, 400)

    def test_tick_matches_elapsed_ticks_in_namespace(self):
        wq = make_world_query()
        ns = _ns(wq, tick=500, start_tick=100)
        self.assertEqual(ns["new"].tick, ns["elapsed_ticks"])

    def test_tick_zero_at_start(self):
        wq = make_world_query()
        dv = _DeltaView(wq, wq, 0)
        self.assertEqual(dv.tick, 0)


# ===========================================================================
# TestSafeBuiltins
# ===========================================================================

class TestSafeBuiltins(unittest.TestCase):

    def setUp(self):
        self.builtins = safe_builtins()

    def test_blocked_names_absent(self):
        
        for name in BLOCKED_NAMES:
            self.assertNotIn(name, self.builtins, f"Blocked: {name}")

    def test_len_present(self):
        self.assertIn("len", self.builtins)

    def test_returns_dict(self):
        self.assertIsInstance(self.builtins, dict)


# ===========================================================================
# TestEvalIntegration
# ===========================================================================

class TestEvalIntegration(unittest.TestCase):

    def test_inventory_condition(self):
        wq = _make_wq(iron_ore=25)
        self.assertTrue(_eval("inventory('iron-ore') >= 25", wq))
        self.assertFalse(_eval("inventory('iron-ore') >= 26", wq))

    def test_charted_chunks_condition(self):
        wq = _make_wq(charted_chunks=10)
        self.assertTrue(_eval("charted_chunks >= 10", wq))

    def test_elapsed_ticks_condition(self):
        self.assertTrue(_eval("elapsed_ticks >= 400", tick=500, start_tick=100))
        self.assertFalse(_eval("elapsed_ticks >= 401", tick=500, start_tick=100))

    def test_new_charted_chunks_condition(self):
        start_wq = _make_wq(charted_chunks=10)
        wq       = _make_wq(charted_chunks=15)
        self.assertTrue(_eval("new.charted_chunks >= 5", wq, start_wq=start_wq))
        self.assertFalse(_eval("new.charted_chunks >= 6", wq, start_wq=start_wq))

    def test_new_inventory_condition(self):
        start_wq = _make_wq(iron_ore=5)
        wq       = _make_wq(iron_ore=15)
        self.assertTrue(_eval("new.inventory('iron-ore') >= 10", wq, start_wq=start_wq))

    def test_new_inventory_not_at_start_baseline_zero(self):
        """Item absent at goal start — all collected items count as new."""
        start_wq = _make_wq(iron_ore=0)
        wq       = _make_wq(iron_ore=5)
        self.assertTrue(_eval("new.inventory('iron-ore') >= 5", wq, start_wq=start_wq))

    def test_new_tick_condition(self):
        self.assertTrue(_eval("new.tick >= 400", tick=500, start_tick=100))

    def test_new_tick_equals_elapsed_ticks(self):
        self.assertTrue(_eval("new.tick == elapsed_ticks", tick=500, start_tick=100))

    def test_compound_condition(self):
        wq = _make_wq(charted_chunks=15, iron_ore=25)
        self.assertTrue(_eval("charted_chunks >= 10 and inventory('iron-ore') >= 20", wq))

    def test_len_builtin_available(self):
        patches = [ResourcePatch("iron-ore", Position(10, 10), 5000, 20)]
        wq = _make_wq(resource_patches=patches)
        self.assertTrue(_eval("len(resources_of_type('iron-ore')) >= 1", wq))

    def test_blocked_name_raises(self):
        with self.assertRaises(Exception):
            _eval("__import__('os')")

    def test_no_delta_when_no_start_wq(self):
        """With no start_wq, all deltas are 0 (start_wq defaults to wq)."""
        wq = _make_wq(charted_chunks=100, iron_ore=50)
        self.assertTrue(_eval("new.charted_chunks == 0", wq))
        self.assertTrue(_eval("new.inventory('iron-ore') == 0", wq))



# ===========================================================================
# TestNaturalObjectsInBbox
# ===========================================================================

class TestNaturalObjectsInBboxMethod(unittest.TestCase):
    def _obj(self, x, y):
        from unittest.mock import MagicMock
        o = MagicMock()
        o.position = Position(x=float(x), y=float(y))
        return o

    def _wq(self, objects):
        ws = WorldState()
        ws.natural_objects = objects
        ws._rebuild_entity_indices()
        return WorldQuery(ws)

    def test_empty(self):
        self.assertEqual(self._wq([]).natural_objects_in_bbox(-16,-16,16,16), [])

    def test_inside_included(self):
        self.assertEqual(len(self._wq([self._obj(0,0)]).natural_objects_in_bbox(-16,-16,16,16)), 1)

    def test_outside_excluded(self):
        self.assertEqual(self._wq([self._obj(100,100)]).natural_objects_in_bbox(-16,-16,16,16), [])

    def test_boundary_inclusive(self):
        for x, y in [(-16,-16),(16,16)]:
            self.assertEqual(len(self._wq([self._obj(x,y)]).natural_objects_in_bbox(-16,-16,16,16)), 1)

    def test_mixed(self):
        wq = self._wq([self._obj(0,0), self._obj(5,5), self._obj(100,0)])
        self.assertEqual(len(wq.natural_objects_in_bbox(-16,-16,16,16)), 2)


# ===========================================================================
# TestBBoxQueryInNamespace
# ===========================================================================

class TestBBoxQueryInNamespace(unittest.TestCase):
    def _obj(self, x, y):
        from unittest.mock import MagicMock
        o = MagicMock()
        o.position = Position(x=float(x), y=float(y))
        return o

    def _wq(self, objects):
        ws = WorldState()
        ws.natural_objects = objects
        ws._rebuild_entity_indices()
        return WorldQuery(ws)

    def test_is_clear_empty(self):
        self.assertTrue(_eval("bbox(-16,-16,16,16).is_clear", self._wq([])))

    def test_is_clear_false_with_object(self):
        self.assertFalse(_eval("bbox(-16,-16,16,16).is_clear", self._wq([self._obj(0,0)])))

    def test_natural_count_zero(self):
        self.assertTrue(_eval("bbox(-16,-16,16,16).natural_count == 0", self._wq([])))

    def test_natural_count_correct(self):
        wq = self._wq([self._obj(0,0), self._obj(5,5)])
        self.assertTrue(_eval("bbox(-16,-16,16,16).natural_count == 2", wq))

    def test_outside_not_counted(self):
        wq = self._wq([self._obj(0,0), self._obj(100,100)])
        self.assertTrue(_eval("bbox(-16,-16,16,16).natural_count == 1", wq))

    def test_bbox_callable_in_namespace(self):
        ns = _ns(self._wq([]))
        self.assertIn("bbox", ns)
        self.assertTrue(callable(ns["bbox"]))

    def test_different_bbox_sizes(self):
        wq = self._wq([self._obj(10,10)])
        self.assertFalse(_eval("bbox(0,0,16,16).is_clear", wq))
        self.assertTrue(_eval("bbox(0,0,8,8).is_clear", wq))

    def test_natural_objects_len(self):
        wq = self._wq([self._obj(5,5)])
        self.assertTrue(_eval("len(bbox(-16,-16,16,16).natural_objects) == 1", wq))




# ===========================================================================
# TestEntityQueryBbox
# ===========================================================================

class TestEntityQueryBbox(unittest.TestCase):
    def _entity(self, eid, x, y, force="player", proto="container"):
        from unittest.mock import MagicMock
        from world import EntityStatus
        e = MagicMock(spec=False)
        e.entity_id = eid
        e.name = "iron-chest"
        e.position = Position(x=float(x), y=float(y))
        e.force = force
        e.prototype_type = proto
        e.status = EntityStatus.WORKING
        e.recipe = None
        return e

    def _wq(self, entities):
        ws = WorldState(entities=entities)
        ws._rebuild_entity_indices()
        return WorldQuery(ws)

    def test_with_bbox_includes_inside(self):
        wq = self._wq([self._entity(1, 0, 0)])
        self.assertEqual(wq.entities().with_bbox(-16,-16,16,16).count(), 1)

    def test_with_bbox_excludes_outside(self):
        wq = self._wq([self._entity(1, 100, 100)])
        self.assertEqual(wq.entities().with_bbox(-16,-16,16,16).count(), 0)

    def test_with_force_filters(self):
        wq = self._wq([self._entity(1, 0, 0, force="player"),
                       self._entity(2, 1, 1, force="enemy")])
        self.assertEqual(wq.entities().with_force("player").count(), 1)

    def test_with_prototype_type(self):
        wq = self._wq([self._entity(1, 0, 0, proto="transport-belt"),
                       self._entity(2, 5, 5, proto="container")])
        self.assertEqual(wq.entities().with_prototype_type("transport-belt").count(), 1)

    def test_nearest_to(self):
        wq = self._wq([self._entity(1, 1, 1), self._entity(2, 50, 50)])
        result = wq.entities().nearest_to(Position(0.0, 0.0))
        self.assertIsNotNone(result)
        self.assertEqual(result.entity_id, 1)

    def test_nearest_to_empty(self):
        self.assertIsNone(self._wq([]).entities().nearest_to(Position(0.0, 0.0)))

    def test_chain_bbox_and_force(self):
        wq = self._wq([self._entity(1, 0, 0, force="player"),
                       self._entity(2, 50, 50, force="player"),
                       self._entity(3, 5, 5, force="enemy")])
        result = wq.entities().with_bbox(-16,-16,16,16).with_force("player").get()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity_id, 1)


class TestBBoxQueryEntities(unittest.TestCase):
    def _entity(self, eid, x, y, force="player"):
        from unittest.mock import MagicMock
        from world import EntityStatus
        e = MagicMock(spec=False)
        e.entity_id = eid
        e.name = "iron-chest"
        e.position = Position(x=float(x), y=float(y))
        e.force = force
        e.prototype_type = "container"
        e.status = EntityStatus.WORKING
        e.recipe = None
        return e

    def _wq(self, entities=None, naturals=None):
        from unittest.mock import MagicMock
        ws = WorldState(entities=entities or [])
        ws.natural_objects = naturals or []
        ws._rebuild_entity_indices()
        return WorldQuery(ws)

    def _nat(self, x, y):
        from unittest.mock import MagicMock
        o = MagicMock()
        o.position = Position(x=float(x), y=float(y))
        return o

    def test_entity_count_zero(self):
        self.assertEqual(self._wq().in_bbox(-16,-16,16,16).entity_count, 0)

    def test_entity_count_correct(self):
        self.assertEqual(self._wq([self._entity(1, 0, 0)]).in_bbox(-16,-16,16,16).entity_count, 1)

    def test_has_player_entities_false(self):
        self.assertFalse(self._wq().in_bbox(-16,-16,16,16).has_player_entities)

    def test_has_player_entities_true(self):
        self.assertTrue(self._wq([self._entity(1, 0, 0)]).in_bbox(-16,-16,16,16).has_player_entities)

    def test_is_buildable_empty(self):
        self.assertTrue(self._wq().in_bbox(-16,-16,16,16).is_buildable)

    def test_is_buildable_false_with_natural(self):
        self.assertFalse(self._wq(naturals=[self._nat(0, 0)]).in_bbox(-16,-16,16,16).is_buildable)

    def test_is_buildable_false_with_entity(self):
        self.assertFalse(self._wq([self._entity(1, 0, 0)]).in_bbox(-16,-16,16,16).is_buildable)

    def test_in_bbox_returns_bboxquery(self):
        from planning.evaluation.condition_namespace import BBoxQuery
        self.assertIsInstance(self._wq().in_bbox(-16,-16,16,16), BBoxQuery)

    def test_entities_method_chains(self):
        wq = self._wq([self._entity(1, 0, 0, force="player"),
                       self._entity(2, 5, 5, force="enemy")])
        result = wq.in_bbox(-16,-16,16,16).entities().with_force("player").count()
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)