"""
tests/unit/world/test_state.py

Tests for world/state.py, world/query.py, and world/writer.py.

This file tests WorldState internals directly (indices, fields) — that is
intentional, this is the unit test for those modules.  Other test files that
need a WorldState only as scaffolding should use the helpers in fixtures.py.

Organised in four sections:
  1. Core dataclass tests  — Position, Inventory, WorldState basics, BeltLane,
                             BeltSegment, ExplorationState
  2. WorldState index tests — internal indices built by __post_init__
  3. WorldQuery tests      — all query methods, connectivity, composable builder
  4. WorldWriter tests     — section replacement, fine-grained mutation,
                             integrate_snapshot

Run with:  python tests/unit/world/test_state.py
"""

from __future__ import annotations

import unittest

from world.state import (
    BeltLane, BeltSegment, BiterBase, DamagedEntity, DestroyedEntity,
    Direction, EntityState, EntityStatus, ExplorationState,
    GroundItem, Inventory, InventorySlot, InserterState,
    LogisticsState, PlayerState, Position, PowerGrid,
    ResourcePatch, ResearchState,
    ThreatState, WorldState,
)
from world.query import WorldQuery
from world.writer import WorldWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity(entity_id, name, x=0.0, y=0.0, status=EntityStatus.WORKING,
            recipe=None, inventory=None) -> EntityState:
    return EntityState(
        entity_id=entity_id, name=name, position=Position(x, y),
        status=status, recipe=recipe, inventory=inventory,
    )


def _inserter(entity_id, x=0.0, y=0.0, active=False,
              pickup_pos=None, drop_pos=None) -> InserterState:
    return InserterState(
        entity_id=entity_id, position=Position(x, y), active=active,
        pickup_position=pickup_pos, drop_position=drop_pos,
    )


def _wq(state: WorldState) -> WorldQuery:
    return WorldQuery(state)


# ===========================================================================
# Section 1 — Core dataclass tests
# ===========================================================================

class TestPosition(unittest.TestCase):
    def test_distance_same_point(self):
        p = Position(0, 0)
        self.assertEqual(p.distance_to(p), 0.0)

    def test_distance_known(self):
        self.assertAlmostEqual(Position(0, 0).distance_to(Position(3, 4)), 5.0)

    def test_frozen(self):
        p = Position(1, 2)
        with self.assertRaises((AttributeError, TypeError)):
            p.x = 99  # type: ignore


class TestInventory(unittest.TestCase):
    def test_count_empty(self):
        self.assertEqual(Inventory().count("iron-plate"), 0)

    def test_count_accumulates(self):
        inv = Inventory(slots=[InventorySlot("iron-plate", 30), InventorySlot("iron-plate", 20)])
        self.assertEqual(inv.count("iron-plate"), 50)

    def test_as_dict(self):
        inv = Inventory(slots=[InventorySlot("coal", 10)])
        self.assertEqual(inv.as_dict(), {"coal": 10})

    def test_is_empty(self):
        self.assertTrue(Inventory().is_empty())
        self.assertFalse(Inventory(slots=[InventorySlot("coal", 1)]).is_empty())


class TestWorldStateBasics(unittest.TestCase):
    def _ws(self) -> WorldState:
        return WorldState(
            tick=3600,
            player=PlayerState(
                position=Position(0, 0),
                inventory=Inventory(slots=[InventorySlot("iron-plate", 100)]),
            ),
            entities=[
                _entity(1, "assembling-machine-1", 5, 5,
                        status=EntityStatus.WORKING, recipe="iron-gear-wheel"),
                _entity(2, "electric-furnace", 10, 10, status=EntityStatus.NO_INPUT),
            ],
            logistics=LogisticsState(
                power=PowerGrid(produced_kw=500, consumed_kw=300, satisfaction=1.0)
            ),
        )

    def test_game_time_seconds(self):
        self.assertEqual(WorldState(tick=600).game_time_seconds, 10.0)

    def test_power_headroom(self):
        self.assertAlmostEqual(self._ws().logistics.power.headroom_kw, 200.0)

    def test_threat_empty_by_default(self):
        self.assertTrue(WorldState().threat.is_empty)

    def test_repr(self):
        r = repr(self._ws())
        self.assertIn("tick=3600", r)
        self.assertIn("entities=2", r)
        self.assertIn("charted_chunks=0", r)

    def test_resource_patch_accepts_mod_string(self):
        patch = ResourcePatch("se-cryonite", Position(100, 200), 50000, 40, 3600)
        self.assertEqual(patch.resource_type, "se-cryonite")
        self.assertEqual(patch.observed_at, 3600)

    def test_ground_items_populated(self):
        ws = WorldState(ground_items=[
            GroundItem("iron-plate", Position(5, 5), 12, observed_at=600, age_ticks=30),
        ])
        self.assertEqual(ws.ground_items[0].count, 12)
        self.assertEqual(ws.ground_items[0].age_ticks, 30)

    def test_threat_state_has_no_damage_fields(self):
        threat = ThreatState()
        self.assertFalse(hasattr(threat, "damaged_entities"))
        self.assertFalse(hasattr(threat, "destroyed_entities"))

    def test_worldstate_damaged_empty_by_default(self):
        self.assertFalse(WorldState().has_damage)
        self.assertEqual(WorldState().damaged_entities, [])

    def test_worldstate_has_damage(self):
        ws = WorldState(damaged_entities=[DamagedEntity(10, "stone-wall", Position(50, 50), 0.4)])
        self.assertTrue(ws.has_damage)

    def test_worldstate_recent_losses(self):
        ws = WorldState(destroyed_entities=[DestroyedEntity("gun-turret", Position(60, 60), 1750, "biter")])
        self.assertEqual(ws.recent_losses[0].cause, "biter")

    def test_worldstate_destroyed_vehicle_cause(self):
        ws = WorldState(destroyed_entities=[DestroyedEntity("wooden-chest", Position(10, 10), 600, "vehicle")])
        self.assertEqual(ws.recent_losses[0].cause, "vehicle")

    def test_worldstate_destroyed_ttl_default(self):
        self.assertEqual(WorldState().destroyed_ttl_ticks, 18_000)

    def test_damage_independent_of_biters(self):
        ws = WorldState(
            threat=ThreatState(),
            damaged_entities=[DamagedEntity(1, "assembling-machine-1", Position(5, 5), 0.7)],
        )
        self.assertTrue(ws.threat.is_empty)
        self.assertTrue(ws.has_damage)


# ===========================================================================
# Section 2 — WorldState index tests
# ===========================================================================

class TestWorldStateIndices(unittest.TestCase):
    """Verify that __post_init__ builds correct internal indices."""

    def _ws(self) -> WorldState:
        return WorldState(entities=[
            _entity(1, "assembling-machine-1", 5, 5,
                    status=EntityStatus.WORKING, recipe="iron-gear-wheel"),
            _entity(2, "assembling-machine-1", 10, 10,
                    status=EntityStatus.NO_INPUT, recipe="iron-gear-wheel"),
            _entity(3, "electric-furnace", 15, 15, status=EntityStatus.WORKING),
        ])

    def test_by_id_populated(self):
        ws = self._ws()
        self.assertIn(1, ws._by_id)
        self.assertIn(3, ws._by_id)
        self.assertNotIn(99, ws._by_id)

    def test_by_name_populated(self):
        ws = self._ws()
        self.assertEqual(len(ws._by_name["assembling-machine-1"]), 2)
        self.assertEqual(len(ws._by_name["electric-furnace"]), 1)
        self.assertNotIn("stone-furnace", ws._by_name)

    def test_by_recipe_populated(self):
        ws = self._ws()
        self.assertEqual(len(ws._by_recipe["iron-gear-wheel"]), 2)

    def test_by_status_populated(self):
        ws = self._ws()
        self.assertEqual(len(ws._by_status[EntityStatus.WORKING]), 2)
        self.assertEqual(len(ws._by_status[EntityStatus.NO_INPUT]), 1)

    def test_entities_without_recipe_not_in_recipe_index(self):
        ws = self._ws()
        for entities in ws._by_recipe.values():
            for e in entities:
                self.assertNotEqual(e.entity_id, 3)

    def test_empty_state_has_empty_indices(self):
        ws = WorldState()
        self.assertEqual(ws._by_id, {})
        self.assertEqual(ws._by_name, {})
        self.assertEqual(ws._by_recipe, {})
        self.assertEqual(ws._by_status, {})

    def test_inserter_indices_none_until_rebuilt(self):
        ws = WorldState(entities=[_entity(1, "iron-chest", 5, 5)])
        self.assertIsNone(ws._inserters_from)
        self.assertIsNone(ws._inserters_to)


# ===========================================================================
# Section 3 — WorldQuery tests
# ===========================================================================

class TestWorldQueryEntityLookups(unittest.TestCase):

    def _ws(self) -> WorldState:
        return WorldState(entities=[
            _entity(1, "assembling-machine-1", 5, 5,
                    status=EntityStatus.WORKING, recipe="iron-gear-wheel"),
            _entity(2, "electric-furnace", 10, 10, status=EntityStatus.NO_INPUT),
            _entity(3, "assembling-machine-1", 15, 15,
                    status=EntityStatus.IDLE, recipe="copper-cable"),
        ])

    def test_entity_by_id_found(self):
        wq = _wq(self._ws())
        e = wq.entity_by_id(1)
        self.assertIsNotNone(e)
        self.assertEqual(e.name, "assembling-machine-1")

    def test_entity_by_id_missing(self):
        self.assertIsNone(_wq(self._ws()).entity_by_id(999))

    def test_entities_by_name(self):
        self.assertEqual(len(_wq(self._ws()).entities_by_name("assembling-machine-1")), 2)

    def test_entities_by_name_empty(self):
        self.assertEqual(_wq(self._ws()).entities_by_name("stone-furnace"), [])

    def test_entities_by_status(self):
        no_input = _wq(self._ws()).entities_by_status(EntityStatus.NO_INPUT)
        self.assertEqual(len(no_input), 1)
        self.assertEqual(no_input[0].entity_id, 2)

    def test_entities_by_recipe(self):
        result = _wq(self._ws()).entities_by_recipe("iron-gear-wheel")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity_id, 1)

    def test_all_entities(self):
        self.assertEqual(len(_wq(self._ws()).all_entities()), 3)

    def test_inventory_count(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot("iron-plate", 100)]),
        ))
        wq = _wq(ws)
        self.assertEqual(wq.inventory_count("iron-plate"), 100)
        self.assertEqual(wq.inventory_count("coal"), 0)

    def test_resources_of_type(self):
        ws = WorldState(resource_map=[
            ResourcePatch("iron-ore", Position(10, 10), 50000, 30),
            ResourcePatch("copper-ore", Position(50, 50), 30000, 20),
            ResourcePatch("iron-ore", Position(200, 200), 10000, 10),
        ])
        wq = _wq(ws)
        self.assertEqual(len(wq.resources_of_type("iron-ore")), 2)
        self.assertEqual(len(wq.resources_of_type("copper-ore")), 1)
        self.assertEqual(wq.resources_of_type("coal"), [])

    def test_section_staleness_never_observed(self):
        self.assertIsNone(_wq(WorldState(tick=1000)).section_staleness("resource_map", 1000))

    def test_section_staleness_fresh(self):
        wq = _wq(WorldState(tick=1000, observed_at={"resource_map": 1000}))
        self.assertEqual(wq.section_staleness("resource_map", 1000), 0)

    def test_section_staleness_stale(self):
        wq = _wq(WorldState(tick=1000, observed_at={"resource_map": 400}))
        self.assertEqual(wq.section_staleness("resource_map", 1000), 600)

    def test_tick_property(self):
        self.assertEqual(_wq(WorldState(tick=7200)).tick, 7200)

    def test_game_time_seconds(self):
        self.assertAlmostEqual(_wq(WorldState(tick=600)).game_time_seconds, 10.0)

    def test_tech_unlocked(self):
        ws = WorldState(research=ResearchState(unlocked=["automation"]))
        wq = _wq(ws)
        self.assertTrue(wq.tech_unlocked("automation"))
        self.assertFalse(wq.tech_unlocked("logistics"))

    def test_charted_chunks_property(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            exploration=ExplorationState(charted_chunks=75),
        ))
        self.assertEqual(_wq(ws).charted_chunks, 75)

    def test_charted_tiles_property(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            exploration=ExplorationState(charted_chunks=10),
        ))
        self.assertEqual(_wq(ws).charted_tiles, 10240)

    def test_charted_area_km2_property(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            exploration=ExplorationState(charted_chunks=1000),
        ))
        self.assertAlmostEqual(_wq(ws).charted_area_km2, 1.024)

    def test_state_passthrough(self):
        ws = WorldState(tick=1234)
        self.assertIs(_wq(ws).state, ws)


class TestWorldQueryConnectivity(unittest.TestCase):

    def _ws_with_inserter(self, chest_pos, pickup_pos, drop_pos=None):
        chest = _entity(1, "iron-chest", *chest_pos)
        ins = _inserter(10, 0, 0,
                        pickup_pos=Position(*pickup_pos),
                        drop_pos=Position(*(drop_pos or (0, -2))))
        return WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))

    def test_inserters_taking_from_match(self):
        ws = self._ws_with_inserter((5.0, 5.0), (5.0, 5.0))
        wq = _wq(ws)
        self.assertEqual(len(wq.inserters_taking_from(1)), 1)
        self.assertEqual(wq.inserters_taking_from(1)[0].entity_id, 10)

    def test_inserters_taking_from_no_match(self):
        ws = self._ws_with_inserter((5.0, 5.0), (5.0, 7.0))
        self.assertEqual(_wq(ws).inserters_taking_from(1), [])

    def test_inserters_taking_from_unknown_entity(self):
        ws = self._ws_with_inserter((5.0, 5.0), (5.0, 5.0))
        self.assertEqual(_wq(ws).inserters_taking_from(999), [])

    def test_inserters_taking_from_null_pickup_excluded(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ins = _inserter(10, 0, 0, pickup_pos=None, drop_pos=Position(5.0, 5.0))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertEqual(_wq(ws).inserters_taking_from(1), [])

    def test_inserters_taking_from_larger_entity(self):
        assembler = _entity(2, "assembling-machine-2", 5.0, 5.0)
        ins = _inserter(20, 0, 0, pickup_pos=Position(6.5, 5.0))
        ws = WorldState(entities=[assembler], logistics=LogisticsState(inserters={20: ins}))
        wq = _wq(ws)
        self.assertEqual(wq.inserters_taking_from(2, tile_width=1, tile_height=1), [])
        self.assertEqual(len(wq.inserters_taking_from(2, tile_width=3, tile_height=3)), 1)

    def test_inserters_delivering_to_match(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ins = _inserter(10, 0, 0, pickup_pos=Position(5.0, 7.0), drop_pos=Position(5.0, 5.0))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertEqual(len(_wq(ws).inserters_delivering_to(1)), 1)

    def test_inserters_delivering_to_no_match(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ins = _inserter(10, 0, 0, pickup_pos=Position(5.0, 7.0), drop_pos=Position(5.0, 8.0))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertEqual(_wq(ws).inserters_delivering_to(1), [])

    def test_inserters_delivering_to_null_drop_excluded(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ins = _inserter(10, 0, 0, pickup_pos=Position(5.0, 7.0), drop_pos=None)
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertEqual(_wq(ws).inserters_delivering_to(1), [])

    def test_inserters_taking_from_type(self):
        chest_a = _entity(1, "iron-chest", 5.0, 5.0)
        chest_b = _entity(2, "iron-chest", 15.0, 5.0)
        other   = _entity(3, "wooden-chest", 25.0, 5.0)
        ins_a = _inserter(10, 0, 0, pickup_pos=Position(5.0, 5.0))
        ins_b = _inserter(11, 0, 0, pickup_pos=Position(15.0, 5.0))
        ins_c = _inserter(12, 0, 0, pickup_pos=Position(25.0, 5.0))
        ws = WorldState(
            entities=[chest_a, chest_b, other],
            logistics=LogisticsState(inserters={10: ins_a, 11: ins_b, 12: ins_c}),
        )
        ids = {i.entity_id for i in _wq(ws).inserters_taking_from_type("iron-chest")}
        self.assertEqual(ids, {10, 11})

    def test_inserters_taking_from_type_unknown(self):
        ws = WorldState(entities=[_entity(1, "iron-chest", 5.0, 5.0)])
        self.assertEqual(_wq(ws).inserters_taking_from_type("steel-chest"), [])

    def test_inserters_delivering_to_type(self):
        chest = _entity(1, "iron-chest", 5.0, 3.0)
        ins = _inserter(10, 0, 0, pickup_pos=Position(5.0, 5.0), drop_pos=Position(5.0, 3.0))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertEqual(len(_wq(ws).inserters_delivering_to_type("iron-chest")), 1)

    def test_multiple_inserters_only_one_matches(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ins_a = _inserter(10, 0, 0, pickup_pos=Position(5.0, 5.0))
        ins_b = _inserter(11, 0, 0, pickup_pos=Position(10.0, 10.0))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins_a, 11: ins_b}))
        result = _wq(ws).inserters_taking_from(1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity_id, 10)

    def test_no_inserters_in_world(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        self.assertEqual(_wq(WorldState(entities=[chest])).inserters_taking_from(1), [])


class TestEntityQueryBuilder(unittest.TestCase):

    def _ws(self) -> WorldState:
        return WorldState(entities=[
            _entity(1, "assembling-machine-1", 5, 5,
                    status=EntityStatus.WORKING, recipe="electronic-circuit"),
            _entity(2, "assembling-machine-1", 10, 10,
                    status=EntityStatus.NO_INPUT, recipe="electronic-circuit"),
            _entity(3, "assembling-machine-1", 15, 15,
                    status=EntityStatus.IDLE, recipe="iron-gear-wheel"),
            _entity(4, "electric-furnace", 20, 20, status=EntityStatus.WORKING),
        ])

    def test_with_name_filters(self):
        result = _wq(self._ws()).entities().with_name("assembling-machine-1").get()
        self.assertEqual(len(result), 3)

    def test_with_recipe_filters(self):
        result = _wq(self._ws()).entities().with_recipe("electronic-circuit").get()
        self.assertEqual(len(result), 2)

    def test_with_status_filters(self):
        result = _wq(self._ws()).entities().with_status(EntityStatus.WORKING).get()
        self.assertEqual(len(result), 2)

    def test_name_and_recipe_chain(self):
        result = (_wq(self._ws()).entities()
                  .with_name("assembling-machine-1")
                  .with_recipe("iron-gear-wheel")
                  .get())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity_id, 3)

    def test_count_terminal(self):
        self.assertEqual(_wq(self._ws()).entities().with_name("electric-furnace").count(), 1)

    def test_first_terminal(self):
        first = _wq(self._ws()).entities().with_name("electric-furnace").first()
        self.assertIsNotNone(first)
        self.assertEqual(first.entity_id, 4)

    def test_first_none_when_empty(self):
        self.assertIsNone(_wq(self._ws()).entities().with_name("stone-furnace").first())

    def test_with_predicate(self):
        result = _wq(self._ws()).entities().with_predicate(lambda e: e.entity_id > 2).get()
        self.assertEqual(len(result), 2)

    def test_with_inserter_input_output(self):
        assembler = _entity(1, "assembling-machine-1", 5, 5, recipe="electronic-circuit")
        feeder    = _inserter(10, 4, 5, pickup_pos=Position(3, 5), drop_pos=Position(5, 5))
        extractor = _inserter(11, 6, 5, pickup_pos=Position(5, 5), drop_pos=Position(7, 5))
        ws = WorldState(
            entities=[assembler],
            logistics=LogisticsState(inserters={10: feeder, 11: extractor}),
        )
        result = (_wq(ws).entities()
                  .with_recipe("electronic-circuit")
                  .with_inserter_input()
                  .with_inserter_output()
                  .get())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity_id, 1)

    def test_with_inserter_input_filters_unwired(self):
        a1 = _entity(1, "assembling-machine-1", 5, 5, recipe="electronic-circuit")
        a2 = _entity(2, "assembling-machine-1", 15, 15, recipe="electronic-circuit")
        feeder = _inserter(10, 4, 5, pickup_pos=Position(3, 5), drop_pos=Position(5, 5))
        ws = WorldState(entities=[a1, a2], logistics=LogisticsState(inserters={10: feeder}))
        result = _wq(ws).entities().with_inserter_input().get()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity_id, 1)

    def test_fully_connected_entities(self):
        assembler = _entity(1, "assembling-machine-1", 5, 5, recipe="electronic-circuit")
        feeder    = _inserter(10, 4, 5, pickup_pos=Position(3, 5), drop_pos=Position(5, 5))
        extractor = _inserter(11, 6, 5, pickup_pos=Position(5, 5), drop_pos=Position(7, 5))
        ws = WorldState(
            entities=[assembler],
            logistics=LogisticsState(inserters={10: feeder, 11: extractor}),
        )
        self.assertEqual(len(_wq(ws).fully_connected_entities("electronic-circuit")), 1)

    def test_fully_connected_entities_not_wired(self):
        assembler = _entity(1, "assembling-machine-1", 5, 5, recipe="electronic-circuit")
        self.assertEqual(_wq(WorldState(entities=[assembler])).fully_connected_entities("electronic-circuit"), [])


# ===========================================================================
# Section 4 — WorldWriter tests
# ===========================================================================

class TestWorldWriterSectionReplacement(unittest.TestCase):

    def test_replace_entities_rebuilds_indices(self):
        ws = WorldState()
        WorldWriter(ws).replace_entities([_entity(1, "stone-furnace", 5, 5)], tick=100)
        self.assertIsNotNone(_wq(ws).entity_by_id(1))
        self.assertEqual(ws.observed_at["entities"], 100)

    def test_replace_logistics_updates_observed_at(self):
        ws = WorldState()
        WorldWriter(ws).replace_logistics(
            LogisticsState(power=PowerGrid(produced_kw=1000, consumed_kw=500)), tick=200
        )
        self.assertAlmostEqual(ws.logistics.power.headroom_kw, 500.0)
        self.assertEqual(ws.observed_at["logistics"], 200)

    def test_replace_research_updates(self):
        ws = WorldState()
        WorldWriter(ws).replace_research(ResearchState(unlocked=["automation"]), tick=300)
        self.assertIn("automation", ws.research.unlocked)
        self.assertEqual(ws.observed_at["research"], 300)

    def test_replace_player(self):
        ws = WorldState()
        WorldWriter(ws).replace_player(
            PlayerState(position=Position(10, 20),
                        inventory=Inventory(slots=[InventorySlot("coal", 50)])),
            tick=400,
        )
        self.assertEqual(ws.player.position, Position(10, 20))
        self.assertEqual(ws.observed_at["player"], 400)


class TestWorldWriterFineGrained(unittest.TestCase):

    def _ws(self) -> WorldState:
        return WorldState(entities=[
            _entity(1, "assembling-machine-1", 5, 5,
                    status=EntityStatus.IDLE, recipe="iron-gear-wheel"),
        ])

    def test_add_entity(self):
        ws = WorldState()
        WorldWriter(ws).add_entity(_entity(1, "stone-furnace", 5, 5))
        self.assertIsNotNone(_wq(ws).entity_by_id(1))

    def test_remove_entity_found(self):
        ws = self._ws()
        self.assertTrue(WorldWriter(ws).remove_entity(1))
        self.assertEqual(ws.entities, [])
        self.assertIsNone(_wq(ws).entity_by_id(1))

    def test_remove_entity_not_found(self):
        ws = self._ws()
        self.assertFalse(WorldWriter(ws).remove_entity(999))

    def test_update_entity_status(self):
        ws = self._ws()
        WorldWriter(ws).update_entity_status(1, EntityStatus.WORKING)
        self.assertEqual(ws.entities[0].status, EntityStatus.WORKING)
        self.assertEqual(len(_wq(ws).entities_by_status(EntityStatus.WORKING)), 1)
        self.assertEqual(_wq(ws).entities_by_status(EntityStatus.IDLE), [])

    def test_update_entity_recipe(self):
        ws = self._ws()
        WorldWriter(ws).update_entity_recipe(1, "copper-cable")
        self.assertEqual(ws.entities[0].recipe, "copper-cable")
        self.assertEqual(_wq(ws).entities_by_recipe("copper-cable"), [ws.entities[0]])
        self.assertEqual(_wq(ws).entities_by_recipe("iron-gear-wheel"), [])

    def test_update_entity_status_not_found(self):
        ws = self._ws()
        self.assertFalse(WorldWriter(ws).update_entity_status(999, EntityStatus.WORKING))

    def test_update_player_inventory(self):
        ws = WorldState()
        WorldWriter(ws).update_player_inventory(Inventory(slots=[InventorySlot("iron-plate", 50)]))
        self.assertEqual(_wq(ws).inventory_count("iron-plate"), 50)

    def test_update_player_position(self):
        ws = WorldState()
        WorldWriter(ws).update_player_position(Position(100, 200))
        self.assertEqual(ws.player.position, Position(100, 200))


class TestWorldWriterIntegrateSnapshot(unittest.TestCase):

    def test_integrate_fresh_snapshot_updates_live(self):
        live = WorldState(tick=100)
        ww = WorldWriter(live)
        snap = WorldState(tick=200, entities=[_entity(1, "stone-furnace", 5, 5)],
                          observed_at={"entities": 200})
        ww.integrate_snapshot(snap)
        self.assertEqual(live.tick, 200)
        self.assertEqual(len(live.entities), 1)
        self.assertEqual(live.observed_at["entities"], 200)

    def test_integrate_stale_snapshot_skips_section(self):
        live = WorldState(tick=300, observed_at={"entities": 300},
                          entities=[_entity(1, "stone-furnace", 5, 5)])
        ww = WorldWriter(live)
        snap = WorldState(tick=100, entities=[_entity(2, "electric-furnace", 10, 10)],
                          observed_at={"entities": 100})
        ww.integrate_snapshot(snap)
        self.assertEqual(live.entities[0].name, "stone-furnace")

    def test_integrate_rebuilds_entity_index(self):
        live = WorldState(tick=0)
        ww = WorldWriter(live)
        snap = WorldState(tick=100, entities=[_entity(42, "iron-chest", 5, 5)],
                          observed_at={"entities": 100})
        ww.integrate_snapshot(snap)
        self.assertIsNotNone(_wq(live).entity_by_id(42))

    def test_integrate_advances_tick(self):
        live = WorldState(tick=50)
        WorldWriter(live).integrate_snapshot(WorldState(tick=150))
        self.assertEqual(live.tick, 150)

    def test_integrate_destroyed_deduplicates(self):
        rec = DestroyedEntity("gun-turret", Position(10, 10), 500, "biter")
        live = WorldState(tick=600, destroyed_entities=[rec],
                          observed_at={"destroyed_entities": 500})
        snap = WorldState(tick=600, destroyed_entities=[rec],
                          observed_at={"destroyed_entities": 600})
        WorldWriter(live).integrate_snapshot(snap)
        self.assertEqual(len(live.destroyed_entities), 1)

    def test_integrate_destroyed_merges_distinct(self):
        rec1 = DestroyedEntity("gun-turret", Position(10, 10), 500, "biter")
        rec2 = DestroyedEntity("stone-wall", Position(20, 20), 550, "biter")
        live = WorldState(tick=600, destroyed_entities=[rec1],
                          observed_at={"destroyed_entities": 500})
        snap = WorldState(tick=600, destroyed_entities=[rec2],
                          observed_at={"destroyed_entities": 600})
        WorldWriter(live).integrate_snapshot(snap)
        self.assertEqual(len(live.destroyed_entities), 2)


# ===========================================================================
# Belt / Exploration / InserterState dataclass tests
# ===========================================================================

class TestExplorationState(unittest.TestCase):

    def test_defaults_zero(self):
        e = ExplorationState()
        self.assertEqual(e.charted_chunks, 0)
        self.assertEqual(e.charted_tiles, 0)
        self.assertAlmostEqual(e.charted_area_km2, 0.0)

    def test_charted_tiles_is_chunks_times_1024(self):
        self.assertEqual(ExplorationState(charted_chunks=10).charted_tiles, 10240)

    def test_charted_area_km2(self):
        self.assertAlmostEqual(ExplorationState(charted_chunks=1000).charted_area_km2, 1.024)

    def test_charted_area_single_chunk(self):
        self.assertAlmostEqual(ExplorationState(charted_chunks=1).charted_area_km2, 0.001024)

    def test_large_exploration(self):
        self.assertAlmostEqual(ExplorationState(charted_chunks=10000).charted_area_km2, 10.24)


class TestWorldStateExploration(unittest.TestCase):

    def test_charted_chunks_shorthand(self):
        ws = WorldState(player=PlayerState(position=Position(0, 0),
                                           exploration=ExplorationState(charted_chunks=75)))
        self.assertEqual(ws.charted_chunks, 75)

    def test_charted_chunks_default_zero(self):
        self.assertEqual(WorldState().charted_chunks, 0)

    def test_exploration_on_player_state(self):
        self.assertEqual(PlayerState(exploration=ExplorationState(charted_chunks=42)).exploration.charted_chunks, 42)

    def test_default_player_has_zero_exploration(self):
        self.assertEqual(PlayerState().exploration.charted_chunks, 0)

    def test_exploration_independent_of_entity_scan(self):
        ws = WorldState(
            player=PlayerState(position=Position(0, 0),
                               exploration=ExplorationState(charted_chunks=100)),
            entities=[], observed_at={},
        )
        self.assertEqual(ws.charted_chunks, 100)


class TestInserterState(unittest.TestCase):
    def test_defaults(self):
        ins = InserterState(entity_id=1, position=Position(0, 0))
        self.assertFalse(ins.active)
        self.assertIsNone(ins.pickup_position)
        self.assertIsNone(ins.drop_position)

    def test_active_flag(self):
        self.assertTrue(InserterState(entity_id=1, position=Position(0, 0), active=True).active)

    def test_positions_stored(self):
        ins = InserterState(
            entity_id=5, position=Position(3.0, 3.0), active=True,
            pickup_position=Position(3.0, 4.0), drop_position=Position(3.0, 2.0),
        )
        self.assertEqual(ins.pickup_position, Position(3.0, 4.0))
        self.assertEqual(ins.drop_position, Position(3.0, 2.0))


class TestLogisticsStateInserters(unittest.TestCase):
    def test_inserters_empty_by_default(self):
        self.assertEqual(LogisticsState().inserters, {})

    def test_inserters_keyed_by_entity_id(self):
        ins = InserterState(entity_id=42, position=Position(0, 0))
        ls = LogisticsState(inserters={42: ins})
        self.assertIs(ls.inserters[42], ins)


class TestBeltLane(unittest.TestCase):

    def test_defaults(self):
        lane = BeltLane()
        self.assertFalse(lane.congested)
        self.assertEqual(lane.items, {})

    def test_is_empty_when_no_items(self):
        self.assertTrue(BeltLane().is_empty())

    def test_is_not_empty_with_items(self):
        self.assertFalse(BeltLane(items={"iron-plate": 3}).is_empty())

    def test_carries_present_item(self):
        self.assertTrue(BeltLane(items={"iron-plate": 4}).carries("iron-plate"))

    def test_carries_absent_item(self):
        self.assertFalse(BeltLane(items={"iron-plate": 4}).carries("copper-plate"))

    def test_carries_zero_count_is_false(self):
        self.assertFalse(BeltLane(items={"iron-plate": 0}).carries("iron-plate"))

    def test_total_items_sums_all(self):
        self.assertEqual(BeltLane(items={"iron-plate": 3, "copper-plate": 2}).total_items(), 5)

    def test_congested_flag(self):
        self.assertTrue(BeltLane(congested=True, items={"iron-plate": 8}).congested)

    def test_multiple_items_on_same_lane(self):
        lane = BeltLane(items={"iron-plate": 3, "copper-plate": 2})
        self.assertTrue(lane.carries("iron-plate"))
        self.assertTrue(lane.carries("copper-plate"))


class TestBeltSegment(unittest.TestCase):

    def _seg(self, lane1_items=None, lane2_items=None, lane1_cong=False, lane2_cong=False):
        return BeltSegment(
            segment_id=1, positions=[Position(0, 0)],
            lane1=BeltLane(congested=lane1_cong, items=lane1_items or {}),
            lane2=BeltLane(congested=lane2_cong, items=lane2_items or {}),
        )

    def test_congested_false_when_both_free(self):
        self.assertFalse(self._seg().congested)

    def test_congested_true_when_lane1_congested(self):
        self.assertTrue(self._seg(lane1_cong=True).congested)

    def test_congested_true_when_lane2_congested(self):
        self.assertTrue(self._seg(lane2_cong=True).congested)

    def test_items_merges_both_lanes(self):
        seg = self._seg(lane1_items={"iron-plate": 4}, lane2_items={"copper-plate": 2})
        self.assertEqual(seg.items.get("iron-plate"), 4)
        self.assertEqual(seg.items.get("copper-plate"), 2)

    def test_items_sums_same_item_across_lanes(self):
        self.assertEqual(
            self._seg(lane1_items={"iron-plate": 4}, lane2_items={"iron-plate": 2}).items["iron-plate"], 6
        )

    def test_carries_item_on_lane1(self):
        self.assertTrue(self._seg(lane1_items={"iron-plate": 3}).carries("iron-plate"))

    def test_carries_item_on_lane2(self):
        self.assertTrue(self._seg(lane2_items={"copper-plate": 2}).carries("copper-plate"))

    def test_carries_false_when_absent(self):
        self.assertFalse(self._seg(lane1_items={"iron-plate": 3}).carries("steel-plate"))

    def test_lanes_are_independent_objects(self):
        seg = self._seg(lane1_items={"iron-plate": 3}, lane2_items={"copper-plate": 2})
        self.assertIsNot(seg.lane1, seg.lane2)


if __name__ == "__main__":
    unittest.main(verbosity=2)