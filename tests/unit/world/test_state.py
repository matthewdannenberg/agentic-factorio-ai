"""
tests/unit/world/test_state.py

Tests for world/state.py

Organised in two sections:
  1. Core dataclass tests  — Position, Inventory, WorldState basics, BeltLane,
                             BeltSegment
  2. Connectivity tests    — InserterState, inserters_taking_from/delivering_to,
                             bounding-box helper

The capability matrix (evaluator condition categories) lives in:
  tests/integration/test_evaluator_capabilities.py

Run with:  python tests/unit/world/test_state.py
"""

from __future__ import annotations

import unittest

from world.state import (
    BeltLane, BeltSegment, BiterBase, DamagedEntity, DestroyedEntity,
    Direction, EntityState, EntityStatus,
    GroundItem, Inventory, InventorySlot, InserterState,
    LogisticsState, PlayerState, Position, PowerGrid,
    ResourceName, ResourcePatch, ResearchState,
    ThreatState, WorldState,
)
def _entity(entity_id, name, x=0.0, y=0.0, status=EntityStatus.WORKING,
            recipe=None, inventory=None) -> EntityState:
    return EntityState(
        entity_id=entity_id,
        name=name,
        position=Position(x, y),
        status=status,
        recipe=recipe,
        inventory=inventory,
    )


def _inserter(entity_id, x=0.0, y=0.0, active=False,
              pickup_pos=None, drop_pos=None) -> InserterState:
    return InserterState(
        entity_id=entity_id,
        position=Position(x, y),
        active=active,
        pickup_position=pickup_pos,
        drop_position=drop_pos,
    )


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
        inv = Inventory(slots=[
            InventorySlot("iron-plate", 30),
            InventorySlot("iron-plate", 20),
        ])
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

    def test_entity_by_id_found(self):
        e = self._ws().entity_by_id(1)
        self.assertIsNotNone(e)
        self.assertEqual(e.name, "assembling-machine-1")

    def test_entity_by_id_missing(self):
        self.assertIsNone(self._ws().entity_by_id(999))

    def test_entities_by_name(self):
        self.assertEqual(len(self._ws().entities_by_name("electric-furnace")), 1)

    def test_entities_by_status(self):
        no_input = self._ws().entities_by_status(EntityStatus.NO_INPUT)
        self.assertEqual(len(no_input), 1)
        self.assertEqual(no_input[0].entity_id, 2)

    def test_inventory_count(self):
        ws = self._ws()
        self.assertEqual(ws.inventory_count("iron-plate"), 100)
        self.assertEqual(ws.inventory_count("coal"), 0)

    def test_power_headroom(self):
        self.assertAlmostEqual(self._ws().logistics.power.headroom_kw, 200.0)

    def test_threat_empty_by_default(self):
        self.assertTrue(WorldState().threat.is_empty)

    def test_repr(self):
        r = repr(self._ws())
        self.assertIn("tick=3600", r)
        self.assertIn("entities=2", r)

    def test_resource_patch_accepts_mod_string(self):
        patch = ResourcePatch("se-cryonite", Position(100, 200), 50000, 40, 3600)
        self.assertEqual(patch.resource_type, "se-cryonite")
        self.assertEqual(patch.observed_at, 3600)

    def test_resource_name_constants(self):
        self.assertEqual(ResourceName.IRON_ORE, "iron-ore")
        self.assertEqual(ResourceName.CRUDE_OIL, "crude-oil")

    def test_resources_of_type(self):
        ws = WorldState(resource_map=[
            ResourcePatch("iron-ore", Position(10, 10), 50000, 30),
            ResourcePatch("copper-ore", Position(50, 50), 30000, 20),
            ResourcePatch("iron-ore", Position(200, 200), 10000, 10),
        ])
        self.assertEqual(len(ws.resources_of_type("iron-ore")), 2)
        self.assertEqual(len(ws.resources_of_type("copper-ore")), 1)
        self.assertEqual(ws.resources_of_type("coal"), [])

    def test_section_staleness_never_observed(self):
        ws = WorldState(tick=1000)
        self.assertIsNone(ws.section_staleness("resource_map", 1000))

    def test_section_staleness_fresh(self):
        ws = WorldState(tick=1000, observed_at={"resource_map": 1000})
        self.assertEqual(ws.section_staleness("resource_map", 1000), 0)

    def test_section_staleness_stale(self):
        ws = WorldState(tick=1000, observed_at={"resource_map": 400})
        self.assertEqual(ws.section_staleness("resource_map", 1000), 600)

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
        ws = WorldState(damaged_entities=[
            DamagedEntity(10, "stone-wall", Position(50, 50), 0.4)
        ])
        self.assertTrue(ws.has_damage)

    def test_worldstate_recent_losses(self):
        ws = WorldState(destroyed_entities=[
            DestroyedEntity("gun-turret", Position(60, 60), 1750, "biter")
        ])
        self.assertEqual(ws.recent_losses[0].cause, "biter")

    def test_worldstate_destroyed_vehicle_cause(self):
        ws = WorldState(destroyed_entities=[
            DestroyedEntity("wooden-chest", Position(10, 10), 600, "vehicle")
        ])
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
# Section 2 — InserterState and connectivity queries
# ===========================================================================

class TestInserterState(unittest.TestCase):
    def test_defaults(self):
        ins = InserterState(entity_id=1, position=Position(0, 0))
        self.assertFalse(ins.active)
        self.assertIsNone(ins.pickup_position)
        self.assertIsNone(ins.drop_position)

    def test_active_flag(self):
        ins = InserterState(entity_id=1, position=Position(0, 0), active=True)
        self.assertTrue(ins.active)

    def test_positions_stored(self):
        ins = InserterState(
            entity_id=5, position=Position(3.0, 3.0), active=True,
            pickup_position=Position(3.0, 4.0),
            drop_position=Position(3.0, 2.0),
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


class TestInsertersTakingFrom(unittest.TestCase):

    def _ws(self, chest_pos, pickup_pos, drop_pos=None):
        chest = _entity(1, "iron-chest", *chest_pos)
        ins = _inserter(10, 0, 0,
                        pickup_pos=Position(*pickup_pos),
                        drop_pos=Position(*(drop_pos or (0, -2))))
        return WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))

    def test_match_when_pickup_on_chest(self):
        ws = self._ws((5.0, 5.0), (5.0, 5.0))
        self.assertEqual(len(ws.inserters_taking_from(1)), 1)
        self.assertEqual(ws.inserters_taking_from(1)[0].entity_id, 10)

    def test_no_match_when_pickup_far_away(self):
        ws = self._ws((5.0, 5.0), (5.0, 7.0))
        self.assertEqual(ws.inserters_taking_from(1), [])

    def test_match_on_entity_edge(self):
        # pickup at exactly centre + 0.5 tile — within epsilon
        ws = self._ws((5.0, 5.0), (5.0, 5.5))
        self.assertEqual(len(ws.inserters_taking_from(1)), 1)

    def test_unknown_entity_id_returns_empty(self):
        ws = self._ws((5.0, 5.0), (5.0, 5.0))
        self.assertEqual(ws.inserters_taking_from(999), [])

    def test_null_pickup_position_excluded(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ins = _inserter(10, 0, 0, pickup_pos=None, drop_pos=Position(5.0, 5.0))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertEqual(ws.inserters_taking_from(1), [])

    def test_multiple_inserters_only_one_matches(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ins_a = _inserter(10, 0, 0, pickup_pos=Position(5.0, 5.0))
        ins_b = _inserter(11, 0, 0, pickup_pos=Position(10.0, 10.0))
        ws = WorldState(entities=[chest],
                        logistics=LogisticsState(inserters={10: ins_a, 11: ins_b}))
        result = ws.inserters_taking_from(1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity_id, 10)

    def test_larger_entity_requires_tile_dimensions(self):
        assembler = _entity(2, "assembling-machine-2", 5.0, 5.0)
        ins = _inserter(20, 0, 0, pickup_pos=Position(6.5, 5.0))
        ws = WorldState(entities=[assembler],
                        logistics=LogisticsState(inserters={20: ins}))
        # 1×1 default — miss
        self.assertEqual(ws.inserters_taking_from(2, tile_width=1, tile_height=1), [])
        # Correct 3×3 — hit
        self.assertEqual(len(ws.inserters_taking_from(2, tile_width=3, tile_height=3)), 1)

    def test_no_inserters_in_world(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ws = WorldState(entities=[chest])
        self.assertEqual(ws.inserters_taking_from(1), [])


class TestInsertersDeliveringTo(unittest.TestCase):

    def _ws(self, chest_pos, drop_pos, pickup_pos=None):
        chest = _entity(1, "iron-chest", *chest_pos)
        ins = _inserter(10, 0, 0,
                        pickup_pos=Position(*(pickup_pos or (0, 2))),
                        drop_pos=Position(*drop_pos))
        return WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))

    def test_match_when_drop_on_chest(self):
        ws = self._ws((5.0, 5.0), (5.0, 5.0))
        self.assertEqual(len(ws.inserters_delivering_to(1)), 1)

    def test_no_match_when_drop_far_away(self):
        ws = self._ws((5.0, 5.0), (5.0, 8.0))
        self.assertEqual(ws.inserters_delivering_to(1), [])

    def test_null_drop_position_excluded(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ins = _inserter(10, 0, 0, pickup_pos=Position(5.0, 7.0), drop_pos=None)
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertEqual(ws.inserters_delivering_to(1), [])

    def test_unknown_entity_id_returns_empty(self):
        ws = self._ws((5.0, 5.0), (5.0, 5.0))
        self.assertEqual(ws.inserters_delivering_to(999), [])


class TestInsertersByType(unittest.TestCase):

    def _ws(self):
        chest_a = _entity(1, "iron-chest", 5.0, 5.0)
        chest_b = _entity(2, "iron-chest", 15.0, 5.0)
        other   = _entity(3, "wooden-chest", 25.0, 5.0)
        ins_a = _inserter(10, 0, 0, pickup_pos=Position(5.0, 5.0),
                          drop_pos=Position(5.0, 3.0))
        ins_b = _inserter(11, 0, 0, pickup_pos=Position(15.0, 5.0),
                          drop_pos=Position(15.0, 3.0))
        ins_c = _inserter(12, 0, 0, pickup_pos=Position(25.0, 5.0),
                          drop_pos=Position(25.0, 3.0))
        return WorldState(
            entities=[chest_a, chest_b, other],
            logistics=LogisticsState(inserters={10: ins_a, 11: ins_b, 12: ins_c}),
        )

    def test_taking_from_type_finds_both_iron_chests(self):
        ws = self._ws()
        ids = {i.entity_id for i in ws.inserters_taking_from_type("iron-chest")}
        self.assertEqual(ids, {10, 11})

    def test_taking_from_type_finds_wooden_chest_only(self):
        ws = self._ws()
        result = ws.inserters_taking_from_type("wooden-chest")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity_id, 12)

    def test_taking_from_type_unknown_returns_empty(self):
        self.assertEqual(self._ws().inserters_taking_from_type("steel-chest"), [])

    def test_delivering_to_type_matches(self):
        # chests are at y=3 in drop_pos — place a chest there to match
        chest = _entity(1, "iron-chest", 5.0, 3.0)
        ins = _inserter(10, 0, 0, pickup_pos=Position(5.0, 5.0),
                        drop_pos=Position(5.0, 3.0))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertEqual(len(ws.inserters_delivering_to_type("iron-chest")), 1)

    def test_delivering_to_type_no_match(self):
        ws = self._ws()  # drop_pos lands at y=3, no chest there
        self.assertEqual(ws.inserters_delivering_to_type("iron-chest"), [])


class TestEntityContainsPosition(unittest.TestCase):

    def _ws_with_entity(self, x, y):
        return WorldState(entities=[_entity(1, "iron-chest", x, y)])

    def test_centre_always_inside(self):
        ws = self._ws_with_entity(10.0, 10.0)
        target = ws.entity_by_id(1)
        self.assertTrue(ws._entity_contains_position(target, Position(10.0, 10.0)))

    def test_far_point_outside(self):
        ws = self._ws_with_entity(10.0, 10.0)
        target = ws.entity_by_id(1)
        self.assertFalse(ws._entity_contains_position(target, Position(12.0, 10.0)))

    def test_larger_tile_dimension_expands_box(self):
        ws = self._ws_with_entity(10.0, 10.0)
        target = ws.entity_by_id(1)
        pt = Position(11.4, 10.0)
        self.assertFalse(ws._entity_contains_position(target, pt, 1, 1))
        self.assertTrue(ws._entity_contains_position(target, pt, 3, 3))



class TestBeltLane(unittest.TestCase):

    def test_defaults(self):
        lane = BeltLane()
        self.assertFalse(lane.congested)
        self.assertEqual(lane.items, {})

    def test_is_empty_when_no_items(self):
        self.assertTrue(BeltLane().is_empty())

    def test_is_not_empty_with_items(self):
        lane = BeltLane(items={"iron-plate": 3})
        self.assertFalse(lane.is_empty())

    def test_carries_present_item(self):
        lane = BeltLane(items={"iron-plate": 4})
        self.assertTrue(lane.carries("iron-plate"))

    def test_carries_absent_item(self):
        lane = BeltLane(items={"iron-plate": 4})
        self.assertFalse(lane.carries("copper-plate"))

    def test_carries_zero_count_is_false(self):
        lane = BeltLane(items={"iron-plate": 0})
        self.assertFalse(lane.carries("iron-plate"))

    def test_total_items_sums_all(self):
        lane = BeltLane(items={"iron-plate": 3, "copper-plate": 2})
        self.assertEqual(lane.total_items(), 5)

    def test_total_items_empty(self):
        self.assertEqual(BeltLane().total_items(), 0)

    def test_congested_flag(self):
        lane = BeltLane(congested=True, items={"iron-plate": 8})
        self.assertTrue(lane.congested)

    def test_multiple_items_on_same_lane(self):
        lane = BeltLane(items={"iron-plate": 3, "copper-plate": 2})
        self.assertTrue(lane.carries("iron-plate"))
        self.assertTrue(lane.carries("copper-plate"))


class TestBeltSegment(unittest.TestCase):

    def _seg(self, lane1_items=None, lane2_items=None,
             lane1_cong=False, lane2_cong=False):
        return BeltSegment(
            segment_id=1,
            positions=[Position(0, 0)],
            lane1=BeltLane(congested=lane1_cong, items=lane1_items or {}),
            lane2=BeltLane(congested=lane2_cong, items=lane2_items or {}),
        )

    def test_congested_false_when_both_free(self):
        self.assertFalse(self._seg().congested)

    def test_congested_true_when_lane1_congested(self):
        self.assertTrue(self._seg(lane1_cong=True).congested)

    def test_congested_true_when_lane2_congested(self):
        self.assertTrue(self._seg(lane2_cong=True).congested)

    def test_congested_true_when_both_congested(self):
        self.assertTrue(self._seg(lane1_cong=True, lane2_cong=True).congested)

    def test_items_merges_both_lanes(self):
        seg = self._seg(lane1_items={"iron-plate": 4}, lane2_items={"copper-plate": 2})
        self.assertEqual(seg.items.get("iron-plate"), 4)
        self.assertEqual(seg.items.get("copper-plate"), 2)

    def test_items_sums_same_item_across_lanes(self):
        seg = self._seg(lane1_items={"iron-plate": 4}, lane2_items={"iron-plate": 2})
        self.assertEqual(seg.items["iron-plate"], 6)

    def test_items_empty_when_both_lanes_empty(self):
        self.assertEqual(self._seg().items, {})

    def test_carries_item_on_lane1(self):
        seg = self._seg(lane1_items={"iron-plate": 3})
        self.assertTrue(seg.carries("iron-plate"))

    def test_carries_item_on_lane2(self):
        seg = self._seg(lane2_items={"copper-plate": 2})
        self.assertTrue(seg.carries("copper-plate"))

    def test_carries_false_when_absent(self):
        seg = self._seg(lane1_items={"iron-plate": 3})
        self.assertFalse(seg.carries("steel-plate"))

    def test_different_items_on_different_lanes(self):
        seg = self._seg(lane1_items={"iron-plate": 4},
                        lane2_items={"copper-plate": 4})
        self.assertTrue(seg.lane1.carries("iron-plate"))
        self.assertTrue(seg.lane2.carries("copper-plate"))
        self.assertFalse(seg.lane1.carries("copper-plate"))
        self.assertFalse(seg.lane2.carries("iron-plate"))

    def test_lanes_are_independent_objects(self):
        seg = self._seg(lane1_items={"iron-plate": 3},
                        lane2_items={"copper-plate": 2})
        self.assertIsNot(seg.lane1, seg.lane2)

if __name__ == "__main__":
    unittest.main(verbosity=2)