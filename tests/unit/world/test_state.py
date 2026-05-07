"""
tests/unit/world/agent/test_state.py

Tests for world/state.py 

Run with:  python tests/unit/world/agent/test_state.py
"""

from __future__ import annotations

import unittest

from world.state import (
    Position, Direction, EntityStatus, ResourceName, ResourceType,
    InventorySlot, Inventory, EntityState, ResourcePatch, GroundItem,
    ResearchState, PowerGrid, LogisticsState,
    DamagedEntity, DestroyedEntity, ThreatState, PlayerState, WorldState,
)


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


class TestWorldState(unittest.TestCase):
    def _ws(self) -> WorldState:
        return WorldState(
            tick=3600,
            player=PlayerState(
                position=Position(0, 0),
                inventory=Inventory(slots=[InventorySlot("iron-plate", 100)]),
            ),
            entities=[
                EntityState(1, "assembling-machine-1", Position(5, 5),
                            status=EntityStatus.WORKING, recipe="iron-gear-wheel"),
                EntityState(2, "electric-furnace", Position(10, 10),
                            status=EntityStatus.NO_INPUT),
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

    def test_resource_patch_is_string_typed(self):
        # ResourceType is str — patch accepts any string, including mod resources
        patch = ResourcePatch(
            resource_type="se-cryonite",   # hypothetical Space Age resource
            position=Position(100, 200),
            amount=50000,
            size=40,
            observed_at=3600,
        )
        self.assertEqual(patch.resource_type, "se-cryonite")
        self.assertEqual(patch.observed_at, 3600)

    def test_resource_name_constants(self):
        # ResourceName provides vanilla string constants
        self.assertEqual(ResourceName.IRON_ORE, "iron-ore")
        self.assertEqual(ResourceName.CRUDE_OIL, "crude-oil")
        # They are just strings — usable anywhere ResourceType (str) is expected
        patch = ResourcePatch("iron-ore", Position(0, 0), 10000, 20)
        self.assertEqual(patch.resource_type, ResourceName.IRON_ORE)

    def test_resources_of_type_string(self):
        ws = WorldState(resource_map=[
            ResourcePatch("iron-ore", Position(10, 10), 50000, 30),
            ResourcePatch("copper-ore", Position(50, 50), 30000, 20),
            ResourcePatch("iron-ore", Position(200, 200), 10000, 10),
        ])
        iron = ws.resources_of_type("iron-ore")
        self.assertEqual(len(iron), 2)
        copper = ws.resources_of_type("copper-ore")
        self.assertEqual(len(copper), 1)
        self.assertEqual(ws.resources_of_type("coal"), [])

    def test_observed_at_defaults_empty(self):
        ws = WorldState()
        self.assertEqual(ws.observed_at, {})

    def test_section_staleness_never_observed(self):
        ws = WorldState(tick=1000)
        self.assertIsNone(ws.section_staleness("resource_map", 1000))

    def test_section_staleness_fresh(self):
        ws = WorldState(tick=1000, observed_at={"resource_map": 1000})
        self.assertEqual(ws.section_staleness("resource_map", 1000), 0)

    def test_section_staleness_stale(self):
        ws = WorldState(tick=1000, observed_at={"resource_map": 400})
        self.assertEqual(ws.section_staleness("resource_map", 1000), 600)

    def test_section_staleness_multiple_sections(self):
        ws = WorldState(tick=500, observed_at={
            "player": 500,
            "entities": 480,
            "resource_map": 100,
        })
        self.assertEqual(ws.section_staleness("player", 500), 0)
        self.assertEqual(ws.section_staleness("entities", 500), 20)
        self.assertEqual(ws.section_staleness("resource_map", 500), 400)
        self.assertIsNone(ws.section_staleness("threat", 500))

    def test_ground_items_default_empty(self):
        ws = WorldState()
        self.assertEqual(ws.ground_items, [])

    def test_ground_items_populated(self):
        ws = WorldState(ground_items=[
            GroundItem(item="iron-plate", position=Position(5, 5), count=12,
                       observed_at=600, age_ticks=30),
            GroundItem(item="copper-plate", position=Position(6, 5), count=3,
                       observed_at=600, age_ticks=10),
        ])
        self.assertEqual(len(ws.ground_items), 2)
        self.assertEqual(ws.ground_items[0].item, "iron-plate")
        self.assertEqual(ws.ground_items[0].count, 12)
        self.assertEqual(ws.ground_items[0].age_ticks, 30)

    def test_threat_state_has_no_damage_fields(self):
        # ThreatState is now biter-only; damage fields live on WorldState
        threat = ThreatState()
        self.assertFalse(hasattr(threat, "damaged_entities"))
        self.assertFalse(hasattr(threat, "destroyed_entities"))
        self.assertFalse(hasattr(threat, "has_damage"))
        self.assertFalse(hasattr(threat, "destroyed_ttl_ticks"))

    def test_worldstate_damaged_empty_by_default(self):
        ws = WorldState()
        self.assertFalse(ws.has_damage)
        self.assertEqual(ws.damaged_entities, [])
        self.assertEqual(ws.destroyed_entities, [])

    def test_worldstate_damaged_entities_biter_cause(self):
        ws = WorldState(damaged_entities=[
            DamagedEntity(
                entity_id=10, name="stone-wall",
                position=Position(50, 50),
                health_fraction=0.4, observed_at=1800,
            )
        ])
        self.assertTrue(ws.has_damage)
        self.assertEqual(ws.damaged_entities[0].health_fraction, 0.4)

    def test_worldstate_destroyed_entities_vehicle_cause(self):
        # Vehicle collisions produce destroyed entities — not biter-gated
        ws = WorldState(destroyed_entities=[
            DestroyedEntity(
                name="wooden-chest", position=Position(10, 10),
                destroyed_at=600, cause="vehicle",
            )
        ])
        losses = ws.recent_losses
        self.assertEqual(len(losses), 1)
        self.assertEqual(losses[0].cause, "vehicle")

    def test_worldstate_destroyed_entities_biter_cause(self):
        ws = WorldState(destroyed_entities=[
            DestroyedEntity(
                name="gun-turret", position=Position(60, 60),
                destroyed_at=1750, cause="biter",
            )
        ])
        self.assertEqual(ws.recent_losses[0].cause, "biter")

    def test_worldstate_destroyed_entities_deconstruct_cause(self):
        ws = WorldState(destroyed_entities=[
            DestroyedEntity(
                name="iron-chest", position=Position(5, 5),
                destroyed_at=300, cause="deconstruct",
            )
        ])
        self.assertEqual(ws.recent_losses[0].cause, "deconstruct")

    def test_worldstate_destroyed_ttl_default(self):
        self.assertEqual(WorldState().destroyed_ttl_ticks, 18_000)

    def test_worldstate_damage_independent_of_biters(self):
        # Damage fields are populated even when ThreatState is empty (biters off)
        ws = WorldState(
            threat=ThreatState(),   # empty — biters off
            damaged_entities=[
                DamagedEntity(1, "assembling-machine-1", Position(5, 5), 0.7)
            ],
        )
        self.assertTrue(ws.threat.is_empty)
        self.assertTrue(ws.has_damage)


if __name__ == "__main__":
    unittest.main(verbosity=2)