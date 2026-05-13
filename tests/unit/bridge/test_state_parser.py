"""
tests/unit/bridge/test_state_parser.py

Tests for bridge/state_parser.py 

Run with:  python tests/unit/bridge/test_state_parser.py
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import json

from bridge.state_parser import StateParser
from world.query import WorldQuery
from world.state import (
    BeltLane,
    Direction,
    EntityStatus,
    InserterState,
    Position,
    WorldState,
)

class TestStateParserFullParse(unittest.TestCase):
    def setUp(self):
        self.parser = StateParser()

    def _parse(self, data: dict) -> WorldState:
        return self.parser.parse(json.dumps(data), current_tick=data.get("tick", 100))

    # --- tick -----------------------------------------------------------------

    def test_tick_from_payload(self):
        state = self._parse({"tick": 3600})
        self.assertEqual(state.tick, 3600)

    def test_tick_from_argument_when_not_in_payload(self):
        state = self.parser.parse(json.dumps({}), current_tick=99)
        self.assertEqual(state.tick, 99)

    # --- player ---------------------------------------------------------------

    def test_player_position(self):
        data = {"tick": 100, "player": {"position": {"x": 10.5, "y": -3.0},
                                         "health": 80.0, "inventory": [], "reachable": []}}
        state = self._parse(data)
        self.assertAlmostEqual(state.player.position.x, 10.5)
        self.assertAlmostEqual(state.player.position.y, -3.0)
        self.assertAlmostEqual(state.player.health, 80.0)

    def test_player_inventory(self):
        data = {"tick": 100, "player": {
            "position": {"x": 0, "y": 0},
            "health": 100.0,
            "inventory": [
                {"item": "iron-plate", "count": 50},
                {"item": "copper-plate", "count": 25},
            ],
            "reachable": [1, 2],
        }}
        state = self._parse(data)
        self.assertEqual(state.player.inventory.count("iron-plate"), 50)
        self.assertEqual(state.player.inventory.count("copper-plate"), 25)
        self.assertEqual(state.player.reachable, [1, 2])

    def test_player_section_stamps_observed_at(self):
        data = {"tick": 200, "player": {"position": {"x": 0, "y": 0}}}
        state = self._parse(data)
        self.assertEqual(state.observed_at.get("player"), 200)

    # --- entities -------------------------------------------------------------

    def test_entity_basic_fields(self):
        data = {"tick": 100, "entities": [{
            "unit_number": 42,
            "name": "assembling-machine-2",
            "position": {"x": 5.0, "y": -5.0},
            "direction": 2,
            "status": "working",
            "recipe": "iron-gear-wheel",
            "energy": 90000.0,
        }]}
        state = self._parse(data)
        self.assertEqual(len(state.entities), 1)
        e = state.entities[0]
        self.assertEqual(e.entity_id, 42)
        self.assertEqual(e.name, "assembling-machine-2")
        self.assertAlmostEqual(e.position.x, 5.0)
        self.assertEqual(e.direction, Direction.EAST)
        self.assertEqual(e.status, EntityStatus.WORKING)
        self.assertEqual(e.recipe, "iron-gear-wheel")
        self.assertAlmostEqual(e.energy, 90000.0)  # raw joules from Lua

    def test_entity_status_no_input(self):
        data = {"tick": 100, "entities": [{
            "unit_number": 1, "name": "assembling-machine-1",
            "position": {"x": 0, "y": 0},
            "status": "item_ingredient_shortage",
        }]}
        state = self._parse(data)
        self.assertEqual(state.entities[0].status, EntityStatus.NO_INPUT)

    def test_entity_status_unknown_falls_back(self):
        data = {"tick": 100, "entities": [{
            "unit_number": 1, "name": "thing",
            "position": {"x": 0, "y": 0},
            "status": "some_future_status",
        }]}
        state = self._parse(data)
        self.assertEqual(state.entities[0].status, EntityStatus.UNKNOWN)

    def test_malformed_entity_skipped(self):
        data = {"tick": 100, "entities": [
            {"bad": "data"},  # missing required fields
            {"unit_number": 7, "name": "stone-furnace",
             "position": {"x": 1, "y": 1}, "status": "idle"},
        ]}
        state = self._parse(data)
        self.assertEqual(len(state.entities), 1)
        self.assertEqual(state.entities[0].entity_id, 7)

    # --- resource_map ---------------------------------------------------------

    def test_resource_patch_vanilla(self):
        data = {"tick": 100, "resource_map": [{
            "resource_type": "iron-ore",
            "position": {"x": 50.0, "y": 30.0},
            "amount": 200000,
            "size": 500,
            "observed_at": 100,
        }]}
        state = self._parse(data)
        self.assertEqual(len(state.resource_map), 1)
        patch = state.resource_map[0]
        # A more general test might also query the KnowledgeBase about resource_type
        self.assertEqual(patch.amount, 200000)
        self.assertEqual(patch.size, 500)

    def test_unknown_resource_name_accepted(self):
        """Parser must NOT reject unknown resource types (Space Age / mods)."""
        data = {"tick": 100, "resource_map": [{
            "resource_type": "se-cryonite",
            "position": {"x": 100.0, "y": 200.0},
            "amount": 50000,
            "size": 100,
        }]}
        state = self._parse(data)
        self.assertEqual(len(state.resource_map), 1)
        self.assertEqual(state.resource_map[0].resource_type, "se-cryonite")

    def test_unknown_resource_registered_in_registry(self):
        """Parser registers unknown resources in the ResourceRegistry when provided."""
        registry = MagicMock()
        parser = StateParser(resource_registry=registry)
        data = {"tick": 100, "resource_map": [{
            "resource_type": "se-cryonite",
            "position": {"x": 0, "y": 0},
            "amount": 1,
            "size": 1,
        }]}
        parser.parse(json.dumps(data), current_tick=100)
        registry.ensure.assert_called_with("se-cryonite")

    # --- ground_items ---------------------------------------------------------

    def test_ground_items(self):
        data = {"tick": 100, "ground_items": [{
            "item": "iron-plate",
            "position": {"x": 1.0, "y": 2.0},
            "count": 30,
            "age_ticks": 120,
        }]}
        state = self._parse(data)
        self.assertEqual(len(state.ground_items), 1)
        gi = state.ground_items[0]
        self.assertEqual(gi.item, "iron-plate")
        self.assertEqual(gi.count, 30)
        self.assertEqual(gi.age_ticks, 120)

    # --- research -------------------------------------------------------------

    def test_research_fields(self):
        data = {"tick": 100, "research": {
            "unlocked": ["automation", "logistics"],
            "in_progress": "steel-processing",
            "queued": ["advanced-material-processing"],
            "science_per_minute": {"automation-science-pack": 12.5},
        }}
        state = self._parse(data)
        r = state.research
        self.assertIn("automation", r.unlocked)
        self.assertEqual(r.in_progress, "steel-processing")
        self.assertEqual(r.queued, ["advanced-material-processing"])
        self.assertAlmostEqual(r.science_per_minute["automation-science-pack"], 12.5)

    def test_research_is_unlocked(self):
        data = {"tick": 100, "research": {"unlocked": ["automation"], "queued": []}}
        state = self._parse(data)
        self.assertTrue(state.research.is_unlocked("automation"))
        self.assertFalse(state.research.is_unlocked("logistics"))

    # --- logistics ------------------------------------------------------------

    def test_power_grid(self):
        data = {"tick": 100, "logistics": {
            "power": {
                "produced_kw": 5000.0,
                "consumed_kw": 3000.0,
                "accumulated_kj": 1000.0,
                "satisfaction": 1.0,
            },
            "belts": [],
            "inserter_activity": {},
        }}
        state = self._parse(data)
        pg = state.logistics.power
        self.assertAlmostEqual(pg.produced_kw, 5000.0)
        self.assertAlmostEqual(pg.consumed_kw, 3000.0)
        self.assertAlmostEqual(pg.headroom_kw, 2000.0)
        self.assertFalse(pg.is_brownout)

    def test_power_brownout(self):
        data = {"tick": 100, "logistics": {
            "power": {"produced_kw": 100.0, "consumed_kw": 100.0,
                      "accumulated_kj": 0.0, "satisfaction": 0.7},
        }}
        state = self._parse(data)
        self.assertTrue(state.logistics.power.is_brownout)

    def test_belt_segment_lane1_congested(self):
        data = {"tick": 100, "logistics": {"belts": [{
            "segment_id": 1,
            "positions": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
            "lane1": {"congested": True,  "items": {"iron-plate": 8}},
            "lane2": {"congested": False, "items": {}},
        }]}}
        state = self._parse(data)
        self.assertEqual(len(state.logistics.belts), 1)
        seg = state.logistics.belts[0]
        self.assertTrue(seg.lane1.congested)
        self.assertFalse(seg.lane2.congested)
        self.assertTrue(seg.congested)          # .congested spans both lanes
        self.assertTrue(seg.lane1.carries("iron-plate"))

    def test_belt_segment_lane2_congested_only(self):
        data = {"tick": 100, "logistics": {"belts": [{
            "segment_id": 1,
            "positions": [{"x": 0, "y": 0}],
            "lane1": {"congested": False, "items": {"copper-plate": 2}},
            "lane2": {"congested": True,  "items": {"copper-plate": 8}},
        }]}}
        seg = self._parse(data).logistics.belts[0]
        self.assertFalse(seg.lane1.congested)
        self.assertTrue(seg.lane2.congested)
        self.assertTrue(seg.congested)

    def test_belt_segment_no_congestion(self):
        data = {"tick": 100, "logistics": {"belts": [{
            "segment_id": 1,
            "positions": [{"x": 0, "y": 0}],
            "lane1": {"congested": False, "items": {}},
            "lane2": {"congested": False, "items": {}},
        }]}}
        self.assertFalse(self._parse(data).logistics.belts[0].congested)

    def test_belt_segment_different_items_per_lane(self):
        data = {"tick": 100, "logistics": {"belts": [{
            "segment_id": 1,
            "positions": [{"x": 0, "y": 0}],
            "lane1": {"congested": False, "items": {"iron-plate": 4}},
            "lane2": {"congested": False, "items": {"copper-plate": 4}},
        }]}}
        seg = self._parse(data).logistics.belts[0]
        self.assertTrue(seg.lane1.carries("iron-plate"))
        self.assertFalse(seg.lane1.carries("copper-plate"))
        self.assertTrue(seg.lane2.carries("copper-plate"))
        self.assertFalse(seg.lane2.carries("iron-plate"))

    def test_belt_segment_mixed_items_same_lane(self):
        """A splitter input tile can have multiple item types on one lane."""
        data = {"tick": 100, "logistics": {"belts": [{
            "segment_id": 1,
            "positions": [{"x": 0, "y": 0}],
            "lane1": {"congested": False, "items": {"iron-plate": 3, "copper-plate": 2}},
            "lane2": {"congested": False, "items": {}},
        }]}}
        seg = self._parse(data).logistics.belts[0]
        self.assertTrue(seg.lane1.carries("iron-plate"))
        self.assertTrue(seg.lane1.carries("copper-plate"))
        self.assertEqual(seg.lane1.total_items(), 5)

    def test_belt_segment_items_property_sums_across_lanes(self):
        """Same item on both lanes: .items should return summed count."""
        data = {"tick": 100, "logistics": {"belts": [{
            "segment_id": 1,
            "positions": [{"x": 0, "y": 0}],
            "lane1": {"congested": False, "items": {"iron-plate": 4}},
            "lane2": {"congested": False, "items": {"iron-plate": 2}},
        }]}}
        seg = self._parse(data).logistics.belts[0]
        self.assertEqual(seg.items["iron-plate"], 6)

    def test_belt_segment_carries_spans_both_lanes(self):
        data = {"tick": 100, "logistics": {"belts": [{
            "segment_id": 1,
            "positions": [{"x": 0, "y": 0}],
            "lane1": {"congested": False, "items": {}},
            "lane2": {"congested": False, "items": {"steel-plate": 1}},
        }]}}
        seg = self._parse(data).logistics.belts[0]
        self.assertTrue(seg.carries("steel-plate"))      # found via lane2
        self.assertFalse(seg.lane1.carries("steel-plate"))

    def test_belt_segment_missing_lane_keys_default_to_empty(self):
        """A segment with no lane1/lane2 keys at all should not crash."""
        data = {"tick": 100, "logistics": {"belts": [{
            "segment_id": 1,
            "positions": [{"x": 0, "y": 0}],
        }]}}
        seg = self._parse(data).logistics.belts[0]
        self.assertTrue(seg.lane1.is_empty())
        self.assertTrue(seg.lane2.is_empty())
        self.assertFalse(seg.congested)

    def test_inserter_activity_new_format_active(self):
        data = {"tick": 100, "logistics": {
            "inserter_activity": {"42": {
                "active": True,
                "pickup_position": {"x": 5.0, "y": 6.0},
                "drop_position":   {"x": 5.0, "y": 4.0},
            }}
        }}
        state = self._parse(data)
        ins = state.logistics.inserters[42]
        self.assertTrue(ins.active)
        self.assertEqual(ins.pickup_position, Position(5.0, 6.0))
        self.assertEqual(ins.drop_position,   Position(5.0, 4.0))

    def test_inserter_activity_new_format_inactive(self):
        data = {"tick": 100, "logistics": {
            "inserter_activity": {"10": {
                "active": False,
                "pickup_position": {"x": 1.0, "y": 2.0},
                "drop_position":   {"x": 1.0, "y": 0.0},
            }}
        }}
        state = self._parse(data)
        self.assertFalse(state.logistics.inserters[10].active)

    def test_inserter_null_drop_position_is_none(self):
        data = {"tick": 100, "logistics": {
            "inserter_activity": {"7": {
                "active": False,
                "pickup_position": {"x": 3.0, "y": 3.0},
                "drop_position": None,
            }}
        }}
        state = self._parse(data)
        self.assertIsNone(state.logistics.inserters[7].drop_position)

    def test_inserter_missing_positions_are_none(self):
        data = {"tick": 100, "logistics": {
            "inserter_activity": {"9": {"active": False}}
        }}
        state = self._parse(data)
        ins = state.logistics.inserters[9]
        self.assertIsNone(ins.pickup_position)
        self.assertIsNone(ins.drop_position)

    def test_inserter_legacy_activity_dict_kept_in_sync(self):
        """inserter_activity (legacy int dict) stays populated alongside inserters."""
        data = {"tick": 100, "logistics": {
            "inserter_activity": {
                "5": {"active": True,  "pickup_position": None, "drop_position": None},
                "6": {"active": False, "pickup_position": None, "drop_position": None},
            }
        }}
        state = self._parse(data)
        self.assertEqual(state.logistics.inserter_activity[5], 1)
        self.assertEqual(state.logistics.inserter_activity[6], 0)

    def test_inserter_entity_id_coerced_to_int(self):
        """JSON object keys are strings; parser must convert to int."""
        data = {"tick": 100, "logistics": {
            "inserter_activity": {"99": {
                "active": True,
                "pickup_position": {"x": 0, "y": 0},
                "drop_position": {"x": 0, "y": 0},
            }}
        }}
        state = self._parse(data)
        self.assertIn(99, state.logistics.inserters)
        self.assertNotIn("99", state.logistics.inserters)

    # --- damaged_entities -----------------------------------------------------

    def test_damaged_entities(self):
        data = {"tick": 100, "damaged_entities": [{
            "entity_id": 55,
            "name": "stone-wall",
            "position": {"x": 10, "y": 10},
            "health_fraction": 0.4,
        }]}
        state = self._parse(data)
        self.assertTrue(state.has_damage)
        d = state.damaged_entities[0]
        self.assertEqual(d.entity_id, 55)
        self.assertAlmostEqual(d.health_fraction, 0.4)

    def test_damaged_entity_health_clamped(self):
        data = {"tick": 100, "damaged_entities": [{
            "entity_id": 1, "name": "wall", "position": {"x": 0, "y": 0},
            "health_fraction": 1.5,  # invalid — should be clamped
        }]}
        state = self._parse(data)
        self.assertLess(state.damaged_entities[0].health_fraction, 1.0)

    # --- destroyed_entities ---------------------------------------------------

    def test_destroyed_entity_cause_vehicle(self):
        data = {"tick": 1000, "destroyed_entities": [{
            "name": "stone-furnace",
            "position": {"x": 5, "y": 5},
            "destroyed_at": 950,
            "cause": "vehicle",
        }]}
        state = self._parse(data)
        self.assertEqual(len(state.destroyed_entities), 1)
        self.assertEqual(state.destroyed_entities[0].cause, "vehicle")

    def test_destroyed_entity_cause_biter(self):
        data = {"tick": 1000, "destroyed_entities": [{
            "name": "assembling-machine-1",
            "position": {"x": 0, "y": 0},
            "destroyed_at": 999,
            "cause": "biter",
        }]}
        state = self._parse(data)
        self.assertEqual(state.destroyed_entities[0].cause, "biter")

    def test_destroyed_entity_cause_unknown_normalised(self):
        data = {"tick": 1000, "destroyed_entities": [{
            "name": "thing",
            "position": {"x": 0, "y": 0},
            "destroyed_at": 999,
            "cause": "some_future_cause",
        }]}
        state = self._parse(data)
        self.assertEqual(state.destroyed_entities[0].cause, "unknown")

    def test_destroyed_entities_pruned_by_ttl(self):
        """Entities older than destroyed_ttl_ticks should be pruned."""
        tick = 100_000
        ttl = 18_000
        data = {
            "tick": tick,
            "destroyed_entities": [
                # Old — should be pruned
                {"name": "old-wall", "position": {"x": 0, "y": 0},
                 "destroyed_at": tick - ttl - 1, "cause": "unknown"},
                # Recent — should survive
                {"name": "new-wall", "position": {"x": 1, "y": 0},
                 "destroyed_at": tick - 100, "cause": "biter"},
            ],
        }
        state = self._parse(data)
        # Only the recent entity survives.
        self.assertEqual(len(state.destroyed_entities), 1)
        self.assertEqual(state.destroyed_entities[0].name, "new-wall")

    def test_destroyed_entities_merge_with_existing(self):
        """parse_partial should merge new events with existing rolling window."""
        parser = StateParser()
        # First parse establishes a destroyed entity.
        raw1 = json.dumps({"tick": 1000, "destroyed_entities": [{
            "name": "wall-1", "position": {"x": 0, "y": 0},
            "destroyed_at": 1000, "cause": "biter",
        }]})
        state = parser.parse(raw1, current_tick=1000)
        self.assertEqual(len(state.destroyed_entities), 1)

        # Second partial update adds another.
        raw2 = json.dumps({"tick": 1100, "destroyed_entities": [{
            "name": "wall-2", "position": {"x": 1, "y": 0},
            "destroyed_at": 1100, "cause": "vehicle",
        }]})
        state = parser.parse_partial(raw2, "destroyed_entities", state)
        self.assertEqual(len(state.destroyed_entities), 2)
        names = {e.name for e in state.destroyed_entities}
        self.assertIn("wall-1", names)
        self.assertIn("wall-2", names)

    # --- threat ---------------------------------------------------------------

    def test_threat_empty_default(self):
        state = self._parse({"tick": 100})
        self.assertTrue(state.threat.is_empty)
        self.assertAlmostEqual(state.threat.evolution_factor, 0.0)

    def test_threat_biter_bases(self):
        data = {"tick": 100, "threat": {
            "biter_bases": [{
                "base_id": 999,
                "position": {"x": 300, "y": 400},
                "size": 12,
                "evolution": 0.3,
            }],
            "pollution_cloud": [],
            "attack_timers": {},
            "evolution_factor": 0.3,
        }}
        state = self._parse(data)
        self.assertFalse(state.threat.is_empty)
        base = state.threat.biter_bases[0]
        self.assertEqual(base.base_id, 999)
        self.assertAlmostEqual(base.evolution, 0.3)

    # --- partial parse --------------------------------------------------------

    def test_partial_parse_only_stamps_present_sections(self):
        parser = StateParser()
        raw = json.dumps({"tick": 500, "player": {
            "position": {"x": 1, "y": 2}, "health": 90.0,
            "inventory": [], "reachable": [],
        }})
        state = WorldState(tick=400)
        state = parser.parse_partial(raw, "player", state)

        # Player section stamped.
        self.assertEqual(state.observed_at.get("player"), 500)
        # Other sections NOT stamped.
        self.assertNotIn("entities", state.observed_at)
        self.assertNotIn("research", state.observed_at)
        self.assertNotIn("logistics", state.observed_at)

    def test_partial_parse_player_only_leaves_safe_defaults(self):
        parser = StateParser()
        raw = json.dumps({"tick": 200, "player": {
            "position": {"x": 0, "y": 0}, "inventory": []
        }})
        state = WorldState(tick=200)
        state = parser.parse_partial(raw, "player", state)
        self.assertEqual(state.entities, [])
        self.assertEqual(state.resource_map, [])
        self.assertEqual(state.ground_items, [])
        self.assertTrue(state.threat.is_empty)

    # --- invalid JSON ---------------------------------------------------------

    def test_invalid_json_returns_safe_default(self):
        state = self.parser.parse("not json at all", current_tick=0)
        self.assertIsInstance(state, WorldState)
        self.assertEqual(state.entities, [])

    def test_empty_string_returns_safe_default(self):
        state = self.parser.parse("", current_tick=0)
        self.assertIsInstance(state, WorldState)

    # --- observed_at completeness --------------------------------------------

    def test_all_sections_stamped_in_full_parse(self):
        full = {
            "tick": 3600,
            "player": {"position": {"x": 0, "y": 0}, "inventory": [], "reachable": []},
            "entities": [],
            "resource_map": [],
            "ground_items": [],
            "research": {"unlocked": [], "queued": []},
            "logistics": {"power": {}, "belts": [], "inserter_activity": {}},
            "damaged_entities": [],
            "destroyed_entities": [],
            "threat": {},
        }
        state = self.parser.parse(json.dumps(full), current_tick=3600)
        for section in ("player", "entities", "resource_map", "ground_items",
                        "research", "logistics", "damaged_entities",
                        "destroyed_entities", "threat"):
            self.assertIn(section, state.observed_at, f"Missing: {section}")
            self.assertEqual(state.observed_at[section], 3600)

    # --- section_staleness ----------------------------------------------------

    def test_section_staleness_never_observed(self):
        state = WorldState(tick=100)
        self.assertIsNone(WorldQuery(state).section_staleness("resource_map", 100))

    def test_section_staleness_fresh(self):
        data = {"tick": 1000, "player": {"position": {"x": 0, "y": 0}, "inventory": []}}
        state = self.parser.parse(json.dumps(data), current_tick=1000)
        self.assertEqual(WorldQuery(state).section_staleness("player", 1000), 0)

    def test_section_staleness_stale(self):
        data = {"tick": 1000, "player": {"position": {"x": 0, "y": 0}, "inventory": []}}
        state = self.parser.parse(json.dumps(data), current_tick=1000)
        self.assertEqual(WorldQuery(state).section_staleness("player", 2000), 1000)

    # --- inventory_count / game_time helpers ---------------------------------

    def test_inventory_count(self):
        data = {"tick": 100, "player": {
            "position": {"x": 0, "y": 0},
            "health": 100.0,
            "inventory": [{"item": "iron-plate", "count": 42}],
            "reachable": [],
        }}
        state = self._parse(data)
        self.assertEqual(WorldQuery(state).inventory_count("iron-plate"), 42)
        self.assertEqual(WorldQuery(state).inventory_count("copper-plate"), 0)

    def test_game_time_seconds(self):
        state = WorldState(tick=3600)
        self.assertAlmostEqual(state.game_time_seconds, 60.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)