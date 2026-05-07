"""
tests/integration/test_StateParser_WorldState.py

Integration tests for StateParser with WorldState

Run with:  python tests/integration/test_StateParser_WorldState.py
"""

from __future__ import annotations

import json
import unittest

from bridge.state_parser import StateParser
from world.state import (
    EntityStatus
)


class TestStateParserWorldStateAPI(unittest.TestCase):
    """Verify that parser output satisfies the WorldState accessor contracts."""

    def setUp(self):
        self.parser = StateParser()

    def _full_state_json(self) -> str:
        return json.dumps({
            "tick": 7200,
            "player": {
                "position": {"x": 12.5, "y": -8.0},
                "health": 65.0,
                "inventory": [
                    {"item": "iron-plate", "count": 100},
                    {"item": "iron-plate", "count": 50},  # two slots same item
                    {"item": "coal", "count": 30},
                ],
                "reachable": [3, 7],
            },
            "entities": [
                {"unit_number": 3, "name": "stone-furnace",
                 "position": {"x": 10, "y": -8}, "status": "working"},
                {"unit_number": 7, "name": "burner-mining-drill",
                 "position": {"x": 14, "y": -8}, "status": "no_minable_resources"},
            ],
            "resource_map": [
                {"resource_type": "iron-ore", "position": {"x": 50, "y": 0},
                 "amount": 150000, "size": 300, "observed_at": 7200},
                {"resource_type": "coal", "position": {"x": -20, "y": 10},
                 "amount": 80000, "size": 150, "observed_at": 7100},
            ],
            "research": {"unlocked": ["automation"], "queued": [], "in_progress": None},
            "logistics": {"power": {"produced_kw": 900.0, "consumed_kw": 750.0,
                                    "accumulated_kj": 0.0, "satisfaction": 1.0}},
            "damaged_entities": [],
            "destroyed_entities": [],
            "threat": {},
        })

    def test_entity_by_id(self):
        state = self.parser.parse(self._full_state_json(), current_tick=7200)
        e = state.entity_by_id(3)
        self.assertIsNotNone(e)
        self.assertEqual(e.name, "stone-furnace")

    def test_entity_by_id_missing(self):
        state = self.parser.parse(self._full_state_json(), current_tick=7200)
        self.assertIsNone(state.entity_by_id(999))

    def test_entities_by_name(self):
        state = self.parser.parse(self._full_state_json(), current_tick=7200)
        drills = state.entities_by_name("burner-mining-drill")
        self.assertEqual(len(drills), 1)

    def test_entities_by_status(self):
        state = self.parser.parse(self._full_state_json(), current_tick=7200)
        working = state.entities_by_status(EntityStatus.WORKING)
        self.assertEqual(len(working), 1)
        self.assertEqual(working[0].name, "stone-furnace")

    def test_resources_of_type(self):
        state = self.parser.parse(self._full_state_json(), current_tick=7200)
        iron = state.resources_of_type("iron-ore")
        self.assertEqual(len(iron), 1)
        self.assertEqual(iron[0].amount, 150000)

    def test_inventory_count_multi_slot(self):
        """Counts across multiple slots of the same item are summed."""
        state = self.parser.parse(self._full_state_json(), current_tick=7200)
        self.assertEqual(state.inventory_count("iron-plate"), 150)  # 100+50

    def test_power_headroom(self):
        state = self.parser.parse(self._full_state_json(), current_tick=7200)
        self.assertAlmostEqual(state.logistics.power.headroom_kw, 150.0)

    def test_recent_losses_empty(self):
        state = self.parser.parse(self._full_state_json(), current_tick=7200)
        self.assertEqual(state.recent_losses, [])

    def test_has_damage_false(self):
        state = self.parser.parse(self._full_state_json(), current_tick=7200)
        self.assertFalse(state.has_damage)


if __name__ == "__main__":
    unittest.main(verbosity=2)