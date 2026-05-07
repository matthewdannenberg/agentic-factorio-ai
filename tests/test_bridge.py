"""
tests/test_bridge.py

Bridge layer tests. All tests run without a live Factorio instance.

Test coverage:
- StateParser: full parse, partial parse, safe defaults, unknown resources,
  destroyed entity causes, pruning of destroyed_entities.
- ActionExecutor: round-trip command generation, recoverable failure, stubs.
- RconClient: mock socket (connection and send/recv protocol).
"""

from __future__ import annotations

import json
import struct
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or tests/ directory.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from bridge.action_executor import ActionExecutor
from bridge.rcon_client import BridgeError, RconClient
from bridge.state_parser import StateParser
from bridge.actions import (
    ApplyBlueprint,
    CraftItem,
    EquipArmor,
    MineEntity,
    MineResource,
    MoveTo,
    NoOp,
    PlaceEntity,
    SelectWeapon,
    SetFilter,
    SetRecipe,
    SetResearchQueue,
    ShootAt,
    StopMovement,
    TransferItems,
    UseItem,
    Wait,
)
from world.state import (
    Direction,
    EntityStatus,
    Position,
    ResourceName,
    WorldState,
)


# ===========================================================================
# StateParser tests
# ===========================================================================

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
        self.assertEqual(patch.resource_type, ResourceName.IRON_ORE)
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

    def test_belt_segment(self):
        data = {"tick": 100, "logistics": {"belts": [{
            "segment_id": 1,
            "positions": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
            "congested": True,
            "item": "iron-plate",
        }]}}
        state = self._parse(data)
        self.assertEqual(len(state.logistics.belts), 1)
        seg = state.logistics.belts[0]
        self.assertTrue(seg.congested)
        self.assertEqual(seg.item, "iron-plate")

    def test_inserter_activity(self):
        data = {"tick": 100, "logistics": {
            "inserter_activity": {"101": 5, "102": 0}
        }}
        state = self._parse(data)
        self.assertEqual(state.logistics.inserter_activity[101], 5)

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
        self.assertIsNone(state.section_staleness("resource_map", 100))

    def test_section_staleness_fresh(self):
        data = {"tick": 1000, "player": {"position": {"x": 0, "y": 0}, "inventory": []}}
        state = self.parser.parse(json.dumps(data), current_tick=1000)
        self.assertEqual(state.section_staleness("player", 1000), 0)

    def test_section_staleness_stale(self):
        data = {"tick": 1000, "player": {"position": {"x": 0, "y": 0}, "inventory": []}}
        state = self.parser.parse(json.dumps(data), current_tick=1000)
        self.assertEqual(state.section_staleness("player", 2000), 1000)

    # --- inventory_count / game_time helpers ---------------------------------

    def test_inventory_count(self):
        data = {"tick": 100, "player": {
            "position": {"x": 0, "y": 0},
            "health": 100.0,
            "inventory": [{"item": "iron-plate", "count": 42}],
            "reachable": [],
        }}
        state = self._parse(data)
        self.assertEqual(state.inventory_count("iron-plate"), 42)
        self.assertEqual(state.inventory_count("copper-plate"), 0)

    def test_game_time_seconds(self):
        state = WorldState(tick=3600)
        self.assertAlmostEqual(state.game_time_seconds, 60.0)


# ===========================================================================
# ActionExecutor tests
# ===========================================================================

class MockRconClient:
    """Fake RCON client that records commands sent and returns canned responses."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self.sent_commands: list[str] = []
        self._default_response = json.dumps({"ok": True})

    def send(self, command: str) -> str:
        self.sent_commands.append(command)
        for prefix, response in self._responses.items():
            if prefix in command:
                return response
        return self._default_response

    def is_connected(self) -> bool:
        return True

    def close(self) -> None:
        pass


class TestActionExecutor(unittest.TestCase):
    def setUp(self):
        self.mock_client = MockRconClient()
        self.executor = ActionExecutor(self.mock_client)

    def _last_command(self) -> str:
        return self.mock_client.sent_commands[-1]

    def _assert_lua(self, substring: str) -> None:
        self.assertIn(substring, self._last_command())

    # --- MOVEMENT -------------------------------------------------------------

    def test_move_to_command(self):
        action = MoveTo(position=Position(10.0, -5.0), pathfind=True)
        result = self.executor.execute(action)
        self.assertTrue(result)
        self._assert_lua("fa.move_to")
        self._assert_lua("10.0")
        self._assert_lua("-5.0")
        self._assert_lua("true")

    def test_move_to_no_pathfind(self):
        action = MoveTo(position=Position(0.0, 0.0), pathfind=False)
        self.executor.execute(action)
        self._assert_lua("false")

    def test_stop_movement_command(self):
        action = StopMovement()
        self.executor.execute(action)
        self._assert_lua("fa.stop_movement")

    # --- MINING ---------------------------------------------------------------

    def test_mine_resource_command(self):
        action = MineResource(position=Position(5.0, 10.0), resource="iron-ore", count=50)
        self.executor.execute(action)
        self._assert_lua("fa.mine_resource")
        self._assert_lua("iron-ore")
        self._assert_lua("50")

    def test_mine_entity_command(self):
        action = MineEntity(entity_id=42)
        self.executor.execute(action)
        self._assert_lua("fa.mine_entity")
        self._assert_lua("42")

    # --- CRAFTING -------------------------------------------------------------

    def test_craft_item_command(self):
        action = CraftItem(recipe="iron-gear-wheel", count=10)
        self.executor.execute(action)
        self._assert_lua("fa.craft_item")
        self._assert_lua("iron-gear-wheel")
        self._assert_lua("10")

    # --- BUILDING -------------------------------------------------------------

    def test_place_entity_command(self):
        action = PlaceEntity(item="stone-furnace", position=Position(3.0, 4.0),
                             direction=Direction.EAST)
        self.executor.execute(action)
        self._assert_lua("fa.place_entity")
        self._assert_lua("stone-furnace")
        self._assert_lua("3.0")
        self._assert_lua("4.0")

    def test_set_recipe_command(self):
        action = SetRecipe(entity_id=7, recipe="copper-cable")
        self.executor.execute(action)
        self._assert_lua("fa.set_recipe")
        self._assert_lua("7")
        self._assert_lua("copper-cable")

    def test_set_filter_command(self):
        action = SetFilter(entity_id=9, slot=0, item="iron-plate")
        self.executor.execute(action)
        self._assert_lua("fa.set_filter")
        self._assert_lua("9")
        self._assert_lua("iron-plate")

    def test_set_filter_clear(self):
        action = SetFilter(entity_id=9, slot=1, item="")
        self.executor.execute(action)
        self._assert_lua("nil")

    def test_apply_blueprint_command(self):
        action = ApplyBlueprint(
            blueprint_string="0eNp...", position=Position(0, 0),
            direction=Direction.NORTH, force_build=False
        )
        self.executor.execute(action)
        self._assert_lua("fa.apply_blueprint")
        self._assert_lua("0eNp...")

    # --- INVENTORY ------------------------------------------------------------

    def test_transfer_items_to_player(self):
        action = TransferItems(entity_id=10, direction="from_entity", item="iron-ore", count=20)
        self.executor.execute(action)
        self._assert_lua("fa.transfer_items")
        self._assert_lua("10")
        self._assert_lua("true")  # from_entity → to_player=true in Lua
        self._assert_lua("iron-ore")

    def test_transfer_items_to_entity(self):
        action = TransferItems(entity_id=10, direction="to_entity", item="coal", count=10)
        self.executor.execute(action)
        self._assert_lua("false")

    # --- RESEARCH -------------------------------------------------------------

    def test_set_research_queue_command(self):
        action = SetResearchQueue(technologies=["automation", "logistics"])
        self.executor.execute(action)
        self._assert_lua("fa.set_research_queue")
        self._assert_lua("automation")
        self._assert_lua("logistics")

    # --- PLAYER ---------------------------------------------------------------

    def test_equip_armor_command(self):
        action = EquipArmor(item="heavy-armor")
        self.executor.execute(action)
        self._assert_lua("fa.equip_armor")
        self._assert_lua("heavy-armor")

    def test_use_item_with_position(self):
        action = UseItem(item="raw-fish", target_position=Position(1.0, 2.0))
        self.executor.execute(action)
        self._assert_lua("fa.use_item")
        self._assert_lua("raw-fish")
        self._assert_lua("1.0")
        self._assert_lua("2.0")

    def test_use_item_without_position(self):
        action = UseItem(item="raw-fish", target_position=None)
        self.executor.execute(action)
        self._assert_lua("nil")

    # --- META -----------------------------------------------------------------

    def test_noop_returns_true(self):
        result = self.executor.execute(NoOp())
        self.assertTrue(result)
        self.assertEqual(len(self.mock_client.sent_commands), 0)

    def test_wait_sleeps_and_returns_true(self):
        import unittest.mock
        with unittest.mock.patch("time.sleep") as mock_sleep:
            result = self.executor.execute(Wait(ticks=60))
        self.assertTrue(result)
        mock_sleep.assert_called_once()
        args = mock_sleep.call_args[0]
        self.assertAlmostEqual(args[0], 1.0, places=2)  # 60/60 = 1s

    # --- VEHICLE stubs --------------------------------------------------------

    def test_enter_vehicle_raises_not_implemented(self):
        from bridge.actions import EnterVehicle
        action = EnterVehicle(entity_id=5)
        with self.assertRaises(NotImplementedError):
            self.executor.execute(action)

    def test_exit_vehicle_raises_not_implemented(self):
        from bridge.actions import ExitVehicle
        action = ExitVehicle()
        with self.assertRaises(NotImplementedError):
            self.executor.execute(action)

    def test_drive_vehicle_raises_not_implemented(self):
        from bridge.actions import DriveVehicle
        action = DriveVehicle(position=Position(0, 0))
        with self.assertRaises(NotImplementedError):
            self.executor.execute(action)

    # --- COMBAT stubs ---------------------------------------------------------

    def test_select_weapon_raises_not_implemented(self):
        action = SelectWeapon(slot=0)
        with self.assertRaises(NotImplementedError):
            self.executor.execute(action)

    def test_shoot_at_entity_raises_not_implemented(self):
        action = ShootAt(target_entity_id=99)
        with self.assertRaises(NotImplementedError):
            self.executor.execute(action)

    def test_shoot_at_position_raises_not_implemented(self):
        action = ShootAt(target_position=Position(10, 10))
        with self.assertRaises(NotImplementedError):
            self.executor.execute(action)

    # --- recoverable failure --------------------------------------------------

    def test_recoverable_failure_returns_false(self):
        client = MockRconClient({"fa.mine_entity": json.dumps({"ok": False, "reason": "out_of_reach"})})
        executor = ActionExecutor(client)
        result = executor.execute(MineEntity(entity_id=1))
        self.assertFalse(result)

    # --- unrecoverable (non-JSON Lua error) -----------------------------------

    def test_non_json_response_raises_bridge_error(self):
        client = MockRconClient({"fa.craft_item": "Error: something went wrong in Lua"})
        executor = ActionExecutor(client)
        with self.assertRaises(BridgeError):
            executor.execute(CraftItem(recipe="iron-plate", count=1))

    # --- unknown action kind --------------------------------------------------

    def test_unknown_action_kind_raises_bridge_error(self):
        fake_action = MagicMock()
        fake_action.kind = "CompletelyUnknownAction"
        with self.assertRaises(BridgeError):
            self.executor.execute(fake_action)


# ===========================================================================
# RconClient tests (mock socket)
# ===========================================================================

def _make_rcon_packet(req_id: int, packet_type: int, body: str) -> bytes:
    """Build a valid RCON response packet."""
    body_bytes = body.encode("utf-8")
    payload = struct.pack("<ii", req_id, packet_type) + body_bytes + b"\x00\x00"
    return struct.pack("<i", len(payload)) + payload


class FakeSocket:
    """Simulates a socket for RCON testing."""

    def __init__(self, recv_data: bytes) -> None:
        self._recv_buf = bytearray(recv_data)
        self.sent_data = bytearray()
        self._closed = False

    def sendall(self, data: bytes) -> None:
        self.sent_data.extend(data)

    def recv(self, n: int) -> bytes:
        chunk = bytes(self._recv_buf[:n])
        del self._recv_buf[:n]
        return chunk

    def close(self) -> None:
        self._closed = True

    def settimeout(self, t: float) -> None:
        pass

    def connect(self, addr: tuple) -> None:
        pass


class TestRconClient(unittest.TestCase):
    def _make_auth_response(self, req_id: int) -> bytes:
        return _make_rcon_packet(req_id, 2, "")

    def _make_exec_response(self, req_id: int, body: str) -> bytes:
        return _make_rcon_packet(req_id, 0, body)

    def _make_client_with_socket(self, sock: FakeSocket) -> RconClient:
        client = RconClient(
            host="localhost", port=25575, password="test",
            timeout_s=1.0, reconnect_attempts=1, reconnect_backoff_s=0.0,
        )
        with patch("socket.socket") as mock_socket_cls:
            mock_socket_cls.return_value = sock
            client.connect()
        return client

    def _make_response_sequence(self, *packets: bytes) -> bytes:
        return b"".join(packets)

    def test_connect_and_is_connected(self):
        # Auth: client sends req_id=1 type 3; server replies with req_id=1 type 2.
        auth_resp = self._make_auth_response(1)
        sock = FakeSocket(auth_resp)
        client = self._make_client_with_socket(sock)
        self.assertTrue(client.is_connected())

    def test_send_returns_response_body(self):
        # Sequence: auth(req=1) then exec response(req=2, body='{"ok":true}')
        auth_resp = self._make_auth_response(1)
        exec_resp = self._make_exec_response(2, '{"ok":true}')
        sock = FakeSocket(auth_resp + exec_resp)
        client = self._make_client_with_socket(sock)
        result = client.send("/c fa.get_tick()")
        self.assertEqual(result.strip(), '{"ok":true}')

    def test_auth_failure_raises_bridge_error(self):
        # Server responds with id=-1 to signal auth failure.
        auth_fail = _make_rcon_packet(-1, 2, "")
        sock = FakeSocket(auth_fail)
        with patch("socket.socket") as mock_socket_cls:
            mock_socket_cls.return_value = sock
            client = RconClient(
                host="localhost", port=25575, password="wrong",
                timeout_s=1.0, reconnect_attempts=1, reconnect_backoff_s=0.0,
            )
            with self.assertRaises(BridgeError):
                client.connect()

    def test_close_marks_disconnected(self):
        auth_resp = self._make_auth_response(1)
        sock = FakeSocket(auth_resp)
        client = self._make_client_with_socket(sock)
        self.assertTrue(client.is_connected())
        client.close()
        self.assertFalse(client.is_connected())

    def test_thread_safety_multiple_sends(self):
        """Multiple threads can call send() without corruption."""
        # This test just checks no exceptions are raised — true socket
        # interleaving can't be simulated without a real server.
        results: list[bool] = []
        errors: list[Exception] = []

        def send_worker(executor: ActionExecutor) -> None:
            try:
                r = executor.execute(NoOp())
                results.append(r)
            except Exception as e:
                errors.append(e)

        mock_client = MockRconClient()
        executor = ActionExecutor(mock_client)
        threads = [threading.Thread(target=send_worker, args=(executor,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)


# ===========================================================================
# Integration: StateParser × existing WorldState API
# ===========================================================================

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
    unittest.main()
