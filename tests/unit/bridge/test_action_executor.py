"""
tests/unit/bridge/test_action_executor.py

Tests for bridge/actions.py and bridge/action_executor.py

Run with:  python tests/unit/bridge/test_action_executor.py
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

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
from bridge.action_executor import ActionExecutor
from bridge.rcon_client import BridgeError
from tests.fixtures import MockRconClient
from world.state import (
    Direction,
    Position
)


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



if __name__ == "__main__":
    unittest.main(verbosity=2)