"""
tests/unit/bridge/test_actions.py

Tests for bridge/actions.py 

Run with:  python tests/unit/bridge/test_actions.py
"""

from __future__ import annotations

import unittest

from bridge.actions import (
    Action, ActionCategory, ALL_ACTION_TYPES, ACTIONS_BY_CATEGORY, actions_for_context,
    MoveTo, StopMovement,
    MineResource, MineEntity,
    CraftItem,
    PlaceEntity, SetRecipe, SetFilter, SetSplitterPriority, ApplyBlueprint,
    TransferItems,
    SetResearchQueue,
    EquipArmor, UseItem,
    EnterVehicle, ExitVehicle, DriveVehicle,
    SelectWeapon, ShootAt, StopShooting,
    Wait, NoOp,
)
from world.state import (
    Direction,
    Position
)

class TestActionCategory(unittest.TestCase):
    def test_all_action_types_have_category(self):
        import dataclasses as dc
        for action_type in ALL_ACTION_TYPES:
            fields = {f.name: f for f in dc.fields(action_type)}
            self.assertIn("category", fields,
                          f"{action_type.__name__} missing category field")

    def test_category_registry_covers_all_types(self):
        registered = set()
        for types in ACTIONS_BY_CATEGORY.values():
            registered.update(types)
        for action_type in ALL_ACTION_TYPES:
            self.assertIn(action_type, registered,
                          f"{action_type.__name__} missing from ACTIONS_BY_CATEGORY")

    def test_combat_bucket(self):
        combat = ACTIONS_BY_CATEGORY[ActionCategory.COMBAT]
        self.assertIn(ShootAt, combat)
        self.assertIn(SelectWeapon, combat)
        self.assertIn(StopShooting, combat)

    def test_vehicle_bucket(self):
        vehicle = ACTIONS_BY_CATEGORY[ActionCategory.VEHICLE]
        self.assertIn(EnterVehicle, vehicle)
        self.assertIn(ExitVehicle, vehicle)
        self.assertIn(DriveVehicle, vehicle)

    def test_movement_bucket_excludes_vehicle(self):
        movement = ACTIONS_BY_CATEGORY[ActionCategory.MOVEMENT]
        self.assertIn(MoveTo, movement)
        self.assertNotIn(DriveVehicle, movement)

    def test_all_ten_categories_present(self):
        expected = {
            ActionCategory.MOVEMENT, ActionCategory.MINING,
            ActionCategory.CRAFTING, ActionCategory.BUILDING,
            ActionCategory.INVENTORY, ActionCategory.RESEARCH,
            ActionCategory.PLAYER, ActionCategory.VEHICLE,
            ActionCategory.COMBAT, ActionCategory.META,
        }
        self.assertEqual(set(ACTIONS_BY_CATEGORY.keys()), expected)


class TestActionsForContext(unittest.TestCase):
    def test_default_has_movement_not_vehicle(self):
        ctx = actions_for_context()
        self.assertIn(MoveTo, ctx)
        self.assertNotIn(DriveVehicle, ctx)
        self.assertNotIn(EnterVehicle, ctx)
        self.assertNotIn(ExitVehicle, ctx)

    def test_default_no_combat(self):
        ctx = actions_for_context()
        self.assertNotIn(ShootAt, ctx)
        self.assertNotIn(SelectWeapon, ctx)
        self.assertNotIn(StopShooting, ctx)

    def test_vehicle_has_drive_not_walk(self):
        ctx = actions_for_context(in_vehicle=True)
        self.assertIn(DriveVehicle, ctx)
        self.assertNotIn(MoveTo, ctx)
        self.assertNotIn(StopMovement, ctx)

    def test_vehicle_no_combat_by_default(self):
        ctx = actions_for_context(in_vehicle=True)
        self.assertNotIn(ShootAt, ctx)

    def test_biters_adds_combat(self):
        ctx = actions_for_context(biters_enabled=True)
        self.assertIn(ShootAt, ctx)
        self.assertIn(SelectWeapon, ctx)
        self.assertIn(StopShooting, ctx)

    def test_biters_still_has_movement(self):
        ctx = actions_for_context(biters_enabled=True)
        self.assertIn(MoveTo, ctx)

    def test_vehicle_and_biters_combined(self):
        ctx = actions_for_context(in_vehicle=True, biters_enabled=True)
        self.assertIn(DriveVehicle, ctx)
        self.assertIn(ShootAt, ctx)
        self.assertNotIn(MoveTo, ctx)

    def test_ungated_actions_always_present(self):
        always = [CraftItem, PlaceEntity, TransferItems, SetResearchQueue,
                  EquipArmor, UseItem, Wait, NoOp]
        contexts = [
            actions_for_context(),
            actions_for_context(in_vehicle=True),
            actions_for_context(biters_enabled=True),
            actions_for_context(in_vehicle=True, biters_enabled=True),
        ]
        for ctx in contexts:
            for action_type in always:
                self.assertIn(action_type, ctx,
                              f"{action_type.__name__} missing from context")


class TestActions(unittest.TestCase):
    """Construction, kind, frozen, and field defaults for all action types."""

    def test_move_to_kind(self):
        self.assertEqual(MoveTo(position=Position(5, 10)).kind, "MoveTo")

    def test_action_frozen(self):
        a = MoveTo(position=Position(0, 0))
        with self.assertRaises((AttributeError, TypeError)):
            a.pathfind = False  # type: ignore

    def test_noop_kind(self):
        self.assertEqual(NoOp().kind, "NoOp")

    def test_wait_ticks(self):
        self.assertEqual(Wait(ticks=300).ticks, 300)

    def test_place_entity_default_direction(self):
        a = PlaceEntity(item="iron-chest", position=Position(3, 3))
        self.assertEqual(a.direction, Direction.NORTH)

    def test_apply_blueprint_defaults(self):
        a = ApplyBlueprint(blueprint_string="0eNr...", position=Position(0, 0))
        self.assertFalse(a.force_build)
        self.assertEqual(a.direction, Direction.NORTH)

    def test_transfer_default_direction(self):
        self.assertEqual(
            TransferItems(entity_id=1, item="coal", count=50).direction, "to_entity"
        )

    def test_mine_resource_default_count(self):
        self.assertEqual(
            MineResource(position=Position(0, 0), resource="iron-ore").count, 1
        )

    def test_drive_vehicle_default_pathfind(self):
        self.assertTrue(DriveVehicle(position=Position(100, 100)).pathfind)

    def test_use_item_no_target(self):
        a = UseItem(item="raw-fish")
        self.assertIsNone(a.target_position)

    def test_use_item_with_target(self):
        a = UseItem(item="poison-capsule", target_position=Position(50, 50))
        self.assertEqual(a.target_position, Position(50, 50))

    def test_equip_armor_category(self):
        a = EquipArmor(item="heavy-armor")
        self.assertEqual(a.item, "heavy-armor")
        self.assertEqual(a.category, ActionCategory.PLAYER)

    def test_select_weapon(self):
        a = SelectWeapon(slot=1)
        self.assertEqual(a.slot, 1)
        self.assertEqual(a.category, ActionCategory.COMBAT)

    def test_shoot_at_entity(self):
        a = ShootAt(target_entity_id=42)
        self.assertEqual(a.target_entity_id, 42)
        self.assertIsNone(a.target_position)

    def test_shoot_at_position(self):
        a = ShootAt(target_position=Position(10, 20))
        self.assertIsNone(a.target_entity_id)
        self.assertEqual(a.target_position, Position(10, 20))

    def test_shoot_at_requires_exactly_one_target_neither(self):
        with self.assertRaises(ValueError):
            ShootAt()

    def test_shoot_at_requires_exactly_one_target_both(self):
        with self.assertRaises(ValueError):
            ShootAt(target_entity_id=1, target_position=Position(0, 0))

    def test_enter_vehicle(self):
        a = EnterVehicle(entity_id=7)
        self.assertEqual(a.entity_id, 7)
        self.assertEqual(a.category, ActionCategory.VEHICLE)

    def test_exit_vehicle_category(self):
        self.assertEqual(ExitVehicle().category, ActionCategory.VEHICLE)

    def test_set_research_queue(self):
        a = SetResearchQueue(technologies=["automation", "logistics"])
        self.assertEqual(a.technologies[0], "automation")
        self.assertEqual(a.category, ActionCategory.RESEARCH)

    def test_stop_movement_category(self):
        self.assertEqual(StopMovement().category, ActionCategory.MOVEMENT)

    def test_mine_entity_category(self):
        self.assertEqual(MineEntity(entity_id=5).category, ActionCategory.MINING)

    def test_set_recipe_category(self):
        self.assertEqual(
            SetRecipe(entity_id=3, recipe="iron-gear-wheel").category,
            ActionCategory.BUILDING
        )

    def test_set_filter_category(self):
        self.assertEqual(
            SetFilter(entity_id=4, slot=0, item="coal").category,
            ActionCategory.BUILDING
        )

    def test_splitter_output_priority_only(self):
        a = SetSplitterPriority(entity_id=5, output_priority="left")
        self.assertEqual(a.output_priority, "left")
        self.assertIsNone(a.input_priority)
        self.assertEqual(a.category, ActionCategory.BUILDING)

    def test_splitter_input_priority_only(self):
        a = SetSplitterPriority(entity_id=5, input_priority="right")
        self.assertEqual(a.input_priority, "right")
        self.assertIsNone(a.output_priority)

    def test_splitter_both_priorities(self):
        a = SetSplitterPriority(entity_id=5, input_priority="left", output_priority="right")
        self.assertEqual(a.input_priority, "left")
        self.assertEqual(a.output_priority, "right")

    def test_splitter_none_priority_resets(self):
        a = SetSplitterPriority(entity_id=5, output_priority="none")
        self.assertEqual(a.output_priority, "none")

    def test_splitter_requires_at_least_one_priority(self):
        with self.assertRaises(ValueError):
            SetSplitterPriority(entity_id=5)

    def test_splitter_rejects_invalid_priority_value(self):
        with self.assertRaises(ValueError):
            SetSplitterPriority(entity_id=5, output_priority="center")

    def test_splitter_in_building_bucket(self):
        self.assertIn(SetSplitterPriority, ACTIONS_BY_CATEGORY[ActionCategory.BUILDING])

    def test_stop_shooting_category(self):
        self.assertEqual(StopShooting().category, ActionCategory.COMBAT)


if __name__ == "__main__":
    unittest.main(verbosity=2)