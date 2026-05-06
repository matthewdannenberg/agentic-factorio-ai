"""
tests/test_core_dataclasses.py

Tests for all core dataclasses — WorldState, Goal, RewardSpec, AuditReport,
Priority, Action types, and AgentState transitions.

No Factorio, no RCON, no LLM. Pure Python stdlib only.
Run with:  python tests/test_core_dataclasses.py
       or: python -m unittest tests.test_core_dataclasses -v
"""

from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.state import (
    Position, Direction, EntityStatus, ResourceName, ResourceType,
    InventorySlot, Inventory, EntityState, ResourcePatch, GroundItem,
    ResearchState, PowerGrid, LogisticsState,
    DamagedEntity, DestroyedEntity, ThreatState, PlayerState, WorldState,
)
from planning.goal import Priority, GoalStatus, RewardSpec, Goal, make_goal
from agent.examiner.audit_report import (
    AuditMode, AnomalySeverity,
    StarvedEntity, Anomaly, BoundingBox, BlueprintCandidate,
    DamagedEntityRecord, DestroyedEntityRecord, AuditReport,
)
from agent.state_machine import AgentState, assert_valid_transition
from bridge.actions import (
    Action, ActionCategory, ALL_ACTION_TYPES, ACTIONS_BY_CATEGORY, actions_for_context,
    MoveTo, StopMovement,
    MineResource, MineEntity,
    CraftItem,
    PlaceEntity, SetRecipe, SetFilter, ApplyBlueprint,
    TransferItems,
    SetResearchQueue,
    EquipArmor, UseItem,
    EnterVehicle, ExitVehicle, DriveVehicle,
    SelectWeapon, ShootAt, StopShooting,
    Wait, NoOp,
)


# ---------------------------------------------------------------------------
# world/state.py
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# planning/goal.py
# ---------------------------------------------------------------------------

class TestPriority(unittest.TestCase):
    def test_ordering(self):
        self.assertLess(Priority.BACKGROUND, Priority.NORMAL)
        self.assertLess(Priority.NORMAL, Priority.URGENT)
        self.assertLess(Priority.URGENT, Priority.EMERGENCY)

    def test_int_values(self):
        self.assertEqual(Priority.NORMAL, 1)
        self.assertEqual(Priority.EMERGENCY, 3)


class TestRewardSpec(unittest.TestCase):
    def test_defaults(self):
        s = RewardSpec()
        self.assertEqual(s.success_reward, 1.0)
        self.assertEqual(s.time_discount, 1.0)

    def test_invalid_time_discount_zero(self):
        with self.assertRaises(ValueError):
            RewardSpec(time_discount=0.0)

    def test_invalid_time_discount_over_one(self):
        with self.assertRaises(ValueError):
            RewardSpec(time_discount=1.1)

    def test_invalid_negative_penalty(self):
        with self.assertRaises(ValueError):
            RewardSpec(failure_penalty=-1.0)

    def test_no_decay_at_one(self):
        s = RewardSpec(success_reward=1.0, time_discount=1.0)
        self.assertEqual(s.discounted_success_reward(99999), 1.0)

    def test_decay_reduces_reward(self):
        s = RewardSpec(time_discount=0.999)
        self.assertLess(s.discounted_success_reward(1000), 1.0)
        self.assertEqual(s.discounted_success_reward(0), 1.0)


class TestGoal(unittest.TestCase):
    def _g(self, priority=Priority.NORMAL) -> Goal:
        return make_goal(
            description="Build 50 iron gears",
            success_condition="inventory_count('iron-gear-wheel') >= 50",
            failure_condition="game_time_seconds > 600",
            priority=priority,
        )

    def test_auto_id(self):
        g = self._g()
        self.assertEqual(len(g.id), 36)

    def test_unique_ids(self):
        self.assertNotEqual(self._g().id, self._g().id)

    def test_default_status_pending(self):
        self.assertEqual(self._g().status, GoalStatus.PENDING)

    def test_activate(self):
        g = self._g()
        g.activate(tick=100)
        self.assertEqual(g.status, GoalStatus.ACTIVE)
        self.assertEqual(g.created_at, 100)

    def test_created_at_preserved_on_resume(self):
        g = self._g()
        g.activate(tick=100)
        g.suspend()
        g.activate(tick=200)
        self.assertEqual(g.created_at, 100)

    def test_suspend_requires_active(self):
        with self.assertRaises(RuntimeError):
            self._g().suspend()

    def test_complete(self):
        g = self._g()
        g.activate(0)
        g.complete(300)
        self.assertEqual(g.status, GoalStatus.COMPLETE)
        self.assertEqual(g.resolved_at, 300)
        self.assertTrue(g.is_terminal)

    def test_fail(self):
        g = self._g()
        g.activate(0)
        g.fail(400)
        self.assertEqual(g.status, GoalStatus.FAILED)
        self.assertTrue(g.is_terminal)

    def test_double_activate_raises(self):
        g = self._g()
        g.activate(0)
        with self.assertRaises(RuntimeError):
            g.activate(10)

    def test_priority_ordering(self):
        self.assertGreater(
            self._g(Priority.EMERGENCY).priority,
            self._g(Priority.BACKGROUND).priority,
        )

    def test_is_active(self):
        g = self._g()
        self.assertFalse(g.is_active)
        g.activate(0)
        self.assertTrue(g.is_active)

    def test_make_goal_with_milestone(self):
        g = make_goal(
            "test", "True", "False",
            milestone_rewards={"coal >= 10": 0.1},
            parent_id="abc",
        )
        self.assertEqual(len(g.reward_spec.milestone_rewards), 1)
        self.assertEqual(g.parent_id, "abc")


# ---------------------------------------------------------------------------
# agent/examiner/audit_report.py
# ---------------------------------------------------------------------------

class TestAuditReport(unittest.TestCase):
    def _mech(self, tick=0) -> AuditReport:
        return AuditReport(
            tick=tick,
            mode=AuditMode.MECHANICAL,
            observation_ticks=600,
            starved_entities=[StarvedEntity(1, "machine-a", 0, 0, ["iron-plate"])],
            power_headroom_kw=-50.0,
            power_satisfaction=0.8,
            production_rates={"iron-plate": 30.0},
            anomalies=[
                Anomaly(AnomalySeverity.CRITICAL, "brownout"),
                Anomaly(AnomalySeverity.INFO, "low iron"),
            ],
        )

    def test_is_brownout(self):
        self.assertTrue(self._mech().is_brownout)

    def test_not_brownout(self):
        r = AuditReport(tick=0, mode=AuditMode.MECHANICAL, power_satisfaction=1.0)
        self.assertFalse(r.is_brownout)

    def test_has_critical(self):
        self.assertTrue(self._mech().has_critical_anomalies)

    def test_no_critical(self):
        r = AuditReport(tick=0, mode=AuditMode.MECHANICAL,
                        anomalies=[Anomaly(AnomalySeverity.INFO, "fine")])
        self.assertFalse(r.has_critical_anomalies)

    def test_production_rate(self):
        r = self._mech()
        self.assertEqual(r.production_rate("iron-plate"), 30.0)
        self.assertEqual(r.production_rate("steel-plate"), 0.0)

    def test_summary_has_mode(self):
        self.assertIn("MECHANICAL", self._mech().summary())

    def test_merge_sums_ticks(self):
        r1 = AuditReport(tick=0, mode=AuditMode.MECHANICAL, observation_ticks=300,
                         production_rates={"iron-plate": 20.0}, power_satisfaction=1.0)
        r2 = AuditReport(tick=300, mode=AuditMode.MECHANICAL, observation_ticks=300,
                         production_rates={"iron-plate": 40.0}, power_satisfaction=0.9)
        merged = r1.merge(r2)
        self.assertEqual(merged.observation_ticks, 600)
        self.assertAlmostEqual(merged.production_rate("iron-plate"), 30.0)
        self.assertAlmostEqual(merged.power_satisfaction, 0.9)
        self.assertTrue(merged.accumulated)
        self.assertEqual(merged.tick, 0)

    def test_merge_deduplicates_entities(self):
        shared = StarvedEntity(1, "m", 0, 0, ["coal"])
        r1 = AuditReport(tick=0, mode=AuditMode.MECHANICAL, starved_entities=[shared])
        r2 = AuditReport(tick=60, mode=AuditMode.MECHANICAL, starved_entities=[shared])
        self.assertEqual(len(r1.merge(r2).starved_entities), 1)

    def test_rich_fields_none_in_mechanical(self):
        r = self._mech()
        self.assertIsNone(r.blueprint_candidates)
        self.assertIsNone(r.llm_observations)

    def test_rich_report(self):
        r = AuditReport(
            tick=0, mode=AuditMode.RICH,
            blueprint_candidates=[
                BlueprintCandidate(
                    region_description="smelting col",
                    bounds=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10),
                    performance_metric="60/min",
                    rationale="threshold met",
                )
            ],
            llm_observations="All good.",
        )
        self.assertEqual(len(r.blueprint_candidates), 1)
        self.assertEqual(r.llm_observations, "All good.")

    def test_merge_joins_llm_observations(self):
        r1 = AuditReport(tick=0, mode=AuditMode.RICH, llm_observations="First.")
        r2 = AuditReport(tick=0, mode=AuditMode.RICH, llm_observations="Second.")
        merged = r1.merge(r2)
        self.assertIn("First.", merged.llm_observations)
        self.assertIn("Second.", merged.llm_observations)


class TestBoundingBox(unittest.TestCase):
    def _bb(self) -> BoundingBox:
        return BoundingBox(min_x=0, min_y=0, max_x=20, max_y=10)

    def test_dimensions(self):
        bb = self._bb()
        self.assertEqual(bb.width, 20)
        self.assertEqual(bb.height, 10)
        self.assertEqual(bb.area, 200)

    def test_center(self):
        bb = self._bb()
        self.assertAlmostEqual(bb.center_x, 10.0)
        self.assertAlmostEqual(bb.center_y, 5.0)

    def test_contains(self):
        bb = self._bb()
        self.assertTrue(bb.contains(10, 5))
        self.assertTrue(bb.contains(0, 0))
        self.assertTrue(bb.contains(20, 10))
        self.assertFalse(bb.contains(21, 5))
        self.assertFalse(bb.contains(10, -1))

    def test_frozen(self):
        bb = self._bb()
        with self.assertRaises((AttributeError, TypeError)):
            bb.min_x = 99  # type: ignore

    def test_repr(self):
        self.assertIn("20×10", repr(self._bb()))

    def test_blueprint_candidate_uses_bounding_box(self):
        bc = BlueprintCandidate(
            region_description="copper smelting row",
            bounds=BoundingBox(min_x=-5, min_y=10, max_x=15, max_y=30),
            performance_metric="45 copper-plate/min",
            rationale="Above threshold for 5 min",
        )
        self.assertEqual(bc.bounds.width, 20)
        self.assertEqual(bc.bounds.height, 20)
        # No center_x/y/radius attributes on BlueprintCandidate itself
        self.assertFalse(hasattr(bc, "radius"))
        self.assertFalse(hasattr(bc, "center_x"))


class TestAuditReportDamage(unittest.TestCase):
    def test_damage_fields_empty_by_default(self):
        r = AuditReport(tick=0, mode=AuditMode.MECHANICAL)
        self.assertEqual(r.damaged_entities, [])
        self.assertEqual(r.destroyed_entities, [])
        self.assertFalse(r.has_structural_damage)

    def test_has_structural_damage_damaged(self):
        r = AuditReport(
            tick=0, mode=AuditMode.MECHANICAL,
            damaged_entities=[
                DamagedEntityRecord(1, "stone-wall", 50.0, 50.0, 0.3)
            ],
        )
        self.assertTrue(r.has_structural_damage)

    def test_has_structural_damage_destroyed(self):
        r = AuditReport(
            tick=0, mode=AuditMode.MECHANICAL,
            destroyed_entities=[
                DestroyedEntityRecord("gun-turret", 60.0, 60.0, 1750, "biter")
            ],
        )
        self.assertTrue(r.has_structural_damage)

    def test_summary_includes_damage_counts(self):
        r = AuditReport(
            tick=0, mode=AuditMode.MECHANICAL,
            damaged_entities=[DamagedEntityRecord(1, "wall", 0, 0, 0.5)],
            destroyed_entities=[DestroyedEntityRecord("turret", 0, 0, 100, "biter")],
        )
        s = r.summary()
        self.assertIn("damaged=1", s)
        self.assertIn("destroyed=1", s)

    def test_merge_unions_damaged_by_entity_id(self):
        r1 = AuditReport(tick=0, mode=AuditMode.MECHANICAL,
                         damaged_entities=[DamagedEntityRecord(1, "wall-a", 0, 0, 0.8)])
        r2 = AuditReport(tick=60, mode=AuditMode.MECHANICAL,
                         damaged_entities=[
                             DamagedEntityRecord(1, "wall-a", 0, 0, 0.6),  # same id
                             DamagedEntityRecord(2, "wall-b", 5, 5, 0.4),  # new
                         ])
        merged = r1.merge(r2)
        self.assertEqual(len(merged.damaged_entities), 2)

    def test_merge_unions_destroyed_deduplicates(self):
        rec = DestroyedEntityRecord("turret", 10.0, 10.0, 500, "biter")
        r1 = AuditReport(tick=0, mode=AuditMode.MECHANICAL, destroyed_entities=[rec])
        r2 = AuditReport(tick=60, mode=AuditMode.MECHANICAL, destroyed_entities=[rec])
        merged = r1.merge(r2)
        self.assertEqual(len(merged.destroyed_entities), 1)

    def test_merge_appends_distinct_destroyed(self):
        r1 = AuditReport(tick=0, mode=AuditMode.MECHANICAL,
                         destroyed_entities=[
                             DestroyedEntityRecord("turret", 10.0, 10.0, 500, "biter")
                         ])
        r2 = AuditReport(tick=60, mode=AuditMode.MECHANICAL,
                         destroyed_entities=[
                             DestroyedEntityRecord("wall", 20.0, 20.0, 560, "vehicle")
                         ])
        merged = r1.merge(r2)
        self.assertEqual(len(merged.destroyed_entities), 2)

    def test_destroyed_entity_record_vehicle_cause(self):
        # "vehicle" is a valid cause — not biter-gated
        r = DestroyedEntityRecord("wooden-chest", 5.0, 5.0, 300, "vehicle")
        self.assertEqual(r.cause, "vehicle")


# ---------------------------------------------------------------------------
# agent/state_machine.py
# ---------------------------------------------------------------------------

class TestStateMachine(unittest.TestCase):
    def _ok(self, frm, to):
        assert_valid_transition(frm, to)  # must not raise

    def _bad(self, frm, to):
        with self.assertRaises(RuntimeError):
            assert_valid_transition(frm, to)

    def test_planning_to_executing(self):
        self._ok(AgentState.PLANNING, AgentState.EXECUTING)

    def test_executing_to_examining(self):
        self._ok(AgentState.EXECUTING, AgentState.EXAMINING)

    def test_executing_to_planning_emergency(self):
        self._ok(AgentState.EXECUTING, AgentState.PLANNING)

    def test_examining_to_planning(self):
        self._ok(AgentState.EXAMINING, AgentState.PLANNING)

    def test_examining_to_waiting(self):
        self._ok(AgentState.EXAMINING, AgentState.WAITING)

    def test_waiting_to_examining(self):
        self._ok(AgentState.WAITING, AgentState.EXAMINING)

    def test_planning_to_waiting_invalid(self):
        self._bad(AgentState.PLANNING, AgentState.WAITING)

    def test_waiting_to_executing_invalid(self):
        self._bad(AgentState.WAITING, AgentState.EXECUTING)

    def test_waiting_to_planning_invalid(self):
        self._bad(AgentState.WAITING, AgentState.PLANNING)


# ---------------------------------------------------------------------------
# bridge/actions.py
# ---------------------------------------------------------------------------

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

    def test_21_action_types_total(self):
        self.assertEqual(len(ALL_ACTION_TYPES), 21)


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

    def test_stop_shooting_category(self):
        self.assertEqual(StopShooting().category, ActionCategory.COMBAT)


if __name__ == "__main__":
    unittest.main(verbosity=2)