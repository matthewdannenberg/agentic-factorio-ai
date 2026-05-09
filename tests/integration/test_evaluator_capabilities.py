"""
tests/integration/test_evaluator_capabilities.py

Integration tests: RewardEvaluator × WorldState capability matrix.

These tests exercise the full path from a WorldState snapshot through the
RewardEvaluator eval namespace to a concrete EvaluationResult.  They are
integration tests (not unit tests) because they couple two modules —
world/state.py and planning/reward_evaluator.py — and verify the contract
between them: that every condition category the LLM is expected to write can
actually be expressed, evaluated, and return the correct result.

This file is the authoritative registry of supported condition categories.
If a condition type cannot be exercised here, it cannot reliably be written
as a goal condition.  Adding a new condition type means adding a test here
first.

Category codes used in test names:
    IV  inventory (player item counts)
    EN  entity placement (counts, status, recipe)
    PR  production / logistics (power, belts, inserters)
    RS  research / technology
    RM  resource map (patches found within scan radius)
    TM  time / tick
    DM  damage / destruction records
    CN  connectivity (inserter-to-entity relationships)
    GI  ground items
    PT  production_rate() -- PROXIMAL throughput via ProductionTracker
    ST  staleness() -- evidence-freshness guards for proximal conditions
    SC  scope -- proximal vs non-proximal boundary tests

Run with:  python tests/integration/test_evaluator_capabilities.py
"""

from __future__ import annotations

import unittest

from planning.goal import RewardSpec, make_goal, Priority
from planning.reward_evaluator import RewardEvaluator
from world.production_tracker import ProductionTracker
from world.state import (
    BeltLane, BeltSegment, DamagedEntity, DestroyedEntity,
    EntityState, EntityStatus,
    GroundItem, Inventory, InventorySlot, InserterState,
    LogisticsState, PlayerState, Position, PowerGrid,
    ResourcePatch, ResearchState, WorldState,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ev(success="False", failure="False", state=None, tick=0, start_tick=0):
    """
    Evaluate a condition pair against a WorldState.  Returns EvaluationResult.
    Uses default RewardSpec (success_reward=1.0, no time pressure).
    """
    ev = RewardEvaluator()
    return ev.evaluate_conditions(
        success_condition=success,
        failure_condition=failure,
        spec=RewardSpec(),
        state=state or WorldState(),
        tick=tick,
        start_tick=start_tick,
    )



def _ev_with_tracker(tracker, success="False", failure="False",
                    state=None, tick=0, start_tick=0):
    ev = RewardEvaluator(tracker=tracker)
    return ev.evaluate_conditions(
        success_condition=success,
        failure_condition=failure,
        spec=RewardSpec(),
        state=state or WorldState(),
        tick=tick,
        start_tick=start_tick,
    )


def _entity(entity_id, name, x=0.0, y=0.0,
            status=EntityStatus.WORKING, recipe=None, inventory=None):
    return EntityState(
        entity_id=entity_id, name=name, position=Position(x, y),
        status=status, recipe=recipe, inventory=inventory,
    )


def _inserter(entity_id, x=0.0, y=0.0, active=False,
              pickup_pos=None, drop_pos=None):
    return InserterState(
        entity_id=entity_id, position=Position(x, y), active=active,
        pickup_position=pickup_pos, drop_position=drop_pos,
    )


def _belt(segment_id, x=0.0, y=0.0,
          lane1_items=None, lane1_congested=False,
          lane2_items=None, lane2_congested=False):
    return BeltSegment(
        segment_id=segment_id,
        positions=[Position(x, y)],
        lane1=BeltLane(congested=lane1_congested, items=lane1_items or {}),
        lane2=BeltLane(congested=lane2_congested, items=lane2_items or {}),
    )


# ===========================================================================
# IV — Inventory
# ===========================================================================

class TestIV_Inventory(unittest.TestCase):

    def test_IV_001_player_has_enough_of_item(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot("iron-plate", 50)]),
        ))
        self.assertTrue(_ev("inventory('iron-plate') >= 50", state=ws).success)

    def test_IV_002_multiple_item_types_all_present(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            inventory=Inventory(slots=[
                InventorySlot("iron-plate", 30),
                InventorySlot("copper-plate", 20),
            ]),
        ))
        r = _ev("inventory('iron-plate') >= 20 and inventory('copper-plate') >= 10",
                state=ws)
        self.assertTrue(r.success)

    def test_IV_003_absent_item_count_is_zero(self):
        self.assertTrue(_ev("inventory('iron-plate') == 0").success)

    def test_IV_004_below_threshold_no_success(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot("iron-plate", 5)]),
        ))
        self.assertFalse(_ev("inventory('iron-plate') >= 50", state=ws).success)

    def test_IV_005_sum_of_stacked_slots(self):
        # Two slots of the same item — inventory() must aggregate them
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            inventory=Inventory(slots=[
                InventorySlot("coal", 100),
                InventorySlot("coal", 100),
            ]),
        ))
        self.assertTrue(_ev("inventory('coal') >= 200", state=ws).success)


# ===========================================================================
# EN — Entity placement
# ===========================================================================

class TestEN_EntityPlacement(unittest.TestCase):

    def test_EN_001_count_entities_of_type(self):
        ws = WorldState(entities=[
            _entity(1, "assembling-machine-1"),
            _entity(2, "assembling-machine-1"),
            _entity(3, "assembling-machine-1"),
        ])
        self.assertTrue(_ev("len(entities('assembling-machine-1')) >= 3", state=ws).success)

    def test_EN_002_entity_type_not_yet_placed(self):
        ws = WorldState(entities=[_entity(1, "iron-chest")])
        self.assertFalse(_ev("len(entities('assembling-machine-1')) >= 1", state=ws).success)

    def test_EN_003_entity_working_on_specific_recipe(self):
        ws = WorldState(entities=[
            _entity(1, "assembling-machine-1",
                    status=EntityStatus.WORKING, recipe="iron-gear-wheel"),
        ])
        r = _ev(
            "any(e.recipe == 'iron-gear-wheel' and e.status.value == 'working' "
            "    for e in entities('assembling-machine-1'))",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_EN_004_any_entity_starved_as_failure(self):
        ws = WorldState(entities=[
            _entity(1, "assembling-machine-1", status=EntityStatus.NO_INPUT),
        ])
        r = _ev(failure="any(e.status.value == 'no_input' for e in state.entities)",
                state=ws)
        self.assertTrue(r.failure)

    def test_EN_005_count_working_entities(self):
        ws = WorldState(entities=[
            _entity(1, "assembling-machine-1", status=EntityStatus.WORKING),
            _entity(2, "assembling-machine-1", status=EntityStatus.WORKING),
            _entity(3, "assembling-machine-1", status=EntityStatus.IDLE),
        ])
        r = _ev("sum(1 for e in state.entities if e.status.value == 'working') >= 2",
                state=ws)
        self.assertTrue(r.success)

    def test_EN_006_no_entities_of_type_placed(self):
        self.assertTrue(_ev("len(entities('iron-chest')) == 0").success)

    def test_EN_007_entity_has_recipe_set(self):
        ws = WorldState(entities=[
            _entity(1, "assembling-machine-2", recipe="electronic-circuit"),
        ])
        r = _ev(
            "any(e.recipe == 'electronic-circuit' "
            "    for e in entities('assembling-machine-2'))",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_EN_008_entity_accessible_via_entity_by_id(self):
        ws = WorldState(entities=[_entity(42, "stone-furnace", 10.0, 10.0)])
        r = _ev("entity_by_id(42) is not None and entity_by_id(42).name == 'stone-furnace'",
                state=ws)
        self.assertTrue(r.success)


# ===========================================================================
# PR — Production / Logistics
# ===========================================================================

class TestPR_Production(unittest.TestCase):

    def test_PR_001_power_produced_above_threshold(self):
        ws = WorldState(logistics=LogisticsState(
            power=PowerGrid(produced_kw=500, consumed_kw=300, satisfaction=1.0)
        ))
        self.assertTrue(_ev("power.produced_kw >= 400", state=ws).success)

    def test_PR_002_no_brownout(self):
        ws = WorldState(logistics=LogisticsState(
            power=PowerGrid(produced_kw=500, consumed_kw=300, satisfaction=1.0)
        ))
        self.assertTrue(_ev("not power.is_brownout", state=ws).success)

    def test_PR_003_brownout_as_failure(self):
        ws = WorldState(logistics=LogisticsState(
            power=PowerGrid(produced_kw=100, consumed_kw=200, satisfaction=0.5)
        ))
        self.assertTrue(_ev(failure="power.is_brownout", state=ws).failure)

    def test_PR_004_power_headroom_positive(self):
        ws = WorldState(logistics=LogisticsState(
            power=PowerGrid(produced_kw=600, consumed_kw=400)
        ))
        self.assertTrue(_ev("power.headroom_kw >= 100", state=ws).success)

    def test_PR_005_count_active_inserters(self):
        ws = WorldState(logistics=LogisticsState(inserters={
            1: _inserter(1, active=True),
            2: _inserter(2, active=True),
            3: _inserter(3, active=False),
        }))
        r = _ev("sum(1 for i in logistics.inserters.values() if i.active) >= 2",
                state=ws)
        self.assertTrue(r.success)

    # Belt tests — two-lane API

    def test_PR_006_belt_lane1_congested(self):
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane1_congested=True, lane1_items={"iron-plate": 8}),
        ]))
        self.assertTrue(_ev("any(b.lane1.congested for b in logistics.belts)",
                            state=ws).success)

    def test_PR_007_belt_lane2_congested_lane1_free(self):
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane1_congested=False, lane2_congested=True,
                  lane1_items={"iron-plate": 3}, lane2_items={"iron-plate": 8}),
        ]))
        r = _ev(
            "any(b.lane2.congested and not b.lane1.congested for b in logistics.belts)",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_PR_008_belt_segment_congested_property_either_lane(self):
        # .congested property is True when either lane is congested
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane2_congested=True),
        ]))
        self.assertTrue(_ev("any(b.congested for b in logistics.belts)", state=ws).success)

    def test_PR_009_belt_no_congestion(self):
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane1_congested=False, lane2_congested=False),
        ]))
        self.assertTrue(
            _ev("not any(b.congested for b in logistics.belts)", state=ws).success
        )

    def test_PR_010_belt_specific_item_on_lane1(self):
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane1_items={"iron-plate": 4}),
        ]))
        r = _ev("any(b.lane1.carries('iron-plate') for b in logistics.belts)", state=ws)
        self.assertTrue(r.success)

    def test_PR_011_belt_specific_item_on_lane2(self):
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane2_items={"copper-plate": 2}),
        ]))
        r = _ev("any(b.lane2.carries('copper-plate') for b in logistics.belts)", state=ws)
        self.assertTrue(r.success)

    def test_PR_012_belt_carries_property_checks_both_lanes(self):
        # .carries() spans both lanes — item only on lane2, still detected
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane2_items={"steel-plate": 1}),
        ]))
        r = _ev("any(b.carries('steel-plate') for b in logistics.belts)", state=ws)
        self.assertTrue(r.success)

    def test_PR_013_belt_mixed_items_on_same_lane(self):
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane1_items={"iron-plate": 3, "copper-plate": 2}),
        ]))
        r = _ev(
            "any(b.lane1.carries('iron-plate') and b.lane1.carries('copper-plate') "
            "    for b in logistics.belts)",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_PR_014_belt_different_items_on_different_lanes(self):
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane1_items={"iron-plate": 4}, lane2_items={"copper-plate": 4}),
        ]))
        r = _ev(
            "any(b.lane1.carries('iron-plate') and b.lane2.carries('copper-plate') "
            "    for b in logistics.belts)",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_PR_015_belt_items_property_merges_both_lanes(self):
        # .items combines both lanes; iron-plate on lane1, copper-plate on lane2
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane1_items={"iron-plate": 4}, lane2_items={"copper-plate": 2}),
        ]))
        r = _ev(
            "any('iron-plate' in b.items and 'copper-plate' in b.items "
            "    for b in logistics.belts)",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_PR_016_belt_items_property_sums_same_item_across_lanes(self):
        # Same item on both lanes — .items should sum to 6
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane1_items={"iron-plate": 4}, lane2_items={"iron-plate": 2}),
        ]))
        r = _ev(
            "any(b.items.get('iron-plate', 0) >= 6 for b in logistics.belts)",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_PR_017_belt_lane_is_empty(self):
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane1_items={}, lane2_items={"iron-plate": 2}),
        ]))
        r = _ev("any(b.lane1.is_empty() for b in logistics.belts)", state=ws)
        self.assertTrue(r.success)


# ===========================================================================
# RS — Research / Technology
# ===========================================================================

class TestRS_Research(unittest.TestCase):

    def test_RS_001_technology_unlocked(self):
        ws = WorldState(research=ResearchState(unlocked=["steel-processing"]))
        self.assertTrue(_ev("tech_unlocked('steel-processing')", state=ws).success)

    def test_RS_002_technology_not_unlocked(self):
        self.assertFalse(_ev("tech_unlocked('steel-processing')").success)

    def test_RS_003_multiple_techs_all_unlocked(self):
        ws = WorldState(research=ResearchState(unlocked=["steel-processing", "automation"]))
        r = _ev("tech_unlocked('steel-processing') and tech_unlocked('automation')",
                state=ws)
        self.assertTrue(r.success)

    def test_RS_004_research_in_progress(self):
        ws = WorldState(research=ResearchState(in_progress="advanced-electronics"))
        self.assertTrue(
            _ev("research.in_progress == 'advanced-electronics'", state=ws).success
        )

    def test_RS_005_research_queue_length(self):
        ws = WorldState(research=ResearchState(queued=["automation", "logistics"]))
        self.assertTrue(_ev("len(research.queued) >= 2", state=ws).success)

    def test_RS_006_nothing_in_progress(self):
        ws = WorldState(research=ResearchState(in_progress=None))
        self.assertTrue(_ev("research.in_progress is None", state=ws).success)


# ===========================================================================
# RM — Resource map
# ===========================================================================

class TestRM_ResourceMap(unittest.TestCase):

    def test_RM_001_iron_patch_found(self):
        ws = WorldState(resource_map=[
            ResourcePatch("iron-ore", Position(100, 100), 50000, 30),
        ])
        self.assertTrue(_ev("len(resources_of_type('iron-ore')) >= 1", state=ws).success)

    def test_RM_002_no_coal_found(self):
        ws = WorldState(resource_map=[
            ResourcePatch("iron-ore", Position(100, 100), 50000, 30),
        ])
        self.assertTrue(_ev("len(resources_of_type('coal')) == 0", state=ws).success)

    def test_RM_003_patch_has_minimum_amount(self):
        ws = WorldState(resource_map=[
            ResourcePatch("iron-ore", Position(100, 100), 80000, 30),
        ])
        r = _ev("any(p.amount >= 50000 for p in resources_of_type('iron-ore'))",
                state=ws)
        self.assertTrue(r.success)

    def test_RM_004_all_required_resource_types_found(self):
        ws = WorldState(resource_map=[
            ResourcePatch("iron-ore",   Position(100, 100), 50000, 30),
            ResourcePatch("copper-ore", Position(200, 100), 30000, 20),
            ResourcePatch("coal",       Position(300, 100), 20000, 15),
        ])
        r = _ev(
            "len(resources_of_type('iron-ore')) >= 1 "
            "and len(resources_of_type('copper-ore')) >= 1 "
            "and len(resources_of_type('coal')) >= 1",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_RM_005_patch_size_threshold(self):
        ws = WorldState(resource_map=[
            ResourcePatch("iron-ore", Position(100, 100), 80000, 150),
        ])
        r = _ev("any(p.size >= 100 for p in resources_of_type('iron-ore'))", state=ws)
        self.assertTrue(r.success)


# ===========================================================================
# TM — Time / Tick
# ===========================================================================

class TestTM_Time(unittest.TestCase):

    def test_TM_001_tick_threshold(self):
        self.assertTrue(_ev("tick >= 3600", tick=3600).success)

    def test_TM_002_game_time_in_seconds(self):
        ws = WorldState(tick=3600)
        self.assertTrue(_ev("state.game_time_seconds >= 60", state=ws).success)

    def test_TM_003_tick_not_yet_reached(self):
        self.assertFalse(_ev("tick >= 9000", tick=3600).success)

    def test_TM_004_elapsed_ticks_in_result(self):
        r = _ev(tick=500, start_tick=100)
        self.assertEqual(r.elapsed_ticks, 400)

    def test_TM_005_time_limit_as_failure(self):
        self.assertTrue(_ev(failure="tick > 600", tick=700).failure)

    def test_TM_006_elapsed_uses_result_not_worldstate_tick(self):
        # elapsed_ticks comes from tick - start_tick, not state.tick
        ws = WorldState(tick=0)   # state.tick is 0
        r = _ev("True", state=ws, tick=300, start_tick=100)
        self.assertEqual(r.elapsed_ticks, 200)


# ===========================================================================
# DM — Damage / Destruction
# ===========================================================================

class TestDM_Damage(unittest.TestCase):

    def test_DM_001_no_structural_damage(self):
        self.assertTrue(_ev("not state.has_damage").success)

    def test_DM_002_damage_as_failure(self):
        ws = WorldState(damaged_entities=[
            DamagedEntity(1, "stone-wall", Position(50, 50), 0.4),
        ])
        self.assertTrue(_ev(failure="state.has_damage", state=ws).failure)

    def test_DM_003_no_recent_losses(self):
        self.assertTrue(_ev("len(state.recent_losses) == 0").success)

    def test_DM_004_biter_destruction_in_losses(self):
        ws = WorldState(destroyed_entities=[
            DestroyedEntity("gun-turret", Position(60, 60), 1750, "biter"),
        ])
        r = _ev("any(e.cause == 'biter' for e in state.recent_losses)", state=ws)
        self.assertTrue(r.success)

    def test_DM_005_health_fraction_critical_as_failure(self):
        ws = WorldState(damaged_entities=[
            DamagedEntity(1, "stone-wall", Position(50, 50), 0.2),
        ])
        r = _ev(failure="any(e.health_fraction < 0.5 for e in state.damaged_entities)",
                state=ws)
        self.assertTrue(r.failure)

    def test_DM_006_count_damaged_entities(self):
        ws = WorldState(damaged_entities=[
            DamagedEntity(1, "stone-wall", Position(50, 50), 0.6),
            DamagedEntity(2, "stone-wall", Position(52, 50), 0.3),
        ])
        self.assertTrue(_ev("len(state.damaged_entities) >= 2", state=ws).success)

    def test_DM_007_deconstruct_cause(self):
        ws = WorldState(destroyed_entities=[
            DestroyedEntity("iron-chest", Position(5, 5), 300, "deconstruct"),
        ])
        r = _ev("any(e.cause == 'deconstruct' for e in state.recent_losses)", state=ws)
        self.assertTrue(r.success)


# ===========================================================================
# CN — Connectivity (inserter ↔ entity)
# ===========================================================================

class TestCN_Connectivity(unittest.TestCase):

    def _chest_with_feeder(self):
        """Chest at (5,5) with one inserter taking from it."""
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ins = _inserter(10, 4.0, 5.0,
                        pickup_pos=Position(5.0, 5.0),
                        drop_pos=Position(4.0, 3.0))
        return WorldState(entities=[chest],
                          logistics=LogisticsState(inserters={10: ins}))

    def test_CN_001_inserter_taking_from_specific_entity(self):
        ws = self._chest_with_feeder()
        self.assertTrue(_ev("len(inserters_from(1)) >= 1", state=ws).success)

    def test_CN_002_inserter_delivering_to_specific_entity(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ins = _inserter(10, 4.0, 3.0,
                        pickup_pos=Position(4.0, 1.0),
                        drop_pos=Position(5.0, 5.0))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertTrue(_ev("len(inserters_to(1)) >= 1", state=ws).success)

    def test_CN_003_no_inserter_feeding_entity(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ins = _inserter(10, 0.0, 0.0,
                        pickup_pos=Position(0.0, 0.0),
                        drop_pos=Position(0.0, -1.0))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertTrue(_ev("len(inserters_to(1)) == 0", state=ws).success)

    def test_CN_004_inserters_taking_from_type_count(self):
        chest_a = _entity(1, "iron-chest", 5.0, 5.0)
        chest_b = _entity(2, "iron-chest", 15.0, 5.0)
        ins_a = _inserter(10, 0, 0, pickup_pos=Position(5.0, 5.0))
        ins_b = _inserter(11, 0, 0, pickup_pos=Position(15.0, 5.0))
        ws = WorldState(
            entities=[chest_a, chest_b],
            logistics=LogisticsState(inserters={10: ins_a, 11: ins_b}),
        )
        self.assertTrue(
            _ev("len(inserters_from_type('iron-chest')) >= 2", state=ws).success
        )

    def test_CN_005_active_inserter_taking_from_entity(self):
        chest = _entity(1, "iron-chest", 5.0, 5.0)
        ins = _inserter(10, 4.0, 5.0, active=True,
                        pickup_pos=Position(5.0, 5.0),
                        drop_pos=Position(4.0, 3.0))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertTrue(_ev("any(i.active for i in inserters_from(1))", state=ws).success)

    def test_CN_006_no_inserters_in_world(self):
        ws = WorldState(entities=[_entity(1, "iron-chest", 5.0, 5.0)])
        self.assertTrue(_ev("len(inserters_from(1)) == 0", state=ws).success)

    def test_CN_007_inserters_from_type_none_found(self):
        self.assertTrue(_ev("len(inserters_from_type('steel-chest')) == 0").success)

    def test_CN_008_delivering_to_type_count(self):
        chest = _entity(1, "iron-chest", 5.0, 3.0)
        ins = _inserter(10, 0, 0,
                        pickup_pos=Position(5.0, 5.0),
                        drop_pos=Position(5.0, 3.0))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertTrue(
            _ev("len(inserters_to_type('iron-chest')) >= 1", state=ws).success
        )

    def test_CN_009_both_ends_of_assembler_connected(self):
        """Assembler is both fed by inserters and has its output extracted."""
        assembler = _entity(1, "assembling-machine-1", 5.0, 5.0)
        feeder    = _inserter(10, 4.0, 5.0,
                              pickup_pos=Position(3.0, 5.0),
                              drop_pos=Position(5.0, 5.0))
        extractor = _inserter(11, 6.0, 5.0,
                              pickup_pos=Position(5.0, 5.0),
                              drop_pos=Position(7.0, 5.0))
        ws = WorldState(
            entities=[assembler],
            logistics=LogisticsState(inserters={10: feeder, 11: extractor}),
        )
        r = _ev("len(inserters_to(1)) >= 1 and len(inserters_from(1)) >= 1", state=ws)
        self.assertTrue(r.success)

    def test_CN_010_inserter_to_belt_tile(self):
        """Inserters can deliver to belt tiles — belts appear in state.entities."""
        belt_entity = _entity(1, "transport-belt", 5.0, 5.0)
        ins = _inserter(10, 5.0, 6.0,
                        pickup_pos=Position(5.0, 7.0),
                        drop_pos=Position(5.0, 5.0))
        ws = WorldState(
            entities=[belt_entity],
            logistics=LogisticsState(inserters={10: ins}),
        )
        self.assertTrue(_ev("len(inserters_to(1)) >= 1", state=ws).success)


# ===========================================================================
# GI — Ground items
# ===========================================================================

class TestGI_GroundItems(unittest.TestCase):

    def test_GI_001_ground_item_present(self):
        ws = WorldState(ground_items=[GroundItem("iron-plate", Position(3, 3), 10)])
        r = _ev("any(g.item == 'iron-plate' for g in state.ground_items)", state=ws)
        self.assertTrue(r.success)

    def test_GI_002_no_ground_items(self):
        self.assertTrue(_ev("len(state.ground_items) == 0").success)

    def test_GI_003_ground_item_count_threshold(self):
        ws = WorldState(ground_items=[GroundItem("iron-plate", Position(3, 3), 25)])
        r = _ev(
            "sum(g.count for g in state.ground_items if g.item == 'iron-plate') >= 20",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_GI_004_multiple_item_types_on_ground(self):
        ws = WorldState(ground_items=[
            GroundItem("iron-plate", Position(3, 3), 10),
            GroundItem("copper-plate", Position(4, 3), 5),
        ])
        r = _ev("any(g.item == 'copper-plate' for g in state.ground_items)", state=ws)
        self.assertTrue(r.success)


# ===========================================================================
# Cross-category — compound conditions combining multiple categories
# ===========================================================================

class TestXC_CompoundConditions(unittest.TestCase):
    """
    Conditions that span multiple categories. These prove the evaluator
    namespace is coherent enough for the LLM to write realistic multi-factor
    goal conditions.
    """

    def test_XC_001_inventory_and_tech(self):
        """Have enough iron AND steel is unlocked."""
        ws = WorldState(
            player=PlayerState(
                position=Position(0, 0),
                inventory=Inventory(slots=[InventorySlot("iron-plate", 100)]),
            ),
            research=ResearchState(unlocked=["steel-processing"]),
        )
        r = _ev(
            "inventory('iron-plate') >= 50 and tech_unlocked('steel-processing')",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_XC_002_entities_working_and_no_brownout(self):
        """Factory is running AND power is healthy."""
        ws = WorldState(
            entities=[_entity(1, "assembling-machine-1", status=EntityStatus.WORKING)],
            logistics=LogisticsState(
                power=PowerGrid(produced_kw=200, consumed_kw=100, satisfaction=1.0)
            ),
        )
        r = _ev(
            "len(entities('assembling-machine-1')) >= 1 and not power.is_brownout",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_XC_003_time_and_inventory(self):
        """Goal completes after at least 60s AND having enough items."""
        ws = WorldState(
            player=PlayerState(
                position=Position(0, 0),
                inventory=Inventory(slots=[InventorySlot("iron-gear-wheel", 50)]),
            ),
        )
        r = _ev(
            "tick >= 3600 and inventory('iron-gear-wheel') >= 50",
            state=ws, tick=4000,
        )
        self.assertTrue(r.success)

    def test_XC_004_failure_if_damage_or_brownout(self):
        """Fail if either structural damage or brownout occurs."""
        ws = WorldState(
            damaged_entities=[DamagedEntity(1, "stone-wall", Position(0, 0), 0.5)],
            logistics=LogisticsState(
                power=PowerGrid(produced_kw=500, consumed_kw=300, satisfaction=1.0)
            ),
        )
        r = _ev(failure="state.has_damage or power.is_brownout", state=ws)
        self.assertTrue(r.failure)

    def test_XC_005_connectivity_and_activity(self):
        """Assembler is connected on both sides AND at least one inserter is active."""
        assembler = _entity(1, "assembling-machine-1", 5.0, 5.0)
        feeder = _inserter(10, 4.0, 5.0, active=True,
                           pickup_pos=Position(3.0, 5.0),
                           drop_pos=Position(5.0, 5.0))
        extractor = _inserter(11, 6.0, 5.0, active=False,
                              pickup_pos=Position(5.0, 5.0),
                              drop_pos=Position(7.0, 5.0))
        ws = WorldState(
            entities=[assembler],
            logistics=LogisticsState(inserters={10: feeder, 11: extractor}),
        )
        r = _ev(
            "len(inserters_to(1)) >= 1 "
            "and len(inserters_from(1)) >= 1 "
            "and any(i.active for i in inserters_to(1))",
            state=ws,
        )
        self.assertTrue(r.success)




# ===========================================================================
# PT — production_rate() via ProductionTracker  [PROXIMAL]
# ===========================================================================

class TestPT_ProductionRate(unittest.TestCase):
    """
    production_rate(item) is PROXIMAL — it only reflects entities within the
    current scan radius, aggregated over the tracker window.
    See CONDITION_SCOPE.md for the full discussion.
    """

    def _tracker_with_history(self, item, count_start, count_end,
                               tick_start=0, tick_end=3600):
        """
        Build a tracker that has seen *item* grow from count_start to count_end
        over the given tick range, using the real tracker's entity-inventory
        delta signal.
        """
        tracker = ProductionTracker()
        e_start = EntityState(
            entity_id=1, name="assembling-machine-1",
            position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot(item, count_start)]),
        )
        e_end = EntityState(
            entity_id=1, name="assembling-machine-1",
            position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot(item, count_end)]),
        )
        tracker.update(WorldState(tick=tick_start, entities=[e_start]))
        tracker.update(WorldState(tick=tick_end, entities=[e_end]))
        return tracker

    def test_PT_001_rate_above_threshold(self):
        # 60 items over 60 seconds = 60/min
        tracker = self._tracker_with_history(
            "iron-plate", count_start=0, count_end=60,
            tick_start=0, tick_end=3600,
        )
        ws = WorldState(tick=3600)
        r = _ev_with_tracker(tracker, "production_rate('iron-plate') >= 60.0",
                             state=ws, tick=3600)
        self.assertTrue(r.success)

    def test_PT_002_rate_below_threshold(self):
        tracker = self._tracker_with_history(
            "iron-plate", count_start=0, count_end=30,
            tick_start=0, tick_end=3600,
        )
        ws = WorldState(tick=3600)
        r = _ev_with_tracker(tracker, "production_rate('iron-plate') >= 60.0",
                             state=ws, tick=3600)
        self.assertFalse(r.success)

    def test_PT_003_unseen_item_returns_zero(self):
        tracker = ProductionTracker()
        r = _ev_with_tracker(tracker, "production_rate('steel-plate') == 0.0")
        self.assertTrue(r.success)

    def test_PT_004_no_tracker_returns_zero_not_raises(self):
        r = _ev("production_rate('iron-plate') == 0.0")
        self.assertTrue(r.success)

    def test_PT_005_rate_as_failure_condition(self):
        tracker = self._tracker_with_history(
            "iron-plate", count_start=0, count_end=10,
            tick_start=0, tick_end=3600,
        )
        ws = WorldState(tick=3600)
        r = _ev_with_tracker(tracker,
                             failure="production_rate('iron-plate') < 30.0",
                             state=ws, tick=3600)
        self.assertTrue(r.failure)

    def test_PT_006_rate_used_as_milestone(self):
        tracker = self._tracker_with_history(
            "copper-plate", count_start=0, count_end=120,
            tick_start=0, tick_end=3600,
        )
        spec = RewardSpec(milestone_rewards={
            "production_rate('copper-plate') >= 100.0": 0.3
        })
        ev = RewardEvaluator(tracker=tracker)
        ws = WorldState(tick=3600)
        result = ev.evaluate_conditions(
            success_condition="False",
            failure_condition="False",
            spec=spec,
            state=ws,
            tick=3600,
            start_tick=0,
        )
        self.assertIn("production_rate('copper-plate') >= 100.0",
                      result.milestones_hit)
        self.assertAlmostEqual(result.reward, 0.3)

    def test_PT_007_single_snapshot_returns_zero(self):
        """A tracker with only one snapshot cannot compute a rate."""
        tracker = ProductionTracker()
        e = EntityState(
            entity_id=1, name="assembling-machine-1",
            position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot("iron-plate", 50)]),
        )
        ws = WorldState(tick=60, entities=[e])
        tracker.update(ws)
        # Only one snapshot — rate() requires at least two.
        r = _ev_with_tracker(tracker, "production_rate('iron-plate') == 0.0",
                             state=ws, tick=60)
        self.assertTrue(r.success)


# ===========================================================================
# ST — staleness() freshness guards  [META]
# ===========================================================================

class TestST_Staleness(unittest.TestCase):
    """
    staleness(section) exposes WorldState.section_staleness() in the eval
    namespace so conditions can guard themselves against stale data.
    See CONDITION_SCOPE.md for usage guidance.
    """

    def test_ST_001_fresh_section_staleness_zero(self):
        ws = WorldState(tick=1000, observed_at={"entities": 1000})
        r = _ev("staleness('entities') == 0", state=ws, tick=1000)
        self.assertTrue(r.success)

    def test_ST_002_stale_section_returns_ticks_elapsed(self):
        ws = WorldState(tick=1000, observed_at={"entities": 700})
        r = _ev("staleness('entities') == 300", state=ws, tick=1000)
        self.assertTrue(r.success)

    def test_ST_003_never_observed_section_returns_none(self):
        ws = WorldState(tick=1000)
        r = _ev("staleness('entities') is None", state=ws, tick=1000)
        self.assertTrue(r.success)

    def test_ST_004_guard_blocks_stale_proximal_condition(self):
        # Entities section is 10 minutes stale — guard prevents evaluation.
        ws = WorldState(
            tick=36000,
            observed_at={"entities": 0},
            entities=[],
        )
        r = _ev(
            "staleness('entities') is not None "
            "and staleness('entities') < 300 "
            "and len(entities('assembling-machine-1')) >= 3",
            state=ws, tick=36000,
        )
        # Guard short-circuits; result is False but for the right reason.
        self.assertFalse(r.success)

    def test_ST_005_guard_passes_when_fresh(self):
        ws = WorldState(
            tick=1000,
            observed_at={"entities": 950},
            entities=[
                EntityState(1, "assembling-machine-1", Position(0, 0),
                            status=EntityStatus.WORKING),
                EntityState(2, "assembling-machine-1", Position(5, 0),
                            status=EntityStatus.WORKING),
                EntityState(3, "assembling-machine-1", Position(10, 0),
                            status=EntityStatus.WORKING),
            ],
        )
        r = _ev(
            "staleness('entities') is not None "
            "and staleness('entities') < 300 "
            "and len(entities('assembling-machine-1')) >= 3",
            state=ws, tick=1000,
        )
        self.assertTrue(r.success)

    def test_ST_006_staleness_in_failure_condition(self):
        # Stale data: guard prevents false failure firing.
        ws = WorldState(
            tick=10000,
            observed_at={"entities": 0},
            entities=[],
        )
        r = _ev(
            failure="staleness('entities') is not None "
                    "and staleness('entities') < 600 "
                    "and len(entities('stone-furnace')) == 0",
            state=ws, tick=10000,
        )
        self.assertFalse(r.failure)

    def test_ST_007_staleness_distinguishes_never_seen_from_stale(self):
        ws_never = WorldState(tick=1000, observed_at={})
        ws_stale = WorldState(tick=1000, observed_at={"entities": 0})
        r_never = _ev("staleness('entities') is None",
                      state=ws_never, tick=1000)
        r_stale = _ev("staleness('entities') == 1000",
                      state=ws_stale, tick=1000)
        self.assertTrue(r_never.success)
        self.assertTrue(r_stale.success)


# ===========================================================================
# SC — scope boundary tests (proximal vs non-proximal)
# ===========================================================================

class TestSC_Scope(unittest.TestCase):
    """
    Verifies that the proximal/non-proximal distinction holds as documented
    in CONDITION_SCOPE.md.  These are semantic correctness tests, not just
    capability tests.
    """

    def test_SC_001_inventory_unaffected_by_empty_entity_scan(self):
        ws = WorldState(
            player=PlayerState(
                position=Position(0, 0),
                inventory=Inventory(slots=[InventorySlot("iron-plate", 100)]),
            ),
            entities=[],
            observed_at={},
        )
        self.assertTrue(_ev("inventory('iron-plate') >= 100", state=ws).success)

    def test_SC_002_tech_unlocked_unaffected_by_empty_entity_scan(self):
        ws = WorldState(
            research=ResearchState(unlocked=["automation"]),
            entities=[],
            observed_at={},
        )
        self.assertTrue(_ev("tech_unlocked('automation')", state=ws).success)

    def test_SC_003_resource_map_accumulates_across_visits(self):
        ws = WorldState(resource_map=[
            ResourcePatch("iron-ore", Position(100, 100), 50000, 30),
            ResourcePatch("iron-ore", Position(500, 500), 40000, 25),
            ResourcePatch("iron-ore", Position(900, 900), 30000, 20),
            ResourcePatch("iron-ore", Position(1300, 100), 60000, 35),
        ])
        r = _ev("len(resources_of_type('iron-ore')) >= 4", state=ws)
        self.assertTrue(r.success)

    def test_SC_004_entities_reflects_scan_radius_only(self):
        # Factory has moved offscreen — entities list is now empty.
        ws = WorldState(entities=[], observed_at={"entities": 100})
        r = _ev("len(entities('assembling-machine-1')) >= 1", state=ws)
        self.assertFalse(r.success)

    def test_SC_005_production_rate_zero_when_offscreen(self):
        tracker = ProductionTracker()
        r = _ev_with_tracker(tracker, "production_rate('iron-plate') == 0.0")
        self.assertTrue(r.success)

    def test_SC_006_non_proximal_plus_proximal_guarded(self):
        # Correct pattern: non-proximal goal with guarded proximal milestone.
        ws = WorldState(
            tick=3600,
            observed_at={"entities": 3500},
            player=PlayerState(
                position=Position(0, 0),
                inventory=Inventory(slots=[InventorySlot("iron-gear-wheel", 200)]),
            ),
            entities=[
                EntityState(1, "assembling-machine-1", Position(0, 0),
                            status=EntityStatus.WORKING, recipe="iron-gear-wheel"),
            ],
            research=ResearchState(unlocked=["automation"]),
        )
        # Non-proximal top-level success
        r_success = _ev("inventory('iron-gear-wheel') >= 200", state=ws, tick=3600)
        self.assertTrue(r_success.success)
        # Guarded proximal milestone
        r_milestone = _ev(
            "staleness('entities') is not None "
            "and staleness('entities') < 300 "
            "and any(e.recipe == 'iron-gear-wheel' for e in state.entities)",
            state=ws, tick=3600,
        )
        self.assertTrue(r_milestone.success)

if __name__ == "__main__":
    unittest.main(verbosity=2)