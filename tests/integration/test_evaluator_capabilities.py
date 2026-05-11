"""
tests/integration/test_evaluator_capabilities.py

Integration tests: RewardEvaluator x WorldQuery capability matrix.

RewardEvaluator now receives a WorldQuery rather than a WorldState.
All helpers wrap WorldState in WorldQuery before passing to the evaluator.

Category codes:
    IV  inventory (player item counts)
    EN  entity placement (counts, status, recipe)
    PR  production / logistics (power, belts, inserters)
    RS  research / technology
    RM  resource map (patches found within scan radius)
    TM  time / tick
    DM  damage / destruction records
    CN  connectivity (inserter-to-entity relationships)
    GI  ground items
    EX  exploration (charted area)  -- NON-PROXIMAL
    PT  production_rate() via ProductionTracker  -- PROXIMAL
    ST  staleness() freshness guards  -- META
    SC  scope boundary tests (proximal vs non-proximal)
    XC  compound conditions (multi-category)

Run with:  python tests/integration/test_evaluator_capabilities.py
"""

from __future__ import annotations

import unittest

from planning.goal import RewardSpec, make_goal, Priority
from planning.reward_evaluator import RewardEvaluator
from world.production_tracker import ProductionTracker
from world.query import WorldQuery
from world.state import (
    BeltLane, BeltSegment, DamagedEntity, DestroyedEntity,
    EntityState, EntityStatus, ExplorationState,
    GroundItem, Inventory, InventorySlot, InserterState,
    LogisticsState, PlayerState, Position, PowerGrid,
    ResourcePatch, ResearchState, WorldState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wq(state: WorldState = None) -> WorldQuery:
    return WorldQuery(state or WorldState())


def _ev(success="False", failure="False", state=None, tick=0, start_tick=0):
    ev = RewardEvaluator()
    return ev.evaluate_conditions(
        success_condition=success,
        failure_condition=failure,
        spec=RewardSpec(),
        wq=_wq(state),
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
        wq=_wq(state),
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
        segment_id=segment_id, positions=[Position(x, y)],
        lane1=BeltLane(congested=lane1_congested, items=lane1_items or {}),
        lane2=BeltLane(congested=lane2_congested, items=lane2_items or {}),
    )


def _ws_with_chart(chunks: int) -> WorldState:
    return WorldState(player=PlayerState(
        position=Position(0, 0),
        exploration=ExplorationState(charted_chunks=chunks),
    ))


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

    def test_IV_002_multiple_item_types(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            inventory=Inventory(slots=[
                InventorySlot("iron-plate", 30),
                InventorySlot("copper-plate", 20),
            ]),
        ))
        r = _ev("inventory('iron-plate') >= 20 and inventory('copper-plate') >= 10", state=ws)
        self.assertTrue(r.success)

    def test_IV_003_absent_item_is_zero(self):
        self.assertTrue(_ev("inventory('iron-plate') == 0").success)

    def test_IV_004_below_threshold_no_success(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot("iron-plate", 5)]),
        ))
        self.assertFalse(_ev("inventory('iron-plate') >= 50", state=ws).success)

    def test_IV_005_stacked_slots_aggregated(self):
        ws = WorldState(player=PlayerState(
            position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot("coal", 100), InventorySlot("coal", 100)]),
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

    def test_EN_002_entity_type_not_placed(self):
        ws = WorldState(entities=[_entity(1, "iron-chest")])
        self.assertFalse(_ev("len(entities('assembling-machine-1')) >= 1", state=ws).success)

    def test_EN_003_entity_working_on_recipe(self):
        ws = WorldState(entities=[
            _entity(1, "assembling-machine-1", status=EntityStatus.WORKING,
                    recipe="iron-gear-wheel"),
        ])
        r = _ev(
            "any(e.recipe == 'iron-gear-wheel' and e.status.value == 'working' "
            "    for e in entities('assembling-machine-1'))",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_EN_004_any_entity_starved_as_failure(self):
        ws = WorldState(entities=[_entity(1, "assembling-machine-1", status=EntityStatus.NO_INPUT)])
        r = _ev(failure="any(e.status.value == 'no_input' for e in state.entities)", state=ws)
        self.assertTrue(r.failure)

    def test_EN_005_count_working_entities(self):
        ws = WorldState(entities=[
            _entity(1, "assembling-machine-1", status=EntityStatus.WORKING),
            _entity(2, "assembling-machine-1", status=EntityStatus.WORKING),
            _entity(3, "assembling-machine-1", status=EntityStatus.IDLE),
        ])
        r = _ev("sum(1 for e in state.entities if e.status.value == 'working') >= 2", state=ws)
        self.assertTrue(r.success)

    def test_EN_006_no_entities_of_type(self):
        self.assertTrue(_ev("len(entities('iron-chest')) == 0").success)


# ===========================================================================
# PR — Production / Logistics
# ===========================================================================

class TestPR_Production(unittest.TestCase):

    def test_PR_001_power_above_threshold(self):
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

    def test_PR_004_power_headroom(self):
        ws = WorldState(logistics=LogisticsState(
            power=PowerGrid(produced_kw=600, consumed_kw=400)
        ))
        self.assertTrue(_ev("power.headroom_kw >= 100", state=ws).success)

    def test_PR_005_active_inserter_count(self):
        ws = WorldState(logistics=LogisticsState(inserters={
            1: InserterState(1, Position(0, 0), active=True),
            2: InserterState(2, Position(0, 0), active=True),
            3: InserterState(3, Position(0, 0), active=False),
        }))
        r = _ev("sum(1 for i in logistics.inserters.values() if i.active) >= 2", state=ws)
        self.assertTrue(r.success)

    def test_PR_006_belt_lane1_congested(self):
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane1_congested=True, lane1_items={"iron-plate": 8}),
        ]))
        self.assertTrue(
            _ev("any(b.lane1.congested for b in logistics.belts)", state=ws).success
        )

    def test_PR_007_belt_different_items_per_lane(self):
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane1_items={"iron-plate": 4}, lane2_items={"copper-plate": 4}),
        ]))
        r = _ev(
            "any(b.lane1.carries('iron-plate') and b.lane2.carries('copper-plate') "
            "    for b in logistics.belts)",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_PR_008_belt_items_property_sums_lanes(self):
        ws = WorldState(logistics=LogisticsState(belts=[
            _belt(1, lane1_items={"iron-plate": 4}, lane2_items={"iron-plate": 2}),
        ]))
        r = _ev("any(b.items.get('iron-plate', 0) >= 6 for b in logistics.belts)", state=ws)
        self.assertTrue(r.success)


# ===========================================================================
# RS — Research
# ===========================================================================

class TestRS_Research(unittest.TestCase):

    def test_RS_001_technology_unlocked(self):
        ws = WorldState(research=ResearchState(unlocked=["steel-processing"]))
        self.assertTrue(_ev("tech_unlocked('steel-processing')", state=ws).success)

    def test_RS_002_not_unlocked(self):
        self.assertFalse(_ev("tech_unlocked('steel-processing')").success)

    def test_RS_003_multiple_techs(self):
        ws = WorldState(research=ResearchState(unlocked=["steel-processing", "automation"]))
        r = _ev("tech_unlocked('steel-processing') and tech_unlocked('automation')", state=ws)
        self.assertTrue(r.success)

    def test_RS_004_in_progress(self):
        ws = WorldState(research=ResearchState(in_progress="advanced-electronics"))
        self.assertTrue(_ev("research.in_progress == 'advanced-electronics'", state=ws).success)

    def test_RS_005_queue_length(self):
        ws = WorldState(research=ResearchState(queued=["automation", "logistics"]))
        self.assertTrue(_ev("len(research.queued) >= 2", state=ws).success)


# ===========================================================================
# RM — Resource map
# ===========================================================================

class TestRM_ResourceMap(unittest.TestCase):

    def test_RM_001_patch_found(self):
        ws = WorldState(resource_map=[ResourcePatch("iron-ore", Position(100, 100), 50000, 30)])
        self.assertTrue(_ev("len(resources_of_type('iron-ore')) >= 1", state=ws).success)

    def test_RM_002_type_not_found(self):
        ws = WorldState(resource_map=[ResourcePatch("iron-ore", Position(100, 100), 50000, 30)])
        self.assertTrue(_ev("len(resources_of_type('coal')) == 0", state=ws).success)

    def test_RM_003_patch_amount_threshold(self):
        ws = WorldState(resource_map=[ResourcePatch("iron-ore", Position(100, 100), 80000, 30)])
        r = _ev("any(p.amount >= 50000 for p in resources_of_type('iron-ore'))", state=ws)
        self.assertTrue(r.success)

    def test_RM_004_all_required_types(self):
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


# ===========================================================================
# TM — Time / Tick
# ===========================================================================

class TestTM_Time(unittest.TestCase):

    def test_TM_001_tick_threshold(self):
        self.assertTrue(_ev("tick >= 3600", tick=3600).success)

    def test_TM_002_game_time_seconds(self):
        ws = WorldState(tick=3600)
        self.assertTrue(_ev("state.game_time_seconds >= 60", state=ws).success)

    def test_TM_003_not_yet_reached(self):
        self.assertFalse(_ev("tick >= 9000", tick=3600).success)

    def test_TM_004_elapsed_ticks(self):
        r = _ev(tick=500, start_tick=100)
        self.assertEqual(r.elapsed_ticks, 400)

    def test_TM_005_time_limit_as_failure(self):
        self.assertTrue(_ev(failure="tick > 600", tick=700).failure)


# ===========================================================================
# DM — Damage / Destruction
# ===========================================================================

class TestDM_Damage(unittest.TestCase):

    def test_DM_001_no_damage(self):
        self.assertTrue(_ev("not state.has_damage").success)

    def test_DM_002_damage_as_failure(self):
        ws = WorldState(damaged_entities=[DamagedEntity(1, "stone-wall", Position(50, 50), 0.4)])
        self.assertTrue(_ev(failure="state.has_damage", state=ws).failure)

    def test_DM_003_biter_kill(self):
        ws = WorldState(destroyed_entities=[
            DestroyedEntity("gun-turret", Position(60, 60), 1750, "biter"),
        ])
        r = _ev("any(e.cause == 'biter' for e in state.recent_losses)", state=ws)
        self.assertTrue(r.success)

    def test_DM_004_health_fraction_critical(self):
        ws = WorldState(damaged_entities=[DamagedEntity(1, "stone-wall", Position(50, 50), 0.2)])
        r = _ev(failure="any(e.health_fraction < 0.5 for e in state.damaged_entities)", state=ws)
        self.assertTrue(r.failure)


# ===========================================================================
# CN — Connectivity
# ===========================================================================

class TestCN_Connectivity(unittest.TestCase):

    def test_CN_001_inserter_taking_from_entity(self):
        chest = _entity(1, "iron-chest", 5, 5)
        ins = _inserter(10, 4, 5, pickup_pos=Position(5, 5), drop_pos=Position(4, 3))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertTrue(_ev("len(inserters_from(1)) >= 1", state=ws).success)

    def test_CN_002_inserter_delivering_to_entity(self):
        chest = _entity(1, "iron-chest", 5, 5)
        ins = _inserter(10, 4, 3, pickup_pos=Position(4, 1), drop_pos=Position(5, 5))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertTrue(_ev("len(inserters_to(1)) >= 1", state=ws).success)

    def test_CN_003_no_inserter_feeding(self):
        chest = _entity(1, "iron-chest", 5, 5)
        ins = _inserter(10, 0, 0, pickup_pos=Position(0, 0), drop_pos=Position(0, -1))
        ws = WorldState(entities=[chest], logistics=LogisticsState(inserters={10: ins}))
        self.assertTrue(_ev("len(inserters_to(1)) == 0", state=ws).success)

    def test_CN_004_type_count(self):
        chest_a = _entity(1, "iron-chest", 5, 5)
        chest_b = _entity(2, "iron-chest", 15, 5)
        ins_a = _inserter(10, 0, 0, pickup_pos=Position(5, 5))
        ins_b = _inserter(11, 0, 0, pickup_pos=Position(15, 5))
        ws = WorldState(
            entities=[chest_a, chest_b],
            logistics=LogisticsState(inserters={10: ins_a, 11: ins_b}),
        )
        self.assertTrue(_ev("len(inserters_from_type('iron-chest')) >= 2", state=ws).success)

    def test_CN_005_both_ends_connected(self):
        assembler = _entity(1, "assembling-machine-1", 5, 5)
        feeder    = _inserter(10, 4, 5, pickup_pos=Position(3, 5), drop_pos=Position(5, 5))
        extractor = _inserter(11, 6, 5, pickup_pos=Position(5, 5), drop_pos=Position(7, 5))
        ws = WorldState(
            entities=[assembler],
            logistics=LogisticsState(inserters={10: feeder, 11: extractor}),
        )
        r = _ev("len(inserters_to(1)) >= 1 and len(inserters_from(1)) >= 1", state=ws)
        self.assertTrue(r.success)

    def test_CN_006_wq_fully_connected_entities(self):
        """Compound query via WorldQuery composable builder in condition string."""
        assembler = _entity(1, "assembling-machine-1", 5, 5, recipe="electronic-circuit")
        feeder    = _inserter(10, 4, 5, pickup_pos=Position(3, 5), drop_pos=Position(5, 5))
        extractor = _inserter(11, 6, 5, pickup_pos=Position(5, 5), drop_pos=Position(7, 5))
        ws = WorldState(
            entities=[assembler],
            logistics=LogisticsState(inserters={10: feeder, 11: extractor}),
        )
        r = _ev("len(wq.fully_connected_entities('electronic-circuit')) >= 1", state=ws)
        self.assertTrue(r.success)


# ===========================================================================
# GI — Ground items
# ===========================================================================

class TestGI_GroundItems(unittest.TestCase):

    def test_GI_001_item_present(self):
        ws = WorldState(ground_items=[GroundItem("iron-plate", Position(3, 3), 10)])
        r = _ev("any(g.item == 'iron-plate' for g in state.ground_items)", state=ws)
        self.assertTrue(r.success)

    def test_GI_002_no_items(self):
        self.assertTrue(_ev("len(state.ground_items) == 0").success)

    def test_GI_003_count_threshold(self):
        ws = WorldState(ground_items=[GroundItem("iron-plate", Position(3, 3), 25)])
        r = _ev(
            "sum(g.count for g in state.ground_items if g.item == 'iron-plate') >= 20",
            state=ws,
        )
        self.assertTrue(r.success)


# ===========================================================================
# EX — Exploration (charted area)  [NON-PROXIMAL]
# ===========================================================================

class TestEX_Exploration(unittest.TestCase):

    def test_EX_001_charted_chunks_threshold(self):
        self.assertTrue(_ev("charted_chunks >= 50", state=_ws_with_chart(50)).success)

    def test_EX_002_charted_chunks_not_met(self):
        self.assertFalse(_ev("charted_chunks >= 50", state=_ws_with_chart(30)).success)

    def test_EX_003_charted_tiles_derived(self):
        self.assertTrue(_ev("charted_tiles == 10240", state=_ws_with_chart(10)).success)

    def test_EX_004_charted_area_km2(self):
        self.assertTrue(_ev("charted_area_km2 >= 1.0", state=_ws_with_chart(1000)).success)

    def test_EX_005_charted_area_not_met(self):
        self.assertFalse(_ev("charted_area_km2 >= 1.0", state=_ws_with_chart(10)).success)

    def test_EX_006_exploration_as_failure(self):
        ws = _ws_with_chart(5)
        r = _ev(failure="tick > 7200 and charted_chunks < 20", state=ws, tick=8000)
        self.assertTrue(r.failure)

    def test_EX_007_not_failure_when_explored_enough(self):
        ws = _ws_with_chart(25)
        r = _ev(failure="tick > 7200 and charted_chunks < 20", state=ws, tick=8000)
        self.assertFalse(r.failure)

    def test_EX_008_exploration_as_milestone(self):
        spec = RewardSpec(milestone_rewards={"charted_chunks >= 25": 0.2})
        ws = _ws_with_chart(30)
        ev = RewardEvaluator()
        result = ev.evaluate_conditions(
            success_condition="False", failure_condition="False",
            spec=spec, wq=_wq(ws), tick=0, start_tick=0,
        )
        self.assertIn("charted_chunks >= 25", result.milestones_hit)
        self.assertAlmostEqual(result.reward, 0.2)

    def test_EX_009_zero_by_default(self):
        self.assertTrue(_ev("charted_chunks == 0").success)

    def test_EX_010_non_proximal_unaffected_by_empty_scan(self):
        ws = WorldState(
            player=PlayerState(position=Position(0, 0),
                               exploration=ExplorationState(charted_chunks=100)),
            entities=[], observed_at={},
        )
        self.assertTrue(_ev("charted_chunks >= 100", state=ws).success)

    def test_EX_011_charted_tiles_expression(self):
        ws = _ws_with_chart(500)
        self.assertTrue(_ev("charted_tiles >= 512000", state=ws).success)

    def test_EX_012_exploration_combined_with_other_non_proximal(self):
        ws = WorldState(
            player=PlayerState(
                position=Position(0, 0),
                inventory=Inventory(slots=[InventorySlot("iron-plate", 50)]),
                exploration=ExplorationState(charted_chunks=30),
            ),
            research=ResearchState(unlocked=["automation"]),
        )
        r = _ev(
            "charted_chunks >= 20 "
            "and inventory('iron-plate') >= 50 "
            "and tech_unlocked('automation')",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_EX_013_progressive_exploration_milestones(self):
        spec = RewardSpec(milestone_rewards={
            "charted_chunks >= 10": 0.1,
            "charted_chunks >= 50": 0.2,
            "charted_chunks >= 100": 0.3,
        })
        ws = _ws_with_chart(75)
        ev = RewardEvaluator()
        result = ev.evaluate_conditions(
            success_condition="charted_chunks >= 200",
            failure_condition="False",
            spec=spec, wq=_wq(ws), tick=0, start_tick=0,
        )
        self.assertIn("charted_chunks >= 10", result.milestones_hit)
        self.assertIn("charted_chunks >= 50", result.milestones_hit)
        self.assertNotIn("charted_chunks >= 100", result.milestones_hit)
        self.assertAlmostEqual(result.reward, 0.3)


# ===========================================================================
# PT — production_rate() via ProductionTracker  [PROXIMAL]
# ===========================================================================

class TestPT_ProductionRate(unittest.TestCase):

    def _tracker_with_history(self, item, count_start, count_end,
                               tick_start=0, tick_end=3600):
        from world.query import WorldQuery
        tracker = ProductionTracker()
        e_start = EntityState(
            entity_id=1, name="assembling-machine-1", position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot(item, count_start)]),
        )
        e_end = EntityState(
            entity_id=1, name="assembling-machine-1", position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot(item, count_end)]),
        )
        tracker.update(WorldQuery(WorldState(tick=tick_start, entities=[e_start])))
        tracker.update(WorldQuery(WorldState(tick=tick_end, entities=[e_end])))
        return tracker

    def test_PT_001_rate_above_threshold(self):
        tracker = self._tracker_with_history("iron-plate", 0, 60, 0, 3600)
        r = _ev_with_tracker(tracker, "production_rate('iron-plate') >= 60.0", tick=3600)
        self.assertTrue(r.success)

    def test_PT_002_rate_below_threshold(self):
        tracker = self._tracker_with_history("iron-plate", 0, 30, 0, 3600)
        r = _ev_with_tracker(tracker, "production_rate('iron-plate') >= 60.0", tick=3600)
        self.assertFalse(r.success)

    def test_PT_003_unseen_item_returns_zero(self):
        tracker = ProductionTracker()
        r = _ev_with_tracker(tracker, "production_rate('steel-plate') == 0.0")
        self.assertTrue(r.success)

    def test_PT_004_no_tracker_returns_zero(self):
        r = _ev("production_rate('iron-plate') == 0.0")
        self.assertTrue(r.success)

    def test_PT_005_rate_as_failure(self):
        tracker = self._tracker_with_history("iron-plate", 0, 10, 0, 3600)
        r = _ev_with_tracker(tracker, failure="production_rate('iron-plate') < 30.0", tick=3600)
        self.assertTrue(r.failure)

    def test_PT_006_rate_as_milestone(self):
        tracker = self._tracker_with_history("copper-plate", 0, 120, 0, 3600)
        spec = RewardSpec(milestone_rewards={"production_rate('copper-plate') >= 100.0": 0.3})
        ev = RewardEvaluator(tracker=tracker)
        result = ev.evaluate_conditions(
            success_condition="False", failure_condition="False",
            spec=spec, wq=_wq(WorldState(tick=3600)), tick=3600, start_tick=0,
        )
        self.assertIn("production_rate('copper-plate') >= 100.0", result.milestones_hit)
        self.assertAlmostEqual(result.reward, 0.3)

    def test_PT_007_single_snapshot_returns_zero(self):
        tracker = ProductionTracker()
        e = EntityState(
            entity_id=1, name="assembling-machine-1", position=Position(0, 0),
            inventory=Inventory(slots=[InventorySlot("iron-plate", 50)]),
        )
        ws = WorldState(tick=60, entities=[e])
        tracker.update(WorldQuery(ws))
        r = _ev_with_tracker(tracker, "production_rate('iron-plate') == 0.0",
                              state=ws, tick=60)
        self.assertTrue(r.success)


# ===========================================================================
# ST — staleness() freshness guards  [META]
# ===========================================================================

class TestST_Staleness(unittest.TestCase):

    def test_ST_001_fresh_section_zero(self):
        ws = WorldState(tick=1000, observed_at={"entities": 1000})
        self.assertTrue(_ev("staleness('entities') == 0", state=ws, tick=1000).success)

    def test_ST_002_stale_section_ticks_elapsed(self):
        ws = WorldState(tick=1000, observed_at={"entities": 700})
        self.assertTrue(_ev("staleness('entities') == 300", state=ws, tick=1000).success)

    def test_ST_003_never_observed_is_none(self):
        ws = WorldState(tick=1000)
        self.assertTrue(_ev("staleness('entities') is None", state=ws, tick=1000).success)

    def test_ST_004_guard_blocks_stale_proximal(self):
        ws = WorldState(tick=36000, observed_at={"entities": 0}, entities=[])
        r = _ev(
            "staleness('entities') is not None "
            "and staleness('entities') < 300 "
            "and len(entities('assembling-machine-1')) >= 3",
            state=ws, tick=36000,
        )
        self.assertFalse(r.success)

    def test_ST_005_guard_passes_when_fresh(self):
        ws = WorldState(
            tick=1000, observed_at={"entities": 950},
            entities=[
                EntityState(i, "assembling-machine-1", Position(i*5, 0),
                            status=EntityStatus.WORKING)
                for i in range(1, 4)
            ],
        )
        r = _ev(
            "staleness('entities') is not None "
            "and staleness('entities') < 300 "
            "and len(entities('assembling-machine-1')) >= 3",
            state=ws, tick=1000,
        )
        self.assertTrue(r.success)

    def test_ST_006_staleness_in_failure(self):
        ws = WorldState(tick=10000, observed_at={"entities": 0}, entities=[])
        r = _ev(
            failure="staleness('entities') is not None "
                    "and staleness('entities') < 600 "
                    "and len(entities('stone-furnace')) == 0",
            state=ws, tick=10000,
        )
        self.assertFalse(r.failure)

    def test_ST_007_never_seen_vs_stale(self):
        ws_never = WorldState(tick=1000, observed_at={})
        ws_stale = WorldState(tick=1000, observed_at={"entities": 0})
        self.assertTrue(_ev("staleness('entities') is None",
                            state=ws_never, tick=1000).success)
        self.assertTrue(_ev("staleness('entities') == 1000",
                            state=ws_stale, tick=1000).success)


# ===========================================================================
# SC — Scope boundary tests
# ===========================================================================

class TestSC_Scope(unittest.TestCase):

    def test_SC_001_inventory_unaffected_by_empty_scan(self):
        ws = WorldState(
            player=PlayerState(position=Position(0, 0),
                               inventory=Inventory(slots=[InventorySlot("iron-plate", 100)])),
            entities=[], observed_at={},
        )
        self.assertTrue(_ev("inventory('iron-plate') >= 100", state=ws).success)

    def test_SC_002_tech_unaffected_by_empty_scan(self):
        ws = WorldState(research=ResearchState(unlocked=["automation"]),
                        entities=[], observed_at={})
        self.assertTrue(_ev("tech_unlocked('automation')", state=ws).success)

    def test_SC_003_exploration_unaffected_by_empty_scan(self):
        ws = WorldState(
            player=PlayerState(position=Position(0, 0),
                               exploration=ExplorationState(charted_chunks=100)),
            entities=[], observed_at={},
        )
        self.assertTrue(_ev("charted_chunks >= 100", state=ws).success)

    def test_SC_004_resource_map_accumulates(self):
        ws = WorldState(resource_map=[
            ResourcePatch("iron-ore", Position(p*100, 100), 50000, 30) for p in range(4)
        ])
        self.assertTrue(_ev("len(resources_of_type('iron-ore')) >= 4", state=ws).success)

    def test_SC_005_entities_proximal_only(self):
        ws = WorldState(entities=[], observed_at={"entities": 100})
        self.assertFalse(_ev("len(entities('assembling-machine-1')) >= 1", state=ws).success)

    def test_SC_006_exploration_monotonic_unrelated_to_proximity(self):
        ws_near = _ws_with_chart(50)
        ws_far = WorldState(
            player=PlayerState(position=Position(5000, 5000),
                               exploration=ExplorationState(charted_chunks=50)),
            entities=[],
        )
        self.assertTrue(_ev("charted_chunks >= 50", state=ws_near).success)
        self.assertTrue(_ev("charted_chunks >= 50", state=ws_far).success)


# ===========================================================================
# XC — Compound conditions
# ===========================================================================

class TestXC_Compound(unittest.TestCase):

    def test_XC_001_inventory_and_tech(self):
        ws = WorldState(
            player=PlayerState(position=Position(0, 0),
                               inventory=Inventory(slots=[InventorySlot("iron-plate", 100)])),
            research=ResearchState(unlocked=["steel-processing"]),
        )
        r = _ev("inventory('iron-plate') >= 50 and tech_unlocked('steel-processing')", state=ws)
        self.assertTrue(r.success)

    def test_XC_002_exploration_and_research(self):
        ws = WorldState(
            player=PlayerState(position=Position(0, 0),
                               exploration=ExplorationState(charted_chunks=30)),
            research=ResearchState(unlocked=["automation"]),
        )
        r = _ev("charted_chunks >= 20 and tech_unlocked('automation')", state=ws)
        self.assertTrue(r.success)

    def test_XC_003_exploration_and_time(self):
        ws = _ws_with_chart(25)
        r = _ev("tick >= 3600 and charted_chunks >= 20", state=ws, tick=4000)
        self.assertTrue(r.success)

    def test_XC_004_exploration_as_failure_with_time(self):
        ws = _ws_with_chart(5)
        r = _ev(failure="tick > 7200 and charted_chunks < 20", state=ws, tick=8000)
        self.assertTrue(r.failure)

    def test_XC_005_all_non_proximal_combined(self):
        ws = WorldState(
            player=PlayerState(
                position=Position(0, 0),
                inventory=Inventory(slots=[InventorySlot("iron-gear-wheel", 200)]),
                exploration=ExplorationState(charted_chunks=30),
            ),
            research=ResearchState(unlocked=["automation"]),
            resource_map=[ResourcePatch("iron-ore", Position(100, 100), 50000, 30)],
        )
        r = _ev(
            "inventory('iron-gear-wheel') >= 200 "
            "and tech_unlocked('automation') "
            "and charted_chunks >= 25 "
            "and len(resources_of_type('iron-ore')) >= 1",
            state=ws,
        )
        self.assertTrue(r.success)

    def test_XC_006_failure_if_damage_or_not_explored(self):
        ws = WorldState(
            player=PlayerState(position=Position(0, 0),
                               exploration=ExplorationState(charted_chunks=5)),
            damaged_entities=[DamagedEntity(1, "wall", Position(0, 0), 0.5)],
        )
        r = _ev(failure="state.has_damage or charted_chunks < 10", state=ws)
        self.assertTrue(r.failure)

    def test_XC_007_wq_builder_compound(self):
        """Condition uses wq composable builder for a multi-criteria entity query."""
        assembler = _entity(1, "assembling-machine-1", 5, 5, recipe="electronic-circuit")
        feeder    = _inserter(10, 4, 5, pickup_pos=Position(3, 5), drop_pos=Position(5, 5))
        extractor = _inserter(11, 6, 5, pickup_pos=Position(5, 5), drop_pos=Position(7, 5))
        ws = WorldState(
            entities=[assembler],
            logistics=LogisticsState(inserters={10: feeder, 11: extractor}),
        )
        r = _ev(
            "wq.entities()"
            "   .with_recipe('electronic-circuit')"
            "   .with_inserter_input()"
            "   .with_inserter_output()"
            "   .count() >= 1",
            state=ws,
        )
        self.assertTrue(r.success)


if __name__ == "__main__":
    unittest.main(verbosity=2)