"""
tests/unit/execution/test_skills.py

Unit tests for execution/skills/:
  NavigateSkill, MineSkill, CraftSkill, DestroySkill

Each skill is tested in isolation via a lightweight WorldQuery stub that
exposes only the fields each skill actually reads. No RCON, no Factorio,
no bridge calls.

Sections
--------
1.  SkillStatus           — enum properties
2.  NavigateSkill         — position target, entity target, stall, reset
3.  MineSkill             — first issue, count completion, stall, reset
4.  CraftSkill            — dispatch, fire-and-forget, depletion, stall
5.  DestroySkill          — first issue, entity gone, stall, reset
6.  Cross-skill           — SkillProtocol structural conformance

Run with:  python -m pytest tests/unit/execution/test_skills.py -v
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

from execution.skills.base import SkillProtocol, SkillStatus
from execution.skills.navigate import (
    NavigateSkill,
    _STALL_GRACE_TICKS,
    _UNREACHABLE_GRACE_TICKS,
    _POSITION_ARRIVAL_TOLERANCE,
    _REDUNDANT_THRESHOLD,
    _STOPPED_THRESHOLD,
)
from execution.skills.mine import MineSkill, _MINING_GRACE_TICKS, _MAX_REISSUE as _MINE_MAX_REISSUE
from execution.skills.craft import CraftSkill, CraftTarget, _CRAFTING_GRACE_TICKS, _MAX_REISSUE as _CRAFT_MAX_REISSUE
from execution.skills.destroy import DestroySkill, _MINING_GRACE_TICKS as _DESTROY_GRACE, _MAX_REISSUE as _DESTROY_MAX_REISSUE
from bridge import MoveTo, StopMovement, MineResource, CraftItem, MineEntity
from world import Position


# ===========================================================================
# WorldQuery stubs
# ===========================================================================

@dataclass
class _PlayerState:
    position: Position = field(default_factory=lambda: Position(x=0.0, y=0.0))
    reachable: list[int] = field(default_factory=list)
    inventory_slots: list = field(default_factory=list)


@dataclass
class _InventorySlot:
    item: str
    count: int


@dataclass
class _EntityState:
    entity_id: int
    position: Position = field(default_factory=lambda: Position(x=0.0, y=0.0))


class _WQ:
    """
    Minimal WorldQuery stub. Only exposes what the skills actually read.
    Mutate fields between ticks to simulate game state changes.
    """

    def __init__(
        self,
        position: Position = None,
        reachable: list[int] = None,
        inventory: dict[str, int] = None,
        entities: list[_EntityState] = None,
        tick: int = 0,
    ):
        self._position  = position or Position(x=0.0, y=0.0)
        self._reachable = reachable or []
        self._inventory = inventory or {}
        self._entities  = {e.entity_id: e for e in (entities or [])}
        self.tick       = tick

        # Build state sub-object that skills read via wq.state.*
        self.state = MagicMock()
        self.state.player.reachable = self._reachable
        self._rebuild_inventory()

    def _rebuild_inventory(self):
        slots = [
            _InventorySlot(item=k, count=v)
            for k, v in self._inventory.items()
            if v > 0
        ]
        self.state.player.inventory.slots = slots

    def player_position(self) -> Position:
        return self._position

    def entity_by_id(self, entity_id: int):
        return self._entities.get(entity_id)

    def inventory_count(self, item: str) -> int:
        return self._inventory.get(item, 0)

    # Mutation helpers for test setup between ticks
    def move_to(self, x: float, y: float) -> None:
        self._position = Position(x=x, y=y)

    def set_inventory(self, inventory: dict[str, int]) -> None:
        self._inventory = dict(inventory)
        self._rebuild_inventory()

    def add_to_inventory(self, item: str, count: int) -> None:
        self._inventory[item] = self._inventory.get(item, 0) + count
        self._rebuild_inventory()

    def remove_entity(self, entity_id: int) -> None:
        self._entities.pop(entity_id, None)

    def set_reachable(self, ids: list[int]) -> None:
        self._reachable.clear()
        self._reachable.extend(ids)
        self.state.player.reachable = self._reachable


_WW = MagicMock()  # WorldWriter — skills don't use it; a single stub is fine


# ===========================================================================
# Section 1 — SkillStatus
# ===========================================================================

class TestSkillStatus(unittest.TestCase):

    def test_all_values_exist(self):
        for name in ("IDLE", "RUNNING", "SUCCEEDED", "FAILED", "STUCK"):
            self.assertIsNotNone(SkillStatus[name])

    def test_terminal_statuses(self):
        self.assertTrue(SkillStatus.SUCCEEDED.is_terminal)
        self.assertTrue(SkillStatus.FAILED.is_terminal)
        self.assertTrue(SkillStatus.STUCK.is_terminal)

    def test_non_terminal_statuses(self):
        self.assertFalse(SkillStatus.IDLE.is_terminal)
        self.assertFalse(SkillStatus.RUNNING.is_terminal)


# ===========================================================================
# Section 2 — NavigateSkill
# ===========================================================================

class TestNavigateSkillStartValidation(unittest.TestCase):

    def setUp(self):
        self.skill = NavigateSkill()

    def test_requires_at_least_one_target(self):
        with self.assertRaises(ValueError):
            self.skill.start()

    def test_rejects_both_targets(self):
        with self.assertRaises(ValueError):
            self.skill.start(
                target_position=Position(x=10, y=10),
                target_entity_id=42,
            )

    def test_idle_before_start(self):
        self.assertEqual(self.skill.status(), SkillStatus.IDLE)

    def test_running_after_start(self):
        self.skill.start(target_position=Position(x=10, y=10))
        self.assertEqual(self.skill.status(), SkillStatus.RUNNING)

    def test_restart_while_running_is_allowed(self):
        self.skill.start(target_position=Position(x=10, y=10))
        self.skill.start(target_position=Position(x=20, y=20))
        self.assertEqual(self.skill.status(), SkillStatus.RUNNING)


class TestNavigateSkillPositionTarget(unittest.TestCase):

    def setUp(self):
        self.skill = NavigateSkill()

    def test_issues_move_to_on_first_tick(self):
        wq = _WQ(position=Position(x=0, y=0))
        self.skill.start(target_position=Position(x=100, y=100))
        actions = self.skill.tick(wq, _WW, tick=0)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MoveTo)

    def test_move_to_has_correct_target(self):
        target = Position(x=50, y=75)
        wq = _WQ(position=Position(x=0, y=0))
        self.skill.start(target_position=target)
        actions = self.skill.tick(wq, _WW, tick=0)
        self.assertAlmostEqual(actions[0].position.x, 50.0)
        self.assertAlmostEqual(actions[0].position.y, 75.0)

    def test_arrives_when_within_tolerance(self):
        # Player within _POSITION_ARRIVAL_TOLERANCE of target
        target = Position(x=10, y=10)
        wq = _WQ(position=Position(x=10.0 + _POSITION_ARRIVAL_TOLERANCE - 0.1, y=10.0))
        self.skill.start(target_position=target)
        # First tick captures start position and issues MoveTo
        self.skill.tick(wq, _WW, tick=0)
        # Move player to arrival position
        wq.move_to(10.0 + _POSITION_ARRIVAL_TOLERANCE - 0.1, 10.0)
        actions = self.skill.tick(wq, _WW, tick=10)
        self.assertEqual(self.skill.status(), SkillStatus.SUCCEEDED)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], StopMovement)

    def test_not_arrived_when_beyond_tolerance(self):
        target = Position(x=0, y=0)
        wq = _WQ(position=Position(x=_POSITION_ARRIVAL_TOLERANCE + 1.0, y=0))
        self.skill.start(target_position=target)
        self.skill.tick(wq, _WW, tick=0)
        self.assertEqual(self.skill.status(), SkillStatus.RUNNING)

    def test_suppresses_redundant_move_to(self):
        # Second tick with same target position should not re-issue MoveTo
        target = Position(x=100, y=100)
        wq = _WQ(position=Position(x=0, y=0))
        self.skill.start(target_position=target)
        self.skill.tick(wq, _WW, tick=0)   # issues MoveTo
        # Move player slightly — still far from target, within redundant threshold
        wq.move_to(0.1, 0.0)
        actions = self.skill.tick(wq, _WW, tick=10)
        self.assertEqual(actions, [])

    def test_reissues_when_target_moves_significantly(self):
        # Entity target: entity moves by more than _REDUNDANT_THRESHOLD
        entity = _EntityState(entity_id=1, position=Position(x=50, y=50))
        wq = _WQ(position=Position(x=0, y=0), entities=[entity])
        self.skill.start(target_entity_id=1)
        self.skill.tick(wq, _WW, tick=0)   # issues MoveTo at (50, 50)
        # Move entity well beyond redundant threshold
        entity.position = Position(x=50 + _REDUNDANT_THRESHOLD + 1.0, y=50)
        wq.move_to(0.1, 0.0)   # player moved a little
        actions = self.skill.tick(wq, _WW, tick=10)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MoveTo)


class TestNavigateSkillEntityTarget(unittest.TestCase):

    def setUp(self):
        self.skill = NavigateSkill()

    def test_succeeds_when_entity_reachable(self):
        entity = _EntityState(entity_id=5, position=Position(x=10, y=10))
        wq = _WQ(
            position=Position(x=8, y=8),
            reachable=[5],
            entities=[entity],
        )
        self.skill.start(target_entity_id=5)
        actions = self.skill.tick(wq, _WW, tick=0)
        self.assertEqual(self.skill.status(), SkillStatus.SUCCEEDED)
        self.assertIsInstance(actions[0], StopMovement)

    def test_fails_when_entity_absent_from_scan(self):
        # Entity not in wq at all
        wq = _WQ(position=Position(x=0, y=0))
        self.skill.start(target_entity_id=99)
        actions = self.skill.tick(wq, _WW, tick=0)
        self.assertEqual(self.skill.status(), SkillStatus.FAILED)
        self.assertIsInstance(actions[0], StopMovement)

    def test_navigates_toward_entity_position(self):
        entity = _EntityState(entity_id=3, position=Position(x=100, y=50))
        wq = _WQ(position=Position(x=0, y=0), entities=[entity])
        self.skill.start(target_entity_id=3)
        actions = self.skill.tick(wq, _WW, tick=0)
        self.assertIsInstance(actions[0], MoveTo)
        self.assertAlmostEqual(actions[0].position.x, 100.0)
        self.assertAlmostEqual(actions[0].position.y, 50.0)


class TestNavigateSkillStall(unittest.TestCase):

    def setUp(self):
        self.skill = NavigateSkill()

    def _run_ticks(self, wq, start_tick, count, delta=10):
        """Run skill for `count` ticks starting at `start_tick`."""
        for i in range(count):
            self.skill.tick(wq, _WW, tick=start_tick + i * delta)

    def test_stuck_after_standard_grace_if_player_moved_then_stopped(self):
        wq = _WQ(position=Position(x=0, y=0))
        self.skill.start(target_position=Position(x=100, y=100))
        # First tick — issues MoveTo; move player slightly so never_moved=False
        self.skill.tick(wq, _WW, tick=0)
        wq.move_to(1.0, 0.0)   # player moved — triggers standard grace path
        # Tick up to just before grace — should still be RUNNING
        ticks_before_grace = _STALL_GRACE_TICKS // 10
        for i in range(1, ticks_before_grace):
            self.skill.tick(wq, _WW, tick=i * 10)
        self.assertEqual(self.skill.status(), SkillStatus.RUNNING)
        # One more tick past grace
        self.skill.tick(wq, _WW, tick=_STALL_GRACE_TICKS + 10)
        self.assertEqual(self.skill.status(), SkillStatus.STUCK)

    def test_stuck_after_unreachable_grace_if_player_never_moved(self):
        wq = _WQ(position=Position(x=0, y=0))
        self.skill.start(target_position=Position(x=100, y=100))
        self.skill.tick(wq, _WW, tick=0)   # captures start position
        # Player never moves — fast grace path
        for i in range(1, _UNREACHABLE_GRACE_TICKS // 10 + 2):
            self.skill.tick(wq, _WW, tick=i * 10)
        self.assertEqual(self.skill.status(), SkillStatus.STUCK)

    def test_stuck_emits_stop_movement(self):
        wq = _WQ(position=Position(x=0, y=0))
        self.skill.start(target_position=Position(x=100, y=100))
        self.skill.tick(wq, _WW, tick=0)
        for i in range(1, _UNREACHABLE_GRACE_TICKS // 10 + 2):
            actions = self.skill.tick(wq, _WW, tick=i * 10)
        self.assertIsInstance(actions[0], StopMovement)

    def test_no_ticks_after_stuck(self):
        wq = _WQ(position=Position(x=0, y=0))
        self.skill.start(target_position=Position(x=100, y=100))
        self.skill.tick(wq, _WW, tick=0)
        for i in range(1, _UNREACHABLE_GRACE_TICKS // 10 + 2):
            self.skill.tick(wq, _WW, tick=i * 10)
        self.assertEqual(self.skill.status(), SkillStatus.STUCK)
        actions = self.skill.tick(wq, _WW, tick=9999)
        self.assertEqual(actions, [])


class TestNavigateSkillReset(unittest.TestCase):

    def test_reset_returns_to_idle(self):
        skill = NavigateSkill()
        skill.start(target_position=Position(x=10, y=10))
        skill.reset()
        self.assertEqual(skill.status(), SkillStatus.IDLE)

    def test_reset_clears_state_for_reuse(self):
        skill = NavigateSkill()
        skill.start(target_position=Position(x=10, y=10))
        wq = _WQ(position=Position(x=0, y=0))
        skill.tick(wq, _WW, tick=0)
        skill.reset()
        # After reset, a new start() should issue a fresh MoveTo
        skill.start(target_position=Position(x=20, y=20))
        actions = skill.tick(wq, _WW, tick=100)
        self.assertIsInstance(actions[0], MoveTo)

    def test_tick_on_idle_returns_empty(self):
        skill = NavigateSkill()
        wq = _WQ(position=Position(x=0, y=0))
        actions = skill.tick(wq, _WW, tick=0)
        self.assertEqual(actions, [])

    def test_tick_on_succeeded_returns_empty(self):
        skill = NavigateSkill()
        target = Position(x=0, y=0)
        wq = _WQ(position=Position(x=0, y=0))
        skill.start(target_position=target)
        skill.tick(wq, _WW, tick=0)  # arrives immediately
        self.assertEqual(skill.status(), SkillStatus.SUCCEEDED)
        actions = skill.tick(wq, _WW, tick=10)
        self.assertEqual(actions, [])

    def test_observe_contains_status(self):
        skill = NavigateSkill()
        skill.start(target_position=Position(x=5, y=5))
        obs = skill.observe()
        self.assertIn("navigate_status", obs)
        self.assertEqual(obs["navigate_status"], "RUNNING")


# ===========================================================================
# Section 3 — MineSkill
# ===========================================================================

class TestMineSkillStart(unittest.TestCase):

    def test_idle_before_start(self):
        self.assertEqual(MineSkill().status(), SkillStatus.IDLE)

    def test_running_after_start(self):
        skill = MineSkill()
        skill.start(Position(x=10, y=10), "iron-ore", count=50)
        self.assertEqual(skill.status(), SkillStatus.RUNNING)

    def test_issues_mine_resource_on_first_tick(self):
        skill = MineSkill()
        skill.start(Position(x=10, y=10), "iron-ore", count=50)
        wq = _WQ()
        actions = skill.tick(wq, _WW, tick=0)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MineResource)

    def test_mine_resource_has_correct_position(self):
        skill = MineSkill()
        pos = Position(x=33, y=44)
        skill.start(pos, "coal", count=10)
        wq = _WQ()
        actions = skill.tick(wq, _WW, tick=0)
        self.assertAlmostEqual(actions[0].position.x, 33.0)
        self.assertAlmostEqual(actions[0].position.y, 44.0)

    def test_mine_resource_count_is_zero(self):
        # count=0 means mine continuously — skill tracks count itself
        skill = MineSkill()
        skill.start(Position(x=0, y=0), "iron-ore", count=5)
        wq = _WQ()
        actions = skill.tick(wq, _WW, tick=0)
        self.assertEqual(actions[0].count, 0)

    def test_no_action_on_second_tick_without_stall(self):
        skill = MineSkill()
        skill.start(Position(x=0, y=0), "iron-ore", count=50)
        wq = _WQ()
        skill.tick(wq, _WW, tick=0)
        actions = skill.tick(wq, _WW, tick=10)
        self.assertEqual(actions, [])


class TestMineSkillCompletion(unittest.TestCase):

    def test_succeeds_when_count_reached(self):
        skill = MineSkill()
        skill.start(Position(x=0, y=0), "iron-ore", count=10)
        wq = _WQ(inventory={"iron-ore": 0})
        skill.tick(wq, _WW, tick=0)   # snapshot taken: 0 iron-ore
        # Add 10 iron-ore to simulate mining
        wq.set_inventory({"iron-ore": 10})
        skill.tick(wq, _WW, tick=10)
        self.assertEqual(skill.status(), SkillStatus.SUCCEEDED)

    def test_not_succeeded_when_count_not_reached(self):
        skill = MineSkill()
        skill.start(Position(x=0, y=0), "iron-ore", count=20)
        wq = _WQ(inventory={"iron-ore": 0})
        skill.tick(wq, _WW, tick=0)
        wq.set_inventory({"iron-ore": 10})   # only 10 of 20
        skill.tick(wq, _WW, tick=10)
        self.assertEqual(skill.status(), SkillStatus.RUNNING)

    def test_count_zero_never_succeeds(self):
        skill = MineSkill()
        skill.start(Position(x=0, y=0), "iron-ore", count=0)
        wq = _WQ(inventory={"iron-ore": 0})
        skill.tick(wq, _WW, tick=0)
        wq.set_inventory({"iron-ore": 1000})
        for i in range(1, 10):
            skill.tick(wq, _WW, tick=i * 10)
        self.assertEqual(skill.status(), SkillStatus.RUNNING)

    def test_delta_measured_from_start_not_absolute(self):
        # Player starts with 5 iron-ore; only needs 3 more
        skill = MineSkill()
        skill.start(Position(x=0, y=0), "iron-ore", count=3)
        wq = _WQ(inventory={"iron-ore": 5})
        skill.tick(wq, _WW, tick=0)   # snapshot: 5
        wq.set_inventory({"iron-ore": 8})   # delta = 3
        skill.tick(wq, _WW, tick=10)
        self.assertEqual(skill.status(), SkillStatus.SUCCEEDED)


class TestMineSkillStall(unittest.TestCase):

    def _tick_to_stall(self, skill, wq, n_reissues):
        """Tick past grace period `n_reissues` times with no inventory change."""
        tick = 0
        for _ in range(n_reissues):
            tick += _MINING_GRACE_TICKS + 10
            skill.tick(wq, _WW, tick=tick)
        return tick

    def test_reissues_on_stall(self):
        skill = MineSkill()
        skill.start(Position(x=0, y=0), "iron-ore", count=50)
        wq = _WQ(inventory={"iron-ore": 0})
        skill.tick(wq, _WW, tick=0)   # first issue
        # Advance past grace with no inventory change
        actions = skill.tick(wq, _WW, tick=_MINING_GRACE_TICKS + 10)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MineResource)

    def test_stuck_after_max_reissues(self):
        skill = MineSkill()
        skill.start(Position(x=0, y=0), "iron-ore", count=50)
        wq = _WQ(inventory={"iron-ore": 0})
        skill.tick(wq, _WW, tick=0)
        self._tick_to_stall(skill, wq, _MINE_MAX_REISSUE + 1)
        self.assertEqual(skill.status(), SkillStatus.STUCK)

    def test_no_stuck_before_max_reissues(self):
        skill = MineSkill()
        skill.start(Position(x=0, y=0), "iron-ore", count=50)
        wq = _WQ(inventory={"iron-ore": 0})
        skill.tick(wq, _WW, tick=0)
        self._tick_to_stall(skill, wq, _MINE_MAX_REISSUE)
        self.assertNotEqual(skill.status(), SkillStatus.STUCK)


class TestMineSkillReset(unittest.TestCase):

    def test_reset_returns_to_idle(self):
        skill = MineSkill()
        skill.start(Position(x=0, y=0), "iron-ore", count=10)
        skill.reset()
        self.assertEqual(skill.status(), SkillStatus.IDLE)

    def test_reset_clears_snapshot(self):
        skill = MineSkill()
        skill.start(Position(x=0, y=0), "iron-ore", count=10)
        wq = _WQ(inventory={"iron-ore": 5})
        skill.tick(wq, _WW, tick=0)   # snapshot = 5
        skill.reset()
        # Start fresh — snapshot should be taken anew
        skill.start(Position(x=0, y=0), "iron-ore", count=3)
        wq.set_inventory({"iron-ore": 0})   # different starting inventory
        skill.tick(wq, _WW, tick=100)      # new snapshot = 0
        wq.set_inventory({"iron-ore": 3})
        skill.tick(wq, _WW, tick=110)
        self.assertEqual(skill.status(), SkillStatus.SUCCEEDED)

    def test_observe_keys(self):
        skill = MineSkill()
        skill.start(Position(x=0, y=0), "iron-ore", count=10)
        obs = skill.observe()
        self.assertIn("mine_status", obs)
        self.assertIn("mine_resource", obs)
        self.assertIn("mine_target_count", obs)
        self.assertIn("mine_reissue_count", obs)


# ===========================================================================
# Section 4 — CraftSkill
# ===========================================================================

class TestCraftSkillStart(unittest.TestCase):

    def test_rejects_empty_targets(self):
        with self.assertRaises(ValueError):
            CraftSkill().start(targets=[])

    def test_idle_before_start(self):
        self.assertEqual(CraftSkill().status(), SkillStatus.IDLE)

    def test_running_after_start(self):
        skill = CraftSkill()
        skill.start([CraftTarget("iron-gear-wheel", "iron-gear-wheel", 5)])
        self.assertEqual(skill.status(), SkillStatus.RUNNING)


class TestCraftSkillDispatch(unittest.TestCase):

    def _make_wq(self, inventory=None):
        return _WQ(inventory=inventory or {"iron-plate": 100})

    def test_dispatches_craft_items_on_first_tick(self):
        skill = CraftSkill()
        skill.start([
            CraftTarget("iron-gear-wheel", "iron-gear-wheel", 5),
            CraftTarget("copper-cable", "copper-cable", 10),
        ])
        wq = self._make_wq()
        actions = skill.tick(wq, _WW, tick=0)
        self.assertEqual(len(actions), 2)
        self.assertTrue(all(isinstance(a, CraftItem) for a in actions))

    def test_craft_item_has_correct_recipe_and_count(self):
        skill = CraftSkill()
        skill.start([CraftTarget("iron-gear-wheel", "iron-gear-wheel", 7)])
        wq = self._make_wq()
        actions = skill.tick(wq, _WW, tick=0)
        self.assertEqual(actions[0].recipe, "iron-gear-wheel")
        self.assertEqual(actions[0].count, 7)

    def test_no_second_dispatch_within_grace(self):
        skill = CraftSkill()
        skill.start(
            [CraftTarget("iron-gear-wheel", "iron-gear-wheel", 5)],
            expected_post_inv={"iron-plate": 90},
        )
        wq = self._make_wq({"iron-plate": 100})
        skill.tick(wq, _WW, tick=0)
        actions = skill.tick(wq, _WW, tick=10)
        self.assertEqual(actions, [])


class TestCraftSkillFireAndForget(unittest.TestCase):

    def test_succeeds_immediately_without_expected_post(self):
        skill = CraftSkill()
        skill.start([CraftTarget("iron-gear-wheel", "iron-gear-wheel", 5)])
        wq = _WQ(inventory={"iron-plate": 100})
        skill.tick(wq, _WW, tick=0)
        self.assertEqual(skill.status(), SkillStatus.SUCCEEDED)

    def test_fire_and_forget_dispatches_actions(self):
        skill = CraftSkill()
        skill.start([CraftTarget("iron-gear-wheel", "iron-gear-wheel", 3)])
        wq = _WQ(inventory={"iron-plate": 100})
        actions = skill.tick(wq, _WW, tick=0)
        self.assertEqual(len(actions), 1)


class TestCraftSkillDepletion(unittest.TestCase):

    def test_succeeds_when_ingredients_depleted(self):
        # Craft 5 gears: costs 10 iron-plate per 5 gears
        skill = CraftSkill()
        skill.start(
            [CraftTarget("iron-gear-wheel", "iron-gear-wheel", 5)],
            expected_post_inv={"iron-plate": 90},   # 100 → 90 = 10 consumed
        )
        wq = _WQ(inventory={"iron-plate": 100})
        skill.tick(wq, _WW, tick=0)          # snapshot: 100
        wq.set_inventory({"iron-plate": 90}) # 10 consumed
        skill.tick(wq, _WW, tick=10)
        self.assertEqual(skill.status(), SkillStatus.SUCCEEDED)

    def test_running_when_partially_depleted(self):
        skill = CraftSkill()
        skill.start(
            [CraftTarget("iron-gear-wheel", "iron-gear-wheel", 5)],
            expected_post_inv={"iron-plate": 90},
        )
        wq = _WQ(inventory={"iron-plate": 100})
        skill.tick(wq, _WW, tick=0)
        wq.set_inventory({"iron-plate": 95})  # only 5 consumed, need 10
        skill.tick(wq, _WW, tick=10)
        self.assertEqual(skill.status(), SkillStatus.RUNNING)

    def test_unrelated_inventory_changes_dont_trigger_success(self):
        # Adding ore to inventory shouldn't look like ingredient depletion
        skill = CraftSkill()
        skill.start(
            [CraftTarget("iron-gear-wheel", "iron-gear-wheel", 5)],
            expected_post_inv={"iron-plate": 90},
        )
        wq = _WQ(inventory={"iron-plate": 100})
        skill.tick(wq, _WW, tick=0)
        # iron-plate unchanged; iron-ore added
        wq.set_inventory({"iron-plate": 100, "iron-ore": 50})
        skill.tick(wq, _WW, tick=10)
        self.assertEqual(skill.status(), SkillStatus.RUNNING)


class TestCraftSkillStall(unittest.TestCase):

    def _tick_to_stall(self, skill, wq, n_reissues):
        tick = 0
        for _ in range(n_reissues):
            tick += _CRAFTING_GRACE_TICKS + 10
            skill.tick(wq, _WW, tick=tick)
        return tick

    def test_reissues_on_stall(self):
        skill = CraftSkill()
        skill.start(
            [CraftTarget("iron-gear-wheel", "iron-gear-wheel", 5)],
            expected_post_inv={"iron-plate": 90},
        )
        wq = _WQ(inventory={"iron-plate": 100})
        skill.tick(wq, _WW, tick=0)
        actions = skill.tick(wq, _WW, tick=_CRAFTING_GRACE_TICKS + 10)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], CraftItem)

    def test_stuck_after_max_reissues(self):
        skill = CraftSkill()
        skill.start(
            [CraftTarget("iron-gear-wheel", "iron-gear-wheel", 5)],
            expected_post_inv={"iron-plate": 90},
        )
        wq = _WQ(inventory={"iron-plate": 100})
        skill.tick(wq, _WW, tick=0)
        self._tick_to_stall(skill, wq, _CRAFT_MAX_REISSUE + 1)
        self.assertEqual(skill.status(), SkillStatus.STUCK)


class TestCraftSkillReset(unittest.TestCase):

    def test_reset_returns_to_idle(self):
        skill = CraftSkill()
        skill.start([CraftTarget("iron-gear-wheel", "iron-gear-wheel", 5)])
        skill.reset()
        self.assertEqual(skill.status(), SkillStatus.IDLE)

    def test_observe_keys(self):
        skill = CraftSkill()
        skill.start([CraftTarget("iron-gear-wheel", "iron-gear-wheel", 5)])
        obs = skill.observe()
        self.assertIn("craft_status", obs)
        self.assertIn("craft_targets", obs)
        self.assertIn("craft_dispatched", obs)
        self.assertIn("craft_reissue_count", obs)


# ===========================================================================
# Section 5 — DestroySkill
# ===========================================================================

class TestDestroySkillStart(unittest.TestCase):

    def test_idle_before_start(self):
        self.assertEqual(DestroySkill().status(), SkillStatus.IDLE)

    def test_running_after_start(self):
        skill = DestroySkill()
        skill.start(entity_id=10)
        self.assertEqual(skill.status(), SkillStatus.RUNNING)


class TestDestroySkillFirstIssue(unittest.TestCase):

    def test_issues_mine_entity_on_first_tick(self):
        entity = _EntityState(entity_id=10, position=Position(x=5, y=5))
        wq = _WQ(entities=[entity])
        skill = DestroySkill()
        skill.start(entity_id=10)
        actions = skill.tick(wq, _WW, tick=0)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MineEntity)
        self.assertEqual(actions[0].entity_id, 10)

    def test_no_action_on_second_tick_within_grace(self):
        entity = _EntityState(entity_id=10)
        wq = _WQ(entities=[entity])
        skill = DestroySkill()
        skill.start(entity_id=10)
        skill.tick(wq, _WW, tick=0)
        actions = skill.tick(wq, _WW, tick=10)
        self.assertEqual(actions, [])


class TestDestroySkillEntityGone(unittest.TestCase):

    def test_succeeds_when_entity_disappears(self):
        entity = _EntityState(entity_id=7)
        wq = _WQ(entities=[entity])
        skill = DestroySkill()
        skill.start(entity_id=7)
        skill.tick(wq, _WW, tick=0)    # issues MineEntity
        wq.remove_entity(7)
        skill.tick(wq, _WW, tick=10)
        self.assertEqual(skill.status(), SkillStatus.SUCCEEDED)

    def test_succeeds_immediately_if_entity_already_gone(self):
        wq = _WQ()   # no entities
        skill = DestroySkill()
        skill.start(entity_id=99)
        skill.tick(wq, _WW, tick=0)
        self.assertEqual(skill.status(), SkillStatus.SUCCEEDED)

    def test_no_actions_after_succeeded(self):
        entity = _EntityState(entity_id=7)
        wq = _WQ(entities=[entity])
        skill = DestroySkill()
        skill.start(entity_id=7)
        skill.tick(wq, _WW, tick=0)
        wq.remove_entity(7)
        skill.tick(wq, _WW, tick=10)
        actions = skill.tick(wq, _WW, tick=20)
        self.assertEqual(actions, [])


class TestDestroySkillStall(unittest.TestCase):

    def _tick_to_stall(self, skill, wq, n_reissues):
        tick = 0
        for _ in range(n_reissues):
            tick += _DESTROY_GRACE + 10
            skill.tick(wq, _WW, tick=tick)
        return tick

    def test_reissues_mine_entity_on_stall(self):
        entity = _EntityState(entity_id=3)
        wq = _WQ(entities=[entity])
        skill = DestroySkill()
        skill.start(entity_id=3)
        skill.tick(wq, _WW, tick=0)
        actions = skill.tick(wq, _WW, tick=_DESTROY_GRACE + 10)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MineEntity)
        self.assertEqual(actions[0].entity_id, 3)

    def test_stuck_after_max_reissues(self):
        entity = _EntityState(entity_id=3)
        wq = _WQ(entities=[entity])
        skill = DestroySkill()
        skill.start(entity_id=3)
        skill.tick(wq, _WW, tick=0)
        self._tick_to_stall(skill, wq, _DESTROY_MAX_REISSUE + 1)
        self.assertEqual(skill.status(), SkillStatus.STUCK)

    def test_reissue_count_tracked(self):
        entity = _EntityState(entity_id=3)
        wq = _WQ(entities=[entity])
        skill = DestroySkill()
        skill.start(entity_id=3)
        skill.tick(wq, _WW, tick=0)
        self._tick_to_stall(skill, wq, 2)
        obs = skill.observe()
        self.assertEqual(obs["destroy_reissue_count"], 2)


class TestDestroySkillReset(unittest.TestCase):

    def test_reset_returns_to_idle(self):
        skill = DestroySkill()
        skill.start(entity_id=5)
        skill.reset()
        self.assertEqual(skill.status(), SkillStatus.IDLE)

    def test_reset_allows_reuse_with_new_entity(self):
        entity_a = _EntityState(entity_id=1)
        entity_b = _EntityState(entity_id=2)
        wq = _WQ(entities=[entity_a, entity_b])
        skill = DestroySkill()

        skill.start(entity_id=1)
        skill.tick(wq, _WW, tick=0)
        wq.remove_entity(1)
        skill.tick(wq, _WW, tick=10)
        self.assertEqual(skill.status(), SkillStatus.SUCCEEDED)

        skill.reset()
        skill.start(entity_id=2)
        actions = skill.tick(wq, _WW, tick=20)
        self.assertEqual(skill.status(), SkillStatus.RUNNING)
        self.assertIsInstance(actions[0], MineEntity)
        self.assertEqual(actions[0].entity_id, 2)

    def test_observe_keys(self):
        skill = DestroySkill()
        skill.start(entity_id=5)
        obs = skill.observe()
        self.assertIn("destroy_status", obs)
        self.assertIn("destroy_entity_id", obs)
        self.assertIn("destroy_reissue_count", obs)


# ===========================================================================
# Section 6 — Cross-skill structural conformance
# ===========================================================================

class TestSkillProtocolConformance(unittest.TestCase):
    """
    Every implemented skill must satisfy SkillProtocol structurally:
    - status() returns SkillStatus
    - reset() returns to IDLE
    - observe() returns a dict
    - tick() on IDLE returns []
    - tick() on terminal status returns []
    """

    def _skills(self):
        nav = NavigateSkill()
        nav.start(target_position=Position(x=10, y=10))

        mine = MineSkill()
        mine.start(Position(x=0, y=0), "iron-ore", count=5)

        craft = CraftSkill()
        craft.start([CraftTarget("iron-gear-wheel", "iron-gear-wheel", 1)])

        destroy = DestroySkill()
        destroy.start(entity_id=1)

        return [nav, mine, craft, destroy]

    def test_status_returns_skill_status(self):
        for skill in self._skills():
            self.assertIsInstance(skill.status(), SkillStatus)

    def test_observe_returns_dict(self):
        for skill in self._skills():
            self.assertIsInstance(skill.observe(), dict)

    def test_reset_returns_to_idle(self):
        for skill in self._skills():
            skill.reset()
            self.assertEqual(skill.status(), SkillStatus.IDLE)

    def test_tick_on_idle_returns_empty(self):
        wq = _WQ()
        skills_idle = [NavigateSkill(), MineSkill(), CraftSkill(), DestroySkill()]
        for skill in skills_idle:
            actions = skill.tick(wq, _WW, tick=0)
            self.assertEqual(actions, [], msg=f"{type(skill).__name__} idle tick not empty")

    def test_is_instance_of_protocol(self):
        for skill in self._skills():
            self.assertIsInstance(skill, SkillProtocol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
