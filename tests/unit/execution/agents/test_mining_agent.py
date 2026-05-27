"""
tests/unit/execution/test_mining_agent.py

Unit tests for execution/agents/mining.py (MiningAgent).

Sections
--------
1.  Stubs and helpers
2.  activate() — task attributes, skill initialisation, StopMining on first tick
3.  gather_resource — MineSkill delegation, outcome observations, progress
4.  clear_region — target list building, navigate→destroy loop,
                   target skipping on nav/destroy stuck, empty region
5.  observe() and progress() cross-check
6.  pending_patches()

Run with:  python -m pytest tests/unit/execution/test_mining_agent.py -v
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

from execution.agents.mining import MiningAgent, _TaskKind, _ClearPhase, _is_natural
from execution.blackboard import Blackboard, EntryCategory, EntryScope
from execution.skills.base import SkillStatus
from world import Position
from bridge import MineResource, MineEntity, MoveTo, StopMining


# ===========================================================================
# Stubs
# ===========================================================================

@dataclass
class _BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class _Task:
    id: str = "task-mine-01"
    description: str = "mine test"
    task_type: str = "gather_resource"
    resource_type: str = "iron-ore"
    target_position: Optional[Position] = None
    count: int = 10
    bbox: Optional[_BBox] = None
    clear_mode: str = "clear_natural"


@dataclass
class _Slot:
    item: str
    count: int


@dataclass
class _EntityState:
    entity_id: int
    name: str = "iron-ore"
    position: Position = field(default_factory=lambda: Position(0.0, 0.0))


class _WQ:
    """Minimal WorldQuery stub for mining tests."""

    def __init__(
        self,
        position: Position = None,
        reachable: list[int] = None,
        inventory: dict[str, int] = None,
        entities: list[_EntityState] = None,
        tick: int = 1,
        crafting_queue_size: int = 0,
    ):
        self._position  = position or Position(x=0.0, y=0.0)
        self._reachable = reachable or []
        self._inventory = dict(inventory or {})
        self._entities  = {e.entity_id: e for e in (entities or [])}
        self.tick       = tick
        self._crafting_queue_size = crafting_queue_size
        self.state      = MagicMock()
        self.state.player.reachable     = self._reachable
        self.state.player.inventory.slots = self._build_slots()
        self.state.entities = list(self._entities.values())

    def _build_slots(self):
        return [_Slot(item=k, count=v) for k, v in self._inventory.items() if v > 0]

    def player_position(self) -> Position:
        return self._position

    def entity_by_id(self, eid: int):
        return self._entities.get(eid)

    def inventory_count(self, item: str) -> int:
        return self._inventory.get(item, 0)

    @property
    def crafting_queue_size(self) -> int:
        return self._crafting_queue_size

    @property
    def nearby_uncharted_chunks(self) -> list:
        return []

    @property
    def charted_chunks(self) -> int:
        return 0

    def move_to(self, x: float, y: float) -> None:
        self._position = Position(x=x, y=y)

    def set_inventory(self, inv: dict[str, int]) -> None:
        self._inventory = dict(inv)
        self.state.player.inventory.slots = self._build_slots()

    def set_reachable(self, ids: list[int]) -> None:
        self._reachable.clear()
        self._reachable.extend(ids)
        self.state.player.reachable = self._reachable

    def remove_entity(self, eid: int) -> None:
        self._entities.pop(eid, None)
        self.state.entities = list(self._entities.values())

    def add_entity(self, e: _EntityState) -> None:
        self._entities[e.entity_id] = e
        self.state.entities = list(self._entities.values())


_WW = MagicMock()
_KB = MagicMock()


def _make_agent() -> MiningAgent:
    return MiningAgent()


def _make_bb() -> Blackboard:
    return Blackboard()


def _obs_of_type(bb: Blackboard, obs_type: str) -> list:
    return [
        e for e in bb.read(category=EntryCategory.OBSERVATION)
        if e.data.get("type") == obs_type
    ]


# ===========================================================================
# Section 2 — activate()
# ===========================================================================

class TestMiningAgentActivate(unittest.TestCase):

    def test_gather_starts_mine_skill(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=10, y=10))
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._task_kind, _TaskKind.GATHER)
        self.assertEqual(agent._mine_skill.status(), SkillStatus.RUNNING)

    def test_gather_missing_position_leaves_skill_idle(self):
        agent = _make_agent()
        task = _Task(target_position=None)
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._mine_skill.status(), SkillStatus.IDLE)

    def test_clear_sets_kind(self):
        agent = _make_agent()
        task = _Task(
            task_type="clear_region",
            bbox=_BBox(0, 0, 50, 50),
        )
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._task_kind, _TaskKind.CLEAR)

    def test_unknown_task_type_sets_unknown_kind(self):
        agent = _make_agent()
        task = _Task(task_type="something_else")
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._task_kind, _TaskKind.UNKNOWN)

    def test_first_tick_emits_stop_mining(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=10, y=10))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        actions = agent.tick(task, bb, _WQ(), _WW, 1, _KB)
        self.assertTrue(any(isinstance(a, StopMining) for a in actions))

    def test_stop_mining_only_on_first_tick(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=10, y=10))
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # StopMining
        actions = agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertFalse(any(isinstance(a, StopMining) for a in actions))

    def test_activate_writes_mining_started(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=10, y=10))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        self.assertEqual(len(_obs_of_type(bb, "mining_started")), 1)

    def test_reactivate_resets_skills(self):
        agent = _make_agent()
        task1 = _Task(target_position=Position(x=10, y=10))
        task2 = _Task(target_position=Position(x=20, y=20))
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task1, bb, wq, _KB)
        agent.tick(task1, bb, wq, _WW, 1, _KB)   # StopMining
        agent.activate(task2, bb, wq, _KB)
        self.assertFalse(agent._gather_outcome_written)
        self.assertEqual(agent._mine_skill.status(), SkillStatus.RUNNING)


# ===========================================================================
# Section 3 — gather_resource
# ===========================================================================

class TestMiningAgentGather(unittest.TestCase):

    def _setup(self, count=10, inv=None):
        agent = _make_agent()
        task = _Task(
            target_position=Position(x=10, y=10),
            resource_type="iron-ore",
            count=count,
        )
        bb = _make_bb()
        wq = _WQ(inventory=inv or {"iron-ore": 0})
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # drain StopMining
        return agent, task, bb, wq

    def test_issues_mine_resource_after_stop(self):
        agent, task, bb, wq = self._setup()
        actions = agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertTrue(any(isinstance(a, MineResource) for a in actions))

    def test_mine_resource_correct_position(self):
        agent, task, bb, wq = self._setup()
        actions = agent.tick(task, bb, wq, _WW, 2, _KB)
        mine = [a for a in actions if isinstance(a, MineResource)][0]
        self.assertAlmostEqual(mine.position.x, 10.0)
        self.assertAlmostEqual(mine.position.y, 10.0)

    def test_mine_resource_correct_resource_type(self):
        agent, task, bb, wq = self._setup()
        actions = agent.tick(task, bb, wq, _WW, 2, _KB)
        mine = [a for a in actions if isinstance(a, MineResource)][0]
        self.assertEqual(mine.resource, "iron-ore")

    def test_gather_succeeded_observation_written(self):
        agent, task, bb, wq = self._setup(count=5)
        agent.tick(task, bb, wq, _WW, 2, _KB)   # first MineResource
        # Simulate count reached
        wq.set_inventory({"iron-ore": 5})
        agent.tick(task, bb, wq, _WW, 3, _KB)   # skill → SUCCEEDED
        agent.tick(task, bb, wq, _WW, 4, _KB)   # outcome written
        self.assertEqual(len(_obs_of_type(bb, "gather_succeeded")), 1)

    def test_gather_succeeded_observation_written_only_once(self):
        agent, task, bb, wq = self._setup(count=5)
        agent.tick(task, bb, wq, _WW, 2, _KB)
        wq.set_inventory({"iron-ore": 5})
        for i in range(3, 8):
            agent.tick(task, bb, wq, _WW, i, _KB)
        self.assertEqual(len(_obs_of_type(bb, "gather_succeeded")), 1)

    def test_gather_stuck_observation_written(self):
        from execution.skills.mine import _MINING_GRACE_TICKS, _MAX_REISSUE
        agent, task, bb, wq = self._setup()
        agent.tick(task, bb, wq, _WW, 2, _KB)   # first MineResource
        tick = 2
        for _ in range(_MAX_REISSUE + 2):
            tick += _MINING_GRACE_TICKS + 10
            agent.tick(task, bb, wq, _WW, tick, _KB)
        # One more tick for outcome to be written
        agent.tick(task, bb, wq, _WW, tick + 1, _KB)
        self.assertEqual(len(_obs_of_type(bb, "gather_stuck")), 1)

    def test_progress_zero_before_mine_issued(self):
        agent, task, bb, wq = self._setup()
        # After StopMining tick but before MineResource
        self.assertAlmostEqual(agent.progress(task, bb, wq, _KB), 0.5)

    def test_progress_one_on_succeeded(self):
        agent, task, bb, wq = self._setup(count=5)
        agent.tick(task, bb, wq, _WW, 2, _KB)
        wq.set_inventory({"iron-ore": 5})
        agent.tick(task, bb, wq, _WW, 3, _KB)
        agent.tick(task, bb, wq, _WW, 4, _KB)
        self.assertAlmostEqual(agent.progress(task, bb, wq, _KB), 1.0)


# ===========================================================================
# Section 4 — clear_region
# ===========================================================================

class TestMiningAgentClearTargetBuilding(unittest.TestCase):

    def _clear_task(self, bbox=None, mode="clear_natural"):
        return _Task(
            task_type="clear_region",
            bbox=bbox or _BBox(0, 0, 100, 100),
            clear_mode=mode,
        )

    def test_builds_target_list_from_entities_in_bbox(self):
        agent = _make_agent()
        entities = [
            _EntityState(1, "tree-01",     Position(x=10, y=10)),
            _EntityState(2, "rock-huge",   Position(x=20, y=20)),
            _EntityState(3, "iron-chest",  Position(x=30, y=30)),  # not natural
        ]
        wq = _WQ(entities=entities)
        task = self._clear_task(mode="clear_natural")
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # StopMining
        agent.tick(task, bb, wq, _WW, 2, _KB)   # build target list
        # Only tree and rock are natural
        total = agent._targets_total
        self.assertEqual(total, 2)

    def test_clear_all_includes_non_natural(self):
        agent = _make_agent()
        entities = [
            _EntityState(1, "tree-01",    Position(x=10, y=10)),
            _EntityState(2, "iron-chest", Position(x=20, y=20)),
        ]
        wq = _WQ(entities=entities)
        task = self._clear_task(mode="clear_all")
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertEqual(agent._targets_total, 2)

    def test_excludes_entities_outside_bbox(self):
        agent = _make_agent()
        entities = [
            _EntityState(1, "tree-01", Position(x=10, y=10)),   # inside
            _EntityState(2, "tree-02", Position(x=200, y=200)), # outside
        ]
        wq = _WQ(entities=entities)
        task = self._clear_task(bbox=_BBox(0, 0, 50, 50))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertEqual(agent._targets_total, 1)

    def test_empty_region_returns_no_actions(self):
        agent = _make_agent()
        wq = _WQ()   # no entities
        task = self._clear_task()
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # StopMining
        actions = agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertEqual(actions, [])

    def test_missing_bbox_logs_and_returns_empty(self):
        agent = _make_agent()
        task = _Task(task_type="clear_region", bbox=None)
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        agent.tick(task, bb, _WQ(), _WW, 1, _KB)
        actions = agent.tick(task, bb, _WQ(), _WW, 2, _KB)
        self.assertEqual(actions, [])


class TestMiningAgentClearLoop(unittest.TestCase):

    def _make_clear_wq(self, entities, reachable=None):
        return _WQ(
            position=Position(x=0, y=0),
            reachable=reachable or [],
            entities=entities,
        )

    def test_navigates_to_unreachable_target(self):
        entity = _EntityState(1, "tree-01", Position(x=50, y=50))
        wq = self._make_clear_wq([entity], reachable=[])
        task = _Task(
            task_type="clear_region",
            bbox=_BBox(0, 0, 100, 100),
            clear_mode="clear_natural",
        )
        agent = _make_agent()
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # StopMining
        agent.tick(task, bb, wq, _WW, 2, _KB)   # build list + pick target
        actions = agent.tick(task, bb, wq, _WW, 3, _KB)
        # Should issue MoveTo since entity is not reachable
        self.assertTrue(any(isinstance(a, MoveTo) for a in actions))
        self.assertEqual(agent._clear_phase, _ClearPhase.NAVIGATE)

    def test_destroys_reachable_target(self):
        entity = _EntityState(1, "tree-01", Position(x=5, y=5))
        wq = self._make_clear_wq([entity], reachable=[1])
        task = _Task(
            task_type="clear_region",
            bbox=_BBox(0, 0, 100, 100),
            clear_mode="clear_natural",
        )
        agent = _make_agent()
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # StopMining
        agent.tick(task, bb, wq, _WW, 2, _KB)   # build list + pick target
        actions = agent.tick(task, bb, wq, _WW, 3, _KB)
        # Entity reachable → MineEntity
        self.assertTrue(any(isinstance(a, MineEntity) for a in actions))
        self.assertEqual(agent._clear_phase, _ClearPhase.DESTROY)

    def test_advances_after_entity_destroyed(self):
        e1 = _EntityState(1, "tree-01", Position(x=5, y=5))
        e2 = _EntityState(2, "rock-01", Position(x=10, y=10))
        wq = self._make_clear_wq([e1, e2], reachable=[1, 2])
        task = _Task(
            task_type="clear_region",
            bbox=_BBox(0, 0, 100, 100),
            clear_mode="clear_natural",
        )
        agent = _make_agent()
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # StopMining
        agent.tick(task, bb, wq, _WW, 2, _KB)   # build list
        agent.tick(task, bb, wq, _WW, 3, _KB)   # MineEntity e1 issued
        # Remove e1 — simulate destruction
        wq.remove_entity(1)
        agent.tick(task, bb, wq, _WW, 4, _KB)   # detects gone, advances
        agent.tick(task, bb, wq, _WW, 5, _KB)   # picks e2
        actions = agent.tick(task, bb, wq, _WW, 6, _KB)
        # Should now be mining e2
        mine = [a for a in actions if isinstance(a, MineEntity)]
        if mine:
            self.assertEqual(mine[0].entity_id, 2)

    def test_skips_target_when_nav_stuck(self):
        from execution.skills.navigate import _UNREACHABLE_GRACE_TICKS
        entity = _EntityState(1, "tree-01", Position(x=500, y=500))
        wq = self._make_clear_wq([entity], reachable=[])
        task = _Task(
            task_type="clear_region",
            bbox=_BBox(0, 0, 1000, 1000),
            clear_mode="clear_natural",
        )
        agent = _make_agent()
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        agent.tick(task, bb, wq, _WW, 2, _KB)
        # Drive navigation to STUCK
        for i in range(1, _UNREACHABLE_GRACE_TICKS // 10 + 10):
            agent.tick(task, bb, wq, _WW, 2 + i * 10, _KB)
            if agent._nav_skill.status() in (
                SkillStatus.STUCK, SkillStatus.IDLE
            ) and agent._current_target is None:
                break
        # Target should be cleared (skipped)
        self.assertIsNone(agent._current_target)


# ===========================================================================
# Section 5 — observe() and progress()
# ===========================================================================

class TestMiningAgentObserveProgress(unittest.TestCase):

    def test_observe_keys_gather(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=10, y=10))
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        obs = agent.observe(task, bb, wq, _KB)
        for key in ("agent", "task_id", "task_kind", "mine_status"):
            self.assertIn(key, obs, f"missing key: {key}")

    def test_observe_keys_clear(self):
        agent = _make_agent()
        task = _Task(task_type="clear_region", bbox=_BBox(0, 0, 50, 50))
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        obs = agent.observe(task, bb, wq, _KB)
        for key in ("agent", "task_id", "task_kind",
                    "clear_targets_remaining", "clear_phase"):
            self.assertIn(key, obs, f"missing key: {key}")

    def test_progress_clear_zero_before_list_built(self):
        agent = _make_agent()
        task = _Task(task_type="clear_region", bbox=_BBox(0, 0, 50, 50))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        self.assertAlmostEqual(agent.progress(task, bb, _WQ(), _KB), 0.0)

    def test_progress_clear_increases_as_targets_cleared(self):
        agent = _make_agent()
        e1 = _EntityState(1, "tree-01", Position(x=5, y=5))
        e2 = _EntityState(2, "tree-02", Position(x=10, y=10))
        wq = _WQ(entities=[e1, e2], reachable=[1, 2])
        task = _Task(
            task_type="clear_region",
            bbox=_BBox(0, 0, 50, 50),
            clear_mode="clear_natural",
        )
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        agent.tick(task, bb, wq, _WW, 2, _KB)   # build list (2 targets)
        p_before = agent.progress(task, bb, wq, _KB)
        wq.remove_entity(1)
        agent.tick(task, bb, wq, _WW, 3, _KB)
        agent.tick(task, bb, wq, _WW, 4, _KB)
        p_after = agent.progress(task, bb, wq, _KB)
        self.assertGreater(p_after, p_before)


# ===========================================================================
# Section 6 — pending_patches()
# ===========================================================================

class TestMiningAgentPendingPatches(unittest.TestCase):

    def test_always_empty(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=10, y=10))
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        for i in range(1, 5):
            agent.tick(task, bb, wq, _WW, i, _KB)
        self.assertEqual(agent.pending_patches(), [])


# ===========================================================================
# _is_natural helper
# ===========================================================================

class TestIsNatural(unittest.TestCase):

    def _e(self, name):
        return _EntityState(entity_id=1, name=name)

    def test_tree_is_natural(self):
        self.assertTrue(_is_natural(self._e("tree-01")))

    def test_rock_is_natural(self):
        self.assertTrue(_is_natural(self._e("rock-huge")))

    def test_boulder_is_natural(self):
        self.assertTrue(_is_natural(self._e("boulder-medium")))

    def test_cliff_is_natural(self):
        self.assertTrue(_is_natural(self._e("cliff-explosives")))

    def test_fish_is_natural(self):
        self.assertTrue(_is_natural(self._e("fish")))

    def test_iron_chest_not_natural(self):
        self.assertFalse(_is_natural(self._e("iron-chest")))

    def test_assembling_machine_not_natural(self):
        self.assertFalse(_is_natural(self._e("assembling-machine-1")))

    def test_name_case_insensitive(self):
        self.assertTrue(_is_natural(self._e("TREE-DEAD-DRY")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
