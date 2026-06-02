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

from execution.agents.mining import MiningAgent, _TaskKind, _ClearPhase, _GatherPhase
from execution.blackboard import Blackboard, EntryCategory, EntryScope
from execution.skills.base import SkillStatus
from world import Position, NaturalObject
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
    entity_types: list = field(default_factory=list)
    item: str = ""
    created_at: int = 0


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
        natural_objects: list = None,
        tick: int = 1,
        crafting_queue_size: int = 0,
    ):
        self._position  = position or Position(x=0.0, y=0.0)
        self._reachable = reachable or []
        self._inventory = dict(inventory or {})
        self._entities  = {e.entity_id: e for e in (entities or [])}
        self._natural_objects = list(natural_objects or [])
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
        if eid in self._entities:
            return self._entities[eid]
        # Natural objects are also findable by entity_id so DestroySkill
        # can confirm they still exist before issuing MineEntity.
        for obj in self._natural_objects:
            if obj.entity_id == eid:
                return obj
        return None

    def inventory_count(self, item: str) -> int:
        return self._inventory.get(item, 0)

    @property
    def crafting_queue_size(self) -> int:
        return self._crafting_queue_size

    @property
    def natural_objects(self) -> list:
        return self._natural_objects

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
        self._natural_objects = [
            o for o in self._natural_objects if o.entity_id != eid
        ]

    def add_entity(self, e: _EntityState) -> None:
        self._entities[e.entity_id] = e
        self.state.entities = list(self._entities.values())

    def set_natural_objects(self, objs: list) -> None:
        self._natural_objects = list(objs)

    # Properties needed by build_full_namespace / RewardEvaluator
    @property
    def charted_tiles(self) -> int:
        return 0

    @property
    def charted_area_km2(self) -> float:
        return 0.0

    @property
    def logistics(self):
        return MagicMock()

    @property
    def power(self):
        return MagicMock()

    @property
    def threat(self):
        return MagicMock()

    @property
    def research(self):
        return MagicMock()

    def tech_unlocked(self, tech: str) -> bool:
        return False

    def resources_of_type(self, resource_type: str) -> list:
        return []

    def inserters_taking_from(self, *a, **kw): return []
    def inserters_delivering_to(self, *a, **kw): return []
    def inserters_taking_from_type(self, *a): return []
    def inserters_delivering_to_type(self, *a): return []

    def entities(self):
        from world.observable.query import EntityQuery
        return EntityQuery([], self)

    def entities_by_name(self, name: str) -> list:
        return []

    def entities_by_status(self, status) -> list:
        return []

    def in_bbox(self, x_min, y_min, x_max, y_max):
        from world.observable.query import BBoxQuery
        return BBoxQuery(self, x_min, y_min, x_max, y_max)

    def natural_objects_in_bbox(self, x_min, y_min, x_max, y_max):
        return [o for o in self._natural_objects
                if x_min <= o.position.x <= x_max and y_min <= o.position.y <= y_max]

    def section_staleness(self, section: str, tick: int):
        return 0

    def _tile_dims(self, name: str):
        return 1, 1


_WW = MagicMock()

def _gather_task(resource_type="iron-ore", count=10):
    """Module-level gather task helper for teardown tests."""
    return _Task(
        task_type="gather_resource",
        resource_type=resource_type,
        target_position=Position(x=10, y=10),
        count=count,
    )

def _clear_task(bbox=None):
    """Module-level clear task helper for teardown tests."""
    return _Task(
        task_type="clear_region",
        bbox=bbox or _BBox(0, 0, 100, 100),
    )

def _make_kb(minable=True):
    """KB stub whose get_entity() returns a proper minable record."""
    record = MagicMock()
    record.is_placeholder = False
    record.minable = minable
    kb = MagicMock()
    kb.get_entity.return_value = record
    return kb

_KB = _make_kb()   # default: all entities are minable


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
        """
        Set up a gather agent ready for MineResource testing.

        The mining agent now has two phases: NAVIGATE → MINE.
        After StopMining (tick 1), tick 2 starts NavigateSkill.
        We force NavigateSkill to SUCCEEDED so the agent transitions
        to the MINE phase, making MineResource appear on tick 3.
        """
        from execution.skills.navigate import NavigateSkill
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
        # Force the NAVIGATE→MINE phase transition directly without ticking
        # _gather_mine (which would issue MineResource and consume the first
        # issue). Set gather_phase directly so tick 2 is the first mine tick.
        agent._nav_skill._status = SkillStatus.SUCCEEDED
        agent._gather_phase = _GatherPhase.MINE
        return agent, task, bb, wq

    def test_issues_mine_resource_after_stop(self):
        agent, task, bb, wq = self._setup()
        # _setup() leaves agent in MINE phase; first mine tick issues MineResource
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
        agent.tick(task, bb, wq, _WW, tick + 1, _KB)
        self.assertEqual(len(_obs_of_type(bb, "gather_stuck")), 1)

    def test_progress_in_mine_phase_running(self):
        """After nav completes, MineSkill is RUNNING (started by activate()).
        progress() returns 0.6 (MINE phase, RUNNING, not yet SUCCEEDED)."""
        agent, task, bb, wq = self._setup()
        self.assertAlmostEqual(agent.progress(task, bb, wq, _KB), 0.6)

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

    def test_builds_target_list_from_natural_objects(self):
        agent = _make_agent()
        natural = [
            NaturalObject(1, "tree-01",   Position(x=10, y=10), "neutral", "tree"),
            NaturalObject(2, "rock-huge", Position(x=20, y=20), "neutral", "simple-entity"),
        ]
        # iron-chest is a player entity — not in natural_objects
        entities = [_EntityState(3, "iron-chest", Position(x=30, y=30))]
        wq = _WQ(entities=entities, natural_objects=natural)
        task = self._clear_task(mode="clear_natural")
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # StopMining
        agent.tick(task, bb, wq, _WW, 2, _KB)   # build target list
        self.assertEqual(agent._targets_total, 2)

    def test_clear_all_includes_both_natural_and_player_built(self):
        agent = _make_agent()
        natural = [NaturalObject(1, "tree-01", Position(x=10, y=10), "neutral", "tree")]
        entities = [_EntityState(2, "iron-chest", Position(x=20, y=20))]
        wq = _WQ(entities=entities, natural_objects=natural)
        task = self._clear_task(mode="clear_all")
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertEqual(agent._targets_total, 2)

    def test_excludes_objects_outside_bbox(self):
        agent = _make_agent()
        natural = [
            NaturalObject(1, "tree-01", Position(x=10, y=10),   "neutral", "tree"),
            NaturalObject(2, "tree-02", Position(x=200, y=200), "neutral", "tree"),
        ]
        wq = _WQ(natural_objects=natural)
        task = self._clear_task(bbox=_BBox(0, 0, 50, 50))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertEqual(agent._targets_total, 1)

    def test_empty_region_returns_no_actions(self):
        agent = _make_agent()
        wq = _WQ()   # no entities, no natural objects
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

    def _make_clear_natural_wq(self, natural_objects, reachable=None):
        return _WQ(
            position=Position(x=0, y=0),
            reachable=reachable or [],
            natural_objects=natural_objects,
        )

    def test_navigates_to_unreachable_natural_target(self):
        nat = [NaturalObject(1, "tree-01", Position(x=50, y=50), "neutral", "tree")]
        wq = self._make_clear_natural_wq(nat, reachable=[])
        task = _Task(
            task_type="clear_region",
            bbox=_BBox(0, 0, 100, 100),
            clear_mode="clear_natural",
        )
        agent = _make_agent()
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)      # StopMining
        actions = agent.tick(task, bb, wq, _WW, 2, _KB)  # build list + navigate
        self.assertTrue(any(isinstance(a, MoveTo) for a in actions))
        self.assertEqual(agent._clear_phase, _ClearPhase.NAVIGATE)

    def test_destroys_reachable_natural_target(self):
        nat = [NaturalObject(1, "tree-01", Position(x=5, y=5), "neutral", "tree")]
        wq = _WQ(
            position=Position(x=0, y=0),
            reachable=[1],
            natural_objects=nat,
        )
        task = _Task(
            task_type="clear_region",
            bbox=_BBox(0, 0, 100, 100),
            clear_mode="clear_natural",
        )
        agent = _make_agent()
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)      # StopMining
        actions = agent.tick(task, bb, wq, _WW, 2, _KB)  # build list + destroy
        self.assertTrue(any(isinstance(a, MineEntity) for a in actions))
        self.assertEqual(agent._clear_phase, _ClearPhase.DESTROY)

    def test_advances_after_natural_target_destroyed(self):
        nat = [
            NaturalObject(1, "tree-01", Position(x=5, y=5),   "neutral", "tree"),
            NaturalObject(2, "rock-01", Position(x=10, y=10), "neutral", "simple-entity"),
        ]
        wq = self._make_clear_natural_wq(nat, reachable=[1, 2])
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
        agent.tick(task, bb, wq, _WW, 3, _KB)   # MineEntity 1 issued
        # Simulate destruction of entity 1 — remove from entity_by_id
        wq.remove_entity(1)
        agent.tick(task, bb, wq, _WW, 4, _KB)   # detects gone, advances
        agent.tick(task, bb, wq, _WW, 5, _KB)   # picks entity 2
        actions = agent.tick(task, bb, wq, _WW, 6, _KB)
        mine = [a for a in actions if isinstance(a, MineEntity)]
        if mine:
            self.assertEqual(mine[0].entity_id, 2)

    def test_skips_target_when_nav_stuck(self):
        from execution.skills.navigate import _UNREACHABLE_GRACE_TICKS
        nat = [NaturalObject(1, "tree-01", Position(x=500, y=500), "neutral", "tree")]
        wq = self._make_clear_natural_wq(nat, reachable=[])
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
        for i in range(1, _UNREACHABLE_GRACE_TICKS // 10 + 10):
            agent.tick(task, bb, wq, _WW, 2 + i * 10, _KB)
            if (agent._nav_skill.status() in (SkillStatus.STUCK, SkillStatus.IDLE)
                    and agent._current_target is None):
                break
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
        # gather observe includes gather_phase plus nav/mine skill sub-keys
        for key in ("agent", "task_id", "task_kind", "gather_phase"):
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
        nat = [
            NaturalObject(1, "tree-01", Position(x=5, y=5),   "neutral", "tree"),
            NaturalObject(2, "tree-02", Position(x=10, y=10), "neutral", "tree"),
        ]
        wq = _WQ(natural_objects=nat, reachable=[1, 2])
        task = _Task(
            task_type="clear_region",
            bbox=_BBox(0, 0, 50, 50),
            clear_mode="clear_natural",
        )
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)     # StopMining
        agent.tick(task, bb, wq, _WW, 2, _KB)     # build list + pick first target
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
# Section — harvest_natural task
# ===========================================================================

class TestMiningAgentHarvest(unittest.TestCase):
    """
    harvest_natural task: mine natural objects by entity type to collect drops.
    Uses the same NAVIGATE→DESTROY machinery as clear_region but loops
    continuously until the task success_condition fires externally.
    """

    def _make_natural_obj(self, name="tree-01", entity_id=1,
                           x=5.0, y=5.0, is_minable=True):
        obj = MagicMock()
        obj.name = name
        obj.entity_id = entity_id
        obj.position = Position(x=x, y=y)
        obj.is_minable = is_minable
        return obj

    def _harvest_task(self, entity_types=None):
        return _Task(
            task_type="harvest_natural",
            entity_types=entity_types or ["tree-01", "tree-02"],
            item="wood",
            count=5,
        )

    def test_harvest_task_sets_harvest_kind(self):
        agent = _make_agent()
        task = self._harvest_task()
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        self.assertEqual(agent._task_kind, _TaskKind.HARVEST)

    def test_harvest_entity_types_stored(self):
        agent = _make_agent()
        task = self._harvest_task(entity_types=["tree-01", "tree-04"])
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._harvest_entity_types, ["tree-01", "tree-04"])

    def test_harvest_task_in_task_types(self):
        self.assertIn("harvest_natural", MiningAgent.TASK_TYPES)

    def test_harvest_tick_returns_empty_when_no_natural_objects(self):
        agent = _make_agent()
        task = self._harvest_task()
        bb = _make_bb()
        wq = _WQ(natural_objects=[])
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # drain StopMining
        actions = agent.tick(task, bb, wq, _WW, 2, _KB)
        # No natural objects in scan → waits
        self.assertFalse(any(isinstance(a, MineResource) for a in actions))

    def test_harvest_ignores_wrong_entity_type(self):
        """Natural objects not in entity_types are ignored."""
        agent = _make_agent()
        task = self._harvest_task(entity_types=["tree-01"])
        wrong_obj = self._make_natural_obj(name="rock-big", entity_id=99)
        bb = _make_bb()
        wq = _WQ(natural_objects=[wrong_obj])
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # StopMining
        actions = agent.tick(task, bb, wq, _WW, 2, _KB)
        # Wrong type — no target selected, no actions
        self.assertIsNone(agent._current_target)

    def test_harvest_selects_matching_entity_type(self):
        """Natural objects matching entity_types are queued as targets."""
        agent = _make_agent()
        task = self._harvest_task(entity_types=["tree-01"])
        tree = self._make_natural_obj(name="tree-01", entity_id=7,
                                       x=10.0, y=10.0)
        bb = _make_bb()
        wq = _WQ(natural_objects=[tree])
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # StopMining
        agent.tick(task, bb, wq, _WW, 2, _KB)   # target selected, nav starts
        self.assertIsNotNone(agent._current_target)
        self.assertEqual(agent._current_target.entity_id, 7)

    def test_harvest_skips_non_minable(self):
        """Non-minable natural objects are filtered out by _build_harvest_targets.
        can_destroy() is KB-authoritative — use a KB that returns minable=False."""
        agent = _make_agent()
        task = self._harvest_task(entity_types=["cliff"])
        cliff = self._make_natural_obj(name="cliff", entity_id=3,
                                        is_minable=False)
        kb_cliff = _make_kb(minable=False)   # KB says cliff is not minable
        bb = _make_bb()
        wq = _WQ(natural_objects=[cliff])
        agent.activate(task, bb, wq, kb_cliff)
        agent.tick(task, bb, wq, _WW, 1, kb_cliff)   # StopMining
        agent.tick(task, bb, wq, _WW, 2, kb_cliff)
        self.assertIsNone(agent._current_target)

    def test_harvest_teardown_returns_stop_mining(self):
        agent = _make_agent()
        task = self._harvest_task()
        agent.activate(task, _make_bb(), _WQ(), _KB)
        actions = agent.teardown()
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], StopMining)

    def test_harvest_observe_includes_entity_types(self):
        agent = _make_agent()
        task = self._harvest_task(entity_types=["tree-01"])
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        obs = agent.observe(task, bb, _WQ(), _KB)
        self.assertIn("harvest_entity_types", obs)
        self.assertIn("tree-01", obs["harvest_entity_types"])

    def test_harvest_advances_when_entity_disappears(self):
        """When current target entity_id is gone from scan, agent picks next."""
        agent = _make_agent()
        task = self._harvest_task(entity_types=["tree-01"])
        tree = self._make_natural_obj(name="tree-01", entity_id=42, x=5.0, y=5.0)
        bb = _make_bb()
        wq = _WQ(natural_objects=[tree])
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # StopMining
        agent.tick(task, bb, wq, _WW, 2, _KB)   # picks tree as target

        self.assertIsNotNone(agent._current_target)
        self.assertEqual(agent._current_target.entity_id, 42)

        # Entity disappears — entity_by_id returns None
        wq2 = _WQ(natural_objects=[])   # empty scan, entity gone
        agent.tick(task, bb, wq2, _WW, 3, _KB)  # should advance (clear target)
        self.assertIsNone(agent._current_target)


# ===========================================================================
# Section — teardown()
# ===========================================================================

class TestMiningAgentTeardown(unittest.TestCase):
    """
    MiningAgent.teardown() is called by the coordinator when a task resolves.

    Contract:
      - GATHER task: always returns [StopMining] (Lua miner may be active)
      - CLEAR task:  returns [] (no persistent miner involved)
      - _pending_stop is always cleared, regardless of task kind
    """

    def test_gather_task_teardown_returns_stop_mining(self):
        """StopMining is always issued after a gather task."""
        agent = _make_agent()
        task = _gather_task()
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        actions = agent.teardown()
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], StopMining)

    def test_gather_teardown_even_after_skill_succeeded(self):
        """StopMining is safe to send even when MineSkill already succeeded.
        The Lua mod treats a redundant StopMining as a no-op."""
        agent = _make_agent()
        task = _gather_task()
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        # Force skill to succeeded state
        agent._mine_skill._status = SkillStatus.SUCCEEDED
        actions = agent.teardown()
        self.assertIsInstance(actions[0], StopMining)

    def test_clear_task_teardown_returns_empty(self):
        """Clear tasks don't use MineSkill persistently — no StopMining needed."""
        agent = _make_agent()
        task = _clear_task()
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        actions = agent.teardown()
        self.assertEqual(actions, [])

    def test_teardown_clears_pending_stop(self):
        """_pending_stop is always cleared after teardown."""
        agent = _make_agent()
        task = _gather_task()
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        agent._pending_stop = True
        agent.teardown()
        self.assertFalse(agent._pending_stop)

    def test_teardown_before_activate_returns_empty(self):
        """teardown() before any task is assigned is safe and returns []."""
        agent = _make_agent()
        actions = agent.teardown()
        self.assertEqual(actions, [])



if __name__ == "__main__":
    unittest.main(verbosity=2)