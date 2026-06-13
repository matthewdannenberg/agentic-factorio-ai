"""
tests/unit/execution/test_coordinator.py

Comprehensive unit tests for execution/coordinator/coordinator.py.

Test organisation
-----------------
1.  Stubs and helpers
2.  GoalFrame dataclass
3.  reset() — state clearing, goal stack initialisation
4.  Tick loop — empty stack, task running, task succeeded/failed/stuck,
    goal completion propagation, sub-goal failure propagation
5.  _handle_collection — already-satisfied, no patches, navigates+gathers,
    gather fails, re-derives after partial gather
6.  _handle_acquire — already-satisfied, natural resource delegates to
    collection, producible but unimplemented, no source
7.  _handle_crafting — already-satisfied, missing ingredients triggers
    acquire sub-goals, crafts when ingredients present, verifies after craft
8.  _handle_explore — target already met, pushes explore task, needs-frontier
    signal triggers re-derive, fallback frontier
9.  _handle_clear_region — undestroyable blocker fails, pushes clear task,
    verifies empty after task
10. _handle_prep_region — factory intersection fails, undestroyable fails,
    logistics warning falls through, delegates to clear_region sub-goal
11. _handle_construction — skips resource check stub, preps region, build stub
12. Stub handlers — production/logistics/byproduct return STUCK,
    research queues tech, noop returns WAITING
13. _push_task — task created with correct attributes, agent activated,
    blackboard entry written, missing agent handled
14. _push_goal — frame pushed onto stack correctly
15. _tick_task — success condition fires, failure condition fires,
    STUCK from agent observe, agent ticked when running
16. drain_patches — empty initially, drained from agent after tick
17. Module-level helpers — _dist, _item_in_nearby_chest, _nearest_frontier,
    _undestroyable_in_bbox, _bbox_is_clear, _bbox_empty_condition,
    _intersects_major_factory, _intersects_logistics

Run with:  python -m pytest tests/unit/execution/test_coordinator.py -v
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

from execution.coordinator.coordinator import (
    RuleBasedCoordinator,
    CoordinatorStatus,
    TaskOutcome,
    GOAL_COLLECTION, GOAL_ACQUIRE, GOAL_CRAFTING, GOAL_EXPLORE,
    GOAL_CLEAR_REGION, GOAL_PREP_REGION, GOAL_CONSTRUCTION,
    GOAL_PRODUCTION, GOAL_LOGISTICS, GOAL_BYPRODUCT, GOAL_RESEARCH,
    GOAL_NOOP,
    _dist, _item_in_nearby_chest,
    _undestroyable_in_bbox, _bbox_is_clear, _bbox_empty_condition,
    _intersects_major_factory, _intersects_logistics,
    _TASK_TIMEOUT_TICKS,
)
from planning.goals.goal import Goal
from planning.planning_item import ItemStatus
from execution.blackboard import Blackboard, EntryCategory, EntryScope
from world import Position, EntityState
from world.observable.state import ChunkCoord


# ===========================================================================
# Section 1 — Stubs and helpers
# ===========================================================================

@dataclass
class _BBox:
    x_min: float; y_min: float; x_max: float; y_max: float

    def __getitem__(self, key: str) -> float:
        return getattr(self, key)


@dataclass
class _ResourcePatch:
    position: Position
    amount: int = 50000
    size: int = 100


@dataclass
class _EntityState:
    entity_id: int
    name: str = "assembling-machine-1"
    prototype_type: str = "assembling-machine"
    position: Position = field(default_factory=lambda: Position(0.0, 0.0))
    force: str = "player"
    is_natural: bool = False


@dataclass
class _Slot:
    item: str
    count: int


class _WQ:
    """Minimal WorldQuery stub for coordinator tests."""

    def __init__(
        self,
        position: Position = None,
        inventory: dict = None,
        resources: dict = None,
        entities: list = None,
        natural_objects: list = None,
        nearby_uncharted: list = None,
        charted: int = 10,
        tech_unlocked: set = None,
        tick: int = 1,
    ):
        self._position         = position or Position(0.0, 0.0)
        self._inventory        = dict(inventory or {})
        self._resources        = resources or {}
        self._entities         = list(entities or [])
        self._natural_objects  = list(natural_objects or [])
        self._nearby_uncharted = list(nearby_uncharted or [])
        self._charted          = charted
        self._tech_unlocked    = set(tech_unlocked or [])
        self.tick              = tick

        self.state = MagicMock()
        self.state.entities = self._entities
        self.state.player.inventory.slots = [
            _Slot(k, v) for k, v in self._inventory.items() if v > 0
        ]

    def player_position(self) -> Position:
        return self._position

    def inventory_count(self, item: str) -> int:
        return self._inventory.get(item, 0)

    def resources_of_type(self, resource_type: str) -> list:
        return self._resources.get(resource_type, [])

    @property
    def natural_objects(self) -> list:
        return self._natural_objects

    @property
    def nearby_uncharted_chunks(self) -> list:
        return self._nearby_uncharted

    @property
    def charted_chunks(self) -> int:
        return self._charted

    def tech_unlocked(self, tech: str) -> bool:
        return tech in self._tech_unlocked

    # Properties added for build_full_namespace / RewardEvaluator compatibility
    @property
    def charted_tiles(self) -> int:
        return self._charted * 1024

    @property
    def charted_area_km2(self) -> float:
        return self._charted * 1024 / 1_000_000

    @property
    def logistics(self):
        return MagicMock()

    @property
    def power(self):
        return MagicMock()

    @property
    def threat(self):
        return MagicMock()

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

    def entity_by_id(self, eid: int):
        return None

    def _tile_dims(self, name: str):
        return 1, 1

    def in_bbox(self, x_min, y_min, x_max, y_max):
        from world.observable.query import BBoxQuery
        return BBoxQuery(self, x_min, y_min, x_max, y_max)

    def natural_objects_in_bbox(self, x_min, y_min, x_max, y_max):
        return [o for o in self._natural_objects
                if x_min <= o.position.x <= x_max and y_min <= o.position.y <= y_max]

    def section_staleness(self, section: str, tick: int):
        return 0

    @property
    def research(self):
        return MagicMock()


_WW = MagicMock()


def _make_registry(agents: dict = None):
    agents = agents or {}
    registry = MagicMock()
    registry.agent_by_id.side_effect = lambda hint: agents.get(hint)
    registry.all_agents.return_value = list(agents.values())
    return registry


def _make_agent(skill_status: str = "RUNNING") -> MagicMock:
    agent = MagicMock()
    agent.activate.return_value = None
    agent.tick.return_value = []
    agent.observe.return_value = {"skill_status": skill_status}
    agent.pending_patches.return_value = []
    return agent


def _make_kb(
    recipes: dict = None,
    entities: dict = None,
    harvestable: dict = None,
) -> MagicMock:
    kb = MagicMock()
    recipes    = recipes    or {}
    entities   = entities   or {}
    harvestable = harvestable or {}

    def recipes_for_product(item):
        return recipes.get(item, [])

    def get_entity(name):
        rec = entities.get(name)
        if rec is None:
            r = MagicMock()
            r.is_placeholder = True
            r.minable = False
            return r
        return rec

    def entities_that_produce(item):
        return harvestable.get(item, [])

    kb.recipes_for_product.side_effect = recipes_for_product
    kb.get_entity.side_effect = get_entity
    kb.entities_that_produce.side_effect = entities_that_produce
    return kb


def _minable_kb(names=None):
    names = names or []
    ents = {}
    for name in names:
        rec = MagicMock()
        rec.is_placeholder = False
        rec.minable = True
        ents[name] = rec
    return _make_kb(entities=ents)


def _make_coord(agents: dict = None, kb=None) -> RuleBasedCoordinator:
    return RuleBasedCoordinator(
        registry=_make_registry(agents or {}),
        kb=kb or _make_kb(),
    )


def _nat(entity_id, name="tree-01", x=5.0, y=5.0, proto="tree"):
    return EntityState(
        entity_id=entity_id, name=name,
        position=Position(x=x, y=y),
        force="neutral", prototype_type=proto,
        is_natural=True,
    )


def _inject_task(coord, success="", failure=""):
    """Inject a mock Task directly onto the coordinator's stack."""
    from planning.tasks.task import Task
    t = Task(
        description=("injected task"),
        success_condition=success,
        failure_condition=failure,
        agent_hint="navigation",
        task_type="test",
        status=ItemStatus.ACTIVE,
        created_at=0,
    )
    coord._stack.append(t)
    coord._active_agent = _make_agent("RUNNING")
    return t


# ===========================================================================
# Section 2 — Goal (replaces GoalFrame)
# ===========================================================================

def _make_goal(goal_type=GOAL_COLLECTION, params=None, step=0,
               status=None, success_condition=""):
    """Helper: construct a Goal with coordinator routing fields set."""
    g = Goal(
        description=goal_type,
        success_condition=success_condition,
        failure_condition="",
        status=status or ItemStatus.ACTIVE,
        step=step,
    )
    g.goal_type = goal_type
    g.params    = params or {}
    return g


class TestGoalAsFrame(unittest.TestCase):
    """Goal carries the step/context/params that GoalFrame used to."""

    def test_defaults(self):
        g = _make_goal(GOAL_COLLECTION)
        self.assertEqual(g.goal_type, GOAL_COLLECTION)
        self.assertEqual(g.step, 0)
        self.assertEqual(g.status, ItemStatus.ACTIVE)
        self.assertEqual(g.params, {})
        self.assertEqual(g.context, {})

    def test_params_independent_across_instances(self):
        g1 = _make_goal(GOAL_COLLECTION, params={"item": "iron-ore"})
        g2 = _make_goal(GOAL_COLLECTION, params={"item": "coal"})
        self.assertIsNot(g1.params, g2.params)

    def test_step_can_be_incremented(self):
        g = _make_goal(GOAL_COLLECTION)
        g.step += 1
        self.assertEqual(g.step, 1)

    def test_context_is_mutable(self):
        g = _make_goal(GOAL_COLLECTION)
        g.context["patch"] = Position(x=10, y=10)
        self.assertIn("patch", g.context)


# ===========================================================================
# Section 3 — reset()
# ===========================================================================

class TestCoordinatorReset(unittest.TestCase):

    def test_clears_goal_stack_and_pushes_new_goal(self):
        coord = _make_coord()
        coord._stack.append(_make_goal(GOAL_NOOP))
        coord._stack.append(_make_goal(GOAL_NOOP))
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 10}, _WQ())
        self.assertEqual(len(coord._stack), 1)
        self.assertEqual(coord._stack[0].goal_type, GOAL_COLLECTION)

    def test_clears_active_task(self):
        coord = _make_coord()
        coord._stack.append(_inject_task(coord))
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 10}, _WQ())
        self.assertFalse(any(hasattr(i, "agent_hint") and i.agent_hint for i in coord._stack))

    def test_clears_active_agent(self):
        coord = _make_coord()
        coord._active_agent = MagicMock()
        coord.reset(GOAL_NOOP, {}, _WQ())
        self.assertIsNone(coord._active_agent)

    def test_clears_blackboard(self):
        coord = _make_coord()
        coord._bb.write(EntryCategory.OBSERVATION, EntryScope.TASK,
                        "test", 0, {"x": 1})
        coord.reset(GOAL_NOOP, {}, _WQ())
        self.assertEqual(coord._bb.read(), [])

    def test_clears_pending_patches(self):
        coord = _make_coord()
        coord._pending_patches.append(MagicMock())
        coord.reset(GOAL_NOOP, {}, _WQ())
        self.assertEqual(coord._pending_patches, [])

    def test_goal_frame_has_correct_params(self):
        coord = _make_coord()
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 50}, _WQ())
        f = coord._stack[0]
        self.assertEqual(f.params["item"], "iron-ore")
        self.assertEqual(f.params["count"], 50)

    def test_goal_frame_starts_at_step_zero(self):
        coord = _make_coord()
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 10}, _WQ())
        self.assertEqual(coord._stack[0].step, 0)


# ===========================================================================
# Section 4 — Tick loop mechanics
# ===========================================================================

class TestTickLoopMechanics(unittest.TestCase):

    def test_empty_stack_returns_complete(self):
        coord = _make_coord()
        status, actions = coord.tick(_WQ(), _WW, 1)
        self.assertEqual(status, CoordinatorStatus.COMPLETE)
        self.assertEqual(actions, [])

    def test_active_task_running_returns_progressing(self):
        agent = _make_agent("RUNNING")
        agent.tick.return_value = [MagicMock()]
        coord = _make_coord(agents={"navigation": agent})
        coord.reset(GOAL_NOOP, {}, _WQ())
        _inject_task(coord)
        coord._active_agent = agent
        status, actions = coord.tick(_WQ(), _WW, 1)
        self.assertEqual(status, CoordinatorStatus.PROGRESSING)

    def test_task_success_advances_goal_step(self):
        coord = _make_coord()
        coord.reset(GOAL_NOOP, {}, _WQ())
        _inject_task(coord, success="True")
        coord.tick(_WQ(), _WW, 1)
        # _active_task is cleared after task resolves; NOOP handler does not
        # push a new task so it stays None.
        self.assertFalse(any(isinstance(i, type(MagicMock())) for i in coord._stack))
        self.assertEqual(coord._stack[-1].step, 1)

    def test_task_failure_returns_stuck(self):
        # When a task fails, the goal is marked failed, the while loop pops
        # it, and the coordinator returns STUCK (empty stack after pop).
        coord = _make_coord()
        coord.reset(GOAL_NOOP, {}, _WQ())
        _inject_task(coord, failure="True")
        status, _ = coord.tick(_WQ(), _WW, 1)
        self.assertEqual(status, CoordinatorStatus.STUCK)

    def test_task_stuck_returns_stuck(self):
        # STUCK agent → goal marked failed → while loop pops it → STUCK status
        agent = _make_agent("STUCK")
        coord = _make_coord()
        coord.reset(GOAL_NOOP, {}, _WQ())
        _inject_task(coord)
        coord._active_agent = agent
        status, _ = coord.tick(_WQ(), _WW, 1)
        self.assertEqual(status, CoordinatorStatus.STUCK)

    def test_task_scope_cleared_after_task_resolves(self):
        # TASK-scoped entries written before resolution are cleared.
        # The NOOP handler doesn't write any new TASK entries, so the scope
        # should be empty after the tick.
        coord = _make_coord()
        coord.reset(GOAL_NOOP, {}, _WQ())
        coord._bb.write(EntryCategory.OBSERVATION, EntryScope.TASK,
                        "agent", 1, {"type": "test"})
        _inject_task(coord, success="True")
        coord.tick(_WQ(), _WW, 1)
        # Only OBSERVATION entries should be cleared; NOOP writes nothing.
        obs_entries = coord._bb.read(
            category=EntryCategory.OBSERVATION, scope=EntryScope.TASK
        )
        self.assertEqual(obs_entries, [])

    def test_completed_goal_pops_stack(self):
        coord = _make_coord()
        coord._stack.append(_make_goal(GOAL_COLLECTION, status=ItemStatus.COMPLETE))
        coord.tick(_WQ(), _WW, 1)
        self.assertEqual(len(coord._stack), 0)

    def test_completed_stack_returns_complete(self):
        coord = _make_coord()
        coord._stack.append(_make_goal(GOAL_COLLECTION, status=ItemStatus.COMPLETE))
        status, _ = coord.tick(_WQ(), _WW, 1)
        self.assertEqual(status, CoordinatorStatus.COMPLETE)

    def test_completed_subgoal_advances_parent_step(self):
        # Use NOOP as parent so the handler doesn't push further sub-goals.
        coord = _make_coord()
        parent = _make_goal(GOAL_NOOP)
        child  = _make_goal(GOAL_COLLECTION,
                            params={"item": "iron-ore", "count": 5},
                            status=ItemStatus.COMPLETE)
        coord._stack.extend([parent, child])
        coord.tick(_WQ(), _WW, 1)
        # child popped, parent.step advanced to 1, NOOP handler runs (WAITING)
        self.assertEqual(coord._stack[0].step, 1)
        self.assertEqual(coord._stack[0].goal_type, GOAL_NOOP)

    def test_failed_subgoal_propagates_to_parent_and_returns_stuck(self):
        # Child fails → parent.failed = True → parent popped → STUCK status.
        coord = _make_coord()
        parent = _make_goal(GOAL_NOOP)
        child  = _make_goal(GOAL_COLLECTION,
                            params={"item": "iron-ore", "count": 5},
                            status=ItemStatus.FAILED)
        coord._stack.extend([parent, child])
        status, _ = coord.tick(_WQ(), _WW, 1)
        self.assertEqual(status, CoordinatorStatus.STUCK)

    def test_failed_top_level_goal_returns_stuck(self):
        coord = _make_coord()
        coord._stack.append(_make_goal(GOAL_COLLECTION, status=ItemStatus.FAILED))
        status, _ = coord.tick(_WQ(), _WW, 1)
        self.assertEqual(status, CoordinatorStatus.STUCK)

    def test_unknown_goal_type_returns_stuck(self):
        coord = _make_coord()
        coord.reset("unknown_goal_xyz", {}, _WQ())
        status, _ = coord.tick(_WQ(), _WW, 1)
        self.assertEqual(status, CoordinatorStatus.STUCK)

    def test_multiple_completed_subgoals_chain_correctly(self):
        # g3(completed) pops → g2.step += 1; g2(completed) pops → g1.step += 1
        # Both pops happen in the same while-loop pass, so g1.step ends at 1
        # (incremented once per completed frame that pops below it).
        # Each frame that pops advances the immediate parent by 1.
        coord = _make_coord()
        g1 = _make_goal(GOAL_NOOP)
        g2 = _make_goal(GOAL_ACQUIRE, params={"item": "x", "count": 1},
                        status=ItemStatus.COMPLETE)
        g3 = _make_goal(GOAL_COLLECTION, params={"item": "x", "count": 1},
                        status=ItemStatus.COMPLETE)
        coord._stack.extend([g1, g2, g3])
        coord.tick(_WQ(), _WW, 1)
        # Both g3 and g2 pop; g1 is the only frame remaining
        self.assertEqual(len(coord._stack), 1)
        self.assertEqual(coord._stack[0].goal_type, GOAL_NOOP)
        # g1.step advanced once (from g2 popping, which itself was advanced by g3)
        self.assertEqual(coord._stack[0].step, 1)


# ===========================================================================
# Section 5 — _handle_collection
# ===========================================================================

class TestHandleCollection(unittest.TestCase):

    def test_already_satisfied_completes_immediately(self):
        coord = _make_coord()
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 10}, _WQ())
        coord.tick(_WQ(inventory={"iron-ore": 10}), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.COMPLETE)

    def test_excess_inventory_also_completes(self):
        coord = _make_coord()
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 5}, _WQ())
        coord.tick(_WQ(inventory={"iron-ore": 100}), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.COMPLETE)

    def test_no_patches_marks_failed(self):
        coord = _make_coord()
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 10}, _WQ())
        coord.tick(_WQ(), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.FAILED)

    def test_patch_found_activates_gather_task_directly(self):
        # MiningAgent handles its own approach navigation — the coordinator
        # pushes a single gather_resource task, not a separate navigate first.
        mine = _make_agent("RUNNING")
        coord = _make_coord(agents={"mining": mine})
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 10}, _WQ())
        wq = _WQ(resources={"iron-ore": [_ResourcePatch(Position(x=100, y=100))]})
        coord.tick(wq, _WW, 1)
        self.assertTrue(any(hasattr(i, "task_type") for i in coord._stack))
        self.assertEqual(coord._stack[-1].task_type, "gather_resource")

    def test_navigate_task_target_position_set(self):
        nav = _make_agent("RUNNING")
        coord = _make_coord(agents={"navigation": nav, "mining": _make_agent()})
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 10}, _WQ())
        patch_pos = Position(x=55, y=77)
        wq = _WQ(resources={"iron-ore": [_ResourcePatch(patch_pos)]})
        coord.tick(wq, _WW, 1)
        self.assertAlmostEqual(coord._stack[-1].target_position.x, 55.0)
        self.assertAlmostEqual(coord._stack[-1].target_position.y, 77.0)

    def test_nearest_patch_selected(self):
        nav = _make_agent("RUNNING")
        coord = _make_coord(agents={"navigation": nav, "mining": _make_agent()})
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 10}, _WQ())
        near = _ResourcePatch(Position(x=10, y=10))
        far  = _ResourcePatch(Position(x=500, y=500))
        wq = _WQ(position=Position(x=0, y=0),
                 resources={"iron-ore": [far, near]})
        coord.tick(wq, _WW, 1)
        self.assertAlmostEqual(
            coord._stack[0].context["patch_pos"].x, 10.0
        )

    def test_patch_context_stored(self):
        nav = _make_agent("RUNNING")
        coord = _make_coord(agents={"navigation": nav, "mining": _make_agent()})
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 10}, _WQ())
        wq = _WQ(resources={"iron-ore": [_ResourcePatch(Position(x=33, y=44))]})
        coord.tick(wq, _WW, 1)
        ctx = coord._stack[0].context
        self.assertAlmostEqual(ctx["patch_pos"].x, 33.0)
        self.assertAlmostEqual(ctx["patch_pos"].y, 44.0)

    def test_gather_task_has_resource_type_and_position(self):
        # Single gather_resource task — no separate nav task.
        mine = _make_agent("RUNNING")
        coord = _make_coord(agents={"mining": mine})
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 5}, _WQ())
        wq = _WQ(resources={"iron-ore": [_ResourcePatch(Position(x=10, y=10))]})
        coord.tick(wq, _WW, 1)
        self.assertEqual(coord._stack[-1].task_type, "gather_resource")
        self.assertEqual(coord._stack[-1].resource_type, "iron-ore")
        self.assertAlmostEqual(coord._stack[-1].target_position.x, 10.0)

    def test_gather_task_resource_type_set_for_coal(self):
        mine = _make_agent("RUNNING")
        coord = _make_coord(agents={"mining": mine})
        coord.reset(GOAL_COLLECTION, {"item": "coal", "count": 20}, _WQ())
        wq = _WQ(resources={"coal": [_ResourcePatch(Position(x=10, y=10))]})
        coord.tick(wq, _WW, 1)
        self.assertEqual(coord._stack[-1].resource_type, "coal")

    def test_inventory_satisfied_after_gather_completes_goal(self):
        mine = _make_agent("RUNNING")
        coord = _make_coord(agents={"mining": mine})
        coord.reset(GOAL_COLLECTION, {"item": "iron-ore", "count": 5}, _WQ())
        patch = _ResourcePatch(Position(x=10, y=10))
        wq = _WQ(resources={"iron-ore": [patch]})
        coord.tick(wq, _WW, 1)   # activates gather_resource task
        # Succeed the gather task
        coord._stack[-1].success_condition = "True"
        coord._stack[-1].failure_condition = ""
        wq_done = _WQ(inventory={"iron-ore": 5},
                      resources={"iron-ore": [patch]})
        coord.tick(wq_done, _WW, 2)
        self.assertEqual(coord._stack[0].status, ItemStatus.COMPLETE)

    def test_step_resets_to_zero_if_insufficient_after_gather(self):
        coord = _make_coord()
        frame = _make_goal(GOAL_COLLECTION, params={"item": "iron-ore", "count": 10}, step=1)
        coord._stack.append(frame)
        wq = _WQ(inventory={"iron-ore": 2})   # not enough
        coord.tick(wq, _WW, 1)
        self.assertEqual(frame.step, 0)


# ===========================================================================
# Section — _handle_collection Path B (harvest_natural)
# ===========================================================================

def _make_harvestable_entity(name="tree-01"):
    rec = MagicMock()
    rec.name = name
    rec.minable = True
    rec.is_placeholder = False
    return rec


class TestHandleCollectionHarvestPath(unittest.TestCase):
    """Path B of _handle_collection: no resource patch, but KB has entities
    that drop the item → push harvest_natural task."""

    def _coord_with_wood_harvestable(self, tree_names=("tree-01",)):
        trees = [_make_harvestable_entity(n) for n in tree_names]
        kb = _make_kb(harvestable={"wood": trees})
        # Must include a mining agent so _push_task("mining") can activate.
        return _make_coord(agents={"mining": _make_agent()}, kb=kb)

    def _activate_wood_goal(self, coord):
        coord.reset(GOAL_COLLECTION, {}, _WQ(),
                    success_condition="new.inventory('wood') >= 5")
        coord._stack[0].params = {"item": "wood", "count": 5}

    def test_harvest_task_pushed_when_no_resource_patch(self):
        coord = self._coord_with_wood_harvestable()
        self._activate_wood_goal(coord)
        coord.tick(_WQ(resources={}), _WW, 1)
        task = coord._stack[-1]
        self.assertIsNotNone(task)
        self.assertEqual(task.task_type, "harvest_natural")

    def test_harvest_task_carries_entity_types(self):
        coord = self._coord_with_wood_harvestable(["tree-01", "tree-02"])
        self._activate_wood_goal(coord)
        coord.tick(_WQ(resources={}), _WW, 1)
        task = coord._stack[-1]
        self.assertIn("tree-01", task.entity_types)
        self.assertIn("tree-02", task.entity_types)

    def test_harvest_task_carries_item_and_count(self):
        coord = self._coord_with_wood_harvestable()
        self._activate_wood_goal(coord)
        coord.tick(_WQ(resources={}), _WW, 1)
        task = coord._stack[-1]
        self.assertEqual(task.item, "wood")
        self.assertEqual(task.count, 5)

    def test_harvest_task_success_condition_uses_absolute_inventory(self):
        coord = self._coord_with_wood_harvestable()
        self._activate_wood_goal(coord)
        coord.tick(_WQ(resources={}, inventory={}), _WW, 1)
        self.assertIn("inventory('wood')", coord._stack[-1].success_condition)

    def test_stuck_when_no_patch_and_no_harvestable(self):
        kb = _make_kb(harvestable={})
        coord = _make_coord(agents={"mining": _make_agent()}, kb=kb)
        coord.reset(GOAL_COLLECTION, {}, _WQ(),
                    success_condition="new.inventory('wood') >= 5")
        coord._stack[0].params = {"item": "wood", "count": 5}
        coord.tick(_WQ(resources={}), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.FAILED)

    def test_resource_patch_takes_priority_over_harvest(self):
        tree = _make_harvestable_entity("tree-01")
        kb = _make_kb(harvestable={"iron-ore": [tree]})
        coord = _make_coord(agents={"mining": _make_agent()}, kb=kb)
        coord.reset(GOAL_COLLECTION, {}, _WQ(),
                    success_condition="new.inventory('iron-ore') >= 5")
        coord._stack[0].params = {"item": "iron-ore", "count": 5}
        patch = MagicMock()
        patch.position = Position(10.0, 10.0)
        coord.tick(_WQ(resources={"iron-ore": [patch]}), _WW, 1)
        self.assertEqual(coord._stack[-1].task_type, "gather_resource")


# ===========================================================================
# Section 6 — _handle_acquire
# ===========================================================================

class TestHandleAcquire(unittest.TestCase):

    def test_already_satisfied_completes(self):
        coord = _make_coord()
        coord.reset(GOAL_ACQUIRE, {"item": "iron-ore", "count": 5}, _WQ())
        coord.tick(_WQ(inventory={"iron-ore": 10}), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.COMPLETE)

    def test_natural_resource_pushes_collection_subgoal(self):
        coord = _make_coord()
        coord.reset(GOAL_ACQUIRE, {"item": "iron-ore", "count": 10}, _WQ())
        wq = _WQ(resources={"iron-ore": [_ResourcePatch(Position(x=50, y=50))]})
        coord.tick(wq, _WW, 1)
        subgoals = [f for f in coord._stack
                    if f.goal_type == GOAL_COLLECTION]
        self.assertEqual(len(subgoals), 1)
        self.assertEqual(subgoals[0].params["item"], "iron-ore")

    def test_collection_subgoal_has_correct_count(self):
        coord = _make_coord()
        coord.reset(GOAL_ACQUIRE, {"item": "iron-ore", "count": 17}, _WQ())
        wq = _WQ(resources={"iron-ore": [_ResourcePatch(Position(x=50, y=50))]})
        coord.tick(wq, _WW, 1)
        subgoal = coord._stack[-1]
        self.assertEqual(subgoal.params["count"], 17)

    def test_producible_unimplemented_fails(self):
        recipe = MagicMock()
        recipe.is_placeholder = False
        kb = _make_kb(recipes={"iron-gear-wheel": [recipe]})
        coord = _make_coord(kb=kb)
        coord.reset(GOAL_ACQUIRE, {"item": "iron-gear-wheel", "count": 5}, _WQ())
        coord.tick(_WQ(), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.FAILED)

    def test_no_source_fails(self):
        coord = _make_coord()
        coord.reset(GOAL_ACQUIRE, {"item": "unobtainium", "count": 1}, _WQ())
        coord.tick(_WQ(), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.FAILED)

    def test_after_collection_completes_checks_inventory(self):
        coord = _make_coord()
        coord.reset(GOAL_ACQUIRE, {"item": "iron-ore", "count": 5}, _WQ())
        wq = _WQ(resources={"iron-ore": [_ResourcePatch(Position(x=50, y=50))]})
        coord.tick(wq, _WW, 1)
        # Mark sub-goal complete
        coord._stack[-1].status = ItemStatus.COMPLETE
        coord.tick(_WQ(inventory={"iron-ore": 5}), _WW, 2)
        self.assertEqual(coord._stack[0].status, ItemStatus.COMPLETE)

    def test_insufficient_after_collection_retries(self):
        coord = _make_coord()
        coord.reset(GOAL_ACQUIRE, {"item": "iron-ore", "count": 10}, _WQ())
        wq = _WQ(resources={"iron-ore": [_ResourcePatch(Position(x=50, y=50))]})
        coord.tick(wq, _WW, 1)
        coord._stack[-1].status = ItemStatus.COMPLETE
        coord.tick(_WQ(inventory={"iron-ore": 3}), _WW, 2)  # not enough
        frame = coord._stack[0]
        self.assertEqual(frame.step, 0)


# ===========================================================================
# Section 7 — _handle_crafting
# ===========================================================================

class TestHandleCrafting(unittest.TestCase):

    def _recipe(self, ingredients=None, output_count=1):
        r = MagicMock()
        r.is_placeholder = False
        r.category = "crafting"
        r.output_count = output_count
        r.ingredients = ingredients or []
        return r

    def _ing(self, item, count):
        i = MagicMock(); i.item = item; i.count = count
        return i

    def test_already_satisfied_completes(self):
        coord = _make_coord()
        coord.reset(GOAL_CRAFTING,
                    {"item": "iron-gear-wheel", "count": 5}, _WQ())
        coord.tick(_WQ(inventory={"iron-gear-wheel": 5}), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.COMPLETE)

    def test_missing_ingredients_push_acquire_subgoals(self):
        ing = self._ing("iron-plate", 2)
        recipe = self._recipe(ingredients=[ing])
        kb = _make_kb(recipes={"iron-gear-wheel": [recipe]})
        coord = _make_coord(kb=kb)
        coord.reset(GOAL_CRAFTING,
                    {"item": "iron-gear-wheel", "recipe": "iron-gear-wheel",
                     "count": 5}, _WQ())
        coord.tick(_WQ(), _WW, 1)
        subgoals = [f for f in coord._stack
                    if f.goal_type == GOAL_ACQUIRE]
        self.assertGreaterEqual(len(subgoals), 1)
        self.assertEqual(subgoals[0].params["item"], "iron-plate")

    def test_ingredients_present_pushes_craft_task(self):
        craft_agent = _make_agent("RUNNING")
        ing = self._ing("iron-plate", 2)
        recipe = self._recipe(ingredients=[ing])
        kb = _make_kb(recipes={"iron-gear-wheel": [recipe]})
        coord = _make_coord(agents={"crafting": craft_agent}, kb=kb)
        coord.reset(GOAL_CRAFTING,
                    {"item": "iron-gear-wheel", "recipe": "iron-gear-wheel",
                     "count": 5}, _WQ())
        coord.tick(_WQ(inventory={"iron-plate": 100}), _WW, 1)
        self.assertIsNotNone(coord._stack[-1])
        self.assertEqual(coord._stack[-1].task_type, "craft_items")

    def test_craft_task_targets_set_correctly(self):
        craft_agent = _make_agent("RUNNING")
        ing = self._ing("iron-plate", 2)
        recipe = self._recipe(ingredients=[ing])
        kb = _make_kb(recipes={"iron-gear-wheel": [recipe]})
        coord = _make_coord(agents={"crafting": craft_agent}, kb=kb)
        coord.reset(GOAL_CRAFTING,
                    {"item": "iron-gear-wheel", "recipe": "iron-gear-wheel",
                     "count": 3}, _WQ())
        coord.tick(_WQ(inventory={"iron-plate": 100}), _WW, 1)
        targets = coord._stack[-1].targets
        self.assertEqual(targets[0]["item"], "iron-gear-wheel")
        self.assertEqual(targets[0]["count"], 3)

    def test_step_2_verifies_inventory_and_completes(self):
        coord = _make_coord()
        frame = _make_goal(GOAL_CRAFTING, 
                          params={"item": "iron-gear-wheel", "count": 5},
                          step=2)
        coord._stack.append(frame)
        coord.tick(_WQ(inventory={"iron-gear-wheel": 5}), _WW, 1)
        self.assertEqual(frame.status, ItemStatus.COMPLETE)

    def test_step_2_insufficient_resets_to_zero(self):
        coord = _make_coord()
        frame = _make_goal(GOAL_CRAFTING, 
                          params={"item": "iron-gear-wheel", "count": 10},
                          step=2)
        coord._stack.append(frame)
        coord.tick(_WQ(inventory={"iron-gear-wheel": 2}), _WW, 1)
        self.assertEqual(frame.step, 0)


# ===========================================================================
# Section 8 — _handle_explore
# ===========================================================================

class TestHandleExplore(unittest.TestCase):

    def test_target_already_met_completes(self):
        coord = _make_coord()
        coord.reset(GOAL_EXPLORE, {"target_chunks": 5}, _WQ())
        frame = coord._stack[0]
        frame.context["start_chunks"] = 0
        coord.tick(_WQ(charted=5), _WW, 1)
        self.assertEqual(frame.status, ItemStatus.COMPLETE)

    def test_pushes_explore_task(self):
        explore_agent = _make_agent("RUNNING")
        coord = _make_coord(agents={"exploration": explore_agent})
        coord.reset(GOAL_EXPLORE, {"target_chunks": 20}, _WQ())
        chunk = ChunkCoord(cx=5, cy=0)
        coord.tick(_WQ(charted=0, nearby_uncharted=[chunk]), _WW, 1)
        self.assertIsNotNone(coord._stack[-1])
        self.assertEqual(coord._stack[-1].task_type, "explore_region")

    def test_task_has_no_frontier_position(self):
        """Coordinator no longer computes or passes frontier_position.
        The ExplorationAgent picks its own frontier from wq.chunk_map /
        nearby_uncharted_chunks autonomously on each tick."""
        explore_agent = _make_agent("RUNNING")
        coord = _make_coord(agents={"exploration": explore_agent})
        coord.reset(GOAL_EXPLORE, {"target_chunks": 20}, _WQ())
        chunk = ChunkCoord(cx=2, cy=3)
        coord.tick(_WQ(charted=0, nearby_uncharted=[chunk]), _WW, 1)
        # frontier_position is not set on the task by the coordinator
        self.assertFalse(hasattr(coord._stack[-1], "frontier_position"),
            "coordinator should not set frontier_position — agent picks its own frontier")

    def test_single_task_pushed_and_coordinator_waits(self):
        """_handle_explore pushes one task at step 0 and transitions to WAITING."""
        explore_agent = _make_agent("RUNNING")
        coord = _make_coord(agents={"exploration": explore_agent})
        coord.reset(GOAL_EXPLORE, {"target_chunks": 20}, _WQ())
        coord.tick(_WQ(charted=0, nearby_uncharted=[]), _WW, 1)
        self.assertIsNotNone(coord._stack[-1])
        self.assertEqual(coord._stack[-1].task_type, "explore_region")
        self.assertEqual(coord._stack[0].step, 1)

    def test_custom_success_condition_passed_to_task(self):
        """When success_condition is in params, it overrides the chunk-count default."""
        explore_agent = _make_agent("RUNNING")
        coord = _make_coord(agents={"exploration": explore_agent})
        cond = "len(resources_of_type('copper-ore')) >= 1"
        coord.reset(GOAL_EXPLORE, {"success_condition": cond}, _WQ())
        coord.tick(_WQ(charted=0, nearby_uncharted=[]), _WW, 1)
        self.assertIsNotNone(coord._stack[-1])
        self.assertEqual(coord._stack[-1].success_condition, cond)

    def test_custom_description_passed_to_task(self):
        """Custom description param is forwarded to the task."""
        explore_agent = _make_agent("RUNNING")
        coord = _make_coord(agents={"exploration": explore_agent})
        coord.reset(
            GOAL_EXPLORE,
            {"success_condition": "len(resources_of_type('oil')) >= 1",
             "description": "Find an oil patch"},
            _WQ(),
        )
        coord.tick(_WQ(charted=0, nearby_uncharted=[]), _WW, 1)
        self.assertIsNotNone(coord._stack[-1])
        self.assertIn("oil", coord._stack[-1].description.lower())

    def test_default_condition_used_when_no_custom(self):
        """Without a custom condition, the task gets the charted_chunks condition."""
        explore_agent = _make_agent("RUNNING")
        coord = _make_coord(agents={"exploration": explore_agent})
        coord.reset(GOAL_EXPLORE, {"target_chunks": 10}, _WQ(charted=50))
        coord.tick(_WQ(charted=50, nearby_uncharted=[]), _WW, 1)
        self.assertIsNotNone(coord._stack[-1])
        self.assertIn("charted_chunks", coord._stack[-1].success_condition)
        self.assertIn("60", coord._stack[-1].success_condition)

    def test_start_chunks_context_initialised_on_first_tick(self):
        explore_agent = _make_agent("RUNNING")
        coord = _make_coord(agents={"exploration": explore_agent})
        coord.reset(GOAL_EXPLORE, {"target_chunks": 20}, _WQ())
        wq = _WQ(charted=7)
        coord.tick(wq, _WW, 1)
        self.assertEqual(coord._stack[0].context["start_chunks"], 7)


# ===========================================================================
# Section 9 — _handle_clear_region
# ===========================================================================

class TestHandleClearRegion(unittest.TestCase):

    def _bbox(self): return _BBox(0, 0, 100, 100)

    def test_undestroyable_blocker_fails_goal(self):
        coord = _make_coord(kb=_make_kb())  # placeholder → minable=False
        coord.reset(GOAL_CLEAR_REGION, {"bbox": self._bbox()}, _WQ())
        cliff = _nat(1, "cliff", x=50, y=50, proto="cliff")
        coord.tick(_WQ(natural_objects=[cliff]), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.FAILED)

    def test_no_blockers_pushes_clear_task(self):
        mine_agent = _make_agent("RUNNING")
        kb = _minable_kb(["tree-01"])
        coord = _make_coord(agents={"mining": mine_agent}, kb=kb)
        coord.reset(GOAL_CLEAR_REGION,
                    {"bbox": self._bbox(), "clear_mode": "clear_natural"},
                    _WQ())
        tree = _nat(1, "tree-01", x=50, y=50)
        coord.tick(_WQ(natural_objects=[tree]), _WW, 1)
        self.assertIsNotNone(coord._stack[-1])
        self.assertEqual(coord._stack[-1].task_type, "clear_region")

    def test_clear_task_bbox_attribute(self):
        mine_agent = _make_agent("RUNNING")
        kb = _minable_kb(["tree-01"])
        coord = _make_coord(agents={"mining": mine_agent}, kb=kb)
        bbox = _BBox(10, 20, 50, 60)
        coord.reset(GOAL_CLEAR_REGION, {"bbox": bbox}, _WQ())
        tree = _nat(1, "tree-01", x=30, y=40)
        coord.tick(_WQ(natural_objects=[tree]), _WW, 1)
        self.assertIs(coord._stack[-1].bbox, bbox)

    def test_clear_task_mode_attribute(self):
        mine_agent = _make_agent("RUNNING")
        kb = _minable_kb(["tree-01"])
        coord = _make_coord(agents={"mining": mine_agent}, kb=kb)
        coord.reset(GOAL_CLEAR_REGION,
                    {"bbox": self._bbox(), "clear_mode": "clear_all"}, _WQ())
        tree = _nat(1, "tree-01", x=50, y=50)
        coord.tick(_WQ(natural_objects=[tree]), _WW, 1)
        self.assertEqual(coord._stack[-1].clear_mode, "clear_all")

    def test_default_mode_is_clear_natural(self):
        mine_agent = _make_agent("RUNNING")
        kb = _minable_kb(["tree-01"])
        coord = _make_coord(agents={"mining": mine_agent}, kb=kb)
        coord.reset(GOAL_CLEAR_REGION, {"bbox": self._bbox()}, _WQ())
        tree = _nat(1, "tree-01", x=50, y=50)
        coord.tick(_WQ(natural_objects=[tree]), _WW, 1)
        self.assertEqual(coord._stack[-1].clear_mode, "clear_natural")

    def test_step_2_empty_bbox_completes(self):
        coord = _make_coord()
        frame = _make_goal(GOAL_CLEAR_REGION, 
                          params={"bbox": self._bbox()}, step=2)
        coord._stack.append(frame)
        coord.tick(_WQ(natural_objects=[]), _WW, 1)
        self.assertEqual(frame.status, ItemStatus.COMPLETE)

    def test_step_2_objects_remain_reissues_clear(self):
        mine_agent = _make_agent("RUNNING")
        kb = _minable_kb(["tree-01"])
        coord = _make_coord(agents={"mining": mine_agent}, kb=kb)
        frame = _make_goal(GOAL_CLEAR_REGION, 
                          params={"bbox": self._bbox()}, step=2)
        coord._stack.append(frame)
        tree = _nat(1, "tree-01", x=50, y=50)
        coord.tick(_WQ(natural_objects=[tree]), _WW, 1)
        self.assertEqual(frame.step, 1)


# ===========================================================================
# Section 10 — _handle_prep_region
# ===========================================================================

class TestHandlePrepRegion(unittest.TestCase):

    def _bbox(self): return _BBox(0, 0, 100, 100)

    def test_major_factory_fails_goal(self):
        coord = _make_coord()
        coord.reset(GOAL_PREP_REGION, {"bbox": self._bbox()}, _WQ())
        asm = _EntityState(1, prototype_type="assembling-machine",
                           position=Position(x=50, y=50))
        coord.tick(_WQ(entities=[asm]), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.FAILED)

    def test_furnace_in_bbox_also_fails(self):
        coord = _make_coord()
        coord.reset(GOAL_PREP_REGION, {"bbox": self._bbox()}, _WQ())
        furnace = _EntityState(1, prototype_type="furnace",
                               position=Position(x=50, y=50))
        coord.tick(_WQ(entities=[furnace]), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.FAILED)

    def test_undestroyable_obstacle_fails_at_step_1(self):
        coord = _make_coord(kb=_make_kb())
        frame = _make_goal(GOAL_PREP_REGION, 
                          params={"bbox": self._bbox()}, step=1)
        coord._stack.append(frame)
        cliff = _nat(1, "cliff", x=50, y=50, proto="cliff")
        coord.tick(_WQ(natural_objects=[cliff]), _WW, 1)
        self.assertEqual(frame.status, ItemStatus.FAILED)

    def test_step_2_proceeds_to_clear_subgoal(self):
        coord = _make_coord()
        frame = _make_goal(GOAL_PREP_REGION, 
                          params={"bbox": self._bbox()}, step=2)
        coord._stack.append(frame)
        coord.tick(_WQ(), _WW, 1)
        subgoals = [f for f in coord._stack
                    if f.goal_type == GOAL_CLEAR_REGION]
        self.assertEqual(len(subgoals), 1)

    def test_step_3_pushes_clear_region_subgoal(self):
        coord = _make_coord()
        frame = _make_goal(GOAL_PREP_REGION, 
                          params={"bbox": self._bbox()}, step=3)
        coord._stack.append(frame)
        coord.tick(_WQ(), _WW, 1)
        subgoals = [f for f in coord._stack
                    if f.goal_type == GOAL_CLEAR_REGION]
        self.assertEqual(len(subgoals), 1)
        self.assertEqual(subgoals[0].params["clear_mode"], "clear_all")

    def test_clear_region_subgoal_has_correct_bbox(self):
        bbox = _BBox(10, 20, 50, 60)
        coord = _make_coord()
        frame = _make_goal(GOAL_PREP_REGION, 
                          params={"bbox": bbox}, step=3)
        coord._stack.append(frame)
        coord.tick(_WQ(), _WW, 1)
        subgoal = coord._stack[-1]
        self.assertIs(subgoal.params["bbox"], bbox)

    def test_step_4_completes_after_subgoal(self):
        coord = _make_coord()
        frame = _make_goal(GOAL_PREP_REGION, 
                          params={"bbox": self._bbox()}, step=4)
        coord._stack.append(frame)
        coord.tick(_WQ(), _WW, 1)
        self.assertEqual(frame.status, ItemStatus.COMPLETE)


# ===========================================================================
# Section 11 — _handle_construction
# ===========================================================================

class TestHandleConstruction(unittest.TestCase):

    def _bbox(self): return _BBox(0, 0, 50, 50)

    def test_step_0_pushes_prep_region_subgoal(self):
        coord = _make_coord()
        coord.reset(GOAL_CONSTRUCTION, {"bbox": self._bbox()}, _WQ())
        coord.tick(_WQ(), _WW, 1)
        subgoals = [f for f in coord._stack
                    if f.goal_type == GOAL_PREP_REGION]
        self.assertEqual(len(subgoals), 1)

    def test_prep_region_subgoal_has_correct_bbox(self):
        bbox = self._bbox()
        coord = _make_coord()
        coord.reset(GOAL_CONSTRUCTION, {"bbox": bbox}, _WQ())
        coord.tick(_WQ(), _WW, 1)
        subgoal = coord._stack[-1]
        self.assertIs(subgoal.params["bbox"], bbox)

    def test_step_2_build_stub_completes_goal(self):
        coord = _make_coord()
        frame = _make_goal(GOAL_CONSTRUCTION, 
                          params={"bbox": self._bbox()}, step=2)
        coord._stack.append(frame)
        coord.tick(_WQ(), _WW, 1)
        self.assertEqual(frame.status, ItemStatus.COMPLETE)


# ===========================================================================
# Section 12 — Stub handlers
# ===========================================================================

class TestStubHandlers(unittest.TestCase):

    def test_production_fails_goal(self):
        coord = _make_coord()
        coord.reset(GOAL_PRODUCTION,
                    {"item": "iron-plate", "rate_per_min": 60.0}, _WQ())
        coord.tick(_WQ(), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.FAILED)

    def test_logistics_fails_goal(self):
        coord = _make_coord()
        coord.reset(GOAL_LOGISTICS, {"connections": []}, _WQ())
        coord.tick(_WQ(), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.FAILED)

    def test_byproduct_fails_goal(self):
        coord = _make_coord()
        coord.reset(GOAL_BYPRODUCT, {"item": "iron-plate"}, _WQ())
        coord.tick(_WQ(), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.FAILED)

    def test_noop_returns_waiting_without_completing(self):
        coord = _make_coord()
        coord.reset(GOAL_NOOP, {}, _WQ())
        status, _ = coord.tick(_WQ(), _WW, 1)
        self.assertEqual(status, CoordinatorStatus.WAITING)
        self.assertNotEqual(coord._stack[0].status, ItemStatus.COMPLETE)
        self.assertNotEqual(coord._stack[0].status, ItemStatus.FAILED)

    def test_noop_is_idempotent_across_ticks(self):
        coord = _make_coord()
        coord.reset(GOAL_NOOP, {}, _WQ())
        for i in range(1, 5):
            status, _ = coord.tick(_WQ(), _WW, i)
            self.assertEqual(status, CoordinatorStatus.WAITING)

    def test_research_already_unlocked_completes(self):
        coord = _make_coord()
        coord.reset(GOAL_RESEARCH, {"tech": "automation"}, _WQ())
        coord.tick(_WQ(tech_unlocked={"automation"}), _WW, 1)
        self.assertEqual(coord._stack[0].status, ItemStatus.COMPLETE)

    def test_research_step_4_pushes_queue_task(self):
        research_agent = _make_agent("RUNNING")
        coord = _make_coord(agents={"crafting": research_agent})
        frame = _make_goal(GOAL_RESEARCH, params={"tech": "automation"}, step=4)
        coord._stack.append(frame)
        coord.tick(_WQ(), _WW, 1)
        self.assertIsNotNone(coord._stack[-1])
        self.assertEqual(coord._stack[-1].task_type, "set_research_queue")

    def test_research_queue_task_has_tech_attribute(self):
        research_agent = _make_agent("RUNNING")
        coord = _make_coord(agents={"crafting": research_agent})
        frame = _make_goal(GOAL_RESEARCH, params={"tech": "steel-processing"},
                          step=4)
        coord._stack.append(frame)
        coord.tick(_WQ(), _WW, 1)
        self.assertEqual(coord._stack[-1].tech, "steel-processing")

    def test_research_step_5_verifies_unlock(self):
        coord = _make_coord()
        frame = _make_goal(GOAL_RESEARCH, 
                          params={"tech": "automation"}, step=5)
        coord._stack.append(frame)
        coord.tick(_WQ(tech_unlocked={"automation"}), _WW, 1)
        self.assertEqual(frame.status, ItemStatus.COMPLETE)

    def test_research_step_5_re_queues_if_not_unlocked(self):
        coord = _make_coord()
        frame = _make_goal(GOAL_RESEARCH, 
                          params={"tech": "automation"}, step=5)
        coord._stack.append(frame)
        coord.tick(_WQ(), _WW, 1)
        self.assertEqual(frame.step, 4)


# ===========================================================================
# Section 13 — _push_task
# ===========================================================================

class TestPushTask(unittest.TestCase):

    def _coord_with_nav(self):
        agent = _make_agent("RUNNING")
        coord = _make_coord(agents={"navigation": agent})
        coord._stack.append(_make_goal(GOAL_COLLECTION))
        return coord, agent

    def test_task_type_set(self):
        coord, _ = self._coord_with_nav()
        coord._push_task("navigate_to_position", "Go", "navigation", tick=1)
        self.assertEqual(coord._stack[-1].task_type, "navigate_to_position")

    def test_description_set(self):
        coord, _ = self._coord_with_nav()
        coord._push_task("navigate_to_position", "Approach patch", "navigation", tick=1)
        self.assertEqual(coord._stack[-1].description, "Approach patch")

    def test_dynamic_attributes_forwarded(self):
        coord, _ = self._coord_with_nav()
        coord._push_task(
            "navigate_to_position", "Go", "navigation", tick=1,
            target_position=Position(x=99, y=88),
            extra_attr="hello",
        )
        self.assertAlmostEqual(coord._stack[-1].target_position.x, 99.0)
        self.assertEqual(coord._stack[-1].extra_attr, "hello")

    def test_agent_activate_called(self):
        coord, agent = self._coord_with_nav()
        coord._push_task("navigate_to_position", "Go", "navigation", tick=1)
        agent.activate.assert_called_once()

    def test_blackboard_intention_written(self):
        coord, _ = self._coord_with_nav()
        coord._push_task("navigate_to_position", "Go", "navigation", tick=1)
        entries = coord._bb.read(category=EntryCategory.INTENTION)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].data["type"], "task_activated")
        self.assertEqual(entries[0].data["agent"], "navigation")

    def test_missing_agent_does_not_set_active_task(self):
        coord = _make_coord(agents={})  # no agents
        coord._stack.append(_make_goal(GOAL_COLLECTION))
        depth_before = len(coord._stack)
        coord._push_task("navigate_to_position", "Go", "navigation", tick=1)
        # No agent registered → _push_task returns early, no Task appended
        self.assertEqual(len(coord._stack), depth_before)

    def test_failure_condition_contains_elapsed_ticks(self):
        coord, _ = self._coord_with_nav()
        coord._push_task("navigate_to_position", "Go", "navigation", tick=1)
        self.assertIn("elapsed_ticks", coord._stack[-1].failure_condition)

    def test_active_agent_set(self):
        coord, agent = self._coord_with_nav()
        coord._push_task("navigate_to_position", "Go", "navigation", tick=1)
        self.assertIs(coord._active_agent, agent)


# ===========================================================================
# Section 14 — _push_goal
# ===========================================================================

class TestPushGoal(unittest.TestCase):

    def test_frame_appended_to_stack(self):
        coord = _make_coord()
        coord._stack.append(_make_goal(GOAL_CONSTRUCTION))
        coord._push_goal(GOAL_PREP_REGION, {"bbox": _BBox(0,0,50,50)})
        self.assertEqual(len(coord._stack), 2)
        self.assertEqual(coord._stack[-1].goal_type, GOAL_PREP_REGION)

    def test_frame_params_preserved(self):
        coord = _make_coord()
        coord._stack.append(_make_goal(GOAL_CONSTRUCTION))
        bbox = _BBox(10, 20, 50, 60)
        coord._push_goal(GOAL_PREP_REGION, {"bbox": bbox})
        self.assertIs(coord._stack[-1].params["bbox"], bbox)

    def test_frame_starts_at_step_zero(self):
        coord = _make_coord()
        coord._stack.append(_make_goal(GOAL_CONSTRUCTION))
        coord._push_goal(GOAL_PREP_REGION, {})
        self.assertEqual(coord._stack[-1].step, 0)

    def test_multiple_subgoals_stack_correctly(self):
        coord = _make_coord()
        coord._stack.append(_make_goal(GOAL_RESEARCH))
        coord._push_goal(GOAL_ACQUIRE, {"item": "iron-plate", "count": 5})
        coord._push_goal(GOAL_COLLECTION, {"item": "iron-plate", "count": 5})
        self.assertEqual(coord._stack[-1].goal_type, GOAL_COLLECTION)
        self.assertEqual(coord._stack[-2].goal_type, GOAL_ACQUIRE)


# ===========================================================================
# Section 15 — _tick_task
# ===========================================================================

class TestTickTask(unittest.TestCase):

    def setUp(self):
        """Each test needs at least a goal on the stack before a task."""
        self.coord = _make_coord(agents={"navigation": _make_agent("RUNNING")})
        self.coord.reset(GOAL_NOOP, {}, _WQ())

    def _task(self, success="", failure=""):
        from planning.tasks.task import Task
        return Task(
            description="test",
            success_condition=success,
            failure_condition=failure,
            agent_hint="navigation",
            task_type="test",
            status=ItemStatus.ACTIVE,
            created_at=0,
        )

    def _push_task(self, coord, success="", failure=""):
        """Push a test Task onto the coordinator stack and set active agent."""
        t = self._task(success, failure)
        coord._stack.append(t)
        coord._active_agent = _make_agent("RUNNING")
        return t

    def test_success_condition_returns_succeeded(self):
        agent = _make_agent("RUNNING")
        self._push_task(self.coord, success="True")
        self.coord._active_agent = agent
        outcome, _ = self.coord._tick_task(_WQ(), _WW, 1)
        self.assertEqual(outcome, TaskOutcome.SUCCEEDED)

    def test_failure_condition_returns_failed(self):
        agent = _make_agent("RUNNING")
        self._push_task(self.coord, failure="True")
        self.coord._active_agent = agent
        outcome, _ = self.coord._tick_task(_WQ(), _WW, 1)
        self.assertEqual(outcome, TaskOutcome.FAILED)

    def test_stuck_agent_observe_returns_stuck(self):
        agent = _make_agent("STUCK")
        self._push_task(self.coord)
        self.coord._active_agent = agent
        outcome, _ = self.coord._tick_task(_WQ(), _WW, 1)
        self.assertEqual(outcome, TaskOutcome.STUCK)

    def test_running_agent_returns_running_with_actions(self):
        action = MagicMock()
        agent = _make_agent("RUNNING")
        agent.tick.return_value = [action]
        self._push_task(self.coord)
        self.coord._active_agent = agent
        outcome, actions = self.coord._tick_task(_WQ(), _WW, 1)
        self.assertEqual(outcome, TaskOutcome.RUNNING)
        self.assertIn(action, actions)

    def test_none_task_returns_succeeded(self):
        # When no Task is on top of the stack, _tick_task returns SUCCEEDED.
        # Use an empty stack to simulate this — no active task.
        coord = _make_coord()
        self.assertEqual(len(coord._stack), 0)
        coord._active_agent = None
        outcome, _ = coord._tick_task(_WQ(), _WW, 1)
        self.assertEqual(outcome, TaskOutcome.SUCCEEDED)

    def test_success_takes_priority_over_failure(self):
        agent = _make_agent("RUNNING")
        self._push_task(self.coord, success="True", failure="True")
        self.coord._active_agent = agent
        outcome, _ = self.coord._tick_task(_WQ(), _WW, 1)
        self.assertEqual(outcome, TaskOutcome.SUCCEEDED)

    def test_agent_ticked_when_running(self):
        agent = _make_agent("RUNNING")
        self._push_task(self.coord)
        self.coord._active_agent = agent
        self.coord._tick_task(_WQ(), _WW, 5)
        agent.tick.assert_called_once()

    def test_agent_not_ticked_on_success(self):
        agent = _make_agent("RUNNING")
        self._push_task(self.coord, success="True")
        self.coord._active_agent = agent
        self.coord._tick_task(_WQ(), _WW, 1)
        agent.tick.assert_not_called()

    def test_navigate_status_also_detected_as_stuck(self):
        agent = MagicMock()
        agent.activate.return_value = None
        agent.tick.return_value = []
        agent.observe.return_value = {"navigate_status": "STUCK"}
        agent.pending_patches.return_value = []
        self._push_task(self.coord)
        self.coord._active_agent = agent
        outcome, _ = self.coord._tick_task(_WQ(), _WW, 1)
        self.assertEqual(outcome, TaskOutcome.STUCK)


# ===========================================================================
# Section 16 — drain_patches
# ===========================================================================

class TestDrainPatches(unittest.TestCase):

    def test_empty_initially(self):
        self.assertEqual(_make_coord().drain_patches(), [])

    def test_returns_patches_and_clears(self):
        coord = _make_coord()
        p1, p2 = MagicMock(), MagicMock()
        coord._pending_patches.extend([p1, p2])
        result = coord.drain_patches()
        self.assertEqual(len(result), 2)
        self.assertIn(p1, result)
        self.assertEqual(coord._pending_patches, [])

    def test_idempotent_second_drain(self):
        coord = _make_coord()
        coord._pending_patches.append(MagicMock())
        coord.drain_patches()
        self.assertEqual(coord.drain_patches(), [])

    def test_agent_patches_drained_during_tick(self):
        patch = MagicMock()
        agent = _make_agent("RUNNING")
        agent.pending_patches.return_value = [patch]
        coord = _make_coord(agents={"navigation": agent})
        coord.reset(GOAL_NOOP, {}, _WQ())
        t = _inject_task(coord)
        t.success_condition = ""
        t.failure_condition = ""
        coord._active_agent = agent
        coord.tick(_WQ(), _WW, 1)
        self.assertIn(patch, coord.drain_patches())


# ===========================================================================
# Section 17 — Module-level helpers
# ===========================================================================

class TestDist(unittest.TestCase):

    def test_zero(self):
        p = Position(x=3.0, y=4.0)
        self.assertAlmostEqual(_dist(p, p), 0.0)

    def test_pythagoras(self):
        self.assertAlmostEqual(_dist(Position(0,0), Position(3,4)), 5.0)

    def test_negative_coordinates(self):
        self.assertAlmostEqual(_dist(Position(-3,0), Position(0,4)), 5.0)


class TestItemInNearbyChest(unittest.TestCase):

    def test_always_false_stub(self):
        self.assertFalse(_item_in_nearby_chest("iron-ore", _WQ()))
        self.assertFalse(_item_in_nearby_chest("anything", _WQ()))


class TestUndestroyableInBbox(unittest.TestCase):

    def _cliff_kb(self):
        kb = MagicMock()
        rec = MagicMock(); rec.is_placeholder = False; rec.minable = False
        kb.get_entity.return_value = rec
        return kb

    def _tree_kb(self):
        kb = MagicMock()
        rec = MagicMock(); rec.is_placeholder = False; rec.minable = True
        kb.get_entity.return_value = rec
        return kb

    def test_non_minable_in_bbox_returned(self):
        bbox = _BBox(0, 0, 100, 100)
        cliff = _nat(1, "cliff", x=50, y=50, proto="cliff")
        result = _undestroyable_in_bbox(bbox, _WQ(natural_objects=[cliff]),
                                        self._cliff_kb())
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], cliff)

    def test_minable_not_returned(self):
        bbox = _BBox(0, 0, 100, 100)
        tree = _nat(1, "tree-01", x=50, y=50)
        result = _undestroyable_in_bbox(bbox, _WQ(natural_objects=[tree]),
                                        self._tree_kb())
        self.assertEqual(result, [])

    def test_outside_bbox_ignored(self):
        bbox = _BBox(0, 0, 50, 50)
        cliff = _nat(1, "cliff", x=200, y=200, proto="cliff")
        result = _undestroyable_in_bbox(bbox, _WQ(natural_objects=[cliff]),
                                        self._cliff_kb())
        self.assertEqual(result, [])

    def test_empty_natural_objects(self):
        bbox = _BBox(0, 0, 100, 100)
        result = _undestroyable_in_bbox(bbox, _WQ(), self._cliff_kb())
        self.assertEqual(result, [])


class TestBboxIsClear(unittest.TestCase):

    def test_no_objects_is_clear(self):
        self.assertTrue(_bbox_is_clear(_BBox(0,0,100,100), _WQ()))

    def test_object_inside_not_clear(self):
        tree = _nat(1, x=50, y=50)
        self.assertFalse(
            _bbox_is_clear(_BBox(0,0,100,100), _WQ(natural_objects=[tree]))
        )

    def test_object_on_boundary_not_clear(self):
        tree = _nat(1, x=100, y=100)
        self.assertFalse(
            _bbox_is_clear(_BBox(0,0,100,100), _WQ(natural_objects=[tree]))
        )

    def test_object_outside_is_clear(self):
        tree = _nat(1, x=200, y=200)
        self.assertTrue(
            _bbox_is_clear(_BBox(0,0,100,100), _WQ(natural_objects=[tree]))
        )


class TestBboxEmptyCondition(unittest.TestCase):

    def test_returns_string(self):
        self.assertIsInstance(_bbox_empty_condition(_BBox(0,0,100,100)), str)

    def test_contains_coordinates(self):
        cond = _bbox_empty_condition(_BBox(10, 20, 50, 60))
        self.assertIn("10", cond)
        self.assertIn("20", cond)
        self.assertIn("50", cond)
        self.assertIn("60", cond)

    def test_contains_staleness_guard(self):
        self.assertIn("staleness", _bbox_empty_condition(_BBox(0,0,100,100)))

    def test_staleness_uses_entities_section(self):
        # Natural objects now live in the unified 'entities' section.
        cond = _bbox_empty_condition(_BBox(0,0,100,100))
        self.assertIn("staleness('entities')", cond)

    def test_references_natural_objects_in_bbox(self):
        # Completion is driven by natural_objects_in_bbox(), which filters
        # the unified entity list by is_natural=True.
        cond = _bbox_empty_condition(_BBox(0,0,100,100))
        self.assertIn("natural_objects_in_bbox", cond)


class TestIntersectsMajorFactory(unittest.TestCase):

    def test_assembler_in_bbox(self):
        asm = _EntityState(1, prototype_type="assembling-machine",
                           position=Position(x=50, y=50))
        self.assertTrue(
            _intersects_major_factory(_BBox(0,0,100,100), _WQ(entities=[asm]))
        )

    def test_all_major_types_detected(self):
        for pt in ["assembling-machine", "furnace", "electric-pole",
                   "lab", "mining-drill", "beacon", "roboport"]:
            e = _EntityState(1, prototype_type=pt, position=Position(x=50,y=50))
            self.assertTrue(
                _intersects_major_factory(_BBox(0,0,100,100),
                                          _WQ(entities=[e])),
                f"{pt} should be major",
            )

    def test_container_not_major(self):
        chest = _EntityState(1, prototype_type="container",
                             position=Position(x=50, y=50))
        self.assertFalse(
            _intersects_major_factory(_BBox(0,0,100,100),
                                      _WQ(entities=[chest]))
        )

    def test_entity_outside_bbox_ignored(self):
        asm = _EntityState(1, prototype_type="assembling-machine",
                           position=Position(x=500, y=500))
        self.assertFalse(
            _intersects_major_factory(_BBox(0,0,100,100), _WQ(entities=[asm]))
        )

    def test_empty_entities(self):
        self.assertFalse(
            _intersects_major_factory(_BBox(0,0,100,100), _WQ())
        )


class TestIntersectsLogistics(unittest.TestCase):

    def test_always_false_stub(self):
        self.assertFalse(_intersects_logistics(_BBox(0,0,100,100), _WQ()))


if __name__ == "__main__":
    unittest.main(verbosity=2)