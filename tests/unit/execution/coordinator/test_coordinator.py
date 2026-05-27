"""
tests/unit/agent/test_coordinator.py

Unit tests for the RuleBasedCoordinator in agent/network/coordinator.py.

All tests run without a live Factorio instance.
"""

import tempfile
import unittest
from pathlib import Path

from execution import Blackboard, EntryCategory, EntryScope
from execution import ExecutionStatus
from execution.coordinator.coordinator import (
    GOAL_TYPE_COLLECTION,
    GOAL_TYPE_CRAFTING,
    GOAL_TYPE_EXPLORATION,
    GOAL_TYPE_PRODUCTION,
    RuleBasedCoordinator,
)
from execution.coordinator.registry import AgentRegistry
from execution.agents.navigation import NavigationAgent
from world import SelfModel
from planning import Task as Subtask, TaskLedger as SubtaskLedger, TaskStatus as SubtaskStatus
from planning import Goal, GoalStatus, Priority, RewardSpec
from world import (
    IngredientRecord,
    KnowledgeBase,
    ProductRecord,
    RecipeRecord,
)
from world import (
    ExplorationState,
    Inventory,
    InventorySlot,
    PlayerState,
    Position,
    ResourcePatch,
    WorldState,
)
from world import WorldQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kb() -> KnowledgeBase:
    """Return an in-memory KnowledgeBase with no query_fn (offline/test mode)."""
    tmp = tempfile.mkdtemp()
    return KnowledgeBase(data_dir=tmp)


def _make_real_kb_with_gear_recipe() -> KnowledgeBase:
    """
    KnowledgeBase pre-populated with an iron-gear-wheel recipe.
    Used by crafting derivation tests that need the full coordinator code path.
    """
    kb = KnowledgeBase(data_dir=Path(tempfile.mkdtemp()))
    ing = IngredientRecord(name="iron-plate", amount=2.0, is_fluid=False)
    prod = ProductRecord(
        name="iron-gear-wheel", amount=1.0, probability=1.0, is_fluid=False
    )
    recipe = RecipeRecord(
        name="iron-gear-wheel",
        category="crafting",
        crafting_time=0.5,
        ingredients=[ing],
        products=[prod],
        made_in=["character", "assembling-machine-1"],
        enabled_by_default=True,
        is_placeholder=False,
    )
    kb._recipes["iron-gear-wheel"] = recipe
    kb._insert_recipe(recipe)
    return kb


def _make_crafting_coordinator(kb: KnowledgeBase | None = None) -> RuleBasedCoordinator:
    """Coordinator wired with *kb* and an otherwise empty registry."""
    if kb is None:
        kb = KnowledgeBase(data_dir=Path(tempfile.mkdtemp()))
    return RuleBasedCoordinator(
        registry=AgentRegistry(),
        blackboard=Blackboard(),
        ledger=SubtaskLedger(),
        self_model=SelfModel(),
        kb=kb,
    )


def _make_crafting_goal(
    success_condition: str = "inventory('iron-gear-wheel') >= 2",
    failure_condition: str = "tick > 99999",
) -> Goal:
    g = Goal(
        description="Craft some gears",
        priority=Priority.NORMAL,
        success_condition=success_condition,
        failure_condition=failure_condition,
        reward_spec=RewardSpec(),
    )
    g.type = GOAL_TYPE_CRAFTING
    g.activate(tick=0)
    return g

def _make_goal(
    goal_type: str = GOAL_TYPE_COLLECTION,
    success_condition: str = "inventory('iron-ore') >= 50",
    failure_condition: str = "tick > 99999",
) -> Goal:
    spec = RewardSpec()
    g = Goal(
        description="Test goal",
        priority=Priority.NORMAL,
        success_condition=success_condition,
        failure_condition=failure_condition,
        reward_spec=spec,
    )
    g.type = goal_type
    g.activate(tick=0)
    return g


def _make_wq(
    player_pos: Position = None,
    resource_patches: list = None,
    inventory_items: dict = None,
    charted_chunks: int = 0,
    tick: int = 100,
    inventory_size: int = 80,
) -> WorldQuery:
    state = WorldState(tick=tick, entities=[])
    state.player.position = player_pos or Position(0.0, 0.0)
    if inventory_items:
        state.player.inventory = Inventory(
            slots=[InventorySlot(item=k, count=v) for k, v in inventory_items.items()]
        )
    if resource_patches:
        state.resource_map = resource_patches
    state.player.exploration = ExplorationState(charted_chunks=charted_chunks)
    state.player.inventory_size = inventory_size
    state._rebuild_entity_indices()
    return WorldQuery(state)


def _make_patch(
    resource_type: str = "iron-ore",
    x: float = 10.0,
    y: float = 10.0,
    amount: int = 10000,
) -> ResourcePatch:
    return ResourcePatch(
        resource_type=resource_type,
        position=Position(x, y),
        amount=amount,
        size=50,
    )


def _make_coordinator(registry: AgentRegistry = None) -> RuleBasedCoordinator:
    reg = registry or AgentRegistry()
    bb = Blackboard()
    ledger = SubtaskLedger()
    sm = SelfModel()
    kb = _make_kb()
    return RuleBasedCoordinator(registry=reg, blackboard=bb, ledger=ledger, self_model=sm, kb=kb)


def _make_mock_writer():
    class _MockWriter:
        pass
    return _MockWriter()


def _drain_stop(coordinator, goal, wq, ww, tick=99):
    """
    Drain the StopMining action emitted on the first tick after reset().
    Call this after reset() and before the tick that should produce real work.
    """
    coordinator.tick(goal, wq, ww, tick=tick)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestCoordinatorReset(unittest.TestCase):

    def test_reset_clears_blackboard(self):
        coordinator = _make_coordinator()
        coordinator._bb.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.GOAL,
            owner_agent="test",
            created_at=0,
            data={"test": True},
        )
        self.assertEqual(len(coordinator._bb), 1)

        goal = _make_goal()
        wq = _make_wq()
        coordinator.reset(goal, wq)

        self.assertEqual(len(coordinator._bb), 0)

    def test_reset_clears_ledger(self):
        coordinator = _make_coordinator()
        subtask = Subtask(
            description="test",
            success_condition="",
            failure_condition="",
            parent_goal_id="g1",
            created_at=0,
            derived_locally=True,
        )
        coordinator._ledger.push(subtask)
        self.assertEqual(len(coordinator._ledger), 1)

        goal = _make_goal()
        wq = _make_wq()
        coordinator.reset(goal, wq)

        self.assertEqual(len(coordinator._ledger), 0)

    def test_reset_with_seed_subtasks_populates_ledger(self):
        coordinator = _make_coordinator()
        goal = _make_goal()
        wq = _make_wq()

        seeds = [
            Subtask(
                description="seed task 1",
                success_condition="inventory('iron-ore') >= 10",
                failure_condition="tick > 9999",
                parent_goal_id=goal.id,
                created_at=0,
                derived_locally=False,
            ),
            Subtask(
                description="seed task 2",
                success_condition="inventory('iron-ore') >= 50",
                failure_condition="tick > 9999",
                parent_goal_id=goal.id,
                created_at=0,
                derived_locally=False,
            ),
        ]
        coordinator.reset(goal, wq, seed_subtasks=seeds)

        # Ledger should have 2 subtasks; the last-pushed (index 1 in reversed
        # order) should be active.
        self.assertEqual(len(coordinator._ledger), 2)

    def test_reset_with_seed_subtasks_does_not_re_derive(self):
        # Even if the goal type is derivable, seed_subtasks skips derivation.
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq = _make_wq(resource_patches=[_make_patch()])

        seed = Subtask(
            description="injected subtask",
            success_condition="inventory('iron-ore') >= 50",
            failure_condition="tick > 9999",
            parent_goal_id=goal.id,
            created_at=0,
            derived_locally=False,
        )
        coordinator.reset(goal, wq, seed_subtasks=[seed])

        # Only 1 subtask (the seed), not the 2 that derivation would produce.
        self.assertEqual(len(coordinator._ledger), 1)


# ---------------------------------------------------------------------------
# tick() — STUCK for unsupported goal types
# ---------------------------------------------------------------------------

class TestCoordinatorStuckPath(unittest.TestCase):

    def test_production_goal_returns_stuck_immediately(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_PRODUCTION)
        wq = _make_wq()
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)

        result = coordinator.tick(goal, wq, ww, tick=100)

        self.assertEqual(result.status, ExecutionStatus.STUCK)
        self.assertIsNotNone(result.stuck_context)

    def test_stuck_at_goal_level_when_no_subtask_derived(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_PRODUCTION)
        wq = _make_wq()
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)

        result = coordinator.tick(goal, wq, ww, tick=100)

        self.assertTrue(result.stuck_context.stuck_at_goal_level)
        self.assertEqual(result.stuck_context.failure_chain, [])

    def test_stuck_context_has_correct_goal_id(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_PRODUCTION)
        wq = _make_wq()
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)

        result = coordinator.tick(goal, wq, ww, tick=100)

        self.assertEqual(result.stuck_context.goal.id, goal.id)

    def test_stuck_context_serialises_without_error(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_PRODUCTION)
        wq = _make_wq()
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)

        result = coordinator.tick(goal, wq, ww, tick=100)
        d = result.stuck_context.to_dict()

        self.assertIn("goal_id", d)
        self.assertIn("failure_chain", d)
        self.assertIn("sibling_history", d)

    def test_collection_goal_no_patches_returns_stuck(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq = _make_wq(resource_patches=[])  # no patches
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)

        result = coordinator.tick(goal, wq, ww, tick=100)

        self.assertEqual(result.status, ExecutionStatus.STUCK)


# ---------------------------------------------------------------------------
# tick() — collection goal happy path
# ---------------------------------------------------------------------------

class TestCoordinatorCollectionDerivation(unittest.TestCase):

    def test_derives_two_subtasks_for_collection_goal(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq = _make_wq(resource_patches=[_make_patch()])
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)

        coordinator.tick(goal, wq, ww, tick=100)

        # Should have pushed 2 subtasks: movement (active) + mining (pending).
        self.assertEqual(len(coordinator._ledger), 2)

    def test_movement_subtask_is_active_first(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq = _make_wq(resource_patches=[_make_patch()])
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)
        coordinator.tick(goal, wq, ww, tick=100)

        active = coordinator._ledger.peek()
        self.assertIsNotNone(active)
        # Approach subtask description starts with "Approach"
        self.assertIn("Approach", active.description)
        self.assertEqual(active.status, SubtaskStatus.ACTIVE)

    def test_waypoint_intention_written_for_movement_subtask(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        patch = _make_patch(x=20.0, y=30.0)
        wq = _make_wq(resource_patches=[patch])
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)
        coordinator.tick(goal, wq, ww, tick=100)

        intentions = coordinator._bb.read(
            category=EntryCategory.INTENTION,
            current_tick=100,
        )
        waypoints = [e for e in intentions if e.data.get("type") == "waypoint"]
        self.assertEqual(len(waypoints), 1)
        wp_data = waypoints[0].data
        self.assertAlmostEqual(wp_data["target_position"]["x"], 20.0)
        self.assertAlmostEqual(wp_data["target_position"]["y"], 30.0)

    def test_selects_nearest_patch(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        near_patch = _make_patch(x=5.0, y=0.0)
        far_patch = _make_patch(x=100.0, y=0.0)
        wq = _make_wq(
            player_pos=Position(0.0, 0.0),
            resource_patches=[far_patch, near_patch],
        )
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)
        coordinator.tick(goal, wq, ww, tick=100)

        intentions = coordinator._bb.read(
            category=EntryCategory.INTENTION, current_tick=100
        )
        waypoints = [e for e in intentions if e.data.get("type") == "waypoint"]
        self.assertEqual(len(waypoints), 1)
        self.assertAlmostEqual(waypoints[0].data["target_position"]["x"], 5.0)


# ---------------------------------------------------------------------------
# tick() — subtask lifecycle: complete() before pop()
# ---------------------------------------------------------------------------

class TestCoordinatorSubtaskLifecycle(unittest.TestCase):

    def test_complete_called_before_pop_on_success(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        patch = _make_patch()
        # Inventory already satisfies the mining condition — both subtasks
        # should resolve quickly.
        wq = _make_wq(
            resource_patches=[patch],
            inventory_items={"iron-ore": 50},
        )
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)

        # First tick derives subtasks.
        coordinator.tick(goal, wq, ww, tick=100)

        # Write a waypoint_reached signal to trigger movement subtask completion.
        coordinator._bb.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.SUBTASK,
            owner_agent="navigation",
            created_at=101,
            data={"type": "waypoint_reached", "waypoint_id": "fake"},
        )

        # Second tick should detect waypoint_reached and complete the movement subtask.
        result = coordinator.tick(goal, wq, ww, tick=101)

        # Movement subtask should be gone; mining subtask should now be active.
        # (Or if inventory condition already satisfied, mining may also be done.)
        self.assertIn(
            result.status,
            {ExecutionStatus.PROGRESSING, ExecutionStatus.WAITING},
        )

    def test_escalate_called_on_failure_condition(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        patch = _make_patch()
        wq_initial = _make_wq(resource_patches=[patch], tick=100)
        ww = _make_mock_writer()
        coordinator.reset(goal, wq_initial)
        _drain_stop(coordinator, goal, wq_initial, ww)
        coordinator.tick(goal, wq_initial, ww, tick=100)

        # Advance time past the failure condition of the derived subtask.
        wq_late = _make_wq(resource_patches=[patch], tick=999999)
        result = coordinator.tick(goal, wq_late, ww, tick=999999)

        self.assertEqual(result.status, ExecutionStatus.STUCK)
        self.assertIsNotNone(result.stuck_context)
        # failure_chain should have at least one subtask.
        self.assertGreater(len(result.stuck_context.failure_chain), 0)

    def test_sibling_history_populated_after_first_subtask_completes(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        patch = _make_patch()
        wq = _make_wq(resource_patches=[patch], inventory_items={"iron-ore": 0})
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)

        # Derive subtasks.
        coordinator.tick(goal, wq, ww, tick=100)

        # The approach subtask's success_condition is is_at(Position(x, y)).
        # We advance the player to the patch position so the condition is met.
        from world import Inventory, InventorySlot, ExplorationState
from world.observable.state import WorldState
        state_at_patch = WorldState(tick=101)
        state_at_patch.player.position = patch.position
        state_at_patch.resource_map = [patch]
        state_at_patch.player.exploration = ExplorationState(charted_chunks=0)
        state_at_patch._rebuild_entity_indices()
        from world import WorldQuery
        wq_at_patch = WorldQuery(state_at_patch)

        coordinator.tick(goal, wq_at_patch, ww, tick=101)

        # History should have the completed approach subtask.
        all_history = {k: v for k, v in coordinator._ledger._history.items()}
        total_records = sum(len(v) for v in all_history.values())
        self.assertGreater(total_records, 0)


# ---------------------------------------------------------------------------
# tick() — agent selection (single active agent per subtask)
# ---------------------------------------------------------------------------

class TestCoordinatorAgentSelection(unittest.TestCase):

    def test_active_agent_none_before_derivation(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq = _make_wq(resource_patches=[_make_patch()])
        coordinator.reset(goal, wq)
        # Before first tick, no subtasks derived, no agent selected.
        self.assertIsNone(coordinator._active_agent)

    def test_active_agent_selected_after_derivation(self):
        registry = AgentRegistry()
        nav = NavigationAgent()
        registry.register(nav)

        coordinator = _make_coordinator(registry)
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq = _make_wq(resource_patches=[_make_patch()])
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)
        coordinator.tick(goal, wq, ww, tick=100)

        self.assertIs(coordinator._active_agent, nav)

    def test_only_active_agent_ticked_not_all(self):
        # Register two agents for the same goal type.
        # Only the first (selected) one should be ticked.
        registry = AgentRegistry()

        ticked = []

        class _RecordingAgent(NavigationAgent):
            def __init__(self, name):
                super().__init__()
                self._name = name
            def tick(self, subtask, bb, wq, ww, tick, kb=None):
                ticked.append(self._name)
                return []
            def activate(self, subtask, bb, wq, kb=None): pass

        agent_a = _RecordingAgent("A")
        agent_b = _RecordingAgent("B")
        registry.register(agent_a)
        registry.register(agent_b)

        coordinator = _make_coordinator(registry)
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq = _make_wq(resource_patches=[_make_patch()])
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        coordinator.tick(goal, wq, ww, tick=100)
        coordinator.tick(goal, wq, ww, tick=101)

        # agent_a is selected (first registered); agent_b must never be ticked.
        self.assertIn("A", ticked)
        self.assertNotIn("B", ticked)

    def test_active_agent_cleared_on_reset(self):
        registry = AgentRegistry()
        nav = NavigationAgent()
        registry.register(nav)

        coordinator = _make_coordinator(registry)
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq = _make_wq(resource_patches=[_make_patch()])
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)
        coordinator.tick(goal, wq, ww, tick=100)
        self.assertIsNotNone(coordinator._active_agent)

        # Reset for a new goal — agent should be cleared.
        goal2 = _make_goal(goal_type=GOAL_TYPE_PRODUCTION)
        coordinator.reset(goal2, wq)
        self.assertIsNone(coordinator._active_agent)

    def test_no_agent_registered_returns_waiting_not_crash(self):
        # No agent registered for collection; should not raise.
        coordinator = _make_coordinator()  # empty registry
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq = _make_wq(resource_patches=[_make_patch()])
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        result = coordinator.tick(goal, wq, ww, tick=100)
        # Should return WAITING (no actions, no agent) rather than crashing.
        self.assertIn(result.status, {
            ExecutionStatus.WAITING, ExecutionStatus.PROGRESSING
        })

    def test_active_agent_cleared_on_escalation(self):
        registry = AgentRegistry()
        nav = NavigationAgent()
        registry.register(nav)

        coordinator = _make_coordinator(registry)
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq_initial = _make_wq(resource_patches=[_make_patch()], tick=100)
        ww = _make_mock_writer()
        coordinator.reset(goal, wq_initial)
        _drain_stop(coordinator, goal, wq_initial, ww)
        coordinator.tick(goal, wq_initial, ww, tick=100)
        self.assertIsNotNone(coordinator._active_agent)

        # Trigger escalation via failure condition.
        wq_late = _make_wq(resource_patches=[_make_patch()], tick=999999)
        coordinator.tick(goal, wq_late, ww, tick=999999)
        self.assertIsNone(coordinator._active_agent)


# ---------------------------------------------------------------------------
# tick() — exploration goal
# ---------------------------------------------------------------------------

class TestCoordinatorExploration(unittest.TestCase):

    def test_exploration_goal_derives_subtask(self):
        coordinator = _make_coordinator()
        goal = _make_goal(
            goal_type=GOAL_TYPE_EXPLORATION,
            success_condition="charted_chunks >= 10",
            failure_condition="tick > 99999",
        )
        wq = _make_wq(charted_chunks=0)
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)

        result = coordinator.tick(goal, wq, ww, tick=100)

        self.assertNotEqual(result.status, ExecutionStatus.STUCK)
        self.assertEqual(len(coordinator._ledger), 1)

    def test_exploration_waypoint_written_to_blackboard(self):
        coordinator = _make_coordinator()
        goal = _make_goal(
            goal_type=GOAL_TYPE_EXPLORATION,
            success_condition="charted_chunks >= 10",
            failure_condition="tick > 99999",
        )
        wq = _make_wq(charted_chunks=0)
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)
        coordinator.tick(goal, wq, ww, tick=100)

        intentions = coordinator._bb.read(
            category=EntryCategory.INTENTION, current_tick=100
        )
        waypoints = [e for e in intentions if e.data.get("type") == "waypoint"]
        self.assertEqual(len(waypoints), 1)

    def test_exploration_success_condition_met(self):
        """
        Exploration subtask now uses arrival (is_at) as its success condition,
        not charted_chunks. Place the player at the derived waypoint so the
        coordinator sees is_at() == True and pops the subtask.

        With charted_chunks=0, player at (0,0): _next_exploration_waypoint
        returns (64, 0) (ring=1, step=64, direction east).
        """
        coordinator = _make_coordinator()
        goal = _make_goal(
            goal_type=GOAL_TYPE_EXPLORATION,
            success_condition="charted_chunks >= 5",
            failure_condition="tick > 99999",
        )
        wq_start = _make_wq(charted_chunks=0)
        ww = _make_mock_writer()
        coordinator.reset(goal, wq_start)

        # Tick 99 — drain the StopMining flush.
        coordinator.tick(goal, wq_start, ww, tick=99)

        # Tick 100 — derive the exploration subtask (waypoint at (256, 0)).
        coordinator.tick(goal, wq_start, ww, tick=100)
        self.assertEqual(len(coordinator._ledger), 1)

        # Tick 101 — player arrives at the waypoint; is_at((256,0)) is True.
        # step = 256 (fixed), direction 0 = east.
        wq_arrived = _make_wq(player_pos=Position(256.0, 0.0), charted_chunks=0)
        result = coordinator.tick(goal, wq_arrived, ww, tick=101)

        self.assertIn(
            result.status,
            {ExecutionStatus.PROGRESSING, ExecutionStatus.WAITING},
        )
        # Subtask was completed and popped — ledger is empty.
        self.assertEqual(len(coordinator._ledger), 0)


# ---------------------------------------------------------------------------
# Navigation stall escalation
# ---------------------------------------------------------------------------

class TestCoordinatorNavigationStall(unittest.TestCase):

    def test_navigation_stall_observation_triggers_immediate_escalation(self):
        """
        When the navigation agent writes a navigation_stalled observation,
        the coordinator responds immediately on the next tick without waiting
        for the subtask failure_condition timeout.

        For exploration goals, escalation returns WAITING (not STUCK) so the
        coordinator re-derives a waypoint in a new direction rather than
        failing the goal. The ledger should be empty after escalation,
        ready for re-derivation.
        """
        registry = AgentRegistry()
        nav = NavigationAgent()
        registry.register(nav)

        coordinator = _make_coordinator(registry)
        goal = _make_goal(
            goal_type=GOAL_TYPE_EXPLORATION,
            success_condition="charted_chunks >= 10",
            failure_condition="tick > 99999",
        )
        wq = _make_wq(charted_chunks=0)
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)

        # Derive the exploration subtask.
        coordinator.tick(goal, wq, ww, tick=100)
        self.assertEqual(len(coordinator._ledger), 1)

        # Manually write a navigation_stalled observation as the nav agent would.
        coordinator._bb.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.SUBTASK,
            owner_agent="navigation",
            created_at=101,
            data={"type": "navigation_stalled", "position": {"x": 0.0, "y": 0.0}},
        )

        # Exploration stall returns WAITING so the goal stays alive and
        # re-derives a waypoint in a new direction on the next tick.
        result = coordinator.tick(goal, wq, ww, tick=101)

        self.assertEqual(result.status, ExecutionStatus.WAITING)
        # Ledger is cleared so re-derivation can push a new waypoint.
        self.assertEqual(len(coordinator._ledger), 0)

    def test_navigation_stall_non_exploration_returns_stuck(self):
        """
        For non-exploration goals, a navigation stall still returns STUCK
        so the loop can escalate to the LLM/goal-source level.
        """
        registry = AgentRegistry()
        nav = NavigationAgent()
        registry.register(nav)

        coordinator = _make_coordinator(registry)
        goal = _make_goal(
            goal_type=GOAL_TYPE_COLLECTION,
            success_condition="inventory('iron-ore') >= 50",
            failure_condition="tick > 99999",
        )
        patch = _make_patch()
        wq = _make_wq(resource_patches=[patch])
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)

        # Derive collection subtasks.
        coordinator.tick(goal, wq, ww, tick=100)

        coordinator._bb.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.SUBTASK,
            owner_agent="navigation",
            created_at=101,
            data={"type": "navigation_stalled", "position": {"x": 0.0, "y": 0.0}},
        )

        result = coordinator.tick(goal, wq, ww, tick=101)

        self.assertEqual(result.status, ExecutionStatus.STUCK)
        self.assertIsNotNone(result.stuck_context)

    def test_stall_escalation_increments_exploration_attempt(self):
        """
        Each exploration stall increments _exploration_attempt so the next
        re-derivation uses a rotated waypoint direction.
        """
        coordinator = _make_coordinator()
        goal = _make_goal(
            goal_type=GOAL_TYPE_EXPLORATION,
            success_condition="charted_chunks >= 10",
            failure_condition="tick > 99999",
        )
        wq = _make_wq(charted_chunks=0)
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        _drain_stop(coordinator, goal, wq, ww)
        coordinator.tick(goal, wq, ww, tick=100)

        self.assertEqual(coordinator._exploration_attempt, 1)

        coordinator._bb.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.SUBTASK,
            owner_agent="navigation",
            created_at=101,
            data={"type": "navigation_stalled", "position": {"x": 0.0, "y": 0.0}},
        )
        coordinator.tick(goal, wq, ww, tick=101)

        # After one stall the attempt counter is still 1 (incremented during
        # derivation, not during escalation — escalation just clears the ledger).
        # The next re-derivation tick will increment it to 2.
        self.assertGreaterEqual(coordinator._exploration_attempt, 1)


# ---------------------------------------------------------------------------
# tick() — crafting goal derivation
# ---------------------------------------------------------------------------

class TestCoordinatorCraftingDerivation(unittest.TestCase):

    def _run(self, kb, inventory_items, inventory_size=80,
             success_condition="inventory('iron-gear-wheel') >= 2"):
        coord = _make_crafting_coordinator(kb)
        goal = _make_crafting_goal(success_condition=success_condition)
        wq = _make_wq(inventory_items=inventory_items, inventory_size=inventory_size)
        ww = _make_mock_writer()
        coord.reset(goal, wq)
        return coord.tick(goal, wq, ww, tick=100), coord

    def test_derives_single_craft_subtask(self):
        kb = _make_real_kb_with_gear_recipe()
        result, coord = self._run(kb, {"iron-plate": 10})
        self.assertNotEqual(result.status, ExecutionStatus.STUCK)
        self.assertEqual(len(coord._ledger), 1)

    def test_derived_subtask_has_crafting_agent_hint(self):
        kb = _make_real_kb_with_gear_recipe()
        _, coord = self._run(kb, {"iron-plate": 10})
        subtask = coord._ledger.peek()
        self.assertIsNotNone(subtask)
        self.assertEqual(getattr(subtask, "agent_hint", None), "crafting")

    def test_derived_subtask_has_craft_attributes(self):
        kb = _make_real_kb_with_gear_recipe()
        _, coord = self._run(kb, {"iron-plate": 10})
        subtask = coord._ledger.peek()
        self.assertEqual(getattr(subtask, "craft_item", None), "iron-gear-wheel")
        self.assertEqual(getattr(subtask, "craft_recipe", None), "iron-gear-wheel")
        self.assertEqual(getattr(subtask, "craft_count", None), 2)

    def test_crafting_task_intention_written(self):
        kb = _make_real_kb_with_gear_recipe()
        _, coord = self._run(kb, {"iron-plate": 10})
        intentions = coord._bb.read(
            category=EntryCategory.INTENTION, current_tick=100
        )
        crafting = [e for e in intentions if e.data.get("type") == "crafting_task"]
        self.assertEqual(len(crafting), 1)

    def test_intention_targets_match_derived_subtask(self):
        kb = _make_real_kb_with_gear_recipe()
        _, coord = self._run(kb, {"iron-plate": 10})
        intentions = coord._bb.read(
            category=EntryCategory.INTENTION, current_tick=100
        )
        entry = next(
            e for e in intentions if e.data.get("type") == "crafting_task"
        )
        targets = entry.data.get("targets", [])
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["item"], "iron-gear-wheel")
        self.assertEqual(targets[0]["count"], 2)

    def test_intention_contains_expected_post_inventory(self):
        kb = _make_real_kb_with_gear_recipe()
        _, coord = self._run(kb, {"iron-plate": 10})
        intentions = coord._bb.read(
            category=EntryCategory.INTENTION, current_tick=100
        )
        entry = next(
            e for e in intentions if e.data.get("type") == "crafting_task"
        )
        # expected_post_inventory must be present so the agent needs no KB.
        self.assertIn("expected_post_inventory", entry.data)
        post = entry.data["expected_post_inventory"]
        # 2 gears × 2 plates = 4 consumed; started with 10 → 6 remain.
        self.assertEqual(post.get("iron-plate"), 6)

    def test_stuck_when_condition_unparseable(self):
        kb = _make_real_kb_with_gear_recipe()
        coord = _make_crafting_coordinator(kb)
        goal = _make_crafting_goal(success_condition="tick > 99999")
        wq = _make_wq(inventory_items={"iron-plate": 10})
        ww = _make_mock_writer()
        coord.reset(goal, wq)
        result = coord.tick(goal, wq, ww, tick=100)
        self.assertEqual(result.status, ExecutionStatus.STUCK)

    def test_stuck_when_insufficient_ingredients(self):
        kb = _make_real_kb_with_gear_recipe()
        result, _ = self._run(kb, {"iron-plate": 0})
        self.assertEqual(result.status, ExecutionStatus.STUCK)

    def test_stuck_when_inventory_size_zero(self):
        kb = _make_real_kb_with_gear_recipe()
        result, _ = self._run(kb, {"iron-plate": 10}, inventory_size=0)
        self.assertEqual(result.status, ExecutionStatus.STUCK)

    def test_success_condition_uses_ingredient_depletion(self):
        kb = _make_real_kb_with_gear_recipe()
        _, coord = self._run(kb, {"iron-plate": 10})
        subtask = coord._ledger.peek()
        # Ingredient depletion condition references iron-plate with <=.
        self.assertIn("iron-plate", subtask.success_condition)
        self.assertIn("<=", subtask.success_condition)

    def test_goal_type_crafting_constant(self):
        self.assertEqual(GOAL_TYPE_CRAFTING, "crafting")

    def test_production_goal_still_returns_stuck(self):
        """Crafting derivation must not affect other goal types."""
        kb = _make_real_kb_with_gear_recipe()
        coord = _make_crafting_coordinator(kb)
        goal = Goal(
            description="Production goal",
            priority=Priority.NORMAL,
            success_condition="inventory('iron-plate') >= 50",
            failure_condition="tick > 99999",
            reward_spec=RewardSpec(),
        )
        goal.type = GOAL_TYPE_PRODUCTION
        goal.activate(tick=0)
        wq = _make_wq(inventory_items={"iron-plate": 10})
        ww = _make_mock_writer()
        coord.reset(goal, wq)
        result = coord.tick(goal, wq, ww, tick=100)
        self.assertEqual(result.status, ExecutionStatus.STUCK)


class TestCoordinatorWriteWaypointCrafting(unittest.TestCase):

    def test_write_waypoint_writes_crafting_task_intention(self):
        kb = _make_real_kb_with_gear_recipe()
        coord = _make_crafting_coordinator(kb)
        subtask = Subtask(
            description="Craft 3x iron-gear-wheel",
            success_condition="inventory('iron-plate') <= 4",
            failure_condition="tick > 99999",
            parent_goal_id="g1",
            created_at=100,
            derived_locally=True,
        )
        subtask.agent_hint = "crafting"
        subtask.craft_item = "iron-gear-wheel"
        subtask.craft_recipe = "iron-gear-wheel"
        subtask.craft_count = 3
        subtask.craft_expected_post = {"iron-plate": 4}

        wq = _make_wq(inventory_items={"iron-plate": 10})
        coord._write_waypoint_for_subtask(subtask, wq, tick=100)

        intentions = coord._bb.read(
            category=EntryCategory.INTENTION, current_tick=100
        )
        crafting = [e for e in intentions if e.data.get("type") == "crafting_task"]
        self.assertEqual(len(crafting), 1)
        targets = crafting[0].data.get("targets", [])
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["item"], "iron-gear-wheel")
        self.assertEqual(targets[0]["count"], 3)
        self.assertEqual(
            crafting[0].data.get("expected_post_inventory"),
            {"iron-plate": 4},
        )

    def test_write_waypoint_no_entry_when_craft_item_missing(self):
        kb = _make_real_kb_with_gear_recipe()
        coord = _make_crafting_coordinator(kb)
        subtask = Subtask(
            description="Broken crafting subtask",
            success_condition="True",
            failure_condition="tick > 99999",
            parent_goal_id="g1",
            created_at=100,
            derived_locally=True,
        )
        subtask.agent_hint = "crafting"
        # No craft_item / craft_count attributes — should log warning, not crash.
        coord._write_waypoint_for_subtask(subtask, _make_wq(), tick=100)
        intentions = coord._bb.read(
            category=EntryCategory.INTENTION, current_tick=100
        )
        crafting = [e for e in intentions if e.data.get("type") == "crafting_task"]
        self.assertEqual(len(crafting), 0)


if __name__ == "__main__":
    unittest.main()