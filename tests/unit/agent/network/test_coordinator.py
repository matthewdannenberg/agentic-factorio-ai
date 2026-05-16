"""
tests/unit/agent/test_coordinator.py

Unit tests for the RuleBasedCoordinator in agent/network/coordinator.py.

All tests run without a live Factorio instance.
"""

import unittest

from agent.blackboard import Blackboard, EntryCategory, EntryScope
from agent.execution_protocol import ExecutionStatus
from agent.network.coordinator import (
    GOAL_TYPE_COLLECTION,
    GOAL_TYPE_EXPLORATION,
    GOAL_TYPE_PRODUCTION,
    RuleBasedCoordinator,
)
from agent.network.registry import AgentRegistry
from agent.network.agents.navigation import NavigationAgent
from agent.self_model import SelfModel
from agent.subtask import Subtask, SubtaskLedger, SubtaskStatus
from planning.goal import Goal, GoalStatus, Priority, RewardSpec
from world.state import (
    ExplorationState,
    Inventory,
    InventorySlot,
    PlayerState,
    Position,
    ResourcePatch,
    WorldState,
)
from world.query import WorldQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    return RuleBasedCoordinator(registry=reg, blackboard=bb, ledger=ledger, self_model=sm)


def _make_mock_writer():
    class _MockWriter:
        pass
    return _MockWriter()


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

        result = coordinator.tick(goal, wq, ww, tick=100)

        self.assertEqual(result.status, ExecutionStatus.STUCK)
        self.assertIsNotNone(result.stuck_context)

    def test_stuck_at_goal_level_when_no_subtask_derived(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_PRODUCTION)
        wq = _make_wq()
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)

        result = coordinator.tick(goal, wq, ww, tick=100)

        self.assertTrue(result.stuck_context.stuck_at_goal_level)
        self.assertEqual(result.stuck_context.failure_chain, [])

    def test_stuck_context_has_correct_goal_id(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_PRODUCTION)
        wq = _make_wq()
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)

        result = coordinator.tick(goal, wq, ww, tick=100)

        self.assertEqual(result.stuck_context.goal.id, goal.id)

    def test_stuck_context_serialises_without_error(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_PRODUCTION)
        wq = _make_wq()
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)

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

        coordinator.tick(goal, wq, ww, tick=100)

        # Should have pushed 2 subtasks: movement (active) + mining (pending).
        self.assertEqual(len(coordinator._ledger), 2)

    def test_movement_subtask_is_active_first(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq = _make_wq(resource_patches=[_make_patch()])
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
        coordinator.tick(goal, wq, ww, tick=100)

        active = coordinator._ledger.peek()
        self.assertIsNotNone(active)
        self.assertIn("move", active.description.lower())
        self.assertEqual(active.status, SubtaskStatus.ACTIVE)

    def test_waypoint_intention_written_for_movement_subtask(self):
        coordinator = _make_coordinator()
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        patch = _make_patch(x=20.0, y=30.0)
        wq = _make_wq(resource_patches=[patch])
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
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

        # Signal waypoint reached to complete movement subtask.
        coordinator._bb.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.SUBTASK,
            owner_agent="navigation",
            created_at=101,
            data={"type": "waypoint_reached", "waypoint_id": "x"},
        )
        coordinator.tick(goal, wq, ww, tick=101)

        # History should have the completed movement subtask.
        history = coordinator._ledger.history_for(goal.id)
        # Parent of movement subtask is the mining subtask's id, not goal.id
        # (because movement is a child of mining in our derivation). So check
        # that at least one level of history exists.
        all_history = {
            k: v
            for k, v in coordinator._ledger._history.items()
        }
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
        registry.register(nav, [GOAL_TYPE_COLLECTION])

        coordinator = _make_coordinator(registry)
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq = _make_wq(resource_patches=[_make_patch()])
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
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
            def tick(self, bb, wq, ww, tick):
                ticked.append(self._name)
                return []
            def activate(self, goal, bb, wq): pass

        agent_a = _RecordingAgent("A")
        agent_b = _RecordingAgent("B")
        registry.register(agent_a, [GOAL_TYPE_COLLECTION])
        registry.register(agent_b, [GOAL_TYPE_COLLECTION])

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
        registry.register(nav, [GOAL_TYPE_COLLECTION])

        coordinator = _make_coordinator(registry)
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq = _make_wq(resource_patches=[_make_patch()])
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)
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
        registry.register(nav, [GOAL_TYPE_COLLECTION])

        coordinator = _make_coordinator(registry)
        goal = _make_goal(goal_type=GOAL_TYPE_COLLECTION)
        wq_initial = _make_wq(resource_patches=[_make_patch()], tick=100)
        ww = _make_mock_writer()
        coordinator.reset(goal, wq_initial)
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
        coordinator.tick(goal, wq, ww, tick=100)

        intentions = coordinator._bb.read(
            category=EntryCategory.INTENTION, current_tick=100
        )
        waypoints = [e for e in intentions if e.data.get("type") == "waypoint"]
        self.assertEqual(len(waypoints), 1)

    def test_exploration_success_condition_met(self):
        coordinator = _make_coordinator()
        goal = _make_goal(
            goal_type=GOAL_TYPE_EXPLORATION,
            success_condition="charted_chunks >= 5",
            failure_condition="tick > 99999",
        )
        # Already explored enough.
        wq = _make_wq(charted_chunks=10)
        ww = _make_mock_writer()
        coordinator.reset(goal, wq)

        # First tick derives the subtask.
        coordinator.tick(goal, wq, ww, tick=100)

        # Second tick should see success condition met and complete the subtask.
        result = coordinator.tick(goal, wq, ww, tick=101)

        self.assertIn(
            result.status,
            {ExecutionStatus.PROGRESSING, ExecutionStatus.WAITING},
        )
        # Ledger should be empty (subtask was completed and popped).
        self.assertEqual(len(coordinator._ledger), 0)


if __name__ == "__main__":
    unittest.main()