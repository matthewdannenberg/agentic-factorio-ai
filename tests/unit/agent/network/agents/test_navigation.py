"""
tests/unit/agent/test_navigation.py

Unit tests for agent/network/agents/navigation.py.

All tests run without a live Factorio instance.
"""

import unittest

from agent.blackboard import Blackboard, EntryCategory, EntryScope
from agent.network.agents.navigation import AGENT_ID, NavigationAgent
from bridge.actions import MineResource, MoveTo
from planning.goal import GoalStatus, Priority, RewardSpec, Goal
from world.state import (
    EntityState,
    Inventory,
    InventorySlot,
    PlayerState,
    Position,
    WorldState,
)
from world.query import WorldQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_goal(goal_type: str = "collection") -> Goal:
    spec = RewardSpec()
    g = Goal(
        description="Test goal",
        priority=Priority.NORMAL,
        success_condition="inventory('iron-ore') >= 50",
        failure_condition="tick > 9999",
        reward_spec=spec,
    )
    g.type = goal_type  # type annotation added by coordinator convention
    return g


def _make_wq(
    player_pos: Position = None,
    reachable: list = None,
    entities: list = None,
    tick: int = 100,
) -> WorldQuery:
    state = WorldState(tick=tick, entities=entities or [])
    state.player.position = player_pos or Position(0.0, 0.0)
    state.player.reachable = reachable or []
    state._rebuild_entity_indices()
    return WorldQuery(state)


def _make_waypoint_entry(
    bb: Blackboard,
    target_pos: Position,
    target_entity_id=None,
    waypoint_type: str = "move",
    tick: int = 100,
):
    return bb.write(
        category=EntryCategory.INTENTION,
        scope=EntryScope.SUBTASK,
        owner_agent="coordinator",
        created_at=tick,
        data={
            "type": "waypoint",
            "waypoint_type": waypoint_type,
            "target_position": {"x": target_pos.x, "y": target_pos.y},
            "target_entity_id": target_entity_id,
            "purpose": "test_waypoint",
        },
    )


def _make_mock_writer():
    """Minimal WorldWriter stub."""
    class _MockWriter:
        pass
    return _MockWriter()


# ---------------------------------------------------------------------------
# activate()
# ---------------------------------------------------------------------------

class TestNavigationAgentActivate(unittest.TestCase):

    def test_activate_writes_position_observation(self):
        agent = NavigationAgent()
        bb = Blackboard()
        wq = _make_wq(player_pos=Position(3.0, 7.0))
        goal = _make_goal()

        agent.activate(goal, bb, wq)

        obs = bb.read(category=EntryCategory.OBSERVATION, current_tick=100)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].data["type"], "player_position")
        self.assertAlmostEqual(obs[0].data["position"]["x"], 3.0)
        self.assertAlmostEqual(obs[0].data["position"]["y"], 7.0)
        self.assertEqual(obs[0].owner_agent, AGENT_ID)

    def test_activate_resets_progress_state(self):
        agent = NavigationAgent()
        bb = Blackboard()
        wq = _make_wq()
        goal = _make_goal()

        # Simulate a previous goal having set state.
        agent._waypoints_completed = 5
        agent._waypoints_total = 5

        agent.activate(goal, bb, wq)

        self.assertEqual(agent._waypoints_completed, 0)
        self.assertEqual(agent._waypoints_total, 0)


# ---------------------------------------------------------------------------
# tick() — movement
# ---------------------------------------------------------------------------

class TestNavigationAgentTick(unittest.TestCase):

    def setUp(self):
        self.agent = NavigationAgent()
        self.bb = Blackboard()
        self.wq = _make_wq(player_pos=Position(0.0, 0.0))
        self.ww = _make_mock_writer()
        self.goal = _make_goal()
        self.agent.activate(self.goal, self.bb, self.wq)
        # Clear the activation observation so tests start clean.
        self.bb.clear_all()

    def test_returns_moveto_when_not_at_waypoint(self):
        _make_waypoint_entry(self.bb, target_pos=Position(50.0, 50.0))
        actions = self.agent.tick(self.bb, self.wq, self.ww, tick=101)

        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MoveTo)
        self.assertTrue(actions[0].pathfind)  # always pathfind=True

    def test_moveto_target_is_waypoint_position(self):
        _make_waypoint_entry(self.bb, target_pos=Position(30.0, -10.0))
        actions = self.agent.tick(self.bb, self.wq, self.ww, tick=101)

        self.assertEqual(len(actions), 1)
        self.assertAlmostEqual(actions[0].position.x, 30.0)
        self.assertAlmostEqual(actions[0].position.y, -10.0)

    def test_returns_empty_when_no_waypoint(self):
        # No waypoint written — should do nothing.
        actions = self.agent.tick(self.bb, self.wq, self.ww, tick=101)
        self.assertEqual(actions, [])

    def test_writes_position_observation_every_tick(self):
        actions = self.agent.tick(self.bb, self.wq, self.ww, tick=101)
        pos_obs = [
            e for e in self.bb.read(current_tick=101)
            if e.data.get("type") == "player_position"
        ]
        self.assertGreaterEqual(len(pos_obs), 1)

    def test_does_not_read_purpose_field(self):
        # Write a waypoint with a purpose that would break things if branched on.
        self.bb.write(
            category=EntryCategory.INTENTION,
            scope=EntryScope.SUBTASK,
            owner_agent="coordinator",
            created_at=101,
            data={
                "type": "waypoint",
                "waypoint_type": "move",
                "target_position": {"x": 10.0, "y": 10.0},
                "target_entity_id": None,
                "purpose": "SHOULD_NOT_CAUSE_BRANCH",
            },
        )
        # Should produce a MoveTo regardless of purpose.
        actions = self.agent.tick(self.bb, self.wq, self.ww, tick=101)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MoveTo)


# ---------------------------------------------------------------------------
# tick() — arrival / waypoint_reached
# ---------------------------------------------------------------------------

class TestNavigationAgentArrival(unittest.TestCase):

    def test_writes_waypoint_reached_when_at_position(self):
        agent = NavigationAgent()
        bb = Blackboard()
        # Player is already at the waypoint position.
        wq = _make_wq(player_pos=Position(10.0, 10.0))
        ww = _make_mock_writer()
        goal = _make_goal()
        agent.activate(goal, bb, wq)
        bb.clear_all()

        _make_waypoint_entry(bb, target_pos=Position(10.0, 10.0), tick=100)
        actions = agent.tick(bb, wq, ww, tick=101)

        # Should return empty — arrival detected.
        self.assertEqual(actions, [])

        # Should have written waypoint_reached observation.
        reached = [
            e for e in bb.read(current_tick=101)
            if e.data.get("type") == "waypoint_reached"
        ]
        self.assertEqual(len(reached), 1)
        self.assertEqual(reached[0].owner_agent, AGENT_ID)

    def test_writes_waypoint_reached_when_entity_reachable(self):
        agent = NavigationAgent()
        bb = Blackboard()
        entity = EntityState(
            entity_id=42,
            name="iron-ore",
            position=Position(5.0, 5.0),
        )
        # Entity is in reachable list.
        wq = _make_wq(
            player_pos=Position(4.0, 5.0),
            reachable=[42],
            entities=[entity],
        )
        ww = _make_mock_writer()
        goal = _make_goal()
        agent.activate(goal, bb, wq)
        bb.clear_all()

        bb.write(
            category=EntryCategory.INTENTION,
            scope=EntryScope.SUBTASK,
            owner_agent="coordinator",
            created_at=100,
            data={
                "type": "waypoint",
                "waypoint_type": "move",
                "target_position": {"x": 5.0, "y": 5.0},
                "target_entity_id": 42,
                "purpose": "approach",
            },
        )
        actions = agent.tick(bb, wq, ww, tick=101)

        self.assertEqual(actions, [])
        reached = [
            e for e in bb.read(current_tick=101)
            if e.data.get("type") == "waypoint_reached"
        ]
        self.assertEqual(len(reached), 1)

    def test_no_waypoint_reached_when_entity_not_reachable(self):
        agent = NavigationAgent()
        bb = Blackboard()
        entity = EntityState(
            entity_id=42,
            name="iron-ore",
            position=Position(50.0, 50.0),
        )
        # Entity NOT in reachable list.
        wq = _make_wq(
            player_pos=Position(0.0, 0.0),
            reachable=[],
            entities=[entity],
        )
        ww = _make_mock_writer()
        goal = _make_goal()
        agent.activate(goal, bb, wq)
        bb.clear_all()

        bb.write(
            category=EntryCategory.INTENTION,
            scope=EntryScope.SUBTASK,
            owner_agent="coordinator",
            created_at=100,
            data={
                "type": "waypoint",
                "waypoint_type": "move",
                "target_position": {"x": 50.0, "y": 50.0},
                "target_entity_id": 42,
                "purpose": "approach",
            },
        )
        actions = agent.tick(bb, wq, ww, tick=101)

        # Should still be moving.
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MoveTo)


# ---------------------------------------------------------------------------
# progress()
# ---------------------------------------------------------------------------

class TestNavigationAgentProgress(unittest.TestCase):

    def test_progress_zero_with_no_waypoints(self):
        agent = NavigationAgent()
        bb = Blackboard()
        wq = _make_wq()
        goal = _make_goal()
        agent.activate(goal, bb, wq)

        self.assertAlmostEqual(agent.progress(bb, wq), 0.0)

    def test_progress_one_when_all_complete(self):
        agent = NavigationAgent()
        bb = Blackboard()
        wq = _make_wq()
        goal = _make_goal()
        agent.activate(goal, bb, wq)

        # Simulate completion of 2 waypoints.
        agent._waypoints_total = 2
        agent._waypoints_completed = 2

        self.assertAlmostEqual(agent.progress(bb, wq), 1.0)

    def test_progress_fraction(self):
        agent = NavigationAgent()
        bb = Blackboard()
        wq = _make_wq()
        goal = _make_goal()
        agent.activate(goal, bb, wq)

        agent._waypoints_total = 4
        agent._waypoints_completed = 1

        self.assertAlmostEqual(agent.progress(bb, wq), 0.25)


# ---------------------------------------------------------------------------
# observe()
# ---------------------------------------------------------------------------

class TestNavigationAgentObserve(unittest.TestCase):

    def test_observe_includes_player_position(self):
        agent = NavigationAgent()
        bb = Blackboard()
        wq = _make_wq(player_pos=Position(12.0, -3.0))
        goal = _make_goal()
        agent.activate(goal, bb, wq)

        obs = agent.observe(bb, wq)
        self.assertIn("player_position", obs)
        self.assertAlmostEqual(obs["player_position"]["x"], 12.0)
        self.assertAlmostEqual(obs["player_position"]["y"], -3.0)

    def test_observe_includes_waypoint_count(self):
        agent = NavigationAgent()
        bb = Blackboard()
        wq = _make_wq()
        goal = _make_goal()
        agent.activate(goal, bb, wq)
        agent._waypoints_completed = 3
        agent._waypoints_total = 5

        obs = agent.observe(bb, wq)
        self.assertEqual(obs["waypoints_completed"], 3)
        self.assertEqual(obs["waypoints_total"], 5)


if __name__ == "__main__":
    unittest.main()
