"""
tests/unit/agent/test_navigation.py

Unit tests for agent/network/agents/navigation.py.

All tests run without a live Factorio instance.

Key differences from the original test file:
  - All AgentProtocol methods now take (subtask, ...) as first argument —
    agents no longer receive Goal objects.
  - On arrival, StopMovement is returned (not a waypoint_reached observation).
  - The coordinator detects arrival by evaluating the subtask success_condition
    (is_at / is_reachable), not by reading a blackboard signal.
"""

import unittest

from agent.blackboard import Blackboard, EntryCategory, EntryScope
from agent.network.agents.navigation import NavigationAgent
from agent.subtask import Subtask
from bridge.actions import MoveTo, StopMovement
from world.state import (
    EntityState,
    Position,
    WorldState,
)
from world.query import WorldQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subtask(description: str = "test movement subtask") -> Subtask:
    return Subtask(
        description=description,
        success_condition="is_at(Position(10.0, 10.0))",
        failure_condition="tick > 9999",
        parent_goal_id="goal-1",
        created_at=0,
        derived_locally=True,
    )


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


def _make_waypoint_entry(bb, target_pos, target_entity_id=None, tick=100):
    return bb.write(
        category=EntryCategory.INTENTION,
        scope=EntryScope.SUBTASK,
        owner_agent="coordinator",
        created_at=tick,
        data={
            "type": "waypoint",
            "waypoint_type": "move",
            "target_position": {"x": target_pos.x, "y": target_pos.y},
            "target_entity_id": target_entity_id,
            "purpose": "test_waypoint",
        },
    )


def _make_mock_writer():
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
        subtask = _make_subtask()

        agent.activate(subtask, bb, wq)

        obs = bb.read(category=EntryCategory.OBSERVATION, current_tick=100)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].data["type"], "player_position")
        self.assertAlmostEqual(obs[0].data["position"]["x"], 3.0)
        self.assertAlmostEqual(obs[0].data["position"]["y"], 7.0)
        self.assertEqual(obs[0].owner_agent, NavigationAgent.AGENT_ID)

    def test_activate_resets_progress_state(self):
        agent = NavigationAgent()
        agent._waypoints_completed = 5
        agent._waypoints_total = 5

        bb = Blackboard()
        wq = _make_wq()
        agent.activate(_make_subtask(), bb, wq)

        self.assertEqual(agent._waypoints_completed, 0)
        self.assertEqual(agent._waypoints_total, 0)

    def test_activate_clears_last_issued_target(self):
        from world.state import Position as P
        agent = NavigationAgent()
        agent._last_issued_target = P(5.0, 5.0)
        agent._last_move_tick = 99

        agent.activate(_make_subtask(), Blackboard(), _make_wq())

        self.assertIsNone(agent._last_issued_target)
        self.assertEqual(agent._last_move_tick, 0)


# ---------------------------------------------------------------------------
# tick() — movement
# ---------------------------------------------------------------------------

class TestNavigationAgentTick(unittest.TestCase):

    def setUp(self):
        self.agent = NavigationAgent()
        self.bb = Blackboard()
        self.wq = _make_wq(player_pos=Position(0.0, 0.0))
        self.ww = _make_mock_writer()
        self.subtask = _make_subtask()
        self.agent.activate(self.subtask, self.bb, self.wq)
        self.bb.clear_all()

    def test_returns_moveto_when_not_at_waypoint(self):
        _make_waypoint_entry(self.bb, target_pos=Position(50.0, 50.0))
        actions = self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=101)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MoveTo)
        self.assertTrue(actions[0].pathfind)

    def test_moveto_target_is_waypoint_position(self):
        _make_waypoint_entry(self.bb, target_pos=Position(30.0, -10.0))
        actions = self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=101)
        self.assertEqual(len(actions), 1)
        self.assertAlmostEqual(actions[0].position.x, 30.0)
        self.assertAlmostEqual(actions[0].position.y, -10.0)

    def test_returns_empty_when_no_waypoint(self):
        actions = self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=101)
        self.assertEqual(actions, [])

    def test_writes_position_observation_every_tick(self):
        self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=101)
        pos_obs = [
            e for e in self.bb.read(current_tick=101)
            if e.data.get("type") == "player_position"
        ]
        self.assertGreaterEqual(len(pos_obs), 1)

    def test_does_not_read_purpose_field(self):
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
        actions = self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=101)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MoveTo)

    def test_redundant_moveto_suppressed_same_target(self):
        """Second tick to same target should not re-issue MoveTo."""
        _make_waypoint_entry(self.bb, target_pos=Position(50.0, 50.0))
        # First tick — issues MoveTo.
        actions1 = self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=101)
        self.assertEqual(len(actions1), 1)
        # Second tick same waypoint, position unchanged — suppressed.
        actions2 = self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=102)
        self.assertEqual(actions2, [])


# ---------------------------------------------------------------------------
# tick() — arrival (StopMovement, no waypoint_reached signal)
# ---------------------------------------------------------------------------

class TestNavigationAgentArrival(unittest.TestCase):

    def test_returns_stop_movement_when_at_position(self):
        """On arrival the agent emits StopMovement; the coordinator detects
        completion by evaluating the subtask success_condition, not by
        reading a blackboard signal."""
        agent = NavigationAgent()
        bb = Blackboard()
        wq = _make_wq(player_pos=Position(10.0, 10.0))
        ww = _make_mock_writer()
        subtask = _make_subtask()
        agent.activate(subtask, bb, wq)
        bb.clear_all()

        _make_waypoint_entry(bb, target_pos=Position(10.0, 10.0), tick=100)
        actions = agent.tick(subtask, bb, wq, ww, tick=101)

        # Should return StopMovement on arrival.
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], StopMovement)

    def test_no_waypoint_reached_observation_written(self):
        """The agent no longer writes waypoint_reached observations — the
        coordinator evaluates the subtask success_condition instead."""
        agent = NavigationAgent()
        bb = Blackboard()
        wq = _make_wq(player_pos=Position(10.0, 10.0))
        ww = _make_mock_writer()
        subtask = _make_subtask()
        agent.activate(subtask, bb, wq)
        bb.clear_all()

        _make_waypoint_entry(bb, target_pos=Position(10.0, 10.0), tick=100)
        agent.tick(subtask, bb, wq, ww, tick=101)

        reached = [
            e for e in bb.read(current_tick=101)
            if e.data.get("type") == "waypoint_reached"
        ]
        self.assertEqual(len(reached), 0)

    def test_returns_stop_movement_when_entity_reachable(self):
        agent = NavigationAgent()
        bb = Blackboard()
        entity = EntityState(entity_id=42, name="iron-ore", position=Position(5.0, 5.0))
        wq = _make_wq(player_pos=Position(4.0, 5.0), reachable=[42], entities=[entity])
        ww = _make_mock_writer()
        subtask = _make_subtask()
        agent.activate(subtask, bb, wq)
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
        actions = agent.tick(subtask, bb, wq, ww, tick=101)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], StopMovement)

    def test_returns_moveto_when_entity_not_reachable(self):
        agent = NavigationAgent()
        bb = Blackboard()
        entity = EntityState(entity_id=42, name="iron-ore", position=Position(50.0, 50.0))
        wq = _make_wq(player_pos=Position(0.0, 0.0), reachable=[], entities=[entity])
        ww = _make_mock_writer()
        subtask = _make_subtask()
        agent.activate(subtask, bb, wq)
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
        actions = agent.tick(subtask, bb, wq, ww, tick=101)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MoveTo)


# ---------------------------------------------------------------------------
# tick() — stall detection
# ---------------------------------------------------------------------------

class TestNavigationAgentStall(unittest.TestCase):

    def test_stall_writes_navigation_stalled_observation(self):
        """
        When the player hasn't moved for _STALL_GRACE_TICKS after a MoveTo,
        the agent writes a navigation_stalled observation so the coordinator
        can escalate immediately rather than waiting for the timeout.
        """
        from agent.network.agents.navigation import _STALL_GRACE_TICKS
        agent = NavigationAgent()
        bb = Blackboard()
        wq = _make_wq(player_pos=Position(0.0, 0.0))
        ww = _make_mock_writer()
        subtask = _make_subtask()
        agent.activate(subtask, bb, wq)
        bb.clear_all()

        # First tick — issues MoveTo, records position.
        _make_waypoint_entry(bb, target_pos=Position(50.0, 50.0))
        agent.tick(subtask, bb, wq, ww, tick=100)

        # Advance tick past grace period without the player moving.
        stall_tick = 100 + _STALL_GRACE_TICKS + 1
        agent.tick(subtask, bb, wq, ww, tick=stall_tick)

        observations = bb.read(category=EntryCategory.OBSERVATION, current_tick=stall_tick)
        stall_obs = [e for e in observations if e.data.get("type") == "navigation_stalled"]
        self.assertEqual(len(stall_obs), 1)

    def test_stall_returns_stop_movement(self):
        """On stall the agent returns StopMovement to halt any in-progress walking."""
        from agent.network.agents.navigation import _STALL_GRACE_TICKS
        agent = NavigationAgent()
        bb = Blackboard()
        wq = _make_wq(player_pos=Position(0.0, 0.0))
        ww = _make_mock_writer()
        subtask = _make_subtask()
        agent.activate(subtask, bb, wq)
        bb.clear_all()

        _make_waypoint_entry(bb, target_pos=Position(50.0, 50.0))
        agent.tick(subtask, bb, wq, ww, tick=100)

        stall_tick = 100 + _STALL_GRACE_TICKS + 1
        actions = agent.tick(subtask, bb, wq, ww, tick=stall_tick)

        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], StopMovement)

    def test_no_stall_if_player_moved(self):
        """No stall signal when the player has moved since the last tick."""
        from agent.network.agents.navigation import _STALL_GRACE_TICKS
        agent = NavigationAgent()
        bb = Blackboard()
        wq_start = _make_wq(player_pos=Position(0.0, 0.0))
        ww = _make_mock_writer()
        subtask = _make_subtask()
        agent.activate(subtask, bb, wq_start)
        bb.clear_all()

        _make_waypoint_entry(bb, target_pos=Position(50.0, 50.0))
        agent.tick(subtask, bb, wq_start, ww, tick=100)

        # Player has moved — update position.
        wq_moved = _make_wq(player_pos=Position(5.0, 0.0))
        stall_tick = 100 + _STALL_GRACE_TICKS + 1
        agent.tick(subtask, bb, wq_moved, ww, tick=stall_tick)

        observations = bb.read(category=EntryCategory.OBSERVATION, current_tick=stall_tick)
        stall_obs = [e for e in observations if e.data.get("type") == "navigation_stalled"]
        self.assertEqual(len(stall_obs), 0)


# ---------------------------------------------------------------------------
# progress() and observe()
# ---------------------------------------------------------------------------

class TestNavigationAgentProgress(unittest.TestCase):

    def _agent_with_subtask(self):
        agent = NavigationAgent()
        agent.activate(_make_subtask(), Blackboard(), _make_wq())
        return agent

    def test_progress_zero_with_no_waypoints(self):
        agent = self._agent_with_subtask()
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq()
        self.assertAlmostEqual(agent.progress(subtask, bb, wq), 0.0)

    def test_progress_one_when_all_complete(self):
        agent = self._agent_with_subtask()
        agent._waypoints_total = 2
        agent._waypoints_completed = 2
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq()
        self.assertAlmostEqual(agent.progress(subtask, bb, wq), 1.0)

    def test_progress_fraction(self):
        agent = self._agent_with_subtask()
        agent._waypoints_total = 4
        agent._waypoints_completed = 1
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq()
        self.assertAlmostEqual(agent.progress(subtask, bb, wq), 0.25)


class TestNavigationAgentObserve(unittest.TestCase):

    def test_observe_includes_player_position(self):
        agent = NavigationAgent()
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq(player_pos=Position(12.0, -3.0))
        agent.activate(subtask, bb, wq)

        obs = agent.observe(subtask, bb, wq)
        self.assertIn("player_position", obs)
        self.assertAlmostEqual(obs["player_position"]["x"], 12.0)
        self.assertAlmostEqual(obs["player_position"]["y"], -3.0)

    def test_observe_includes_waypoint_count(self):
        agent = NavigationAgent()
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq()
        agent.activate(subtask, bb, wq)
        agent._waypoints_completed = 3
        agent._waypoints_total = 5

        obs = agent.observe(subtask, bb, wq)
        self.assertEqual(obs["waypoints_completed"], 3)
        self.assertEqual(obs["waypoints_total"], 5)

    def test_observe_includes_subtask_id(self):
        agent = NavigationAgent()
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq()
        agent.activate(subtask, bb, wq)

        obs = agent.observe(subtask, bb, wq)
        self.assertIn("subtask_id", obs)
        self.assertEqual(obs["subtask_id"], subtask.id[:8])


if __name__ == "__main__":
    unittest.main()