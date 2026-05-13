"""
tests/unit/agent/test_registry.py

Tests for agent/network/registry.py

Run with:  python -m pytest tests/unit/agent/test_registry.py -v
       or:  python -m unittest tests.unit.agent.test_registry
"""

from __future__ import annotations

import unittest

from bridge.actions import Action
from agent.network.agent_protocol import AgentProtocol
from agent.network.registry import AgentRegistry


class _StubAgent(AgentProtocol):
    """Minimal agent satisfying AgentProtocol for registry tests."""

    def __init__(self, name: str) -> None:
        self.name = name

    def activate(self, goal, blackboard, wq): pass
    def tick(self, blackboard, wq, ww, tick): return []
    def observe(self, blackboard, wq): return {}
    def progress(self, blackboard, wq): return 0.0

    def __repr__(self):
        return f"StubAgent({self.name!r})"


class TestAgentRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = AgentRegistry()
        self.nav = _StubAgent("nav")
        self.prod = _StubAgent("prod")
        self.spatial = _StubAgent("spatial")

    def test_register_single_goal_type(self):
        self.registry.register(self.nav, ["exploration"])
        agents = self.registry.agents_for_goal("exploration")
        self.assertEqual(len(agents), 1)
        self.assertIs(agents[0], self.nav)

    def test_register_multiple_goal_types(self):
        self.registry.register(self.nav, ["exploration", "construction"])
        self.assertIn(self.nav, self.registry.agents_for_goal("exploration"))
        self.assertIn(self.nav, self.registry.agents_for_goal("construction"))

    def test_unknown_goal_type_returns_empty(self):
        self.registry.register(self.nav, ["exploration"])
        self.assertEqual(self.registry.agents_for_goal("production"), [])

    def test_multiple_agents_same_goal_type(self):
        self.registry.register(self.nav, ["construction"])
        self.registry.register(self.spatial, ["construction"])
        agents = self.registry.agents_for_goal("construction")
        self.assertEqual(len(agents), 2)
        ids = [id(a) for a in agents]
        self.assertIn(id(self.nav), ids)
        self.assertIn(id(self.spatial), ids)

    def test_all_agents_includes_all_registered(self):
        self.registry.register(self.nav, ["exploration"])
        self.registry.register(self.prod, ["production"])
        self.registry.register(self.spatial, ["construction"])
        all_a = self.registry.all_agents()
        self.assertEqual(len(all_a), 3)

    def test_all_agents_no_duplicates_when_registered_multiple_types(self):
        self.registry.register(self.nav, ["exploration", "construction"])
        all_a = self.registry.all_agents()
        self.assertEqual(len(all_a), 1)

    def test_same_agent_registered_twice_not_duplicated_in_bucket(self):
        self.registry.register(self.nav, ["exploration"])
        self.registry.register(self.nav, ["exploration"])
        agents = self.registry.agents_for_goal("exploration")
        self.assertEqual(len(agents), 1)

    def test_register_empty_goal_types_raises(self):
        with self.assertRaises(ValueError):
            self.registry.register(self.nav, [])

    def test_agents_for_goal_returns_copy(self):
        self.registry.register(self.nav, ["exploration"])
        agents = self.registry.agents_for_goal("exploration")
        agents.append(self.prod)
        self.assertEqual(len(self.registry.agents_for_goal("exploration")), 1)

    def test_all_agents_returns_copy(self):
        self.registry.register(self.nav, ["exploration"])
        all_a = self.registry.all_agents()
        all_a.append(self.prod)
        self.assertEqual(len(self.registry.all_agents()), 1)

    def test_registration_order_preserved_in_bucket(self):
        self.registry.register(self.nav, ["construction"])
        self.registry.register(self.spatial, ["construction"])
        self.registry.register(self.prod, ["construction"])
        agents = self.registry.agents_for_goal("construction")
        self.assertIs(agents[0], self.nav)
        self.assertIs(agents[1], self.spatial)
        self.assertIs(agents[2], self.prod)

    def test_registered_goal_types(self):
        self.registry.register(self.nav, ["exploration"])
        self.registry.register(self.prod, ["production", "research"])
        types = set(self.registry.registered_goal_types())
        self.assertSetEqual(types, {"exploration", "production", "research"})

    def test_repr(self):
        self.registry.register(self.nav, ["exploration"])
        r = repr(self.registry)
        self.assertIn("AgentRegistry", r)


class TestStubCoordinator(unittest.TestCase):
    """Verify the stub coordinator from coordinator.py satisfies its protocol."""

    def test_stub_returns_waiting(self):
        from agent.network.coordinator import StubCoordinator
        from agent.execution_protocol import ExecutionStatus
        from planning.goal import make_goal

        stub = StubCoordinator()
        goal = make_goal("do nothing", "True", "False")
        result = stub.tick(goal=goal, wq=None, ww=None, tick=0)
        self.assertEqual(result.status, ExecutionStatus.WAITING)
        self.assertEqual(result.actions, [])
        self.assertIsNone(result.stuck_context)

    def test_stub_reset_does_not_raise(self):
        from agent.network.coordinator import StubCoordinator
        from planning.goal import make_goal

        stub = StubCoordinator()
        goal = make_goal("do nothing", "True", "False")
        stub.reset(goal=goal, wq=None)   # should not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
