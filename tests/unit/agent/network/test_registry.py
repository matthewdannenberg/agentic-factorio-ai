"""
tests/unit/agent/test_registry.py

Tests for agent/network/registry.py (updated for new API: register(agent) only,
no goal-type arguments).

Run with:  pytest tests/unit/agent/test_registry.py -v
"""

from __future__ import annotations

import unittest

from agent.network.agent_protocol import AgentProtocol
from agent.network.registry import AgentRegistry


class _StubAgent(AgentProtocol):
    """Minimal agent satisfying AgentProtocol for registry tests."""

    AGENT_ID = "stub"

    def __init__(self, name: str) -> None:
        self.name = name

    def activate(self, subtask, blackboard, wq): pass
    def tick(self, subtask, blackboard, wq, ww, tick): return []
    def observe(self, subtask, blackboard, wq): return {}
    def progress(self, subtask, blackboard, wq): return 0.0

    def __repr__(self):
        return f"StubAgent({self.name!r})"


class _NavAgent(_StubAgent):
    AGENT_ID = "navigation"
    def __init__(self): super().__init__("nav")

class _MineAgent(_StubAgent):
    AGENT_ID = "mining"
    def __init__(self): super().__init__("mining")

class _ProdAgent(_StubAgent):
    AGENT_ID = "production"
    def __init__(self): super().__init__("prod")


class TestAgentRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = AgentRegistry()
        self.nav   = _NavAgent()
        self.mine  = _MineAgent()
        self.prod  = _ProdAgent()

    # --- Basic registration ---

    def test_register_single_agent(self):
        self.registry.register(self.nav)
        self.assertEqual(len(self.registry), 1)
        self.assertIn(self.nav, self.registry.all_agents())

    def test_register_multiple_agents(self):
        self.registry.register(self.nav)
        self.registry.register(self.mine)
        self.registry.register(self.prod)
        self.assertEqual(len(self.registry), 3)

    def test_duplicate_registration_ignored(self):
        self.registry.register(self.nav)
        self.registry.register(self.nav)
        self.assertEqual(len(self.registry), 1)

    def test_all_agents_returns_copy(self):
        self.registry.register(self.nav)
        agents = self.registry.all_agents()
        agents.append(self.mine)
        self.assertEqual(len(self.registry.all_agents()), 1)

    def test_registration_order_preserved(self):
        self.registry.register(self.nav)
        self.registry.register(self.mine)
        self.registry.register(self.prod)
        agents = self.registry.all_agents()
        self.assertIs(agents[0], self.nav)
        self.assertIs(agents[1], self.mine)
        self.assertIs(agents[2], self.prod)

    # --- agent_by_id ---

    def test_agent_by_id_finds_correct_agent(self):
        self.registry.register(self.nav)
        self.registry.register(self.mine)
        result = self.registry.agent_by_id("mining")
        self.assertIs(result, self.mine)

    def test_agent_by_id_returns_none_for_unknown_id(self):
        self.registry.register(self.nav)
        self.assertIsNone(self.registry.agent_by_id("nonexistent"))

    def test_agent_by_id_returns_none_for_empty_registry(self):
        self.assertIsNone(self.registry.agent_by_id("navigation"))

    def test_agent_by_id_returns_first_match(self):
        # Two agents with the same AGENT_ID — first registered wins.
        class _AltNav(_StubAgent):
            AGENT_ID = "navigation"
            def __init__(self): super().__init__("alt_nav")

        alt = _AltNav()
        self.registry.register(self.nav)
        self.registry.register(alt)
        self.assertIs(self.registry.agent_by_id("navigation"), self.nav)

    # --- repr ---

    def test_repr(self):
        self.registry.register(self.nav)
        r = repr(self.registry)
        self.assertIn("AgentRegistry", r)
        self.assertIn("1", r)


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
        stub.reset(goal=goal, wq=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)