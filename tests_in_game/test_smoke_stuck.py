"""
tests/integration/test_smoke_stuck.py

Smoke test 2 — STUCK path.

REQUIRES A LIVE FACTORIO INSTANCE. Submits a "production" goal that the Phase 6
coordinator cannot derive. Asserts that ExecutionStatus.STUCK is returned
immediately with a correctly populated StuckContext.

This test validates that the escalation path works end-to-end before Phase 11
wires in a real LLM at the other end.
"""

import unittest

from planning.goal import Goal, Priority, RewardSpec
from agent.execution_protocol import ExecutionStatus
from agent.network.coordinator import GOAL_TYPE_PRODUCTION, RuleBasedCoordinator
from agent.network.registry import AgentRegistry
from agent.self_model import SelfModel
from agent.blackboard import Blackboard
from agent.subtask import SubtaskLedger

_FACTORIO_AVAILABLE = False
try:
    from bridge.rcon_client import RconClient
    import config
    client = RconClient(
        host=config.RCON_HOST,
        port=config.RCON_PORT,
        password=config.RCON_PASSWORD,
        timeout_s=2.0,
    )
    client.close()
    _FACTORIO_AVAILABLE = True
except Exception:
    pass


@unittest.skipUnless(_FACTORIO_AVAILABLE, "Factorio RCON not available")
class TestSmokeStuck(unittest.TestCase):
    """
    Smoke test: production goal returns STUCK on the first coordinator tick.
    """

    def _build_coordinator(self) -> RuleBasedCoordinator:
        registry = AgentRegistry()
        # No agents registered for "production" in Phase 6.
        bb = Blackboard()
        ledger = SubtaskLedger()
        return RuleBasedCoordinator(
            registry=registry,
            blackboard=bb,
            ledger=ledger,
            self_model=SelfModel(),
        )

    def test_production_goal_returns_stuck_immediately(self):
        from bridge.rcon_client import RconClient
        from bridge.state_parser import StateParser
        from world.state import WorldState
        from world.query import WorldQuery
        from world.writer import WorldWriter
        import config

        spec = RewardSpec()
        goal = Goal(
            description="Produce 10 iron gear wheels",
            priority=Priority.NORMAL,
            success_condition="inventory('iron-gear-wheel') >= 10",
            failure_condition="tick > 99999",
            reward_spec=spec,
        )
        goal.type = GOAL_TYPE_PRODUCTION
        goal.activate(tick=0)

        coordinator = self._build_coordinator()

        client = RconClient(
            host=config.RCON_HOST,
            port=config.RCON_PORT,
            password=config.RCON_PASSWORD,
        )
        parser = StateParser()
        live_state = WorldState()
        ww = WorldWriter(live_state)
        wq = WorldQuery(live_state)

        raw = client.send(
            f"/c rcon.print(fa.get_state({{"
            f"radius={config.LOCAL_SCAN_RADIUS}, "
            f"resource_radius={config.RESOURCE_SCAN_RADIUS}, "
            f"item_radius={config.GROUND_ITEM_SCAN_RADIUS}"
            f"}}))"
        )
        snapshot = parser.parse(raw, live_state.tick)
        ww.integrate_snapshot(snapshot)
        client.close()

        coordinator.reset(goal, wq)
        result = coordinator.tick(goal, wq, None, tick=live_state.tick)

        self.assertEqual(result.status, ExecutionStatus.STUCK)
        self.assertIsNotNone(result.stuck_context)
        self.assertTrue(result.stuck_context.stuck_at_goal_level)

        # StuckContext must serialise cleanly.
        d = result.stuck_context.to_dict()
        self.assertEqual(d["goal_id"], goal.id)
        self.assertEqual(d["failure_chain"], [])