"""
tests/integration/test_smoke_exploration.py

Smoke test 3 — Exploration goal.

REQUIRES A LIVE FACTORIO INSTANCE. Submits an exploration goal and asserts that
charted_chunks >= 10 is achieved within the time limit.
"""

import unittest

from planning.goal import Goal, GoalStatus, Priority, RewardSpec
from agent.network.coordinator import GOAL_TYPE_EXPLORATION, RuleBasedCoordinator
from agent.network.agents.navigation import NavigationAgent
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
class TestSmokeExploration(unittest.TestCase):
    """
    Smoke test: exploration goal completes with charted_chunks >= 10.
    """

    MAX_TICKS = 54_000   # 15 minutes

    def test_exploration_goal_completes(self):
        from bridge.rcon_client import RconClient
        from bridge.state_parser import StateParser
        from bridge.action_executor import ActionExecutor
        from world.state import WorldState
        from world.query import WorldQuery
        from world.writer import WorldWriter
        from planning.reward_evaluator import RewardEvaluator
        import config
        import time

        spec = RewardSpec(success_reward=1.0)
        goal = Goal(
            description="Chart at least 10 chunks",
            priority=Priority.NORMAL,
            success_condition="charted_chunks >= 10",
            failure_condition=f"tick > {self.MAX_TICKS}",
            reward_spec=spec,
        )
        goal.type = GOAL_TYPE_EXPLORATION
        goal.activate(tick=0)

        registry = AgentRegistry()
        nav = NavigationAgent()
        registry.register(nav, [GOAL_TYPE_EXPLORATION])

        bb = Blackboard()
        ledger = SubtaskLedger()
        coordinator = RuleBasedCoordinator(
            registry=registry,
            blackboard=bb,
            ledger=ledger,
            self_model=SelfModel(),
        )

        client = RconClient(
            host=config.RCON_HOST,
            port=config.RCON_PORT,
            password=config.RCON_PASSWORD,
        )
        parser = StateParser()
        executor = ActionExecutor(client)
        live_state = WorldState()
        ww = WorldWriter(live_state)
        wq = WorldQuery(live_state)
        evaluator = RewardEvaluator()

        coordinator.reset(goal, wq)
        start_tick: int = 0

        try:
            for _ in range(self.MAX_TICKS // config.TICK_INTERVAL):
                raw = client.send(
                    f"/c rcon.print(fa.get_state({{"
                    f"radius={config.LOCAL_SCAN_RADIUS}, "
                    f"resource_radius={config.RESOURCE_SCAN_RADIUS}, "
                    f"item_radius={config.GROUND_ITEM_SCAN_RADIUS}"
                    f"}}))"
                )
                snapshot = parser.parse(raw, live_state.tick)
                ww.integrate_snapshot(snapshot)
                tick = live_state.tick
                if start_tick == 0:
                    start_tick = tick

                result_flags = evaluator.evaluate(goal, wq, tick, start_tick)
                if result_flags.success:
                    goal.complete(tick)
                    break

                result = coordinator.tick(goal, wq, ww, tick)
                for action in result.actions:
                    executor.execute(action)

                time.sleep(config.TICK_INTERVAL / 60.0)

        finally:
            client.close()

        self.assertEqual(goal.status, GoalStatus.COMPLETE)
        self.assertGreaterEqual(wq.charted_chunks, 10)