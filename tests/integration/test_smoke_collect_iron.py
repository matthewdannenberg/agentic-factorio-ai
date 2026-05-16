"""
tests/integration/test_smoke_collect_iron.py

Smoke test 1 — "Collect 50 iron ore" (happy path).

REQUIRES A LIVE FACTORIO INSTANCE. This test will be skipped if the RCON
connection cannot be established. Run only when Factorio is running with:

    factorio --start-server-load-latest \
             --rcon-port 25575 \
             --rcon-password factorio

This test validates the entire pipeline:
    bridge → WorldState → WorldQuery → coordinator → SubtaskLedger →
    navigation agent → action executor → RCON → Factorio →
    StateParser → WorldState → RewardEvaluator → goal resolution.
"""

import unittest

from planning.goal import Goal, GoalStatus, Priority, RewardSpec
from agent.network.coordinator import GOAL_TYPE_COLLECTION, RuleBasedCoordinator
from agent.network.agents.navigation import NavigationAgent
from agent.network.registry import AgentRegistry
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
        timeout=2.0,
    )
    client.close()
    _FACTORIO_AVAILABLE = True
except Exception:
    pass


@unittest.skipUnless(_FACTORIO_AVAILABLE, "Factorio RCON not available")
class TestSmokeCollectIron(unittest.TestCase):
    """
    End-to-end smoke test: collect 50 iron ore.

    Starting conditions assumed:
      - Fresh game or save with iron ore patches within scan radius.
      - Player inventory does not yet contain 50 iron-ore.

    The test runs the main tick loop for up to MAX_TICKS and asserts that the
    goal's success condition is met before timeout.
    """

    MAX_TICKS = 36_000   # 10 minutes at 60 tps

    def _build_system(self):
        """Wire up the coordinator, navigation agent, and registry."""
        registry = AgentRegistry()
        nav = NavigationAgent()
        registry.register(nav, [GOAL_TYPE_COLLECTION])

        bb = Blackboard()
        ledger = SubtaskLedger()
        coordinator = RuleBasedCoordinator(
            registry=registry,
            blackboard=bb,
            ledger=ledger,
        )
        return coordinator, nav

    def test_collect_50_iron_ore(self):
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
            description="Collect 50 iron ore",
            priority=Priority.NORMAL,
            success_condition="inventory('iron-ore') >= 50",
            failure_condition="tick > 36000",
            reward_spec=spec,
        )
        goal.type = GOAL_TYPE_COLLECTION
        goal.activate(tick=0)

        coordinator, nav = self._build_system()

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

        try:
            for poll in range(self.MAX_TICKS // config.TICK_INTERVAL):
                raw = client.send("/c rcon.print(fa.get_state())")
                snapshot = parser.parse(raw)
                ww.integrate_snapshot(snapshot)
                tick = live_state.tick

                # Check success.
                result_flags = evaluator.evaluate_conditions(goal, wq, tick)
                if result_flags.get("success"):
                    goal.complete(tick)
                    break

                # Tick coordinator.
                result = coordinator.tick(goal, wq, ww, tick)
                for action in result.actions:
                    executor.execute(action)

                time.sleep(config.TICK_INTERVAL / 60.0)

        finally:
            client.close()

        self.assertEqual(
            goal.status,
            GoalStatus.COMPLETE,
            "Goal did not complete within time limit",
        )
        self.assertGreaterEqual(
            wq.inventory_count("iron-ore"),
            50,
            "Player inventory does not contain 50 iron-ore after goal completion",
        )
