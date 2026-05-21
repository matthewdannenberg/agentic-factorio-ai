"""
tests_in_game/conftest.py

Shared pytest fixtures for in-game tests.

All fixtures in this file are available to every test in tests_in_game/
without explicit import — pytest discovers conftest.py automatically.

Session structure
-----------------
One RCON connection is opened for the whole test session and closed at the
end. The KnowledgeBase is shared across all tests (it accumulates entity and
recipe knowledge as tests run, matching production behaviour). Everything else
— Blackboard, SubtaskLedger, SelfModel, BehavioralMemory — is freshly
constructed for each goal run so tests are isolated from each other.

Skipping without a game
-----------------------
If the RCON connection cannot be established, the entire session is skipped
with a clear message. Individual test files do not need their own skip guards.

Usage in a test file
--------------------
    def test_something(run_goal, wq_snapshot):
        entry = GoalQueueEntry(
            description="Collect 10 iron ore",
            success_condition="inventory('iron-ore') >= 10",
            failure_condition="tick > 3600",
            goal_type="collection",
        )
        stats, wq = run_goal(entry)
        assert stats.goals_completed == 1
        assert wq.inventory_count("iron-ore") >= 10

    def test_multi_goal(run_goals):
        # Two goals sharing KB state — goals run sequentially, state resets
        # between them except for the KnowledgeBase.
        stats, wq = run_goals([entry_a, entry_b])
        assert stats.goals_completed == 2
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tempfile
import atexit

import pytest

import config
from agent.blackboard import Blackboard
from agent.loop import FactorioLoop, LoopConfig
from agent.memory.behavioral import SQLiteBehavioralMemory
from agent.network.agents.mining import MiningAgent
from agent.network.agents.navigation import NavigationAgent
from agent.network.coordinator import (
    GOAL_TYPE_COLLECTION,
    GOAL_TYPE_EXPLORATION,
    RuleBasedCoordinator,
)
from agent.network.registry import AgentRegistry
from agent.self_model import SelfModel
from agent.subtask import SubtaskLedger
from bridge.action_executor import ActionExecutor
from bridge.rcon_client import RconClient
from bridge.state_parser import StateParser
from llm.goal_source import GoalQueue, GoalQueueEntry
from planning.reward_evaluator import RewardEvaluator
from world.knowledge import KnowledgeBase
from world.state import WorldState
from world.query import WorldQuery
from world.writer import WorldWriter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RCON availability — skip the whole session if Factorio isn't running
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "in_game: marks tests that require a live Factorio instance",
    )


@pytest.fixture(scope="session")
def rcon_client():
    """
    Session-scoped RCON client. Skips the session if Factorio is unreachable.
    Closed automatically when the session ends.
    """
    try:
        client = RconClient(
            host=config.RCON_HOST,
            port=config.RCON_PORT,
            password=config.RCON_PASSWORD,
            timeout_s=config.RCON_TIMEOUT_S,
        )
        # Smoke-test the connection.
        client.send("/c rcon.print('ping')")
    except Exception as exc:
        pytest.skip(
            f"Factorio RCON not available ({exc}). "
            "Start Factorio with --rcon-port 25575 --rcon-password factorio "
            "and load a map before running in-game tests."
        )

    yield client
    client.close()


# ---------------------------------------------------------------------------
# Shared KB — persists across the session, accumulates as tests run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def knowledge_base():
    """
    Session-scoped KnowledgeBase. Uses a temporary directory so no permanent
    file is left on disk after the session. The directory and its contents are
    cleaned up when the session ends.

    KnowledgeBase takes data_dir (a Path) rather than a db_path directly —
    it constructs the SQLite path internally as data_dir / "knowledge.db".
    """
    tmp = tempfile.mkdtemp(prefix="factorio_agent_test_kb_")

    def _cleanup():
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    atexit.register(_cleanup)
    return KnowledgeBase(data_dir=tmp)


# ---------------------------------------------------------------------------
# run_goal / run_goals — the core test execution fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def run_goal(rcon_client, knowledge_base):
    """
    Function fixture: run a single GoalQueueEntry through the full agent loop.

    Usage:
        stats, wq = run_goal(entry)

    Returns (LoopStats, WorldQuery) where WorldQuery reflects the world state
    at the end of the run. Each call gets a fresh Blackboard, SubtaskLedger,
    SelfModel, and BehavioralMemory — only the KnowledgeBase persists.

    The LoopConfig used here sets shutdown_on_empty_queue=True so the loop
    exits as soon as the single goal resolves (success or failure).
    """
    def _run(entry: GoalQueueEntry):
        return _execute_goals([entry], rcon_client, knowledge_base)
    return _run


@pytest.fixture
def run_goals(rcon_client, knowledge_base):
    """
    Function fixture: run a list of GoalQueueEntry objects sequentially.

    Goals share the KnowledgeBase but each goal gets a fresh coordinator,
    blackboard, and self-model — exactly as in production. Use this when
    you want to test multi-goal sequences that accumulate world knowledge.

    Usage:
        stats, wq = run_goals([entry_a, entry_b])
        assert stats.goals_completed == 2
    """
    def _run(entries: list[GoalQueueEntry]):
        return _execute_goals(entries, rcon_client, knowledge_base)
    return _run


# ---------------------------------------------------------------------------
# Internal: build and run the loop
# ---------------------------------------------------------------------------

def _execute_goals(
    entries: list[GoalQueueEntry],
    client: RconClient,
    kb: KnowledgeBase,
) -> tuple:
    """
    Build a complete agent stack and run the given goals to completion.

    Returns (LoopStats, WorldQuery).
    """
    # Fresh per-run components.
    nav_agent  = NavigationAgent()
    mine_agent = MiningAgent()

    registry = AgentRegistry()
    registry.register(nav_agent)
    registry.register(mine_agent)

    blackboard = Blackboard()
    ledger     = SubtaskLedger()
    sm         = SelfModel()
    mem        = SQLiteBehavioralMemory(db_path=":memory:")

    coordinator = RuleBasedCoordinator(
        registry=registry,
        blackboard=blackboard,
        ledger=ledger,
        self_model=sm,
    )

    queue     = GoalQueue(entries)
    evaluator = RewardEvaluator()

    # Wire the KB's query_fn to the live RCON client. The KB's _query() method
    # passes expressions like 'rcon.print(fa.get_recipe_prototype("x"))' directly
    # to query_fn. client.send() expects a full Factorio console command, so we
    # must prepend "/c " here.
    kb._query_fn = lambda expr: client.send(f"/c {expr}")

    # StateParser accepts resource_registry (not knowledge_base) — it handles
    # resource patch name registration. KB population (recipes, techs, entities)
    # happens separately via kb.ensure_* calls driven by WorldWriter/coordinator.
    # The KB's query_fn above allows those calls to fetch from Factorio.
    parser    = StateParser()
    executor  = ActionExecutor(client)

    cfg = LoopConfig(
        tick_interval=config.TICK_INTERVAL,
        local_scan_radius=config.LOCAL_SCAN_RADIUS,
        resource_scan_radius=config.RESOURCE_SCAN_RADIUS,
        ground_item_scan_radius=config.GROUND_ITEM_SCAN_RADIUS,
        shutdown_on_empty_queue=True,
    )

    loop = FactorioLoop(
        client=client,
        parser=parser,
        executor=executor,
        coordinator=coordinator,
        goal_source=queue,
        behavioral_mem=mem,
        self_model=sm,
        evaluator=evaluator,
        cfg=cfg,
    )

    stats = loop.run()

    # Warm the KB: ensure_* for all resource types observed in the final world
    # state. This triggers query_fn calls to Factorio for any names not yet in
    # the KB, populating recipe and tech data for the knowledge tests.
    final_wq = loop._wq
    for patch in final_wq.state.resource_map:
        try:
            kb.ensure_resource(patch.resource_type)
        except Exception:
            pass
    for entity in final_wq.state.entities:
        try:
            kb.ensure_entity(entity.name)
        except Exception:
            pass

    # Capture a final world state snapshot for assertions.
    # We reach into the loop's internal wq — this is acceptable in tests.
    final_wq = loop._wq

    mem.close()
    return stats, final_wq