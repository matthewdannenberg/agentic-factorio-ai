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
    def test_something(run_goal):
        entry = GoalQueueEntry(
            description="Collect 10 iron ore",
            success_condition="inventory('iron-ore') >= 10",
            failure_condition="tick > 3600",
            goal_type="collection",
        )
        # Tuple unpacking (backwards-compatible with all existing tests):
        stats, wq = run_goal(entry)
        assert stats.goals_completed == 1
        assert wq.inventory_count("iron-ore") >= 10

        # Or access the richer RunResult directly:
        result = run_goal(entry)
        print(result.final_tick, result.kb_summary)

    def test_multi_goal(run_goals):
        # Two goals sharing KB state — goals run sequentially, state resets
        # between them except for the KnowledgeBase.
        stats, wq = run_goals([entry_a, entry_b])
        assert stats.goals_completed == 2
"""

from __future__ import annotations

import logging
import tempfile
import atexit
from dataclasses import dataclass, field
from typing import Iterator

import pytest

import config
from agent.blackboard import Blackboard
from agent.loop import FactorioLoop, LoopConfig, LoopStats
from agent.memory.behavioral import SQLiteBehavioralMemory
from agent.network.agents.mining import MiningAgent
from agent.network.agents.navigation import NavigationAgent
from agent.network.coordinator import RuleBasedCoordinator
from agent.network.registry import AgentRegistry
from agent.self_model import SelfModel
from agent.subtask import SubtaskLedger
from bridge.action_executor import ActionExecutor
from bridge.prototype_query import make_prototype_query_fn
from bridge.rcon_client import RconClient
from bridge.state_parser import StateParser
from llm.goal_source import GoalQueue, GoalQueueEntry
from planning.reward_evaluator import RewardEvaluator
from world.knowledge import KnowledgeBase
from world.query import WorldQuery

log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def configure_logging():
    """Ensure DEBUG logging is active for all in-game tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


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
# RCON diagnostics — enables bugfixing via print statements in Factorio
# ---------------------------------------------------------------------------

@pytest.fixture
def rcon(rcon_client):
    """Convenience fixture for raw RCON commands during debugging."""
    def _send(cmd: str) -> str:
        result = rcon_client.send(cmd)
        print(f"\nRCON >> {cmd}\nRCON << {result}")
        return result
    return _send


# ---------------------------------------------------------------------------
# RunResult — richer return type from _execute_goals
#
# Supports tuple unpacking as `stats, wq = run_goal(entry)` so all existing
# tests continue to work without modification. Richer fields (final_tick,
# kb_summary, goals_attempted) are available as named attributes when needed.
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    stats: LoopStats
    wq: WorldQuery
    goals_attempted: list[str]  # entry.description for each goal in the run
    final_tick: int             # wq.tick at end of run
    kb_summary: dict            # kb.summary() at end of run

    def __iter__(self) -> Iterator:
        """Yield (stats, wq) so tests can unpack as: stats, wq = run_goal(entry)."""
        yield self.stats
        yield self.wq


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
# run_goal / run_goals — the core test execution fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def run_goal(rcon_client, knowledge_base):
    """
    Function fixture: run a single GoalQueueEntry through the full agent loop.

    Usage (tuple unpacking — compatible with all existing tests):
        stats, wq = run_goal(entry)

    Or access the full result:
        result = run_goal(entry)
        print(result.final_tick, result.kb_summary)

    Each call gets a fresh Blackboard, SubtaskLedger, SelfModel, and
    BehavioralMemory — only the KnowledgeBase persists across calls.

    The LoopConfig sets shutdown_on_empty_queue=True so the loop exits as
    soon as the single goal resolves (success or failure).
    """
    def _run(entry: GoalQueueEntry) -> RunResult:
        return _execute_goals([entry], rcon_client, knowledge_base)
    return _run


@pytest.fixture
def run_goals(rcon_client, knowledge_base):
    """
    Function fixture: run a list of GoalQueueEntry objects sequentially.

    Goals share the KnowledgeBase but each goal gets a fresh coordinator,
    blackboard, and self-model — exactly as in production.

    Usage (tuple unpacking — compatible with all existing tests):
        stats, wq = run_goals([entry_a, entry_b])
        assert stats.goals_completed == 2
    """
    def _run(entries: list[GoalQueueEntry]) -> RunResult:
        return _execute_goals(entries, rcon_client, knowledge_base)
    return _run


# ---------------------------------------------------------------------------
# Internal: build and run the loop
# ---------------------------------------------------------------------------

def _execute_goals(
    entries: list[GoalQueueEntry],
    client: RconClient,
    kb: KnowledgeBase,
) -> RunResult:
    """
    Build a complete agent stack and run the given goals to completion.

    Returns a RunResult. Supports tuple unpacking as (stats, wq) for
    backwards compatibility with existing tests.
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

    # Wire the KB to Factorio via the bridge's prototype query factory.
    # make_prototype_query_fn owns all Lua construction and mod namespacing —
    # knowledge.py only ever calls query_fn("recipe", "iron-gear-wheel") etc.
    kb._query_fn = make_prototype_query_fn(client)

    # StateParser accepts resource_registry (not knowledge_base) — it handles
    # resource patch name registration. KB population (recipes, techs, entities)
    # happens separately via kb.ensure_* calls driven by WorldWriter/coordinator.
    # The KB's query_fn above allows those calls to fetch from Factorio.
    parser   = StateParser()
    executor = ActionExecutor(client)

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

    # Capture final world state. We reach into the loop's internal _wq —
    # acceptable in tests.
    final_wq = loop._wq

    # Warm the KB: ensure_* for all resource types and entities observed in
    # the final world state. Triggers query_fn calls to Factorio for any names
    # not yet in the KB, populating recipe and tech data for the knowledge tests.
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

    mem.close()

    return RunResult(
        stats=stats,
        wq=final_wq,
        goals_attempted=[e.description for e in entries],
        final_tick=final_wq.tick,
        kb_summary=kb.summary(),
    )