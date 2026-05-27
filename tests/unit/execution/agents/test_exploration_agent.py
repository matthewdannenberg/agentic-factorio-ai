"""
tests/unit/execution/test_exploration_agent.py

Unit tests for execution/agents/exploration.py (ExplorationAgent).

Sections
--------
1. Stubs and helpers
2. activate() — task attributes, initial phase, blackboard observations
3. APPROACH phase — skill delegation, transition to SCAN
4. SCAN phase — target selection, chunk tracking, needs-frontier signal
5. Phase transitions — APPROACH→SCAN on succeed/stuck/failed
6. observe() and progress()
7. pending_patches()

Run with:  python -m pytest tests/unit/execution/test_exploration_agent.py -v
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

from execution.agents.exploration import (
    ExplorationAgent,
    _Phase,
    _chunk_centre,
    _nearest_chunk,
    _CHUNK_SIZE,
)
from execution.blackboard import Blackboard, EntryCategory, EntryScope
from execution.skills.base import SkillStatus
from world import Position
from world.observable.state import ChunkCoord
from bridge import MoveTo, StopMovement


# ===========================================================================
# Section 1 — Stubs
# ===========================================================================

@dataclass
class _Task:
    id: str = "task-explore-01"
    description: str = "explore region"
    task_type: str = "explore_region"
    frontier_position: Optional[Position] = None
    home_position: Optional[Position] = None


class _WQ:
    """WorldQuery stub with mutable nearby_uncharted_chunks and position."""

    def __init__(
        self,
        position: Position = None,
        reachable: list[int] = None,
        charted: int = 10,
        nearby_uncharted: list = None,
        tick: int = 1,
    ):
        self._position          = position or Position(x=16.0, y=16.0)
        self._reachable         = reachable or []
        self._charted_chunks    = charted
        self._nearby_uncharted  = list(nearby_uncharted or [])
        self.tick               = tick
        self.state              = MagicMock()
        self.state.player.reachable = self._reachable
        self.state.player.inventory.slots = []
        self.state.player.crafting_queue_size = 0

    def player_position(self) -> Position:
        return self._position

    def entity_by_id(self, eid):
        return None

    def inventory_count(self, item: str) -> int:
        return 0

    @property
    def charted_chunks(self) -> int:
        return self._charted_chunks

    @property
    def nearby_uncharted_chunks(self) -> list:
        return self._nearby_uncharted

    @property
    def crafting_queue_size(self) -> int:
        return 0

    def move_to(self, x: float, y: float) -> None:
        self._position = Position(x=x, y=y)

    def set_nearby_uncharted(self, chunks: list) -> None:
        self._nearby_uncharted = list(chunks)

    def set_charted(self, n: int) -> None:
        self._charted_chunks = n


_WW = MagicMock()
_KB = MagicMock()


def _make_agent() -> ExplorationAgent:
    return ExplorationAgent()


def _make_bb() -> Blackboard:
    return Blackboard()


def _obs_of_type(bb: Blackboard, obs_type: str) -> list:
    return [
        e for e in bb.read(category=EntryCategory.OBSERVATION)
        if e.data.get("type") == obs_type
    ]


def _chunk(cx: int, cy: int) -> ChunkCoord:
    return ChunkCoord(cx=cx, cy=cy)


# ===========================================================================
# Section 2 — activate()
# ===========================================================================

class TestExplorationAgentActivate(unittest.TestCase):

    def test_initial_phase_is_approach_when_frontier_given(self):
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=200, y=200))
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._phase, _Phase.APPROACH)

    def test_initial_phase_is_scan_when_no_frontier(self):
        agent = _make_agent()
        task = _Task()   # no frontier_position
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._phase, _Phase.SCAN)

    def test_skill_running_after_frontier_activate(self):
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=200, y=200))
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._skill.status(), SkillStatus.RUNNING)

    def test_skill_idle_when_no_frontier_and_no_uncharted(self):
        # No frontier, no nearby uncharted — skill stays IDLE until first tick
        agent = _make_agent()
        task = _Task()
        wq = _WQ(nearby_uncharted=[])
        agent.activate(task, _make_bb(), wq, _KB)
        self.assertEqual(agent._skill.status(), SkillStatus.IDLE)

    def test_activate_writes_exploration_started(self):
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=100, y=100))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        self.assertEqual(len(_obs_of_type(bb, "exploration_started")), 1)

    def test_exploration_started_contains_charted_count(self):
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=100, y=100))
        bb = _make_bb()
        wq = _WQ(charted=42)
        agent.activate(task, bb, wq, _KB)
        obs = _obs_of_type(bb, "exploration_started")[0]
        self.assertEqual(obs.data["charted_chunks_at_start"], 42)

    def test_exploration_started_contains_frontier(self):
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=128, y=96))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        obs = _obs_of_type(bb, "exploration_started")[0]
        self.assertAlmostEqual(obs.data["frontier_position"]["x"], 128.0)
        self.assertAlmostEqual(obs.data["frontier_position"]["y"], 96.0)

    def test_exploration_started_frontier_none_when_absent(self):
        agent = _make_agent()
        task = _Task()
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        obs = _obs_of_type(bb, "exploration_started")[0]
        self.assertIsNone(obs.data["frontier_position"])

    def test_home_defaults_to_player_position(self):
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=100, y=100))
        wq = _WQ(position=Position(x=16, y=16))
        agent.activate(task, _make_bb(), wq, _KB)
        self.assertAlmostEqual(agent._home_position.x, 16.0)
        self.assertAlmostEqual(agent._home_position.y, 16.0)

    def test_home_uses_task_attribute_when_set(self):
        agent = _make_agent()
        task = _Task(
            frontier_position=Position(x=100, y=100),
            home_position=Position(x=5, y=5),
        )
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertAlmostEqual(agent._home_position.x, 5.0)

    def test_reactivate_clears_state(self):
        agent = _make_agent()
        task1 = _Task(frontier_position=Position(x=100, y=100))
        task2 = _Task(frontier_position=Position(x=300, y=300))
        bb = _make_bb()
        wq = _WQ(nearby_uncharted=[_chunk(5, 5)])
        agent.activate(task1, bb, wq, _KB)
        # Drain skill to SUCCEEDED
        wq.move_to(100.0 + 0.5, 100.0)   # arrive at frontier
        agent.tick(task1, bb, wq, _WW, 1, _KB)
        # Reactivate
        agent.activate(task2, bb, _WQ(), _KB)
        self.assertFalse(agent._needs_new_frontier)
        self.assertEqual(agent._chunks_approached, 0)
        self.assertFalse(agent._outcome_written)


# ===========================================================================
# Section 3 — APPROACH phase
# ===========================================================================

class TestExplorationAgentApproach(unittest.TestCase):

    def _approach_task(self, fx=200.0, fy=200.0) -> _Task:
        return _Task(frontier_position=Position(x=fx, y=fy))

    def test_approach_issues_move_to_on_first_tick(self):
        agent = _make_agent()
        task = self._approach_task()
        wq = _WQ(position=Position(x=0, y=0))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        actions = agent.tick(task, bb, wq, _WW, 1, _KB)
        self.assertTrue(any(isinstance(a, MoveTo) for a in actions))

    def test_approach_writes_position_observation(self):
        agent = _make_agent()
        task = self._approach_task()
        wq = _WQ()
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        self.assertTrue(len(_obs_of_type(bb, "player_position")) >= 1)

    def test_approach_transitions_to_scan_on_skill_succeed(self):
        agent = _make_agent()
        # Place player at the frontier — skill succeeds on first tick
        task = _Task(frontier_position=Position(x=16.0, y=16.0))
        wq = _WQ(position=Position(x=16.0, y=16.0),
                 nearby_uncharted=[_chunk(1, 0)])
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        # First tick: skill.tick() fires → SUCCEEDED
        agent.tick(task, bb, wq, _WW, 1, _KB)
        # Second tick: _tick_approach sees SUCCEEDED → transitions to SCAN
        agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertEqual(agent._phase, _Phase.SCAN)

    def test_approach_transitions_to_scan_on_skill_stuck(self):
        from execution.skills.navigate import _UNREACHABLE_GRACE_TICKS
        agent = _make_agent()
        task = self._approach_task(fx=500, fy=500)
        wq = _WQ(position=Position(x=0, y=0),
                 nearby_uncharted=[_chunk(3, 3)])
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        # Run until skill becomes STUCK
        for i in range(1, _UNREACHABLE_GRACE_TICKS // 10 + 5):
            agent.tick(task, bb, wq, _WW, i * 10, _KB)
            if agent._phase == _Phase.SCAN:
                break
        self.assertEqual(agent._phase, _Phase.SCAN)

    def test_approach_phase_stays_approach_while_running(self):
        agent = _make_agent()
        task = self._approach_task(fx=500, fy=500)
        wq = _WQ(position=Position(x=0, y=0))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        # Two ticks — not enough to stall
        agent.tick(task, bb, wq, _WW, 1, _KB)
        agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertEqual(agent._phase, _Phase.APPROACH)


# ===========================================================================
# Section 4 — SCAN phase
# ===========================================================================

class TestExplorationAgentScan(unittest.TestCase):

    def _enter_scan(self, agent, task, bb, wq):
        """Helper: activate with no frontier to enter SCAN immediately."""
        no_frontier_task = _Task()
        no_frontier_task.id = task.id
        agent.activate(no_frontier_task, bb, wq, _KB)

    def test_scan_issues_move_to_toward_uncharted_chunk(self):
        agent = _make_agent()
        task = _Task()
        wq = _WQ(
            position=Position(x=0, y=0),
            nearby_uncharted=[_chunk(2, 0)],  # to the east
        )
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        actions = agent.tick(task, bb, wq, _WW, 1, _KB)
        self.assertTrue(any(isinstance(a, MoveTo) for a in actions))

    def test_scan_targets_nearest_chunk(self):
        """Agent should move toward the closer of two uncharted chunks."""
        agent = _make_agent()
        task = _Task()
        wq = _WQ(
            position=Position(x=0, y=0),
            nearby_uncharted=[
                _chunk(10, 0),  # far east
                _chunk(1, 0),   # near east
            ],
        )
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        actions = agent.tick(task, bb, wq, _WW, 1, _KB)
        move_actions = [a for a in actions if isinstance(a, MoveTo)]
        self.assertEqual(len(move_actions), 1)
        # Target should be centre of chunk (1, 0) = (48, 16)
        expected = _chunk_centre(1, 0)
        self.assertAlmostEqual(move_actions[0].position.x, expected.x, places=1)
        self.assertAlmostEqual(move_actions[0].position.y, expected.y, places=1)

    def test_scan_increments_chunks_approached(self):
        agent = _make_agent()
        task = _Task()
        wq = _WQ(
            position=Position(x=0, y=0),
            nearby_uncharted=[_chunk(1, 0)],
        )
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        self.assertEqual(agent._chunks_approached, 1)

    def test_scan_does_not_reissue_for_same_target(self):
        """If the nearest chunk hasn't changed significantly, don't restart skill."""
        agent = _make_agent()
        task = _Task()
        wq = _WQ(
            position=Position(x=0, y=0),
            nearby_uncharted=[_chunk(1, 0)],
        )
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # starts skill
        chunks_before = agent._chunks_approached
        agent.tick(task, bb, wq, _WW, 2, _KB)   # same target
        self.assertEqual(agent._chunks_approached, chunks_before)

    def test_scan_retargets_when_chunk_changes(self):
        """Skill is restarted when the nearest uncharted chunk changes."""
        agent = _make_agent()
        task = _Task()
        wq = _WQ(
            position=Position(x=0, y=0),
            nearby_uncharted=[_chunk(1, 0)],
        )
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # targets chunk(1,0)
        # Move player; shift nearby uncharted to a very different chunk
        wq.move_to(100, 0)
        wq.set_nearby_uncharted([_chunk(5, 0)])
        agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertEqual(agent._chunks_approached, 2)

    def test_scan_writes_needs_frontier_when_empty(self):
        agent = _make_agent()
        task = _Task()
        wq = _WQ(position=Position(x=0, y=0), nearby_uncharted=[])
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        self.assertTrue(agent._needs_new_frontier)
        self.assertEqual(len(_obs_of_type(bb, "exploration_needs_frontier")), 1)

    def test_scan_needs_frontier_written_only_once(self):
        """The needs_frontier observation is not re-written on subsequent ticks."""
        agent = _make_agent()
        task = _Task()
        wq = _WQ(nearby_uncharted=[])
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        for i in range(1, 5):
            agent.tick(task, bb, wq, _WW, i, _KB)
        self.assertEqual(len(_obs_of_type(bb, "exploration_needs_frontier")), 1)

    def test_scan_clears_needs_frontier_when_uncharted_reappears(self):
        """If uncharted chunks appear after the signal, the flag clears."""
        agent = _make_agent()
        task = _Task()
        wq = _WQ(nearby_uncharted=[])
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # sets _needs_new_frontier
        self.assertTrue(agent._needs_new_frontier)
        # New chunks appear
        wq.set_nearby_uncharted([_chunk(1, 0)])
        agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertFalse(agent._needs_new_frontier)

    def test_scan_needs_frontier_contains_charted_count(self):
        agent = _make_agent()
        task = _Task()
        wq = _WQ(nearby_uncharted=[], charted=55)
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        obs = _obs_of_type(bb, "exploration_needs_frontier")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].data["charted_chunks"], 55)

    def test_scan_stuck_on_chunk_resets_skill(self):
        """If the skill gets stuck navigating to a chunk, it resets to IDLE."""
        from execution.skills.navigate import _UNREACHABLE_GRACE_TICKS
        agent = _make_agent()
        task = _Task()
        wq = _WQ(
            position=Position(x=0, y=0),
            nearby_uncharted=[_chunk(1, 0)],
        )
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # starts skill → RUNNING
        # Drive ticks until STUCK is detected AND reset has occurred (IDLE)
        for i in range(1, _UNREACHABLE_GRACE_TICKS // 10 + 10):
            agent.tick(task, bb, wq, _WW, 1 + i * 10, _KB)
            if agent._skill.status() == SkillStatus.IDLE:
                break
        self.assertEqual(agent._skill.status(), SkillStatus.IDLE)


# ===========================================================================
# Section 5 — Phase transitions
# ===========================================================================

class TestExplorationAgentPhaseTransitions(unittest.TestCase):

    def test_approach_to_scan_on_succeed(self):
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=16.0, y=16.0))
        wq = _WQ(position=Position(x=16.0, y=16.0),
                 nearby_uncharted=[_chunk(1, 0)])
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # skill → SUCCEEDED
        agent.tick(task, bb, wq, _WW, 2, _KB)   # _tick_approach sees SUCCEEDED → SCAN
        self.assertEqual(agent._phase, _Phase.SCAN)

    def test_approach_to_scan_on_failed(self):
        """FAILED skill also transitions to SCAN on the following tick."""
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=500, y=500))
        wq = _WQ(position=Position(x=0, y=0),
                 nearby_uncharted=[_chunk(2, 2)])
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        # Force skill to FAILED directly
        agent._skill._status = SkillStatus.FAILED
        # _tick_approach now sees FAILED → transitions to SCAN
        agent.tick(task, bb, wq, _WW, 1, _KB)
        self.assertEqual(agent._phase, _Phase.SCAN)

    def test_scan_does_not_go_back_to_approach(self):
        """Once in SCAN the agent stays in SCAN."""
        agent = _make_agent()
        task = _Task()   # no frontier → SCAN immediately
        wq = _WQ(nearby_uncharted=[_chunk(1, 0)])
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        for i in range(1, 6):
            agent.tick(task, bb, wq, _WW, i, _KB)
        self.assertEqual(agent._phase, _Phase.SCAN)


# ===========================================================================
# Section 6 — observe() and progress()
# ===========================================================================

class TestExplorationAgentObserveProgress(unittest.TestCase):

    def test_observe_keys_present(self):
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=100, y=100))
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        obs = agent.observe(task, bb, wq, _KB)
        for key in ("agent", "task_id", "phase", "player_position",
                    "skill_status", "charted_chunks", "nearby_uncharted",
                    "needs_new_frontier", "chunks_approached"):
            self.assertIn(key, obs, f"missing key: {key}")

    def test_observe_agent_id(self):
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=100, y=100))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        self.assertEqual(agent.observe(task, bb, _WQ(), _KB)["agent"], "exploration")

    def test_observe_phase_approach(self):
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=100, y=100))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        self.assertEqual(agent.observe(task, bb, _WQ(), _KB)["phase"], "APPROACH")

    def test_observe_phase_scan(self):
        agent = _make_agent()
        task = _Task()
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        self.assertEqual(agent.observe(task, bb, _WQ(), _KB)["phase"], "SCAN")

    def test_progress_approach_running(self):
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=500, y=500))
        bb = _make_bb()
        wq = _WQ(position=Position(x=0, y=0))
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)  # skill now RUNNING
        p = agent.progress(task, bb, wq, _KB)
        self.assertAlmostEqual(p, 0.1)

    def test_progress_scan_with_uncharted(self):
        agent = _make_agent()
        task = _Task()
        bb = _make_bb()
        wq = _WQ(nearby_uncharted=[_chunk(1, 0)])
        agent.activate(task, bb, wq, _KB)
        p = agent.progress(task, bb, wq, _KB)
        self.assertAlmostEqual(p, 0.5)

    def test_progress_scan_no_uncharted(self):
        agent = _make_agent()
        task = _Task()
        bb = _make_bb()
        wq = _WQ(nearby_uncharted=[])
        agent.activate(task, bb, wq, _KB)
        p = agent.progress(task, bb, wq, _KB)
        self.assertAlmostEqual(p, 0.9)

    def test_observe_nearby_uncharted_count(self):
        agent = _make_agent()
        task = _Task()
        bb = _make_bb()
        wq = _WQ(nearby_uncharted=[_chunk(1, 0), _chunk(0, 1), _chunk(-1, 0)])
        agent.activate(task, bb, wq, _KB)
        obs = agent.observe(task, bb, wq, _KB)
        self.assertEqual(obs["nearby_uncharted"], 3)


# ===========================================================================
# Section 7 — pending_patches
# ===========================================================================

class TestExplorationAgentPendingPatches(unittest.TestCase):

    def test_always_empty(self):
        agent = _make_agent()
        task = _Task(frontier_position=Position(x=100, y=100))
        bb = _make_bb()
        wq = _WQ(nearby_uncharted=[_chunk(1, 0)])
        agent.activate(task, bb, wq, _KB)
        for i in range(1, 5):
            agent.tick(task, bb, wq, _WW, i, _KB)
        self.assertEqual(agent.pending_patches(), [])


# ===========================================================================
# Module-level helper tests
# ===========================================================================

class TestExplorationHelpers(unittest.TestCase):

    def test_chunk_centre_zero_zero(self):
        c = _chunk_centre(0, 0)
        self.assertAlmostEqual(c.x, _CHUNK_SIZE / 2.0)
        self.assertAlmostEqual(c.y, _CHUNK_SIZE / 2.0)

    def test_chunk_centre_nonzero(self):
        c = _chunk_centre(3, 2)
        self.assertAlmostEqual(c.x, 3 * _CHUNK_SIZE + _CHUNK_SIZE / 2.0)
        self.assertAlmostEqual(c.y, 2 * _CHUNK_SIZE + _CHUNK_SIZE / 2.0)

    def test_chunk_centre_negative(self):
        c = _chunk_centre(-1, -1)
        self.assertAlmostEqual(c.x, -_CHUNK_SIZE + _CHUNK_SIZE / 2.0)
        self.assertAlmostEqual(c.y, -_CHUNK_SIZE + _CHUNK_SIZE / 2.0)

    def test_nearest_chunk_single(self):
        chunks = [_chunk(2, 0)]
        player = Position(x=0, y=0)
        result = _nearest_chunk(chunks, player)
        self.assertEqual(result.cx, 2)
        self.assertEqual(result.cy, 0)

    def test_nearest_chunk_picks_closer(self):
        chunks = [_chunk(10, 0), _chunk(1, 0)]
        player = Position(x=0, y=0)
        result = _nearest_chunk(chunks, player)
        self.assertEqual(result.cx, 1)

    def test_nearest_chunk_diagonal(self):
        chunks = [_chunk(1, 1), _chunk(0, 2)]
        # chunk(1,1) centre: (48,48), dist^2 = 48^2+48^2 = 4608
        # chunk(0,2) centre: (16,80), dist^2 = 16^2+80^2 = 6656
        player = Position(x=0, y=0)
        result = _nearest_chunk(chunks, player)
        self.assertEqual((result.cx, result.cy), (1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)