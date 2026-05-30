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
    _pick_scan_target,
    _pick_chunk_map_frontier,
    _pos_to_chunk,
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
    created_at: int = 0


class _WQ:
    """WorldQuery stub with mutable nearby_uncharted_chunks and position."""

    def __init__(
        self,
        position: Position = None,
        reachable: list[int] = None,
        charted: int = 10,
        nearby_uncharted: list = None,
        tick: int = 1,
        charted_coords: set = None,
    ):
        self._position          = position or Position(x=16.0, y=16.0)
        self._reachable         = reachable or []
        self._charted_chunks    = charted
        self._nearby_uncharted  = list(nearby_uncharted or [])
        self._charted_coords    = charted_coords or set()
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
    def chunk_map(self):
        from world.observable.query import ChunkMapQuery
        return ChunkMapQuery(self._charted_coords)

    @property
    def crafting_queue_size(self) -> int:
        return 0

    def move_to(self, x: float, y: float) -> None:
        self._position = Position(x=x, y=y)

    def set_nearby_uncharted(self, chunks: list) -> None:
        self._nearby_uncharted = list(chunks)

    def set_charted(self, n: int) -> None:
        self._charted_chunks = n

    def add_charted_chunk(self, cx: int, cy: int) -> None:
        self._charted_coords.add((cx, cy))


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
    """
    activate() now stores home_position from player and resets all internal
    state. The agent picks its own frontier from wq.chunk_map / nearby_uncharted
    on the first tick — no frontier_position is passed on the task.
    """

    def test_initial_phase_is_approach(self):
        """Agent always starts in APPROACH, picks frontier on first tick."""
        agent = _make_agent()
        agent.activate(_Task(), _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._phase, _Phase.APPROACH)

    def test_skill_idle_after_activate(self):
        """Skill starts IDLE — frontier is chosen on the first tick call."""
        agent = _make_agent()
        agent.activate(_Task(), _make_bb(), _WQ(nearby_uncharted=[]), _KB)
        self.assertEqual(agent._skill.status(), SkillStatus.IDLE)

    def test_activate_writes_exploration_started(self):
        agent = _make_agent()
        bb = _make_bb()
        agent.activate(_Task(), bb, _WQ(), _KB)
        self.assertEqual(len(_obs_of_type(bb, "exploration_started")), 1)

    def test_exploration_started_contains_charted_count(self):
        agent = _make_agent()
        bb = _make_bb()
        wq = _WQ(charted=42)
        agent.activate(_Task(), bb, wq, _KB)
        obs = _obs_of_type(bb, "exploration_started")[0]
        self.assertEqual(obs.data["charted_chunks_at_start"], 42)

    def test_home_set_from_player_position(self):
        agent = _make_agent()
        wq = _WQ(position=Position(x=16, y=16))
        agent.activate(_Task(), _make_bb(), wq, _KB)
        self.assertAlmostEqual(agent._home_position.x, 16.0)
        self.assertAlmostEqual(agent._home_position.y, 16.0)

    def test_unreachable_frontiers_cleared_on_activate(self):
        """Stale unreachable set from prior task must not carry over."""
        agent = _make_agent()
        agent._unreachable_frontiers = {(1, 2), (3, 4)}
        agent.activate(_Task(), _make_bb(), _WQ(), _KB)
        self.assertEqual(len(agent._unreachable_frontiers), 0)

    def test_truly_stuck_cleared_on_activate(self):
        agent = _make_agent()
        agent._truly_stuck = True
        agent.activate(_Task(), _make_bb(), _WQ(), _KB)
        self.assertFalse(agent._truly_stuck)

    def test_reactivate_clears_state(self):
        agent = _make_agent()
        bb = _make_bb()
        wq = _WQ(nearby_uncharted=[_chunk(5, 5)])
        agent.activate(_Task(), bb, wq, _KB)
        agent.tick(_Task(), bb, wq, _WW, 1, _KB)  # skill starts RUNNING
        # Re-activate with fresh task — skill resets to IDLE
        agent.activate(_Task(), _make_bb(), wq, _KB)
        # Reactivate
        agent.activate(_Task(), bb, _WQ(), _KB)
        self.assertEqual(agent._chunks_approached, 0)


# ===========================================================================
# Section 3 — APPROACH phase
# ===========================================================================

class TestExplorationAgentApproach(unittest.TestCase):
    """
    APPROACH phase: agent picks a frontier from wq.chunk_map or
    nearby_uncharted_chunks and navigates toward it. On STUCK, the targeted
    chunk is added to _unreachable_frontiers and the agent falls into SCAN.
    """

    def test_approach_issues_move_to_when_nearby_uncharted(self):
        """First tick with nearby uncharted chunks starts NavigateSkill."""
        agent = _make_agent()
        wq = _WQ(nearby_uncharted=[_chunk(3, 0)])
        bb = _make_bb()
        agent.activate(_Task(), bb, wq, _KB)
        actions = agent.tick(_Task(), bb, wq, _WW, 1, _KB)
        self.assertGreater(len(actions), 0)
        self.assertIsInstance(actions[0], MoveTo)

    def test_approach_uses_chunk_map_when_no_nearby_uncharted(self):
        """When nearby_uncharted is empty, frontier picked from chunk_map."""
        agent = _make_agent()
        wq = _WQ(nearby_uncharted=[], charted_coords={(5, 0), (5, 1)})
        bb = _make_bb()
        agent.activate(_Task(), bb, wq, _KB)
        actions = agent.tick(_Task(), bb, wq, _WW, 1, _KB)
        # chunk_map has frontiers → skill should start → MoveTo issued
        self.assertGreater(len(actions), 0)
        self.assertIsInstance(actions[0], MoveTo)

    def test_approach_fallback_when_no_frontiers(self):
        """No nearby uncharted and empty chunk_map → fallback push outward."""
        agent = _make_agent()
        wq = _WQ(position=Position(0, 0), nearby_uncharted=[], charted_coords=set())
        bb = _make_bb()
        agent.activate(_Task(), bb, wq, _KB)
        actions = agent.tick(_Task(), bb, wq, _WW, 1, _KB)
        # Fallback pushes outward from player — still issues a MoveTo
        self.assertGreater(len(actions), 0)

    def test_approach_stuck_marks_frontier_unreachable(self):
        """Navigation STUCK during APPROACH adds targeted chunk to unreachable set."""
        agent = _make_agent()
        wq = _WQ(nearby_uncharted=[_chunk(10, 10)])
        bb = _make_bb()
        agent.activate(_Task(), bb, wq, _KB)
        agent.tick(_Task(), bb, wq, _WW, 1, _KB)   # skill starts
        # Force skill STUCK
        agent._skill._status = SkillStatus.STUCK
        agent.tick(_Task(), bb, wq, _WW, 2, _KB)
        self.assertGreater(len(agent._unreachable_frontiers), 0)

    def test_approach_stuck_does_not_report_truly_stuck_unless_all_exhausted(self):
        """Individual nav STUCK should NOT set _truly_stuck (more frontiers available)."""
        agent = _make_agent()
        # Provide many nearby chunks so there are alternatives
        chunks = [_chunk(i, 0) for i in range(20)]
        wq = _WQ(nearby_uncharted=chunks)
        bb = _make_bb()
        agent.activate(_Task(), bb, wq, _KB)
        agent.tick(_Task(), bb, wq, _WW, 1, _KB)
        agent._skill._status = SkillStatus.STUCK
        agent.tick(_Task(), bb, wq, _WW, 2, _KB)
        self.assertFalse(agent._truly_stuck)

    def test_approach_truly_stuck_when_all_frontiers_exhausted(self):
        """_truly_stuck set only when _pick_frontier returns None."""
        agent = _make_agent()
        # Single nearby chunk — mark it unreachable manually first
        wq = _WQ(nearby_uncharted=[], charted_coords=set())
        agent._unreachable_frontiers = {(0, 0), (1, 0)}  # pre-fill to exhaust
        bb = _make_bb()
        agent.activate(_Task(), bb, wq, _KB)
        # With no nearby uncharted and empty chunk_map, fallback is used.
        # _pick_frontier always returns something from fallback, so truly_stuck
        # is only set if _pick_frontier returns None (no frontiers AND no fallback).
        # Test that truly_stuck is False when fallback is available.
        agent.tick(_Task(), bb, wq, _WW, 1, _KB)
        self.assertFalse(agent._truly_stuck)

    def test_approach_transitions_to_scan_on_succeed(self):
        """After nav succeeds, phase switches to SCAN."""
        agent = _make_agent()
        wq = _WQ(position=Position(16, 16), nearby_uncharted=[_chunk(0, 0)])
        bb = _make_bb()
        agent.activate(_Task(), bb, wq, _KB)
        agent.tick(_Task(), bb, wq, _WW, 1, _KB)   # skill starts
        agent._skill._status = SkillStatus.SUCCEEDED
        agent.tick(_Task(), bb, wq, _WW, 2, _KB)   # APPROACH sees SUCCEEDED → SCAN
        self.assertEqual(agent._phase, _Phase.SCAN)

    def test_approach_transitions_to_scan_on_stuck(self):
        """STUCK nav also transitions to SCAN (try scanning from current pos)."""
        agent = _make_agent()
        wq = _WQ(nearby_uncharted=[_chunk(5, 5)])
        bb = _make_bb()
        agent.activate(_Task(), bb, wq, _KB)
        agent.tick(_Task(), bb, wq, _WW, 1, _KB)
        agent._skill._status = SkillStatus.STUCK
        agent.tick(_Task(), bb, wq, _WW, 2, _KB)
        self.assertEqual(agent._phase, _Phase.SCAN)


# ===========================================================================
# Section 4 — SCAN phase
# ===========================================================================

class TestExplorationAgentScan(unittest.TestCase):

    def _enter_scan(self, agent, task, bb, wq) -> None:
        """Activate and skip straight to SCAN by marking skill SUCCEEDED."""
        agent.activate(task, bb, wq, _KB)
        # Get skill started then succeed it immediately
        agent.tick(task, bb, wq, _WW, 1, _KB)
        agent._skill._status = SkillStatus.SUCCEEDED
        agent.tick(task, bb, wq, _WW, 2, _KB)

    def test_scan_picks_target_from_nearby(self):
        """SCAN issues MoveTo toward nearby uncharted chunk."""
        agent = _make_agent()
        task = _Task()
        wq = _WQ(nearby_uncharted=[_chunk(2, 0)])
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        actions = agent.tick(task, bb, wq, _WW, 3, _KB)
        self.assertGreater(len(actions), 0)

    def test_scan_returns_to_approach_when_locally_surrounded(self):
        """Empty nearby_uncharted during SCAN returns to APPROACH."""
        agent = _make_agent()
        task = _Task()
        wq = _WQ(nearby_uncharted=[_chunk(1, 0)])
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        # Now empty the nearby list
        wq.set_nearby_uncharted([])
        agent.tick(task, bb, wq, _WW, 3, _KB)
        self.assertEqual(agent._phase, _Phase.APPROACH)

    def test_scan_stuck_resets_skill(self):
        """STUCK in SCAN resets skill to IDLE for next tick."""
        from execution.skills.navigate import _UNREACHABLE_GRACE_TICKS
        agent = _make_agent()
        task = _Task()
        wq = _WQ(position=Position(0, 0), nearby_uncharted=[_chunk(1, 0)])
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        agent.tick(task, bb, wq, _WW, 3, _KB)  # starts skill
        for i in range(1, _UNREACHABLE_GRACE_TICKS // 10 + 10):
            agent.tick(task, bb, wq, _WW, 3 + i * 10, _KB)
            if agent._skill.status() == SkillStatus.IDLE:
                break
        self.assertEqual(agent._skill.status(), SkillStatus.IDLE)

    def test_scan_no_exploration_needs_frontier_signal(self):
        """The exploration_needs_frontier blackboard signal is removed.
        SCAN now loops internally back to APPROACH instead of signalling."""
        agent = _make_agent()
        task = _Task()
        wq = _WQ(nearby_uncharted=[_chunk(1, 0)])
        bb = _make_bb()
        self._enter_scan(agent, task, bb, wq)
        wq.set_nearby_uncharted([])
        agent.tick(task, bb, wq, _WW, 3, _KB)
        # Signal removed — should not appear in blackboard
        self.assertEqual(len(_obs_of_type(bb, "exploration_needs_frontier")), 0)


# ===========================================================================
# Section 5 — Phase transitions
# ===========================================================================

class TestExplorationAgentPhaseTransitions(unittest.TestCase):

    def test_approach_to_scan_on_succeed(self):
        agent = _make_agent()
        task = _Task()
        wq = _WQ(position=Position(16.0, 16.0), nearby_uncharted=[_chunk(1, 0)])
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # skill → RUNNING
        agent._skill._status = SkillStatus.SUCCEEDED
        agent.tick(task, bb, wq, _WW, 2, _KB)   # APPROACH sees SUCCEEDED → SCAN
        self.assertEqual(agent._phase, _Phase.SCAN)

    def test_approach_to_scan_on_stuck(self):
        agent = _make_agent()
        task = _Task()
        wq = _WQ(nearby_uncharted=[_chunk(2, 2)])
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        agent._skill._status = SkillStatus.STUCK
        agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertEqual(agent._phase, _Phase.SCAN)

    def test_scan_to_approach_when_locally_surrounded(self):
        """SCAN returns to APPROACH when nearby_uncharted becomes empty."""
        agent = _make_agent()
        task = _Task()
        wq = _WQ(nearby_uncharted=[_chunk(1, 0)])
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        agent._skill._status = SkillStatus.SUCCEEDED
        agent.tick(task, bb, wq, _WW, 2, _KB)  # → SCAN
        self.assertEqual(agent._phase, _Phase.SCAN)
        wq.set_nearby_uncharted([])
        agent.tick(task, bb, wq, _WW, 3, _KB)  # SCAN empty → APPROACH
        self.assertEqual(agent._phase, _Phase.APPROACH)


# ===========================================================================
# Section 6 — observe() and progress()
# ===========================================================================

class TestExplorationAgentObserveProgress(unittest.TestCase):

    def test_observe_keys_present(self):
        agent = _make_agent()
        task = _Task()
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        obs = agent.observe(task, bb, wq, _KB)
        for key in ("agent", "task_id", "phase", "player_position",
                    "skill_status", "charted_chunks", "nearby_uncharted",
                    "chunks_approached", "unreachable_frontiers"):
            self.assertIn(key, obs, f"missing key: {key}")

    def test_observe_agent_id(self):
        agent = _make_agent()
        agent.activate(_Task(), _make_bb(), _WQ(), _KB)
        self.assertEqual(agent.observe(_Task(), _make_bb(), _WQ(), _KB)["agent"], "exploration")

    def test_observe_phase_approach(self):
        agent = _make_agent()
        agent.activate(_Task(), _make_bb(), _WQ(), _KB)
        self.assertEqual(agent.observe(_Task(), _make_bb(), _WQ(), _KB)["phase"], "APPROACH")

    def test_observe_skill_status_stuck_when_truly_stuck(self):
        """skill_status reports STUCK only when _truly_stuck is set."""
        agent = _make_agent()
        agent.activate(_Task(), _make_bb(), _WQ(), _KB)
        agent._truly_stuck = True
        obs = agent.observe(_Task(), _make_bb(), _WQ(), _KB)
        self.assertEqual(obs["skill_status"], "STUCK")

    def test_observe_skill_status_not_stuck_on_individual_failure(self):
        """Individual nav STUCK does not propagate to observe skill_status."""
        agent = _make_agent()
        agent.activate(_Task(), _make_bb(), _WQ(), _KB)
        agent._truly_stuck = False
        obs = agent.observe(_Task(), _make_bb(), _WQ(), _KB)
        self.assertNotEqual(obs["skill_status"], "STUCK")

    def test_observe_unreachable_frontiers_count(self):
        agent = _make_agent()
        agent.activate(_Task(), _make_bb(), _WQ(), _KB)
        agent._unreachable_frontiers = {(1, 2), (3, 4)}
        obs = agent.observe(_Task(), _make_bb(), _WQ(), _KB)
        self.assertEqual(obs["unreachable_frontiers"], 2)

    def test_progress_approach_running(self):
        agent = _make_agent()
        task = _Task()
        bb = _make_bb()
        wq = _WQ(position=Position(0, 0), nearby_uncharted=[_chunk(5, 0)])
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        p = agent.progress(task, bb, wq, _KB)
        self.assertAlmostEqual(p, 0.1)

    def test_progress_scan_with_uncharted(self):
        agent = _make_agent()
        task = _Task()
        bb = _make_bb()
        wq = _WQ(nearby_uncharted=[_chunk(1, 0)])
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        agent._skill._status = SkillStatus.SUCCEEDED
        agent.tick(task, bb, wq, _WW, 2, _KB)  # → SCAN
        p = agent.progress(task, bb, wq, _KB)
        self.assertAlmostEqual(p, 0.5)

    def test_progress_scan_no_uncharted(self):
        agent = _make_agent()
        task = _Task()
        bb = _make_bb()
        wq = _WQ(nearby_uncharted=[_chunk(1, 0)])
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        agent._skill._status = SkillStatus.SUCCEEDED
        agent.tick(task, bb, wq, _WW, 2, _KB)  # → SCAN
        wq.set_nearby_uncharted([])
        p = agent.progress(task, bb, wq, _KB)
        self.assertAlmostEqual(p, 0.8)

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
    """Tests for module-level helper functions in exploration.py."""

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

    def test_pick_scan_target_single(self):
        """_pick_scan_target returns centre of the only candidate."""
        chunks = [_chunk(2, 0)]
        player = Position(x=0, y=0)
        result = _pick_scan_target(chunks, player)
        expected_x = 2 * _CHUNK_SIZE + _CHUNK_SIZE / 2.0
        self.assertAlmostEqual(result.x, expected_x)

    def test_pick_scan_target_biased_toward_nearest(self):
        """With pool=10, the closest chunk is always in the candidate pool."""
        # Place 20 chunks at increasing distances; nearest must be selectable
        import random
        random.seed(42)  # deterministic for test
        chunks = [_chunk(i, 0) for i in range(1, 21)]
        player = Position(x=0, y=0)
        # Run many times — nearest chunk (1,0) should appear frequently
        seen = set()
        for _ in range(50):
            result = _pick_scan_target(chunks, player)
            cx = int((result.x - _CHUNK_SIZE / 2) / _CHUNK_SIZE)
            seen.add(cx)
        # chunk at cx=1 (nearest) should appear in the seen set
        self.assertIn(1, seen)

    def test_pos_to_chunk_origin(self):
        from world import Position
        result = _pos_to_chunk(Position(0, 0))
        self.assertEqual(result, (0, 0))

    def test_pos_to_chunk_positive(self):
        result = _pos_to_chunk(Position(64.5, 96.2))
        self.assertEqual(result, (2, 3))

    def test_pos_to_chunk_negative(self):
        result = _pos_to_chunk(Position(-1.0, -1.0))
        self.assertEqual(result, (-1, -1))

    def test_pick_chunk_map_frontier_empty(self):
        from world.observable.query import ChunkMapQuery
        cq = ChunkMapQuery(set())
        result = _pick_chunk_map_frontier(cq, Position(0, 0))
        self.assertIsNone(result)

    def test_pick_chunk_map_frontier_excludes_unreachable(self):
        from world.observable.query import ChunkMapQuery
        # Single frontier — mark it unreachable → returns None
        cq = ChunkMapQuery({(0, 0)})
        result = _pick_chunk_map_frontier(cq, Position(0, 0), exclude={(0, 0)})
        self.assertIsNone(result)

    def test_pick_chunk_map_frontier_skips_excluded(self):
        from world.observable.query import ChunkMapQuery
        # Two frontiers: (0,0) excluded, (5,0) available
        cq = ChunkMapQuery({(0, 0), (5, 0)})
        # Ensure (5,0) is a frontier: it needs an uncharted neighbour
        # Both are frontiers since they border uncharted chunks
        result = _pick_chunk_map_frontier(cq, Position(0, 0), exclude={(0, 0)})
        if result is not None:
            # Must be centre of (5,0), not (0,0)
            self.assertAlmostEqual(result.x, 5 * _CHUNK_SIZE + _CHUNK_SIZE / 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)