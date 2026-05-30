"""
tests/unit/execution/test_navigation_agent.py

Unit tests for execution/agents/navigation.py (NavigationAgent).

All movement logic lives in NavigateSkill, which is already tested in
test_skills.py. These tests focus on the agent's own responsibilities:
  - Reading task attributes correctly on activate()
  - Delegating to NavigateSkill and returning its actions
  - Writing correct blackboard observations
  - Translating skill status to task outcome observations
  - progress() and observe() returning sensible values
  - pending_patches() always empty

Sections
--------
1. Stubs and helpers
2. activate() — task attribute reading, blackboard writes, error cases
3. tick() — action delegation, position observation, terminal handling
4. Outcome observations — succeeded / stuck / failed blackboard entries
5. observe() and progress()
6. pending_patches()

Run with:  python -m pytest tests/unit/execution/test_navigation_agent.py -v
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

from execution.agents.navigation import NavigationAgent
from execution.blackboard import Blackboard, EntryCategory, EntryScope
from execution.skills.base import SkillStatus
from world import Position
from bridge import MoveTo, StopMovement


# ===========================================================================
# Section 1 — Stubs
# ===========================================================================

@dataclass
class _Task:
    """Minimal task stub with dynamic attribute support."""
    id: str = "task-0001"
    description: str = "navigate test"
    task_type: str = "navigate_to_position"
    target_position: Optional[Position] = None
    target_entity_id: Optional[int] = None


@dataclass
class _EntityState:
    entity_id: int
    position: Position = field(default_factory=lambda: Position(0.0, 0.0))


class _WQ:
    """Minimal WorldQuery stub for navigation tests."""

    def __init__(
        self,
        position: Position = None,
        reachable: list[int] = None,
        entities: list[_EntityState] = None,
        tick: int = 1,
    ):
        self._position = position or Position(x=0.0, y=0.0)
        self._reachable = reachable or []
        self._entities = {e.entity_id: e for e in (entities or [])}
        self.tick = tick
        self.state = MagicMock()
        self.state.player.reachable = self._reachable
        self.state.player.inventory.slots = []
        self.state.player.crafting_queue_size = 0

    def player_position(self) -> Position:
        return self._position

    def entity_by_id(self, eid: int):
        return self._entities.get(eid)

    def inventory_count(self, item: str) -> int:
        return 0

    @property
    def charted_chunks(self) -> int:
        return 0

    @property
    def nearby_uncharted_chunks(self) -> list:
        return []

    @property
    def crafting_queue_size(self) -> int:
        return 0

    def move_to(self, x: float, y: float) -> None:
        self._position = Position(x=x, y=y)

    def set_reachable(self, ids: list[int]) -> None:
        self._reachable.clear()
        self._reachable.extend(ids)
        self.state.player.reachable = self._reachable


_WW = MagicMock()
_KB = MagicMock()


def _make_agent() -> NavigationAgent:
    return NavigationAgent()


def _make_bb() -> Blackboard:
    return Blackboard()


def _obs_types(bb: Blackboard, tick: int = 0) -> list[str]:
    """Return all observation type strings written to the blackboard."""
    return [
        e.data.get("type", "")
        for e in bb.read(category=EntryCategory.OBSERVATION, current_tick=tick)
    ]


# ===========================================================================
# Section 2 — activate()
# ===========================================================================

class TestNavigationAgentActivate(unittest.TestCase):

    def test_position_task_starts_skill(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=100, y=50))
        wq = _WQ()
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        # Skill should be RUNNING after start()
        self.assertEqual(agent._skill.status(), SkillStatus.RUNNING)

    def test_entity_task_starts_skill(self):
        entity = _EntityState(entity_id=7, position=Position(x=20, y=20))
        agent = _make_agent()
        task = _Task(task_type="navigate_to_entity", target_entity_id=7)
        wq = _WQ(entities=[entity])
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        self.assertEqual(agent._skill.status(), SkillStatus.RUNNING)

    def test_no_target_leaves_skill_idle(self):
        agent = _make_agent()
        task = _Task()   # no target
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        self.assertEqual(agent._skill.status(), SkillStatus.IDLE)

    def test_activate_writes_navigation_started_observation(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=10, y=10))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        self.assertIn("navigation_started", _obs_types(bb))

    def test_navigation_started_contains_task_id(self):
        agent = _make_agent()
        task = _Task(id="task-abc", target_position=Position(x=10, y=10))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        obs = [e for e in bb.read() if e.data.get("type") == "navigation_started"]
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].data["task_id"], "task-abc")

    def test_reactivate_resets_skill(self):
        """Calling activate() twice resets skill for the new task."""
        agent = _make_agent()
        task1 = _Task(target_position=Position(x=100, y=100))
        task2 = _Task(target_position=Position(x=200, y=200))
        wq = _WQ()
        bb = _make_bb()
        agent.activate(task1, bb, wq, _KB)
        agent.tick(task1, bb, wq, _WW, 1, _KB)
        agent.activate(task2, bb, wq, _KB)
        # Skill should be RUNNING fresh for task2
        self.assertEqual(agent._skill.status(), SkillStatus.RUNNING)
        self.assertFalse(agent._outcome_written)

    def test_position_stored_in_observation(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=55, y=66))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        obs = [e for e in bb.read() if e.data.get("type") == "navigation_started"]
        self.assertAlmostEqual(obs[0].data["target_pos"]["x"], 55.0)
        self.assertAlmostEqual(obs[0].data["target_pos"]["y"], 66.0)


# ===========================================================================
# Section 3 — tick()
# ===========================================================================

class TestNavigationAgentTick(unittest.TestCase):

    def test_tick_returns_move_to_on_first_tick(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=100, y=100))
        wq = _WQ(position=Position(x=0, y=0))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        actions = agent.tick(task, bb, wq, _WW, 1, _KB)
        self.assertTrue(any(isinstance(a, MoveTo) for a in actions))

    def test_tick_writes_position_observation(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=100, y=100))
        wq = _WQ(position=Position(x=0, y=0))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        self.assertIn("player_position", _obs_types(bb))

    def test_tick_idle_returns_empty(self):
        """If activate() had no target, tick() returns nothing."""
        agent = _make_agent()
        task = _Task()
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        actions = agent.tick(task, bb, _WQ(), _WW, 1, _KB)
        self.assertEqual(actions, [])

    def test_tick_after_terminal_returns_empty(self):
        """Once the skill reaches a terminal state, tick() returns []."""
        agent = _make_agent()
        # Player already at target — skill succeeds on first tick
        task = _Task(target_position=Position(x=0, y=0))
        wq = _WQ(position=Position(x=0, y=0))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # skill → SUCCEEDED
        actions = agent.tick(task, bb, wq, _WW, 2, _KB)
        self.assertEqual(actions, [])

    def test_outcome_written_only_once(self):
        """Outcome observation is written exactly once when skill becomes terminal."""
        agent = _make_agent()
        task = _Task(target_position=Position(x=0, y=0))
        wq = _WQ(position=Position(x=0, y=0))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # skill → SUCCEEDED here
        agent.tick(task, bb, wq, _WW, 2, _KB)   # already terminal
        agent.tick(task, bb, wq, _WW, 3, _KB)

        outcome_obs = [
            e for e in bb.read()
            if e.data.get("type") == "navigation_succeeded"
        ]
        self.assertEqual(len(outcome_obs), 1)

    def test_entity_target_arrives_when_reachable(self):
        entity = _EntityState(entity_id=5, position=Position(x=10, y=10))
        wq = _WQ(
            position=Position(x=9, y=9),
            reachable=[5],
            entities=[entity],
        )
        agent = _make_agent()
        task = _Task(task_type="navigate_to_entity", target_entity_id=5)
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        actions = agent.tick(task, bb, wq, _WW, 1, _KB)
        self.assertEqual(agent._skill.status(), SkillStatus.SUCCEEDED)
        self.assertTrue(any(isinstance(a, StopMovement) for a in actions))


# ===========================================================================
# Section 4 — Outcome observations
# ===========================================================================

class TestNavigationAgentOutcomes(unittest.TestCase):

    def test_succeeded_observation_written(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=0, y=0))
        wq = _WQ(position=Position(x=0, y=0))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        self.assertIn("navigation_succeeded", _obs_types(bb))

    def test_failed_observation_for_missing_entity(self):
        """Entity not in wq → skill FAILED → navigation_failed written."""
        wq = _WQ()   # no entities
        agent = _make_agent()
        task = _Task(task_type="navigate_to_entity", target_entity_id=99)
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        self.assertIn("navigation_failed", _obs_types(bb))

    def test_stuck_observation_after_grace(self):
        """Skill transitions to STUCK → navigation_stuck written."""
        from execution.skills.navigate import _UNREACHABLE_GRACE_TICKS
        agent = _make_agent()
        task = _Task(target_position=Position(x=500, y=500))
        wq = _WQ(position=Position(x=0, y=0))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        # Drive ticks until STUCK
        for i in range(1, _UNREACHABLE_GRACE_TICKS // 10 + 5):
            agent.tick(task, bb, wq, _WW, i * 10, _KB)
            if agent._skill.status() == SkillStatus.STUCK:
                break
        self.assertIn("navigation_stuck", _obs_types(bb))

    def test_outcome_observation_contains_task_id(self):
        agent = _make_agent()
        task = _Task(id="task-xyz", target_position=Position(x=0, y=0))
        wq = _WQ(position=Position(x=0, y=0))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        outcome = [e for e in bb.read()
                   if e.data.get("type") == "navigation_succeeded"][0]
        self.assertEqual(outcome.data["task_id"], "task-xyz")

    def test_outcome_observation_contains_skill_observe(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=0, y=0))
        wq = _WQ(position=Position(x=0, y=0))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        outcome = [e for e in bb.read()
                   if e.data.get("type") == "navigation_succeeded"][0]
        self.assertIn("skill_observe", outcome.data)
        self.assertIn("navigate_status", outcome.data["skill_observe"])


# ===========================================================================
# Section 5 — observe() and progress()
# ===========================================================================

class TestNavigationAgentObserveProgress(unittest.TestCase):

    def test_observe_keys_present(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=10, y=10))
        wq = _WQ()
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        obs = agent.observe(task, bb, wq, _KB)
        for key in ("agent", "task_id", "task_type", "player_position",
                    "skill_status", "navigate_status"):
            self.assertIn(key, obs, f"missing key: {key}")

    def test_observe_agent_id(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=10, y=10))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        self.assertEqual(agent.observe(task, bb, _WQ(), _KB)["agent"], "navigation")

    def test_progress_zero_before_move_issued(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=100, y=100))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        # No tick yet — no MoveTo issued
        self.assertAlmostEqual(agent.progress(task, bb, _WQ(), _KB), 0.0)

    def test_progress_half_when_running(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=100, y=100))
        wq = _WQ(position=Position(x=0, y=0))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # issues MoveTo
        p = agent.progress(task, bb, wq, _KB)
        self.assertAlmostEqual(p, 0.5)

    def test_progress_one_when_succeeded(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=0, y=0))
        wq = _WQ(position=Position(x=0, y=0))
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        self.assertAlmostEqual(agent.progress(task, bb, wq, _KB), 1.0)

    def test_progress_zero_on_failed(self):
        agent = _make_agent()
        task = _Task(task_type="navigate_to_entity", target_entity_id=99)
        wq = _WQ()
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # FAILED immediately
        self.assertAlmostEqual(agent.progress(task, bb, wq, _KB), 0.0)


# ===========================================================================
# Section 6 — pending_patches
# ===========================================================================

class TestNavigationAgentPendingPatches(unittest.TestCase):

    def test_always_empty_before_activate(self):
        self.assertEqual(_make_agent().pending_patches(), [])

    def test_always_empty_after_activate(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=10, y=10))
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent.pending_patches(), [])

    def test_always_empty_after_ticks(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=100, y=100))
        wq = _WQ()
        bb = _make_bb()
        agent.activate(task, bb, wq, _KB)
        for i in range(1, 5):
            agent.tick(task, bb, wq, _WW, i, _KB)
        self.assertEqual(agent.pending_patches(), [])


# ===========================================================================
# Section — teardown()
# ===========================================================================

class TestNavigationAgentTeardown(unittest.TestCase):
    """
    NavigationAgent.teardown() always returns [StopMovement].

    The Lua mod's persistent on_tick walker keeps moving until explicitly
    stopped. Teardown ensures a clean slate regardless of nav state.
    """

    def test_teardown_returns_stop_movement(self):
        agent = _make_agent()
        actions = agent.teardown()
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], StopMovement)

    def test_teardown_after_activate_returns_stop_movement(self):
        agent = _make_agent()
        task = _Task(target_position=Position(x=100, y=100))
        agent.activate(task, _make_bb(), _WQ(), _KB)
        actions = agent.teardown()
        self.assertIsInstance(actions[0], StopMovement)

    def test_teardown_while_running_returns_stop_movement(self):
        """StopMovement is issued even mid-navigation."""
        agent = _make_agent()
        task = _Task(target_position=Position(x=100, y=100))
        bb = _make_bb()
        wq = _WQ(position=Position(0, 0))
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # skill now RUNNING
        actions = agent.teardown()
        self.assertIsInstance(actions[0], StopMovement)

    def test_teardown_idempotent(self):
        """Calling teardown twice is safe — both return [StopMovement]."""
        agent = _make_agent()
        task = _Task(target_position=Position(x=10, y=10))
        agent.activate(task, _make_bb(), _WQ(), _KB)
        first = agent.teardown()
        second = agent.teardown()
        self.assertIsInstance(first[0], StopMovement)
        self.assertIsInstance(second[0], StopMovement)



if __name__ == "__main__":
    unittest.main(verbosity=2)