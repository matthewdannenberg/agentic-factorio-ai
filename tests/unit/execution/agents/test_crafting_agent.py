"""
tests/unit/execution/test_crafting_agent.py

Unit tests for execution/agents/crafting.py (CraftingAgent).

Sections
--------
1.  Stubs and helpers
2.  activate() — target parsing, skill initialisation, edge cases
3.  tick() — action delegation, terminal handling, outcome once
4.  Outcome observations — succeeded / stuck
5.  observe() and progress()
6.  pending_patches()

Run with:  python -m pytest tests/unit/execution/test_crafting_agent.py -v
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

from execution.agents.crafting import CraftingAgent
from execution.blackboard import Blackboard, EntryCategory, EntryScope
from execution.skills.base import SkillStatus
from bridge import CraftItem


# ===========================================================================
# Stubs
# ===========================================================================

@dataclass
class _Slot:
    item: str
    count: int


@dataclass
class _Task:
    id: str = "task-craft-01"
    description: str = "craft test"
    task_type: str = "craft_items"
    targets: list = field(default_factory=list)


class _WQ:
    def __init__(
        self,
        inventory: dict[str, int] = None,
        tick: int = 1,
        crafting_queue_size: int = 0,
    ):
        self._inventory = dict(inventory or {})
        self.tick = tick
        self._crafting_queue_size = crafting_queue_size
        self.state = MagicMock()
        self.state.player.inventory.slots = self._build_slots()
        self.state.player.reachable = []
        self.state.player.crafting_queue_size = crafting_queue_size

    def _build_slots(self):
        return [_Slot(k, v) for k, v in self._inventory.items() if v > 0]

    def player_position(self):
        from world import Position
        return Position(x=0.0, y=0.0)

    def entity_by_id(self, eid):
        return None

    def inventory_count(self, item: str) -> int:
        return self._inventory.get(item, 0)

    @property
    def crafting_queue_size(self) -> int:
        return self._crafting_queue_size

    @property
    def nearby_uncharted_chunks(self) -> list:
        return []

    @property
    def charted_chunks(self) -> int:
        return 0

    def set_inventory(self, inv: dict[str, int]) -> None:
        self._inventory = dict(inv)
        self.state.player.inventory.slots = self._build_slots()

    def set_crafting_queue_size(self, n: int) -> None:
        self._crafting_queue_size = n
        self.state.player.crafting_queue_size = n


_WW = MagicMock()
_KB = MagicMock()


def _make_agent() -> CraftingAgent:
    return CraftingAgent()


def _make_bb() -> Blackboard:
    return Blackboard()


def _obs_of_type(bb: Blackboard, obs_type: str) -> list:
    return [
        e for e in bb.read(category=EntryCategory.OBSERVATION)
        if e.data.get("type") == obs_type
    ]


def _gear_targets(count=5):
    return [{"item": "iron-gear-wheel", "recipe": "iron-gear-wheel", "count": count}]


# ===========================================================================
# Section 2 — activate()
# ===========================================================================

class TestCraftingAgentActivate(unittest.TestCase):

    def test_valid_targets_start_skill(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._skill.status(), SkillStatus.RUNNING)

    def test_empty_targets_leaves_skill_idle(self):
        agent = _make_agent()
        task = _Task(targets=[])
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._skill.status(), SkillStatus.IDLE)

    def test_zero_count_target_is_filtered(self):
        agent = _make_agent()
        task = _Task(targets=[
            {"item": "iron-gear-wheel", "recipe": "iron-gear-wheel", "count": 0},
        ])
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._skill.status(), SkillStatus.IDLE)

    def test_missing_item_target_is_filtered(self):
        agent = _make_agent()
        task = _Task(targets=[{"recipe": "iron-gear-wheel", "count": 5}])
        agent.activate(task, _make_bb(), _WQ(), _KB)
        self.assertEqual(agent._skill.status(), SkillStatus.IDLE)

    def test_recipe_defaults_to_item_when_absent(self):
        agent = _make_agent()
        task = _Task(targets=[{"item": "iron-gear-wheel", "count": 5}])
        agent.activate(task, _make_bb(), _WQ(), _KB)
        # Skill is RUNNING — recipe was inferred from item name
        self.assertEqual(agent._skill.status(), SkillStatus.RUNNING)

    def test_activate_writes_crafting_started(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        self.assertEqual(len(_obs_of_type(bb, "crafting_started")), 1)

    def test_crafting_started_contains_targets(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets(count=7))
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        obs = _obs_of_type(bb, "crafting_started")[0]
        self.assertEqual(len(obs.data["targets"]), 1)
        self.assertEqual(obs.data["targets"][0]["item"], "iron-gear-wheel")
        self.assertEqual(obs.data["targets"][0]["count"], 7)

    def test_multiple_targets_all_parsed(self):
        agent = _make_agent()
        task = _Task(targets=[
            {"item": "iron-gear-wheel", "recipe": "iron-gear-wheel", "count": 5},
            {"item": "copper-cable",    "recipe": "copper-cable",    "count": 10},
        ])
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        obs = _obs_of_type(bb, "crafting_started")[0]
        self.assertEqual(len(obs.data["targets"]), 2)

    def test_reactivate_resets_skill_and_flag(self):
        agent = _make_agent()
        task1 = _Task(targets=_gear_targets())
        task2 = _Task(targets=[{"item": "copper-cable", "recipe": "copper-cable", "count": 3}])
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task1, bb, wq, _KB)
        agent.tick(task1, bb, wq, _WW, 1, _KB)
        agent.activate(task2, bb, wq, _KB)
        self.assertFalse(agent._outcome_written)
        self.assertEqual(agent._skill.status(), SkillStatus.RUNNING)


# ===========================================================================
# Section 3 — tick()
# ===========================================================================

class TestCraftingAgentTick(unittest.TestCase):

    def test_idle_skill_returns_empty(self):
        agent = _make_agent()
        task = _Task(targets=[])
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        actions = agent.tick(task, bb, _WQ(), _WW, 1, _KB)
        self.assertEqual(actions, [])

    def test_first_tick_dispatches_craft_items(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        actions = agent.activate(task, bb, _WQ(), _KB) or \
                  agent.tick(task, bb, _WQ(), _WW, 1, _KB)
        # tick() should return CraftItem actions
        actions = agent.tick(task, bb, _WQ(), _WW, 1, _KB)
        self.assertTrue(any(isinstance(a, CraftItem) for a in actions))

    def test_craft_item_correct_recipe_and_count(self):
        agent = _make_agent()
        task = _Task(targets=[{"item": "iron-gear-wheel",
                               "recipe": "iron-gear-wheel", "count": 7}])
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        actions = agent.tick(task, bb, _WQ(), _WW, 1, _KB)
        ci = [a for a in actions if isinstance(a, CraftItem)][0]
        self.assertEqual(ci.recipe, "iron-gear-wheel")
        self.assertEqual(ci.count, 7)

    def test_terminal_skill_returns_empty(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # dispatch
        # Confirm via queue
        wq.set_crafting_queue_size(1)
        agent.tick(task, bb, wq, _WW, 2, _KB)   # SUCCEEDED
        actions = agent.tick(task, bb, wq, _WW, 3, _KB)
        self.assertEqual(actions, [])

    def test_outcome_written_only_once(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        wq.set_crafting_queue_size(1)
        for i in range(2, 7):
            agent.tick(task, bb, wq, _WW, i, _KB)
        self.assertEqual(len(_obs_of_type(bb, "crafting_succeeded")), 1)


# ===========================================================================
# Section 4 — Outcome observations
# ===========================================================================

class TestCraftingAgentOutcomes(unittest.TestCase):

    def test_succeeded_via_queue(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # dispatch
        wq.set_crafting_queue_size(2)
        agent.tick(task, bb, wq, _WW, 2, _KB)   # SUCCEEDED on this tick
        agent.tick(task, bb, wq, _WW, 3, _KB)   # outcome written on next tick
        self.assertEqual(len(_obs_of_type(bb, "crafting_succeeded")), 1)

    def test_succeeded_via_inventory_fallback(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets(count=5))
        bb = _make_bb()
        wq = _WQ(inventory={})
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # dispatch
        wq.set_inventory({"iron-gear-wheel": 5})
        agent.tick(task, bb, wq, _WW, 2, _KB)
        agent.tick(task, bb, wq, _WW, 3, _KB)
        self.assertEqual(len(_obs_of_type(bb, "crafting_succeeded")), 1)

    def test_stuck_observation_written(self):
        from execution.skills.craft import _CRAFTING_GRACE_TICKS, _MAX_REISSUE
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        wq = _WQ(inventory={})
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # dispatch
        tick = 1
        for _ in range(_MAX_REISSUE + 2):
            tick += _CRAFTING_GRACE_TICKS + 10
            agent.tick(task, bb, wq, _WW, tick, _KB)
        # One more tick to write outcome
        agent.tick(task, bb, wq, _WW, tick + 1, _KB)
        self.assertEqual(len(_obs_of_type(bb, "crafting_stuck")), 1)

    def test_outcome_contains_task_id(self):
        agent = _make_agent()
        task = _Task(id="task-xyz", targets=_gear_targets())
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        wq.set_crafting_queue_size(1)
        agent.tick(task, bb, wq, _WW, 2, _KB)
        agent.tick(task, bb, wq, _WW, 3, _KB)
        obs = _obs_of_type(bb, "crafting_succeeded")
        self.assertEqual(obs[0].data["task_id"], "task-xyz")

    def test_outcome_contains_skill_observe(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        wq.set_crafting_queue_size(1)
        agent.tick(task, bb, wq, _WW, 2, _KB)
        agent.tick(task, bb, wq, _WW, 3, _KB)
        obs = _obs_of_type(bb, "crafting_succeeded")
        self.assertIn("skill_observe", obs[0].data)


# ===========================================================================
# Section 5 — observe() and progress()
# ===========================================================================

class TestCraftingAgentObserveProgress(unittest.TestCase):

    def test_observe_keys_present(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        obs = agent.observe(task, bb, wq, _KB)
        for key in ("agent", "task_id", "skill_status",
                    "craft_status", "craft_targets", "craft_dispatched"):
            self.assertIn(key, obs, f"missing key: {key}")

    def test_observe_agent_id(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        self.assertEqual(agent.observe(task, bb, _WQ(), _KB)["agent"], "crafting")

    def test_progress_zero_before_dispatch(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        agent.activate(task, bb, _WQ(), _KB)
        # No tick yet — dispatched=False
        self.assertAlmostEqual(agent.progress(task, bb, _WQ(), _KB), 0.0)

    def test_progress_half_after_dispatch(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)   # dispatch
        self.assertAlmostEqual(agent.progress(task, bb, wq, _KB), 0.5)

    def test_progress_one_on_succeeded(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        agent.tick(task, bb, wq, _WW, 1, _KB)
        wq.set_crafting_queue_size(1)
        agent.tick(task, bb, wq, _WW, 2, _KB)
        agent.tick(task, bb, wq, _WW, 3, _KB)
        self.assertAlmostEqual(agent.progress(task, bb, wq, _KB), 1.0)


# ===========================================================================
# Section 6 — pending_patches()
# ===========================================================================

class TestCraftingAgentPendingPatches(unittest.TestCase):

    def test_always_empty_before_activate(self):
        self.assertEqual(_make_agent().pending_patches(), [])

    def test_always_empty_after_ticks(self):
        agent = _make_agent()
        task = _Task(targets=_gear_targets())
        bb = _make_bb()
        wq = _WQ()
        agent.activate(task, bb, wq, _KB)
        for i in range(1, 5):
            agent.tick(task, bb, wq, _WW, i, _KB)
        self.assertEqual(agent.pending_patches(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
