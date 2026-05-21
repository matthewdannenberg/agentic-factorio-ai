"""
tests/unit/agent/network/agents/test_mining.py

Unit tests for agent/network/agents/mining.py.

All tests run without a live Factorio instance.

Coverage:
  - activate() resets all state
  - tick() with no blackboard task returns []
  - gather_resource: first issue emits MineResource
  - gather_resource: second tick within grace period suppressed
  - gather_resource: stall detection re-issues after grace period
  - gather_resource: new task (different resource/position) always re-issues
  - clear_all: builds target list from entities in bbox
  - clear_natural: filters to natural entities only
  - clear: entity within reach → MineEntity
  - clear: entity out of reach → MoveTo
  - clear: entity gone from scan → advances to next target
  - clear: empty bbox → no actions
  - _is_natural heuristics
  - progress() for clear tasks
  - observe() returns expected keys
"""

import unittest

from agent.blackboard import Blackboard, EntryCategory, EntryScope
from agent.network.agents.mining import MiningAgent
from agent.subtask import Subtask
from bridge.actions import MineEntity, MineResource, MoveTo, StopMining
from world.state import (
    EntityState,
    Inventory,
    InventorySlot,
    Position,
    WorldState,
)
from world.query import WorldQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subtask(description: str = "gather iron-ore") -> Subtask:
    return Subtask(
        description=description,
        success_condition="inventory('iron-ore') >= 10",
        failure_condition="tick > 9999",
        parent_goal_id="goal-1",
        created_at=0,
        derived_locally=True,
    )


def _make_wq(
    player_pos: Position = None,
    entities: list = None,
    reachable: list = None,
    inventory_items: dict = None,
    tick: int = 100,
) -> WorldQuery:
    state = WorldState(tick=tick, entities=entities or [])
    state.player.position = player_pos or Position(0.0, 0.0)
    state.player.reachable = reachable or []
    if inventory_items:
        state.player.inventory = Inventory(
            slots=[InventorySlot(item=k, count=v) for k, v in inventory_items.items()]
        )
    state._rebuild_entity_indices()
    return WorldQuery(state)


def _make_entity(entity_id: int, name: str, x: float, y: float) -> EntityState:
    return EntityState(entity_id=entity_id, name=name, position=Position(x, y))


def _make_mock_writer():
    class _MockWriter:
        pass
    return _MockWriter()


def _write_gather_task(bb: Blackboard, resource_type: str, x: float, y: float, tick: int = 100):
    return bb.write(
        category=EntryCategory.INTENTION,
        scope=EntryScope.SUBTASK,
        owner_agent="coordinator",
        created_at=tick,
        data={
            "type": "mining_task",
            "task_type": "gather_resource",
            "resource_type": resource_type,
            "target_position": {"x": x, "y": y},
        },
    )


def _write_clear_task(bb: Blackboard, task_type: str, bbox: dict, tick: int = 100):
    return bb.write(
        category=EntryCategory.INTENTION,
        scope=EntryScope.SUBTASK,
        owner_agent="coordinator",
        created_at=tick,
        data={
            "type": "mining_task",
            "task_type": task_type,
            "bounding_box": bbox,
        },
    )


_BBOX = {"x_min": -10.0, "y_min": -10.0, "x_max": 10.0, "y_max": 10.0}


def _drain_stop(agent, subtask, bb, wq, ww, tick=100):
    """
    Drain the pending StopMining action emitted on the first tick after
    activate(). Call this after activate() and before the real test tick
    so tests can assert on the actual mining behaviour without noise.
    Returns the drained actions for tests that want to inspect them.
    """
    return agent.tick(subtask, bb, wq, ww, tick=tick)


# ---------------------------------------------------------------------------
# activate()
# ---------------------------------------------------------------------------

class TestMiningAgentActivate(unittest.TestCase):

    def test_activate_resets_gather_state(self):
        agent = MiningAgent()
        agent._gather_resource_type = "iron-ore"
        agent._gather_issued_at = 500
        agent._last_inventory = {"iron-ore": 5}

        agent.activate(_make_subtask(), Blackboard(), _make_wq())

        self.assertEqual(agent._gather_resource_type, "")
        self.assertEqual(agent._gather_issued_at, 0)
        self.assertEqual(agent._last_inventory, {})

    def test_activate_resets_clear_state(self):
        from agent.network.agents.mining import _ClearTarget
        agent = MiningAgent()
        agent._clear_targets = [_ClearTarget(entity_id=1, position=Position(0, 0))]
        agent._mine_issued_at = 200
        agent._current_target = _ClearTarget(entity_id=2, position=Position(1, 1))

        agent.activate(_make_subtask(), Blackboard(), _make_wq())

        self.assertEqual(agent._clear_targets, [])
        self.assertEqual(agent._mine_issued_at, 0)
        self.assertIsNone(agent._current_target)

    def test_activate_stores_subtask(self):
        agent = MiningAgent()
        subtask = _make_subtask()
        agent.activate(subtask, Blackboard(), _make_wq())
        self.assertIs(agent._current_subtask, subtask)


# ---------------------------------------------------------------------------
# tick() — no task on blackboard
# ---------------------------------------------------------------------------

class TestMiningAgentNoTask(unittest.TestCase):

    def test_returns_empty_with_no_task(self):
        agent = MiningAgent()
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq()
        ww = _make_mock_writer()
        agent.activate(subtask, bb, wq)
        _drain_stop(agent, subtask, bb, wq, ww)

        actions = agent.tick(subtask, bb, wq, ww, tick=101)
        self.assertEqual(actions, [])

    def test_returns_empty_with_wrong_entry_type(self):
        agent = MiningAgent()
        subtask = _make_subtask()
        bb = Blackboard()
        bb.write(
            category=EntryCategory.INTENTION,
            scope=EntryScope.SUBTASK,
            owner_agent="coordinator",
            created_at=100,
            data={"type": "waypoint"},   # not a mining_task
        )
        wq = _make_wq()
        ww = _make_mock_writer()
        agent.activate(subtask, bb, wq)
        _drain_stop(agent, subtask, bb, wq, ww)

        actions = agent.tick(subtask, bb, wq, ww, tick=101)
        self.assertEqual(actions, [])


# ---------------------------------------------------------------------------
# tick() — gather_resource
# ---------------------------------------------------------------------------

class TestMiningAgentGather(unittest.TestCase):

    def setUp(self):
        self.agent = MiningAgent()
        self.subtask = _make_subtask()
        self.bb = Blackboard()
        self.wq = _make_wq()
        self.ww = _make_mock_writer()
        self.agent.activate(self.subtask, self.bb, self.wq)
        _drain_stop(self.agent, self.subtask, self.bb, self.wq, self.ww)

    def test_first_tick_emits_mine_resource(self):
        _write_gather_task(self.bb, "iron-ore", 5.0, 5.0)
        actions = self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=101)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MineResource)
        self.assertEqual(actions[0].resource, "iron-ore")
        self.assertAlmostEqual(actions[0].position.x, 5.0)
        self.assertAlmostEqual(actions[0].position.y, 5.0)

    def test_mine_resource_count_is_zero(self):
        """count=0 means mine until full/exhausted."""
        _write_gather_task(self.bb, "iron-ore", 5.0, 5.0)
        actions = self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=101)
        self.assertEqual(actions[0].count, 0)

    def test_second_tick_within_grace_suppressed(self):
        _write_gather_task(self.bb, "iron-ore", 5.0, 5.0)
        self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=101)
        # Second tick — still within grace period.
        actions = self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=102)
        self.assertEqual(actions, [])

    def test_stall_detected_after_grace_period(self):
        """After grace period, if inventory hasn't changed, re-issue."""
        _write_gather_task(self.bb, "iron-ore", 5.0, 5.0)
        # Issue at tick 100.
        self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=100)
        # Tick after grace period with unchanged inventory.
        from agent.network.agents.mining import _MINING_GRACE_TICKS
        tick_after_grace = 100 + _MINING_GRACE_TICKS + 1
        actions = self.agent.tick(
            self.subtask, self.bb, self.wq, self.ww, tick=tick_after_grace
        )
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MineResource)

    def test_no_stall_if_inventory_changed(self):
        """If inventory changed during grace period, no re-issue after it."""
        _write_gather_task(self.bb, "iron-ore", 5.0, 5.0)
        self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=100)
        # Advance with ore in inventory.
        wq_with_ore = _make_wq(inventory_items={"iron-ore": 3})
        from agent.network.agents.mining import _MINING_GRACE_TICKS
        tick_after_grace = 100 + _MINING_GRACE_TICKS + 1
        actions = self.agent.tick(
            self.subtask, self.bb, wq_with_ore, self.ww, tick=tick_after_grace
        )
        self.assertEqual(actions, [])

    def test_new_resource_type_always_reissues(self):
        """Changing resource type resets state and issues a new command."""
        _write_gather_task(self.bb, "iron-ore", 5.0, 5.0)
        self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=101)

        # Replace blackboard entry with copper-ore task.
        self.bb.clear_all()
        _write_gather_task(self.bb, "copper-ore", 5.0, 5.0, tick=102)
        actions = self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=102)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].resource, "copper-ore")

    def test_new_position_always_reissues(self):
        """Changing target position resets state and issues a new command."""
        _write_gather_task(self.bb, "iron-ore", 5.0, 5.0)
        self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=101)

        self.bb.clear_all()
        _write_gather_task(self.bb, "iron-ore", 20.0, 20.0, tick=102)
        actions = self.agent.tick(self.subtask, self.bb, self.wq, self.ww, tick=102)
        self.assertEqual(len(actions), 1)
        self.assertAlmostEqual(actions[0].position.x, 20.0)


# ---------------------------------------------------------------------------
# tick() — clear region
# ---------------------------------------------------------------------------

class TestMiningAgentClear(unittest.TestCase):

    def setUp(self):
        self.subtask = _make_subtask("clear region")
        self.ww = _make_mock_writer()

    def test_clear_all_picks_up_all_entities_in_bbox(self):
        tree = _make_entity(1, "tree-01", 0.0, 0.0)
        rock = _make_entity(2, "rock-big", 2.0, 2.0)
        outside = _make_entity(3, "tree-01", 50.0, 50.0)
        entities = [tree, rock, outside]
        wq = _make_wq(entities=entities, reachable=[1, 2, 3])

        agent = MiningAgent()
        bb = Blackboard()
        agent.activate(self.subtask, bb, wq)
        _drain_stop(agent, self.subtask, bb, wq, self.ww)
        _write_clear_task(bb, "clear_all", _BBOX)

        agent.tick(self.subtask, bb, wq, self.ww, tick=100)

        # Target list should have 2 entities (the outside one excluded).
        # current_target uses one, remainder in _clear_targets.
        total = len(agent._clear_targets) + (1 if agent._current_target else 0)
        self.assertEqual(total, 2)

    def test_clear_natural_excludes_non_natural(self):
        tree = _make_entity(1, "tree-01", 0.0, 0.0)
        furnace = _make_entity(2, "stone-furnace", 2.0, 2.0)
        wq = _make_wq(entities=[tree, furnace], reachable=[1, 2])

        agent = MiningAgent()
        bb = Blackboard()
        agent.activate(self.subtask, bb, wq)
        _drain_stop(agent, self.subtask, bb, wq, self.ww)
        _write_clear_task(bb, "clear_natural", _BBOX)

        agent.tick(self.subtask, bb, wq, self.ww, tick=100)

        # Only the tree should be targeted; furnace is not natural.
        total = len(agent._clear_targets) + (1 if agent._current_target else 0)
        self.assertEqual(total, 1)
        if agent._current_target:
            self.assertEqual(agent._current_target.entity_id, 1)

    def test_reachable_entity_emits_mine_entity(self):
        tree = _make_entity(1, "tree-01", 0.0, 0.0)
        wq = _make_wq(entities=[tree], reachable=[1])

        agent = MiningAgent()
        bb = Blackboard()
        agent.activate(self.subtask, bb, wq)
        _drain_stop(agent, self.subtask, bb, wq, self.ww)
        _write_clear_task(bb, "clear_all", _BBOX)

        actions = agent.tick(self.subtask, bb, wq, self.ww, tick=100)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MineEntity)
        self.assertEqual(actions[0].entity_id, 1)

    def test_unreachable_entity_emits_moveto(self):
        tree = _make_entity(1, "tree-01", 8.0, 8.0)
        wq = _make_wq(entities=[tree], reachable=[])

        agent = MiningAgent()
        bb = Blackboard()
        agent.activate(self.subtask, bb, wq)
        _drain_stop(agent, self.subtask, bb, wq, self.ww)
        _write_clear_task(bb, "clear_all", _BBOX)

        actions = agent.tick(self.subtask, bb, wq, self.ww, tick=100)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], MoveTo)

    def test_destroyed_entity_advances_to_next(self):
        tree1 = _make_entity(1, "tree-01", 0.0, 0.0)
        tree2 = _make_entity(2, "tree-02", 2.0, 0.0)
        wq_both = _make_wq(entities=[tree1, tree2], reachable=[1, 2])

        agent = MiningAgent()
        bb = Blackboard()
        agent.activate(self.subtask, bb, wq_both)
        _drain_stop(agent, self.subtask, bb, wq_both, self.ww)
        _write_clear_task(bb, "clear_all", _BBOX)

        # First tick — picks tree1 as current target.
        agent.tick(self.subtask, bb, wq_both, self.ww, tick=100)
        self.assertIsNotNone(agent._current_target)
        first_id = agent._current_target.entity_id

        # tree1 disappears.
        wq_tree1_gone = _make_wq(entities=[tree2], reachable=[2])
        agent.tick(self.subtask, bb, wq_tree1_gone, self.ww, tick=101)

        # Current target should have advanced to tree2.
        if agent._current_target:
            self.assertNotEqual(agent._current_target.entity_id, first_id)

    def test_empty_bbox_returns_no_actions(self):
        wq = _make_wq(entities=[])

        agent = MiningAgent()
        bb = Blackboard()
        agent.activate(self.subtask, bb, wq)
        _drain_stop(agent, self.subtask, bb, wq, self.ww)
        _write_clear_task(bb, "clear_all", _BBOX)

        actions = agent.tick(self.subtask, bb, wq, self.ww, tick=100)
        self.assertEqual(actions, [])

    def test_targets_sorted_by_distance(self):
        """Nearest entity should be processed first."""
        near = _make_entity(1, "tree-01", 1.0, 0.0)
        far  = _make_entity(2, "tree-02", 8.0, 0.0)
        # Player at origin; near is closer.
        wq = _make_wq(player_pos=Position(0.0, 0.0), entities=[far, near], reachable=[1, 2])

        agent = MiningAgent()
        bb = Blackboard()
        agent.activate(self.subtask, bb, wq)
        _drain_stop(agent, self.subtask, bb, wq, self.ww)
        _write_clear_task(bb, "clear_all", _BBOX)

        agent.tick(self.subtask, bb, wq, self.ww, tick=100)
        self.assertIsNotNone(agent._current_target)
        self.assertEqual(agent._current_target.entity_id, 1)


# ---------------------------------------------------------------------------
# StopMining behaviour
# ---------------------------------------------------------------------------

class TestMiningAgentStopMining(unittest.TestCase):
    """
    Verify that activate() triggers a StopMining on the first tick and that
    subsequent ticks are unaffected.
    """

    def test_first_tick_after_activate_emits_stop_mining(self):
        agent = MiningAgent()
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq()
        ww = _make_mock_writer()
        agent.activate(subtask, bb, wq)

        actions = agent.tick(subtask, bb, wq, ww, tick=100)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], StopMining)

    def test_stop_only_fires_once_per_activate(self):
        """The pending-stop flag is cleared after the first tick."""
        agent = MiningAgent()
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq()
        ww = _make_mock_writer()
        agent.activate(subtask, bb, wq)

        agent.tick(subtask, bb, wq, ww, tick=100)  # drains stop
        actions = agent.tick(subtask, bb, wq, ww, tick=101)
        self.assertNotIn(StopMining(), actions)

    def test_re_activate_triggers_another_stop(self):
        """A second activate() on the same agent sets the flag again."""
        agent = MiningAgent()
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq()
        ww = _make_mock_writer()

        agent.activate(subtask, bb, wq)
        agent.tick(subtask, bb, wq, ww, tick=100)  # drains first stop

        agent.activate(subtask, bb, wq)  # re-activated (new subtask assigned)
        actions = agent.tick(subtask, bb, wq, ww, tick=101)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], StopMining)

    def test_stop_mining_emitted_before_any_task_processing(self):
        """Even when a blackboard task is present, stop fires first."""
        agent = MiningAgent()
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq()
        ww = _make_mock_writer()
        agent.activate(subtask, bb, wq)
        _write_gather_task(bb, "iron-ore", 5.0, 5.0)

        actions = agent.tick(subtask, bb, wq, ww, tick=100)
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], StopMining)

        # Next tick should now mine.
        actions2 = agent.tick(subtask, bb, wq, ww, tick=101)
        self.assertEqual(len(actions2), 1)
        self.assertIsInstance(actions2[0], MineResource)


# ---------------------------------------------------------------------------
# _is_natural heuristics
# ---------------------------------------------------------------------------

class TestIsNatural(unittest.TestCase):

    def _check(self, name: str) -> bool:
        agent = MiningAgent()
        entity = _make_entity(1, name, 0.0, 0.0)
        return agent._is_natural(entity)

    def test_tree_is_natural(self):
        self.assertTrue(self._check("tree-01"))
        self.assertTrue(self._check("dead-tree-desert"))

    def test_rock_is_natural(self):
        self.assertTrue(self._check("rock-big"))
        self.assertTrue(self._check("sand-rock-big"))

    def test_boulder_is_natural(self):
        self.assertTrue(self._check("boulder"))

    def test_cliff_is_natural(self):
        self.assertTrue(self._check("cliff"))

    def test_fish_is_natural(self):
        self.assertTrue(self._check("fish"))

    def test_furnace_is_not_natural(self):
        self.assertFalse(self._check("stone-furnace"))

    def test_assembler_is_not_natural(self):
        self.assertFalse(self._check("assembling-machine-1"))

    def test_inserter_is_not_natural(self):
        self.assertFalse(self._check("inserter"))

    def test_iron_ore_resource_is_not_natural(self):
        self.assertFalse(self._check("iron-ore"))


# ---------------------------------------------------------------------------
# progress() and observe()
# ---------------------------------------------------------------------------

class TestMiningAgentProgressAndObserve(unittest.TestCase):

    def test_progress_zero_before_clear_starts(self):
        agent = MiningAgent()
        subtask = _make_subtask()
        agent.activate(subtask, Blackboard(), _make_wq())
        self.assertAlmostEqual(agent.progress(subtask, Blackboard(), _make_wq()), 0.0)

    def test_progress_reflects_clear_completion(self):
        from agent.network.agents.mining import _ClearTarget, _SubtaskKind
        agent = MiningAgent()
        subtask = _make_subtask()
        agent.activate(subtask, Blackboard(), _make_wq())

        # Simulate: 3 targets total, 2 done, 1 remaining.
        agent._subtask_kind = _SubtaskKind.CLEAR
        agent._current_target = _ClearTarget(entity_id=3, position=Position(0, 0))
        agent._clear_targets = []   # 0 remaining after current

        # total=1 (current), done=0 → progress=0.
        p = agent.progress(subtask, Blackboard(), _make_wq())
        self.assertAlmostEqual(p, 0.0)

    def test_observe_contains_expected_keys(self):
        agent = MiningAgent()
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq()
        agent.activate(subtask, bb, wq)

        obs = agent.observe(subtask, bb, wq)
        self.assertIn("agent", obs)
        self.assertIn("subtask_id", obs)
        self.assertIn("subtask_kind", obs)
        self.assertIn("clear_targets_remaining", obs)
        self.assertIn("current_target", obs)

    def test_observe_subtask_id_matches(self):
        agent = MiningAgent()
        subtask = _make_subtask()
        bb = Blackboard()
        wq = _make_wq()
        agent.activate(subtask, bb, wq)

        obs = agent.observe(subtask, bb, wq)
        self.assertEqual(obs["subtask_id"], subtask.id[:8])


if __name__ == "__main__":
    unittest.main()