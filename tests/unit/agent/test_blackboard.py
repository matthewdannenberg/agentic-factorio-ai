"""
tests/unit/agent/test_blackboard.py

Tests for agent/blackboard.py

Run with:  python -m pytest tests/unit/agent/test_blackboard.py -v
       or:  python -m unittest tests.unit.agent.test_blackboard
"""

from __future__ import annotations

import unittest

from agent.blackboard import Blackboard, BlackboardEntry, EntryCategory, EntryScope


class TestBlackboardWrite(unittest.TestCase):
    def setUp(self):
        self.bb = Blackboard()

    def test_write_returns_entry(self):
        entry = self.bb.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.GOAL,
            owner_agent="nav",
            created_at=100,
            data={"key": "val"},
        )
        self.assertIsInstance(entry, BlackboardEntry)
        self.assertEqual(entry.category, EntryCategory.OBSERVATION)
        self.assertEqual(entry.scope, EntryScope.GOAL)
        self.assertEqual(entry.owner_agent, "nav")
        self.assertEqual(entry.created_at, 100)
        self.assertEqual(entry.data["key"], "val")
        self.assertIsNone(entry.expires_at)

    def test_write_with_expiry(self):
        entry = self.bb.write(
            category=EntryCategory.RESERVATION,
            scope=EntryScope.SUBTASK,
            owner_agent="spatial",
            created_at=50,
            data={},
            expires_at=200,
        )
        self.assertEqual(entry.expires_at, 200)

    def test_write_assigns_unique_ids(self):
        e1 = self.bb.write(
            category=EntryCategory.INTENTION,
            scope=EntryScope.GOAL,
            owner_agent="prod",
            created_at=0,
            data={},
        )
        e2 = self.bb.write(
            category=EntryCategory.INTENTION,
            scope=EntryScope.GOAL,
            owner_agent="prod",
            created_at=0,
            data={},
        )
        self.assertNotEqual(e1.id, e2.id)

    def test_len_increases_on_write(self):
        self.assertEqual(len(self.bb), 0)
        self.bb.write(EntryCategory.OBSERVATION, EntryScope.GOAL, "a", 0, {})
        self.assertEqual(len(self.bb), 1)
        self.bb.write(EntryCategory.OBSERVATION, EntryScope.GOAL, "b", 0, {})
        self.assertEqual(len(self.bb), 2)


class TestBlackboardRead(unittest.TestCase):
    def setUp(self):
        self.bb = Blackboard()
        self.bb.write(EntryCategory.OBSERVATION, EntryScope.GOAL,    "nav",  10, {"x": 1})
        self.bb.write(EntryCategory.INTENTION,   EntryScope.SUBTASK, "prod", 20, {"x": 2})
        self.bb.write(EntryCategory.RESERVATION, EntryScope.GOAL,    "spatial", 30, {"x": 3})
        self.bb.write(EntryCategory.OBSERVATION, EntryScope.SUBTASK, "nav",  40, {"x": 4})

    def test_read_all(self):
        results = self.bb.read()
        self.assertEqual(len(results), 4)

    def test_filter_by_category(self):
        obs = self.bb.read(category=EntryCategory.OBSERVATION)
        self.assertEqual(len(obs), 2)
        self.assertTrue(all(e.category == EntryCategory.OBSERVATION for e in obs))

    def test_filter_by_scope_goal(self):
        goal_entries = self.bb.read(scope=EntryScope.GOAL)
        self.assertEqual(len(goal_entries), 2)
        self.assertTrue(all(e.scope == EntryScope.GOAL for e in goal_entries))

    def test_filter_by_scope_subtask(self):
        sub_entries = self.bb.read(scope=EntryScope.SUBTASK)
        self.assertEqual(len(sub_entries), 2)

    def test_filter_by_owner(self):
        nav_entries = self.bb.read(owner_agent="nav")
        self.assertEqual(len(nav_entries), 2)

    def test_combined_filter(self):
        results = self.bb.read(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.GOAL,
            owner_agent="nav",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].data["x"], 1)

    def test_get_by_id(self):
        entry = self.bb.write(
            EntryCategory.OBSERVATION, EntryScope.GOAL, "agent", 0, {"v": 99}
        )
        found = self.bb.get(entry.id)
        self.assertIsNotNone(found)
        self.assertEqual(found.data["v"], 99)

    def test_get_unknown_id_returns_none(self):
        self.assertIsNone(self.bb.get("nonexistent-id"))


class TestBlackboardClear(unittest.TestCase):
    def setUp(self):
        self.bb = Blackboard()
        self.bb.write(EntryCategory.OBSERVATION, EntryScope.GOAL,    "a", 0, {})
        self.bb.write(EntryCategory.OBSERVATION, EntryScope.GOAL,    "b", 0, {})
        self.bb.write(EntryCategory.INTENTION,   EntryScope.SUBTASK, "c", 0, {})
        self.bb.write(EntryCategory.RESERVATION, EntryScope.SUBTASK, "d", 0, {})

    def test_clear_subtask_scope_removes_only_subtask(self):
        removed = self.bb.clear_scope(EntryScope.SUBTASK)
        self.assertEqual(removed, 2)
        remaining = self.bb.read()
        self.assertEqual(len(remaining), 2)
        self.assertTrue(all(e.scope == EntryScope.GOAL for e in remaining))

    def test_goal_scoped_entries_survive_subtask_clear(self):
        self.bb.clear_scope(EntryScope.SUBTASK)
        goal_entries = self.bb.read(scope=EntryScope.GOAL)
        self.assertEqual(len(goal_entries), 2)

    def test_clear_all(self):
        removed = self.bb.clear_all()
        self.assertEqual(removed, 4)
        self.assertEqual(len(self.bb), 0)
        self.assertEqual(self.bb.read(), [])

    def test_clear_all_idempotent(self):
        self.bb.clear_all()
        removed_again = self.bb.clear_all()
        self.assertEqual(removed_again, 0)


class TestBlackboardExpiry(unittest.TestCase):
    def setUp(self):
        self.bb = Blackboard()

    def test_expired_entry_not_returned_by_read(self):
        self.bb.write(
            EntryCategory.OBSERVATION, EntryScope.GOAL, "agent", 0, {"v": 1},
            expires_at=100,
        )
        # Before expiry
        results = self.bb.read(current_tick=99)
        self.assertEqual(len(results), 1)
        # At and after expiry
        results = self.bb.read(current_tick=100)
        self.assertEqual(len(results), 0)
        results = self.bb.read(current_tick=200)
        self.assertEqual(len(results), 0)

    def test_non_expiring_entry_always_returned(self):
        self.bb.write(
            EntryCategory.OBSERVATION, EntryScope.GOAL, "agent", 0, {},
            expires_at=None,
        )
        self.assertEqual(len(self.bb.read(current_tick=9999)), 1)

    def test_prune_expired_removes_correct_entries(self):
        self.bb.write(
            EntryCategory.OBSERVATION, EntryScope.GOAL, "a", 0, {}, expires_at=50
        )
        self.bb.write(
            EntryCategory.OBSERVATION, EntryScope.GOAL, "b", 0, {}, expires_at=200
        )
        self.bb.write(
            EntryCategory.OBSERVATION, EntryScope.GOAL, "c", 0, {}
        )
        pruned = self.bb.prune_expired(current_tick=100)
        self.assertEqual(pruned, 1)
        self.assertEqual(len(self.bb), 2)

    def test_expired_entry_excluded_from_snapshot(self):
        self.bb.write(
            EntryCategory.OBSERVATION, EntryScope.GOAL, "agent", 0, {"v": 42},
            expires_at=50,
        )
        snap = self.bb.snapshot(current_tick=100)
        self.assertEqual(len(snap), 0)


class TestBlackboardSnapshot(unittest.TestCase):
    def setUp(self):
        self.bb = Blackboard()

    def test_snapshot_contains_all_live_entries(self):
        e1 = self.bb.write(EntryCategory.OBSERVATION, EntryScope.GOAL, "a", 10, {"k": "v"})
        e2 = self.bb.write(EntryCategory.INTENTION,   EntryScope.SUBTASK, "b", 20, {"k": 2})
        snap = self.bb.snapshot()
        self.assertIn(e1.id, snap)
        self.assertIn(e2.id, snap)

    def test_snapshot_entry_structure(self):
        entry = self.bb.write(
            EntryCategory.RESERVATION, EntryScope.GOAL, "spatial", 99,
            {"tile": (5, 5)},
        )
        snap = self.bb.snapshot()
        s = snap[entry.id]
        self.assertEqual(s["id"], entry.id)
        self.assertEqual(s["category"], "RESERVATION")
        self.assertEqual(s["scope"], "GOAL")
        self.assertEqual(s["owner_agent"], "spatial")
        self.assertEqual(s["created_at"], 99)
        self.assertIsNone(s["expires_at"])

    def test_snapshot_is_shallow_copy(self):
        e = self.bb.write(EntryCategory.OBSERVATION, EntryScope.GOAL, "x", 0, {"z": 1})
        snap = self.bb.snapshot()
        snap[e.id]["data"]["z"] = 999   # mutate snapshot
        # Original should be unaffected (data is a new dict in the snapshot)
        self.assertEqual(e.data["z"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
