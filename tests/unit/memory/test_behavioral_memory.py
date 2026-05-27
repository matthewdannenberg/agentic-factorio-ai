"""
tests/unit/agent/test_behavioral_memory.py

Tests for agent/memory/behavioral.py

Run with:  python -m pytest tests/unit/agent/test_behavioral_memory.py -v
       or:  python -m unittest tests.unit.agent.test_behavioral_memory

Uses :memory: SQLite databases so no files are written during testing.
"""

from __future__ import annotations

import unittest

from agent.memory.behavioral import (
    GoalOutcome,
    PerformanceStats,
    SQLiteBehavioralMemory,
    StrategyRecord,
)
from agent.self_model import SelfModel, SelfModelNode, NodeType, NodeStatus, BoundingBox
from world.state import Position


def _make_mem() -> SQLiteBehavioralMemory:
    return SQLiteBehavioralMemory(db_path=":memory:")


def _outcome(success: bool = True, reward: float = 1.0, ticks: int = 600) -> GoalOutcome:
    return GoalOutcome(success=success, reward=reward, ticks_elapsed=ticks)


def _node(label="test") -> SelfModelNode:
    return SelfModelNode(
        type=NodeType.PRODUCTION_LINE,
        status=NodeStatus.ACTIVE,
        bounding_box=BoundingBox(Position(0, 0), Position(10, 10)),
        label=label,
        throughput={"iron-plate": 30.0},
        created_at=0,
    )


class TestRecordAndQuery(unittest.TestCase):
    def setUp(self):
        self.mem = _make_mem()

    def tearDown(self):
        self.mem.close()

    def test_record_and_retrieve(self):
        self.mem.record_outcome(
            "production",
            {"context": "early_game"},
            _outcome(success=True, reward=0.9, ticks=500),
            ticks_elapsed=500,
        )
        records = self.mem.query_strategies("production", {})
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r.goal_type, "production")
        self.assertTrue(r.outcome.success)
        self.assertAlmostEqual(r.outcome.reward, 0.9)

    def test_multiple_records_same_goal_type(self):
        for i in range(3):
            self.mem.record_outcome(
                "exploration", {}, _outcome(success=True, ticks=100 * i), ticks_elapsed=100 * i
            )
        records = self.mem.query_strategies("exploration", {})
        self.assertEqual(len(records), 3)

    def test_query_different_goal_type_returns_empty(self):
        self.mem.record_outcome("production", {}, _outcome(), ticks_elapsed=0)
        records = self.mem.query_strategies("exploration", {})
        self.assertEqual(records, [])

    def test_context_summary_preserved(self):
        ctx = {"tick": 1200, "tech": "automation", "items": ["iron-plate"]}
        self.mem.record_outcome("research", ctx, _outcome(), ticks_elapsed=0)
        records = self.mem.query_strategies("research", {})
        self.assertEqual(records[0].context_summary["tick"], 1200)
        self.assertEqual(records[0].context_summary["tech"], "automation")

    def test_failure_outcome_preserved(self):
        self.mem.record_outcome(
            "production", {}, _outcome(success=False, reward=0.0), ticks_elapsed=999
        )
        records = self.mem.query_strategies("production", {})
        self.assertFalse(records[0].outcome.success)


class TestPerformanceHistory(unittest.TestCase):
    def setUp(self):
        self.mem = _make_mem()

    def tearDown(self):
        self.mem.close()

    def test_empty_history_returns_safe_defaults(self):
        stats = self.mem.get_performance_history("production")
        self.assertEqual(stats.goal_type, "production")
        self.assertEqual(stats.total_attempts, 0)
        self.assertAlmostEqual(stats.success_rate, 0.0)
        self.assertAlmostEqual(stats.mean_ticks, 0.0)
        self.assertAlmostEqual(stats.mean_reward, 0.0)

    def test_one_success_gives_100_percent(self):
        self.mem.record_outcome("production", {}, _outcome(True, 1.0, 300), 300)
        stats = self.mem.get_performance_history("production")
        self.assertEqual(stats.total_attempts, 1)
        self.assertAlmostEqual(stats.success_rate, 1.0)
        self.assertAlmostEqual(stats.mean_ticks, 300.0)
        self.assertAlmostEqual(stats.mean_reward, 1.0)

    def test_mixed_outcomes_aggregated(self):
        self.mem.record_outcome("prod", {}, _outcome(True, 1.0, 400), 400)
        self.mem.record_outcome("prod", {}, _outcome(False, 0.0, 200), 200)
        stats = self.mem.get_performance_history("prod")
        self.assertEqual(stats.total_attempts, 2)
        self.assertAlmostEqual(stats.success_rate, 0.5)
        self.assertAlmostEqual(stats.mean_ticks, 300.0)
        self.assertAlmostEqual(stats.mean_reward, 0.5)

    def test_performance_separated_by_goal_type(self):
        self.mem.record_outcome("production", {}, _outcome(True, 1.0, 100), 100)
        self.mem.record_outcome("exploration", {}, _outcome(False, 0.0, 200), 200)
        p_stats = self.mem.get_performance_history("production")
        e_stats = self.mem.get_performance_history("exploration")
        self.assertAlmostEqual(p_stats.success_rate, 1.0)
        self.assertAlmostEqual(e_stats.success_rate, 0.0)


class TestPersistenceAcrossRestart(unittest.TestCase):
    """
    Simulates close-and-reopen by using a temp file rather than :memory:.
    We use a fresh :memory: db for each phase but verify the SQL schema is
    idempotent (CREATE TABLE IF NOT EXISTS).
    """

    def test_schema_creation_idempotent(self):
        """Opening the same :memory: db twice doesn't blow up."""
        mem = SQLiteBehavioralMemory(db_path=":memory:")
        mem.record_outcome("production", {}, _outcome(), ticks_elapsed=0)
        mem.close()
        # Re-using same object after close (reconnects on next call)
        mem._conn = None
        records = mem.query_strategies("production", {})
        # :memory: doesn't persist after close; we just verify no crash
        mem.close()

    def test_file_based_persistence(self):
        """Records survive a close/reopen cycle when using a real file."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            mem1 = SQLiteBehavioralMemory(db_path=db_path)
            mem1.record_outcome("production", {"k": "v"}, _outcome(True, 0.8, 500), 500)
            mem1.close()

            mem2 = SQLiteBehavioralMemory(db_path=db_path)
            records = mem2.query_strategies("production", {})
            mem2.close()

            self.assertEqual(len(records), 1)
            self.assertAlmostEqual(records[0].outcome.reward, 0.8)
            self.assertEqual(records[0].context_summary["k"], "v")
        finally:
            os.unlink(db_path)


class TestSpatialPatternRecording(unittest.TestCase):
    def setUp(self):
        self.mem = _make_mem()

    def tearDown(self):
        self.mem.close()

    def test_record_spatial_pattern_does_not_raise(self):
        sm = SelfModel()
        n1 = _node("iron prod")
        n2 = _node("belt run")
        sm.add_node(n1)
        sm.add_node(n2)
        from agent.self_model import EdgeType
        sm.add_edge(n1.id, n2.id, EdgeType.FEEDS_INTO)
        # Should not raise
        self.mem.record_spatial_pattern(sm, label="iron-belt-pattern")

    def test_record_empty_subgraph(self):
        sm = SelfModel()
        self.mem.record_spatial_pattern(sm, label="empty")


class TestContextManager(unittest.TestCase):
    def test_with_statement_closes_on_exit(self):
        with SQLiteBehavioralMemory(db_path=":memory:") as mem:
            mem.record_outcome("production", {}, _outcome(), ticks_elapsed=0)
            self.assertIsNotNone(mem._conn)
        # After the with block, connection should be closed
        self.assertIsNone(mem._conn)

    def test_with_statement_closes_on_exception(self):
        mem = None
        try:
            with SQLiteBehavioralMemory(db_path=":memory:") as m:
                mem = m
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass
        self.assertIsNone(mem._conn)


if __name__ == "__main__":
    unittest.main(verbosity=2)