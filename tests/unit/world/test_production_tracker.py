"""
tests/unit/world/test_production_tracker.py

Unit tests for world/production_tracker.py

WorldState/WorldQuery construction is handled via fixtures helpers so that a
WorldState signature change does not require edits here.

Run with:  python -m unittest tests.unit.world.test_production_tracker
"""

from __future__ import annotations

import unittest

from world.production_tracker import ProductionSummary, ProductionTracker
from world.state import EntityStatus
from tests.fixtures import make_world_query, make_inventory_entity


class TestProductionTrackerEmpty(unittest.TestCase):
    def setUp(self):
        self.tracker = ProductionTracker()

    def test_rate_returns_zero_with_no_history(self):
        self.assertEqual(self.tracker.rate("iron-plate"), 0.0)

    def test_rate_returns_zero_with_one_snapshot(self):
        self.tracker.update(make_world_query(tick=100))
        self.assertEqual(self.tracker.rate("iron-plate"), 0.0)

    def test_rates_all_empty_with_no_history(self):
        self.assertEqual(self.tracker.rates_all(), {})

    def test_rates_all_empty_with_one_snapshot(self):
        self.tracker.update(make_world_query(tick=100))
        self.assertEqual(self.tracker.rates_all(), {})

    def test_is_stalled_false_with_no_history(self):
        self.assertFalse(self.tracker.is_stalled("iron-plate"))


class TestProductionTrackerInventoryDiff(unittest.TestCase):
    def setUp(self):
        self.tracker = ProductionTracker()

    def test_rate_correct_from_two_snapshots(self):
        e1 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 60})
        self.tracker.update(make_world_query(tick=0, entities=[e1]))
        self.tracker.update(make_world_query(tick=3600, entities=[e2]))
        self.assertAlmostEqual(self.tracker.rate("iron-plate", window_ticks=3600), 60.0, places=2)

    def test_rate_zero_when_window_exceeds_history(self):
        e1 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 10})
        self.tracker.update(make_world_query(tick=1000, entities=[e1]))
        self.tracker.update(make_world_query(tick=1100, entities=[e2]))
        self.assertGreater(self.tracker.rate("iron-plate", window_ticks=1_000_000), 0.0)

    def test_rate_for_untracked_item_is_zero(self):
        e1 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 10})
        self.tracker.update(make_world_query(tick=0, entities=[e1]))
        self.tracker.update(make_world_query(tick=3600, entities=[e2]))
        self.assertEqual(self.tracker.rate("copper-plate"), 0.0)

    def test_rates_all_covers_all_tracked_items(self):
        e1 = make_inventory_entity(1, "chest", {"iron-plate": 0, "copper-plate": 0})
        e2 = make_inventory_entity(1, "chest", {"iron-plate": 30, "copper-plate": 15})
        self.tracker.update(make_world_query(tick=0, entities=[e1]))
        self.tracker.update(make_world_query(tick=3600, entities=[e2]))
        rates = self.tracker.rates_all(window_ticks=3600)
        self.assertIn("iron-plate", rates)
        self.assertIn("copper-plate", rates)

    def test_no_production_when_inventory_unchanged(self):
        e = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 10})
        self.tracker.update(make_world_query(tick=0, entities=[e]))
        self.tracker.update(make_world_query(tick=3600, entities=[e]))
        self.assertEqual(self.tracker.rate("iron-plate"), 0.0)

    def test_multiple_entities_summed(self):
        e1a = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 0})
        e2a = make_inventory_entity(2, "assembling-machine-1", {"iron-plate": 0})
        e1b = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 30})
        e2b = make_inventory_entity(2, "assembling-machine-1", {"iron-plate": 30})
        self.tracker.update(make_world_query(tick=0, entities=[e1a, e2a]))
        self.tracker.update(make_world_query(tick=3600, entities=[e1b, e2b]))
        self.assertAlmostEqual(self.tracker.rate("iron-plate", window_ticks=3600), 60.0, places=1)


class TestProductionTrackerIsStalled(unittest.TestCase):
    def setUp(self):
        self.tracker = ProductionTracker()

    def test_is_stalled_false_when_rate_positive(self):
        e1 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 60})
        self.tracker.update(make_world_query(tick=0, entities=[e1]))
        self.tracker.update(make_world_query(tick=3600, entities=[e2]))
        self.assertFalse(self.tracker.is_stalled("iron-plate", window_ticks=3600))

    def test_is_stalled_true_when_was_produced_then_stopped(self):
        e1 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 10})
        e3 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 10})
        self.tracker.update(make_world_query(tick=0, entities=[e1]))
        self.tracker.update(make_world_query(tick=600, entities=[e2]))
        self.tracker.update(make_world_query(tick=1800, entities=[e3]))
        self.assertTrue(self.tracker.is_stalled("iron-plate", window_ticks=600))

    def test_is_stalled_false_for_never_tracked_item(self):
        e = make_inventory_entity(1, "chest", {"iron-plate": 0})
        self.tracker.update(make_world_query(tick=0, entities=[e]))
        self.tracker.update(make_world_query(tick=600, entities=[e]))
        self.assertFalse(self.tracker.is_stalled("copper-plate"))

    def test_is_stalled_with_query(self):
        tracker = ProductionTracker()
        e1 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 0},
                                    status=EntityStatus.WORKING)
        e2 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 0},
                                    status=EntityStatus.NO_INPUT)
        tracker.update(make_world_query(tick=0, entities=[e1]))
        wq2 = make_world_query(tick=600, entities=[e2])
        tracker.update(wq2)
        self.assertTrue(tracker.is_stalled_with_query("iron-plate", wq2, window_ticks=600))

    def test_is_stalled_with_query_false_when_producing(self):
        tracker = ProductionTracker()
        e1 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 60})
        tracker.update(make_world_query(tick=0, entities=[e1]))
        wq2 = make_world_query(tick=3600, entities=[e2])
        tracker.update(wq2)
        self.assertFalse(tracker.is_stalled_with_query("iron-plate", wq2, window_ticks=3600))


class TestProductionTrackerGapHandling(unittest.TestCase):
    def test_gap_does_not_fabricate_production(self):
        tracker = ProductionTracker()
        e1 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = make_inventory_entity(1, "assembling-machine-1", {"iron-plate": 10})
        tracker.update(make_world_query(tick=0, entities=[e1]))
        tracker.update(make_world_query(tick=600, entities=[e2]))
        tracker.update(make_world_query(tick=50600, entities=[e2]))
        rate = tracker.rate("iron-plate", window_ticks=60000)
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 100.0)

    def test_update_after_long_pause_works(self):
        tracker = ProductionTracker()
        tracker.update(make_world_query(tick=1000))
        tracker.update(make_world_query(tick=100000))
        self.assertEqual(tracker.rate("iron-plate"), 0.0)


class TestProductionTrackerSummary(unittest.TestCase):
    def test_summary_returns_production_summary(self):
        self.assertIsInstance(ProductionTracker().summary(), ProductionSummary)

    def test_summary_fields_populated(self):
        tracker = ProductionTracker()
        e1 = make_inventory_entity(1, "chest", {"iron-plate": 0, "copper-plate": 0})
        e2 = make_inventory_entity(1, "chest", {"iron-plate": 60, "copper-plate": 30})
        tracker.update(make_world_query(tick=0, entities=[e1]))
        tracker.update(make_world_query(tick=3600, entities=[e2]))
        summary = tracker.summary(window_ticks=3600)
        self.assertIsInstance(summary.rates, dict)
        self.assertIsInstance(summary.stalled_items, list)
        self.assertIsInstance(summary.top_producers, list)
        self.assertEqual(summary.window_ticks, 3600)

    def test_summary_tick_range(self):
        tracker = ProductionTracker()
        tracker.update(make_world_query(tick=100))
        tracker.update(make_world_query(tick=3700))
        summary = tracker.summary()
        self.assertEqual(summary.tick_start, 100)
        self.assertEqual(summary.tick_end, 3700)

    def test_summary_empty_tracker(self):
        tracker = ProductionTracker()
        summary = tracker.summary()
        self.assertEqual(summary.rates, {})
        self.assertEqual(summary.stalled_items, [])
        self.assertEqual(summary.top_producers, [])


class TestProductionTrackerInserterActivity(unittest.TestCase):
    def test_inserter_activity_delta_tracked(self):
        tracker = ProductionTracker()
        tracker.update(make_world_query(tick=0, inserter_activity={1: 100}))
        tracker.update(make_world_query(tick=3600, inserter_activity={1: 200}))
        self.assertNotIn("__inserter_moves__", tracker.rates_all())

    def test_inserter_activity_no_fabrication_on_reset(self):
        tracker = ProductionTracker()
        tracker.update(make_world_query(tick=0, inserter_activity={1: 500}))
        tracker.update(make_world_query(tick=3600, inserter_activity={1: 10}))
        self.assertGreaterEqual(tracker.rate("iron-plate"), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)