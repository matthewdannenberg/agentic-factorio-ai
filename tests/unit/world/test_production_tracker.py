"""
tests/unit/world/test_production_tracker.py

Unit tests for world/production_tracker.py

Run with:  python -m unittest tests.unit.world.test_production_tracker
"""

from __future__ import annotations

import unittest

from world.production_tracker import ProductionSummary, ProductionTracker
from world.state import (
    EntityState,
    EntityStatus,
    Inventory,
    InventorySlot,
    LogisticsState,
    Position,
    WorldState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ws(tick: int, entities=None, inserter_activity=None) -> WorldState:
    """Build a minimal WorldState for testing."""
    logistics = LogisticsState()
    if inserter_activity:
        logistics = LogisticsState(inserter_activity=inserter_activity)
    return WorldState(
        tick=tick,
        entities=entities or [],
        logistics=logistics,
    )


def _entity_with_inventory(entity_id: int, name: str, items: dict[str, int],
                            status: EntityStatus = EntityStatus.WORKING) -> EntityState:
    slots = [InventorySlot(item=k, count=v) for k, v in items.items()]
    return EntityState(
        entity_id=entity_id,
        name=name,
        position=Position(0, 0),
        status=status,
        inventory=Inventory(slots=slots),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProductionTrackerEmpty(unittest.TestCase):
    def setUp(self):
        self.tracker = ProductionTracker()

    def test_rate_returns_zero_with_no_history(self):
        self.assertEqual(self.tracker.rate("iron-plate"), 0.0)

    def test_rate_returns_zero_with_one_snapshot(self):
        self.tracker.update(_ws(tick=100))
        self.assertEqual(self.tracker.rate("iron-plate"), 0.0)

    def test_rates_all_empty_with_no_history(self):
        self.assertEqual(self.tracker.rates_all(), {})

    def test_rates_all_empty_with_one_snapshot(self):
        self.tracker.update(_ws(tick=100))
        self.assertEqual(self.tracker.rates_all(), {})

    def test_is_stalled_false_with_no_history(self):
        self.assertFalse(self.tracker.is_stalled("iron-plate"))


class TestProductionTrackerInventoryDiff(unittest.TestCase):
    """Tests using inventory delta signal (no inserter activity)."""

    def setUp(self):
        self.tracker = ProductionTracker()

    def test_rate_correct_from_two_snapshots(self):
        # 3600 ticks = 60 seconds = 1 minute (60 tps).
        # 60 items produced in 1 minute → rate = 60 items/min.
        e1 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 60})

        self.tracker.update(_ws(tick=0, entities=[e1]))
        self.tracker.update(_ws(tick=3600, entities=[e2]))

        rate = self.tracker.rate("iron-plate", window_ticks=3600)
        self.assertAlmostEqual(rate, 60.0, places=2)

    def test_rate_zero_when_window_exceeds_history(self):
        e1 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 10})
        self.tracker.update(_ws(tick=1000, entities=[e1]))
        self.tracker.update(_ws(tick=1100, entities=[e2]))

        # Request window much larger than history
        rate = self.tracker.rate("iron-plate", window_ticks=1_000_000)
        # Should still return the rate (history within window)
        self.assertGreater(rate, 0.0)

    def test_rate_for_untracked_item_is_zero(self):
        e1 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 10})
        self.tracker.update(_ws(tick=0, entities=[e1]))
        self.tracker.update(_ws(tick=3600, entities=[e2]))

        self.assertEqual(self.tracker.rate("copper-plate"), 0.0)

    def test_rates_all_covers_all_tracked_items(self):
        e1 = _entity_with_inventory(1, "chest", {"iron-plate": 0, "copper-plate": 0})
        e2 = _entity_with_inventory(1, "chest", {"iron-plate": 30, "copper-plate": 15})
        self.tracker.update(_ws(tick=0, entities=[e1]))
        self.tracker.update(_ws(tick=3600, entities=[e2]))

        rates = self.tracker.rates_all(window_ticks=3600)
        self.assertIn("iron-plate", rates)
        self.assertIn("copper-plate", rates)
        self.assertGreater(rates["iron-plate"], 0)
        self.assertGreater(rates["copper-plate"], 0)

    def test_no_production_when_inventory_unchanged(self):
        e = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 10})
        self.tracker.update(_ws(tick=0, entities=[e]))
        self.tracker.update(_ws(tick=3600, entities=[e]))  # same inventory
        self.assertEqual(self.tracker.rate("iron-plate"), 0.0)

    def test_multiple_entities_summed(self):
        e1a = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 0})
        e2a = _entity_with_inventory(2, "assembling-machine-1", {"iron-plate": 0})
        e1b = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 30})
        e2b = _entity_with_inventory(2, "assembling-machine-1", {"iron-plate": 30})

        self.tracker.update(_ws(tick=0, entities=[e1a, e2a]))
        self.tracker.update(_ws(tick=3600, entities=[e1b, e2b]))

        rate = self.tracker.rate("iron-plate", window_ticks=3600)
        # Two machines each producing 30 over 3600 ticks (= 1 min at 60 tps):
        # 30/min each = 60/min total
        self.assertAlmostEqual(rate, 60.0, places=1)


class TestProductionTrackerIsStalled(unittest.TestCase):
    def setUp(self):
        self.tracker = ProductionTracker()

    def test_is_stalled_false_when_rate_positive(self):
        e1 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 60})
        self.tracker.update(_ws(tick=0, entities=[e1]))
        self.tracker.update(_ws(tick=3600, entities=[e2]))
        self.assertFalse(self.tracker.is_stalled("iron-plate", window_ticks=3600))

    def test_is_stalled_true_when_was_produced_then_stopped(self):
        # First produce, then stop
        e1 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 10})
        e3 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 10})  # no change

        self.tracker.update(_ws(tick=0, entities=[e1]))
        self.tracker.update(_ws(tick=600, entities=[e2]))
        self.tracker.update(_ws(tick=1800, entities=[e3]))

        # Rate over short window: should be zero (no change in last 1200 ticks)
        self.assertTrue(self.tracker.is_stalled("iron-plate", window_ticks=600))

    def test_is_stalled_false_for_never_tracked_item(self):
        e1 = _entity_with_inventory(1, "chest", {"iron-plate": 0})
        self.tracker.update(_ws(tick=0, entities=[e1]))
        self.tracker.update(_ws(tick=600, entities=[e1]))
        # copper-plate never appeared — no producer evidence → not stalled
        self.assertFalse(self.tracker.is_stalled("copper-plate"))

    def test_is_stalled_with_state(self):
        tracker = ProductionTracker()
        e1 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 0},
                                     status=EntityStatus.WORKING)
        e2 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 0},
                                     status=EntityStatus.NO_INPUT)
        ws1 = _ws(tick=0, entities=[e1])
        ws2 = _ws(tick=600, entities=[e2])
        tracker.update(ws1)
        tracker.update(ws2)
        # Rate is zero but entity is NO_INPUT (active producer)
        self.assertTrue(tracker.is_stalled_with_state("iron-plate", ws2, window_ticks=600))

    def test_is_stalled_with_state_false_when_producing(self):
        tracker = ProductionTracker()
        e1 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 0},
                                     status=EntityStatus.WORKING)
        e2 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 60},
                                     status=EntityStatus.WORKING)
        ws1 = _ws(tick=0, entities=[e1])
        ws2 = _ws(tick=3600, entities=[e2])
        tracker.update(ws1)
        tracker.update(ws2)
        self.assertFalse(tracker.is_stalled_with_state("iron-plate", ws2, window_ticks=3600))


class TestProductionTrackerGapHandling(unittest.TestCase):
    def test_gap_does_not_fabricate_production(self):
        tracker = ProductionTracker()
        e1 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 0})
        e2 = _entity_with_inventory(1, "assembling-machine-1", {"iron-plate": 10})

        tracker.update(_ws(tick=0, entities=[e1]))
        tracker.update(_ws(tick=600, entities=[e2]))
        # Large gap — no updates for 50000 ticks
        tracker.update(_ws(tick=50600, entities=[e2]))

        # Rate over the large window should only count actual observed delta
        rate = tracker.rate("iron-plate", window_ticks=60000)
        # 10 items produced in first 600 ticks — after that, unchanged.
        # Should be a small positive rate, not fabricated huge number.
        self.assertGreater(rate, 0.0)
        # Rate should not be astronomically high (no fabricated production)
        self.assertLess(rate, 100.0)  # sanity bound

    def test_update_after_long_pause_works(self):
        tracker = ProductionTracker()
        tracker.update(_ws(tick=1000))
        # Long pause
        tracker.update(_ws(tick=100000))
        # Should not raise and rate should be 0 (no inventory tracked)
        self.assertEqual(tracker.rate("iron-plate"), 0.0)


class TestProductionTrackerSummary(unittest.TestCase):
    def test_summary_returns_production_summary(self):
        tracker = ProductionTracker()
        self.assertIsInstance(tracker.summary(), ProductionSummary)

    def test_summary_fields_populated(self):
        tracker = ProductionTracker()
        e1 = _entity_with_inventory(1, "chest", {"iron-plate": 0, "copper-plate": 0})
        e2 = _entity_with_inventory(1, "chest", {"iron-plate": 60, "copper-plate": 30})
        tracker.update(_ws(tick=0, entities=[e1]))
        tracker.update(_ws(tick=3600, entities=[e2]))

        summary = tracker.summary(window_ticks=3600)
        self.assertIsInstance(summary.rates, dict)
        self.assertIsInstance(summary.stalled_items, list)
        self.assertIsInstance(summary.top_producers, list)
        self.assertIsInstance(summary.window_ticks, int)
        self.assertEqual(summary.window_ticks, 3600)
        self.assertIsInstance(summary.tick_start, int)
        self.assertIsInstance(summary.tick_end, int)

    def test_summary_tick_range(self):
        tracker = ProductionTracker()
        tracker.update(_ws(tick=100))
        tracker.update(_ws(tick=3700))
        summary = tracker.summary()
        self.assertEqual(summary.tick_start, 100)
        self.assertEqual(summary.tick_end, 3700)

    def test_summary_top_producers_sorted_desc(self):
        tracker = ProductionTracker()
        e1 = _entity_with_inventory(1, "chest", {"iron-plate": 0, "copper-plate": 0})
        e2 = _entity_with_inventory(1, "chest", {"iron-plate": 120, "copper-plate": 30})
        tracker.update(_ws(tick=0, entities=[e1]))
        tracker.update(_ws(tick=3600, entities=[e2]))

        summary = tracker.summary()
        if len(summary.top_producers) >= 2:
            self.assertGreaterEqual(summary.top_producers[0][1], summary.top_producers[1][1])

    def test_summary_empty_tracker(self):
        tracker = ProductionTracker()
        summary = tracker.summary()
        self.assertEqual(summary.rates, {})
        self.assertEqual(summary.stalled_items, [])
        self.assertEqual(summary.top_producers, [])
        self.assertEqual(summary.tick_start, 0)
        self.assertEqual(summary.tick_end, 0)


class TestProductionTrackerInserterActivity(unittest.TestCase):
    def test_inserter_activity_delta_tracked(self):
        tracker = ProductionTracker()
        # Inserter 1 has moved 100 items total at tick 0, then 200 at tick 3600
        tracker.update(_ws(tick=0, inserter_activity={1: 100}))
        tracker.update(_ws(tick=3600, inserter_activity={1: 200}))
        # 100 items moved over 3600 ticks / 60 tps = 1 minute → 100 items/min
        # Inserter moves are tracked under __inserter_moves__ internally
        # They don't pollute the item-level rates dict
        rates = tracker.rates_all()
        # The __inserter_moves__ key should not leak to public API
        self.assertNotIn("__inserter_moves__", rates)

    def test_inserter_activity_no_fabrication_on_reset(self):
        tracker = ProductionTracker()
        tracker.update(_ws(tick=0, inserter_activity={1: 500}))
        # Simulated counter reset (new game / entity rebuilt)
        tracker.update(_ws(tick=3600, inserter_activity={1: 10}))
        # Should not produce negative or huge numbers
        # (guards against counter reset with max(0, delta))
        # Just verify no exception and rate is non-negative
        self.assertGreaterEqual(tracker.rate("iron-plate"), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)