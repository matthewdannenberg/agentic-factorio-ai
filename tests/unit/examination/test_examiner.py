"""
tests/unit/agent/test_examiner.py

Tests for agent/examiner/*.py

Run with:  python tests/unit/agent/test_examiner.py
"""

from __future__ import annotations

import unittest

from agent.examiner.audit_report import (
    AuditMode, AnomalySeverity,
    StarvedEntity, Anomaly, BoundingBox, BlueprintCandidate,
    DamagedEntityRecord, DestroyedEntityRecord, AuditReport,
)


class TestAuditReport(unittest.TestCase):
    def _mech(self, tick=0) -> AuditReport:
        return AuditReport(
            tick=tick,
            mode=AuditMode.MECHANICAL,
            observation_ticks=600,
            starved_entities=[StarvedEntity(1, "machine-a", 0, 0, ["iron-plate"])],
            power_headroom_kw=-50.0,
            power_satisfaction=0.8,
            production_rates={"iron-plate": 30.0},
            anomalies=[
                Anomaly(AnomalySeverity.CRITICAL, "brownout"),
                Anomaly(AnomalySeverity.INFO, "low iron"),
            ],
        )

    def test_is_brownout(self):
        self.assertTrue(self._mech().is_brownout)

    def test_not_brownout(self):
        r = AuditReport(tick=0, mode=AuditMode.MECHANICAL, power_satisfaction=1.0)
        self.assertFalse(r.is_brownout)

    def test_has_critical(self):
        self.assertTrue(self._mech().has_critical_anomalies)

    def test_no_critical(self):
        r = AuditReport(tick=0, mode=AuditMode.MECHANICAL,
                        anomalies=[Anomaly(AnomalySeverity.INFO, "fine")])
        self.assertFalse(r.has_critical_anomalies)

    def test_production_rate(self):
        r = self._mech()
        self.assertEqual(r.production_rate("iron-plate"), 30.0)
        self.assertEqual(r.production_rate("steel-plate"), 0.0)

    def test_summary_has_mode(self):
        self.assertIn("MECHANICAL", self._mech().summary())

    def test_merge_sums_ticks(self):
        r1 = AuditReport(tick=0, mode=AuditMode.MECHANICAL, observation_ticks=300,
                         production_rates={"iron-plate": 20.0}, power_satisfaction=1.0)
        r2 = AuditReport(tick=300, mode=AuditMode.MECHANICAL, observation_ticks=300,
                         production_rates={"iron-plate": 40.0}, power_satisfaction=0.9)
        merged = r1.merge(r2)
        self.assertEqual(merged.observation_ticks, 600)
        self.assertAlmostEqual(merged.production_rate("iron-plate"), 30.0)
        self.assertAlmostEqual(merged.power_satisfaction, 0.9)
        self.assertTrue(merged.accumulated)
        self.assertEqual(merged.tick, 0)

    def test_merge_deduplicates_entities(self):
        shared = StarvedEntity(1, "m", 0, 0, ["coal"])
        r1 = AuditReport(tick=0, mode=AuditMode.MECHANICAL, starved_entities=[shared])
        r2 = AuditReport(tick=60, mode=AuditMode.MECHANICAL, starved_entities=[shared])
        self.assertEqual(len(r1.merge(r2).starved_entities), 1)

    def test_rich_fields_none_in_mechanical(self):
        r = self._mech()
        self.assertIsNone(r.blueprint_candidates)
        self.assertIsNone(r.llm_observations)

    def test_rich_report(self):
        r = AuditReport(
            tick=0, mode=AuditMode.RICH,
            blueprint_candidates=[
                BlueprintCandidate(
                    region_description="smelting col",
                    bounds=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10),
                    performance_metric="60/min",
                    rationale="threshold met",
                )
            ],
            llm_observations="All good.",
        )
        self.assertEqual(len(r.blueprint_candidates), 1)
        self.assertEqual(r.llm_observations, "All good.")

    def test_merge_joins_llm_observations(self):
        r1 = AuditReport(tick=0, mode=AuditMode.RICH, llm_observations="First.")
        r2 = AuditReport(tick=0, mode=AuditMode.RICH, llm_observations="Second.")
        merged = r1.merge(r2)
        self.assertIn("First.", merged.llm_observations)
        self.assertIn("Second.", merged.llm_observations)


class TestBoundingBox(unittest.TestCase):
    def _bb(self) -> BoundingBox:
        return BoundingBox(min_x=0, min_y=0, max_x=20, max_y=10)

    def test_dimensions(self):
        bb = self._bb()
        self.assertEqual(bb.width, 20)
        self.assertEqual(bb.height, 10)
        self.assertEqual(bb.area, 200)

    def test_center(self):
        bb = self._bb()
        self.assertAlmostEqual(bb.center_x, 10.0)
        self.assertAlmostEqual(bb.center_y, 5.0)

    def test_contains(self):
        bb = self._bb()
        self.assertTrue(bb.contains(10, 5))
        self.assertTrue(bb.contains(0, 0))
        self.assertTrue(bb.contains(20, 10))
        self.assertFalse(bb.contains(21, 5))
        self.assertFalse(bb.contains(10, -1))

    def test_frozen(self):
        bb = self._bb()
        with self.assertRaises((AttributeError, TypeError)):
            bb.min_x = 99  # type: ignore

    def test_repr(self):
        self.assertIn("20×10", repr(self._bb()))

    def test_blueprint_candidate_uses_bounding_box(self):
        bc = BlueprintCandidate(
            region_description="copper smelting row",
            bounds=BoundingBox(min_x=-5, min_y=10, max_x=15, max_y=30),
            performance_metric="45 copper-plate/min",
            rationale="Above threshold for 5 min",
        )
        self.assertEqual(bc.bounds.width, 20)
        self.assertEqual(bc.bounds.height, 20)
        # No center_x/y/radius attributes on BlueprintCandidate itself
        self.assertFalse(hasattr(bc, "radius"))
        self.assertFalse(hasattr(bc, "center_x"))


class TestAuditReportDamage(unittest.TestCase):
    def test_damage_fields_empty_by_default(self):
        r = AuditReport(tick=0, mode=AuditMode.MECHANICAL)
        self.assertEqual(r.damaged_entities, [])
        self.assertEqual(r.destroyed_entities, [])
        self.assertFalse(r.has_structural_damage)

    def test_has_structural_damage_damaged(self):
        r = AuditReport(
            tick=0, mode=AuditMode.MECHANICAL,
            damaged_entities=[
                DamagedEntityRecord(1, "stone-wall", 50.0, 50.0, 0.3)
            ],
        )
        self.assertTrue(r.has_structural_damage)

    def test_has_structural_damage_destroyed(self):
        r = AuditReport(
            tick=0, mode=AuditMode.MECHANICAL,
            destroyed_entities=[
                DestroyedEntityRecord("gun-turret", 60.0, 60.0, 1750, "biter")
            ],
        )
        self.assertTrue(r.has_structural_damage)

    def test_summary_includes_damage_counts(self):
        r = AuditReport(
            tick=0, mode=AuditMode.MECHANICAL,
            damaged_entities=[DamagedEntityRecord(1, "wall", 0, 0, 0.5)],
            destroyed_entities=[DestroyedEntityRecord("turret", 0, 0, 100, "biter")],
        )
        s = r.summary()
        self.assertIn("damaged=1", s)
        self.assertIn("destroyed=1", s)

    def test_merge_unions_damaged_by_entity_id(self):
        r1 = AuditReport(tick=0, mode=AuditMode.MECHANICAL,
                         damaged_entities=[DamagedEntityRecord(1, "wall-a", 0, 0, 0.8)])
        r2 = AuditReport(tick=60, mode=AuditMode.MECHANICAL,
                         damaged_entities=[
                             DamagedEntityRecord(1, "wall-a", 0, 0, 0.6),  # same id
                             DamagedEntityRecord(2, "wall-b", 5, 5, 0.4),  # new
                         ])
        merged = r1.merge(r2)
        self.assertEqual(len(merged.damaged_entities), 2)

    def test_merge_unions_destroyed_deduplicates(self):
        rec = DestroyedEntityRecord("turret", 10.0, 10.0, 500, "biter")
        r1 = AuditReport(tick=0, mode=AuditMode.MECHANICAL, destroyed_entities=[rec])
        r2 = AuditReport(tick=60, mode=AuditMode.MECHANICAL, destroyed_entities=[rec])
        merged = r1.merge(r2)
        self.assertEqual(len(merged.destroyed_entities), 1)

    def test_merge_appends_distinct_destroyed(self):
        r1 = AuditReport(tick=0, mode=AuditMode.MECHANICAL,
                         destroyed_entities=[
                             DestroyedEntityRecord("turret", 10.0, 10.0, 500, "biter")
                         ])
        r2 = AuditReport(tick=60, mode=AuditMode.MECHANICAL,
                         destroyed_entities=[
                             DestroyedEntityRecord("wall", 20.0, 20.0, 560, "vehicle")
                         ])
        merged = r1.merge(r2)
        self.assertEqual(len(merged.destroyed_entities), 2)

    def test_destroyed_entity_record_vehicle_cause(self):
        # "vehicle" is a valid cause — not biter-gated
        r = DestroyedEntityRecord("wooden-chest", 5.0, 5.0, 300, "vehicle")
        self.assertEqual(r.cause, "vehicle")


if __name__ == "__main__":
    unittest.main(verbosity=2)