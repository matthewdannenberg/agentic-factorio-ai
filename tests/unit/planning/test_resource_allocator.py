"""
tests/unit/planning/test_resource_allocator.py

Tests for planning/resource_allocator.py

Run with:  python -m pytest tests/unit/planning/test_resource_allocator.py -v
       or: python tests/unit/planning/test_resource_allocator.py
"""

from __future__ import annotations

import unittest

from planning.goal import Priority
from planning.resource_allocator import ResourceAllocator


class TestResourceAllocatorPassthrough(unittest.TestCase):
    """Initial implementation is a pass-through — all requests granted."""

    def setUp(self):
        self.alloc = ResourceAllocator()

    def test_request_action_slot_all_priorities(self):
        for p in Priority:
            with self.subTest(priority=p):
                self.assertTrue(self.alloc.request_action_slot(p))

    def test_request_llm_call_all_priorities(self):
        for p in Priority:
            with self.subTest(priority=p):
                self.assertTrue(self.alloc.request_llm_call(p))

    def test_tick_does_not_raise(self):
        try:
            self.alloc.tick()
        except Exception as exc:
            self.fail(f"tick() raised unexpectedly: {exc}")

    def test_multiple_ticks_do_not_raise(self):
        for _ in range(100):
            self.alloc.tick()

    def test_request_after_tick_still_granted(self):
        self.alloc.tick()
        self.assertTrue(self.alloc.request_action_slot(Priority.NORMAL))
        self.assertTrue(self.alloc.request_llm_call(Priority.NORMAL))


class TestResourceAllocatorInterface(unittest.TestCase):
    """Verify the interface matches what callers expect."""

    def test_callable_with_background_priority(self):
        alloc = ResourceAllocator()
        self.assertTrue(alloc.request_action_slot(Priority.BACKGROUND))
        self.assertTrue(alloc.request_llm_call(Priority.BACKGROUND))

    def test_callable_with_emergency_priority(self):
        alloc = ResourceAllocator()
        self.assertTrue(alloc.request_action_slot(Priority.EMERGENCY))
        self.assertTrue(alloc.request_llm_call(Priority.EMERGENCY))

    def test_returns_bool_type(self):
        alloc = ResourceAllocator()
        result = alloc.request_action_slot(Priority.NORMAL)
        self.assertIsInstance(result, bool)
        result = alloc.request_llm_call(Priority.NORMAL)
        self.assertIsInstance(result, bool)

    def test_diagnostic_action_slots_used_increments(self):
        alloc = ResourceAllocator()
        alloc.request_action_slot(Priority.NORMAL)
        alloc.request_action_slot(Priority.NORMAL)
        self.assertEqual(alloc.action_slots_used, 2)

    def test_diagnostic_action_slots_reset_on_tick(self):
        alloc = ResourceAllocator()
        alloc.request_action_slot(Priority.NORMAL)
        alloc.tick()
        self.assertEqual(alloc.action_slots_used, 0)

    def test_diagnostic_llm_calls_used_increments(self):
        alloc = ResourceAllocator()
        alloc.request_llm_call(Priority.NORMAL)
        alloc.request_llm_call(Priority.URGENT)
        self.assertEqual(alloc.llm_calls_used, 2)

    def test_custom_budget_parameters_accepted(self):
        alloc = ResourceAllocator(action_slots_per_tick=5, llm_calls_per_hour=10)
        self.assertTrue(alloc.request_action_slot(Priority.NORMAL))


if __name__ == "__main__":
    unittest.main(verbosity=2)
