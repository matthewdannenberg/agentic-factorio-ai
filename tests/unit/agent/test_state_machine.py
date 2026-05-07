"""
tests/unit/agent/test_state_machine.py

Tests for agent/state_machine.py 

Run with:  python tests/unit/agent/test_state_machine.py
"""

from __future__ import annotations

import unittest

from agent.state_machine import AgentState, assert_valid_transition

class TestStateMachine(unittest.TestCase):
    def _ok(self, frm, to):
        assert_valid_transition(frm, to)  # must not raise

    def _bad(self, frm, to):
        with self.assertRaises(RuntimeError):
            assert_valid_transition(frm, to)

    def test_planning_to_executing(self):
        self._ok(AgentState.PLANNING, AgentState.EXECUTING)

    def test_executing_to_examining(self):
        self._ok(AgentState.EXECUTING, AgentState.EXAMINING)

    def test_executing_to_planning_emergency(self):
        self._ok(AgentState.EXECUTING, AgentState.PLANNING)

    def test_examining_to_planning(self):
        self._ok(AgentState.EXAMINING, AgentState.PLANNING)

    def test_examining_to_waiting(self):
        self._ok(AgentState.EXAMINING, AgentState.WAITING)

    def test_waiting_to_examining(self):
        self._ok(AgentState.WAITING, AgentState.EXAMINING)

    def test_planning_to_waiting_invalid(self):
        self._bad(AgentState.PLANNING, AgentState.WAITING)

    def test_waiting_to_executing_invalid(self):
        self._bad(AgentState.WAITING, AgentState.EXECUTING)

    def test_waiting_to_planning_invalid(self):
        self._bad(AgentState.WAITING, AgentState.PLANNING)


if __name__ == "__main__":
    unittest.main(verbosity=2)