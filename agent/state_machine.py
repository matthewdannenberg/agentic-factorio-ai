"""
agent/state_machine.py

AgentState — the top-level state machine enum for the agent loop.

Consumed by:  agent/loop.py
Informed by:  agent/examiner/audit_report.py, llm/budget.py

This module only defines the states and their valid transitions. Transition
logic lives in agent/loop.py.

State semantics
---------------
PLANNING   LLM is being called to set a new Goal + RewardSpec (or revise one).
           Entry: start of run, after goal completion, after rich examination.
           Exit:  Goal created → EXECUTING.

EXECUTING  Pure-code execution of the active Goal via primitives.
           No LLM calls are made in this state.
           Exit:  goal complete → EXAMINING(rich)
                  rate limited / anomaly / scheduled → EXAMINING(mechanical)
                  emergency (biters) → PLANNING (preempt)

EXAMINING  Examination of factory state. Two modes:
           RICH       — LLM available; full reflection + curation.
           MECHANICAL — LLM unavailable; pure-code health checks.
           Exit:  rich → PLANNING
                  mechanical → WAITING (accumulate AuditReports until LLM back)

WAITING    LLM is rate-limited or unavailable. Mechanical examination continues
           in a loop, accumulating reports. No planning or LLM calls.
           Exit:  LLM available again → EXAMINING(rich) with accumulated report.
"""

from __future__ import annotations

from enum import Enum, auto


class AgentState(Enum):
    PLANNING   = auto()
    EXECUTING  = auto()
    EXAMINING  = auto()
    WAITING    = auto()


class ExamineMode(Enum):
    RICH       = auto()
    MECHANICAL = auto()


# ---------------------------------------------------------------------------
# Valid transitions — enforced by agent/loop.py
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.PLANNING:  frozenset({AgentState.EXECUTING}),
    AgentState.EXECUTING: frozenset({AgentState.EXAMINING, AgentState.PLANNING}),
    AgentState.EXAMINING: frozenset({AgentState.PLANNING, AgentState.WAITING}),
    AgentState.WAITING:   frozenset({AgentState.EXAMINING}),
}


def assert_valid_transition(from_state: AgentState, to_state: AgentState) -> None:
    allowed = VALID_TRANSITIONS.get(from_state, frozenset())
    if to_state not in allowed:
        raise RuntimeError(
            f"Invalid state transition: {from_state.name} → {to_state.name}. "
            f"Allowed from {from_state.name}: "
            f"{', '.join(s.name for s in allowed) or 'none'}"
        )
