"""
agent/network/coordinator.py

CoordinatorProtocol — the interface the execution layer exposes inward to the
agent network.

The coordinator is the internal orchestrator of the multi-agent network. It
routes goals to agents, manages the subtask stack and blackboard lifecycle,
resolves action conflicts, and assembles ExecutionResult.

This module provides:
  - CoordinatorProtocol: the interface (classes satisfying it are injected)
  - StubCoordinator: a minimal Phase 5 implementation that satisfies the
    protocol and does nothing. Replaced in Phase 6 with the rule-based
    coordinator.

Rules
-----
- No learning logic. No game-playing behavior in this module.
- The stub is intentionally inert — it returns WAITING with no actions on
  every tick. Phase 6 provides the real implementation.
- The coordinator is the only component that may write to the subtask stack
  and clear blackboard scopes.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from agent.execution_protocol import ExecutionResult, ExecutionStatus
from planning.goal import Goal

if TYPE_CHECKING:
    from world.query import WorldQuery
    from world.writer import WorldWriter


# ---------------------------------------------------------------------------
# CoordinatorProtocol
# ---------------------------------------------------------------------------

class CoordinatorProtocol:
    """
    Interface for the execution network's internal coordinator.

    The ExecutionLayerProtocol delegates to this coordinator. External callers
    never interact with the coordinator directly — they go through
    ExecutionLayerProtocol.

    Methods
    -------
    reset(goal, wq)
        Prepare for a new goal. Clears the blackboard, subtask stack, and
        any goal-scoped state. Called by the execution layer on goal start.

    tick(goal, wq, ww, tick) -> ExecutionResult
        Run one coordination cycle: poll agents, resolve conflicts, manage
        subtask lifecycle, assemble and return ExecutionResult.
    """

    def reset(
        self,
        goal: Goal,
        wq: "WorldQuery",
        seed_subtasks: Optional[list] = None,
    ) -> None:
        """
        Prepare for a new goal.

        seed_subtasks: optional Subtask list injected by the LLM after an
        escalation. The coordinator pre-populates its ledger from these instead
        of starting derivation from scratch. See ExecutionLayerProtocol.reset().
        """
        raise NotImplementedError

    def tick(
        self,
        goal: Goal,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> ExecutionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# StubCoordinator
# ---------------------------------------------------------------------------

class StubCoordinator(CoordinatorProtocol):
    """
    Minimal stub coordinator for Phase 5.

    Satisfies CoordinatorProtocol. Does nothing meaningful:
      - reset() is a no-op.
      - tick() always returns ExecutionResult(actions=[], status=WAITING).

    Replaced by the rule-based coordinator in Phase 6.
    """

    def reset(
        self,
        goal: Goal,
        wq: "WorldQuery",
        seed_subtasks: Optional[list] = None,
    ) -> None:
        """No-op. Real reset logic comes in Phase 6."""
        pass

    def tick(
        self,
        goal: Goal,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> ExecutionResult:
        """Return WAITING with no actions every tick."""
        return ExecutionResult(actions=[], status=ExecutionStatus.WAITING)