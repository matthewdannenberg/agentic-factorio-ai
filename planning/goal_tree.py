"""
planning/goal_tree.py

GoalTree — runtime manager for the agent's goal hierarchy.

Produced by:  agent/strategic.py and agent/tactical.py (add goals here)
Consumed by:  agent/loop.py, agent/execution.py

Rules:
- Pure logic. No LLM calls. No RCON. No WorldState mutation.
- Goal lifecycle transitions are delegated to Goal methods (which enforce
  valid state changes and raise RuntimeError on violations).
- Preemption is LIFO: suspended goals resume in most-recently-suspended order.
- A parent goal is not complete until all its children are complete.
"""

from __future__ import annotations

import logging
from typing import Optional

from planning.goal import Goal, GoalStatus, Priority

log = logging.getLogger(__name__)


class GoalTree:
    """
    Authoritative store of all goals in the current run.

    Internal invariant: at most one goal is ACTIVE at any time (the leaf goal
    currently being executed). All others are PENDING, SUSPENDED, COMPLETE,
    or FAILED.

    Preemption stack: when a new goal outranks the active goal, the active goal
    is pushed onto `_suspended_stack` (LIFO). On resolution, the top of the
    stack resumes.
    """

    def __init__(self) -> None:
        self._goals: dict[str, Goal] = {}          # id → Goal, all statuses
        self._suspended_stack: list[str] = []       # ids of suspended goals, LIFO
        self._active_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_goal(self, goal: Goal) -> None:
        """
        Add a goal to the tree. If it outranks the current active goal,
        preempt (suspend) the current goal and activate the new one.
        If the tree is empty, activate immediately.
        """
        self._goals[goal.id] = goal

        current = self._active_goal_obj()

        if current is None:
            # Tree is empty or all goals resolved — activate directly
            if goal.status == GoalStatus.PENDING:
                goal.activate(tick=0)
            self._active_id = goal.id
        elif goal.priority > current.priority:
            # Preempt: suspend current, activate new
            log.debug(
                "Preempting %s (priority=%s) with %s (priority=%s)",
                current.id[:8], current.priority.name,
                goal.id[:8], goal.priority.name,
            )
            current.suspend()
            self._suspended_stack.append(current.id)
            goal.activate(tick=0)
            self._active_id = goal.id
        else:
            # Lower or equal priority — leave as PENDING, don't activate
            pass

    def active_goal(self) -> Optional[Goal]:
        """The currently executing goal, or None if the tree is empty/idle."""
        return self._active_goal_obj()

    def complete_active(self, tick: int) -> Optional[Goal]:
        """
        Mark the active goal complete. Checks parent completion. Resumes the
        most-recently-suspended goal (if any). Returns the next active goal.
        """
        current = self._active_goal_obj()
        if current is None:
            return None

        current.complete(tick)
        log.debug("Goal complete: %s", current.id[:8])

        self._active_id = None
        self._maybe_complete_parent(current, tick)
        return self._resume_or_next(tick)

    def fail_active(self, tick: int, reason: str = "") -> Optional[Goal]:
        """
        Mark the active goal failed. Resumes the most-recently-suspended goal
        (if any). Returns the next active goal.
        """
        current = self._active_goal_obj()
        if current is None:
            return None

        if reason:
            log.warning("Goal failed (%s): %s", reason, current.id[:8])
        current.fail(tick)
        self._active_id = None
        return self._resume_or_next(tick)

    def all_goals(self) -> list[Goal]:
        """All goals in any status, insertion order."""
        return list(self._goals.values())

    def pending_goals(self) -> list[Goal]:
        """PENDING goals sorted by priority descending (highest priority first)."""
        return sorted(
            (g for g in self._goals.values() if g.status == GoalStatus.PENDING),
            key=lambda g: g.priority,
            reverse=True,
        )

    def goal_by_id(self, goal_id: str) -> Optional[Goal]:
        """Look up a goal by its UUID. Returns None if not found."""
        return self._goals.get(goal_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_goal_obj(self) -> Optional[Goal]:
        if self._active_id is None:
            return None
        return self._goals.get(self._active_id)

    def _resume_or_next(self, tick: int) -> Optional[Goal]:
        """
        After the active goal resolves, resume the top of the suspended stack.
        If the stack is empty, activate the highest-priority pending goal.
        """
        # Try to resume most-recently-suspended
        while self._suspended_stack:
            candidate_id = self._suspended_stack.pop()
            candidate = self._goals.get(candidate_id)
            if candidate is None or candidate.is_terminal:
                continue  # Already resolved somehow — skip
            candidate.activate(tick=tick)
            self._active_id = candidate_id
            log.debug("Resumed suspended goal: %s", candidate_id[:8])
            return candidate

        # No suspended goals — promote the highest-priority pending goal
        pending = self.pending_goals()
        if pending:
            next_goal = pending[0]
            next_goal.activate(tick=tick)
            self._active_id = next_goal.id
            log.debug("Activated pending goal: %s", next_goal.id[:8])
            return next_goal

        self._active_id = None
        return None

    def _maybe_complete_parent(self, completed_child: Goal, tick: int) -> None:
        """
        If a child goal completes, check whether its parent can now be marked
        complete (all children complete).
        """
        parent_id = completed_child.parent_id
        if parent_id is None:
            return

        parent = self._goals.get(parent_id)
        if parent is None or parent.is_terminal:
            return

        # Find all direct children of the parent
        children = [
            g for g in self._goals.values()
            if g.parent_id == parent_id
        ]
        if all(c.status == GoalStatus.COMPLETE for c in children):
            # All children done — complete the parent too
            if parent.status == GoalStatus.ACTIVE:
                parent.complete(tick)
                log.debug("Parent goal auto-completed: %s", parent_id[:8])
                self._maybe_complete_parent(parent, tick)
