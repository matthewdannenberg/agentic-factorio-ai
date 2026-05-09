"""
planning/resource_allocator.py

ResourceAllocator — priority-weighted allocation of shared agent resources.

Consumed by:  agent/loop.py, agent/execution.py, llm/budget.py

Rules:
- Pure logic. No LLM calls. No RCON. No side effects outside this object.
- Initial implementation is a clean pass-through: every request is granted.
- Interface is designed for future contention when BITERS_ENABLED=True.
  The `tick()` method exists for future per-tick budget-reset logic.
  The `Priority` parameter exists so callers never need to change their
  call sites when contention logic is introduced.
"""

from __future__ import annotations

import logging

from planning.goal import Priority

log = logging.getLogger(__name__)


class ResourceAllocator:
    """
    Allocates shared agent resources (action slots, LLM call budget) by priority.

    Current implementation: pass-through. All requests are granted regardless
    of priority or budget state. This satisfies the interface contract while
    the biter threat module is disabled.

    Future contention design (placeholder):
    - action_slots_per_tick: max actions the execution layer may take per tick.
      BACKGROUND goals get fewer slots when URGENT/EMERGENCY goals are active.
    - llm_calls_per_hour: rolling budget tracked by llm/budget.py; allocator
      gates LLM call requests when the budget is near exhaustion, except for
      EMERGENCY priority.
    """

    def __init__(
        self,
        action_slots_per_tick: int = 10,
        llm_calls_per_hour: int = 30,
    ) -> None:
        self._action_slots_per_tick = action_slots_per_tick
        self._llm_calls_per_hour = llm_calls_per_hour
        # Reserved for future use
        self._action_slots_used_this_tick: int = 0
        self._llm_calls_used_this_hour: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_action_slot(self, priority: Priority) -> bool:
        """
        True if the agent may take an action at this priority level.

        Current: always True (pass-through).
        Future: gate BACKGROUND/NORMAL when URGENT/EMERGENCY goals are contending.
        """
        self._action_slots_used_this_tick += 1
        return True

    def request_llm_call(self, priority: Priority) -> bool:
        """
        True if an LLM call is permitted at this priority level.

        Current: always True (pass-through).
        Future: deny BACKGROUND when budget is low; always allow EMERGENCY.
        """
        self._llm_calls_used_this_hour += 1
        return True

    def tick(self) -> None:
        """
        Called each game tick to refresh per-tick budgets.
        Current: resets action slot counter only (pass-through).
        """
        self._action_slots_used_this_tick = 0

    # ------------------------------------------------------------------
    # Diagnostic helpers (for examination layer)
    # ------------------------------------------------------------------

    @property
    def action_slots_used(self) -> int:
        """Action slots consumed this tick (informational)."""
        return self._action_slots_used_this_tick

    @property
    def llm_calls_used(self) -> int:
        """LLM calls consumed this tracking window (informational)."""
        return self._llm_calls_used_this_hour
