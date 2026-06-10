"""
planning/planning_item.py

PlanningItem — unified base class for Goals and Tasks.

Both Goal and Task represent "something to achieve" with condition-driven
lifecycle. They differ in origin and routing:
  - Goal: produced by LLM or coordinator hierarchy; handled by coordinator
    goal handlers; carries priority and reward shape.
  - Task: produced by coordinator goal handlers; routed to a specific agent;
        carries agent_hint and runtime params.

The unified stack (coordinator._stack: list[PlanningItem]) holds both.
The top of the stack is always the current unit of work:
  - If Task: tick its agent, evaluate success/failure, pop when done.
  - If Goal: run its goal handler, which may push sub-Goals or Tasks on top.

Parent relationships
--------------------
parent_id points to the PlanningItem that created this one — always another
PlanningItem, whether Goal or Task. This supports the failure pipeline:

    Goal A → Task B + Goal C (stack: [A, C, B], top=B)
    B fails → push diagnose_failure(parent=B) on top
    Diagnosis resolves → B retried or stack revised
    A preserved throughout

The parent chain is Goal-or-Task all the way up, with None at the root.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# ItemStatus — unified lifecycle enum
# ---------------------------------------------------------------------------

class ItemStatus(Enum):
    """
    Lifecycle states shared by both Goals and Tasks.

    PENDING   : Created but not yet the active item.
    ACTIVE    : Currently being executed (top of stack).
    SUSPENDED : Preempted by a higher-priority item; resumable.
    COMPLETE  : Success condition met; item has been popped.
    FAILED    : Failure condition met; item has been popped.
    ESCALATED : Failed and referred to the LLM for decomposition.
                Distinguishes "gave up silently" from "sent to LLM".
    """
    PENDING   = auto()
    ACTIVE    = auto()
    SUSPENDED = auto()
    COMPLETE  = auto()
    FAILED    = auto()
    ESCALATED = auto()


# ---------------------------------------------------------------------------
# PlanningItem — base class
# ---------------------------------------------------------------------------

@dataclass
class PlanningItem:
    """
    Base class for Goal and Task.

    Carries all fields common to both: identity, lifecycle, condition strings,
    and parent linkage.

    Subclasses add their distinctive fields:
      Goal: priority, reward_spec, step (coordinator handler state), context
      Task: agent_hint, params, derived_locally
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Human-readable
    description: str = ""

    # Condition strings — evaluated by RewardEvaluator
    success_condition: str = ""
    failure_condition: str = ""

    # Lifecycle
    status: ItemStatus = ItemStatus.PENDING
    created_at: int = 0
    resolved_at: Optional[int] = None

    # Hierarchy — id of the PlanningItem that created this one, or None
    parent_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def activate(self, tick: int = 0) -> None:
        if self.status not in (ItemStatus.PENDING, ItemStatus.SUSPENDED):
            raise RuntimeError(
                f"Cannot activate {type(self).__name__} in status "
                f"{self.status.name}; expected PENDING or SUSPENDED"
            )
        self.status = ItemStatus.ACTIVE
        if self.created_at == 0:
            self.created_at = tick

    def suspend(self) -> None:
        if self.status != ItemStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot suspend {type(self).__name__} in status "
                f"{self.status.name}; expected ACTIVE"
            )
        self.status = ItemStatus.SUSPENDED

    def complete(self, tick: int) -> None:
        if self.status != ItemStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot complete {type(self).__name__} in status "
                f"{self.status.name}; expected ACTIVE"
            )
        self.status = ItemStatus.COMPLETE
        self.resolved_at = tick

    def fail(self, tick: int) -> None:
        if self.status != ItemStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot fail {type(self).__name__} in status "
                f"{self.status.name}; expected ACTIVE"
            )
        self.status = ItemStatus.FAILED
        self.resolved_at = tick

    def escalate(self, tick: int) -> None:
        if self.status != ItemStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot escalate {type(self).__name__} in status "
                f"{self.status.name}; expected ACTIVE"
            )
        self.status = ItemStatus.ESCALATED
        self.resolved_at = tick

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            ItemStatus.COMPLETE, ItemStatus.FAILED, ItemStatus.ESCALATED
        )

    @property
    def is_active(self) -> bool:
        return self.status == ItemStatus.ACTIVE