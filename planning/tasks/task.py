"""
planning/tasks/task.py

Task (née Subtask) — a locally-derived prerequisite task produced by the execution network.

Structurally similar to Goal but created by the coordinator/agents rather than
the LLM. Tasks are derived from KB production chain queries and self-model
state, not from LLM calls — unless derivation fails and the LLM injects one
via escalation (in which case derived_locally=False).

TaskRecord — the resolved record of a completed or failed Task.
Written to the TaskLedger when a Task is popped. Captures enough
information to reconstruct the attempted work tree for LLM escalation prompts.

See planning/tasks/task_ledger.py for the TaskLedger that manages the
live stack and history log.

Rules
-----
- Pure data + ledger operations. No LLM calls. No RCON. No game state reads.
- Status transitions are enforced; illegal transitions raise RuntimeError.
- The ledger does not evaluate success/failure conditions — the coordinator
  does that via RewardEvaluator, then calls complete() or fail() on the
  Task before popping it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# TaskStatus
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    """
    Lifecycle state of a single Task.

    PENDING   : Created but not yet the active Task (waiting below top).
    ACTIVE    : Currently being executed (topmost on the live stack).
    COMPLETE  : Success condition was met; has been popped from the stack.
    FAILED    : Failure condition met or derivation failed; has been popped.
    ESCALATED : Failed and passed to the LLM for decomposition. A subtype of
                failure that StuckContext records explicitly, so the prompt
                layer can distinguish "failed silently" from "was escalated".
    """
    PENDING   = auto()
    ACTIVE    = auto()
    COMPLETE  = auto()
    FAILED    = auto()
    ESCALATED = auto()


# Compact outcome type used in TaskRecord
TaskOutcome = Literal["complete", "failed", "escalated"]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """
    A single prerequisite task in the execution network.

    Fields
    ------
    id                : UUID string, auto-generated.
    description       : Human-readable statement of what must be achieved.
    success_condition : Evaluable expression string (same format as Goal).
                        Evaluated by RewardEvaluator against WorldQuery.
    failure_condition : Evaluable expression string. Triggers escalation.
    parent_goal_id    : ID of the Goal this Task ultimately serves.
    parent_task_id : ID of the Task that spawned this one (nested
                        Tasks), or None if derived directly from the goal.
    created_at        : Game tick at creation.
    status            : Current lifecycle state.
    derived_locally   : True if derived by the coordinator/agents from KB and
                        self-model. False if injected by the LLM after
                        escalation.
    resolved_at       : Game tick at terminal transition, or None if ongoing.
    """
    description: str
    success_condition: str
    failure_condition: str
    parent_goal_id: str
    created_at: int
    derived_locally: bool
    parent_task_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    resolved_at: Optional[int] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Transition PENDING → ACTIVE."""
        if self.status != TaskStatus.PENDING:
            raise RuntimeError(
                f"Cannot activate task in status {self.status.name}; "
                "expected PENDING"
            )
        self.status = TaskStatus.ACTIVE

    def complete(self, tick: int) -> None:
        """Transition ACTIVE → COMPLETE."""
        if self.status != TaskStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot complete task in status {self.status.name}; "
                "expected ACTIVE"
            )
        self.status = TaskStatus.COMPLETE
        self.resolved_at = tick

    def fail(self, tick: int) -> None:
        """Transition ACTIVE → FAILED."""
        if self.status != TaskStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot fail task in status {self.status.name}; "
                "expected ACTIVE"
            )
        self.status = TaskStatus.FAILED
        self.resolved_at = tick

    def escalate(self, tick: int) -> None:
        """
        Transition ACTIVE → ESCALATED.

        Called instead of fail() when the coordinator is building a
        StuckContext. Distinguishes "gave up silently" from "sent to LLM"
        so the history log and prompt layer can tell them apart.
        """
        if self.status != TaskStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot escalate task in status {self.status.name}; "
                "expected ACTIVE"
            )
        self.status = TaskStatus.ESCALATED
        self.resolved_at = tick

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            TaskStatus.COMPLETE,
            TaskStatus.FAILED,
            TaskStatus.ESCALATED,
        )

    @property
    def is_active(self) -> bool:
        return self.status == TaskStatus.ACTIVE

    @property
    def parent_id(self) -> str:
        """
        The id of this task's immediate parent.

        Returns parent_task_id if this is a nested task, or
        parent_goal_id if this is a top-level task. Used as the key when
        recording this task in the ledger's history log.
        """
        return self.parent_task_id if self.parent_task_id else self.parent_goal_id

    def __repr__(self) -> str:
        origin = "local" if self.derived_locally else "injected"
        return (
            f"Task({self.id[:8]}… status={self.status.name} "
            f"[{origin}] desc={self.description!r})"
        )


# ---------------------------------------------------------------------------
# TaskRecord
# ---------------------------------------------------------------------------

@dataclass
class TaskRecord:
    """
    The resolved record of a Task that has left the live stack.

    Written to the TaskLedger history when a Task is popped after
    reaching a terminal status. Enough information to reconstruct the
    attempted work tree for LLM escalation prompts.

    Fields
    ------
    task         : The resolved Task (with terminal status set).
    outcome      : Compact label — "complete", "failed", or "escalated".
    children_ids : IDs of Tasks that were derived from this one (whose
                   parent_task_id equals this Task's id). Populated by
                   the ledger from its own history at record-creation time.
    """
    task: Task
    outcome: TaskOutcome
    children_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for LLM prompt construction."""
        return {
            "id": self.task.id,
            "description": self.task.description,
            "outcome": self.outcome,
            "derived_locally": self.task.derived_locally,
            "created_at": self.task.created_at,
            "resolved_at": self.task.resolved_at,
            "children_ids": list(self.children_ids),
        }