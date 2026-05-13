"""
agent/subtask.py

Subtask — a locally-derived prerequisite task produced by the execution network.

Structurally similar to Goal but created by the coordinator/agents rather than
the LLM. Subtasks are derived from KB production chain queries and self-model
state, not from LLM calls — unless derivation fails and the LLM injects one
via escalation (in which case derived_locally=False).

SubtaskRecord — the resolved record of a completed or failed subtask.
Written to the SubtaskLedger when a subtask is popped. Captures enough
information to reconstruct the attempted work tree for LLM escalation prompts.

SubtaskLedger — replaces the simple SubtaskStack.
Combines a live LIFO stack (the active nesting chain) with a history log
(all resolved subtasks, keyed by their parent). Together they let the
coordinator produce a complete picture of what was attempted at every level
when building a StuckContext for escalation.

Why not a plain stack?
----------------------
A plain stack discards completed siblings the moment they are popped. When
Task1 and Task2 are both prerequisites of Goal1, and Task1 completes before
Task2 gets stuck, a plain stack has no memory of Task1. The LLM escalation
prompt would then describe Task2's failure with no indication that Task1 was
already done — potentially causing the LLM to re-derive work that is complete,
or to produce a decomposition that conflicts with what Task1 produced.

The ledger's history log retains every resolved SubtaskRecord under its parent's
id, so the full sibling context is available at escalation time.

Rules
-----
- Pure data + ledger operations. No LLM calls. No RCON. No game state reads.
- Status transitions are enforced; illegal transitions raise RuntimeError.
- The ledger does not evaluate success/failure conditions — the coordinator
  does that via RewardEvaluator, then calls complete() or fail() on the
  subtask before popping it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# SubtaskStatus
# ---------------------------------------------------------------------------

class SubtaskStatus(Enum):
    """
    Lifecycle state of a single subtask.

    PENDING   : Created but not yet the active subtask (waiting below top).
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


# Compact outcome type used in SubtaskRecord
SubtaskOutcome = Literal["complete", "failed", "escalated"]


# ---------------------------------------------------------------------------
# Subtask
# ---------------------------------------------------------------------------

@dataclass
class Subtask:
    """
    A single prerequisite task in the execution network.

    Fields
    ------
    id                : UUID string, auto-generated.
    description       : Human-readable statement of what must be achieved.
    success_condition : Evaluable expression string (same format as Goal).
                        Evaluated by RewardEvaluator against WorldQuery.
    failure_condition : Evaluable expression string. Triggers escalation.
    parent_goal_id    : ID of the Goal this subtask ultimately serves.
    parent_subtask_id : ID of the Subtask that spawned this one (nested
                        subtasks), or None if derived directly from the goal.
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
    parent_subtask_id: Optional[str] = None
    status: SubtaskStatus = SubtaskStatus.PENDING
    resolved_at: Optional[int] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Transition PENDING → ACTIVE."""
        if self.status != SubtaskStatus.PENDING:
            raise RuntimeError(
                f"Cannot activate subtask in status {self.status.name}; "
                "expected PENDING"
            )
        self.status = SubtaskStatus.ACTIVE

    def complete(self, tick: int) -> None:
        """Transition ACTIVE → COMPLETE."""
        if self.status != SubtaskStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot complete subtask in status {self.status.name}; "
                "expected ACTIVE"
            )
        self.status = SubtaskStatus.COMPLETE
        self.resolved_at = tick

    def fail(self, tick: int) -> None:
        """Transition ACTIVE → FAILED."""
        if self.status != SubtaskStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot fail subtask in status {self.status.name}; "
                "expected ACTIVE"
            )
        self.status = SubtaskStatus.FAILED
        self.resolved_at = tick

    def escalate(self, tick: int) -> None:
        """
        Transition ACTIVE → ESCALATED.

        Called instead of fail() when the coordinator is building a
        StuckContext. Distinguishes "gave up silently" from "sent to LLM"
        so the history log and prompt layer can tell them apart.
        """
        if self.status != SubtaskStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot escalate subtask in status {self.status.name}; "
                "expected ACTIVE"
            )
        self.status = SubtaskStatus.ESCALATED
        self.resolved_at = tick

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            SubtaskStatus.COMPLETE,
            SubtaskStatus.FAILED,
            SubtaskStatus.ESCALATED,
        )

    @property
    def is_active(self) -> bool:
        return self.status == SubtaskStatus.ACTIVE

    @property
    def parent_id(self) -> str:
        """
        The id of this subtask's immediate parent.

        Returns parent_subtask_id if this is a nested subtask, or
        parent_goal_id if this is a top-level subtask. Used as the key when
        recording this subtask in the ledger's history log.
        """
        return self.parent_subtask_id if self.parent_subtask_id else self.parent_goal_id

    def __repr__(self) -> str:
        origin = "local" if self.derived_locally else "injected"
        return (
            f"Subtask({self.id[:8]}… status={self.status.name} "
            f"[{origin}] desc={self.description!r})"
        )


# ---------------------------------------------------------------------------
# SubtaskRecord
# ---------------------------------------------------------------------------

@dataclass
class SubtaskRecord:
    """
    The resolved record of a subtask that has left the live stack.

    Written to the SubtaskLedger history when a subtask is popped after
    reaching a terminal status. Enough information to reconstruct the
    attempted work tree for LLM escalation prompts.

    Fields
    ------
    subtask      : The resolved Subtask (with terminal status set).
    outcome      : Compact label — "complete", "failed", or "escalated".
    children_ids : IDs of subtasks that were derived from this one (whose
                   parent_subtask_id equals this subtask's id). Populated by
                   the ledger from its own history at record-creation time.
    """
    subtask: Subtask
    outcome: SubtaskOutcome
    children_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for LLM prompt construction."""
        return {
            "id": self.subtask.id,
            "description": self.subtask.description,
            "outcome": self.outcome,
            "derived_locally": self.subtask.derived_locally,
            "created_at": self.subtask.created_at,
            "resolved_at": self.subtask.resolved_at,
            "children_ids": list(self.children_ids),
        }


# ---------------------------------------------------------------------------
# SubtaskLedger
# ---------------------------------------------------------------------------

class SubtaskLedger:
    """
    Live stack + history log for the coordinator's subtask lifecycle.

    Live stack
    ----------
    Behaves like the previous SubtaskStack: push/pop/peek, LIFO ordering,
    auto-activates the newly revealed top on pop. At most one subtask has
    status ACTIVE at a time (the top of the stack).

    History log
    -----------
    A dict mapping parent_id → list[SubtaskRecord], where parent_id is either
    a goal id (for top-level subtasks) or a subtask id (for nested ones). Each
    entry is written when a subtask is popped after reaching a terminal status.

    This preserves the full sibling context at every nesting level. When
    Task1 completes and is popped, its record remains under goal_id. When
    Task2 then gets stuck, failure_chain() and sibling_history() together
    reconstruct:

        Goal1
        ├── Task1  ✓  (history under goal_id)
        └── Task2  ✗  (live stack, top)
            ├── TaskA  ✓  (history under task2.id)
            └── Task3  ✗  (live stack, bottom of Task2 nesting)

    Pop discipline
    --------------
    The coordinator must call complete(), fail(), or escalate() on the topmost
    subtask before calling pop(). pop() enforces this — it raises RuntimeError
    if the top is not terminal. This keeps the history log's outcome field
    accurate without requiring pop() to infer intent.

    Clearing
    --------
    clear() removes both the live stack and the history log. Called on goal
    reset. History is goal-scoped — it does not survive across goals.
    """

    def __init__(self) -> None:
        # Live stack: index 0 = bottom (oldest/outermost), -1 = top (ACTIVE)
        self._stack: list[Subtask] = []
        # History: parent_id -> [SubtaskRecord, ...] in resolution order
        self._history: dict[str, list[SubtaskRecord]] = {}

    # ------------------------------------------------------------------
    # Stack operations
    # ------------------------------------------------------------------

    def push(self, subtask: Subtask) -> None:
        """
        Push a subtask onto the live stack.

        The newly pushed subtask is always activated immediately — it is a
        prerequisite that must complete before the subtask below it can
        continue. The previous top (if any) transitions back to PENDING to
        reflect that it is suspended waiting for this prerequisite.

        This is analogous to a call stack: pushing a new frame suspends the
        current frame. Popping the top resumes the frame below.

        Raises ValueError if the subtask is not PENDING.
        """
        if subtask.status != SubtaskStatus.PENDING:
            raise ValueError(
                f"Only PENDING subtasks can be pushed; got {subtask.status.name}"
            )
        # Suspend the current top — it is waiting for this prerequisite.
        if self._stack and self._stack[-1].status == SubtaskStatus.ACTIVE:
            self._stack[-1].status = SubtaskStatus.PENDING
        # Activate the incoming subtask immediately.
        subtask.activate()
        self._stack.append(subtask)

    def pop(self) -> Optional[Subtask]:
        """
        Remove the topmost subtask, record it in the history log, and return it.

        The topmost subtask must already be in a terminal status (COMPLETE,
        FAILED, or ESCALATED). The coordinator is responsible for calling
        complete(), fail(), or escalate() before calling pop(). This ensures
        the history log's outcome field is always accurate.

        After the pop, if the newly revealed top is PENDING it is activated.

        Returns None if the stack is empty.
        Raises RuntimeError if the top is not terminal.
        """
        if not self._stack:
            return None

        subtask = self._stack[-1]
        if not subtask.is_terminal:
            raise RuntimeError(
                f"Cannot pop subtask {subtask.id[:8]}… with status "
                f"{subtask.status.name}; call complete(), fail(), or "
                "escalate() first"
            )

        self._stack.pop()

        # Determine compact outcome label
        outcome: SubtaskOutcome = (
            "complete"  if subtask.status == SubtaskStatus.COMPLETE  else
            "escalated" if subtask.status == SubtaskStatus.ESCALATED else
            "failed"
        )

        # children_ids: any subtasks already recorded in history under this id
        children_ids = [
            rec.subtask.id
            for rec in self._history.get(subtask.id, [])
        ]
        record = SubtaskRecord(
            subtask=subtask,
            outcome=outcome,
            children_ids=children_ids,
        )
        self._history.setdefault(subtask.parent_id, []).append(record)

        # Activate the newly revealed top — it was suspended waiting for
        # the prerequisite that just completed.
        if self._stack and self._stack[-1].status == SubtaskStatus.PENDING:
            self._stack[-1].activate()

        return subtask

    def peek(self) -> Optional[Subtask]:
        """Return the topmost (currently ACTIVE) subtask, or None."""
        return self._stack[-1] if self._stack else None

    def clear(self) -> None:
        """Remove all live stack entries and history. Called on goal reset."""
        self._stack.clear()
        self._history.clear()

    # ------------------------------------------------------------------
    # Escalation context queries
    # ------------------------------------------------------------------

    def failure_chain(self) -> list[Subtask]:
        """
        Return the current live stack from outermost to innermost.

        This is the nesting chain from the direct child of the goal down to
        the currently stuck (topmost) subtask. The last element is the
        immediate failure point.

        Example: if the stack is [Task2 (bottom), Task3 (top)], returns
        [Task2, Task3] — Task2 is the direct child of Goal1, Task3 is the
        leaf where execution is stuck.

        Returns an empty list if no subtasks have been pushed (stuck at
        goal level before any subtask was derived).
        """
        return list(self._stack)   # bottom=0 → top=-1

    def sibling_history(
        self,
        chain: list[Subtask],
        goal_id: str,
    ) -> dict[str, list[SubtaskRecord]]:
        """
        Return resolved sibling records at each level of the nesting chain.

        For each level (goal → chain[0] → chain[1] → … → chain[-1]), returns
        the SubtaskRecords of all siblings that were attempted and resolved
        before the current live subtask at that level was pushed.

        Parameters
        ----------
        chain   : Result of failure_chain() — the current live nesting.
        goal_id : ID of the currently active goal (used as the root key).

        Returns
        -------
        Dict keyed by parent id. Present only for levels that have at least
        one resolved sibling. Each value is ordered by resolution time
        (earliest first).

        Example — Goal1 → [Task1✓, Task2(stuck)] / Task2 → [TaskA✓, Task3(stuck)]:

            {
                goal_id:  [SubtaskRecord(Task1, "complete")],
                task2.id: [SubtaskRecord(TaskA, "complete")],
            }

        Levels with no resolved siblings are omitted from the result, so the
        caller can iterate only over levels where something was actually tried.
        """
        # Parent ids for each level: goal itself, then each non-leaf in chain
        parent_ids = [goal_id] + [t.id for t in chain[:-1]]
        return {
            pid: list(self._history[pid])
            for pid in parent_ids
            if pid in self._history
        }

    # ------------------------------------------------------------------
    # Direct history access
    # ------------------------------------------------------------------

    def history_for(self, parent_id: str) -> list[SubtaskRecord]:
        """
        Return resolved SubtaskRecords whose parent is parent_id, in
        resolution order. Returns an empty list if none exist.
        """
        return list(self._history.get(parent_id, []))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of subtasks currently on the live stack."""
        return len(self._stack)

    def __bool__(self) -> bool:
        return bool(self._stack)

    def __repr__(self) -> str:
        history_count = sum(len(v) for v in self._history.values())
        return (
            f"SubtaskLedger(stack_depth={len(self._stack)}, "
            f"history={history_count})"
        )