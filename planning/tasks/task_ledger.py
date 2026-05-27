"""
planning/tasks/task_ledger.py

TaskLedger (née SubtaskLedger) — live stack + history log for the coordinator's
task lifecycle.

See planning/tasks/task.py for the Task, TaskStatus, and TaskRecord types.
"""

from __future__ import annotations

from typing import Optional

from planning.tasks.task import Task, TaskStatus, TaskRecord, TaskOutcome


class TaskLedger:
    """
    Live stack + history log for the coordinator's Task lifecycle.

    Live stack
    ----------
    Behaves like the previous TaskStack: push/pop/peek, LIFO ordering,
    auto-activates the newly revealed top on pop. At most one Task has
    status ACTIVE at a time (the top of the stack).

    History log
    -----------
    A dict mapping parent_id → list[TaskRecord], where parent_id is either
    a goal id (for top-level Tasks) or a Task id (for nested ones). Each
    entry is written when a Task is popped after reaching a terminal status.

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
    Task before calling pop(). pop() enforces this — it raises RuntimeError
    if the top is not terminal. This keeps the history log's outcome field
    accurate without requiring pop() to infer intent.

    Clearing
    --------
    clear() removes both the live stack and the history log. Called on goal
    reset. History is goal-scoped — it does not survive across goals.
    """

    def __init__(self) -> None:
        # Live stack: index 0 = bottom (oldest/outermost), -1 = top (ACTIVE)
        self._stack: list[Task] = []
        # History: parent_id -> [TaskRecord, ...] in resolution order
        self._history: dict[str, list[TaskRecord]] = {}

    # ------------------------------------------------------------------
    # Stack operations
    # ------------------------------------------------------------------

    def push(self, task: Task) -> None:
        """
        Push a Task onto the live stack.

        The newly pushed Task is always activated immediately — it is a
        prerequisite that must complete before the Task below it can
        continue. The previous top (if any) transitions back to PENDING to
        reflect that it is suspended waiting for this prerequisite.

        This is analogous to a call stack: pushing a new frame suspends the
        current frame. Popping the top resumes the frame below.

        Raises ValueError if the Task is not PENDING.
        """
        if task.status != TaskStatus.PENDING:
            raise ValueError(
                f"Only PENDING tasks can be pushed; got {task.status.name}"
            )
        # Suspend the current top — it is waiting for this prerequisite.
        if self._stack and self._stack[-1].status == TaskStatus.ACTIVE:
            self._stack[-1].status = TaskStatus.PENDING
        # Activate the incoming task immediately.
        task.activate()
        self._stack.append(task)

    def pop(self) -> Optional[Task]:
        """
        Remove the topmost Task, record it in the history log, and return it.

        The topmost Task must already be in a terminal status (COMPLETE,
        FAILED, or ESCALATED). The coordinator is responsible for calling
        complete(), fail(), or escalate() before calling pop(). This ensures
        the history log's outcome field is always accurate.

        After the pop, if the newly revealed top is PENDING it is activated.

        Returns None if the stack is empty.
        Raises RuntimeError if the top is not terminal.
        """
        if not self._stack:
            return None

        task = self._stack[-1]
        if not task.is_terminal:
            raise RuntimeError(
                f"Cannot pop task {task.id[:8]}… with status "
                f"{task.status.name}; call complete(), fail(), or "
                "escalate() first"
            )

        self._stack.pop()

        # Determine compact outcome label
        outcome: TaskOutcome = (
            "complete"  if task.status == TaskStatus.COMPLETE  else
            "escalated" if task.status == TaskStatus.ESCALATED else
            "failed"
        )

        # children_ids: any tasks already recorded in history under this id
        children_ids = [
            rec.task.id
            for rec in self._history.get(task.id, [])
        ]
        record = TaskRecord(
            task=task,
            outcome=outcome,
            children_ids=children_ids,
        )
        self._history.setdefault(task.parent_id, []).append(record)

        # Activate the newly revealed top — it was suspended waiting for
        # the prerequisite that just completed.
        if self._stack and self._stack[-1].status == TaskStatus.PENDING:
            self._stack[-1].activate()

        return task

    def peek(self) -> Optional[Task]:
        """Return the topmost (currently ACTIVE) Task, or None."""
        return self._stack[-1] if self._stack else None

    def clear(self) -> None:
        """Remove all live stack entries and history. Called on goal reset."""
        self._stack.clear()
        self._history.clear()

    # ------------------------------------------------------------------
    # Escalation context queries
    # ------------------------------------------------------------------

    def failure_chain(self) -> list[Task]:
        """
        Return the current live stack from outermost to innermost.

        This is the nesting chain from the direct child of the goal down to
        the currently stuck (topmost) Task. The last element is the
        immediate failure point.

        Example: if the stack is [Task2 (bottom), Task3 (top)], returns
        [Task2, Task3] — Task2 is the direct child of Goal1, Task3 is the
        leaf where execution is stuck.

        Returns an empty list if no Tasks have been pushed (stuck at
        goal level before any Task was derived).
        """
        return list(self._stack)   # bottom=0 → top=-1

    def sibling_history(
        self,
        chain: list[Task],
        goal_id: str,
    ) -> dict[str, list[TaskRecord]]:
        """
        Return resolved sibling records at each level of the nesting chain.

        For each level (goal → chain[0] → chain[1] → … → chain[-1]), returns
        the TaskRecords of all siblings that were attempted and resolved
        before the current live Task at that level was pushed.

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
                goal_id:  [TaskRecord(Task1, "complete")],
                task2.id: [TaskRecord(TaskA, "complete")],
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

    def history_for(self, parent_id: str) -> list[TaskRecord]:
        """
        Return resolved TaskRecords whose parent is parent_id, in
        resolution order. Returns an empty list if none exist.
        """
        return list(self._history.get(parent_id, []))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of Tasks currently on the live stack."""
        return len(self._stack)

    def __bool__(self) -> bool:
        return bool(self._stack)

    def __repr__(self) -> str:
        history_count = sum(len(v) for v in self._history.values())
        return (
            f"TaskLedger(stack_depth={len(self._stack)}, "
            f"history={history_count})"
        )