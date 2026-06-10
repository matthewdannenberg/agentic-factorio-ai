"""
planning/tasks/task.py  (revised)

Task — a concrete work item routed to a specific agent.

Replaces the previous Task + TaskStatus model.
Task now inherits from PlanningItem, gaining the unified lifecycle enum
(ItemStatus) and shared fields.

Task-specific additions
-----------------------
agent_hint      : Which agent handles this task (e.g. "navigation", "mining").
params          : Runtime parameters the agent reads (target_position, bbox,
                  entity_types, etc.).
derived_locally : True if produced by the coordinator; False if LLM-injected.
task_type       : String tag identifying the task variant within an agent
                  (e.g. "gather_resource", "harvest_natural", "navigate_to").

TaskStatus / TaskRecord / TaskLedger
-------------------------------------
TaskStatus is retained as a thin alias over ItemStatus. TaskRecord and
TaskLedger are retained unchanged — the ledger is populated when a Task
pops off the unified stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from planning.planning_item import PlanningItem, ItemStatus
from typing import Literal


# ---------------------------------------------------------------------------
# TaskStatus — alias for backward compatibility
# ---------------------------------------------------------------------------

TaskStatus = ItemStatus

# TaskOutcome — compact resolution label used by TaskRecord and TaskLedger
TaskOutcome = Literal["complete", "failed", "escalated"]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class Task(PlanningItem):
    """
    A concrete work item assigned to a specific agent.

    Produced by: coordinator goal handlers (_handle_collection etc.)
    Handled by:  the agent identified by agent_hint.

    Fields beyond PlanningItem
    --------------------------
    agent_hint      : Agent ID string — matched against AgentProtocol.AGENT_ID.
    task_type       : Variant string within the agent (e.g. "gather_resource").
    params          : Arbitrary dict read by the agent at activation.
    derived_locally : True if produced by coordinator logic; False if the LLM
                      injected it after escalation.

    Backward-compat aliases
    -----------------------
    parent_goal_id  : Deprecated — use parent_id instead. Accepted in
                      __init__ for compatibility with existing tests and task
                      ledger code; sets parent_id if parent_id is not given.
    parent_task_id  : Deprecated — use parent_id instead. Overrides
                      parent_goal_id if both are supplied.
    """
    agent_hint:      str  = ""
    task_type:       str  = ""
    params:          dict = field(default_factory=dict)
    derived_locally: bool = True

    def __init__(self, **kwargs):
        # Extract deprecated compat fields before passing to PlanningItem
        parent_goal_id = kwargs.pop("parent_goal_id", None)
        parent_task_id = kwargs.pop("parent_task_id", None)
        # Resolve parent_id: parent_task_id overrides parent_goal_id
        if "parent_id" not in kwargs:
            kwargs["parent_id"] = parent_task_id or parent_goal_id
        # Set Task-specific fields with defaults
        self.agent_hint      = kwargs.pop("agent_hint",      "")
        self.task_type       = kwargs.pop("task_type",       "")
        self.params          = kwargs.pop("params",          {})
        self.derived_locally = kwargs.pop("derived_locally", True)
        # Delegate remaining fields to PlanningItem via object.__setattr__
        # (can't call super().__init__ on a dataclass easily, so set directly)
        for attr, default in [
            ("id",                None),
            ("description",       ""),
            ("success_condition", ""),
            ("failure_condition", ""),
            ("status",            ItemStatus.PENDING),
            ("created_at",        0),
            ("resolved_at",       None),
            ("parent_id",         None),
        ]:
            object.__setattr__(self, attr,
                               kwargs.pop(attr, default if attr != "id" else
                               __import__("uuid").uuid4().__str__()))
        if kwargs:
            raise TypeError(f"Task() got unexpected keyword arguments: {list(kwargs)}")

    @property
    def parent_goal_id(self) -> str:
        """Deprecated compat alias for parent_id."""
        return self.parent_id or ""

    @property
    def parent_task_id(self) -> Optional[str]:
        """Deprecated: returns parent_id if it looks like a task parent."""
        return self.parent_id

    def __repr__(self) -> str:
        origin = "local" if self.derived_locally else "injected"
        return (
            f"Task({self.id[:8]}… type={self.task_type!r} "
            f"agent={self.agent_hint!r} status={self.status.name} "
            f"[{origin}] desc={self.description!r})"
        )


# ---------------------------------------------------------------------------
# TaskRecord — retained from previous version unchanged
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc  # noqa: E402

from planning.planning_item import ItemStatus as _IS  # noqa: E402


@dataclass
class TaskRecord:
    """Resolved record written to TaskLedger when a Task pops off the stack."""
    task: Task
    outcome: str          # "complete" | "failed" | "escalated"
    children_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id":             self.task.id,
            "description":    self.task.description,
            "outcome":        self.outcome,
            "derived_locally": self.task.derived_locally,
            "created_at":     self.task.created_at,
            "resolved_at":    self.task.resolved_at,
            "children_ids":   list(self.children_ids),
        }