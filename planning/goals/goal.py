"""
planning/goals/goal.py  (revised)

Goal — a strategic or tactical planning objective.

Replaces the previous Goal + GoalStatus + GoalTree-activation model.
Goal now inherits from PlanningItem, gaining the unified lifecycle enum
(ItemStatus) and shared fields (id, description, conditions, parent_id).

Goal-specific additions
-----------------------
priority    : Priority enum for preemption ordering.
reward_spec : Reward shape, evaluated by RewardEvaluator on resolution.
step        : Coordinator handler state — which decision point the handler
              has reached. Replaces GoalFrame.step.
context     : Coordinator handler state — arbitrary dict for per-handler
              persistence across ticks. Replaces GoalFrame.context.

The coordinator never inspects Goal internals directly; it reads step/context
through the item it pops off the stack and treats them as opaque carry-along
state for the handler.

GoalStatus / GoalTree
---------------------
GoalStatus is retained as a thin alias over ItemStatus so existing code that
imports GoalStatus continues to work. GoalTree is retained unchanged for the
Phase 11 LLM layer — it is not wired into the execution path yet.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from planning.planning_item import PlanningItem, ItemStatus


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    BACKGROUND = 0
    NORMAL     = 1
    URGENT     = 2
    EMERGENCY  = 3


# ---------------------------------------------------------------------------
# GoalStatus — alias for backward compatibility
# ---------------------------------------------------------------------------

GoalStatus = ItemStatus   # import GoalStatus still works; values are the same


# ---------------------------------------------------------------------------
# RewardSpec
# ---------------------------------------------------------------------------

@dataclass
class RewardSpec:
    success_reward: float = 1.0
    failure_penalty: float = 0.5
    milestone_rewards: dict[str, float] = field(default_factory=dict)
    time_discount: float = 1.0
    calibration_notes: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.time_discount <= 1.0:
            raise ValueError(
                f"time_discount must be in (0.0, 1.0], got {self.time_discount}"
            )
        if self.failure_penalty < 0.0:
            raise ValueError(
                f"failure_penalty must be non-negative, got {self.failure_penalty}"
            )

    def discounted_success_reward(self, ticks_elapsed: int) -> float:
        return self.success_reward * (self.time_discount ** ticks_elapsed)


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------

@dataclass
class Goal(PlanningItem):
    """
    A strategic or tactical planning objective.

    Produced by: LLM layer (Phase 11) or coordinator hierarchy.
    Handled by: coordinator goal handlers (_handle_collection, etc.)

    Fields beyond PlanningItem
    --------------------------
    priority    : Governs preemption. Higher priority Goals push lower ones
                  into SUSPENDED status.
    reward_spec : Reward shape evaluated on resolution.
    step        : Coordinator handler step counter (replaces GoalFrame.step).
                  0 = handler not yet entered; >0 = handler has been entered
                  and should resume at this step.
    context     : Coordinator handler carry-along state (replaces
                  GoalFrame.context). Mutable dict; handler-specific.
    """
    priority:    Priority  = Priority.NORMAL
    reward_spec: RewardSpec = field(default_factory=RewardSpec)
    step:        int        = 0
    context:     dict       = field(default_factory=dict)

    @property
    def ticks_elapsed(self) -> int:
        """Ticks since activation, or 0 if not yet started."""
        if self.created_at == 0:
            return 0
        end = self.resolved_at if self.resolved_at is not None else self.created_at
        return max(0, end - self.created_at)

    def __repr__(self) -> str:
        return (
            f"Goal({self.id[:8]}… priority={self.priority.name} "
            f"status={self.status.name} step={self.step} "
            f"desc={self.description!r})"
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_goal(
    description: str,
    success_condition: str,
    failure_condition: str,
    priority: Priority = Priority.NORMAL,
    success_reward: float = 1.0,
    failure_penalty: float = 0.5,
    milestone_rewards: Optional[dict[str, float]] = None,
    time_discount: float = 1.0,
    calibration_notes: str = "",
    parent_id: Optional[str] = None,
    created_at: int = 0,
) -> Goal:
    spec = RewardSpec(
        success_reward=success_reward,
        failure_penalty=failure_penalty,
        milestone_rewards=milestone_rewards or {},
        time_discount=time_discount,
        calibration_notes=calibration_notes,
    )
    return Goal(
        description=description,
        priority=priority,
        success_condition=success_condition,
        failure_condition=failure_condition,
        reward_spec=spec,
        parent_id=parent_id,
        created_at=created_at,
    )