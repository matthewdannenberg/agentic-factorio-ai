"""
planning/goal.py

Goal, RewardSpec, Priority, GoalStatus — the planning layer's core types.

Produced by:  agent/strategic.py and agent/tactical.py (LLM layers)
Consumed by:  planning/goal_tree.py, planning/reward_evaluator.py, agent/execution.py

Rules:
- Pure data. No LLM calls. No RCON.
- Condition expressions (success_condition, failure_condition, milestone keys)
  are strings that reward_evaluator.py evaluates against a live WorldState.
  They are opaque here — this module does not execute them.
- All enum values that biter support will need (EMERGENCY, URGENT) are defined
  now even though they are unused until BITERS_ENABLED=True.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    """
    Ordered by urgency (higher int = higher urgency).
    IntEnum so goals can be compared directly: p1 > p2.

    EMERGENCY and URGENT are unused until biters are enabled, but they are
    defined now so the goal tree, preemption logic, and resource allocator
    can be written once and never revisited.
    """
    BACKGROUND = 0   # Long-running background work; freely preemptible
    NORMAL     = 1   # Standard factory-building goals
    URGENT     = 2   # Time-sensitive but not life-threatening (reserved)
    EMERGENCY  = 3   # Drop everything — biter attack, critical power failure


# ---------------------------------------------------------------------------
# GoalStatus
# ---------------------------------------------------------------------------

class GoalStatus(Enum):
    PENDING   = "pending"     # Created but not yet started
    ACTIVE    = "active"      # Currently being pursued
    SUSPENDED = "suspended"   # Preempted by higher-priority goal; resumable
    COMPLETE  = "complete"    # Success condition met
    FAILED    = "failed"      # Failure condition met or explicitly abandoned


# ---------------------------------------------------------------------------
# RewardSpec
# ---------------------------------------------------------------------------

@dataclass
class RewardSpec:
    """
    A structured definition of success, failure, milestones, and time pressure
    for a single Goal. Produced by the LLM alongside the Goal; evaluated
    mechanically by planning/reward_evaluator.py — no LLM during execution.

    Condition expressions in milestone_rewards are the same opaque string
    format as Goal.success_condition / failure_condition.

    Fields
    ------
    success_reward      : Reward on goal completion. Typically 1.0 for normal goals.
    failure_penalty     : Penalty (positive float, subtracted) on failure.
    milestone_rewards   : Partial rewards for intermediate achievements.
                          Keys are evaluable condition strings; values are floats.
    time_discount       : Per-tick decay factor applied to success_reward.
                          Final reward = success_reward * (time_discount ** ticks_elapsed).
                          Set to 1.0 for no time pressure.
    calibration_notes   : The LLM's own forward-looking assessment of how well this
                          spec captures what actually matters. Read by the reflection
                          call after goal resolution.
    """
    success_reward: float = 1.0
    failure_penalty: float = 0.5
    milestone_rewards: dict[str, float] = field(default_factory=dict)
    time_discount: float = 1.0          # 1.0 = no decay; <1.0 = timed pressure
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
        """Apply time discount to success reward."""
        return self.success_reward * (self.time_discount ** ticks_elapsed)


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    """
    A single planning objective, produced by the strategic or tactical LLM.

    Condition strings (success_condition, failure_condition, milestone keys in
    reward_spec) are evaluated by planning/reward_evaluator.py against a live
    WorldState. This class stores them as opaque strings only.

    The goal tree (planning/goal_tree.py) manages parent/child relationships
    and status transitions. Goals do not manage their own children.

    Fields
    ------
    description         : Human-readable statement of the objective.
    priority            : Priority enum — governs preemption in the goal tree.
    success_condition   : Evaluable expression; True when goal is achieved.
    failure_condition   : Evaluable expression; True when goal should be abandoned.
    reward_spec         : Associated RewardSpec for mechanical reward evaluation.
    parent_id           : ID of the parent Goal, or None for top-level goals.
    status              : Current lifecycle state.
    created_at          : Game tick at creation.
    resolved_at         : Game tick at completion/failure, or None if ongoing.
    id                  : UUID4 string, auto-generated.
    """
    description: str
    priority: Priority
    success_condition: str
    failure_condition: str
    reward_spec: RewardSpec

    # Hierarchy
    parent_id: Optional[str] = None

    # Lifecycle
    status: GoalStatus = GoalStatus.PENDING
    created_at: int = 0          # Game tick
    resolved_at: Optional[int] = None

    # Identity — always last so callers can use positional args up to reward_spec
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def activate(self, tick: int) -> None:
        if self.status not in (GoalStatus.PENDING, GoalStatus.SUSPENDED):
            raise RuntimeError(
                f"Cannot activate goal in status {self.status}; "
                "expected PENDING or SUSPENDED"
            )
        self.status = GoalStatus.ACTIVE
        if self.created_at == 0:
            self.created_at = tick

    def suspend(self) -> None:
        if self.status != GoalStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot suspend goal in status {self.status}; expected ACTIVE"
            )
        self.status = GoalStatus.SUSPENDED

    def complete(self, tick: int) -> None:
        self.status = GoalStatus.COMPLETE
        self.resolved_at = tick

    def fail(self, tick: int) -> None:
        self.status = GoalStatus.FAILED
        self.resolved_at = tick

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.status in (GoalStatus.COMPLETE, GoalStatus.FAILED)

    @property
    def is_active(self) -> bool:
        return self.status == GoalStatus.ACTIVE

    @property
    def ticks_elapsed(self) -> Optional[int]:
        """Ticks since activation, or None if not yet started."""
        if self.created_at == 0:
            return None
        end = self.resolved_at if self.resolved_at is not None else self.created_at
        return max(0, end - self.created_at)

    def __repr__(self) -> str:
        return (
            f"Goal({self.id[:8]}… priority={self.priority.name} "
            f"status={self.status.name} desc={self.description!r})"
        )


# ---------------------------------------------------------------------------
# Factory helpers (make it easy for LLM output parsers to construct goals)
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
    """
    Convenience constructor — keeps LLM output parsers free of RewardSpec
    boilerplate. All Goal + RewardSpec fields are exposed at one level.
    """
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
