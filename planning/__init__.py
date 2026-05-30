"""
planning/

Goal and task lifecycle management. Pure logic — no LLM calls, no RCON.

"Planning" means: what are we trying to do, has it succeeded or failed,
and what reward did we earn?

Subpackages
-----------
goals/       Goal, GoalTree, GoalSource, GoalQueue — what the agent is
             trying to accomplish and where goals come from.
tasks/       Task (née Subtask), TaskLedger — the coordinator's derived
             work units handed to agents.
evaluation/  RewardEvaluator, condition_namespace, ResourceAllocator —
             mechanical evaluation of condition strings and reward shaping.

Public surface
--------------
All imports from planning/ go through this file.
"""

# --- Goals ------------------------------------------------------------------
from planning.goals.goal import Goal, GoalStatus, Priority, RewardSpec, make_goal
from planning.goals.goal_tree import GoalTree
from planning.goals.goal_source import GoalSource, GoalQueue, GoalQueueEntry

# --- Tasks ------------------------------------------------------------------
from planning.tasks.task import Task, TaskStatus, TaskRecord
from planning.tasks.task_ledger import TaskLedger

# --- Evaluation -------------------------------------------------------------
from planning.evaluation.reward_evaluator import RewardEvaluator, EvaluationResult
from planning.evaluation.condition_namespace import (
    build_core_namespace,
    safe_builtins,
    BLOCKED_NAMES,
)
from planning.evaluation.condition_parser import params_from_condition
from planning.evaluation.resource_allocator import ResourceAllocator

__all__ = [
    # Goals
    "Goal", "GoalStatus", "Priority", "RewardSpec", "make_goal",
    "GoalTree",
    "GoalSource", "GoalQueue", "GoalQueueEntry",
    # Tasks
    "Task", "TaskStatus", "TaskRecord",
    "TaskLedger",
    # Evaluation
    "RewardEvaluator", "EvaluationResult",
    "build_core_namespace", "safe_builtins", "BLOCKED_NAMES",
    params_from_condition, "ResourceAllocator",
]
