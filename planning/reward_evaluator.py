"""
planning/reward_evaluator.py

RewardEvaluator — mechanically evaluates a Goal's conditions and RewardSpec
against a WorldState.

Rules:
- Pure computation. No LLM calls. No RCON. No side effects.
- Condition strings come from the agent's own LLM outputs — eval() is
  acceptable. All exceptions are caught and treated as non-triggering.
- Dangerous modules (os, subprocess, sys, etc.) are excluded from the
  eval namespace.
- Time discounting applies to success_reward and failure_penalty only.
  Milestone rewards are point-in-time and are not discounted.

Design note:
  success_condition and failure_condition live on Goal (not RewardSpec).
  RewardSpec holds the *numeric* reward shape. The evaluator accepts the full
  Goal so it can read both. A lower-level evaluate_conditions() helper accepts
  raw strings directly (useful for tests and introspection).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from planning.goal import Goal, RewardSpec
from world.state import WorldState

log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    The result of evaluating a Goal against a WorldState snapshot.

    Fields
    ------
    success         : True if the success_condition evaluated to a truthy value.
    failure         : True if the failure_condition evaluated to a truthy value.
    reward          : Total reward accumulated this evaluation.
    milestones_hit  : Condition strings of milestones that triggered this tick.
    elapsed_ticks   : Ticks since goal start (current_tick - start_tick).
    """
    success: bool = False
    failure: bool = False
    reward: float = 0.0
    milestones_hit: list[str] = field(default_factory=list)
    elapsed_ticks: int = 0


class RewardEvaluator:
    """
    Evaluates a Goal's conditions and RewardSpec against a live WorldState snapshot.

    Primary API:
        result = evaluator.evaluate(goal, state, tick=3600, start_tick=3000)

    Lower-level API (raw condition strings):
        result = evaluator.evaluate_conditions(
            success_condition, failure_condition, spec, state, tick, start_tick
        )
    """

    _BLOCKED_NAMES = frozenset({
        "os", "sys", "subprocess", "importlib", "builtins",
        "open", "exec", "compile", "__import__",
    })

    def evaluate(
        self,
        goal: Goal,
        state: WorldState,
        tick: int,
        start_tick: int,
    ) -> EvaluationResult:
        return self.evaluate_conditions(
            success_condition=goal.success_condition,
            failure_condition=goal.failure_condition,
            spec=goal.reward_spec,
            state=state,
            tick=tick,
            start_tick=start_tick,
        )

    def evaluate_conditions(
        self,
        success_condition: str,
        failure_condition: str,
        spec: RewardSpec,
        state: WorldState,
        tick: int,
        start_tick: int,
    ) -> EvaluationResult:
        elapsed = max(0, tick - start_tick)
        ns = self._build_namespace(state, tick)

        success = self._eval_bool(success_condition, ns, "success_condition")
        failure = self._eval_bool(failure_condition, ns, "failure_condition")

        reward = 0.0
        milestones_hit: list[str] = []

        if success:
            reward += spec.discounted_success_reward(elapsed)
        elif failure:
            reward -= spec.discounted_success_reward(elapsed) * spec.failure_penalty

        for condition, milestone_reward in spec.milestone_rewards.items():
            if self._eval_bool(condition, ns, f"milestone({condition!r})"):
                reward += milestone_reward
                milestones_hit.append(condition)

        return EvaluationResult(
            success=success,
            failure=failure,
            reward=reward,
            milestones_hit=milestones_hit,
            elapsed_ticks=elapsed,
        )

    def _build_namespace(self, state: WorldState, tick: int) -> dict:
        ns: dict = {
            "state":     state,
            "inventory": state.inventory_count,
            "entities":  state.entities_by_name,
            "tick":      tick,
            "research":  state.research,
            "logistics": state.logistics,
            "threat":    state.threat,
        }
        # Strip dangerous builtins
        raw_builtins = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)  # type: ignore
        ns["__builtins__"] = {
            k: v for k, v in raw_builtins.items()
            if k not in self._BLOCKED_NAMES
        }
        return ns

    def _eval_bool(self, expression: str, namespace: dict, label: str) -> bool:
        if not expression or not expression.strip():
            return False
        try:
            return bool(eval(expression, namespace))  # noqa: S307
        except Exception as exc:
            log.warning("RewardEvaluator: exception in %s %r: %s", label, expression, exc)
            return False
