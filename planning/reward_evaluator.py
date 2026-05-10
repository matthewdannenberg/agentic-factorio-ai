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

Condition scope:
  Not all conditions have the same evidence quality. See CONDITION_SCOPE.md
  for the authoritative breakdown of PROXIMAL vs NON-PROXIMAL conditions and
  the staleness helpers available in the eval namespace.

  Short version:
    NON-PROXIMAL (safe anywhere):
      inventory(item)        player carries inventory everywhere
      tech_unlocked(tech)    research state is global
      resources_of_type(t)   accumulated across all visits
      charted_chunks         force chart size, global and monotonic
      charted_tiles          charted_chunks x 1024
      charted_area_km2       charted_tiles / 1,000,000
      tick / game_time_seconds  always current

    PROXIMAL (requires player proximity):
      production_rate(item)  scan radius only, tracker-backed
      entities(name)         scan radius only
      staleness(section)     use to guard proximal conditions explicitly
      logistics / power      scan radius only
      inserters_from/to      scan radius only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from planning.goal import Goal, RewardSpec
from world.state import WorldState

if TYPE_CHECKING:
    from world.production_tracker import ProductionTrackerProtocol

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

    Parameters
    ----------
    tracker : ProductionTrackerProtocol, optional
        A ProductionTracker (or compatible object) that provides
        production_rate(item) -> float.  When provided, the eval namespace
        gains a production_rate() shorthand.  When None, production_rate()
        always returns 0.0 and logs a warning on first use.
    """

    _BLOCKED_NAMES = frozenset({
        "os", "sys", "subprocess", "importlib", "builtins",
        "open", "exec", "compile", "__import__",
    })

    def __init__(self, tracker: Optional["ProductionTrackerProtocol"] = None) -> None:
        self._tracker = tracker

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
        # production_rate: use tracker if available, else a safe zero-returning stub
        # that warns once so the LLM prompt designer notices the gap.
        if self._tracker is not None:
            production_rate = self._tracker.rate
        else:
            def production_rate(item: str) -> float:  # type: ignore[misc]
                log.warning(
                    "production_rate(%r) called but no ProductionTracker is "
                    "attached to this RewardEvaluator — returning 0.0. "
                    "Pass tracker= when constructing RewardEvaluator in the "
                    "main loop. See CONDITION_SCOPE.md.", item
                )
                return 0.0

        # staleness(section) -> int | None
        # Exposes WorldState.section_staleness() as a first-class namespace
        # function so conditions can guard themselves:
        #   staleness('entities') is not None and staleness('entities') < 300
        def staleness(section: str) -> Optional[int]:
            return state.section_staleness(section, tick)

        ns: dict = {
            # Top-level state
            "state":     state,
            "tick":      tick,
            # Inventory — NON-PROXIMAL (travels with player)
            "inventory": state.inventory_count,
            # Exploration — NON-PROXIMAL (global force chart, monotonically increasing)
            "charted_chunks":   state.player.exploration.charted_chunks,
            "charted_tiles":    state.player.exploration.charted_tiles,
            "charted_area_km2": state.player.exploration.charted_area_km2,
            # Entities — PROXIMAL (scan radius only)
            "entities":  state.entities_by_name,
            "entity_by_id": state.entity_by_id,
            "entities_by_status": state.entities_by_status,
            # Research — NON-PROXIMAL (global)
            "research":  state.research,
            "tech_unlocked": state.research.is_unlocked,
            # Production rates — PROXIMAL (scan radius only, tracker-backed)
            "production_rate": production_rate,
            # Staleness guard — use to make proximal conditions self-defending
            "staleness": staleness,
            # Logistics sub-objects — PROXIMAL
            "logistics": state.logistics,
            "power":     state.logistics.power,
            "threat":    state.threat,
            # Connectivity queries — PROXIMAL
            "inserters_from":      state.inserters_taking_from,
            "inserters_to":        state.inserters_delivering_to,
            "inserters_from_type": state.inserters_taking_from_type,
            "inserters_to_type":   state.inserters_delivering_to_type,
            # Resource patches — NON-PROXIMAL (accumulated across all visits)
            "resources_of_type": state.resources_of_type,
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