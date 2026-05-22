"""
planning/reward_evaluator.py

RewardEvaluator — mechanically evaluates a Goal's conditions and RewardSpec
against a WorldQuery.

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

  The evaluator now receives a WorldQuery rather than a WorldState directly.
  This enforces the access boundary: the evaluator never touches WorldState
  fields directly; it uses only the WorldQuery interface.  WorldQuery.state
  is exposed as a namespace name so that existing condition strings like
  "state.game_time_seconds" and "state.ground_items" continue to work.

Condition scope:
  See CONDITION_SCOPE.md for the authoritative PROXIMAL vs NON-PROXIMAL
  breakdown and staleness helpers.

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

if TYPE_CHECKING:
    from world.query import WorldQuery
    from world.production_tracker import ProductionTrackerProtocol

log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    The result of evaluating a Goal against a WorldQuery snapshot.

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


class _DeltaView:
    """
    Provides delta access to world state relative to a goal-activation snapshot.

    Exposed in condition strings as the ``new`` namespace name:

        new.charted_chunks >= 5          # scalar property delta via __getattr__
        new.charted_tiles >= 5120        # any wq scalar property works automatically
        new.inventory('iron-ore') >= 10  # parameterized — explicit method
        new.resource_patches('coal') >= 1  # parameterized — explicit method
        new.elapsed_ticks > 1200         # ticks since goal activation

    Scalar wq properties get automatic delta support via __getattr__ — adding
    a new property to WorldQuery requires no changes here. Only parameterized
    lookups (inventory, resource_patches) need explicit methods.
    """

    def __init__(self, wq: "WorldQuery", snapshot: dict, elapsed_ticks: int = 0) -> None:
        # Use object.__setattr__ to avoid triggering our own __getattr__
        object.__setattr__(self, '_wq', wq)
        object.__setattr__(self, '_snapshot', snapshot)
        object.__setattr__(self, '_elapsed_ticks', elapsed_ticks)

    def __getattr__(self, name: str):
        """
        Automatic delta for scalar WorldQuery properties.
        new.charted_chunks → wq.charted_chunks - snapshot['charted_chunks']
        Clamped to 0. Raises AttributeError if the wq property doesn't exist.
        """
        wq       = object.__getattribute__(self, '_wq')
        snapshot = object.__getattribute__(self, '_snapshot')
        try:
            current = getattr(wq, name)
        except AttributeError:
            raise AttributeError(f"_DeltaView: no property {name!r} on WorldQuery")
        if callable(current):
            raise AttributeError(
                f"_DeltaView: {name!r} is callable — use an explicit method instead"
            )
        baseline = snapshot.get(name, current)
        try:
            return max(0, current - baseline)
        except TypeError:
            # Non-numeric property — return raw value, delta doesn't apply
            return current

    @property
    def tick(self) -> int:
        """Ticks elapsed since goal activation (current_tick - start_tick).
        Mirrors the top-level `tick` name: new.tick is the delta of tick.
        """
        return object.__getattribute__(self, '_elapsed_ticks')

    def inventory(self, item: str) -> int:
        """Net items of *item* collected since goal activation (clamped to 0)."""
        wq       = object.__getattribute__(self, '_wq')
        snapshot = object.__getattribute__(self, '_snapshot')
        snap_inv = snapshot.get("inventory", {})
        return max(0, wq.inventory_count(item)
                   - snap_inv.get(item, wq.inventory_count(item)))

    def resource_patches(self, resource_type: str) -> int:
        """Resource patches of *resource_type* discovered since goal activation."""
        wq       = object.__getattribute__(self, '_wq')
        snapshot = object.__getattribute__(self, '_snapshot')
        snap_rc  = snapshot.get("resource_counts", {})
        current  = len(wq.resources_of_type(resource_type))
        return max(0, current - snap_rc.get(resource_type, current))


class RewardEvaluator:
    """
    Evaluates a Goal's conditions and RewardSpec against a live WorldQuery.

    Primary API:
        result = evaluator.evaluate(goal, wq, tick=3600, start_tick=3000)

    Lower-level API (raw condition strings):
        result = evaluator.evaluate_conditions(
            success_condition, failure_condition, spec, wq, tick, start_tick
        )

    Parameters
    ----------
    tracker : ProductionTrackerProtocol, optional
        Provides production_rate(item) -> float.  When None, production_rate()
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
        wq: "WorldQuery",
        tick: int,
        start_tick: int,
        start_snapshot: Optional[dict] = None,
    ) -> EvaluationResult:
        return self.evaluate_conditions(
            success_condition=goal.success_condition,
            failure_condition=goal.failure_condition,
            spec=goal.reward_spec,
            wq=wq,
            tick=tick,
            start_tick=start_tick,
            start_snapshot=start_snapshot,
        )

    def evaluate_conditions(
        self,
        success_condition: str,
        failure_condition: str,
        spec: RewardSpec,
        wq: "WorldQuery",
        tick: int,
        start_tick: int,
        start_snapshot: Optional[dict] = None,
    ) -> EvaluationResult:
        elapsed = max(0, tick - start_tick)
        ns = self._build_namespace(wq, tick, elapsed, start_snapshot or {})

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

    def _build_namespace(self, wq: "WorldQuery", tick: int, elapsed_ticks: int = 0, start_snapshot: dict = None) -> dict:
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

        def staleness(section: str) -> Optional[int]:
            return wq.section_staleness(section, tick)

        ns: dict = {
            # The underlying WorldState, exposed for condition strings that
            # reference state.game_time_seconds, state.ground_items, etc.
            # Consumers should prefer the named namespace names below, but
            # "state.X" forms remain valid for backwards compatibility.
            "state":     wq.state,
            "tick":      tick,
            # elapsed_ticks: ticks since this goal was activated.
            # Use instead of raw tick for time-based conditions so
            # they are independent of absolute game clock.
            # e.g. failure_condition="elapsed_ticks > 1200"
            # elapsed_ticks: kept as a convenience alias for new.tick.
            # Prefer new.tick for consistency with the delta framework.
            "elapsed_ticks": elapsed_ticks,
            # Delta conditions — relative to goal activation snapshot.
            # Access via the `new` object: new.charted_chunks, new.inventory('iron-ore'), etc.
            # Adding a new delta only requires a method on _DeltaView — no per-item boilerplate.
            "new": _DeltaView(wq, start_snapshot or {}, elapsed_ticks),
            # WorldQuery itself, for composable query use in conditions.
            "wq":        wq,
            # Inventory — NON-PROXIMAL
            "inventory": wq.inventory_count,
            # Exploration — NON-PROXIMAL
            "charted_chunks":   wq.charted_chunks,
            "charted_tiles":    wq.charted_tiles,
            "charted_area_km2": wq.charted_area_km2,
            # Entities — PROXIMAL (scan radius only)
            "entities":           wq.entities_by_name,
            "entity_by_id":       wq.entity_by_id,
            "entities_by_status": wq.entities_by_status,
            # Research — NON-PROXIMAL
            "research":      wq.research,
            "tech_unlocked": wq.tech_unlocked,
            # Production rates — PROXIMAL (scan radius + tracker window)
            "production_rate": production_rate,
            # Staleness guard — META
            "staleness": staleness,
            # Logistics sub-objects — PROXIMAL
            "logistics": wq.logistics,
            "power":     wq.power,
            "threat":    wq.threat,
            # Connectivity queries — PROXIMAL
            "inserters_from":      wq.inserters_taking_from,
            "inserters_to":        wq.inserters_delivering_to,
            "inserters_from_type": wq.inserters_taking_from_type,
            "inserters_to_type":   wq.inserters_delivering_to_type,
            # Resource patches — NON-PROXIMAL
            "resources_of_type": wq.resources_of_type,
        }
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