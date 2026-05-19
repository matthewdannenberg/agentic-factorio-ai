"""
llm/goal_source.py

GoalSource — the interface the loop uses to obtain goals and handle escalation.

This module provides:
  - GoalSource       : abstract protocol (what the loop depends on)
  - GoalQueue        : manual / recorded goal sequence implementation
  - GoalQueueEntry   : serialisable representation of one queued goal

The GoalQueue fills the role of the LLM in Phases 6–10. It holds an ordered
list of goals and dispenses them one at a time when the loop requests the next
goal. When the queue is exhausted the loop exits cleanly.

Long-term value
---------------
A GoalQueue loaded from a JSON file is a recorded goal sequence — the exact
output a real LLM produced on a previous run. Playing it back lets you:
  - Reproduce a run deterministically (given the same world seed)
  - Run the agent without making API calls while debugging lower layers
  - Build regression tests against a known goal sequence

When the real LLM client lands in Phase 11 it will implement GoalSource
directly. The loop will not change — only the injected implementation differs.

Stuck handling
--------------
GoalQueue.handle_stuck() logs the StuckContext and returns [] (no seed
subtasks). This means the coordinator stays stuck until the subtask's
failure_condition fires and the goal is marked failed. The loop then requests
the next goal from the queue and continues.

A real LLM implementation would inspect the StuckContext, decompose the
failure, and return seed subtasks for the coordinator to resume with.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from planning.goal import Goal, GoalStatus, Priority, make_goal

if TYPE_CHECKING:
    from agent.subtask import Subtask
    from agent.execution_protocol import StuckContext

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GoalQueueEntry — serialisable form of a queued goal
# ---------------------------------------------------------------------------

@dataclass
class GoalQueueEntry:
    """
    A single goal in a GoalQueue, in a form that can be serialised to JSON
    and reconstructed without loss.

    All fields map directly to make_goal() parameters so that
    GoalQueueEntry.to_goal() is a thin wrapper with no interpretation.
    """
    description: str
    success_condition: str
    failure_condition: str
    goal_type: str = "collection"          # matches GOAL_TYPE_* constants
    priority: str = "NORMAL"               # Priority enum name
    success_reward: float = 1.0
    failure_penalty: float = 0.5
    milestone_rewards: dict = field(default_factory=dict)
    time_discount: float = 1.0
    calibration_notes: str = ""

    def to_goal(self) -> Goal:
        """Construct a Goal from this entry."""
        priority = Priority[self.priority]
        goal = make_goal(
            description=self.description,
            success_condition=self.success_condition,
            failure_condition=self.failure_condition,
            priority=priority,
            success_reward=self.success_reward,
            failure_penalty=self.failure_penalty,
            milestone_rewards=self.milestone_rewards,
            time_discount=self.time_discount,
            calibration_notes=self.calibration_notes,
        )
        # Store the goal type as a dynamic attribute — the coordinator reads it
        # via getattr(goal, "type", "") to determine derivation strategy.
        goal.type = self.goal_type
        return goal

    @classmethod
    def from_goal(cls, goal: Goal) -> "GoalQueueEntry":
        """
        Capture a live Goal as a GoalQueueEntry for recording.

        Used by the loop to persist LLM-produced goals for later replay.
        """
        return cls(
            description=goal.description,
            success_condition=goal.success_condition,
            failure_condition=goal.failure_condition,
            goal_type=getattr(goal, "type", "collection"),
            priority=goal.priority.name,
            success_reward=goal.reward_spec.success_reward,
            failure_penalty=goal.reward_spec.failure_penalty,
            milestone_rewards=dict(goal.reward_spec.milestone_rewards),
            time_discount=goal.reward_spec.time_discount,
            calibration_notes=goal.reward_spec.calibration_notes,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GoalQueueEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# GoalSource — the protocol the loop depends on
# ---------------------------------------------------------------------------

class GoalSource:
    """
    Abstract interface for anything that produces Goals for the loop.

    The loop calls next_goal() at startup and after each goal resolves.
    It calls handle_stuck() whenever the execution network reports STUCK.

    Concrete implementations:
      GoalQueue — manual / recorded sequence (this module)
      LLMClient — real LLM, Phase 11 (llm/client.py, not yet implemented)
    """

    def next_goal(self, context: dict) -> Optional[Goal]:
        """
        Return the next Goal to pursue, or None to signal clean shutdown.

        Parameters
        ----------
        context : dict
            Summary of current world state and performance history, as the
            examination layer would provide. GoalQueue ignores this; a real
            LLM uses it to select and configure the next goal.
        """
        raise NotImplementedError

    def handle_stuck(self, stuck_context: "StuckContext") -> list["Subtask"]:
        """
        Called when the execution network reports STUCK.

        Returns a list of seed Subtasks for the coordinator to resume with,
        or [] to let the goal fail naturally (timeout → mark failed → next
        goal). GoalQueue always returns [].

        Parameters
        ----------
        stuck_context : StuckContext
            Full execution context at the time of the STUCK report.
        """
        raise NotImplementedError

    def record_outcome(self, goal: Goal, reward: float) -> None:
        """
        Notify the goal source that a goal has resolved.

        GoalQueue logs the outcome. A real LLM client may use this to
        calibrate future goals or update a reflection record.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# GoalQueue — manual / recorded goal sequence
# ---------------------------------------------------------------------------

class GoalQueue(GoalSource):
    """
    An ordered sequence of goals dispensed one at a time.

    Construct from a Python list for programmatic use (tests, scripts):

        queue = GoalQueue([
            GoalQueueEntry(
                description="Collect 50 iron ore",
                success_condition="inventory('iron-ore') >= 50",
                failure_condition="tick > 36000",
                goal_type="collection",
            ),
        ])

    Load from a JSON file for replay:

        queue = GoalQueue.from_file("runs/goals_2026-05-19.json")

    Save to a JSON file to record a session:

        queue.save("runs/goals_2026-05-19.json")

    Parameters
    ----------
    entries      : Ordered list of GoalQueueEntry objects.
    loop_forever : If True, restart from the beginning when the queue is
                   exhausted instead of returning None. Useful for stress
                   testing. Default False.
    """

    def __init__(
        self,
        entries: list[GoalQueueEntry],
        loop_forever: bool = False,
    ) -> None:
        self._entries: list[GoalQueueEntry] = list(entries)
        self._index: int = 0
        self._loop_forever = loop_forever
        self._outcomes: list[dict] = []   # recorded for save()

    # ------------------------------------------------------------------
    # GoalSource interface
    # ------------------------------------------------------------------

    def next_goal(self, context: dict) -> Optional[Goal]:
        """
        Return the next goal in the queue, or None when exhausted.

        Logs the goal description so the operator can see what the agent
        is about to attempt.
        """
        if self._index >= len(self._entries):
            if self._loop_forever:
                log.info("GoalQueue: restarting from beginning")
                self._index = 0
            else:
                log.info("GoalQueue: all goals exhausted — signalling shutdown")
                return None

        entry = self._entries[self._index]
        self._index += 1
        goal = entry.to_goal()
        log.info(
            "GoalQueue: dispensing goal %d/%d — %s",
            self._index, len(self._entries), goal.description,
        )
        return goal

    def handle_stuck(self, stuck_context: "StuckContext") -> list["Subtask"]:
        """
        Log the stuck context and return no seed subtasks.

        The goal will eventually fail via its failure_condition and the loop
        will request the next goal. A real LLM would decompose the failure
        and return seed subtasks here.
        """
        log.warning(
            "GoalQueue.handle_stuck: goal %s stuck at %s — no decomposition "
            "available (GoalQueue is a stub; real LLM handles this in Phase 11). "
            "Goal will fail naturally via failure_condition.",
            stuck_context.goal.id[:8],
            (
                stuck_context.immediate_failure.description
                if stuck_context.immediate_failure
                else "goal level"
            ),
        )
        return []

    def record_outcome(self, goal: Goal, reward: float) -> None:
        """Record the outcome of a completed or failed goal."""
        outcome = {
            "goal_id": goal.id,
            "description": goal.description,
            "status": goal.status.name,
            "reward": reward,
            "ticks_elapsed": goal.ticks_elapsed,
        }
        self._outcomes.append(outcome)
        log.info(
            "GoalQueue: goal %s resolved — status=%s reward=%.3f ticks=%s",
            goal.id[:8],
            goal.status.name,
            reward,
            goal.ticks_elapsed,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path, loop_forever: bool = False) -> "GoalQueue":
        """
        Load a GoalQueue from a JSON file.

        The file must be a JSON array of GoalQueueEntry dicts, as produced
        by GoalQueue.save().

        Raises FileNotFoundError or json.JSONDecodeError on bad input.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        entries = [GoalQueueEntry.from_dict(d) for d in raw]
        log.info(
            "GoalQueue: loaded %d goals from %s", len(entries), path
        )
        return cls(entries, loop_forever=loop_forever)

    def save(self, path: str | Path) -> None:
        """
        Save the current queue entries to a JSON file.

        The outcomes list (recorded by record_outcome()) is saved alongside
        the entries in a separate key, so a saved file captures both the
        plan and what actually happened.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "goals": [e.to_dict() for e in self._entries],
            "outcomes": self._outcomes,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log.info("GoalQueue: saved %d goals to %s", len(self._entries), path)

    @classmethod
    def load_with_outcomes(cls, path: str | Path) -> "GoalQueue":
        """
        Load from a file that contains both goals and outcomes (as saved by
        save()). Outcomes are restored into _outcomes for inspection.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        entries = [GoalQueueEntry.from_dict(d) for d in data.get("goals", [])]
        queue = cls(entries)
        queue._outcomes = data.get("outcomes", [])
        return queue

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def remaining(self) -> int:
        """Number of goals not yet dispensed."""
        return max(0, len(self._entries) - self._index)

    def outcomes(self) -> list[dict]:
        """Recorded outcomes for all resolved goals this session."""
        return list(self._outcomes)

    def append(self, entry: GoalQueueEntry) -> None:
        """Add a goal to the end of the queue at runtime."""
        self._entries.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"GoalQueue({len(self._entries)} goals, "
            f"index={self._index}, "
            f"remaining={self.remaining()})"
        )
