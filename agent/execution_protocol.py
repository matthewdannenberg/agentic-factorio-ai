"""
agent/execution_protocol.py

ExecutionLayerProtocol — the sole interface between the planning layer and the
agent execution network.

The planning layer (main loop, goal tree, examination layer) interacts with
execution exclusively through this protocol. The multi-agent system behind it
is invisible to all callers.

Rules
-----
- Pure interface definitions and dataclasses. No learning logic. No game interaction.
- ExecutionResult carries everything the main loop needs from one tick.
- StuckContext carries everything the LLM layer needs to reason about a stuck
  goal: the full nesting chain from goal to failure point, and the resolved
  sibling history at every level of that chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

from bridge.actions import Action
from planning.goal import Goal

if TYPE_CHECKING:
    from agent.subtask import Subtask, SubtaskRecord
    from world.query import WorldQuery
    from world.writer import WorldWriter


# ---------------------------------------------------------------------------
# ExecutionStatus
# ---------------------------------------------------------------------------

class ExecutionStatus(Enum):
    """
    Coarse signal from the execution network back to the main loop.

    PROGRESSING : The network is making observable progress toward the goal.
                  Normal operating state; continue ticking.
    WAITING     : Blocked on a game condition (crafting timer, belt filling,
                  research completing). No action is useful right now; the
                  main loop may reduce poll frequency.
    STUCK       : The network cannot advance. The LLM layer should be
                  consulted. ExecutionResult.stuck_context is populated.
    COMPLETE    : The execution network believes the goal has been achieved.
                  The main loop should trigger examination and goal resolution.
    """
    PROGRESSING = auto()
    WAITING     = auto()
    STUCK       = auto()
    COMPLETE    = auto()


# ---------------------------------------------------------------------------
# StuckContext
# ---------------------------------------------------------------------------

@dataclass
class StuckContext:
    """
    Diagnostic payload produced when the execution network reports STUCK.

    Passed to the LLM layer so it can reason about the blocked goal and
    produce targeted decomposition subgoals. Carries a complete picture of
    what was attempted at every level of the subtask hierarchy — not just
    the immediate failure point.

    Fields
    ------
    goal            : The Goal that is stuck.

    failure_chain   : The live subtask nesting chain at the time of failure,
                      ordered outermost-to-innermost. The last element is the
                      subtask where execution is stuck. Empty if the network
                      got stuck at goal level before deriving any subtask.

                      Example: [Task2, Task3] means Task2 is the direct child
                      of the goal, and Task3 (nested inside Task2) is the
                      actual failure point.

    sibling_history : Resolved sibling SubtaskRecords at each level of the
                      chain, keyed by the parent's id. Levels with no resolved
                      siblings are omitted.

                      Example: {goal.id: [Task1✓], task2.id: [TaskA✓]}
                      means Task1 was completed before Task2 was attempted,
                      and TaskA was completed before Task3 was attempted.

                      This lets the LLM see the full tree:
                          Goal1
                          ├── Task1  ✓  (sibling_history[goal.id])
                          └── Task2  ✗  (failure_chain[0])
                              ├── TaskA  ✓  (sibling_history[task2.id])
                              └── Task3  ✗  (failure_chain[1])  ← stuck here

    blackboard_snapshot : Plain-dict snapshot of the blackboard at time of
                          failure. Produced by Blackboard.snapshot(). Contains
                          observations, intentions, and reservations that were
                          live at the moment the network declared STUCK.

    Why the full chain matters
    --------------------------
    The LLM must decide at which level to intervene. It might:
      - Decompose the leaf failure (Task3) if it's a narrow, fixable gap
      - Discard the parent's approach (Task2) if Task3's failure reveals a
        flawed strategy, and propose a different Task2 decomposition
      - In extreme cases, revise the goal itself

    None of these decisions can be made well if the LLM only sees the leaf.
    The chain and sibling history together give it the strategic context it
    needs without requiring access to the full execution network internals.
    """
    goal: Goal
    failure_chain: list["Subtask"]
    sibling_history: dict[str, list["SubtaskRecord"]]
    blackboard_snapshot: dict

    def to_dict(self) -> dict:
        """
        Serialise to a plain dict for logging or LLM prompt construction.

        Renders the goal, the failure chain (outermost to innermost), and the
        sibling history at each level. Subtask objects are represented by their
        id and description only; full SubtaskRecord dicts are included for
        siblings (they carry outcome and children_ids).
        """
        return {
            "goal_id": self.goal.id,
            "goal_description": self.goal.description,
            "failure_chain": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status.name,
                    "derived_locally": t.derived_locally,
                }
                for t in self.failure_chain
            ],
            "sibling_history": {
                parent_id: [rec.to_dict() for rec in records]
                for parent_id, records in self.sibling_history.items()
            },
            "blackboard_snapshot": self.blackboard_snapshot,
        }

    @property
    def immediate_failure(self) -> Optional["Subtask"]:
        """
        The deepest subtask in the chain — the immediate failure point.
        None if the network got stuck at goal level (empty chain).
        """
        return self.failure_chain[-1] if self.failure_chain else None

    @property
    def stuck_at_goal_level(self) -> bool:
        """True if no subtask had been derived when the network got stuck."""
        return len(self.failure_chain) == 0


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """
    The complete output of one execution tick.

    Returned by ExecutionLayerProtocol.tick() and CoordinatorProtocol.tick().

    Fields
    ------
    actions      : Ordered list of actions to dispatch this tick. May be empty
                   (e.g. when status is WAITING).
    status       : Coarse signal describing the network's current state.
    stuck_context: Populated when status is STUCK; None otherwise. The main
                   loop passes this to the LLM layer for decomposition.
    """
    actions: list[Action]
    status: ExecutionStatus
    stuck_context: Optional[StuckContext] = None

    def __post_init__(self) -> None:
        if self.status == ExecutionStatus.STUCK and self.stuck_context is None:
            raise ValueError(
                "ExecutionResult with status STUCK must carry a stuck_context"
            )
        if self.status != ExecutionStatus.STUCK and self.stuck_context is not None:
            raise ValueError(
                f"ExecutionResult with status {self.status.name} must not carry "
                "a stuck_context (only STUCK results carry one)"
            )


# ---------------------------------------------------------------------------
# ExecutionLayerProtocol
# ---------------------------------------------------------------------------

class ExecutionLayerProtocol:
    """
    Protocol for the agent execution network.

    The planning layer calls reset() at the start of each new goal and tick()
    on every poll cycle. progress() and observe() are available to the
    examination layer and main loop at any time.

    Concrete implementations satisfy this interface by providing all four
    methods with compatible signatures. The concrete type is injected at
    startup; callers depend only on this interface.

    Methods
    -------
    reset(goal, wq, seed_subtasks=None)
        Prepare the network for a new goal. Clears the blackboard, subtask
        ledger, and any goal-scoped state. Queries behavioral memory for
        warm-starting candidates.

        seed_subtasks: optional list of Subtask objects injected by the LLM
        after an escalation (derived_locally=False). When provided, the
        coordinator pre-populates its ledger from these rather than starting
        derivation from scratch. This is how LLM decompositions re-enter the
        execution network at the right level without bypassing the subtask
        lifecycle machinery.

    tick(goal, wq, ww, tick) -> ExecutionResult
        Advance the execution network by one poll cycle. Agents read from wq,
        write actions via ww-backed Action objects, and update the blackboard.
        Returns the assembled ExecutionResult.

    progress(goal, wq) -> float
        Estimate of progress toward the goal in [0.0, 1.0]. Derived from the
        network's internal state — not from RewardEvaluator. Used by the
        examination layer and main loop for monitoring.

    observe(wq) -> dict
        Flat dict of named observations from the network's current state.
        Used by the examination layer to build the structured factory summary.
    """

    def reset(
        self,
        goal: Goal,
        wq: "WorldQuery",
        seed_subtasks: Optional[list["Subtask"]] = None,
    ) -> None:
        raise NotImplementedError

    def tick(
        self,
        goal: Goal,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> ExecutionResult:
        raise NotImplementedError

    def progress(self, goal: Goal, wq: "WorldQuery") -> float:
        raise NotImplementedError

    def observe(self, wq: "WorldQuery") -> dict:
        raise NotImplementedError