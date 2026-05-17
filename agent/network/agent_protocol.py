"""
agent/network/agent_protocol.py

AgentProtocol — the interface every agent in the execution network must satisfy.

The coordinator interacts with individual agents exclusively through this
interface. Concrete agent implementations (navigation, mining, production,
spatial-logistics) are substitutable without touching the coordinator.

Boundary
--------
Agents interact with Subtasks, not Goals. Goals are the coordinator's concern.
The coordinator translates a Goal into Subtasks and hands each Subtask to the
appropriate agent. Agents never receive or inspect the Goal object — they know
only what they need to do from the active Subtask and the blackboard.

Rules
-----
- Pure interface. No learning logic. No game interaction.
- Agents receive the active Subtask, the blackboard, and WorldQuery on every
  tick; they do not hold references to any of these between ticks (except the
  subtask stored in activate() for context).
- Agents return a list of Actions from tick(). The coordinator assembles the
  final ExecutionResult.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bridge.actions import Action

if TYPE_CHECKING:
    from agent.blackboard import Blackboard
    from agent.subtask import Subtask
    from world.query import WorldQuery
    from world.writer import WorldWriter


class AgentProtocol:
    """
    Interface for a single agent in the execution network.

    Methods
    -------
    activate(subtask, blackboard, wq)
        Called once when a new subtask is assigned to this agent. The agent
        configures its internal state relative to the subtask and may write
        initial observations to the blackboard. Does not return anything.

        Called by the coordinator when a subtask becomes active (i.e. when
        the subtask is pushed to the top of the ledger), not at goal start.
        Each new subtask triggers a fresh activate() call.

    tick(subtask, blackboard, wq, ww, tick) -> list[Action]
        Called every poll cycle while this agent owns the active subtask.
        The agent reads WorldQuery and the blackboard, writes observations
        and reservations to the blackboard, and returns its candidate action
        list. The coordinator selects from and orders the returned actions.

    observe(subtask, blackboard, wq) -> dict
        Return a flat dict of named observations from this agent's current
        internal state. Called by the coordinator to build the combined
        observation for examination and progress tracking.

    progress(subtask, blackboard, wq) -> float
        Return an estimate of progress toward the current subtask in
        [0.0, 1.0]. Derived from the agent's internal state, not from
        RewardEvaluator. The coordinator aggregates progress across subtasks.
    """

    def activate(
        self,
        subtask: "Subtask",
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> None:
        raise NotImplementedError

    def tick(
        self,
        subtask: "Subtask",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        raise NotImplementedError

    def observe(
        self,
        subtask: "Subtask",
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> dict:
        raise NotImplementedError

    def progress(
        self,
        subtask: "Subtask",
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> float:
        raise NotImplementedError