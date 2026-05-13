"""
agent/network/agent_protocol.py

AgentProtocol — the interface every agent in the execution network must satisfy.

The coordinator interacts with individual agents exclusively through this
interface. Concrete agent implementations (navigation, production,
spatial-logistics) are substitutable without touching the coordinator.

Rules
-----
- Pure interface. No learning logic. No game interaction.
- Agents receive the blackboard and WorldQuery on every tick; they do not
  hold references to either between ticks.
- Agents return a list of Actions from tick(). The coordinator resolves
  conflicts and assembles the final ExecutionResult.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bridge.actions import Action
from planning.goal import Goal

if TYPE_CHECKING:
    from agent.blackboard import Blackboard
    from world.query import WorldQuery
    from world.writer import WorldWriter


class AgentProtocol:
    """
    Interface for a single agent in the execution network.

    Methods
    -------
    activate(goal, blackboard, wq)
        Called once when the goal is assigned to this agent. The agent
        configures its observation space relative to the goal structure
        and reads any warm-start data from behavioral memory. Does not
        return anything — side effects go to the blackboard.

    tick(blackboard, wq, ww, tick) -> list[Action]
        Called every poll cycle. The agent reads WorldQuery and the
        blackboard, writes observations and reservations to the blackboard,
        and returns its candidate action list. The coordinator selects from
        and orders the returned actions.

    observe(blackboard, wq) -> dict
        Return a flat dict of named observations from this agent's current
        internal state. Called by the coordinator to build the combined
        observation for examination and progress tracking.

    progress(blackboard, wq) -> float
        Return an estimate of progress toward the current goal in [0.0, 1.0].
        Derived from the agent's internal state, not from RewardEvaluator.
        The coordinator aggregates progress across agents.
    """

    def activate(
        self,
        goal: Goal,
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> None:
        raise NotImplementedError

    def tick(
        self,
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        raise NotImplementedError

    def observe(
        self,
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> dict:
        raise NotImplementedError

    def progress(
        self,
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> float:
        raise NotImplementedError
