"""
execution/agents/base.py

AgentProtocol — the interface every agent in the execution network must satisfy.

The coordinator interacts with individual agents exclusively through this
interface. Concrete agent implementations (navigation, mining, production,
spatial-logistics) are substitutable without touching the coordinator.

Boundary
--------
Agents interact with Tasks, not Goals. Goals are the coordinator's concern.
The coordinator translates a Goal into Tasks and hands each Task to the
appropriate agent. Agents never receive or inspect the Goal object — they know
only what they need to do from the active Task and the blackboard.

Rules
-----
- Pure interface. No learning logic. No game interaction.
- Agents receive the active Task, the blackboard, WorldQuery, and
  KnowledgeBase on every tick; they do not hold references to any of these
  between ticks (except the task stored in activate() for context).
- Agents return a list of Actions from tick(). The coordinator assembles the
  final ExecutionResult.
- kb is provided to all methods for consistency. Agents that don't need it
  (navigation, mining) simply ignore it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bridge import Action
from world.model.patch import SelfModelPatch

if TYPE_CHECKING:
    from execution.blackboard import Blackboard
    from planning.tasks.task import Task
    from world import KnowledgeBase
    from world import WorldQuery
    from world import WorldWriter


class AgentProtocol:
    """
    Interface for a single agent in the execution network.

    Methods
    -------
    activate(task, blackboard, wq, kb)
        Called once when a new task is assigned to this agent. The agent
        configures its internal state relative to the task and may write
        initial observations to the blackboard. Does not return anything.

        Called by the coordinator when a task becomes active (i.e. when
        the task is pushed to the top of the ledger), not at goal start.
        Each new task triggers a fresh activate() call.

    tick(task, blackboard, wq, ww, tick, kb) -> list[Action]
        Called every poll cycle while this agent owns the active task.
        The agent reads WorldQuery and the blackboard, writes observations
        and reservations to the blackboard, and returns its candidate action
        list. The coordinator selects from and orders the returned actions.

    observe(task, blackboard, wq, kb) -> dict
        Return a flat dict of named observations from this agent's current
        internal state. Called by the coordinator to build the combined
        observation for examination and progress tracking.

    progress(task, blackboard, wq, kb) -> float
        Return an estimate of progress toward the current task in
        [0.0, 1.0]. Derived from the agent's internal state, not from
        RewardEvaluator. The coordinator aggregates progress across tasks.

    pending_patches() -> list[SelfModelPatch]
        Return and clear all self-model patches accumulated since the last
        call. Called by the coordinator after each tick. Default returns []
        — agents that never produce patches need not override.
    """

    def activate(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> None:
        raise NotImplementedError

    def tick(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
        kb: "KnowledgeBase",
    ) -> list[Action]:
        raise NotImplementedError

    def observe(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> dict:
        raise NotImplementedError

    def progress(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> float:
        raise NotImplementedError

    def pending_patches(self) -> list[SelfModelPatch]:
        """
        Return and clear accumulated self-model patches since the last call.

        Called by the coordinator after each agent tick. Agents that produce
        patches override this; agents that do not need not override.
        """
        return []