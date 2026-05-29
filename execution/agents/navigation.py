"""
execution/agents/navigation.py

NavigationAgent — walks the player to a position or entity.

Task types handled
------------------
  navigate_to_position   Walk to a tile-space position.
  navigate_to_entity     Walk to within reach of a specific entity.

Task attributes (set by coordinator)
-------------------------------------
  task.target_position   : Position   — for navigate_to_position
  task.target_entity_id  : int        — for navigate_to_entity

Task outcomes
-------------
  COMPLETE  : NavigateSkill reported SUCCEEDED (player arrived).
  FAILED    : NavigateSkill reported FAILED (target entity not found).
  STUCK     : NavigateSkill reported STUCK (pathfinder stalled).

Design
------
This agent is intentionally thin. All movement logic — MoveTo suppression,
stall detection, arrival detection, grace periods — lives in NavigateSkill.
The agent's job is:

  1. On activate(): read target from task, start NavigateSkill.
  2. Each tick(): tick the skill, return its actions.
  3. On skill terminal status: write a blackboard observation summarising
     the outcome.

The blackboard is used only for observations (no waypoint entries — the task
carries the target directly). The coordinator reads skill status via the task
evaluation cycle; no side-channel signals needed.

Self-model patches
------------------
NavigationAgent does not produce self-model patches.

Rules
-----
- No LLM calls. No RCON. No KnowledgeBase access.
- All state cleared on activate().
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from execution.agents.base import AgentProtocol
from execution.blackboard import EntryCategory, EntryScope
from execution.skills.navigate import NavigateSkill
from execution.skills.base import SkillStatus
from bridge import Action, StopMovement
from world import Position
from world.model.patch import SelfModelPatch

if TYPE_CHECKING:
    from execution.blackboard import Blackboard
    from planning.tasks.task import Task
    from world import KnowledgeBase, WorldQuery, WorldWriter

log = logging.getLogger(__name__)


class NavigationAgent(AgentProtocol):
    """
    Walks the player to a position or entity target.

    Wraps NavigateSkill; adds blackboard observations and translates skill
    status into signals the coordinator can act on via task evaluation.
    """

    AGENT_ID = "navigation"

    # Task type strings this agent handles.
    TASK_TYPES = frozenset({"navigate_to_position", "navigate_to_entity"})

    def __init__(self) -> None:
        self._skill = NavigateSkill()
        self._task_type: str = ""
        self._outcome_written: bool = False

    # ------------------------------------------------------------------
    # AgentProtocol
    # ------------------------------------------------------------------

    def activate(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> None:
        self._skill.reset()
        self._outcome_written = False
        self._task_type = getattr(task, "task_type", "navigate_to_position")

        target_pos: Optional[Position] = getattr(task, "target_position", None)
        target_entity_id: Optional[int] = getattr(task, "target_entity_id", None)

        if target_pos is not None:
            self._skill.start(target_position=target_pos)
        elif target_entity_id is not None:
            self._skill.start(target_entity_id=target_entity_id)
        else:
            log.error(
                "NavigationAgent: task %s has neither target_position nor "
                "target_entity_id — agent will idle",
                task.id[:8],
            )
            return

        pos = wq.player_position()
        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.TASK,
            owner_agent=self.AGENT_ID,
            created_at=task.created_at if wq is None else wq.tick,
            data={
                "type":          "navigation_started",
                "task_id":       task.id,
                "task_type":     self._task_type,
                "from":          {"x": pos.x, "y": pos.y},
                "target_pos":    {"x": target_pos.x, "y": target_pos.y}
                                 if target_pos else None,
                "target_entity": target_entity_id,
            },
        )
        log.debug(
            "NavigationAgent activated: task=%s type=%s target_pos=%s entity=%s",
            task.id[:8], self._task_type, target_pos, target_entity_id,
        )

    def tick(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
        kb: "KnowledgeBase",
    ) -> list[Action]:
        if self._skill.status() == SkillStatus.IDLE:
            return []

        if self._skill.status().is_terminal:
            return []

        actions = self._skill.tick(wq, ww, tick)

        # Write position observation every tick for coordinator/examination.
        pos = wq.player_position()
        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.TASK,
            owner_agent=self.AGENT_ID,
            created_at=tick,
            data={"type": "player_position", "position": {"x": pos.x, "y": pos.y}},
        )

        status = self._skill.status()
        if status.is_terminal and not self._outcome_written:
            self._outcome_written = True
            self._write_outcome(task, blackboard, tick, pos, status)

        return actions

    def observe(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> dict:
        pos = wq.player_position()
        obs = self._skill.observe()
        obs.update({
            "agent":           self.AGENT_ID,
            "task_id":         task.id[:8],
            "task_type":       self._task_type,
            "player_position": {"x": pos.x, "y": pos.y},
            "skill_status":    self._skill.status().name,
        })
        return obs

    def progress(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> float:
        status = self._skill.status()
        if status == SkillStatus.SUCCEEDED:
            return 1.0
        # NavigateSkill has no intermediate progress metric; report 0.5 once
        # a MoveTo has been issued (player is in motion toward target).
        if status == SkillStatus.RUNNING:
            return 0.5 if self._skill._last_issued_target is not None else 0.0
        return 0.0

    def teardown(self) -> list[Action]:
        """
        Halt any in-progress Lua movement when the task ends.

        NavigateSkill issues MoveTo actions that drive a persistent
        on_tick walker in the Lua mod. Without an explicit stop, the
        player continues walking to the previous target after the task
        resolves, which can interfere with subsequent goals.
        """
        self._skill.reset()
        return [StopMovement()]

    def pending_patches(self) -> list[SelfModelPatch]:
        return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_outcome(
        self,
        task: "Task",
        blackboard: "Blackboard",
        tick: int,
        pos: Position,
        status: SkillStatus,
    ) -> None:
        obs_type = {
            SkillStatus.SUCCEEDED: "navigation_succeeded",
            SkillStatus.STUCK:     "navigation_stuck",
            SkillStatus.FAILED:    "navigation_failed",
        }.get(status, "navigation_unknown")

        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.TASK,
            owner_agent=self.AGENT_ID,
            created_at=tick,
            data={
                "type":          obs_type,
                "task_id":       task.id,
                "position":      {"x": pos.x, "y": pos.y},
                "skill_observe": self._skill.observe(),
            },
        )
        if status == SkillStatus.STUCK:
            log.warning(
                "NavigationAgent: task %s STUCK at %s", task.id[:8], pos
            )
        else:
            log.debug(
                "NavigationAgent: task %s → %s at %s",
                task.id[:8], status.name, pos,
            )