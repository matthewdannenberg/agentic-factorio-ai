"""
execution/agents/crafting.py

CraftingAgent — queues hand-crafting jobs and confirms acceptance.

Task types handled
------------------
  craft_items   Hand-craft one or more items.

Task attributes (set by coordinator)
-------------------------------------
  task.targets : list[dict]
      Each entry: {"item": str, "recipe": str, "count": int}

Task outcomes
-------------
  COMPLETE  : CraftSkill SUCCEEDED — queue populated or items in inventory.
  STUCK     : CraftSkill STUCK — dispatch failed after _MAX_REISSUE attempts.

Design
------
This agent is a thin wrapper around CraftSkill. The confirmation logic
(crafting_queue_size > 0 or items in inventory) lives entirely in the skill.
The agent's job is:

  1. On activate(): read task.targets, build CraftTarget list, start skill.
  2. Each tick(): tick the skill, return its actions.
  3. On terminal status: write outcome observation to blackboard.

Self-model patches
------------------
CraftingAgent does not write factory graph patches.

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
from execution.skills.base import SkillStatus
from execution.skills.craft import CraftSkill, CraftTarget
from bridge import Action
from world.model.patch import SelfModelPatch

if TYPE_CHECKING:
    from execution.blackboard import Blackboard
    from planning.tasks.task import Task
    from world import KnowledgeBase, WorldQuery, WorldWriter

log = logging.getLogger(__name__)


class CraftingAgent(AgentProtocol):
    """
    Hand-crafting agent.

    Wraps CraftSkill; reads targets from task attributes and writes
    outcome observations to the blackboard.
    """

    AGENT_ID = "crafting"

    TASK_TYPES = frozenset({"craft_items"})

    def __init__(self) -> None:
        self._skill = CraftSkill()
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

        raw_targets: list[dict] = getattr(task, "targets", [])
        targets = [
            CraftTarget(
                item=str(t.get("item", "")),
                recipe=str(t.get("recipe", t.get("item", ""))),
                count=int(t.get("count", 0)),
            )
            for t in raw_targets
            if t.get("item") and int(t.get("count", 0)) > 0
        ]

        if targets:
            self._skill.start(targets=targets)
        else:
            log.error(
                "CraftingAgent: task %s has no valid targets in task.targets",
                task.id[:8],
            )

        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.TASK,
            owner_agent=self.AGENT_ID,
            created_at=task.created_at if wq is None else wq.tick,
            data={
                "type":    "crafting_started",
                "task_id": task.id,
                "targets": [{"item": t.item, "count": t.count} for t in targets],
            },
        )
        log.debug(
            "CraftingAgent activated: task=%s targets=%s",
            task.id[:8],
            ", ".join(f"{t.count}x {t.item}" for t in targets),
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

        status = self._skill.status()
        if status.is_terminal and not self._outcome_written:
            self._outcome_written = True
            obs_type = (
                "crafting_succeeded" if status == SkillStatus.SUCCEEDED
                else "crafting_stuck"
            )
            blackboard.write(
                category=EntryCategory.OBSERVATION,
                scope=EntryScope.TASK,
                owner_agent=self.AGENT_ID,
                created_at=tick,
                data={
                    "type":          obs_type,
                    "task_id":       task.id,
                    "skill_observe": self._skill.observe(),
                },
            )
            if status == SkillStatus.STUCK:
                log.warning(
                    "CraftingAgent: STUCK — task %s", task.id[:8]
                )

        return actions

    def observe(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> dict:
        obs = self._skill.observe()
        obs.update({
            "agent":        self.AGENT_ID,
            "task_id":      task.id[:8],
            "skill_status": self._skill.status().name,
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
        if status == SkillStatus.RUNNING:
            return 0.5 if self._skill._dispatched else 0.0
        return 0.0

    def pending_patches(self) -> list[SelfModelPatch]:
        return []