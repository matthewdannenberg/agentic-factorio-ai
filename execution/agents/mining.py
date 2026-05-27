"""
execution/agents/mining.py

MiningAgent — resource gathering and terrain clearing.

Task types handled
------------------
  gather_resource
      Mine N units of a named resource from a known patch.
      The coordinator navigates the player to the patch first
      (NavigationAgent), then activates this agent.

  clear_region
      Remove entities from a bounding box. Two modes:
        clear_natural — trees, rocks, cliffs, fish (name heuristics).
        clear_all     — every entity in the bbox.
      The coordinator navigates the player to the region first.

Task attributes (set by coordinator)
-------------------------------------
  gather_resource:
      task.resource_type   : str       — e.g. "iron-ore"
      task.target_position : Position  — tile to mine from
      task.count           : int       — units to collect (0 = unlimited)

  clear_region:
      task.bbox            : BoundingBox
      task.clear_mode      : str       — "clear_natural" | "clear_all"

Task outcomes
-------------
  COMPLETE  : MineSkill SUCCEEDED (count reached), or all clear targets gone.
  STUCK     : MineSkill STUCK (inventory stalled), or DestroySkill STUCK.

Internal loop for clear_region
-------------------------------
For each entity in target list:
  1. Check is_reachable(). If yes → start DestroySkill.
  2. If not reachable → start NavigateSkill toward entity position.
  3. When NavigateSkill SUCCEEDED → check reachable again.
  4. When DestroySkill SUCCEEDED → advance to next target.
  5. When DestroySkill STUCK → skip this target, log warning.

Self-model patches
------------------
MiningAgent does not write factory graph patches. Gathering and clearing
leave no persistent factory record beyond inventory changes.

Rules
-----
- No LLM calls. No RCON. No KnowledgeBase access.
- All state cleared on activate().
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

from execution.agents.base import AgentProtocol
from execution.blackboard import EntryCategory, EntryScope
from execution.predicates import is_reachable
from execution.skills.base import SkillStatus
from execution.skills.navigate import NavigateSkill
from execution.skills.mine import MineSkill
from execution.skills.destroy import DestroySkill
from bridge import Action, StopMining
from world import Position
from world.model.patch import SelfModelPatch

if TYPE_CHECKING:
    from execution.blackboard import Blackboard
    from planning.tasks.task import Task
    from world import KnowledgeBase, WorldQuery, WorldWriter, BoundingBox

log = logging.getLogger(__name__)


class _TaskKind(Enum):
    GATHER  = auto()
    CLEAR   = auto()
    UNKNOWN = auto()


@dataclass
class _ClearTarget:
    entity_id: int
    position: Position


class _ClearPhase(Enum):
    NAVIGATE = auto()   # moving toward target entity
    DESTROY  = auto()   # actively mining target entity


class MiningAgent(AgentProtocol):
    """
    Resource gathering and terrain clearing agent.

    Uses MineSkill for gathering, NavigateSkill + DestroySkill for clearing.
    """

    AGENT_ID = "mining"

    TASK_TYPES = frozenset({"gather_resource", "clear_region"})

    def __init__(self) -> None:
        self._task_kind: _TaskKind = _TaskKind.UNKNOWN
        self._pending_stop: bool = False

        # Skills
        self._mine_skill    = MineSkill()
        self._nav_skill     = NavigateSkill()
        self._destroy_skill = DestroySkill()

        # Gather state
        self._gather_outcome_written: bool = False

        # Clear state
        self._clear_targets: list[_ClearTarget] = []
        self._current_target: Optional[_ClearTarget] = None
        self._clear_phase: _ClearPhase = _ClearPhase.NAVIGATE
        self._targets_total: int = 0

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
        # Reset all skills and state.
        self._mine_skill.reset()
        self._nav_skill.reset()
        self._destroy_skill.reset()
        self._task_kind = _TaskKind.UNKNOWN
        self._gather_outcome_written = False
        self._clear_targets = []
        self._current_target = None
        self._clear_phase = _ClearPhase.NAVIGATE
        self._targets_total = 0
        # Halt any persistent Lua miner from a previous task.
        self._pending_stop = True

        task_type = getattr(task, "task_type", "")

        if task_type == "gather_resource":
            self._task_kind = _TaskKind.GATHER
            pos: Optional[Position] = getattr(task, "target_position", None)
            resource: str = getattr(task, "resource_type", "")
            count: int = getattr(task, "count", 0)
            if pos and resource:
                self._mine_skill.start(position=pos, resource=resource, count=count)
            else:
                log.error(
                    "MiningAgent: gather_resource task %s missing "
                    "target_position or resource_type",
                    task.id[:8],
                )

        elif task_type == "clear_region":
            self._task_kind = _TaskKind.CLEAR
            # Target list built on first tick (needs wq.state.entities scan).

        else:
            log.warning(
                "MiningAgent: unknown task_type %r on task %s",
                task_type, task.id[:8],
            )

        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.TASK,
            owner_agent=self.AGENT_ID,
            created_at=wq.tick,
            data={
                "type":      "mining_started",
                "task_id":   task.id,
                "task_type": task_type,
            },
        )
        log.debug(
            "MiningAgent activated: task=%s type=%s",
            task.id[:8], task_type,
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
        if self._pending_stop:
            self._pending_stop = False
            return [StopMining()]

        if self._task_kind == _TaskKind.GATHER:
            return self._tick_gather(task, blackboard, wq, ww, tick)
        elif self._task_kind == _TaskKind.CLEAR:
            return self._tick_clear(task, blackboard, wq, ww, tick)
        return []

    def observe(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> dict:
        obs: dict = {
            "agent":      self.AGENT_ID,
            "task_id":    task.id[:8],
            "task_kind":  self._task_kind.name,
        }
        if self._task_kind == _TaskKind.GATHER:
            obs.update(self._mine_skill.observe())
        elif self._task_kind == _TaskKind.CLEAR:
            obs.update({
                "clear_targets_remaining": len(self._clear_targets),
                "clear_current_entity":    (
                    self._current_target.entity_id
                    if self._current_target else None
                ),
                "clear_phase":             self._clear_phase.name,
            })
            obs.update(self._destroy_skill.observe())
        return obs

    def progress(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> float:
        if self._task_kind == _TaskKind.GATHER:
            status = self._mine_skill.status()
            if status == SkillStatus.SUCCEEDED:
                return 1.0
            if status == SkillStatus.RUNNING:
                return 0.5
            return 0.0
        if self._task_kind == _TaskKind.CLEAR:
            if self._targets_total == 0:
                return 0.0
            remaining = len(self._clear_targets) + (
                1 if self._current_target else 0
            )
            return min(1.0, 1.0 - remaining / self._targets_total)
        return 0.0

    def pending_patches(self) -> list[SelfModelPatch]:
        return []

    # ------------------------------------------------------------------
    # Gather
    # ------------------------------------------------------------------

    def _tick_gather(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        status = self._mine_skill.status()

        if status == SkillStatus.IDLE:
            # activate() couldn't start the skill (missing params).
            return []

        if status.is_terminal and not self._gather_outcome_written:
            self._gather_outcome_written = True
            obs_type = (
                "gather_succeeded" if status == SkillStatus.SUCCEEDED
                else "gather_stuck"
            )
            blackboard.write(
                category=EntryCategory.OBSERVATION,
                scope=EntryScope.TASK,
                owner_agent=self.AGENT_ID,
                created_at=tick,
                data={
                    "type":          obs_type,
                    "task_id":       task.id,
                    "skill_observe": self._mine_skill.observe(),
                },
            )
            if status == SkillStatus.STUCK:
                log.warning(
                    "MiningAgent: gather STUCK — task %s", task.id[:8]
                )

        if status.is_terminal:
            return []

        return self._mine_skill.tick(wq, ww, tick)

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def _tick_clear(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        # Build target list on first tick.
        if not self._clear_targets and self._current_target is None \
                and self._targets_total == 0:
            self._build_target_list(task, wq)
            if not self._clear_targets and self._current_target is None:
                log.info("MiningAgent: no targets in clear region — done")
                return []

        # Advance if current target was destroyed.
        if self._current_target is not None:
            if wq.entity_by_id(self._current_target.entity_id) is None:
                log.debug(
                    "MiningAgent: entity %d gone, advancing",
                    self._current_target.entity_id,
                )
                self._current_target = None
                self._nav_skill.reset()
                self._destroy_skill.reset()
                self._clear_phase = _ClearPhase.NAVIGATE

        # Pick next target.
        if self._current_target is None:
            if not self._clear_targets:
                return []
            self._current_target = self._clear_targets.pop(0)
            self._nav_skill.reset()
            self._destroy_skill.reset()
            self._clear_phase = _ClearPhase.NAVIGATE
            log.debug(
                "MiningAgent: targeting entity %d",
                self._current_target.entity_id,
            )

        # Route to the right phase.
        if self._clear_phase == _ClearPhase.NAVIGATE:
            return self._clear_navigate(task, blackboard, wq, ww, tick)
        else:
            return self._clear_destroy(task, blackboard, wq, ww, tick)

    def _clear_navigate(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        """Move toward current target until it's reachable, then switch to DESTROY."""
        target = self._current_target

        # Already reachable — skip navigation.
        if is_reachable(target.entity_id, wq):
            self._nav_skill.reset()
            self._clear_phase = _ClearPhase.DESTROY
            self._destroy_skill.start(entity_id=target.entity_id)
            return self._clear_destroy(task, blackboard, wq, ww, tick)

        nav_status = self._nav_skill.status()

        if nav_status == SkillStatus.IDLE:
            self._nav_skill.start(target_position=target.position)
        elif nav_status == SkillStatus.SUCCEEDED:
            # Arrived — switch to DESTROY regardless of is_reachable
            # (entity may be slightly off-centre from its tile position).
            self._nav_skill.reset()
            self._clear_phase = _ClearPhase.DESTROY
            self._destroy_skill.start(entity_id=target.entity_id)
            return self._clear_destroy(task, blackboard, wq, ww, tick)
        elif nav_status == SkillStatus.STUCK:
            log.warning(
                "MiningAgent: nav stuck reaching entity %d — skipping",
                target.entity_id,
            )
            self._current_target = None
            self._nav_skill.reset()
            self._clear_phase = _ClearPhase.NAVIGATE
            return []

        return self._nav_skill.tick(wq, ww, tick)

    def _clear_destroy(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        """Mine the current target entity."""
        d_status = self._destroy_skill.status()

        if d_status == SkillStatus.IDLE:
            # Shouldn't normally happen — start defensively.
            self._destroy_skill.start(entity_id=self._current_target.entity_id)

        if d_status == SkillStatus.SUCCEEDED:
            # Entity gone — advance handled at top of _tick_clear next tick.
            return []

        if d_status == SkillStatus.STUCK:
            log.warning(
                "MiningAgent: destroy STUCK on entity %d — skipping",
                self._current_target.entity_id,
            )
            self._current_target = None
            self._destroy_skill.reset()
            self._clear_phase = _ClearPhase.NAVIGATE
            return []

        return self._destroy_skill.tick(wq, ww, tick)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_target_list(self, task: "Task", wq: "WorldQuery") -> None:
        bbox = getattr(task, "bbox", None)
        if bbox is None:
            log.warning(
                "MiningAgent: clear_region task %s has no bbox", task.id[:8]
            )
            return

        clear_mode: str = getattr(task, "clear_mode", "clear_natural")
        natural_only = (clear_mode == "clear_natural")

        targets: list[_ClearTarget] = []
        for entity in wq.state.entities:
            pos = entity.position
            if not (bbox.x_min <= pos.x <= bbox.x_max
                    and bbox.y_min <= pos.y <= bbox.y_max):
                continue
            if natural_only and not _is_natural(entity):
                continue
            targets.append(_ClearTarget(entity_id=entity.entity_id, position=pos))

        player_pos = wq.player_position()
        targets.sort(
            key=lambda t: math.hypot(
                t.position.x - player_pos.x,
                t.position.y - player_pos.y,
            )
        )
        self._clear_targets = targets
        self._targets_total = len(targets)
        log.info(
            "MiningAgent: %d targets in clear region (mode=%s)",
            len(targets), clear_mode,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _is_natural(entity) -> bool:
    """
    True if the entity appears to be a natural world object.
    Uses name heuristics — KB-free. Replace with prototype_type check
    when EntityState gains that field (Phase 9).
    """
    name = entity.name.lower()
    return (
        "tree"    in name
        or "rock"    in name
        or "boulder" in name
        or "cliff"   in name
        or "fish"    in name
    )