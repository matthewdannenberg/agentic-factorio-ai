"""
execution/agents/mining.py

MiningAgent — resource gathering and terrain clearing.

Task types handled
------------------
  gather_resource
      Mine N units of a named resource from a known patch.
      The agent navigates to the patch internally (NAVIGATE phase) before
      mining (MINE phase), so no prior NavigationAgent task is required.

  clear_region
      Remove entities from a bounding box. Two modes:
        clear_natural — trees, rocks, cliffs, fish (name heuristics).
        clear_all     — every entity in the bbox.
      The agent navigates to each target internally.

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
from execution.predicates import is_reachable, can_destroy
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
    HARVEST = auto()   # mine natural objects for their drops (wood, fish, etc.)
    UNKNOWN = auto()


@dataclass
class _ClearTarget:
    entity_id: int
    position: Position


class _GatherPhase(Enum):
    NAVIGATE = auto()   # walking to the resource patch
    MINE     = auto()   # mining once in range


class _ClearPhase(Enum):
    NAVIGATE = auto()   # moving toward target entity
    DESTROY  = auto()   # actively mining target entity


class MiningAgent(AgentProtocol):
    """
    Resource gathering and terrain clearing agent.

    Uses MineSkill for gathering, NavigateSkill + DestroySkill for clearing.
    """

    AGENT_ID = "mining"

    TASK_TYPES = frozenset({"gather_resource", "clear_region", "harvest_natural"})

    def __init__(self) -> None:
        self._task_kind: _TaskKind = _TaskKind.UNKNOWN
        self._pending_stop: bool = False

        # Skills
        self._mine_skill    = MineSkill()
        self._nav_skill     = NavigateSkill()
        self._destroy_skill = DestroySkill()

        # Gather state
        self._gather_outcome_written: bool = False
        self._gather_phase: _GatherPhase = _GatherPhase.NAVIGATE

        # Clear state
        self._clear_targets: list[_ClearTarget] = []
        self._current_target: Optional[_ClearTarget] = None
        self._clear_phase: _ClearPhase = _ClearPhase.NAVIGATE
        self._targets_total: int = 0

        # Harvest state
        self._harvest_entity_types: list[str] = []

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
        self._gather_phase = _GatherPhase.NAVIGATE
        self._clear_targets = []
        self._current_target = None
        self._clear_phase = _ClearPhase.NAVIGATE
        self._targets_total = 0
        self._harvest_entity_types = []
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

        elif task_type == "harvest_natural":
            self._task_kind = _TaskKind.HARVEST
            self._harvest_entity_types = list(getattr(task, "entity_types", []))
            if not self._harvest_entity_types:
                log.error(
                    "MiningAgent: harvest_natural task %s missing entity_types",
                    task.id[:8],
                )

        else:
            log.warning(
                "MiningAgent: unknown task_type %r on task %s",
                task_type, task.id[:8],
            )

        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.TASK,
            owner_agent=self.AGENT_ID,
            created_at=task.created_at if wq is None else wq.tick,
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
            return self._tick_clear(task, blackboard, wq, ww, tick, kb)
        elif self._task_kind == _TaskKind.HARVEST:
            return self._tick_harvest(task, blackboard, wq, ww, tick, kb)
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
            obs["gather_phase"] = self._gather_phase.name
            if self._gather_phase == _GatherPhase.NAVIGATE:
                obs.update(self._nav_skill.observe())
            else:
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
        elif self._task_kind == _TaskKind.HARVEST:
            obs.update({
                "harvest_entity_types":    self._harvest_entity_types[:3],
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
            if self._gather_phase == _GatherPhase.NAVIGATE:
                nav_status = self._nav_skill.status()
                if nav_status == SkillStatus.RUNNING:
                    return 0.2
                return 0.0
            status = self._mine_skill.status()
            if status == SkillStatus.SUCCEEDED:
                return 1.0
            if status == SkillStatus.RUNNING:
                return 0.6
            return 0.3
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

    def teardown(self) -> list[Action]:
        """
        Called by the coordinator immediately after a task resolves.

        Issues StopMining whenever MineSkill is or was running — i.e. whenever
        the Lua persistent miner may be active. This covers two cases:

          1. _pending_stop is True: activate() set it to stop a *previous*
             task's miner before this one started (pre-existing behaviour).

          2. MineSkill is RUNNING or was recently running (task completed via
             the goal-level evaluator before the skill reached a terminal state).
             In this case _pending_stop is already False (cleared on first tick),
             but the Lua miner is still active and must be halted.

        Always safe to send StopMining redundantly — it is a no-op if nothing
        is mining.
        """
        self._pending_stop = False
        if self._task_kind in (_TaskKind.GATHER, _TaskKind.HARVEST):
            # Always issue StopMining for gather/harvest tasks — the Lua
            # persistent miner may still be active when the task resolves.
            log.debug("MiningAgent: teardown — issuing StopMining")
            return [StopMining()]
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
        """
        Two-phase gather loop: navigate to the patch, then mine.

        NAVIGATE phase: walk to target_position using NavigateSkill.
          - If already in range (MineSkill would succeed immediately),
            skip straight to MINE.
          - On nav SUCCEEDED or STUCK: transition to MINE regardless
            (STUCK means we got as close as pathfinding allows; MineSkill
            will determine whether we're actually in range).

        MINE phase: run MineSkill until SUCCEEDED or STUCK.
        """
        if self._gather_phase == _GatherPhase.NAVIGATE:
            return self._gather_navigate(task, blackboard, wq, ww, tick)
        return self._gather_mine(task, blackboard, wq, ww, tick)

    def _gather_navigate(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        """Walk to the resource patch, then hand off to _gather_mine."""
        mine_status = self._mine_skill.status()
        if mine_status == SkillStatus.IDLE:
            # activate() couldn't start MineSkill (missing params).
            return []

        # Check if MineSkill can start immediately (player already in range).
        # We do this by attempting a tick — if it issues actions the player
        # is close enough; if it returns STUCK straight away we need to
        # navigate first. Instead, use the nav approach unconditionally for
        # simplicity: always navigate to target_position first, then mine.
        # NavigateSkill will return SUCCEEDED immediately if already nearby.
        nav_status = self._nav_skill.status()

        if nav_status == SkillStatus.IDLE:
            target_pos = getattr(task, "target_position", None)
            if target_pos is not None:
                self._nav_skill.start(target_position=target_pos)
            else:
                # No position to navigate to — go straight to mining.
                log.debug(
                    "MiningAgent: no target_position on task %s, "
                    "skipping navigation",
                    task.id[:8],
                )
                self._gather_phase = _GatherPhase.MINE
                return self._gather_mine(task, blackboard, wq, ww, tick)

        nav_status = self._nav_skill.status()

        if nav_status in (SkillStatus.SUCCEEDED, SkillStatus.STUCK,
                          SkillStatus.FAILED):
            if nav_status == SkillStatus.STUCK:
                log.debug(
                    "MiningAgent: nav STUCK reaching patch — "
                    "attempting mine from current position",
                )
            self._nav_skill.reset()
            self._gather_phase = _GatherPhase.MINE
            return self._gather_mine(task, blackboard, wq, ww, tick)

        return self._nav_skill.tick(wq, ww, tick)

    def _gather_mine(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        """Run MineSkill to completion."""
        status = self._mine_skill.status()

        if status == SkillStatus.IDLE:
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
        kb: "KnowledgeBase",
    ) -> list[Action]:
        # Build target list on first tick.
        if not self._clear_targets and self._current_target is None \
                and self._targets_total == 0:
            self._build_target_list(task, wq, kb, blackboard, tick)
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
    # Harvest (harvest_natural task)
    # ------------------------------------------------------------------

    def _tick_harvest(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
        kb: "KnowledgeBase",
    ) -> list[Action]:
        """
        Mine natural objects of specific entity types to collect item drops.

        Unlike clear_region (bounded by a bbox, completes when region empty),
        harvest_natural loops continuously — when the target queue empties it
        rebuilds from the next scan, and keeps harvesting until the task's
        success_condition fires externally (evaluated by the coordinator).
        """
        # Advance if current target was destroyed or gone from scan.
        if self._current_target is not None:
            if wq.entity_by_id(self._current_target.entity_id) is None:
                log.debug(
                    "MiningAgent: harvest target %d gone — advancing",
                    self._current_target.entity_id,
                )
                self._current_target = None
                self._nav_skill.reset()
                self._destroy_skill.reset()
                self._clear_phase = _ClearPhase.NAVIGATE

        # Pick next target from queue, or rebuild from current scan.
        if self._current_target is None:
            if not self._clear_targets:
                self._build_harvest_targets(wq, kb)
            if not self._clear_targets:
                log.debug(
                    "MiningAgent: no harvestable %s in scan radius — waiting",
                    self._harvest_entity_types[:3],
                )
                return []
            self._current_target = self._clear_targets.pop(0)
            self._nav_skill.reset()
            self._destroy_skill.reset()
            self._clear_phase = _ClearPhase.NAVIGATE
            log.debug(
                "MiningAgent: harvesting entity %d",
                self._current_target.entity_id,
            )

        # Route to nav or destroy phase (reuse clear phase machinery).
        if self._clear_phase == _ClearPhase.NAVIGATE:
            return self._clear_navigate(task, blackboard, wq, ww, tick)
        return self._clear_destroy(task, blackboard, wq, ww, tick)

    def _build_harvest_targets(
        self,
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> None:
        """
        Scan wq.natural_objects for entities in _harvest_entity_types and
        queue them as _ClearTarget entries, nearest-first.
        Called each time the queue empties during a harvest_natural task.
        """
        type_set = set(self._harvest_entity_types)
        targets: list[_ClearTarget] = []
        for obj in wq.natural_objects:
            if obj.name not in type_set:
                continue
            if not obj.is_minable or not can_destroy(obj, kb):
                continue
            targets.append(_ClearTarget(entity_id=obj.entity_id, position=obj.position))
        player_pos = wq.player_position()
        targets.sort(
            key=lambda t: math.hypot(
                t.position.x - player_pos.x,
                t.position.y - player_pos.y,
            )
        )
        self._clear_targets = targets
        log.debug(
            "MiningAgent: harvest scan: %d %s targets",
            len(targets), self._harvest_entity_types[:3],
        )


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_target_list(
        self,
        task: "Task",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
        blackboard: "Blackboard",
        tick: int,
    ) -> None:
        bbox = getattr(task, "bbox", None)
        if bbox is None:
            log.warning(
                "MiningAgent: clear_region task %s has no bbox", task.id[:8]
            )
            return

        clear_mode: str = getattr(task, "clear_mode", "clear_natural")

        targets: list[_ClearTarget] = []
        blocked: list = []   # objects in bbox that can't be destroyed yet
        if clear_mode == "clear_natural":
            # Use wq.natural_objects — the dedicated natural-object scan that
            # returns trees, rocks, and cliffs. These are excluded from
            # wq.state.entities because they have no unit_number in Factorio.
            for obj in wq.natural_objects:
                pos = obj.position
                if not (bbox.x_min <= pos.x <= bbox.x_max
                        and bbox.y_min <= pos.y <= bbox.y_max):
                    continue
                if not obj.is_minable:
                    log.debug(
                        "MiningAgent: skipping non-minable %s at %s",
                        obj.name, pos,
                    )
                    blocked.append(obj.name)
                    continue
                if not can_destroy(obj, kb):
                    log.debug(
                        "MiningAgent: %s at %s requires trigger/special "
                        "action — not yet supported, skipping",
                        obj.name, pos,
                    )
                    blocked.append(obj.name)
                    continue
                targets.append(_ClearTarget(entity_id=obj.entity_id, position=pos))
        else:
            # clear_all — use both natural objects and player-built entities
            for obj in wq.natural_objects:
                pos = obj.position
                if not (bbox.x_min <= pos.x <= bbox.x_max
                        and bbox.y_min <= pos.y <= bbox.y_max):
                    continue
                if not obj.is_minable or not can_destroy(obj, kb):
                    blocked.append(obj.name)
                    continue
                targets.append(_ClearTarget(entity_id=obj.entity_id, position=pos))
            for entity in wq.state.entities:
                pos = entity.position
                if bbox.x_min <= pos.x <= bbox.x_max and bbox.y_min <= pos.y <= bbox.y_max:
                    targets.append(_ClearTarget(
                        entity_id=entity.entity_id, position=pos
                    ))

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
            "MiningAgent: %d targets in clear region (mode=%s), %d blocked",
            len(targets), clear_mode, len(blocked),
        )

        if blocked:
            # Some objects in the bbox cannot be destroyed with current
            # capabilities (trigger-mining not yet supported, or unknown
            # prototype). Notify the coordinator so it can decide whether
            # to treat the clear as partial success or failure.
            blackboard.write(
                category=EntryCategory.OBSERVATION,
                scope=EntryScope.TASK,
                owner_agent=self.AGENT_ID,
                created_at=tick,
                data={
                    "type":          "clear_partially_blocked",
                    "task_id":       task.id,
                    "blocked_names": list(set(blocked)),
                    "blocked_count": len(blocked),
                },
            )
            log.warning(
                "MiningAgent: %d object(s) in clear region cannot be "
                "destroyed — %s",
                len(blocked), list(set(blocked)),
            )