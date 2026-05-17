"""
agent/network/agents/mining.py

MiningAgent — handles resource gathering and obstacle clearing.

Satisfies AgentProtocol. Owns two subtask types:

  gather_resource
    Mine N of a named resource from a known patch. The approach walk has
    already been completed by the NavigationAgent before this agent is
    activated. The mining agent starts mining immediately, monitors inventory,
    and re-issues the mining command if it stalls (player drifted out of
    reach, resource tile exhausted and must move to adjacent tile).

  clear_region
    Given a bounding box, remove all obstacles from it. Two modes:

      clear_all     — mine every entity in the region regardless of faction.
      clear_natural — mine only entities whose force is "neutral" and whose
                      type is not "resource" (trees, rocks, cliffs, etc.).
                      Uses entity name heuristics at runtime — no KB dependency.

    For each target entity: if already within reach, mine it. If not, move
    to it first (the agent handles its own local positioning within the box),
    then mine it. Large-distance transit to the region is handled by the
    NavigationAgent before this agent is activated.

Mining model
------------
fa.mine_resource() and fa.mine_entity() set player.mining_state, which is
continuous: once set, the player mines until the resource is exhausted or the
state is cleared. The agent issues the command once and then does nothing until
a stall is detected (inventory unchanged after a grace period).

Rules
-----
- No LLM calls. No RCON. Satisfies AgentProtocol.
- No KB dependency. Natural-object detection uses entity name heuristics.
- Agents receive Subtasks, not Goals. No Goal is stored or inspected.
- All state between ticks stored on the instance. Cleared on activate().
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

from agent.blackboard import EntryCategory, EntryScope
from agent.network.agent_protocol import AgentProtocol
from agent.preconditions import is_reachable
from bridge.actions import Action, MineEntity, MineResource, MoveTo, StopMovement
from world.state import Position

if TYPE_CHECKING:
    from agent.blackboard import Blackboard
    from agent.subtask import Subtask
    from world.query import WorldQuery
    from world.writer import WorldWriter

log = logging.getLogger(__name__)

AGENT_ID = "mining"

# Default player reach radius in tiles when not available from WorldQuery.
_DEFAULT_REACH = 6.0

# Ticks to wait after issuing a mining command before checking for stall.
_MINING_GRACE_TICKS = 30

# Ticks to wait after issuing a move command before checking for stall.
_MOVE_GRACE_TICKS = 10

# Position change below this threshold (tiles/tick) → considered stalled.
_STOPPED_THRESHOLD = 0.05


class _SubtaskKind(Enum):
    GATHER  = auto()
    CLEAR   = auto()
    UNKNOWN = auto()


@dataclass
class _ClearTarget:
    entity_id: int
    position: Position


class MiningAgent(AgentProtocol):
    """
    Mining and destruction agent.

    Reads mining_task INTENTION entries from the blackboard (written by the
    coordinator) and executes them. Handles its own local positioning within
    a target region for clear_region tasks; long-distance transit is the
    NavigationAgent's job.
    """

    def __init__(self) -> None:
        self._current_subtask: Optional["Subtask"] = None
        self._subtask_kind: _SubtaskKind = _SubtaskKind.UNKNOWN

        # Gather state
        self._gather_resource_type: str = ""
        self._gather_target_pos: Optional[Position] = None
        self._gather_issued_at: int = 0
        self._last_inventory: dict = {}

        # Clear state
        self._clear_targets: list[_ClearTarget] = []
        self._clear_natural_only: bool = False
        self._current_target: Optional[_ClearTarget] = None
        self._mine_issued_at: int = 0
        self._move_issued_at: int = 0
        self._last_position: Optional[Position] = None

    # ------------------------------------------------------------------
    # AgentProtocol
    # ------------------------------------------------------------------

    def activate(
        self,
        subtask: "Subtask",
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> None:
        """
        Called when a new subtask is assigned to this agent.
        Resets all internal state for the new subtask.
        """
        self._current_subtask = subtask
        self._subtask_kind = _SubtaskKind.UNKNOWN
        self._gather_resource_type = ""
        self._gather_target_pos = None
        self._gather_issued_at = 0
        self._last_inventory = {}
        self._clear_targets = []
        self._clear_natural_only = False
        self._current_target = None
        self._mine_issued_at = 0
        self._move_issued_at = 0
        self._last_position = None
        log.debug(
            "MiningAgent activated for subtask %s: %s",
            subtask.id[:8], subtask.description,
        )

    def tick(
        self,
        subtask: "Subtask",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        """
        One tick. Reads the active mining_task entry and executes it.
        Issues commands only when necessary (new task or stall detected).
        """
        task = self._find_active_task(blackboard, tick)
        if task is None:
            return []

        task_type = task.data.get("task_type", "")

        if task_type == "gather_resource":
            return self._tick_gather(task.data, wq, tick)
        elif task_type in ("clear_all", "clear_natural"):
            return self._tick_clear(task.data, wq, tick)
        else:
            log.warning("MiningAgent: unknown task_type %r", task_type)
            return []

    def observe(
        self,
        subtask: "Subtask",
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> dict:
        return {
            "agent": AGENT_ID,
            "subtask_id": subtask.id[:8],
            "subtask_kind": self._subtask_kind.name,
            "clear_targets_remaining": len(self._clear_targets),
            "current_target": (
                self._current_target.entity_id if self._current_target else None
            ),
        }

    def progress(
        self,
        subtask: "Subtask",
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> float:
        if self._subtask_kind == _SubtaskKind.CLEAR:
            total = len(self._clear_targets) + (1 if self._current_target else 0)
            if total == 0:
                return 0.0
            done = total - len(self._clear_targets) - (1 if self._current_target else 0)
            return min(1.0, done / total)
        return 0.0

    # ------------------------------------------------------------------
    # Gather
    # ------------------------------------------------------------------

    def _tick_gather(
        self,
        task_data: dict,
        wq: "WorldQuery",
        tick: int,
    ) -> list[Action]:
        self._subtask_kind = _SubtaskKind.GATHER

        resource_type = task_data.get("resource_type", "")
        target_pos_dict = task_data.get("target_position")
        if not resource_type or not target_pos_dict:
            log.warning("MiningAgent: gather_resource task missing fields")
            return []

        target_pos = Position(x=target_pos_dict["x"], y=target_pos_dict["y"])

        is_new = (
            self._gather_resource_type != resource_type
            or self._gather_target_pos != target_pos
        )
        if is_new:
            self._gather_resource_type = resource_type
            self._gather_target_pos = target_pos
            self._gather_issued_at = 0
            self._last_inventory = {}

        grace_elapsed = (tick - self._gather_issued_at) >= _MINING_GRACE_TICKS
        is_stalled = grace_elapsed and self._is_inventory_stalled(wq)

        if self._gather_issued_at == 0 or is_stalled:
            if is_stalled:
                log.debug("MiningAgent: gather stall detected, re-issuing")
            self._gather_issued_at = tick
            self._last_inventory = self._inventory_snapshot(wq)
            return [MineResource(
                position=target_pos,
                resource=resource_type,
                count=0,  # 0 = mine until full / exhausted
            )]

        return []

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def _tick_clear(
        self,
        task_data: dict,
        wq: "WorldQuery",
        tick: int,
    ) -> list[Action]:
        self._subtask_kind = _SubtaskKind.CLEAR
        task_type = task_data.get("task_type", "clear_all")
        self._clear_natural_only = (task_type == "clear_natural")

        # Build target list on first tick.
        if not self._clear_targets and self._current_target is None:
            self._build_target_list(task_data, wq)
            if not self._clear_targets:
                log.info("MiningAgent: no targets found in region")
                return []

        # Advance if current target was destroyed.
        if self._current_target is not None:
            if wq.entity_by_id(self._current_target.entity_id) is None:
                log.debug(
                    "MiningAgent: target %d gone, advancing",
                    self._current_target.entity_id,
                )
                self._current_target = None
                self._mine_issued_at = 0
                self._move_issued_at = 0

        # Pick next target.
        if self._current_target is None:
            if not self._clear_targets:
                return []
            self._current_target = self._clear_targets.pop(0)
            self._mine_issued_at = 0
            self._move_issued_at = 0
            log.debug("MiningAgent: advancing to entity %d", self._current_target.entity_id)

        current_pos = wq.player_position()

        if is_reachable(self._current_target.entity_id, wq):
            return self._issue_mine(self._current_target, wq, tick)

        return self._issue_move(self._current_target.position, current_pos, tick)

    def _build_target_list(self, task_data: dict, wq: "WorldQuery") -> None:
        bbox = task_data.get("bounding_box")
        if not bbox:
            log.warning("MiningAgent: clear task has no bounding_box")
            return

        x_min = bbox.get("x_min", -1e9)
        y_min = bbox.get("y_min", -1e9)
        x_max = bbox.get("x_max",  1e9)
        y_max = bbox.get("y_max",  1e9)

        targets = []
        for entity in wq.state.entities:
            pos = entity.position
            if not (x_min <= pos.x <= x_max and y_min <= pos.y <= y_max):
                continue
            if self._clear_natural_only and not self._is_natural(entity):
                continue
            targets.append(_ClearTarget(entity_id=entity.entity_id, position=pos))

        player_pos = wq.player_position()
        targets.sort(key=lambda t: t.position.distance_to(player_pos))
        self._clear_targets = targets
        log.info("MiningAgent: %d targets in clear region", len(targets))

    def _is_natural(self, entity) -> bool:
        """
        True if the entity is a natural world object (not player-built).

        Uses entity name heuristics — KB-free. When EntityState gains a
        prototype_type field (Phase 9), replace with a type check against
        _NATURAL_TYPES = {"tree", "simple-entity", "cliff", "fish"}.
        """
        name = entity.name.lower()
        return (
            "tree" in name
            or "rock" in name
            or "boulder" in name
            or "cliff" in name
            or "fish" in name
        )

    def _issue_mine(
        self,
        target: _ClearTarget,
        wq: "WorldQuery",
        tick: int,
    ) -> list[Action]:
        grace_elapsed = (tick - self._mine_issued_at) >= _MINING_GRACE_TICKS
        entity_still_present = wq.entity_by_id(target.entity_id) is not None

        if self._mine_issued_at == 0 or (grace_elapsed and entity_still_present):
            if self._mine_issued_at > 0:
                log.debug("MiningAgent: mine stall on %d, re-issuing", target.entity_id)
            self._mine_issued_at = tick
            return [MineEntity(entity_id=target.entity_id)]

        return []

    def _issue_move(
        self,
        target_pos: Position,
        current_pos: Position,
        tick: int,
    ) -> list[Action]:
        is_new = self._move_issued_at == 0
        grace_elapsed = (tick - self._move_issued_at) >= _MOVE_GRACE_TICKS
        is_stalled = grace_elapsed and self._is_movement_stalled(current_pos)

        if is_new or is_stalled:
            if is_stalled:
                log.debug("MiningAgent: move stall, re-issuing")
            self._move_issued_at = tick
            self._last_position = current_pos
            return [MoveTo(position=target_pos, pathfind=True)]

        self._last_position = current_pos
        return []

    def _is_movement_stalled(self, current_pos: Position) -> bool:
        if self._last_position is None:
            return False
        dx = current_pos.x - self._last_position.x
        dy = current_pos.y - self._last_position.y
        return math.sqrt(dx * dx + dy * dy) < _STOPPED_THRESHOLD

    # ------------------------------------------------------------------
    # Inventory stall detection
    # ------------------------------------------------------------------

    def _is_inventory_stalled(self, wq: "WorldQuery") -> bool:
        return self._inventory_snapshot(wq) == self._last_inventory

    def _inventory_snapshot(self, wq: "WorldQuery") -> dict:
        return {
            slot.item: slot.count
            for slot in wq.state.player.inventory.slots
        }

    # ------------------------------------------------------------------
    # Blackboard reading
    # ------------------------------------------------------------------

    def _find_active_task(
        self,
        blackboard: "Blackboard",
        tick: int,
    ) -> Optional[object]:
        intentions = blackboard.read(
            category=EntryCategory.INTENTION,
            current_tick=tick,
        )
        tasks = [e for e in intentions if e.data.get("type") == "mining_task"]
        return tasks[0] if tasks else None