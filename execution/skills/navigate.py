"""
execution/skills/navigate.py

NavigateSkill — walk the player to a target position or entity.

Extracted from NavigationAgent. Contains all movement logic, stall detection,
redundant-command suppression, and arrival detection. NavigationAgent will
become a thin wrapper around this skill.

start() parameters
------------------
target_position  : Tile-space Position to walk to. Mutually exclusive with
                   target_entity_id, but one must be provided.
target_entity_id : System entity ID (sys_id assigned by WorldWriter) to walk
                   toward. Uses the entity's current position from wq on each
                   tick, so the skill tracks moving entities correctly.
                   Arrival is detected via is_reachable(sys_id, wq) — True
                   when the entity's sys_id is in the player's reachable set.

Status transitions
------------------
IDLE      → RUNNING  : start() called with a valid target.
RUNNING   → SUCCEEDED: arrival check passes (is_at or is_reachable).
RUNNING   → STUCK    : player has not moved for more than the grace period
                       since the last MoveTo was issued.
RUNNING   → FAILED   : target_entity_id given but entity not found in wq on
                       the first tick (target disappeared before we started).
Any       → IDLE     : reset() called.

Stall detection
---------------
Two grace periods exist for the two common stall cases:

_STALL_GRACE_TICKS (180 ticks ≈ 3 s)
    Standard grace. Gives the pathfinder time to compute and the character
    time to start moving. Used when the player has moved since activation.

_UNREACHABLE_GRACE_TICKS (30 ticks ≈ 0.5 s)
    Fast grace. When the player has not moved at all since start() was called,
    the pathfinder almost certainly returned "unreachable" almost immediately.
    Transition to STUCK quickly so the agent can try a different approach.

MoveTo suppression
------------------
A new MoveTo is issued only when:
  1. start() was just called (first tick).
  2. The target has moved by more than _REDUNDANT_THRESHOLD tiles since the
     last issued command (entity target changed position).
  3. A stall is detected — this is handled by transitioning to STUCK rather
     than re-issuing, which is a change from the old NavigationAgent behaviour.
     The agent/coordinator is responsible for re-starting if desired.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, TYPE_CHECKING

from execution.skills.base import SkillProtocol, SkillStatus
from execution.predicates import is_at, is_reachable
from bridge import Action, MoveTo, StopMovement
from world import Position

if TYPE_CHECKING:
    from world import WorldQuery, WorldWriter

log = logging.getLogger(__name__)

# Arrival tolerance (tiles) for position-only targets.
_POSITION_ARRIVAL_TOLERANCE = 1.5

# Suppress a new MoveTo if the target has moved less than this many tiles
# since the last issued command — avoids cancelling an in-flight path request.
_REDUNDANT_THRESHOLD = 0.5

# Standard grace: time for the pathfinder to respond and the player to begin.
_STALL_GRACE_TICKS = 180

# Fast grace: used when the player has not moved at all since start(). If the
# pathfinder returned "unreachable" the player will be stationary from tick 1.
_UNREACHABLE_GRACE_TICKS = 30

# Position change below this threshold between ticks → considered stationary.
_STOPPED_THRESHOLD = 0.05




class NavigateSkill(SkillProtocol):
    """
    Walk the player character to a target.

    Usage
    -----
        skill = NavigateSkill()

        # By position:
        skill.start(target_position=Position(x=100, y=50))

        # By entity:
        skill.start(target_entity_id=42)

        # Each tick:
        actions = skill.tick(wq, ww, tick)
        if skill.status() == SkillStatus.SUCCEEDED:
            ...
        elif skill.status() == SkillStatus.STUCK:
            # handle stall — retry or escalate
            skill.reset()
    """

    def __init__(self) -> None:
        self._status: SkillStatus = SkillStatus.IDLE
        self._target_position: Optional[Position] = None
        self._target_entity_id: Optional[int] = None
        self._last_issued_target: Optional[Position] = None
        self._last_move_tick: int = 0
        self._last_position: Optional[Position] = None
        self._start_position: Optional[Position] = None   # position at start()

    # ------------------------------------------------------------------
    # SkillProtocol
    # ------------------------------------------------------------------

    def start(
        self,
        target_position: Optional[Position] = None,
        target_entity_id: Optional[int] = None,
    ) -> None:
        """
        Initialise for a new navigation job.

        Exactly one of target_position or target_entity_id must be supplied.
        Calling start() while RUNNING resets and restarts with the new target.
        """
        if target_position is None and target_entity_id is None:
            raise ValueError(
                "NavigateSkill.start() requires target_position or target_entity_id"
            )
        if target_position is not None and target_entity_id is not None:
            raise ValueError(
                "NavigateSkill.start() takes target_position OR target_entity_id, not both"
            )
        self._target_position   = target_position
        self._target_entity_id  = target_entity_id
        self._last_issued_target = None
        self._last_move_tick     = 0
        self._last_position      = None
        self._start_position     = None   # set on first tick from wq
        self._status = SkillStatus.RUNNING
        log.debug(
            "NavigateSkill started: pos=%s entity_id=%s",
            target_position, target_entity_id,
        )

    def tick(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        if self._status != SkillStatus.RUNNING:
            return []

        pos = wq.player_position()

        # Capture start position on first tick so we can detect never-moved.
        if self._start_position is None:
            self._start_position = pos

        # --- Resolve concrete target position ---
        target_pos = self._resolve_target(wq)
        if target_pos is None:
            # Entity target requested but not found in wq — it's gone.
            log.warning(
                "NavigateSkill: entity %s not found in wq — FAILED",
                self._target_entity_id,
            )
            self._status = SkillStatus.FAILED
            self._last_position = pos
            return [StopMovement()]

        # --- Arrival ---
        if self._check_arrival(wq, target_pos):
            log.debug("NavigateSkill: arrived at target")
            self._status = SkillStatus.SUCCEEDED
            self._last_position = pos
            return [StopMovement()]

        # --- Stall detection ---
        never_moved = self._is_same_position(pos, self._start_position)
        grace = _UNREACHABLE_GRACE_TICKS if never_moved else _STALL_GRACE_TICKS
        grace_elapsed = (tick - self._last_move_tick) >= grace
        is_stalled = grace_elapsed and self._is_stalled(pos)

        if is_stalled:
            log.warning(
                "NavigateSkill: stall after %d ticks at %s — STUCK",
                tick - self._last_move_tick, pos,
            )
            self._status = SkillStatus.STUCK
            self._last_position = pos
            return [StopMovement()]

        # --- Issue MoveTo if needed ---
        if self._should_issue(target_pos):
            log.debug("NavigateSkill: issuing MoveTo %s", target_pos)
            self._last_issued_target = target_pos
            self._last_move_tick = tick
            self._last_position = pos
            return [MoveTo(position=target_pos, pathfind=True)]

        self._last_position = pos
        return []

    def status(self) -> SkillStatus:
        return self._status

    def reset(self) -> None:
        self._status             = SkillStatus.IDLE
        self._target_position    = None
        self._target_entity_id   = None
        self._last_issued_target = None
        self._last_move_tick     = 0
        self._last_position      = None
        self._start_position     = None

    def observe(self) -> dict:
        return {
            "navigate_status": self._status.name,
            "navigate_target_pos": (
                {"x": self._target_position.x, "y": self._target_position.y}
                if self._target_position else None
            ),
            "navigate_target_entity": self._target_entity_id,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_target(self, wq: "WorldQuery") -> Optional[Position]:
        """Resolve the current target to a tile-space Position, or None."""
        if self._target_entity_id is not None:
            entity = wq.entity_by_id(self._target_entity_id)
            if entity is None:
                return None
            return entity.position
        return self._target_position

    def _check_arrival(self, wq: "WorldQuery", target_pos: Position) -> bool:
        if self._target_entity_id is not None:
            return is_reachable(self._target_entity_id, wq)
        return is_at(target_pos, wq, tolerance=_POSITION_ARRIVAL_TOLERANCE)

    def _should_issue(self, target_pos: Position) -> bool:
        """True if a new MoveTo should be issued for this target."""
        if self._last_issued_target is None:
            return True
        dx = target_pos.x - self._last_issued_target.x
        dy = target_pos.y - self._last_issued_target.y
        return math.sqrt(dx * dx + dy * dy) >= _REDUNDANT_THRESHOLD

    def _is_stalled(self, current_pos: Position) -> bool:
        if self._last_position is None:
            return False
        dx = current_pos.x - self._last_position.x
        dy = current_pos.y - self._last_position.y
        return math.sqrt(dx * dx + dy * dy) < _STOPPED_THRESHOLD

    def _is_same_position(self, a: Position, b: Position) -> bool:
        dx = a.x - b.x
        dy = a.y - b.y
        return math.sqrt(dx * dx + dy * dy) < _STOPPED_THRESHOLD