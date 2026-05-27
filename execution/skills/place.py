"""
execution/skills/place.py

PlaceSkill — place an entity from inventory at a tile position.

Phase 7 stub. Interface is complete; implementation deferred to Phase 7
when the construction agent is built. The skill transitions immediately to
FAILED so any agent using it before implementation is visible in tests.

start() parameters
------------------
item      : Inventory item name to place (e.g. "assembling-machine-1").
position  : Tile-space Position to place the entity.
direction : Direction enum value for the entity's facing.

Status transitions (stub)
--------------------------
IDLE    → RUNNING : start() called.
RUNNING → FAILED  : immediately on first tick (not yet implemented).

Implementation notes (Phase 7)
-------------------------------
- Issue PlaceEntity action.
- Confirm placement by polling entity scan for a new entity of the right
  type at (or near) the target position.
- SUCCEEDED once confirmed. FAILED if the position is blocked (another
  entity present, tile collision) after _MAX_ATTEMPTS retries.
- STUCK if entity not visible in scan radius after placement (player
  may need to move closer before confirmation is reliable).
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from execution.skills.base import SkillProtocol, SkillStatus
from bridge import Action
from world import Direction, Position

if TYPE_CHECKING:
    from world import WorldQuery, WorldWriter

log = logging.getLogger(__name__)


class PlaceSkill(SkillProtocol):
    """Place an entity at a position. Phase 7 stub."""

    def __init__(self) -> None:
        self._status: SkillStatus = SkillStatus.IDLE
        self._item: str = ""
        self._position: Optional[Position] = None
        self._direction: Optional[Direction] = None

    def start(
        self,
        item: str,
        position: Position,
        direction: Direction,
    ) -> None:
        """
        Initialise for a placement job.

        Parameters
        ----------
        item      : Inventory item to place.
        position  : Target tile position.
        direction : Entity facing direction.
        """
        self._item      = item
        self._position  = position
        self._direction = direction
        self._status    = SkillStatus.RUNNING
        log.debug("PlaceSkill started: item=%s pos=%s dir=%s", item, position, direction)

    def tick(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        if self._status != SkillStatus.RUNNING:
            return []
        log.warning("PlaceSkill: not yet implemented — FAILED")
        self._status = SkillStatus.FAILED
        return []

    def status(self) -> SkillStatus:
        return self._status

    def reset(self) -> None:
        self._status    = SkillStatus.IDLE
        self._item      = ""
        self._position  = None
        self._direction = None

    def observe(self) -> dict:
        return {
            "place_status":    self._status.name,
            "place_item":      self._item,
            "place_position":  (
                {"x": self._position.x, "y": self._position.y}
                if self._position else None
            ),
        }
