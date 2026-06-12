"""
execution/skills/destroy.py

DestroySkill — mine (deconstruct) a specific entity.

Handles issuing MineEntity, detecting completion, and stall re-issue.
Presence and reachability checks delegate entirely to predicates.py —
no game mechanics are reimplemented here.

start() parameters
------------------
entity_id    : Id of the entity to destroy. 0 for natural objects (trees,
               rocks) which have no unit number in Factorio 2.x.
position     : World position of the target. Required when entity_id == 0
               for proximity-based presence detection via is_present().
target_name  : Prototype name. Used by is_present() to distinguish natural
               objects at similar positions.

Status transitions
------------------
IDLE      -> RUNNING  : start() called.
RUNNING   -> SUCCEEDED: is_present() returns False (entity gone).
RUNNING   -> SUCCEEDED: entity already gone on first tick — advance loop.
RUNNING   -> STUCK    : entity persists after _MAX_REISSUE re-issues.
Any       -> IDLE     : reset() called.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from execution.skills.base import SkillProtocol, SkillStatus
from execution.predicates import is_present, is_reachable
from bridge import Action, MineEntity

if TYPE_CHECKING:
    from world import WorldQuery, WorldWriter, Position

log = logging.getLogger(__name__)

_MINING_GRACE_TICKS = 300   # 5 s at 60 tps / TICK_INTERVAL=10
_MAX_REISSUE        = 3


class DestroySkill(SkillProtocol):

    def __init__(self) -> None:
        self._status:        SkillStatus          = SkillStatus.IDLE
        self._entity_id:     Optional[int]        = None
        self._position:      Optional["Position"] = None
        self._target_name:   str                  = ""
        self._issued_at:     int                  = 0
        self._reissue_count: int                  = 0

    def start(
        self,
        entity_id:   int,
        position:    "Optional[Position]" = None,
        target_name: str = "",
    ) -> None:
        self._entity_id     = entity_id
        self._position      = position
        self._target_name   = target_name
        self._issued_at     = 0
        self._reissue_count = 0
        self._status        = SkillStatus.RUNNING
        log.debug(
            "DestroySkill started: entity_id=%d name=%s",
            entity_id, target_name,
        )

    def tick(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        if self._status != SkillStatus.RUNNING:
            return []

        if not self.is_target_present(wq):
            log.debug(
                "DestroySkill: entity %d (%s) gone — SUCCEEDED",
                self._entity_id, self._target_name,
            )
            self._status = SkillStatus.SUCCEEDED
            return []

        if self._issued_at == 0:
            self._issued_at = tick
            log.debug("DestroySkill: issuing MineEntity %d", self._entity_id)
            return [MineEntity(entity_id=self._entity_id)]

        grace_elapsed = (tick - self._issued_at) >= _MINING_GRACE_TICKS
        if grace_elapsed:
            if self._reissue_count >= _MAX_REISSUE:
                log.warning(
                    "DestroySkill: %d re-issues on entity %d (%s) — STUCK",
                    _MAX_REISSUE, self._entity_id, self._target_name,
                )
                self._status = SkillStatus.STUCK
                return []
            self._reissue_count += 1
            self._issued_at = tick
            log.debug(
                "DestroySkill: stall on entity %d, re-issuing (%d/%d)",
                self._entity_id, self._reissue_count, _MAX_REISSUE,
            )
            return [MineEntity(entity_id=self._entity_id)]

        return []

    def status(self) -> SkillStatus:
        return self._status

    def reset(self) -> None:
        self._status        = SkillStatus.IDLE
        self._entity_id     = None
        self._position      = None
        self._target_name   = ""
        self._issued_at     = 0
        self._reissue_count = 0

    def observe(self) -> dict:
        return {
            "destroy_status":        self._status.name,
            "destroy_entity_id":     self._entity_id,
            "destroy_reissue_count": self._reissue_count,
        }

    def is_target_present(self, wq: "WorldQuery") -> bool:
        """True if the target still exists. Delegates to predicates.is_present()."""
        if self._entity_id is None:
            return False
        pos = self._position
        if pos is None and self._entity_id == 0:
            return True   # no position recorded — assume present, avoid spurious SUCCEEDED
        return is_present(self._entity_id, pos, wq, name=self._target_name)

    def is_target_reachable(self, wq: "WorldQuery") -> bool:
        """
        True if the target entity is within game-engine mining reach.

        For real entities: delegates to predicates.is_reachable() which uses
        the actual reachable set reported by the Lua mod — no hardcoded
        distance constants.

        For natural objects (entity_id == 0): always False — reach is
        determined by navigation arrival, since natural objects have no
        unit number to look up in the reachable set.
        """
        if self._entity_id is None or self._entity_id == 0:
            return False
        return is_reachable(self._entity_id, wq)