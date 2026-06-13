"""
execution/skills/destroy.py

DestroySkill — mine (deconstruct) a specific entity.

Handles issuing MineEntity, detecting completion, and stall re-issue.
Presence and reachability checks delegate entirely to predicates.py —
no game mechanics are reimplemented here.

start() parameters
------------------
sys_id : System-assigned entity ID (assigned by WorldWriter, not the raw
         Factorio unit_number). Works uniformly for both placed entities and
         natural objects — the old entity_id=0 special case no longer exists.

The MineEntity bridge action requires the raw Factorio unit_number, not the
sys_id. DestroySkill retrieves it via ww.factorio_id_for(sys_id) on each
tick where an action is issued. For natural objects this returns 0, which
the bridge handles as a position-based mine command.

Status transitions
------------------
IDLE      -> RUNNING  : start() called.
RUNNING   -> SUCCEEDED: is_present() returns False (entity gone).
RUNNING   -> SUCCEEDED: entity already gone on first tick.
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
    from world import WorldQuery, WorldWriter

log = logging.getLogger(__name__)

_MINING_GRACE_TICKS = 300   # 5 s at 60 tps / TICK_INTERVAL=10
_MAX_REISSUE        = 3


class DestroySkill(SkillProtocol):

    def __init__(self) -> None:
        self._status:        SkillStatus  = SkillStatus.IDLE
        self._sys_id:        Optional[int] = None
        self._issued_at:     int           = 0
        self._reissue_count: int           = 0

    def start(self, sys_id: int) -> None:
        """
        Initialise for a new destruction job.

        Parameters
        ----------
        sys_id : System entity ID as assigned by WorldWriter. Works for both
                 placed entities and natural objects.
        """
        self._sys_id        = sys_id
        self._issued_at     = 0
        self._reissue_count = 0
        self._status        = SkillStatus.RUNNING
        log.debug("DestroySkill started: sys_id=%d", sys_id)

    def tick(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        if self._status != SkillStatus.RUNNING:
            return []

        if not self.is_target_present(wq):
            log.debug("DestroySkill: entity sys_id=%d gone — SUCCEEDED", self._sys_id)
            self._status = SkillStatus.SUCCEEDED
            return []

        factorio_id = ww.factorio_id_for(self._sys_id)

        if self._issued_at == 0:
            self._issued_at = tick
            log.debug(
                "DestroySkill: issuing MineEntity sys_id=%d factorio_id=%d",
                self._sys_id, factorio_id,
            )
            return [MineEntity(entity_id=factorio_id)]

        grace_elapsed = (tick - self._issued_at) >= _MINING_GRACE_TICKS
        if grace_elapsed:
            if self._reissue_count >= _MAX_REISSUE:
                log.warning(
                    "DestroySkill: %d re-issues on sys_id=%d — STUCK",
                    _MAX_REISSUE, self._sys_id,
                )
                self._status = SkillStatus.STUCK
                return []
            self._reissue_count += 1
            self._issued_at = tick
            log.debug(
                "DestroySkill: stall on sys_id=%d, re-issuing (%d/%d)",
                self._sys_id, self._reissue_count, _MAX_REISSUE,
            )
            return [MineEntity(entity_id=factorio_id)]

        return []

    def status(self) -> SkillStatus:
        return self._status

    def reset(self) -> None:
        self._status        = SkillStatus.IDLE
        self._sys_id        = None
        self._issued_at     = 0
        self._reissue_count = 0

    def observe(self) -> dict:
        return {
            "destroy_status":        self._status.name,
            "destroy_entity_id":     self._sys_id,
            "destroy_reissue_count": self._reissue_count,
        }

    def is_target_present(self, wq: "WorldQuery") -> bool:
        """True if the target still exists. Delegates to predicates.is_present()."""
        if self._sys_id is None:
            return False
        return is_present(self._sys_id, wq)

    def is_target_reachable(self, wq: "WorldQuery") -> bool:
        """
        True if the target entity is within game-engine reach.

        Delegates to predicates.is_reachable() which checks the sys_id-keyed
        reachable set populated by WorldWriter. Works uniformly for placed
        entities and natural objects — no special cases.
        """
        if self._sys_id is None:
            return False
        return is_reachable(self._sys_id, wq)