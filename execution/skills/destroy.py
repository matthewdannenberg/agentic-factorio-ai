"""
execution/skills/destroy.py

DestroySkill — mine (deconstruct) a specific placed entity by entity id.

Handles issuing MineEntity, detecting completion when the entity disappears
from the scan, and stall re-issue when the entity persists past the grace
period.

This is a single-target skill. Iterating over a list of targets to clear a
region is the responsibility of the agent using this skill.

start() parameters
------------------
entity_id : Id of the entity to destroy.

Status transitions
------------------
IDLE      -> RUNNING  : start() called.
RUNNING   -> SUCCEEDED: entity_id no longer found in wq (entity destroyed).
RUNNING   -> SUCCEEDED: entity_id not in wq on the very first tick — it was
                        already gone before we started. Treat as success so
                        the agent's clear loop advances rather than stalling.
RUNNING   -> STUCK    : entity still present after _MAX_REISSUE re-issues
                        of MineEntity — something is preventing destruction
                        (entity has health, wrong tool, blocked).
Any       -> IDLE     : reset() called.

Mining model
------------
fa.mine_entity() sets a persistent mining_state in Lua that mines the
entity continuously until it is destroyed or stop_mining() is called.
The skill issues the command once and re-issues only on stall detection.

The skill does NOT issue StopMining on SUCCEEDED — the agent is responsible
for stopping the miner between targets, since starting the next DestroySkill
will issue a fresh MineEntity command anyway. Issuing StopMining between each
target would introduce an unnecessary extra RCON round-trip.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from execution.skills.base import SkillProtocol, SkillStatus
from bridge import Action, MineEntity

if TYPE_CHECKING:
    from world import WorldQuery, WorldWriter

log = logging.getLogger(__name__)

# Ticks after issuing MineEntity before checking for stall. Entities take
# multiple swings to mine; at 60 tps and TICK_INTERVAL=10, 300 ticks = 5 s.
_MINING_GRACE_TICKS = 300

# Maximum re-issue attempts before declaring STUCK.
_MAX_REISSUE = 3


class DestroySkill(SkillProtocol):
    """
    Deconstruct a specific placed entity.

    Usage
    -----
        skill = DestroySkill()
        skill.start(entity_id=42)

        while skill.status() == SkillStatus.RUNNING:
            actions = skill.tick(wq, ww, tick)
            dispatch(actions)

        if skill.status() == SkillStatus.SUCCEEDED:
            skill.reset()
            # continue to next target
        elif skill.status() == SkillStatus.STUCK:
            # entity resisting destruction — skip or escalate
            skill.reset()
    """

    def __init__(self) -> None:
        self._status: SkillStatus = SkillStatus.IDLE
        self._entity_id: Optional[int] = None
        self._issued_at: int = 0
        self._reissue_count: int = 0

    # ------------------------------------------------------------------
    # SkillProtocol
    # ------------------------------------------------------------------

    def start(self, entity_id: int) -> None:
        """
        Initialise for a destruction job.

        Parameters
        ----------
        entity_id : Id of the entity to destroy. Must be present in wq on
                    the first tick, or the skill immediately reports SUCCEEDED
                    (already destroyed — treat as done, advance the loop).
        """
        self._entity_id     = entity_id
        self._issued_at     = 0
        self._reissue_count = 0
        self._status        = SkillStatus.RUNNING
        log.debug("DestroySkill started: entity_id=%d", entity_id)

    def tick(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        if self._status != SkillStatus.RUNNING:
            return []

        entity_present = wq.entity_by_id(self._entity_id) is not None

        # Entity already gone — succeeded (either we destroyed it or it was
        # already absent when we started).
        if not entity_present:
            log.debug(
                "DestroySkill: entity %d gone — SUCCEEDED", self._entity_id
            )
            self._status = SkillStatus.SUCCEEDED
            return []

        # First issue.
        if self._issued_at == 0:
            self._issued_at = tick
            log.debug("DestroySkill: issuing MineEntity %d", self._entity_id)
            return [MineEntity(entity_id=self._entity_id)]

        # Stall detection — entity still present after grace period.
        grace_elapsed = (tick - self._issued_at) >= _MINING_GRACE_TICKS
        if grace_elapsed:
            if self._reissue_count >= _MAX_REISSUE:
                log.warning(
                    "DestroySkill: %d re-issues on entity %d with no result — STUCK",
                    _MAX_REISSUE, self._entity_id,
                )
                self._status = SkillStatus.STUCK
                return []

            self._reissue_count += 1
            self._issued_at = tick
            log.debug(
                "DestroySkill: mine stall on entity %d, re-issuing (%d/%d)",
                self._entity_id, self._reissue_count, _MAX_REISSUE,
            )
            return [MineEntity(entity_id=self._entity_id)]

        return []

    def status(self) -> SkillStatus:
        return self._status

    def reset(self) -> None:
        self._status        = SkillStatus.IDLE
        self._entity_id     = None
        self._issued_at     = 0
        self._reissue_count = 0

    def observe(self) -> dict:
        return {
            "destroy_status":        self._status.name,
            "destroy_entity_id":     self._entity_id,
            "destroy_reissue_count": self._reissue_count,
        }