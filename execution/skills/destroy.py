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
from execution.predicates import is_reachable
from bridge import Action, MineEntity

if TYPE_CHECKING:
    from world import WorldQuery, WorldWriter
    from world import Position

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
        self._position: Optional["Position"] = None
        self._target_name: Optional[str] = None   # entity name for entity_id=0 targets
        self._issued_at: int = 0
        self._reissue_count: int = 0

    # ------------------------------------------------------------------
    # SkillProtocol
    # ------------------------------------------------------------------

    def start(
        self,
        entity_id: int,
        position: Optional["Position"] = None,
        target_name: Optional[str] = None,
    ) -> None:
        """
        Initialise for a destruction job.

        Parameters
        ----------
        entity_id   : Id of the entity to destroy. In Factorio 2.x, natural
                      objects like trees have no unit_number and use entity_id=0.
                      For entity_id=0, position must be supplied.
        position    : Tile position of the target. Required when entity_id=0.
        target_name : Entity prototype name (e.g. "tree-08"). Required when
                      entity_id=0 so that is_target_present() can distinguish
                      the original tree from a stump or other entity that may
                      appear at the same position after the tree is felled.
        """
        self._entity_id     = entity_id
        self._position      = position
        self._target_name   = target_name
        self._issued_at     = 0
        self._reissue_count = 0
        self._status        = SkillStatus.RUNNING
        log.debug("DestroySkill started: entity_id=%d name=%s", entity_id, target_name)

    def is_target_present(self, wq: "WorldQuery") -> bool:
        """
        True if the target entity is still present in the world scan.

        For entity_id=0 (natural objects without unit_number, e.g. trees in
        Factorio 2.x): checks wq.natural_objects by proximity to _position.
        For normal entities: checks wq.entity_by_id.

        Used by the agent's navigation phase to decide when to stop navigating
        and switch to destruction, and by the harvest loop to detect target
        removal between DestroySkill cycles.
        """
        if self._entity_id is None:
            return False
        if self._entity_id == 0 and self._position is not None:
            pos = self._position
            # Match by name AND position: a stump or different entity at the
            # same spot must not count as "still present". If target_name is
            # unknown, fall back to proximity-only (conservative).
            return any(
                abs(o.position.x - pos.x) < 2
                and abs(o.position.y - pos.y) < 2
                and (self._target_name is None or o.name == self._target_name)
                for o in wq.natural_objects
            )
        return wq.entity_by_id(self._entity_id) is not None

    def is_target_reachable(self, wq: "WorldQuery") -> bool:
        """
        True if the target is close enough to mine without further navigation.

        For entity_id=0 (trees): always returns False — there is no unit_number
        to look up in player.reachable, so the agent must navigate to the target
        position and rely on nav SUCCEEDED as the trigger to start destroying.

        For normal entities: delegates to is_reachable(), which checks the
        Factorio engine's actual reach list populated by the bridge.

        Used by _clear_navigate so it doesn't need to know about entity_id=0.
        """
        if self._entity_id is None:
            return False
        if self._entity_id == 0:
            return False   # no unit_number — navigate to position, not entity
        return is_reachable(self._entity_id, wq)

    def tick(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        if self._status != SkillStatus.RUNNING:
            return []

        entity_present = self.is_target_present(wq)

        # Entity already gone — succeeded.
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
        self._position      = None
        self._target_name   = None
        self._issued_at     = 0
        self._reissue_count = 0

    def observe(self) -> dict:
        return {
            "destroy_status":        self._status.name,
            "destroy_entity_id":     self._entity_id,
            "destroy_reissue_count": self._reissue_count,
        }