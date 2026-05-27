"""
execution/skills/clear.py

ClearSkill — remove all (or all natural) entities from a bounding box.

Extracted from MiningAgent._tick_clear(). Handles building the target list,
local positioning to reach each entity (using NavigateSkill internally),
issuing MineEntity, and advancing through the list as targets are destroyed.

start() parameters
------------------
bbox       : BoundingBox defining the region to clear.
clear_mode : "clear_natural" (default) — remove only natural objects
             (trees, rocks, cliffs, fish) identified by name heuristic.
             "clear_all" — remove every entity in the bbox regardless.

Status transitions
------------------
IDLE      → RUNNING  : start() called.
RUNNING   → SUCCEEDED: target list exhausted (all entities destroyed).
RUNNING   → STUCK    : current target entity stall — MineEntity re-issued
                       _MAX_REISSUE times with no entity disappearing.
Any       → IDLE     : reset() called.

Local positioning
-----------------
ClearSkill uses a NavigateSkill instance internally to move the player to
each unreachable target. This is local positioning only — moving a few tiles
within the clearing region. Long-distance transit to the region is handled
by the coordinator before the skill is started.

Natural object detection
------------------------
Uses entity name heuristics (KB-free). When EntityState gains a prototype_type
field, replace _is_natural() with a type check against the canonical set:
{"tree", "simple-entity", "cliff", "fish"}.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from execution.preconditions import is_reachable
from execution.skills.base import SkillProtocol, SkillStatus
from execution.skills.navigate import NavigateSkill
from bridge import Action, MineEntity, StopMining
from world import BoundingBox, Position

if TYPE_CHECKING:
    from world import WorldQuery, WorldWriter

log = logging.getLogger(__name__)

# Ticks after issuing MineEntity before checking for stall.
_MINING_GRACE_TICKS = 300

# Maximum re-issues per target before declaring STUCK.
_MAX_REISSUE = 3


@dataclass
class _ClearTarget:
    entity_id: int
    position: Position


class ClearSkill(SkillProtocol):
    """
    Remove entities from a bounding box.

    Usage
    -----
        skill = ClearSkill()
        skill.start(bbox=BoundingBox(...), clear_mode="clear_natural")

        while skill.status() == SkillStatus.RUNNING:
            actions = skill.tick(wq, ww, tick)
            dispatch(actions)
    """

    def __init__(self) -> None:
        self._status: SkillStatus = SkillStatus.IDLE
        self._bbox: Optional[BoundingBox] = None
        self._clear_natural_only: bool = True
        self._targets: list[_ClearTarget] = []
        self._current: Optional[_ClearTarget] = None
        self._mine_issued_at: int = 0
        self._reissue_count: int = 0
        self._list_built: bool = False
        self._navigate = NavigateSkill()

    # ------------------------------------------------------------------
    # SkillProtocol
    # ------------------------------------------------------------------

    def start(
        self,
        bbox: BoundingBox,
        clear_mode: str = "clear_natural",
    ) -> None:
        """
        Initialise for a new clearing job.

        Parameters
        ----------
        bbox       : Region to clear.
        clear_mode : "clear_natural" or "clear_all".
        """
        self._bbox               = bbox
        self._clear_natural_only = (clear_mode == "clear_natural")
        self._targets            = []
        self._current            = None
        self._mine_issued_at     = 0
        self._reissue_count      = 0
        self._list_built         = False
        self._navigate.reset()
        self._status = SkillStatus.RUNNING
        log.debug(
            "ClearSkill started: bbox=%s mode=%s", bbox, clear_mode
        )

    def tick(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        if self._status != SkillStatus.RUNNING:
            return []

        # Build target list on first tick (needs entity scan from wq).
        if not self._list_built:
            self._build_target_list(wq)
            self._list_built = True
            if not self._targets and self._current is None:
                log.info("ClearSkill: no targets in bbox — SUCCEEDED")
                self._status = SkillStatus.SUCCEEDED
                return [StopMining()]

        # Advance if the current target has been destroyed.
        if self._current is not None:
            if wq.entity_by_id(self._current.entity_id) is None:
                log.debug(
                    "ClearSkill: entity %d gone, advancing",
                    self._current.entity_id,
                )
                self._current = None
                self._mine_issued_at = 0
                self._reissue_count  = 0
                self._navigate.reset()

        # Pick next target when current slot is empty.
        if self._current is None:
            if not self._targets:
                log.info("ClearSkill: all targets cleared — SUCCEEDED")
                self._status = SkillStatus.SUCCEEDED
                return [StopMining()]
            self._current = self._targets.pop(0)
            self._mine_issued_at = 0
            self._reissue_count  = 0
            self._navigate.reset()
            log.debug("ClearSkill: targeting entity %d", self._current.entity_id)

        # Reach check — navigate to the target if not within range.
        if not is_reachable(self._current.entity_id, wq):
            return self._navigate_to(self._current.position, wq, ww, tick)

        # Within reach — mine it.
        return self._issue_mine(tick, wq)

    def status(self) -> SkillStatus:
        return self._status

    def reset(self) -> None:
        self._status             = SkillStatus.IDLE
        self._bbox               = None
        self._clear_natural_only = True
        self._targets            = []
        self._current            = None
        self._mine_issued_at     = 0
        self._reissue_count      = 0
        self._list_built         = False
        self._navigate.reset()

    def observe(self) -> dict:
        return {
            "clear_status":           self._status.name,
            "clear_targets_remaining": len(self._targets),
            "clear_current_entity":   (
                self._current.entity_id if self._current else None
            ),
            "clear_reissue_count":    self._reissue_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _navigate_to(
        self,
        target_pos: Position,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        """Use the embedded NavigateSkill for local positioning."""
        nav = self._navigate
        if nav.status() == SkillStatus.IDLE:
            nav.start(target_position=target_pos)
        elif nav.status() == SkillStatus.STUCK:
            # Can't reach this target; skip it.
            log.warning(
                "ClearSkill: can't reach entity %d — skipping",
                self._current.entity_id if self._current else -1,
            )
            self._current = None
            nav.reset()
            return []
        elif nav.status() == SkillStatus.SUCCEEDED:
            nav.reset()
            return []
        return nav.tick(wq, ww, tick)

    def _issue_mine(self, tick: int, wq: "WorldQuery") -> list[Action]:
        if self._mine_issued_at == 0:
            self._mine_issued_at = tick
            return [MineEntity(entity_id=self._current.entity_id)]

        grace_elapsed = (tick - self._mine_issued_at) >= _MINING_GRACE_TICKS
        entity_present = wq.entity_by_id(self._current.entity_id) is not None

        if grace_elapsed and entity_present:
            if self._reissue_count >= _MAX_REISSUE:
                log.warning(
                    "ClearSkill: %d re-issues on entity %d with no result — STUCK",
                    _MAX_REISSUE, self._current.entity_id,
                )
                self._status = SkillStatus.STUCK
                return []
            self._reissue_count += 1
            log.debug(
                "ClearSkill: mine stall on entity %d, re-issuing (%d/%d)",
                self._current.entity_id, self._reissue_count, _MAX_REISSUE,
            )
            self._mine_issued_at = tick
            return [MineEntity(entity_id=self._current.entity_id)]

        return []

    def _build_target_list(self, wq: "WorldQuery") -> None:
        if self._bbox is None:
            return
        bb = self._bbox
        targets = []
        for entity in wq.state.entities:
            pos = entity.position
            if not (bb.x_min <= pos.x <= bb.x_max and bb.y_min <= pos.y <= bb.y_max):
                continue
            if self._clear_natural_only and not self._is_natural(entity):
                continue
            targets.append(_ClearTarget(entity_id=entity.entity_id, position=pos))

        player_pos = wq.player_position()
        targets.sort(key=lambda t: t.position.distance_to(player_pos))
        self._targets = targets
        log.info("ClearSkill: %d targets in bbox", len(targets))

    @staticmethod
    def _is_natural(entity) -> bool:
        """
        True if the entity appears to be a natural world object.

        Uses name heuristics — KB-free. Replace with prototype_type check
        in Phase 9 when EntityState gains that field.
        """
        name = entity.name.lower()
        return (
            "tree"    in name
            or "rock"    in name
            or "boulder" in name
            or "cliff"   in name
            or "fish"    in name
        )
