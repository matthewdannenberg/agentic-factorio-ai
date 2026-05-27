"""
execution/skills/mine.py

MineSkill — mine a resource patch until count units have been collected.

Extracted from MiningAgent._tick_gather(). Handles issuing MineResource,
monitoring inventory for stall detection, and succeeding once the target
count has been reached.

start() parameters
------------------
position      : Tile-space Position of the resource patch to mine.
resource      : Resource type name (e.g. "iron-ore", "coal").
count         : Number of units to collect. The skill monitors inventory
                delta to detect completion. 0 means mine until exhausted
                or externally stopped (success condition never fires).

Status transitions
------------------
IDLE      → RUNNING  : start() called.
RUNNING   → SUCCEEDED: inventory delta >= count since start() (if count > 0).
RUNNING   → STUCK    : inventory unchanged after _MINING_GRACE_TICKS since
                       last MineResource command; re-issue attempted up to
                       _MAX_REISSUE before transitioning to STUCK.
Any       → IDLE     : reset() called.

Mining model
------------
fa.mine_resource() sets a persistent mining_state in Lua that mines the
tile continuously until the resource is exhausted or stop_mining() is called.
The skill issues the command once and re-issues only on stall detection.
The skill does NOT issue StopMining on success — the coordinator or agent
is responsible for stopping the miner after the skill reports SUCCEEDED,
since another skill may immediately continue on the same tile.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from execution.skills.base import SkillProtocol, SkillStatus
from bridge import Action, MineResource
from world import Position

if TYPE_CHECKING:
    from world import WorldQuery, WorldWriter

log = logging.getLogger(__name__)

# Ticks after issuing MineResource before checking for inventory stall.
# Mining a single ore tile takes several seconds; this must be long enough
# that we don't re-issue during a normal swing cycle.
# At 60 tps and TICK_INTERVAL=10: 300 ticks ≈ 5 seconds.
_MINING_GRACE_TICKS = 300

# Maximum re-issue attempts before declaring STUCK. After this many re-issues
# without any inventory change, something structural is wrong (patch exhausted
# in another session, player not reaching tile, inventory full).
_MAX_REISSUE = 3


class MineSkill(SkillProtocol):
    """
    Mine a resource patch until the target count is collected.

    Usage
    -----
        skill = MineSkill()
        skill.start(position=Position(x=10, y=20), resource="iron-ore", count=50)

        while skill.status() == SkillStatus.RUNNING:
            actions = skill.tick(wq, ww, tick)
            dispatch(actions)

        if skill.status() == SkillStatus.SUCCEEDED:
            ...
        elif skill.status() == SkillStatus.STUCK:
            # inventory stalled — try moving to adjacent tile or escalate
            skill.reset()
    """

    def __init__(self) -> None:
        self._status: SkillStatus = SkillStatus.IDLE
        self._position: Optional[Position] = None
        self._resource: str = ""
        self._count: int = 0
        self._issued_at: int = 0
        self._reissue_count: int = 0
        self._snapshot_taken: bool = False
        self._snapshot_at_start: dict[str, int] = {}
        self._last_inventory: dict[str, int] = {}

    # ------------------------------------------------------------------
    # SkillProtocol
    # ------------------------------------------------------------------

    def start(
        self,
        position: Position,
        resource: str,
        count: int = 0,
    ) -> None:
        """
        Initialise for a new mining job.

        Parameters
        ----------
        position : Position of the resource patch tile to mine.
        resource : Resource type name (e.g. "iron-ore").
        count    : Units to collect before reporting SUCCEEDED. 0 = mine
                   indefinitely (SUCCEEDED never fires; agent must reset()).
        """
        self._position   = position
        self._resource   = resource
        self._count      = count
        self._issued_at  = 0
        self._reissue_count = 0
        self._snapshot_taken    = False
        self._snapshot_at_start = {}
        self._last_inventory    = {}
        self._status = SkillStatus.RUNNING
        log.debug(
            "MineSkill started: resource=%s pos=%s count=%d",
            resource, position, count,
        )

    def tick(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        if self._status != SkillStatus.RUNNING:
            return []

        current_inv = self._inventory_snapshot(wq)

        # Capture inventory at first tick so we can measure delta.
        if not self._snapshot_taken:
            self._snapshot_at_start = current_inv
            self._snapshot_taken = True

        # --- Completion check (only when count > 0) ---
        if self._count > 0:
            delta = (
                current_inv.get(self._resource, 0)
                - self._snapshot_at_start.get(self._resource, 0)
            )
            if delta >= self._count:
                log.debug(
                    "MineSkill: collected %d %s — SUCCEEDED", delta, self._resource
                )
                self._status = SkillStatus.SUCCEEDED
                return []

        # --- Issue or re-issue MineResource ---
        if self._issued_at == 0:
            # First issue.
            self._issued_at = tick
            self._last_inventory = current_inv
            return [MineResource(
                position=self._position,
                resource=self._resource,
                count=0,  # 0 = mine continuously; skill handles count tracking
            )]

        # --- Stall detection ---
        grace_elapsed = (tick - self._issued_at) >= _MINING_GRACE_TICKS
        if grace_elapsed and current_inv == self._last_inventory:
            if self._reissue_count >= _MAX_REISSUE:
                log.warning(
                    "MineSkill: %d re-issues exhausted with no inventory change "
                    "— STUCK (resource=%s pos=%s)",
                    _MAX_REISSUE, self._resource, self._position,
                )
                self._status = SkillStatus.STUCK
                return []

            self._reissue_count += 1
            log.debug(
                "MineSkill: inventory stall, re-issuing (attempt %d/%d)",
                self._reissue_count, _MAX_REISSUE,
            )
            self._issued_at = tick
            self._last_inventory = current_inv
            return [MineResource(
                position=self._position,
                resource=self._resource,
                count=0,
            )]

        # Update last-known inventory each tick for stall detection.
        self._last_inventory = current_inv
        return []

    def status(self) -> SkillStatus:
        return self._status

    def reset(self) -> None:
        self._status            = SkillStatus.IDLE
        self._position          = None
        self._resource          = ""
        self._count             = 0
        self._issued_at         = 0
        self._reissue_count     = 0
        self._snapshot_taken    = False
        self._snapshot_at_start = {}
        self._last_inventory    = {}

    def observe(self) -> dict:
        return {
            "mine_status":        self._status.name,
            "mine_resource":      self._resource,
            "mine_target_count":  self._count,
            "mine_reissue_count": self._reissue_count,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _inventory_snapshot(self, wq: "WorldQuery") -> dict[str, int]:
        return {
            slot.item: slot.count
            for slot in wq.state.player.inventory.slots
        }