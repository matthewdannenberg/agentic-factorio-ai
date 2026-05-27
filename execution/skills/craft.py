"""
execution/skills/craft.py

CraftSkill — queue hand-crafting jobs and confirm they were accepted.

start() parameters
------------------
targets : List of CraftTarget — the items and counts to craft.

Status transitions
------------------
IDLE      -> RUNNING  : start() called with at least one target.
RUNNING   -> SUCCEEDED: crafting_queue_size > 0 (primary — Factorio accepted
                        the jobs and the queue is populated), OR all target
                        items are present in inventory at the required counts
                        (fallback — jobs finished before we could observe the
                        queue, which is fine: the player has the items).
RUNNING   -> STUCK    : queue empty AND items not in inventory after
                        _CRAFTING_GRACE_TICKS, re-issued _MAX_REISSUE times.
Any       -> IDLE     : reset() called.

Confirmation model
------------------
fa.craft_item() queues jobs in Factorio's crafting queue. Ingredients are
consumed immediately when queued; output arrives as recipes complete.

The skill confirms acceptance via two complementary signals:

  Primary   : crafting_queue_size > 0
              Unambiguous: Factorio populated the queue.

  Fallback  : inventory counts for all targets met
              Handles the case where crafting completed so quickly that the
              queue was already empty by the time we polled. If the items are
              in inventory, the jobs ran successfully.

No snapshot, no ingredient depletion arithmetic, no expected_post_inv needed.
The coordinator uses preconditions.post_crafting_inventory for its own
task success conditions; the skill does not need that information.

Edge case noted (not handled, assessed as negligible risk)
-----------------------------------------------------------
If a prior long-running queue is still active when a new CraftSkill starts,
and the new dispatch is dropped over RCON, the skill might falsely report
SUCCEEDED because the old queue is still populated. This requires: a prior
queue still running, a fast intermediate task, an immediate new craft task,
and a dropped RCON command — all simultaneously. Re-issue logic on the next
stall cycle would catch it. Probability assessed as negligible.

Rules
-----
- No LLM calls. No RCON. No KnowledgeBase. No snapshot.
- Agents receive Tasks, not Goals. No Goal stored or inspected.
- All state between ticks stored on the instance. Cleared on reset().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from execution.skills.base import SkillProtocol, SkillStatus
from bridge import Action, CraftItem

if TYPE_CHECKING:
    from world import WorldQuery, WorldWriter

log = logging.getLogger(__name__)

# Ticks after issuing CraftItem before checking for stall.
# At 60 tps and TICK_INTERVAL=10: 60 ticks = 1 second.
_CRAFTING_GRACE_TICKS = 60

# Maximum re-issue attempts before declaring STUCK.
_MAX_REISSUE = 3


@dataclass
class CraftTarget:
    """One item type to craft."""
    item: str    # output item name
    recipe: str  # recipe name (may differ from item for some recipes)
    count: int   # units to craft


class CraftSkill(SkillProtocol):
    """
    Hand-craft a list of items.

    Usage
    -----
        targets = [CraftTarget(item="iron-gear-wheel", recipe="iron-gear-wheel", count=10)]

        skill = CraftSkill()
        skill.start(targets=targets)

        while skill.status() == SkillStatus.RUNNING:
            actions = skill.tick(wq, ww, tick)
            dispatch(actions)

        if skill.status() == SkillStatus.SUCCEEDED:
            ...   # queue was populated or items already in inventory
        elif skill.status() == SkillStatus.STUCK:
            ...   # dispatch failed repeatedly — escalate
    """

    def __init__(self) -> None:
        self._status: SkillStatus = SkillStatus.IDLE
        self._targets: list[CraftTarget] = []
        self._issued_at: int = 0
        self._dispatched: bool = False
        self._reissue_count: int = 0

    # ------------------------------------------------------------------
    # SkillProtocol
    # ------------------------------------------------------------------

    def start(self, targets: list[CraftTarget]) -> None:
        """
        Initialise for a new crafting job.

        Parameters
        ----------
        targets : Items to craft (item, recipe, count). Must be non-empty.
        """
        if not targets:
            raise ValueError("CraftSkill.start() requires at least one target")
        self._targets       = list(targets)
        self._issued_at     = 0
        self._dispatched    = False
        self._reissue_count = 0
        self._status        = SkillStatus.RUNNING
        log.debug(
            "CraftSkill started: %d target(s) — %s",
            len(targets),
            ", ".join(f"{t.count}x {t.item}" for t in targets),
        )

    def tick(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        if self._status != SkillStatus.RUNNING:
            return []

        # --- First dispatch ---
        if not self._dispatched:
            actions = self._build_actions()
            self._dispatched = True
            self._issued_at  = tick
            log.info(
                "CraftSkill: dispatching %d CraftItem(s) — %s",
                len(actions),
                ", ".join(f"{t.count}x {t.item}" for t in self._targets),
            )
            return actions

        # --- Confirmation check ---
        # Primary: queue populated means Factorio accepted the jobs.
        # Fallback: items already in inventory means jobs finished quickly.
        if wq.crafting_queue_size > 0:
            log.debug("CraftSkill: queue_size=%d — SUCCEEDED", wq.crafting_queue_size)
            self._status = SkillStatus.SUCCEEDED
            return []

        if self._items_in_inventory(wq):
            log.debug("CraftSkill: all target items in inventory — SUCCEEDED")
            self._status = SkillStatus.SUCCEEDED
            return []

        # --- Stall detection ---
        grace_elapsed = (tick - self._issued_at) >= _CRAFTING_GRACE_TICKS
        if grace_elapsed:
            if self._reissue_count >= _MAX_REISSUE:
                log.warning(
                    "CraftSkill: %d re-issues with no queue or inventory "
                    "confirmation — STUCK",
                    _MAX_REISSUE,
                )
                self._status = SkillStatus.STUCK
                return []

            self._reissue_count += 1
            self._issued_at = tick
            log.debug(
                "CraftSkill: stall, re-issuing (%d/%d)",
                self._reissue_count, _MAX_REISSUE,
            )
            return self._build_actions()

        return []

    def status(self) -> SkillStatus:
        return self._status

    def reset(self) -> None:
        self._status        = SkillStatus.IDLE
        self._targets       = []
        self._issued_at     = 0
        self._dispatched    = False
        self._reissue_count = 0

    def observe(self) -> dict:
        return {
            "craft_status":        self._status.name,
            "craft_targets":       [{"item": t.item, "count": t.count}
                                    for t in self._targets],
            "craft_dispatched":    self._dispatched,
            "craft_reissue_count": self._reissue_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_actions(self) -> list[Action]:
        return [CraftItem(recipe=t.recipe, count=t.count) for t in self._targets]

    def _items_in_inventory(self, wq: "WorldQuery") -> bool:
        """True if all targets have their required counts in player inventory."""
        for target in self._targets:
            if wq.inventory_count(target.item) < target.count:
                return False
        return True