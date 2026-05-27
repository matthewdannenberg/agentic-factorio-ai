"""
execution/skills/craft.py

CraftSkill — queue hand-crafting jobs and confirm ingredient consumption.

Extracted from CraftingAgent. Handles issuing CraftItem commands, monitoring
ingredient depletion as the confirmation signal, and stall re-issue logic.

start() parameters
------------------
targets              : List of CraftTarget — the items and counts to craft.
expected_post_inv    : Dict mapping item name → expected inventory count
                       *after* all crafting orders are accepted. Written by
                       the coordinator (via preconditions.post_crafting_inventory)
                       so the skill needs no KnowledgeBase access.

Status transitions
------------------
IDLE      → RUNNING  : start() called with at least one target.
RUNNING   → SUCCEEDED: total ingredient depletion equals expected depletion.
RUNNING   → STUCK    : no ingredient movement after _CRAFTING_GRACE_TICKS,
                       re-issued _MAX_REISSUE times without result.
Any       → IDLE     : reset() called.

Crafting model
--------------
fa.craft_item() queues jobs in Factorio's crafting queue. Ingredients are
consumed *immediately* when queued — output arrives later. Ingredient
depletion is therefore the correct confirmation signal, not output arrival.

All CraftItem commands are issued on the first tick. The skill then watches
for ingredient movement. If ingredients haven't moved after the grace period,
the commands are re-issued.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from execution.skills.base import SkillProtocol, SkillStatus
from bridge import Action, CraftItem

if TYPE_CHECKING:
    from world import WorldQuery, WorldWriter

log = logging.getLogger(__name__)

# Ticks after issuing CraftItem before checking for ingredient stall.
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
        expected_post = {"iron-plate": 80}   # pre-computed by coordinator

        skill = CraftSkill()
        skill.start(targets=targets, expected_post_inv=expected_post)

        while skill.status() == SkillStatus.RUNNING:
            actions = skill.tick(wq, ww, tick)
            dispatch(actions)
    """

    def __init__(self) -> None:
        self._status: SkillStatus = SkillStatus.IDLE
        self._targets: list[CraftTarget] = []
        self._expected_post: dict[str, int] = {}
        self._snapshot: dict[str, int] = {}    # inventory at start()
        self._issued_at: int = 0
        self._dispatched: bool = False
        self._reissue_count: int = 0

    # ------------------------------------------------------------------
    # SkillProtocol
    # ------------------------------------------------------------------

    def start(
        self,
        targets: list[CraftTarget],
        expected_post_inv: Optional[dict[str, int]] = None,
    ) -> None:
        """
        Initialise for a new crafting job.

        Parameters
        ----------
        targets           : Items to craft (item, recipe, count).
        expected_post_inv : Inventory state expected after all jobs are
                            accepted. If None, the skill will issue the
                            commands but always reports SUCCEEDED immediately
                            after dispatch (fire-and-forget mode).
        """
        if not targets:
            raise ValueError("CraftSkill.start() requires at least one target")
        self._targets        = list(targets)
        self._expected_post  = dict(expected_post_inv) if expected_post_inv else {}
        self._snapshot       = {}
        self._issued_at      = 0
        self._dispatched     = False
        self._reissue_count  = 0
        self._status = SkillStatus.RUNNING
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

        current_inv = self._inventory_snapshot(wq)

        # Capture snapshot on first tick.
        if not self._snapshot:
            self._snapshot = current_inv

        # Fire-and-forget mode: no expected_post_inv supplied.
        if not self._expected_post:
            if not self._dispatched:
                actions = self._build_actions()
                self._dispatched = True
                self._issued_at  = tick
                self._status = SkillStatus.SUCCEEDED
                return actions
            return []

        # --- Completion check ---
        if self._dispatched and self._ingredients_depleted(current_inv):
            log.debug("CraftSkill: ingredient depletion confirmed — SUCCEEDED")
            self._status = SkillStatus.SUCCEEDED
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

        # --- Stall detection and re-issue ---
        grace_elapsed = (tick - self._issued_at) >= _CRAFTING_GRACE_TICKS
        if grace_elapsed and self._total_ingredient_depletion(current_inv) == 0:
            if self._reissue_count >= _MAX_REISSUE:
                log.warning(
                    "CraftSkill: %d re-issues with no ingredient movement — STUCK",
                    _MAX_REISSUE,
                )
                self._status = SkillStatus.STUCK
                return []
            self._reissue_count += 1
            log.debug(
                "CraftSkill: ingredient stall, re-issuing (%d/%d)",
                self._reissue_count, _MAX_REISSUE,
            )
            actions = self._build_actions()
            self._issued_at = tick
            return actions

        return []

    def status(self) -> SkillStatus:
        return self._status

    def reset(self) -> None:
        self._status        = SkillStatus.IDLE
        self._targets       = []
        self._expected_post = {}
        self._snapshot      = {}
        self._issued_at     = 0
        self._dispatched    = False
        self._reissue_count = 0

    def observe(self) -> dict:
        return {
            "craft_status":        self._status.name,
            "craft_targets":       [{"item": t.item, "count": t.count} for t in self._targets],
            "craft_dispatched":    self._dispatched,
            "craft_reissue_count": self._reissue_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_actions(self) -> list[Action]:
        return [CraftItem(recipe=t.recipe, count=t.count) for t in self._targets]

    def _ingredients_depleted(self, current_inv: dict[str, int]) -> bool:
        """True if all expected ingredient reductions have been observed."""
        return self._total_ingredient_depletion(current_inv) >= self._total_expected_depletion()

    def _total_ingredient_depletion(self, current_inv: dict[str, int]) -> int:
        """
        Total ingredient units consumed toward the expected post-crafting state.
        Each item's contribution is capped at its expected reduction to prevent
        unrelated inventory changes from inflating the signal.
        """
        total = 0
        for item, expected_after in self._expected_post.items():
            before = self._snapshot.get(item, 0)
            expected_consumed = max(0, before - expected_after)
            if expected_consumed == 0:
                continue
            actual_after = current_inv.get(item, 0)
            actually_consumed = max(0, before - actual_after)
            total += min(actually_consumed, expected_consumed)
        return total

    def _total_expected_depletion(self) -> int:
        total = 0
        for item, expected_after in self._expected_post.items():
            before = self._snapshot.get(item, 0)
            total += max(0, before - expected_after)
        return total

    def _inventory_snapshot(self, wq: "WorldQuery") -> dict[str, int]:
        return {
            slot.item: slot.count
            for slot in wq.state.player.inventory.slots
        }
