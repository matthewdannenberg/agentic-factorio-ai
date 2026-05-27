"""
execution/agents/crafting.py

CraftingAgent — queues hand-crafting jobs and confirms ingredient consumption.

Satisfies AgentProtocol. Owns one task type:

  craft_items
    Craft one or more items by hand. The coordinator writes a list of
    {item, recipe, count} targets to the blackboard as a crafting_task
    INTENTION entry, along with the pre-computed expected_post_inventory.
    The agent emits CraftItem actions for each target, then monitors
    ingredient depletion as confirmation that Factorio accepted the requests.

Crafting model
--------------
fa.craft_item() queues a crafting job in Factorio's crafting queue.
Ingredients are consumed *immediately* when the job is queued — the
output items arrive in a trickle as crafting completes. This is the
key property the agent exploits: ingredient consumption is the
confirmation signal, not output arrival.

The agent issues all CraftItem commands on the first tick. It then
watches for ingredient depletion toward the expected_post_inventory
written by the coordinator. If depletion isn't observed after
_CRAFTING_GRACE_TICKS, the command is re-issued — this handles dropped
RCON commands without being fooled by normal crafting delays on the
output side.

Stall detection
---------------
The coordinator computes expected_post_inventory (via post_crafting_inventory
in preconditions.py) and writes it directly into the INTENTION entry.
The agent reads it at activate() time and needs no KB access for cost
calculations — the coordinator has already done that work.

Success detection
-----------------
The coordinator derives a success_condition based on ingredient depletion
(or output accumulation as a fallback). The agent does not evaluate
success conditions — the coordinator handles that as normal.

Multi-target crafting
---------------------
A single craft_items task may request multiple item types. All CraftItem
commands are issued on the same tick. expected_post_inventory covers the
combined effect of all targets sequentially.

Rules
-----
- No LLM calls. No RCON. Satisfies AgentProtocol.
- No KB access needed — all cost information comes from the coordinator
  via the INTENTION entry.
- Agents receive Tasks, not Goals. No Goal is stored or inspected.
- All state between ticks stored on the instance. Cleared on activate().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from execution.blackboard import EntryCategory, EntryScope
from execution.agents.base import AgentProtocol
from bridge import Action, CraftItem

if TYPE_CHECKING:
    from execution.blackboard import Blackboard
    from planning.tasks.task import Task
    from world import KnowledgeBase
    from world import WorldQuery
    from world import WorldWriter

log = logging.getLogger(__name__)

AGENT_ID = "crafting"

# Ticks to wait after issuing CraftItem commands before checking for
# ingredient stall. Crafting consumption is instantaneous in Factorio,
# but state polling has a lag of up to TICK_INTERVAL ticks. This grace
# period avoids a false stall on the very first poll after dispatch.
# At 60 tps and TICK_INTERVAL=10: 60 ticks = 1 second — plenty of margin.
_CRAFTING_GRACE_TICKS = 60

# Maximum re-issue attempts before giving up and returning stale state.
# If ingredients haven't moved after this many re-issues, something is
# structurally wrong (e.g. recipe locked, inventory full mid-craft).
_MAX_REISSUE_ATTEMPTS = 3


@dataclass
class _CraftTarget:
    """One item type to craft."""
    item: str    # output item name
    recipe: str  # recipe name (may differ from item for some recipes)
    count: int   # number of units to craft


class CraftingAgent(AgentProtocol):
    """
    Hand-crafting agent.

    Reads a crafting_task INTENTION entry from the blackboard (written by the
    coordinator) and emits CraftItem actions. The INTENTION entry carries both
    the list of targets and the pre-computed expected_post_inventory, so the
    agent requires no KB access for stall detection.
    """

    AGENT_ID = "crafting"

    def __init__(self) -> None:
        self._current_task: Optional["Task"] = None
        self._targets: list[_CraftTarget] = []
        # Inventory snapshot at activation.
        self._snapshot: dict[str, int] = {}
        # Expected inventory after all crafting orders are accepted, written
        # by the coordinator into the INTENTION entry.
        self._expected_post: dict[str, int] = {}
        self._issued_at: int = 0
        self._dispatched: bool = False
        self._reissue_count: int = 0

    # ------------------------------------------------------------------
    # AgentProtocol
    # ------------------------------------------------------------------

    def activate(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> None:
        self._current_task = task
        self._targets = []
        self._snapshot = self._inventory_snapshot(wq)
        self._expected_post = dict(self._snapshot)
        self._issued_at = 0
        self._dispatched = False
        self._reissue_count = 0

        task = self._find_task(blackboard, wq.tick)
        if task is not None:
            self._targets = self._resolve_targets(task.data)
            self._expected_post = dict(task.data.get("expected_post_inventory",
                                                      self._snapshot))

        log.debug(
            "CraftingAgent activated for task %s: %d target(s) — %s",
            task.id[:8], len(self._targets), task.description,
        )

    def tick(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
        kb: "KnowledgeBase",
    ) -> list[Action]:
        # Resolve targets lazily if activate() couldn't find the entry.
        if not self._targets and not self._dispatched:
            task = self._find_task(blackboard, tick)
            if task is None:
                return []
            self._targets = self._resolve_targets(task.data)
            self._expected_post = dict(task.data.get("expected_post_inventory",
                                                      self._snapshot))
            if not self._targets:
                log.warning("CraftingAgent: crafting_task entry has no valid targets")
                return []

        # First dispatch or re-issue after stall.
        if not self._dispatched or self._should_reissue(wq, tick):
            if self._dispatched:
                self._reissue_count += 1
                if self._reissue_count > _MAX_REISSUE_ATTEMPTS:
                    log.warning(
                        "CraftingAgent: %d re-issue attempts exhausted for "
                        "task %s — ingredient consumption not detected",
                        _MAX_REISSUE_ATTEMPTS, task.id[:8],
                    )
                    return []
                log.debug(
                    "CraftingAgent: ingredient stall detected, re-issuing "
                    "(attempt %d/%d)",
                    self._reissue_count, _MAX_REISSUE_ATTEMPTS,
                )

            actions = [CraftItem(recipe=t.recipe, count=t.count) for t in self._targets]
            self._dispatched = True
            self._issued_at = tick
            log.info(
                "CraftingAgent: issuing %d CraftItem command(s) — %s",
                len(actions),
                ", ".join(f"{t.count}x {t.item}" for t in self._targets),
            )
            return actions

        return []

    def observe(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> dict:
        depleted = self._total_ingredient_depletion(wq)
        expected = self._total_expected_depletion()
        return {
            "agent": AGENT_ID,
            "task_id": task.id[:8],
            "targets": [{"item": t.item, "count": t.count} for t in self._targets],
            "dispatched": self._dispatched,
            "reissue_count": self._reissue_count,
            "ingredient_depletion": depleted,
            "expected_depletion": expected,
        }

    def progress(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> float:
        if not self._dispatched:
            return 0.0
        expected = self._total_expected_depletion()
        if expected == 0:
            return 1.0
        return min(1.0, self._total_ingredient_depletion(wq) / expected)

    # ------------------------------------------------------------------
    # Stall detection
    # ------------------------------------------------------------------

    def _should_reissue(self, wq: "WorldQuery", tick: int) -> bool:
        if tick - self._issued_at < _CRAFTING_GRACE_TICKS:
            return False
        return self._total_ingredient_depletion(wq) == 0

    def _total_ingredient_depletion(self, wq: "WorldQuery") -> int:
        """
        Total ingredient units consumed toward the expected post-crafting state.

        Each item's contribution is capped at its expected reduction so that
        external consumption of the same ingredient doesn't inflate the signal.
        """
        current = self._inventory_snapshot(wq)
        total = 0
        for item, expected_after in self._expected_post.items():
            before = self._snapshot.get(item, 0)
            expected_consumed = max(0, before - expected_after)
            if expected_consumed == 0:
                continue
            actual_after = current.get(item, 0)
            actually_consumed = max(0, before - actual_after)
            total += min(actually_consumed, expected_consumed)
        return total

    def _total_expected_depletion(self) -> int:
        """Total ingredient units expected to be consumed across all targets."""
        total = 0
        for item, expected_after in self._expected_post.items():
            before = self._snapshot.get(item, 0)
            total += max(0, before - expected_after)
        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_targets(self, task_data: dict) -> list[_CraftTarget]:
        result = []
        for raw in task_data.get("targets", []):
            item = raw.get("item", "")
            recipe_name = raw.get("recipe", item)
            count = int(raw.get("count", 0))
            if not item or count <= 0:
                continue
            result.append(_CraftTarget(item=item, recipe=recipe_name, count=count))
        return result

    def _find_task(self, blackboard: "Blackboard", tick: int) -> Optional[object]:
        intentions = blackboard.read(
            category=EntryCategory.INTENTION,
            current_tick=tick,
        )
        tasks = [e for e in intentions if e.data.get("type") == "crafting_task"]
        return tasks[0] if tasks else None

    def _inventory_snapshot(self, wq: "WorldQuery") -> dict[str, int]:
        snapshot: dict[str, int] = {}
        for slot in wq.state.player.inventory.slots:
            snapshot[slot.item] = snapshot.get(slot.item, 0) + slot.count
        return snapshot