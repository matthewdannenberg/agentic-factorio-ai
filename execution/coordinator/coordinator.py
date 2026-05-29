"""
execution/coordinator/coordinator.py

RuleBasedCoordinator — hierarchical goal decomposition for the execution layer.

Architecture
------------
The coordinator maintains two stacks:

  _goal_stack   : list[GoalFrame]
      A stack of pending goals in descending priority (top = active).
      Each GoalFrame carries the Goal object, a step counter, and any
      derived context (patch position, bbox, etc.) needed across ticks.
      When the top goal completes, it is popped and the parent resumes.

  _task_ledger  : TaskLedger
      The flat list of concrete Tasks sent to agents. When the ledger is
      non-empty, the coordinator ticks the active agent rather than
      performing goal derivation.

Tick loop
---------
Each tick:
  1. If a Task is active, evaluate its success/failure and tick the agent.
  2. If no Task is active, look at the top GoalFrame and call its handler.
  3. The handler either pushes a new Task (ledger), pushes a sub-goal
     (goal_stack), or determines the current goal is complete/failed.

Goal type vocabulary
--------------------
GOAL_COLLECTION    "collection"     Get N of item from resource patches
GOAL_ACQUIRE       "acquire"        Get N of item by any means
GOAL_CRAFTING      "crafting"       Hand-craft N of item
GOAL_EXPLORE       "exploration"    Chart N new chunks
GOAL_CLEAR_REGION  "clear_region"   Clear natural objects from a bbox
GOAL_PREP_REGION   "prep_region"    Safely clear a region (factory-aware)
GOAL_CONSTRUCTION  "construction"   Build infrastructure in a region
GOAL_PRODUCTION    "production"     Establish item production at a rate
GOAL_LOGISTICS     "logistics"      Connect factory nodes with belts/inserters
GOAL_BYPRODUCT     "byproduct"      Route/consume all output of a factory node
GOAL_RESEARCH      "research"       Unlock a technology
GOAL_NOOP          "noop"           Idle — ask LLM for the next goal

Unimplemented stubs
-------------------
Several handlers are stubs pending later phases:

  _handle_prep_region     — STUB: logistics rerouting (Phase 7/8)
  _handle_construction    — STUB: build task (Phase 7 RL agent)
  _handle_production      — STUB: region delineation + logistics + byproducts
  _handle_logistics       — STUB: build-line task (Phase 9)
  _handle_byproduct       — STUB: similar to production
  _handle_research        — STUB: science production + lab construction
  _handle_noop            — STUB: ask LLM

Each stub returns WAITING (not STUCK) — the coordinator knows conceptually what
to do but lacks the execution capability. STUCK is reserved for cases where
the coordinator has genuinely run out of options.

Rules
-----
- No LLM calls. No RCON. Pure coordination logic.
- Goal handlers must not call each other directly — push to _goal_stack.
- Tasks are the only thing sent to agents. Goals are coordinator-internal.
- EntryScope.TASK (not GOAL) for task-scoped blackboard entries.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

from execution.blackboard import Blackboard, EntryCategory, EntryScope
from execution.skills.base import SkillStatus
from world import Position
from world.model.patch import SelfModelPatch

from planning import Task

if TYPE_CHECKING:
    from execution.agents.base import AgentProtocol
    from world import KnowledgeBase, WorldQuery, WorldWriter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Goal type constants
# ---------------------------------------------------------------------------

GOAL_COLLECTION   = "collection"
GOAL_ACQUIRE      = "acquire"
GOAL_CRAFTING     = "crafting"
GOAL_EXPLORE      = "exploration"
GOAL_CLEAR_REGION = "clear_region"
GOAL_PREP_REGION  = "prep_region"
GOAL_CONSTRUCTION = "construction"
GOAL_PRODUCTION   = "production"
GOAL_LOGISTICS    = "logistics"
GOAL_BYPRODUCT    = "byproduct"
GOAL_RESEARCH     = "research"
GOAL_NOOP         = "noop"

# ---------------------------------------------------------------------------
# Task timeout constants
# ---------------------------------------------------------------------------

_TASK_TIMEOUT_TICKS     = 18_000   # 5 min — general tasks
_NAV_TASK_TIMEOUT_TICKS =  1_800   # 30 s  — navigation only


# ---------------------------------------------------------------------------
# GoalFrame — one entry on the goal stack
# ---------------------------------------------------------------------------

@dataclass
class GoalFrame:
    """
    A goal and all coordinator-derived context needed to resume it.

    Pushed onto _goal_stack. The top frame is the currently active goal.
    When a sub-goal is pushed, the parent frame is suspended until the
    sub-goal's frame is popped.

    Fields
    ------
    goal_type   : Goal type string (GOAL_* constant).
    params      : Arbitrary dict of goal parameters (item name, count, bbox,
                  rate, etc.). Populated at push time; read by the handler.
    step        : Which step of the handler's decision tree we are on.
                  Incremented each time the handler advances.
    completed   : True when the goal has been achieved. The coordinator pops
                  the frame on the next tick.
    failed      : True when the goal cannot be achieved. The coordinator
                  escalates on the next tick.
    context     : Mutable dict for handler-specific state that must persist
                  across ticks (e.g. which resource patch was chosen, which
                  part of a region has been cleared).
    """
    goal_type: str
    params: dict = field(default_factory=dict)
    step: int = 0
    completed: bool = False
    failed: bool = False
    context: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TaskResult — what a task handler returns
# ---------------------------------------------------------------------------

class TaskOutcome(Enum):
    RUNNING   = auto()   # task is active, continue ticking
    SUCCEEDED = auto()   # task complete — advance goal step
    FAILED    = auto()   # task failed — escalate or skip
    STUCK     = auto()   # agent reported stuck — escalate goal


# ---------------------------------------------------------------------------
# CoordinatorStatus — what the coordinator returns to the loop each tick
# ---------------------------------------------------------------------------

class CoordinatorStatus(Enum):
    PROGRESSING = auto()
    WAITING     = auto()   # no current work, but not stuck
    STUCK       = auto()   # escalate to LLM
    COMPLETE    = auto()   # top-level goal achieved


# ---------------------------------------------------------------------------
# RuleBasedCoordinator
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# StubCoordinator (retained for reference / tests that import it)
# ---------------------------------------------------------------------------

class StubCoordinator:
    """
    Minimal no-op coordinator stub retained for Phase 6 test compatibility.

    Matches the old CoordinatorProtocol signature (goal, wq, ww, tick) and
    returns an ExecutionResult so existing tests that import StubCoordinator
    continue to work unchanged.
    """

    def reset(self, goal=None, wq=None, seed_subtasks=None) -> None:
        pass

    def tick(self, goal=None, wq=None, ww=None, tick: int = 0):
        from execution.protocol import ExecutionResult, ExecutionStatus
        return ExecutionResult(actions=[], status=ExecutionStatus.WAITING)


class RuleBasedCoordinator:
    """
    Hierarchical rule-based coordinator.

    Usage
    -----
    coordinator = RuleBasedCoordinator(registry, kb)
    coordinator.reset(goal_type, params, wq)

    Each tick:
        status, actions = coordinator.tick(wq, ww, tick)
    """

    def __init__(
        self,
        registry,          # AgentRegistry
        kb: "KnowledgeBase",
        blackboard: Optional[Blackboard] = None,
    ) -> None:
        self._registry  = registry
        self._kb        = kb
        self._bb        = blackboard or Blackboard()

        self._goal_stack: list[GoalFrame] = []
        self._active_task: Optional["Task"] = None
        self._active_agent: Optional["AgentProtocol"] = None
        self._pending_patches: list[SelfModelPatch] = []
        # Set when the top-level goal fails internally (frame.failed on the
        # only remaining frame). Prevents subsequent ticks from returning COMPLETE
        # (empty stack) before the loop's stuck-retry mechanism fails the goal.
        self._top_level_failed: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(
        self,
        goal_type: str,
        params: dict,
        wq: "WorldQuery",
    ) -> None:
        """
        Begin a new top-level goal.

        Clears all internal state and pushes a single GoalFrame for the
        given goal type. The first tick will call the appropriate handler.
        """
        self._goal_stack.clear()
        self._active_task = None
        self._active_agent = None
        self._pending_patches.clear()
        self._top_level_failed = False
        self._bb.clear_all()

        self._goal_stack.append(GoalFrame(goal_type=goal_type, params=params))
        log.info(
            "Coordinator reset: goal_type=%s params=%s", goal_type, params
        )

    def tick(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        One coordination cycle. Returns (status, actions).

        If a task is active, ticks the owning agent and evaluates
        task success/failure. Otherwise, runs the top goal handler.
        """
        if not self._goal_stack:
            if self._top_level_failed:
                return CoordinatorStatus.STUCK, []
            return CoordinatorStatus.COMPLETE, []

        # --- Tick active task ---
        if self._active_task is not None:
            outcome, actions = self._tick_task(wq, ww, tick)

            # Drain patches from the agent.
            if self._active_agent is not None:
                self._pending_patches.extend(
                    self._active_agent.pending_patches()
                )

            if outcome == TaskOutcome.RUNNING:
                return CoordinatorStatus.PROGRESSING, actions

            # Task resolved — clear it and let the handler decide what's next.
            self._active_task = None
            if self._active_agent is not None:
                self._active_agent = None
            self._bb.clear_scope(EntryScope.TASK)

            if outcome == TaskOutcome.SUCCEEDED:
                top = self._goal_stack[-1]
                top.step += 1
                log.debug(
                    "Task succeeded — goal %s advancing to step %d",
                    top.goal_type, top.step,
                )
            elif outcome in (TaskOutcome.FAILED, TaskOutcome.STUCK):
                top = self._goal_stack[-1]
                top.failed = True
                log.warning(
                    "Task %s on goal %s — marking goal failed",
                    outcome.name, top.goal_type,
                )

        # --- Check for completed / failed goals and propagate upward ---
        while self._goal_stack:
            top = self._goal_stack[-1]
            if top.completed:
                self._goal_stack.pop()
                if self._goal_stack:
                    # Parent resumes — advance its step.
                    self._goal_stack[-1].step += 1
                    log.debug(
                        "Sub-goal %s completed — parent %s advancing to step %d",
                        top.goal_type,
                        self._goal_stack[-1].goal_type,
                        self._goal_stack[-1].step,
                    )
                continue
            if top.failed:
                self._goal_stack.pop()
                if self._goal_stack:
                    # Parent sub-goal failed — mark parent failed too.
                    self._goal_stack[-1].failed = True
                    log.warning(
                        "Sub-goal %s failed — propagating to parent %s",
                        top.goal_type,
                        self._goal_stack[-1].goal_type,
                    )
                else:
                    # Top-level goal failed — set flag so subsequent ticks
                    # continue returning STUCK until the loop resets us.
                    self._top_level_failed = True
                    return CoordinatorStatus.STUCK, []
                continue
            break

        if not self._goal_stack:
            if self._top_level_failed:
                return CoordinatorStatus.STUCK, []
            return CoordinatorStatus.COMPLETE, []

        # --- Run top goal handler ---
        return self._dispatch_goal(self._goal_stack[-1], wq, tick)

    def drain_patches(self) -> list[SelfModelPatch]:
        """Return and clear accumulated self-model patches."""
        patches = list(self._pending_patches)
        self._pending_patches.clear()
        return patches

    # ------------------------------------------------------------------
    # Goal dispatch
    # ------------------------------------------------------------------

    def _dispatch_goal(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        handler = {
            GOAL_COLLECTION:   self._handle_collection,
            GOAL_ACQUIRE:      self._handle_acquire,
            GOAL_CRAFTING:     self._handle_crafting,
            GOAL_EXPLORE:      self._handle_explore,
            GOAL_CLEAR_REGION: self._handle_clear_region,
            GOAL_PREP_REGION:  self._handle_prep_region,
            GOAL_CONSTRUCTION: self._handle_construction,
            GOAL_PRODUCTION:   self._handle_production,
            GOAL_LOGISTICS:    self._handle_logistics,
            GOAL_BYPRODUCT:    self._handle_byproduct,
            GOAL_RESEARCH:     self._handle_research,
            GOAL_NOOP:         self._handle_noop,
        }.get(frame.goal_type)

        if handler is None:
            log.warning(
                "No handler for goal type %r — returning STUCK", frame.goal_type
            )
            return CoordinatorStatus.STUCK, []

        return handler(frame, wq, tick)

    # ------------------------------------------------------------------
    # Goal handlers
    # Each handler takes (frame, wq, tick) and returns (status, actions).
    # Handlers are called repeatedly across ticks — frame.step tracks
    # which decision point the handler has reached.
    # ------------------------------------------------------------------

    # ── Collection ──────────────────────────────────────────────────────

    def _handle_collection(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        Collect N units of a resource.

        params: {item, count}

        Decision tree
        -------------
        Step 0 — Source selection:
          a) If item already in inventory ≥ count → complete immediately.
          b) If item available in a reachable chest → STUB: open_chest task
             (Phase 7 — requires InteractSkill).
          c) If a known resource patch exists → push gather_resource Task.
          d) Else → STUCK (no known source — explorer should find one first).

        Step 1 — After gather task:
          If inventory ≥ count → complete.
          Else → back to step 0 (re-derive, patch may have shifted).
        """
        item  = frame.params.get("item", "")
        count = frame.params.get("count", 0)

        if not item:
            log.warning("Collection goal missing 'item' param — STUCK")
            frame.failed = True
            return CoordinatorStatus.STUCK, []

        # Already have enough.
        if wq.inventory_count(item) >= count:
            log.info("Collection goal already satisfied: %dx %s", count, item)
            frame.completed = True
            return CoordinatorStatus.PROGRESSING, []

        if frame.step == 0:
            # Check for item in chests.
            # STUB: chest extraction not yet implemented (Phase 7 InteractSkill).
            # When implemented: find nearest chest containing item, push
            # navigate_to_entity + extract_from_chest task sequence.
            if _item_in_nearby_chest(item, wq):
                log.warning(
                    "Collection: %s found in chest but chest interaction "
                    "is not yet implemented (Phase 7) — falling through to mining",
                    item,
                )
                # Fall through to mining rather than returning STUCK so the
                # goal can still complete via the resource patch.

            # Find nearest resource patch.
            patches = wq.resources_of_type(item)
            if not patches:
                log.warning(
                    "Collection: no known %s patches — STUCK "
                    "(explore to find resource first)",
                    item,
                )
                frame.failed = True
                return CoordinatorStatus.STUCK, []

            player_pos = wq.player_position()
            nearest = min(
                patches,
                key=lambda p: _dist(p.position, player_pos),
            )
            frame.context["patch_pos"] = nearest.position

            # Push a single gather_resource task. MiningAgent handles its
            # own approach navigation internally via NavigateSkill — a
            # separate navigate_to_position task would cause two navigation
            # hops (coordinator nav to patch centre, then agent nav to
            # mining position) where one is sufficient.
            gather_target = wq.inventory_count(item) + count
            self._push_task(
                task_type   = "gather_resource",
                description = f"Gather {count}x {item}",
                agent_hint  = "mining",
                tick        = tick,
                resource_type     = item,
                target_position   = nearest.position,
                count             = count,
                success_condition = f"inventory('{item}') >= {gather_target}",
            )
            return CoordinatorStatus.PROGRESSING, []

        # Step ≥ 1: check if we have enough now.
        if wq.inventory_count(item) >= count:
            frame.completed = True
            return CoordinatorStatus.PROGRESSING, []

        # Not enough — reset to step 0 to re-derive (patch may have shifted).
        frame.step = 0
        return CoordinatorStatus.WAITING, []

    # ── Acquire (Generalised Collection) ────────────────────────────────

    def _handle_acquire(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        Acquire N units of an item by any means (mine, produce, or retrieve).

        params: {item, count}

        Decision tree
        -------------
        Step 0 — Source selection:
          a) If inventory already sufficient → complete.
          b) If natural resource (known patch) → Collection sub-goal.
          c) If producible (KB has a recipe and we can eventually get inputs)
             → Production sub-goal.
          d) Else → STUCK.
        """
        item  = frame.params.get("item", "")
        count = frame.params.get("count", 0)

        if not item:
            log.warning("Acquire goal missing 'item' param — STUCK")
            frame.failed = True
            return CoordinatorStatus.STUCK, []

        if wq.inventory_count(item) >= count:
            frame.completed = True
            return CoordinatorStatus.PROGRESSING, []

        if frame.step == 0:
            # Is it a natural resource we can mine?
            if wq.resources_of_type(item):
                log.info("Acquire %dx %s via Collection", count, item)
                self._push_goal(GOAL_COLLECTION, {"item": item, "count": count})
                return CoordinatorStatus.PROGRESSING, []

            # Can it be produced?
            # STUB: Production goal not yet implemented (Phase 8).
            # When implemented: check KB for a recipe chain, then push
            # GOAL_PRODUCTION with the target item and rate.
            recipes = self._kb.recipes_for_product(item) if self._kb else []
            if recipes:
                log.warning(
                    "Acquire: %s is producible but GOAL_PRODUCTION is not yet "
                    "implemented (Phase 8) — STUCK",
                    item,
                )
                frame.failed = True
                return CoordinatorStatus.STUCK, []

            # No resource patch and no recipe — STUCK.
            #
            # Known gap: items obtained by clearing natural objects (most
            # notably wood from trees in the base game) fall into this
            # branch. Wood is not a mineable resource patch and has no
            # crafting recipe, so _handle_acquire cannot satisfy a wood
            # request directly.
            #
            # The correct long-term fix is a third acquisition category:
            # "harvest from natural object". This requires:
            #
            #   1. KnowledgeBase.EntityRecord gaining a field:
            #          mining_products: dict[str, int]
            #      populated from proto.mineable_properties.products at
            #      runtime via fa.get_entity_prototype. This keeps the
            #      knowledge learned, not hardcoded.
            #
            #   2. A KB query:
            #          kb.entities_that_produce(item) -> list[str]
            #      returning entity names whose mining drops the target
            #      item (e.g. "wood" → ["tree-01", "tree-dead-dry", ...]).
            #
            #   3. A "harvest_natural" task type: MiningAgent locates
            #      entities of the returned type in wq.natural_objects,
            #      navigates to them, and destroys them. The scan radius
            #      already provides natural_objects so this is feasible.
            #
            # This is deferred to Phase 7. Until then:
            #   - In the base game, wood demands are small enough that
            #     incidental tree clearing from prep_region / clear_region
            #     goals typically covers them without a dedicated acquire.
            #   - A scenario or mod requiring large quantities of a
            #     natural-object drop will STUCK and require LLM escalation.
            #
            # DO NOT work around this with `if item == "wood"` — that
            # hardcodes a Factorio item name and breaks total-conversion mods.
            log.warning(
                "Acquire: no known source for %s — STUCK. "
                "(If this is a natural-object drop such as wood, see the "
                "harvest_natural gap comment in _handle_acquire.)",
                item,
            )
            frame.failed = True
            return CoordinatorStatus.STUCK, []

        # Step ≥ 1: sub-goal (collection or production) completed.
        if wq.inventory_count(item) >= count:
            frame.completed = True
        else:
            frame.step = 0   # retry
        return CoordinatorStatus.PROGRESSING, []

    # ── Crafting ─────────────────────────────────────────────────────────

    def _handle_crafting(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        Hand-craft N of an item.

        params: {item, recipe, count}

        Decision tree
        -------------
        Step 0 — Ingredient check:
          If any ingredient is insufficient → Acquire sub-goal for each.
        Step 1 (or 0 if ingredients are all present) — Craft:
          Push craft_items Task.
        Step 2 — Verify:
          If inventory ≥ count → complete.
          Else → back to step 0.
        """
        item   = frame.params.get("item", "")
        recipe = frame.params.get("recipe", item)
        count  = frame.params.get("count", 0)

        if not item:
            log.warning("Crafting goal missing 'item' param — STUCK")
            frame.failed = True
            return CoordinatorStatus.STUCK, []

        if wq.inventory_count(item) >= count:
            frame.completed = True
            return CoordinatorStatus.PROGRESSING, []

        if frame.step == 0:
            # Check ingredients.
            missing = _missing_ingredients(item, count, wq, self._kb)
            if missing:
                log.info(
                    "Crafting %dx %s: missing ingredients %s — acquiring",
                    count, item, missing,
                )
                for ing_item, ing_count in missing.items():
                    self._push_goal(
                        GOAL_ACQUIRE,
                        {"item": ing_item, "count": ing_count},
                    )
                # Step will advance to 1 after all acquire sub-goals complete.
                return CoordinatorStatus.PROGRESSING, []

            # Ingredients present — craft.
            frame.step = 1

        if frame.step == 1:
            self._push_task(
                task_type   = "craft_items",
                description = f"Craft {count}x {item}",
                agent_hint  = "crafting",
                tick        = tick,
                targets=[{"item": item, "recipe": recipe, "count": count}],
                success_condition = (
                    f"crafting_queue_size > 0 or "
                    f"inventory('{item}') >= "
                    f"{wq.inventory_count(item) + count}"
                ),
            )
            return CoordinatorStatus.PROGRESSING, []

        # Step 2 — verify arrival.
        if wq.inventory_count(item) >= count:
            frame.completed = True
        else:
            frame.step = 0   # retry from ingredient check
        return CoordinatorStatus.PROGRESSING, []

    # ── Exploration ───────────────────────────────────────────────────────

    def _handle_explore(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        Chart at least N new chunks.

        params: {target_chunks}

        Decision tree
        -------------
        Check if charted_chunks has reached the target each step.
        Push an explore_region Task toward the nearest frontier.
        When the task returns needs_frontier, push a new task toward the
        next frontier. Repeat until target is reached.
        """
        target = frame.params.get("target_chunks", 0)
        start  = frame.context.setdefault("start_chunks", wq.charted_chunks)

        if wq.charted_chunks - start >= target:
            frame.completed = True
            return CoordinatorStatus.PROGRESSING, []

        if frame.step == 0 or frame.context.get("needs_frontier"):
            frame.context["needs_frontier"] = False

            frontier = _nearest_frontier(wq)
            if frontier is None:
                # No frontier — surrounded on all sides. Goal complete by
                # default if charted_chunks is non-zero, else stuck.
                if wq.charted_chunks > start:
                    frame.completed = True
                else:
                    log.warning("Explore: no frontier found — STUCK")
                    frame.failed = True
                return CoordinatorStatus.PROGRESSING, []

            self._push_task(
                task_type   = "explore_region",
                description = (
                    f"Explore toward frontier at {frontier}"
                ),
                agent_hint  = "exploration",
                tick        = tick,
                frontier_position = frontier,
                success_condition = (
                    f"charted_chunks >= {start + target}"
                ),
            )
            return CoordinatorStatus.PROGRESSING, []

        # Check for needs_frontier signal from exploration agent.
        obs = self._bb.read(category=EntryCategory.OBSERVATION)
        if any(e.data.get("type") == "exploration_needs_frontier" for e in obs):
            frame.context["needs_frontier"] = True
            frame.step = 0

        return CoordinatorStatus.WAITING, []

    # ── Clear Region ──────────────────────────────────────────────────────

    def _handle_clear_region(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        Remove natural obstacles from a bounding box.

        params: {bbox, clear_mode}   (clear_mode: "clear_natural" | "clear_all")

        Decision tree
        -------------
        Step 0 — Check for undestroyable blockers:
          Scan natural_objects in bbox for objects where can_destroy=False.
          If any present → fail immediately (caller should use prep_region
          which has a softer handling, or wait for cliff-explosives tech).
        Step 1 — Push clear Task.
        Step 2 — Verify region empty (no natural_objects in bbox) → complete.
        """
        bbox       = frame.params.get("bbox")
        clear_mode = frame.params.get("clear_mode", "clear_natural")

        if bbox is None:
            log.warning("Clear region goal missing 'bbox' param — STUCK")
            frame.failed = True
            return CoordinatorStatus.STUCK, []

        if frame.step == 0:
            # Check for undestroyable objects.
            blocked = _undestroyable_in_bbox(bbox, wq, self._kb)
            if blocked:
                # Known gap: cliffs (and any mod entity with minable=False)
                # make a clear_region goal fail immediately with no recovery
                # path. This is the correct conservative behaviour right now,
                # but it means any bbox that contains a cliff is permanently
                # unclearable until the following capabilities are added:
                #
                #   1. Technology change detection.
                #      EntityRecord.minable can change when the player
                #      researches a technology (e.g. "cliff-explosives" makes
                #      cliffs minable). Currently the KB learns minable=False
                #      at startup and never re-queries. The examination layer
                #      (Phase 10) or a dedicated on_research_finished hook
                #      should invalidate affected EntityRecords and re-query
                #      fa.get_entity_prototype when research completes.
                #
                #   2. UseItemOnEntity bridge action.
                #      Even once the technology is researched and minable
                #      becomes True, destroying a cliff requires placing a
                #      cliff explosive via a UseItemOnEntity action, not the
                #      plain MineEntity that DestroySkill currently issues.
                #      DestroySkill needs a trigger_item parameter and a
                #      corresponding new bridge action (Phase 7).
                #
                #   3. Coordinator retry after technology unlock.
                #      Once (1) and (2) are in place, the coordinator should
                #      not permanently fail a goal because cliffs are present.
                #      Instead it could: push a GOAL_RESEARCH sub-goal to
                #      unlock cliff-explosives, push a GOAL_ACQUIRE sub-goal
                #      to obtain the explosives, then retry the clear. This
                #      would make cliff clearing fully automatic. For now,
                #      STUCK is the only honest response.
                #
                # DO NOT special-case "cliff" by name here — use the minable
                # flag from the KB, which is already what _undestroyable_in_bbox
                # does. The fix is in the KB update loop and bridge action, not
                # in naming specific entity types in coordinator logic.
                log.warning(
                    "Clear region: %d undestroyable object(s) in bbox — FAIL. "
                    "(Likely cliffs requiring explosives. See cliff gap comment "
                    "in _handle_clear_region for the full fix path.)",
                    len(blocked),
                )
                frame.failed = True
                return CoordinatorStatus.STUCK, []
            frame.step = 1

        if frame.step == 1:
            self._push_task(
                task_type   = "clear_region",
                description = f"Clear {clear_mode} in {bbox}",
                agent_hint  = "mining",
                tick        = tick,
                bbox        = bbox,
                clear_mode  = clear_mode,
                success_condition = _bbox_empty_condition(bbox),
            )
            return CoordinatorStatus.PROGRESSING, []

        # Step 2 — verify.
        if _bbox_is_clear(bbox, wq):
            frame.completed = True
        else:
            log.info("Clear region: obstacles remain — re-issuing clear task")
            frame.step = 1
        return CoordinatorStatus.PROGRESSING, []

    # ── Prepare Region (Soft Clear) ───────────────────────────────────────

    def _handle_prep_region(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        Safely clear a region, accounting for factory infrastructure.

        params: {bbox}

        Decision tree
        -------------
        Step 0 — Factory intersection check:
          If bbox intersects major factory infrastructure (assemblers, furnaces,
          power poles, etc.) → FAIL immediately. The coordinator should never
          demolish the existing factory without explicit intent.

        Step 1 — Undestroyable obstacle check:
          If bbox contains objects where can_destroy=False (cliffs without
          explosives) → FAIL. The construction goal should choose a different
          region, or wait for cliff-explosives technology.

        Step 2 — Logistics rerouting (STUB):
          If bbox intersects belts or inserters → STUB: spin off a logistics
          rerouting sub-goal to move the belts out of the region.
          NOT YET IMPLEMENTED (Phase 9 — requires spatial-logistics agent).

        Step 3 — Clear natural objects → GOAL_CLEAR_REGION sub-goal.
        """
        bbox = frame.params.get("bbox")

        if bbox is None:
            log.warning("Prep region goal missing 'bbox' param — STUCK")
            frame.failed = True
            return CoordinatorStatus.STUCK, []

        if frame.step == 0:
            if _intersects_major_factory(bbox, wq):
                log.warning(
                    "Prep region: bbox intersects major factory infrastructure "
                    "— refusing to clear. Choose a different region."
                )
                frame.failed = True
                return CoordinatorStatus.STUCK, []
            frame.step = 1

        if frame.step == 1:
            if _undestroyable_in_bbox(bbox, wq, self._kb):
                log.warning(
                    "Prep region: undestroyable obstacles in bbox "
                    "(cliffs without explosives). "
                    "Waiting for cliff-explosives tech or choose a new region."
                )
                frame.failed = True
                return CoordinatorStatus.STUCK, []
            frame.step = 2

        if frame.step == 2:
            # STUB: logistics rerouting.
            # When implemented (Phase 9): check if any belt/inserter intersects
            # bbox using wq.logistics.belts and wq.state.entities, then push
            # GOAL_LOGISTICS with a "reroute_around_bbox" instruction.
            if _intersects_logistics(bbox, wq):
                log.warning(
                    "Prep region: bbox intersects belts/inserters but logistics "
                    "rerouting is not yet implemented (Phase 9 spatial-logistics "
                    "agent). Proceeding with clear anyway — belts will be "
                    "destroyed and will need manual reconnection."
                )
                # Intentional fall-through: warn and continue rather than
                # blocking the whole build pipeline. Remove this when Phase 9
                # logistics rerouting is implemented.
            frame.step = 3

        if frame.step == 3:
            self._push_goal(
                GOAL_CLEAR_REGION,
                {"bbox": bbox, "clear_mode": "clear_all"},
            )
            return CoordinatorStatus.PROGRESSING, []

        # Step 4: sub-goal completed → done.
        frame.completed = True
        return CoordinatorStatus.PROGRESSING, []

    # ── Construction ──────────────────────────────────────────────────────

    def _handle_construction(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        Build infrastructure in a region.

        params: {bbox, blueprint_hint (optional), resource_buffer_multiplier}

        Decision tree
        -------------
        Step 0 — Resource check:
          Estimate building requirements from blueprint_hint (or defer to the
          build agent's own assessment). If resources are insufficient by a
          healthy buffer → GOAL_ACQUIRE sub-goal for each missing resource.

        Step 1 — Prepare region:
          → GOAL_PREP_REGION sub-goal.

        Step 2 — Build (STUB):
          Push a build task to the construction agent.
          NOT YET IMPLEMENTED — the Phase 7 RL construction agent is needed.

        STUB NOTE: Steps 0 and 2 are stubs. Step 1 (prep region) is
        implemented and will run correctly.
        """
        bbox = frame.params["bbox"]

        if frame.step == 0:
            # STUB: resource requirement estimation.
            # When implemented (Phase 7): query KB for likely building
            # materials given blueprint_hint, check inventory with buffer,
            # push GOAL_ACQUIRE sub-goals for any shortfall.
            log.info(
                "Construction: resource check not yet implemented (Phase 7) "
                "— skipping to region prep."
            )
            frame.step = 1

        if frame.step == 1:
            self._push_goal(GOAL_PREP_REGION, {"bbox": bbox})
            return CoordinatorStatus.PROGRESSING, []

        if frame.step == 2:
            # STUB: build task.
            # When implemented (Phase 7): push a "build" task to the
            # construction agent with the blueprint and bbox.
            log.warning(
                "Construction: build task not yet implemented (Phase 7 "
                "RL construction agent) — marking complete as a no-op."
            )
            frame.completed = True
            return CoordinatorStatus.WAITING, []

        frame.completed = True
        return CoordinatorStatus.PROGRESSING, []

    # ── Production ────────────────────────────────────────────────────────

    def _handle_production(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        Establish production of an item at a given rate.

        params: {item, rate_per_min, output_storage (optional)}

        Decision tree
        -------------
        Step 0 — Delineate build region:
          Find a region not intersecting existing factory.
          STUB: Region delineation algorithm not yet implemented (Phase 8).
          Requires: factory self-model graph for obstacle avoidance.

        Step 1 — Construction sub-goal:
          → GOAL_CONSTRUCTION for the delineated region.

        Step 2 — Connect inputs (recipe recursion):
          For each ingredient of item:
            If produced elsewhere in factory → GOAL_LOGISTICS (connect).
            Else → GOAL_PRODUCTION (recurse).

        Step 3 — Handle byproducts:
          For each output that isn't the primary item:
            → GOAL_BYPRODUCT sub-goal.

        Step 4 — Output storage (optional):
          If output_storage param is True → build a storage container.
          STUB: storage task not yet implemented (Phase 7).

        STUB NOTE: All steps are stubs pending Phase 8 and the factory
        self-model (Phase 10 examination layer).
        """
        item         = frame.params.get("item", "")
        rate_per_min = frame.params.get("rate_per_min", 1.0)

        log.warning(
            "GOAL_PRODUCTION (%s at %.1f/min): not yet implemented "
            "(Phase 8 — requires RL production agent and factory self-model). "
            "Returning STUCK.",
            item, rate_per_min,
        )
        frame.failed = True
        return CoordinatorStatus.STUCK, []

    # ── Logistics ─────────────────────────────────────────────────────────

    def _handle_logistics(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        Connect factory nodes with belts and inserters.

        params: {connections: list[{from_node, to_node, item}]}

        Decision tree
        -------------
        For each connection in params["connections"]:
          Step 0 — Push build-line task for this connection.
          Step 1 — Verify connection established (self-model check).
          Advance to next connection.

        STUB NOTE: Build-line task not yet implemented (Phase 9
        spatial-logistics agent). Requires spatial path planning between
        factory nodes.
        """
        log.warning(
            "GOAL_LOGISTICS: not yet implemented "
            "(Phase 9 — requires spatial-logistics agent). "
            "Returning STUCK."
        )
        frame.failed = True
        return CoordinatorStatus.STUCK, []

    # ── Byproduct ─────────────────────────────────────────────────────────

    def _handle_byproduct(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        Route or consume all output from a production node to prevent backlog.

        params: {source_node_id, item}

        Decision tree
        -------------
        Step 0 — Delineate build region (as in GOAL_PRODUCTION).
          STUB: region delineation not yet implemented.

        Step 1 — Construction sub-goal for consumer infrastructure.

        Step 2 — Connect source output to consumer input → GOAL_LOGISTICS.

        Step 3 — Handle further byproducts of the consumer → GOAL_BYPRODUCT
          (every output of the consumer is itself a byproduct).

        STUB NOTE: All steps are stubs pending Phase 8.
        """
        item = frame.params.get("item", "unknown")
        log.warning(
            "GOAL_BYPRODUCT (%s): not yet implemented "
            "(Phase 8). Returning STUCK.",
            item,
        )
        frame.failed = True
        return CoordinatorStatus.STUCK, []

    # ── Research ──────────────────────────────────────────────────────────

    def _handle_research(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        Unlock a technology.

        params: {tech}

        Decision tree
        -------------
        Step 0 — Already unlocked check:
          If tech_unlocked(tech) → complete immediately.

        Step 1 — Science production check:
          If not producing sufficient science packs of the required types
          at a sufficient rate → GOAL_PRODUCTION sub-goal for each pack type.
          STUB: science rate check requires self-model (Phase 10).

        Step 2 — Lab check:
          If no labs placed, or insufficient labs for the science rate
          → GOAL_CONSTRUCTION sub-goal for a lab region.
          STUB: lab count assessment requires self-model (Phase 10).

        Step 3 — Science routing check:
          If science packs not connected to labs → GOAL_LOGISTICS sub-goal.
          STUB: Phase 9.

        Step 4 — Wait / fill queue:
          If nothing else to do: push a SetResearchQueue task to ensure
          the tech is queued, then return WAITING.

        STUB NOTE: Steps 1–3 require Phase 9/10. Step 4 (queue the research)
        is implementable now.
        """
        tech = frame.params.get("tech", "")

        if not tech:
            log.warning("Research goal missing 'tech' param — STUCK")
            frame.failed = True
            return CoordinatorStatus.STUCK, []

        if frame.step == 0:
            if wq.tech_unlocked(tech):
                log.info("Research goal already satisfied: %s", tech)
                frame.completed = True
                return CoordinatorStatus.PROGRESSING, []
            frame.step = 1

        if frame.step == 1:
            # STUB: science production check (Phase 10).
            # When implemented: read wq.research.science_per_minute for each
            # required pack type, compare to estimated lab consumption rate,
            # push GOAL_PRODUCTION for any shortfall.
            log.info(
                "Research %s: science production check not yet implemented "
                "(Phase 10) — assuming sufficient science.",
                tech,
            )
            frame.step = 2

        if frame.step == 2:
            # STUB: lab sufficiency check (Phase 10).
            # When implemented: count labs via self-model, compare to required
            # throughput, push GOAL_CONSTRUCTION for more labs if needed.
            log.info(
                "Research %s: lab check not yet implemented (Phase 10).",
                tech,
            )
            frame.step = 3

        if frame.step == 3:
            # STUB: science routing check (Phase 9).
            log.info(
                "Research %s: logistics check not yet implemented (Phase 9).",
                tech,
            )
            frame.step = 4

        if frame.step == 4:
            # Queue the research. SetResearchQueue is an existing bridge action.
            # This is the one fully-implemented step.
            self._push_task(
                task_type   = "set_research_queue",
                description = f"Queue technology: {tech}",
                agent_hint  = "crafting",  # placeholder — needs a research agent
                tick        = tick,
                tech        = tech,
                success_condition = f"tech_unlocked('{tech}')",
            )
            return CoordinatorStatus.PROGRESSING, []

        # Step 5: verify.
        if wq.tech_unlocked(tech):
            frame.completed = True
        else:
            frame.step = 4   # re-queue
        return CoordinatorStatus.PROGRESSING, []

    # ── No-op ─────────────────────────────────────────────────────────────

    def _handle_noop(
        self,
        frame: GoalFrame,
        wq: "WorldQuery",
        tick: int,
    ) -> tuple[CoordinatorStatus, list]:
        """
        Idle — ask the LLM for the next goal.

        STUB: LLM integration implemented in Phase 11.
        Until then, returns WAITING indefinitely.
        """
        log.info("No-op goal: waiting for LLM to provide next goal (Phase 11).")
        return CoordinatorStatus.WAITING, []

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def _push_task(
        self,
        task_type: str,
        description: str,
        agent_hint: str,
        tick: int,
        success_condition: str = "",
        **params,
    ) -> None:
        """
        Create a task and set it as the active task.

        Uses Task from the project's task module. task_type is stored
        as a dynamic attribute (Task has no task_type field natively).
        All extra keyword arguments are also set as dynamic attributes so
        agents can read them directly.
        """
        task = Task(
            description       = description,
            success_condition = success_condition,
            failure_condition = f"elapsed_ticks > {_TASK_TIMEOUT_TICKS}",
            parent_goal_id    = "coordinator",
            created_at        = tick,
            derived_locally   = True,
        )
        task.task_type  = task_type
        task.agent_hint = agent_hint
        for k, v in params.items():
            setattr(task, k, v)

        # Select and activate the owning agent.
        agent = self._registry.agent_by_id(agent_hint)
        if agent is None:
            log.warning(
                "No agent registered for hint %r — task %s cannot be activated",
                agent_hint, task.id[:8],
            )
            return

        self._active_task  = task
        self._active_agent = agent
        agent.activate(task, self._bb, None, self._kb)  # wq not available here

        self._bb.write(
            category   = EntryCategory.INTENTION,
            scope      = EntryScope.TASK,
            owner_agent = "coordinator",
            created_at = tick,
            data       = {
                "type":        "task_activated",
                "task_type":   task_type,
                "description": description,
                "agent":       agent_hint,
            },
        )
        log.debug(
            "Task activated: type=%s agent=%s desc=%s",
            task_type, agent_hint, description,
        )

    def _push_goal(self, goal_type: str, params: dict) -> None:
        """Push a sub-goal onto the goal stack."""
        self._goal_stack.append(GoalFrame(goal_type=goal_type, params=params))
        log.debug("Sub-goal pushed: %s %s", goal_type, params)

    # ------------------------------------------------------------------
    # Task evaluation
    # ------------------------------------------------------------------

    def _tick_task(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> tuple[TaskOutcome, list]:
        """Tick the active task's agent and evaluate completion."""
        task  = self._active_task
        agent = self._active_agent

        if task is None or agent is None:
            return TaskOutcome.SUCCEEDED, []

        # Evaluate success condition.
        if task.success_condition and self._eval(task.success_condition, wq, tick):
            log.info("Task complete: %s", task.description)
            return TaskOutcome.SUCCEEDED, []

        # Evaluate failure condition.
        if task.failure_condition and self._eval(task.failure_condition, wq, tick):
            log.warning("Task failed: %s", task.description)
            return TaskOutcome.FAILED, []

        # Check agent skill status via observe().
        obs = agent.observe(task, self._bb, wq, self._kb)
        skill_status = obs.get("skill_status") or obs.get("navigate_status")
        if skill_status == SkillStatus.STUCK.name:
            log.warning("Agent STUCK on task: %s", task.description)
            return TaskOutcome.STUCK, []

        # Tick the agent.
        actions = agent.tick(task, self._bb, wq, ww, tick, self._kb)
        return TaskOutcome.RUNNING, actions

    def _eval(self, condition: str, wq: "WorldQuery", tick: int) -> bool:
        """Evaluate a condition string against the current WorldQuery."""
        if not condition:
            return False
        # Fast-path for literal booleans — avoids needing a full namespace.
        # Used in tests and for simple coordinator-derived conditions.
        if condition.strip() == "True":
            return True
        if condition.strip() == "False":
            return False
        try:
            from planning import build_core_namespace
            ns = build_core_namespace(wq, tick, 0, None)
            return bool(eval(condition, {"__builtins__": {}}, ns))  # noqa: S307
        except Exception as exc:
            log.debug("Condition eval failed (%r): %s", condition, exc)
            return False


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions — no coordinator state)
# ---------------------------------------------------------------------------

def _dist(a: Position, b: Position) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _item_in_nearby_chest(item: str, wq: "WorldQuery") -> bool:
    """
    True if any chest in the scan radius contains the given item.

    STUB: Chest inventory scanning not yet implemented. Always returns False
    until InteractSkill (Phase 7) is available.
    """
    # When implemented: iterate wq.state.entities, filter by prototype_type
    # == "container" or category == "STORAGE", check entity.inventory for item.
    return False


def _missing_ingredients(
    item: str,
    count: int,
    wq: "WorldQuery",
    kb,
) -> dict[str, int]:
    """
    Return {ingredient: shortfall} for crafting count of item.

    Uses KB recipe data. Returns {} if KB is unavailable or recipe unknown.
    """
    if kb is None:
        return {}

    current = {
        slot.item: slot.count
        for slot in wq.state.player.inventory.slots
    }
    recipes = kb.recipes_for_product(item)
    if not recipes:
        return {}

    handcraft = next(
        (r for r in recipes if not r.is_placeholder and r.category == "crafting"),
        None,
    )
    if handcraft is None:
        return {}

    missing = {}
    batches = math.ceil(count / max(handcraft.output_count, 1))
    for ing in handcraft.ingredients:
        needed  = ing.count * batches
        have    = current.get(ing.item, 0)
        if have < needed:
            missing[ing.item] = needed - have
    return missing


def _nearest_frontier(wq: "WorldQuery") -> Optional[Position]:
    """
    Return the tile-space centre of the nearest uncharted chunk, or None.

    Reads wq.nearby_uncharted_chunks if available, falling back to a
    simple outward search pattern from the player's position.
    """
    nearby = getattr(wq, "nearby_uncharted_chunks", [])
    if nearby:
        player = wq.player_position()
        best   = min(
            nearby,
            key=lambda c: math.hypot(
                c.cx * 32 + 16 - player.x,
                c.cy * 32 + 16 - player.y,
            ),
        )
        return Position(x=best.cx * 32 + 16.0, y=best.cy * 32 + 16.0)

    # Fallback: push outward from current position in a simple spiral.
    pos    = wq.player_position()
    radius = max(64.0, math.sqrt(wq.charted_chunks) * 32.0)
    return Position(x=pos.x + radius, y=pos.y)


def _undestroyable_in_bbox(bbox, wq: "WorldQuery", kb) -> list:
    """
    Return natural objects in bbox that cannot be destroyed with MineEntity.

    Uses can_destroy() from execution.predicates, which reads
    EntityRecord.minable from the KnowledgeBase — no entity names hardcoded.
    """
    from execution.predicates import can_destroy
    result = []
    for obj in getattr(wq, "natural_objects", []):
        pos = obj.position
        if (bbox.x_min <= pos.x <= bbox.x_max
                and bbox.y_min <= pos.y <= bbox.y_max):
            if not can_destroy(obj, kb):
                result.append(obj)
    return result


def _bbox_is_clear(bbox, wq: "WorldQuery") -> bool:
    """True if no natural objects remain in the bbox."""
    for obj in getattr(wq, "natural_objects", []):
        pos = obj.position
        if (bbox.x_min <= pos.x <= bbox.x_max
                and bbox.y_min <= pos.y <= bbox.y_max):
            return False
    return True


def _bbox_empty_condition(bbox) -> str:
    """Build a success condition string for a clear task."""
    return (
        f"staleness('entities') is not None and "
        f"staleness('entities') < 300 and "
        f"not any("
        f"  {bbox.x_min} <= e.position.x <= {bbox.x_max} and "
        f"  {bbox.y_min} <= e.position.y <= {bbox.y_max} "
        f"  for e in state.entities"
        f")"
    )


def _intersects_major_factory(bbox, wq: "WorldQuery") -> bool:
    """
    True if the bbox overlaps any major factory entity (assembler, furnace,
    power pole, lab, mining drill, etc.).

    Uses prototype_type to categorise entities — no name hardcoding.
    Major types: assembling-machine, furnace, electric-pole, lab,
    mining-drill, beacon, roboport.
    """
    MAJOR_TYPES = {
        "assembling-machine", "furnace", "electric-pole", "lab",
        "mining-drill", "beacon", "roboport",
    }
    for entity in wq.state.entities:
        pos = entity.position
        if (bbox.x_min <= pos.x <= bbox.x_max
                and bbox.y_min <= pos.y <= bbox.y_max):
            if entity.prototype_type in MAJOR_TYPES:
                return True
    return False


def _intersects_logistics(bbox, wq: "WorldQuery") -> bool:
    """
    True if the bbox overlaps any belt segment or inserter.

    STUB: Belt positions are not currently tracked with precise tile
    coordinates in WorldState — logistics.belts gives flow data, not
    positions. Returns False until the spatial-logistics model (Phase 9)
    provides position data for belts and inserters.
    """
    # When implemented (Phase 9): check belt segment positions and
    # inserter positions against bbox.
    return False