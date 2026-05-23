"""
agent/network/coordinator.py

CoordinatorProtocol and rule-based coordinator implementation for Phase 6.

This module provides:
  - CoordinatorProtocol: the interface (unchanged from Phase 5)
  - StubCoordinator: the Phase 5 no-op stub (retained for reference)
  - RuleBasedCoordinator: the Phase 6 implementation

RuleBasedCoordinator
--------------------
Manages the subtask ledger and blackboard lifecycle. Routes goals to registered
agents. Derives subtask trees for "collection" and "exploration" goal types.
Returns STUCK immediately for all other goal types.

Goal type vocabulary (stable across phases — appear in behavioral memory):
    GOAL_TYPE_COLLECTION   = "collection"
    GOAL_TYPE_PRODUCTION   = "production"
    GOAL_TYPE_CONSTRUCTION = "construction"
    GOAL_TYPE_EXPLORATION  = "exploration"
    GOAL_TYPE_RESEARCH     = "research"

Subtask derivation for "collection"
------------------------------------
1. Find the nearest known resource patch of the required type in resource_map.
2. Push a mining subtask (success: inventory(item) >= N).
3. Push an approach subtask as a prerequisite (success: is_at(patch_position)).
   Because of call-stack semantics, push mining first then movement — movement
   executes first and mining activates after movement completes.

Subtask derivation for "exploration"
--------------------------------------
1. Push a movement subtask that walks the player in a simple search pattern
   until charted_chunks reaches the success threshold.

Ledger lifecycle discipline
----------------------------
The coordinator is the ONLY component that calls lifecycle methods on subtasks.
Correct sequence for every subtask pop:
  1. Call subtask.complete(tick), subtask.fail(tick), or subtask.escalate(tick)
  2. Then call ledger.pop()
pop() raises RuntimeError if the top is not terminal.

StuckContext construction order
--------------------------------
Build the context BEFORE escalating any subtask — failure_chain() reads the
live stack, which is mutated by the escalation/pop sequence.

Rules
-----
- No LLM calls. No RCON. No game-playing behavior in this module.
- The coordinator does not call agents directly during subtask evaluation —
  agents write to the blackboard and the coordinator reads from it.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Optional, TYPE_CHECKING

from agent.blackboard import Blackboard, EntryCategory, EntryScope
from agent.execution_protocol import (
    ExecutionResult,
    ExecutionStatus,
    StuckContext,
)
from bridge.actions import StopMining
from agent.subtask import Subtask, SubtaskLedger, SubtaskStatus
from planning.goal import Goal
from world.state import Position

if TYPE_CHECKING:
    from agent.network.registry import AgentRegistry
    from agent.self_model import SelfModelProtocol
    from world.knowledge import KnowledgeBase
    from world.query import WorldQuery
    from world.writer import WorldWriter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Goal type constants — stable identifiers used in behavioral memory records.
# Define here; import from here wherever these strings are needed.
# ---------------------------------------------------------------------------

GOAL_TYPE_COLLECTION   = "collection"
GOAL_TYPE_PRODUCTION   = "production"
GOAL_TYPE_CONSTRUCTION = "construction"
GOAL_TYPE_EXPLORATION  = "exploration"
GOAL_TYPE_RESEARCH     = "research"
GOAL_TYPE_CLEAR        = "clear_region"

# Goal types the Phase 6 coordinator can derive subtask trees for.
_DERIVABLE_TYPES = {
    GOAL_TYPE_COLLECTION,
    GOAL_TYPE_EXPLORATION,
    GOAL_TYPE_CLEAR,
}

# Subtask failure timeout: if a subtask's explicit failure_condition has not
# fired but the subtask has been active for this many ticks, escalate anyway.
#
# PLACEHOLDER — this is a blunt instrument. The right escalation signal is
# lack of progress (execution_layer.progress() stalling), not elapsed time.
# Time-based escalation fires too early during training (learned agents
# legitimately flounder while improving) and may fire too late in a genuine
# deadlock if the constant is set generously. Revisit before Phase 8 when
# the first learned agent runs and sets a baseline for realistic subtask
# durations. The baked-in absolute tick deadline in derived subtask
# failure_conditions (tick > start + timeout) is also problematic — a subtask
# derived at tick 100 and one derived at tick 50,000 have different effective
# timeouts. Both issues should be addressed together at Phase 8.
_SUBTASK_TIMEOUT_TICKS = 18_000   # 5 minutes at 60 tps — adjust at Phase 8
# Shorter timeout for pure navigation subtasks: if the player stalls on an
# obstacle, we want to re-derive a new waypoint within ~30 seconds rather
# than waiting the full 5-minute general timeout.
_NAV_SUBTASK_TIMEOUT_TICKS = 1_800   # 30 seconds at 60 tps


# ---------------------------------------------------------------------------
# CoordinatorProtocol
# ---------------------------------------------------------------------------

class CoordinatorProtocol:
    """
    Interface for the execution network's internal coordinator.

    The ExecutionLayerProtocol delegates to this coordinator. External callers
    never interact with the coordinator directly — they go through
    ExecutionLayerProtocol.
    """

    def reset(
        self,
        goal: Goal,
        wq: "WorldQuery",
        seed_subtasks: Optional[list] = None,
    ) -> None:
        raise NotImplementedError

    def tick(
        self,
        goal: Goal,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> ExecutionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# StubCoordinator (Phase 5 — retained for reference)
# ---------------------------------------------------------------------------

class StubCoordinator(CoordinatorProtocol):
    """
    Minimal stub coordinator for Phase 5.

    Retained for reference. Replaced by RuleBasedCoordinator in Phase 6.
    """

    def reset(
        self,
        goal: Goal,
        wq: "WorldQuery",
        seed_subtasks: Optional[list] = None,
    ) -> None:
        pass

    def tick(
        self,
        goal: Goal,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> ExecutionResult:
        return ExecutionResult(actions=[], status=ExecutionStatus.WAITING)


# ---------------------------------------------------------------------------
# RuleBasedCoordinator
# ---------------------------------------------------------------------------

class RuleBasedCoordinator(CoordinatorProtocol):
    """
    Rule-based coordinator for Phase 6.

    Manages the SubtaskLedger, Blackboard, and AgentRegistry. Derives
    subtask trees for "collection" and "exploration" goals. Returns STUCK
    for all other goal types.

    Agent selection
    ---------------
    At any given time exactly one agent owns the active subtask. The
    coordinator selects the owning agent when a subtask is pushed onto the
    ledger and records that selection in _active_agent. Only that agent is
    ticked until the subtask resolves, at which point the coordinator selects
    the owner for the next subtask. This replaces the earlier poll-all approach
    which ticked every agent registered for the goal type simultaneously —
    incorrect for a system where subtasks have clear single owners.

    Agent selection logic (Phase 6): all registered agents for the goal type
    are candidates; the first one returned by the registry is selected. In
    Phase 8+ this will be driven by subtask type tags.

    Self-model
    ----------
    The self-model is injected so the coordinator can query existing
    infrastructure before deriving subtasks (e.g. check whether a production
    line already exists before deciding to build one). In Phase 6 the model
    is empty at run start and is not yet queried during derivation, but the
    injection point is established here so later phases do not need to change
    the constructor signature.

    Parameters
    ----------
    registry    : AgentRegistry — maps goal types to agents.
    blackboard  : Blackboard — shared working memory (cleared by reset()).
    ledger      : SubtaskLedger — tracks the live subtask stack and history.
    self_model  : SelfModelProtocol — the agent's model of built infrastructure.
    """

    def __init__(
        self,
        registry: "AgentRegistry",
        blackboard: Blackboard,
        ledger: SubtaskLedger,
        self_model: "SelfModelProtocol",
        kb: "KnowledgeBase",
    ) -> None:
        self._registry = registry
        self._bb = blackboard
        self._ledger = ledger
        self._sm = self_model
        self._kb = kb
        self._current_goal: Optional[Goal] = None
        self._active_agent = None   # the single agent owning the current subtask
        # Tick at which the current active subtask was activated (for timeout).
        # PLACEHOLDER — see _SUBTASK_TIMEOUT_TICKS comment above.
        self._subtask_activated_at: int = 0
        self._goal_start_tick: int = 0
        self._goal_start_snapshot: Optional["WorldQuery"] = None
        # Set to True when _derive_subtasks fails so we don't re-attempt
        # derivation on every subsequent tick (which would spam STUCK events).
        # Cleared by reset() when a new goal starts.
        self._derivation_failed: bool = False
        # Emit StopMining on the first tick after reset() so any in-progress
        # Lua mining is halted before navigation or other subtasks begin.
        self._pending_stop_mining: bool = False
        # Counts how many times exploration derivation has been attempted
        # (including after escalation). Used to rotate waypoint direction so
        # repeated stalls try different directions rather than the same one.
        self._exploration_attempt: int = 0

    # ------------------------------------------------------------------
    # CoordinatorProtocol
    # ------------------------------------------------------------------

    def reset(
        self,
        goal: Goal,
        wq: "WorldQuery",
        seed_subtasks: Optional[list[Subtask]] = None,
    ) -> None:
        """
        Prepare for a new goal.

        Clears the blackboard, ledger, and active agent. Agents are NOT
        activated here — activate() is called on the selected agent only when
        a subtask becomes active (via _activate_agent_for_subtask). This
        enforces the boundary: agents interact with Subtasks, not Goals.

        If seed_subtasks are provided (post-LLM-escalation), pushes them onto
        the cleared ledger in order and activates the agent for the first one.
        """
        self._bb.clear_all()
        self._ledger.clear()
        self._current_goal = goal
        self._active_agent = None
        self._subtask_activated_at = 0
        self._derivation_failed = False
        self._pending_stop_mining = True
        self._exploration_attempt = 0
        self._goal_start_tick = wq.tick
        self._goal_start_snapshot = {}  # populated by loop via set_start_snapshot()

        if seed_subtasks:
            for subtask in reversed(seed_subtasks):
                self._ledger.push(subtask)
            first = self._ledger.peek()
            if first is not None:
                self._active_agent = self._select_agent(goal)
                if self._active_agent is not None:
                    self._active_agent.activate(first, self._bb, wq, self._kb)
            log.info(
                "Coordinator reset with %d seed subtasks for goal %s",
                len(seed_subtasks),
                goal.id[:8],
            )
        else:
            log.info(
                "Coordinator reset for goal %s (%s)",
                goal.id[:8],
                getattr(goal, "type", "unknown"),
            )

    def tick(
        self,
        goal: Goal,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> ExecutionResult:
        """
        One coordination cycle.

        1. Check if the goal type is derivable. If not, return STUCK immediately.
        2. If no subtask is active, derive the subtask tree.
        3. Evaluate the active subtask's success/failure conditions.
        4. Poll agents for actions.
        5. Return ExecutionResult.
        """
        goal_type = getattr(goal, "type", "")

        # --- Halt any in-progress Lua mining from a previous goal ---
        # reset() sets this flag; we drain it here so StopMining is the very
        # first action dispatched at the start of every goal — before the
        # navigation agent moves the player, while mining might still be active.
        if self._pending_stop_mining:
            self._pending_stop_mining = False
            return ExecutionResult(
                actions=[StopMining()],
                status=ExecutionStatus.PROGRESSING,
            )

        # --- Fast-path: unsupported goal type ---
        if goal_type not in _DERIVABLE_TYPES and not self._ledger:
            return self._stuck_at_goal_level(goal, tick)

        # --- Derive subtasks if ledger is empty ---
        if not self._ledger:
            if self._derivation_failed:
                # Already failed to derive on a previous tick — don't retry
                # every tick, which would spam STUCK events. Return WAITING so
                # the loop continues polling until the failure_condition fires.
                return ExecutionResult(actions=[], status=ExecutionStatus.WAITING)

            result = self._derive_subtasks(goal, wq, tick)
            if result is not None:
                # Derivation failed — mark so we don't retry next tick.
                self._derivation_failed = True
                return result
            # Derivation succeeded — select and activate agent for first subtask.
            first = self._ledger.peek()
            if first is not None:
                self._active_agent = self._select_agent(goal)
                if self._active_agent is not None:
                    self._active_agent.activate(first, self._bb, wq, self._kb)

        # --- Evaluate active subtask ---
        active = self._ledger.peek()
        if active is None:
            # Ledger is empty and we've already derived — goal may be complete.
            return ExecutionResult(actions=[], status=ExecutionStatus.WAITING)

        # Track activation time for timeout.
        if self._subtask_activated_at == 0:
            self._subtask_activated_at = tick

        # Check for navigation stall signal — escalate immediately rather than
        # waiting for the failure_condition timeout. The navigation agent writes
        # this when it detects the player is stopped after the grace period.
        observations = self._bb.read(category=EntryCategory.OBSERVATION, current_tick=tick)
        if any(e.data.get("type") == "navigation_stalled" for e in observations):
            log.warning(
                "Coordinator: navigation stall detected on subtask %s — escalating",
                active.id[:8],
            )
            return self._escalate(goal, tick)

        # Check success condition.
        if self._evaluate_condition(active.success_condition, wq, tick):
            log.info(
                "Subtask %s complete: %s", active.id[:8], active.description
            )
            active.complete(tick)
            self._ledger.pop()
            self._bb.clear_scope(EntryScope.SUBTASK)
            self._subtask_activated_at = 0

            # Select and activate agent for the next subtask.
            next_subtask = self._ledger.peek()
            if next_subtask is not None:
                self._active_agent = self._select_agent(goal)
                if self._active_agent is not None:
                    self._active_agent.activate(next_subtask, self._bb, wq, self._kb)
                self._write_waypoint_for_subtask(next_subtask, wq, tick)
            else:
                self._active_agent = None

            return ExecutionResult(actions=[], status=ExecutionStatus.PROGRESSING)

        # Check timeout.
        if (
            tick - self._subtask_activated_at > _SUBTASK_TIMEOUT_TICKS
            and self._subtask_activated_at > 0
        ):
            log.warning(
                "Subtask %s timed out after %d ticks",
                active.id[:8],
                tick - self._subtask_activated_at,
            )
            return self._escalate(goal, tick)

        # Check failure condition.
        if self._evaluate_condition(active.failure_condition, wq, tick):
            log.warning(
                "Subtask %s failure condition met: %s",
                active.id[:8],
                active.description,
            )
            return self._escalate(goal, tick)

        # --- Tick the active agent ---
        actions = self._tick_active_agent(active, self._bb, wq, ww, tick, self._kb)

        return ExecutionResult(
            actions=actions,
            status=ExecutionStatus.PROGRESSING if actions else ExecutionStatus.WAITING,
        )

    # ------------------------------------------------------------------
    # Subtask derivation
    # ------------------------------------------------------------------

    def _derive_subtasks(
        self,
        goal: Goal,
        wq: "WorldQuery",
        tick: int,
    ) -> Optional[ExecutionResult]:
        """
        Derive a subtask tree for the goal. Returns None on success (subtasks
        pushed onto ledger), or an ExecutionResult(STUCK) on failure.
        """
        goal_type = getattr(goal, "type", "")

        if goal_type == GOAL_TYPE_COLLECTION:
            return self._derive_collection(goal, wq, tick)
        elif goal_type == GOAL_TYPE_EXPLORATION:
            result = self._derive_exploration(goal, wq, tick, self._exploration_attempt)
            self._exploration_attempt += 1
            return result
        elif goal_type == GOAL_TYPE_CLEAR:
            return self._derive_clear(goal, wq, tick)
        else:
            return self._stuck_at_goal_level(goal, tick)

    def _derive_collection(
        self,
        goal: Goal,
        wq: "WorldQuery",
        tick: int,
    ) -> Optional[ExecutionResult]:
        """
        Derive subtasks for a collection goal.

        Parses the resource type and target count from the goal's
        success_condition ("inventory('iron-ore') >= 50") and finds the
        nearest known resource patch. Pushes mining and movement subtasks.

        Returns None if subtasks were derived successfully, STUCK otherwise.
        """
        resource_type, target_count = _parse_collection_condition(
            goal.success_condition
        )
        if resource_type is None:
            log.warning(
                "Cannot parse collection condition: %s", goal.success_condition
            )
            return self._stuck_at_goal_level(goal, tick)

        # Find the nearest resource patch.
        patches = wq.resources_of_type(resource_type)
        if not patches:
            log.warning(
                "No known patches of %s in resource_map", resource_type
            )
            return self._stuck_at_goal_level(goal, tick)

        player_pos = wq.player_position()
        nearest = min(patches, key=lambda p: p.position.distance_to(player_pos))
        patch_pos = nearest.position

        # Two subtasks: approach (navigation agent) then gather (mining agent).
        # Push gather first (suspended), approach second (activates immediately).
        # Call-stack semantics: last pushed = first executed.
        # Use an absolute inventory threshold for the gather subtask rather
        # than goal.success_condition verbatim. This handles new.inventory()
        # goals correctly: the subtask needs to know when to stop mining,
        # expressed as a concrete absolute count.
        current_count = wq.inventory_count(resource_type)
        gather_target = current_count + target_count
        gather_subtask = Subtask(
            description=f"Gather {target_count} {resource_type}",
            success_condition=f"inventory('{resource_type}') >= {gather_target}",
            failure_condition=f"tick > {tick + _SUBTASK_TIMEOUT_TICKS}",
            parent_goal_id=goal.id,
            created_at=tick,
            derived_locally=True,
        )
        gather_subtask.agent_hint = "mining"
        gather_subtask.resource_type = resource_type
        gather_subtask.target_position = patch_pos

        # Success condition: player is within interaction range of the patch.
        # is_at(pos, tolerance) is in the condition eval namespace.
        approach_success = (
            f"is_at(Position({patch_pos.x}, {patch_pos.y}), tolerance=1.5)"
        )
        approach_subtask = Subtask(
            description=(
                f"Approach {resource_type} patch at "
                f"({patch_pos.x:.0f}, {patch_pos.y:.0f})"
            ),
            success_condition=approach_success,
            failure_condition=f"tick > {tick + _SUBTASK_TIMEOUT_TICKS}",
            parent_goal_id=goal.id,
            parent_subtask_id=gather_subtask.id,
            created_at=tick,
            derived_locally=True,
        )
        approach_subtask.agent_hint = "navigation"
        approach_subtask.target_position = patch_pos

        self._ledger.push(gather_subtask)
        self._ledger.push(approach_subtask)

        # Write the navigation waypoint for the approach subtask.
        self._bb.write(
            category=EntryCategory.INTENTION,
            scope=EntryScope.SUBTASK,
            owner_agent="coordinator",
            created_at=tick,
            data={
                "type": "waypoint",
                "waypoint_type": "move",
                "target_position": {"x": patch_pos.x, "y": patch_pos.y},
                "target_entity_id": None,
                "purpose": f"approach_{resource_type}_patch",
            },
        )

        log.info(
            "Derived collection subtasks: approach (%s, %s) then gather %d %s",
            patch_pos.x, patch_pos.y, target_count, resource_type,
        )
        return None  # success

    def _derive_exploration(
        self,
        goal: Goal,
        wq: "WorldQuery",
        tick: int,
        attempt: int = 0,
    ) -> Optional[ExecutionResult]:
        """
        Derive subtasks for an exploration goal.

        Parses the target chunk count from the success_condition
        ("charted_chunks >= 10") and pushes a movement subtask that walks
        the player in a simple outward spiral pattern. The success condition
        is evaluated against charted_chunks directly.

        Returns None on success, STUCK on failure.
        """
        target_chunks = _parse_exploration_condition(goal.success_condition)
        if target_chunks is None:
            log.warning(
                "Cannot parse exploration condition: %s", goal.success_condition
            )
            return self._stuck_at_goal_level(goal, tick)

        # Determine the next unexplored waypoint in a simple grid pattern.
        player_pos = wq.player_position()
        next_pos = _next_exploration_waypoint(player_pos, wq.charted_chunks, target_chunks, attempt)

        # The subtask success condition is arrival at the waypoint — NOT the
        # goal's charted_chunks condition. The goal's chunk condition is already
        # evaluated every tick by RewardEvaluator and will fire as soon as enough
        # chunks are charted, regardless of where the player is. Using it as the
        # subtask condition caused the subtask to only complete when charted_chunks
        # reached the full target, meaning a single waypoint had to do all the
        # work. Arrival-based success lets the coordinator re-derive a new waypoint
        # after each leg, building up chunk coverage incrementally.
        arrival_condition = (
            f"is_at(Position({next_pos.x}, {next_pos.y}), 3.0)"
        )
        exploration_subtask = Subtask(
            description=f"Walk to exploration waypoint ({next_pos.x:.0f}, {next_pos.y:.0f})",
            success_condition=arrival_condition,
            failure_condition=f"tick > {tick + _NAV_SUBTASK_TIMEOUT_TICKS}",
            parent_goal_id=goal.id,
            created_at=tick,
            derived_locally=True,
        )
        exploration_subtask.agent_hint = "navigation"
        exploration_subtask.target_position = next_pos
        self._ledger.push(exploration_subtask)

        self._bb.write(
            category=EntryCategory.INTENTION,
            scope=EntryScope.SUBTASK,
            owner_agent="coordinator",
            created_at=tick,
            data={
                "type": "waypoint",
                "waypoint_type": "move",
                "target_position": {"x": next_pos.x, "y": next_pos.y},
                "target_entity_id": None,
                "purpose": "exploration_waypoint",
            },
        )

        log.info(
            "Derived exploration subtask: chart %d chunks (currently %d)",
            target_chunks,
            wq.charted_chunks,
        )
        return None

    def _derive_clear(
        self,
        goal: Goal,
        wq: "WorldQuery",
        tick: int,
    ) -> Optional[ExecutionResult]:
        """
        Derive a subtask for a clear_region goal.

        The goal's metadata must carry a 'bounding_box' dict:
            {"x_min": f, "y_min": f, "x_max": f, "y_max": f}
        and optionally 'clear_mode': "clear_all" | "clear_natural" (default).

        The success condition is evaluated normally — typically a tick-based
        upper bound. The mining agent clears all targets in the box and then
        sits idle; the success condition fires at the bound or can be written
        as a proximal entity-count check if the caller prefers.
        """
        bbox = getattr(goal, "bounding_box", None)
        if not bbox:
            log.warning("clear_region goal %s has no bounding_box attribute", goal.id[:8])
            return self._stuck_at_goal_level(goal, tick)

        clear_mode = getattr(goal, "clear_mode", "clear_natural")

        clear_subtask = Subtask(
            description=f"Clear {clear_mode} in region ({bbox})",
            success_condition=goal.success_condition,
            failure_condition=f"tick > {tick + _SUBTASK_TIMEOUT_TICKS}",
            parent_goal_id=goal.id,
            created_at=tick,
            derived_locally=True,
        )
        clear_subtask.agent_hint = "mining"
        clear_subtask.bounding_box = bbox
        clear_subtask.clear_mode = clear_mode
        self._ledger.push(clear_subtask)

        # Write the mining_task INTENTION directly.
        self._bb.write(
            category=EntryCategory.INTENTION,
            scope=EntryScope.SUBTASK,
            owner_agent="coordinator",
            created_at=tick,
            data={
                "type": "mining_task",
                "task_type": clear_mode,
                "bounding_box": bbox,
            },
        )

        log.info(
            "Derived clear subtask: %s in %s", clear_mode, bbox
        )
        return None

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    def _stuck_at_goal_level(
        self, goal: Goal, tick: int
    ) -> ExecutionResult:
        """Return STUCK with an empty failure chain (stuck before any subtask)."""
        ctx = StuckContext(
            goal=goal,
            failure_chain=[],
            sibling_history={},
            blackboard_snapshot=self._bb.snapshot(tick),
        )
        return ExecutionResult(
            actions=[],
            status=ExecutionStatus.STUCK,
            stuck_context=ctx,
        )

    def _escalate(self, goal: Goal, tick: int) -> ExecutionResult:
        """
        Build StuckContext from the current ledger state and return STUCK.

        Capture chain and siblings BEFORE escalating — failure_chain() reads
        the live stack, which is mutated by the escalation/pop sequence.
        """
        chain = self._ledger.failure_chain()
        siblings = self._ledger.sibling_history(chain, goal.id)
        snapshot = self._bb.snapshot(tick)

        # Escalate and pop from innermost (top) to outermost.
        for subtask in reversed(chain):
            if subtask.status == SubtaskStatus.ACTIVE:
                subtask.escalate(tick)
            self._ledger.pop()

        self._bb.clear_all()
        self._subtask_activated_at = 0
        self._active_agent = None

        ctx = StuckContext(
            goal=goal,
            failure_chain=chain,
            sibling_history=siblings,
            blackboard_snapshot=snapshot,
        )
        # For exploration goals, a stuck subtask just means "that direction
        # was blocked — try another". Return WAITING so the coordinator
        # re-derives a new waypoint (in a rotated direction) on the next tick
        # rather than propagating STUCK to the loop and failing the goal.
        if getattr(goal, "type", "") == GOAL_TYPE_EXPLORATION:
            log.info(
                "Coordinator: exploration subtask stuck — re-deriving waypoint "
                "(attempt %d)", self._exploration_attempt
            )
            self._derivation_failed = False
            self._active_agent = None
            return ExecutionResult(actions=[], status=ExecutionStatus.WAITING)

        return ExecutionResult(
            actions=[],
            status=ExecutionStatus.STUCK,
            stuck_context=ctx,
        )

    # ------------------------------------------------------------------
    # Agent selection and ticking
    # ------------------------------------------------------------------

    def set_start_snapshot(self, start_wq: "WorldQuery") -> None:
        """
        Called by the loop immediately after reset() to provide a WorldQuery
        snapshot taken at goal activation. Used to populate the `new` delta
        object in subtask condition namespaces.
        """
        self._goal_start_snapshot = start_wq

    def _select_agent(self, goal: Goal):
        """
        Select the agent that will own the current active subtask.

        Routing:
          1. Read agent_hint from the active subtask.
          2. Use registry.agent_by_id(hint) to find the matching agent.
          3. If no hint or no match, fall back to the first registered agent.

        The registry is not queried by goal type — agents are selected purely
        by the subtask's agent_hint matching an agent's AGENT_ID class attribute.
        """
        active = self._ledger.peek()
        if active is not None:
            hint = getattr(active, "agent_hint", None)
            if hint:
                agent = self._registry.agent_by_id(hint)
                if agent is not None:
                    log.debug(
                        "Agent %r selected by hint for subtask %s",
                        hint, active.id[:8],
                    )
                    return agent
                log.warning(
                    "agent_hint %r not found in registry; falling back to first",
                    hint,
                )

        agents = self._registry.all_agents()
        if not agents:
            log.warning("No agents registered — cannot select owner")
            return None
        if len(agents) > 1:
            log.debug(
                "No hint on active subtask; selecting first of %d registered agents",
                len(agents),
            )
        return agents[0]

    def _tick_active_agent(
        self,
        subtask: Subtask,
        blackboard: Blackboard,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
        kb: "KnowledgeBase",
    ) -> list:
        """
        Tick the single agent that owns the current active subtask.

        Passes the active Subtask to the agent — agents interact with
        Subtasks, not Goals. Only _active_agent is ticked; all other
        registered agents are dormant until selected for a future subtask.
        """
        if self._active_agent is None:
            return []
        try:
            return self._active_agent.tick(subtask, blackboard, wq, ww, tick, kb)
        except Exception:
            log.exception("Active agent %s raised during tick", self._active_agent)
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate_condition(
        self,
        condition: str,
        wq: "WorldQuery",
        tick: int,
    ) -> bool:
        """
        Evaluate a condition expression string against the current WorldQuery.

        Uses the same eval-based mechanism as RewardEvaluator. Returns False
        on any exception (missing data, stale state, parse error).

        Empty condition strings evaluate to False (no automatic completion).
        """
        if not condition:
            return False
        try:
            ns = _build_condition_namespace(wq, tick, self._goal_start_tick, self._goal_start_snapshot)
            return bool(eval(condition, {"__builtins__": {}}, ns))  # noqa: S307
        except Exception as exc:
            log.debug(
                "Condition eval failed (%r): %s", condition, exc
            )
            return False

    def _write_waypoint_for_subtask(
        self,
        subtask: Subtask,
        wq: "WorldQuery",
        tick: int,
    ) -> None:
        """
        Write a blackboard INTENTION for a newly-activated subtask.

        Navigation subtasks get a "waypoint" INTENTION entry, with the
        target position read directly from subtask.target_position (stored
        at derivation time — no description parsing needed).

        Mining subtasks get a "mining_task" INTENTION entry, with the
        resource type and patch position read from subtask attributes.

        The agent_hint on the subtask determines which type to write.
        """
        hint = getattr(subtask, "agent_hint", None)

        if hint == "navigation":
            pos: Optional[Position] = getattr(subtask, "target_position", None)
            if pos is None:
                log.warning(
                    "Navigation subtask %s has no target_position attribute",
                    subtask.id[:8],
                )
                return
            self._bb.write(
                category=EntryCategory.INTENTION,
                scope=EntryScope.SUBTASK,
                owner_agent="coordinator",
                created_at=tick,
                data={
                    "type": "waypoint",
                    "waypoint_type": "move",
                    "target_position": {"x": pos.x, "y": pos.y},
                    "target_entity_id": None,
                    "purpose": "approach",
                },
            )

        elif hint == "mining":
            # Check if this is a clear subtask (has bounding_box) or a gather.
            bbox = getattr(subtask, "bounding_box", None)
            clear_mode = getattr(subtask, "clear_mode", None)

            if bbox is not None:
                # Clear subtask — re-write the mining_task INTENTION.
                self._bb.write(
                    category=EntryCategory.INTENTION,
                    scope=EntryScope.SUBTASK,
                    owner_agent="coordinator",
                    created_at=tick,
                    data={
                        "type": "mining_task",
                        "task_type": clear_mode or "clear_natural",
                        "bounding_box": bbox,
                    },
                )
                return

            resource_type: str = getattr(subtask, "resource_type", "")
            pos = getattr(subtask, "target_position", None)

            if not resource_type or pos is None:
                # Fall back to re-deriving from the resource map.
                resource_type = self._parse_resource_type_from_description(
                    subtask.description
                )
                if resource_type:
                    patches = wq.resources_of_type(resource_type)
                    if patches:
                        player_pos = wq.player_position()
                        nearest = min(
                            patches,
                            key=lambda p: p.position.distance_to(player_pos),
                        )
                        pos = nearest.position

            if not resource_type or pos is None:
                log.warning(
                    "Cannot write mining_task for subtask %s — missing "
                    "resource_type or target_position",
                    subtask.id[:8],
                )
                return

            self._bb.write(
                category=EntryCategory.INTENTION,
                scope=EntryScope.SUBTASK,
                owner_agent="coordinator",
                created_at=tick,
                data={
                    "type": "mining_task",
                    "task_type": "gather_resource",
                    "resource_type": resource_type,
                    "target_position": {"x": pos.x, "y": pos.y},
                },
            )

    def _parse_resource_type_from_description(self, description: str) -> str:
        """
        Parse the resource type from "Gather N resource-type".
        Returns the last word of the description.
        """
        parts = description.strip().split()
        return parts[-1] if len(parts) >= 3 else ""


# ---------------------------------------------------------------------------
# Condition parsing helpers
# ---------------------------------------------------------------------------

def _parse_collection_condition(condition: str):
    """
    Parse collection goal success conditions into (resource_type, target_count).

    Handles both forms:
        "inventory('iron-ore') >= 50"     -> ("iron-ore", 50)
        "new.inventory('iron-ore') >= 5"  -> ("iron-ore", 5)

    Returns (None, None) if neither form matches.
    """
    import re as _re
    m = _re.search(r"(?:new\.)?inventory\(['\"]([^'\"]+)['\"]\)\s*>=\s*(\d+)", condition)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def _parse_exploration_condition(condition: str):
    """
    Parse "charted_chunks >= 10" → 10.
    Returns None if the condition doesn't match.
    """
    import re
    m = re.search(r"charted_chunks\s*>=\s*(\d+)", condition)
    if m:
        return int(m.group(1))
    return None


def _next_exploration_waypoint(
    player_pos: Position,
    current_chunks: int,
    target_chunks: int,
    attempt: int = 0,
) -> Position:
    """
    Return the next waypoint for outward exploration from the world origin.

    Waypoints are absolute positions (not relative to player) so each leg
    moves into genuinely new territory. Step grows with charted chunk count:
    128 tiles (4 chunks) minimum, one ring wider per sqrt(chunks) increment.
    The `attempt` offset rotates the cardinal direction so repeated stalls
    try a different direction rather than retrying the same blocked path.

    This is a Phase 6 placeholder. Phase 9 replaces with frontier-based
    exploration.
    """
    # Fixed step of 256 tiles (8 chunks) from origin. This is intentionally
    # oversized to reliably clear any pre-generated spawn area on any map.
    # The frontier-based planner in Phase 9 will replace this entirely.
    # attempt offset adds 64 tiles per retry so repeated stalls push further.
    step = 256 + (attempt // 4) * 64

    # Absolute coordinates from origin. Rotate direction by attempt so
    # repeated stalls (e.g. unreachable terrain) try different cardinals.
    # Starting east avoids the crashed ship which spawns north in default maps.
    direction = (current_chunks + attempt) % 4
    if direction == 0:
        return Position(step, 0.0)     # east
    elif direction == 1:
        return Position(0.0, step)     # south
    elif direction == 2:
        return Position(-step, 0.0)    # west
    else:
        return Position(0.0, -step)    # north


def _build_condition_namespace(wq: "WorldQuery", tick: int, start_tick: int = 0, start_wq: "WorldQuery" = None) -> dict:
    """
    Build the eval namespace for subtask condition evaluation.

    Starts from the shared core (planning.condition_namespace.build_core_namespace)
    and adds coordinator-specific positional predicates: is_at, is_reachable,
    Position. These are needed for navigation subtask success conditions but
    are not part of the general goal condition namespace.

    Any entry added to build_core_namespace is automatically available here
    without any changes to this function.
    """
    from planning.condition_namespace import build_core_namespace, safe_builtins
    from agent.preconditions import is_at as _is_at, is_reachable as _is_reachable

    def is_at(pos: Position, tolerance: float = 1.5) -> bool:
        return _is_at(pos, wq, tolerance)

    def is_reachable(entity_id: int) -> bool:
        return _is_reachable(entity_id, wq)

    ns = build_core_namespace(wq, tick, start_tick, start_wq)
    ns["__builtins__"] = safe_builtins()

    # Coordinator-specific extras: positional predicates for navigation
    # subtask success conditions (is_at, is_reachable, Position).
    ns.update({
        "is_at":       is_at,
        "is_reachable": is_reachable,
        "Position":    Position,
    })
    return ns