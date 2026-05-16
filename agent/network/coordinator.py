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
3. Push a movement subtask as a prerequisite (success: waypoint_reached).
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
from agent.subtask import Subtask, SubtaskLedger, SubtaskStatus
from planning.goal import Goal
from world.state import Position

if TYPE_CHECKING:
    from agent.network.registry import AgentRegistry
    from agent.self_model import SelfModelProtocol
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

# Goal types the Phase 6 coordinator can derive subtask trees for.
_DERIVABLE_TYPES = {GOAL_TYPE_COLLECTION, GOAL_TYPE_EXPLORATION}

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
    ) -> None:
        self._registry = registry
        self._bb = blackboard
        self._ledger = ledger
        self._sm = self_model
        self._current_goal: Optional[Goal] = None
        self._active_agent = None   # the single agent owning the current subtask
        # Tick at which the current active subtask was activated (for timeout).
        # PLACEHOLDER — see _SUBTASK_TIMEOUT_TICKS comment above.
        self._subtask_activated_at: int = 0

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

        Clears the blackboard, ledger, and active agent. If seed_subtasks are
        provided (post-LLM-escalation), pushes them onto the cleared ledger in
        order instead of deriving from scratch, and selects the agent for the
        first subtask.
        """
        self._bb.clear_all()
        self._ledger.clear()
        self._current_goal = goal
        self._active_agent = None
        self._subtask_activated_at = 0

        if seed_subtasks:
            for subtask in reversed(seed_subtasks):
                self._ledger.push(subtask)
            self._active_agent = self._select_agent(goal)
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

        # Activate all agents registered for this goal type so they can
        # configure their observation spaces. Only _active_agent will be
        # ticked; the others are dormant until selected.
        goal_type = getattr(goal, "type", "")
        agents = self._registry.agents_for_goal(goal_type)
        for agent in agents:
            agent.activate(goal, self._bb, wq)

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

        # --- Fast-path: unsupported goal type ---
        if goal_type not in _DERIVABLE_TYPES and not self._ledger:
            return self._stuck_at_goal_level(goal, tick)

        # --- Derive subtasks if ledger is empty ---
        if not self._ledger:
            result = self._derive_subtasks(goal, wq, tick)
            if result is not None:
                return result  # STUCK from derivation failure
            # Derivation succeeded — select the agent for the first subtask.
            self._active_agent = self._select_agent(goal)

        # --- Evaluate active subtask ---
        active = self._ledger.peek()
        if active is None:
            # Ledger is empty and we've already derived — goal may be complete.
            return ExecutionResult(actions=[], status=ExecutionStatus.WAITING)

        # Track activation time for timeout.
        if self._subtask_activated_at == 0:
            self._subtask_activated_at = tick

        # Check success condition.
        if self._evaluate_condition(active.success_condition, wq, tick):
            log.info(
                "Subtask %s complete: %s", active.id[:8], active.description
            )
            active.complete(tick)
            self._ledger.pop()
            self._bb.clear_scope(EntryScope.SUBTASK)
            self._subtask_activated_at = 0

            # Write waypoint and select owner for the next subtask.
            next_subtask = self._ledger.peek()
            if next_subtask is not None:
                self._active_agent = self._select_agent(goal)
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

        # Check for waypoint_reached signal from navigation agent.
        if self._check_waypoint_reached(tick):
            log.debug("Coordinator detected waypoint_reached for subtask %s", active.id[:8])
            active.complete(tick)
            self._ledger.pop()
            self._bb.clear_scope(EntryScope.SUBTASK)
            self._subtask_activated_at = 0

            next_subtask = self._ledger.peek()
            if next_subtask is not None:
                self._active_agent = self._select_agent(goal)
                self._write_waypoint_for_subtask(next_subtask, wq, tick)
            else:
                self._active_agent = None

            return ExecutionResult(actions=[], status=ExecutionStatus.PROGRESSING)

        # --- Tick the active agent ---
        actions = self._tick_active_agent(self._bb, wq, ww, tick)

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
            return self._derive_exploration(goal, wq, tick)
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

        # Build subtasks. Push mining first (it will be suspended while
        # movement is active), then push movement (executes first).
        mining_subtask = Subtask(
            description=f"Mine {target_count} {resource_type}",
            success_condition=goal.success_condition,
            failure_condition=f"tick > {tick + _SUBTASK_TIMEOUT_TICKS}",
            parent_goal_id=goal.id,
            created_at=tick,
            derived_locally=True,
        )
        movement_subtask = Subtask(
            description=f"Move to {resource_type} patch at ({patch_pos.x:.0f}, {patch_pos.y:.0f})",
            success_condition="",   # Completed via waypoint_reached signal
            failure_condition=f"tick > {tick + _SUBTASK_TIMEOUT_TICKS}",
            parent_goal_id=goal.id,
            parent_subtask_id=mining_subtask.id,
            created_at=tick,
            derived_locally=True,
        )

        # Push mining first (suspended), movement second (activates immediately).
        self._ledger.push(mining_subtask)
        self._ledger.push(movement_subtask)

        # Write waypoint for the movement subtask.
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
            "Derived collection subtasks: move to (%s, %s) then mine %d %s",
            patch_pos.x, patch_pos.y, target_count, resource_type,
        )
        return None  # success

    def _derive_exploration(
        self,
        goal: Goal,
        wq: "WorldQuery",
        tick: int,
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
        next_pos = _next_exploration_waypoint(player_pos, wq.charted_chunks, target_chunks)

        exploration_subtask = Subtask(
            description=f"Explore until {target_chunks} chunks charted",
            success_condition=goal.success_condition,
            failure_condition=f"tick > {tick + _SUBTASK_TIMEOUT_TICKS * 3}",
            parent_goal_id=goal.id,
            created_at=tick,
            derived_locally=True,
        )
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
        return ExecutionResult(
            actions=[],
            status=ExecutionStatus.STUCK,
            stuck_context=ctx,
        )

    # ------------------------------------------------------------------
    # Agent selection and ticking
    # ------------------------------------------------------------------

    def _select_agent(self, goal: Goal):
        """
        Select the agent that will own the current active subtask.

        Phase 6: returns the first agent registered for the goal type, or
        None if no agents are registered. This is sufficient while there is
        only one agent per goal type. In Phase 8+ this should be driven by
        a subtask type tag so that different subtask types within the same
        goal can route to different agents.
        """
        goal_type = getattr(goal, "type", "")
        agents = self._registry.agents_for_goal(goal_type)
        if not agents:
            log.warning(
                "No agents registered for goal type %r — cannot select owner",
                goal_type,
            )
            return None
        if len(agents) > 1:
            log.debug(
                "%d agents registered for %r; selecting first. "
                "Revisit at Phase 8 when subtask type tags are introduced.",
                len(agents), goal_type,
            )
        return agents[0]

    def _tick_active_agent(
        self,
        blackboard: Blackboard,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list:
        """
        Tick the single agent that owns the current active subtask.

        Only _active_agent is ticked. All other registered agents are dormant
        until selected as the owner of a future subtask. Returns the agent's
        candidate action list, or an empty list if no agent is selected.
        """
        if self._active_agent is None:
            return []
        try:
            return self._active_agent.tick(blackboard, wq, ww, tick)
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
            ns = _build_condition_namespace(wq, tick)
            return bool(eval(condition, {"__builtins__": {}}, ns))  # noqa: S307
        except Exception as exc:
            log.debug(
                "Condition eval failed (%r): %s", condition, exc
            )
            return False

    def _check_waypoint_reached(self, tick: int) -> bool:
        """
        True if the navigation agent has written a waypoint_reached observation
        in the current subtask scope.
        """
        observations = self._bb.read(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.SUBTASK,
            current_tick=tick,
        )
        return any(
            e.data.get("type") == "waypoint_reached"
            for e in observations
        )

    def _write_waypoint_for_subtask(
        self,
        subtask: Subtask,
        wq: "WorldQuery",
        tick: int,
    ) -> None:
        """
        Write a waypoint INTENTION for a newly-activated subtask.

        For mining subtasks, the patch position was recorded during derivation
        and must be re-derived from the resource_map. This is a best-effort
        write; if the patch is no longer visible, the mining subtask will
        time out and escalate.
        """
        # Mining subtasks: write a mine_resource waypoint toward the nearest patch.
        desc = subtask.description.lower()
        if "mine" in desc:
            # Extract resource type from description "Mine N resource-type".
            parts = desc.split()
            if len(parts) >= 3:
                resource_type = parts[-1]  # last word
                patches = wq.resources_of_type(resource_type)
                if patches:
                    player_pos = wq.player_position()
                    nearest = min(
                        patches, key=lambda p: p.position.distance_to(player_pos)
                    )
                    self._bb.write(
                        category=EntryCategory.INTENTION,
                        scope=EntryScope.SUBTASK,
                        owner_agent="coordinator",
                        created_at=tick,
                        data={
                            "type": "waypoint",
                            "waypoint_type": "mine_resource",
                            "target_position": {
                                "x": nearest.position.x,
                                "y": nearest.position.y,
                            },
                            "target_entity_id": None,
                            "purpose": f"mine_{resource_type}",
                        },
                    )


# ---------------------------------------------------------------------------
# Condition parsing helpers
# ---------------------------------------------------------------------------

def _parse_collection_condition(condition: str):
    """
    Parse "inventory('iron-ore') >= 50" → ("iron-ore", 50).
    Returns (None, None) if the condition doesn't match expected format.
    """
    import re
    m = re.search(r"inventory\(['\"]([^'\"]+)['\"]\)\s*>=\s*(\d+)", condition)
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
) -> Position:
    """
    Return the next waypoint position for a simple outward exploration pattern.

    Uses a concentric square spiral: each ring is 64 tiles (2 chunks) wide.
    The pattern is stateless — given player position and current charted count,
    the same waypoint is always returned (deterministic).

    This is a Phase 6 placeholder. Phase 9 will replace with a proper
    frontier-based exploration planner.
    """
    # Estimate how far out we need to go based on chunks already charted.
    # Each ring of radius r covers roughly (2r)^2 / 1024 chunks.
    ring = max(1, int(math.sqrt(current_chunks)))
    step = 64 * ring  # tiles

    # Emit a waypoint in a N→E→S→W pattern based on ring parity.
    direction = current_chunks % 4
    if direction == 0:
        return Position(player_pos.x, player_pos.y - step)   # north
    elif direction == 1:
        return Position(player_pos.x + step, player_pos.y)   # east
    elif direction == 2:
        return Position(player_pos.x, player_pos.y + step)   # south
    else:
        return Position(player_pos.x - step, player_pos.y)   # west


def _build_condition_namespace(wq: "WorldQuery", tick: int) -> dict:
    """
    Build a minimal eval namespace for subtask condition evaluation.

    Mirrors the key entries from RewardEvaluator's namespace. Only the
    subset needed for Phase 6 subtask conditions is included.
    """
    def inventory(item: str) -> int:
        return wq.inventory_count(item)

    return {
        "inventory": inventory,
        "charted_chunks": wq.charted_chunks,
        "charted_tiles": wq.charted_tiles,
        "charted_area_km2": wq.charted_area_km2,
        "tick": tick,
        "wq": wq,
        "state": wq.state,
        "True": True,
        "False": False,
        "len": len,
        "any": any,
        "all": all,
        "sum": sum,
        "min": min,
        "max": max,
    }