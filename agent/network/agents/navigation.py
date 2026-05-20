"""
agent/network/agents/navigation.py

NavigationAgent — rule-based movement agent for Phase 6.

Satisfies AgentProtocol. Handles movement only: walking the player character
toward waypoint targets specified by the coordinator on the blackboard.

Scope
-----
The navigation agent's sole responsibility is movement:
  - Read waypoint INTENTION entries from the blackboard
  - Emit MoveTo actions toward the target position or entity
  - Detect arrival using is_reachable() (entity targets) or is_at() (position
    targets) — never a hardcoded reach constant
  - Write position OBSERVATION entries each tick
  - Emit StopMovement on arrival so the character does not overshoot
  - Emit StopMovement on arrival so the character does not overshoot

What the navigation agent does NOT do
--------------------------------------
- Mine, craft, or interact with any entity (MiningAgent owns that)
- Decide where to go (coordinator writes waypoints to the blackboard)
- Inspect or store the Goal — agents receive Subtasks only
- Read the `purpose` field of a waypoint entry or branch on it
- Evaluate subtask success conditions (coordinator does that)
- Write to the subtask ledger

Movement model — one-shot with suppression
-------------------------------------------
fa.move_to() triggers a pathfinding request in the Lua mod; the mod then
drives walking_state every game tick along the computed path. The Python side
should issue MoveTo as rarely as possible to avoid cancelling in-flight path
requests.

A new MoveTo is issued only when:
  1. A new waypoint is assigned (different waypoint blackboard entry).
  2. The target position has changed by more than _REDUNDANT_THRESHOLD tiles
     from the last-issued target (e.g. an entity has moved).
  3. The player has been stationary for more than _STALL_GRACE_TICKS ticks
     since the last command AND the target hasn't changed — indicating the
     path was lost or the pathfinder returned unreachable.

Condition 3 has a minimum wait of _STALL_GRACE_TICKS (default ~3 seconds)
to give the pathfinder time to return a result before declaring a stall.

Rules
-----
- No LLM calls. No RCON. Satisfies AgentProtocol interface.
- All state between ticks stored on the instance. Cleared on activate().
- Always pathfind=True in MoveTo.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, TYPE_CHECKING

from agent.blackboard import EntryCategory, EntryScope
from agent.network.agent_protocol import AgentProtocol
from agent.preconditions import is_at, is_reachable
from bridge.actions import Action, MoveTo, StopMovement
from world.state import Position

if TYPE_CHECKING:
    from agent.blackboard import Blackboard
    from agent.subtask import Subtask
    from world.query import WorldQuery
    from world.writer import WorldWriter

log = logging.getLogger(__name__)

AGENT_ID = "navigation"

# Arrival tolerance (tiles) for position-only waypoints (no entity_id).
_POSITION_ARRIVAL_TOLERANCE = 1.5

# Suppress a new MoveTo if the new target is within this many tiles of the
# last-issued target — avoids cancelling an in-flight pathfinding request
# just because the waypoint position is slightly different.
_REDUNDANT_THRESHOLD = 0.5

# Minimum ticks after issuing a MoveTo before stall detection fires.
# Gives the Lua pathfinder time to compute and the character time to start
# moving. At 60 tps and TICK_INTERVAL=10, 180 ticks ≈ 3 seconds of polls.
_STALL_GRACE_TICKS = 180

# Position change below this threshold (tiles) per tick → considered stalled.
_STOPPED_THRESHOLD = 0.05


class NavigationAgent(AgentProtocol):
    """
    Rule-based navigation agent.

    Issues MoveTo only when the waypoint changes, the target moves
    significantly, or a genuine stall is detected after a grace period.
    """

    def __init__(self) -> None:
        self._current_subtask: Optional["Subtask"] = None
        self._waypoints_completed: int = 0
        self._waypoints_total: int = 0
        self._last_issued_waypoint_id: Optional[str] = None
        self._last_issued_target: Optional[Position] = None
        self._last_move_tick: int = 0
        self._last_position: Optional[Position] = None

    # ------------------------------------------------------------------
    # AgentProtocol
    # ------------------------------------------------------------------

    def activate(
        self,
        subtask: "Subtask",
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> None:
        """
        Called when a new subtask is assigned to this agent.
        Resets all internal state for the new subtask.
        """
        self._current_subtask = subtask
        self._waypoints_completed = 0
        self._waypoints_total = 0
        self._last_issued_waypoint_id = None
        self._last_issued_target = None
        self._last_move_tick = 0
        self._last_position = None

        pos = wq.player_position()
        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.SUBTASK,
            owner_agent=AGENT_ID,
            created_at=wq.tick,
            data={
                "type": "player_position",
                "position": {"x": pos.x, "y": pos.y},
                "subtask_id": subtask.id,
            },
        )
        log.debug(
            "NavigationAgent activated at %s for subtask %s: %s",
            pos, subtask.id[:8], subtask.description,
        )

    def tick(
        self,
        subtask: "Subtask",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        """
        One navigation tick. Returns at most one action.

        Returns MoveTo only on a new waypoint or stall.
        Returns StopMovement on arrival.
        Returns [] on all other ticks.
        """
        pos = wq.player_position()

        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.SUBTASK,
            owner_agent=AGENT_ID,
            created_at=tick,
            data={"type": "player_position", "position": {"x": pos.x, "y": pos.y}},
        )

        waypoint = self._find_active_waypoint(blackboard, tick)
        if waypoint is None:
            self._last_position = pos
            return []

        waypoint_data = waypoint.data
        target_entity_id: Optional[int] = waypoint_data.get("target_entity_id")
        target_pos_dict: Optional[dict] = waypoint_data.get("target_position")

        # --- Arrival detection ---
        # The coordinator evaluates the subtask's success_condition to detect
        # completion — no side-channel signal needed here. On arrival we simply
        # stop moving so the character doesn't overshoot the target.
        if self._check_arrival(target_entity_id, target_pos_dict, wq):
            self._last_issued_waypoint_id = None
            self._waypoints_completed += 1
            self._last_position = pos
            return [StopMovement()]

        # --- Movement ---
        is_new_waypoint = waypoint.id != self._last_issued_waypoint_id
        target_pos = self._resolve_target(target_entity_id, target_pos_dict, wq)

        if target_pos is None:
            self._last_position = pos
            return []

        is_redundant = self._is_redundant_target(target_pos)
        grace_elapsed = (tick - self._last_move_tick) >= _STALL_GRACE_TICKS
        is_stalled = grace_elapsed and self._is_stalled(pos)

        should_issue = is_new_waypoint or not is_redundant or is_stalled

        if should_issue:
            if is_stalled and not is_new_waypoint:
                log.debug(
                    "NavigationAgent: stall detected after %d ticks, re-issuing",
                    tick - self._last_move_tick,
                )
            action = MoveTo(position=target_pos, pathfind=True)
            self._last_issued_waypoint_id = waypoint.id
            self._last_issued_target = target_pos
            self._last_move_tick = tick
            self._last_position = pos
            return [action]

        self._last_position = pos
        return []

    def observe(
        self,
        subtask: "Subtask",
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> dict:
        pos = wq.player_position()
        return {
            "agent": AGENT_ID,
            "subtask_id": subtask.id[:8],
            "player_position": {"x": pos.x, "y": pos.y},
            "waypoints_completed": self._waypoints_completed,
            "waypoints_total": self._waypoints_total,
            "has_active_waypoint": self._last_issued_waypoint_id is not None,
        }

    def progress(
        self,
        subtask: "Subtask",
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> float:
        if self._waypoints_total == 0:
            return 0.0
        return min(1.0, self._waypoints_completed / self._waypoints_total)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_active_waypoint(
        self,
        blackboard: "Blackboard",
        tick: int,
    ) -> Optional[object]:
        intentions = blackboard.read(
            category=EntryCategory.INTENTION,
            current_tick=tick,
        )
        waypoints = [e for e in intentions if e.data.get("type") == "waypoint"]
        if not waypoints:
            return None
        waypoint = waypoints[0]
        if waypoint.id != self._last_issued_waypoint_id:
            self._waypoints_total = max(
                self._waypoints_total,
                self._waypoints_completed + len(waypoints),
            )
        return waypoint

    def _check_arrival(
        self,
        target_entity_id: Optional[int],
        target_pos_dict: Optional[dict],
        wq: "WorldQuery",
    ) -> bool:
        if target_entity_id is not None:
            return is_reachable(target_entity_id, wq)
        if target_pos_dict is not None:
            target = Position(x=target_pos_dict["x"], y=target_pos_dict["y"])
            return is_at(target, wq, tolerance=_POSITION_ARRIVAL_TOLERANCE)
        return False

    def _resolve_target(
        self,
        target_entity_id: Optional[int],
        target_pos_dict: Optional[dict],
        wq: "WorldQuery",
    ) -> Optional[Position]:
        """Resolve waypoint data to a concrete Position, or None."""
        if target_entity_id is not None:
            entity = wq.entity_by_id(target_entity_id)
            if entity is not None:
                return entity.position
        if target_pos_dict is not None:
            return Position(x=target_pos_dict["x"], y=target_pos_dict["y"])
        log.warning("NavigationAgent: waypoint has no resolvable target position")
        return None

    def _is_redundant_target(self, target: Position) -> bool:
        """
        True if *target* is within _REDUNDANT_THRESHOLD of the last-issued
        target — meaning a MoveTo to this position was already sent and the
        pathfinder is likely still executing it.
        """
        if self._last_issued_target is None:
            return False
        dx = target.x - self._last_issued_target.x
        dy = target.y - self._last_issued_target.y
        return math.sqrt(dx * dx + dy * dy) < _REDUNDANT_THRESHOLD

    def _is_stalled(self, current_pos: Position) -> bool:
        """
        True if the player hasn't moved since the last tick — the pathfinder
        may have returned unreachable or the path was lost.
        Only meaningful after _STALL_GRACE_TICKS have elapsed.
        """
        if self._last_position is None:
            return False
        dx = current_pos.x - self._last_position.x
        dy = current_pos.y - self._last_position.y
        return math.sqrt(dx * dx + dy * dy) < _STOPPED_THRESHOLD