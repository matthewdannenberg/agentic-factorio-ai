"""
agent/network/agents/navigation.py

NavigationAgent — rule-based movement agent for Phase 6.

Satisfies AgentProtocol. Handles movement only: walking the player character
toward waypoint targets specified by the coordinator on the blackboard.

Scope
-----
The navigation agent's responsibility is strictly movement:
  - Read waypoint INTENTION entries from the blackboard
  - Emit MoveTo actions toward the target position or entity
  - Detect arrival using is_reachable() (entity targets) or is_at() (position
    targets) — never a hardcoded reach constant
  - Write position OBSERVATION entries each tick so the coordinator knows
    where the player is
  - Write a "waypoint_reached" OBSERVATION when arrival is detected

Phase 6 concession — mining
---------------------------
For Phase 6, the navigation agent also handles MineResource and MineEntity
actions when the active subtask is a mining subtask (identified by a
"mine_resource" or "mine_entity" waypoint type). This is a pragmatic
simplification: mining is position-dependent and physically trivial. It is
NOT an architectural commitment — a future inventory-management agent will
own item interaction. This concession is documented here and in the Phase 6
brief so it is not mistaken for intentional design.

What the navigation agent does NOT do
--------------------------------------
- It does not decide where to go (that is the coordinator's job)
- It does not read the `purpose` field of a waypoint entry or branch on it
- It does not contain spatial reasoning beyond "move toward target position"
- It does not evaluate goal success conditions
- It does not write to the subtask ledger

Rules
-----
- No LLM calls. No RCON. Satisfies AgentProtocol interface.
- All state between ticks is stored on the agent instance (current waypoint,
  waypoints completed count). Cleared on activate().
- Pathfinding: always pathfind=True in MoveTo actions.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from agent.blackboard import EntryCategory, EntryScope
from agent.network.agent_protocol import AgentProtocol
from agent.preconditions import is_at, is_reachable
from bridge.actions import Action, MineEntity, MineResource, MoveTo
from world.state import Position

if TYPE_CHECKING:
    from agent.blackboard import Blackboard
    from planning.goal import Goal
    from world.query import WorldQuery
    from world.writer import WorldWriter

log = logging.getLogger(__name__)

AGENT_ID = "navigation"

# Tolerance (tiles) for position-only waypoint arrival when no entity_id
# is present. Derived from what "close enough to interact" means for resource
# patches. Not used for entity targets — those use is_reachable() instead.
_POSITION_ARRIVAL_TOLERANCE = 1.5


class NavigationAgent(AgentProtocol):
    """
    Rule-based navigation agent.

    Reads waypoint INTENTION entries from the blackboard and emits MoveTo
    (and, as a Phase 6 concession, mining) actions to execute them.

    State
    -----
    _current_waypoint_id : blackboard entry id of the waypoint being pursued.
    _waypoints_completed : count of waypoints completed since activate().
    _waypoints_total     : count of waypoints seen since activate() (for progress).
    _goal                : the current goal (stored for context in observations).
    """

    def __init__(self) -> None:
        self._current_waypoint_id: Optional[str] = None
        self._waypoints_completed: int = 0
        self._waypoints_total: int = 0
        self._goal: Optional["Goal"] = None

    # ------------------------------------------------------------------
    # AgentProtocol
    # ------------------------------------------------------------------

    def activate(
        self,
        goal: "Goal",
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> None:
        """
        Reset navigation state and write an initial position observation.

        Called once when a new goal is assigned. Clears any carry-over state
        from a previous goal (blackboard is already cleared by the coordinator,
        but agent-internal state needs its own reset).
        """
        self._goal = goal
        self._current_waypoint_id = None
        self._waypoints_completed = 0
        self._waypoints_total = 0

        pos = wq.player_position()
        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.GOAL,
            owner_agent=AGENT_ID,
            created_at=wq.tick,
            data={
                "type": "player_position",
                "position": {"x": pos.x, "y": pos.y},
            },
        )
        log.debug("NavigationAgent activated at %s for goal %s", pos, goal.id[:8])

    def tick(
        self,
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        """
        One navigation tick. Returns at most one action.

        Sequence:
        1. Write current position observation.
        2. Find the active waypoint INTENTION entry.
        3. If arrived → write waypoint_reached OBSERVATION, clear current
           waypoint tracking, return empty list (coordinator handles transition).
        4. If not arrived → emit MoveTo (or mining action for mining waypoints).
        5. If no waypoint present → return empty list.
        """
        pos = wq.player_position()

        # Always write a fresh position observation so the coordinator can
        # track the player's location without querying WorldQuery itself.
        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.SUBTASK,
            owner_agent=AGENT_ID,
            created_at=tick,
            data={
                "type": "player_position",
                "position": {"x": pos.x, "y": pos.y},
            },
        )

        waypoint = self._find_active_waypoint(blackboard, tick)
        if waypoint is None:
            return []

        waypoint_data = waypoint.data
        target_entity_id: Optional[int] = waypoint_data.get("target_entity_id")
        target_pos_dict: Optional[dict] = waypoint_data.get("target_position")
        waypoint_type: str = waypoint_data.get("waypoint_type", "move")

        # --- Arrival detection ---
        arrived = self._check_arrival(
            target_entity_id=target_entity_id,
            target_pos_dict=target_pos_dict,
            wq=wq,
        )

        if arrived:
            self._signal_waypoint_reached(blackboard, waypoint, tick)
            self._current_waypoint_id = None
            self._waypoints_completed += 1
            return []

        # --- Emit action ---
        if waypoint_type == "mine_resource":
            # Phase 6 mining concession: emit MineResource when at the patch
            # (arrival not yet signalled means we're still approaching).
            return self._approach_action(target_entity_id, target_pos_dict)

        if waypoint_type == "mine_entity":
            # Phase 6 mining concession: emit MineEntity when reachable.
            if target_entity_id is not None:
                return [MineEntity(entity_id=target_entity_id)]
            return self._approach_action(None, target_pos_dict)

        # Default: emit MoveTo toward the target.
        return self._approach_action(target_entity_id, target_pos_dict, wq=wq)

    def observe(
        self,
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> dict:
        """
        Return a flat dict of current navigation state for examination.
        """
        pos = wq.player_position()
        return {
            "agent": AGENT_ID,
            "player_position": {"x": pos.x, "y": pos.y},
            "waypoints_completed": self._waypoints_completed,
            "waypoints_total": self._waypoints_total,
            "has_active_waypoint": self._current_waypoint_id is not None,
        }

    def progress(
        self,
        blackboard: "Blackboard",
        wq: "WorldQuery",
    ) -> float:
        """
        Fraction of waypoints in this goal completed, in [0.0, 1.0].

        Returns 0.0 if no waypoints have been issued yet.
        """
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
        """
        Return the current waypoint INTENTION entry, or None.

        Reads all non-expired INTENTION entries and returns the first one
        whose data["type"] == "waypoint". Tracks the entry id so that
        repeated ticks on the same waypoint are handled correctly.

        Also updates _waypoints_total for progress tracking.
        """
        intentions = blackboard.read(
            category=EntryCategory.INTENTION,
            current_tick=tick,
        )
        waypoints = [e for e in intentions if e.data.get("type") == "waypoint"]

        if not waypoints:
            self._current_waypoint_id = None
            return None

        # Take the first (oldest) unfinished waypoint.
        waypoint = waypoints[0]

        # Update total count whenever we see new waypoints.
        seen_ids = {e.id for e in waypoints}
        if waypoint.id != self._current_waypoint_id:
            # New waypoint — update tracking.
            self._current_waypoint_id = waypoint.id
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
        """
        True if the player has arrived at the waypoint target.

        Entity targets: use is_reachable() — derives reach from WorldQuery.
        Position targets: use is_at() with _POSITION_ARRIVAL_TOLERANCE.
        """
        if target_entity_id is not None:
            return is_reachable(target_entity_id, wq)

        if target_pos_dict is not None:
            target = Position(
                x=target_pos_dict["x"],
                y=target_pos_dict["y"],
            )
            return is_at(target, wq, tolerance=_POSITION_ARRIVAL_TOLERANCE)

        return False

    def _approach_action(
        self,
        target_entity_id: Optional[int],
        target_pos_dict: Optional[dict],
        wq: Optional["WorldQuery"] = None,
    ) -> list[Action]:
        """
        Return a MoveTo action toward the target.

        If an entity_id is provided and wq is available, resolve the entity's
        position from WorldQuery. Otherwise fall back to target_pos_dict.
        """
        target_pos: Optional[Position] = None

        if target_entity_id is not None and wq is not None:
            entity = wq.entity_by_id(target_entity_id)
            if entity is not None:
                target_pos = entity.position

        if target_pos is None and target_pos_dict is not None:
            target_pos = Position(
                x=target_pos_dict["x"],
                y=target_pos_dict["y"],
            )

        if target_pos is None:
            log.warning(
                "NavigationAgent: waypoint has neither a resolvable entity nor "
                "a target_position; returning empty action list"
            )
            return []

        return [MoveTo(position=target_pos, pathfind=True)]

    def _signal_waypoint_reached(
        self,
        blackboard: "Blackboard",
        waypoint,
        tick: int,
    ) -> None:
        """
        Write a waypoint_reached OBSERVATION to the blackboard.

        The coordinator reads this on the next tick and performs the subtask
        transition (complete the movement subtask, activate the next one).
        """
        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.SUBTASK,
            owner_agent=AGENT_ID,
            created_at=tick,
            data={
                "type": "waypoint_reached",
                "waypoint_id": waypoint.id,
                "target_entity_id": waypoint.data.get("target_entity_id"),
                "target_position": waypoint.data.get("target_position"),
            },
        )
        log.debug(
            "NavigationAgent: waypoint_reached (entry %s)",
            waypoint.id[:8],
        )
