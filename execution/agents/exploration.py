"""
execution/agents/exploration.py

ExplorationAgent — charts new territory by navigating toward uncharted chunks.

Task types handled
------------------
  explore_region   Explore until the task's success condition is met (typically
                   new.charted_chunks >= N, evaluated by the coordinator).

Task attributes (set by coordinator)
-------------------------------------
  task.frontier_position : Position
      Tile-space centre of the nearest frontier chunk at task creation.
      The agent navigates here first, then switches to local uncharted-chunk
      tracking. The coordinator computes this from ChunkGrid.nearest_frontier_position().

  task.home_position : Position (optional)
      Position the coordinator considers the "base" for this exploration task.
      Used only for logging / future distance-budget checks. Defaults to the
      player's position at activate() time if absent.

Exploration loop
----------------
The agent runs a two-phase loop per frontier:

  Phase 1 — APPROACH
    Navigate to task.frontier_position using NavigateSkill. This gets the
    player to the edge of charted territory, within range of uncharted chunks.
    Transition to SCAN when NavigateSkill reports SUCCEEDED or STUCK (if
    STUCK, try scanning anyway — the player may be close enough).

  Phase 2 — SCAN
    Each tick: read wq.nearby_uncharted_chunks. Pick the closest uncharted
    chunk, navigate toward its tile-space centre. As the player moves, new
    chunks are charted and Lua emits them via newly_charted_chunks. When
    nearby_uncharted_chunks is empty the player is surrounded by charted
    territory — signal the coordinator by setting _needs_new_frontier = True
    and returning to APPROACH with a stale frontier (coordinator will provide
    a new task). Until then the agent keeps walking.

Coordinator interaction
-----------------------
The task's success/failure conditions are evaluated by the coordinator each
tick against the WorldQuery. The agent does not evaluate them itself. When
the coordinator decides the task is complete (or failed/timed out), it pops
the task and the agent is deactivated.

When nearby_uncharted_chunks is empty the agent writes a
"exploration_needs_frontier" OBSERVATION to the blackboard. The coordinator
reads this and either:
  a) derives a new explore_region task with an updated frontier_position, or
  b) escalates if no more frontiers exist in ChunkGrid.

Self-model patches
------------------
ExplorationAgent does not write factory graph patches. Chunk charting is
handled by the loop draining wq.newly_charted_chunks directly into ChunkGrid
each tick — the agent does not need to do this.

Rules
-----
- No LLM calls. No RCON. No KnowledgeBase access.
- All state cleared on activate().
"""

from __future__ import annotations

import logging
import math
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

from execution.agents.base import AgentProtocol
from execution.blackboard import EntryCategory, EntryScope
from execution.skills.navigate import NavigateSkill
from execution.skills.base import SkillStatus
from bridge import Action
from world import Position
from world.model.patch import SelfModelPatch

if TYPE_CHECKING:
    from execution.blackboard import Blackboard
    from planning.tasks.task import Task
    from world import KnowledgeBase, WorldQuery, WorldWriter

log = logging.getLogger(__name__)

# Tile-space size of one chunk (32×32 tiles).
_CHUNK_SIZE = 32


class _Phase(Enum):
    APPROACH = auto()   # navigating to initial frontier_position
    SCAN     = auto()   # walking toward nearby uncharted chunks


class ExplorationAgent(AgentProtocol):
    """
    Charts new territory by navigating toward uncharted chunks.

    Uses a single NavigateSkill instance, switching its target between the
    initial frontier position (APPROACH phase) and the closest uncharted
    chunk centre (SCAN phase).
    """

    AGENT_ID = "exploration"

    TASK_TYPES = frozenset({"explore_region"})

    def __init__(self) -> None:
        self._skill = NavigateSkill()
        self._phase: _Phase = _Phase.APPROACH
        self._home_position: Optional[Position] = None
        self._frontier_position: Optional[Position] = None
        self._current_uncharted_target: Optional[Position] = None
        self._needs_new_frontier: bool = False
        self._chunks_approached: int = 0
        self._outcome_written: bool = False

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
        self._skill.reset()
        self._phase = _Phase.APPROACH
        self._needs_new_frontier = False
        self._chunks_approached = 0
        self._outcome_written = False
        self._current_uncharted_target = None

        player_pos = wq.player_position() if wq is not None else None
        self._home_position = getattr(task, "home_position", None) or player_pos
        self._frontier_position = getattr(task, "frontier_position", None)

        if self._frontier_position is not None:
            self._skill.start(target_position=self._frontier_position)
            log.debug(
                "ExplorationAgent activated: task=%s frontier=%s",
                task.id[:8], self._frontier_position,
            )
        else:
            # No frontier provided — go straight to SCAN from current position.
            log.debug(
                "ExplorationAgent: task=%s no frontier_position, starting SCAN",
                task.id[:8],
            )
            self._phase = _Phase.SCAN

        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.TASK,
            owner_agent=self.AGENT_ID,
            created_at=task.created_at if wq is None else wq.tick,
            data={
                "type":              "exploration_started",
                "task_id":           task.id,
                "frontier_position": (
                    {"x": self._frontier_position.x, "y": self._frontier_position.y}
                    if self._frontier_position else None
                ),
                "charted_chunks_at_start": wq.charted_chunks if wq is not None else 0,
            },
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
        if self._phase == _Phase.APPROACH:
            return self._tick_approach(task, blackboard, wq, ww, tick)
        else:
            return self._tick_scan(task, blackboard, wq, ww, tick)

    def observe(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> dict:
        pos = wq.player_position()
        obs = self._skill.observe()
        obs.update({
            "agent":                 self.AGENT_ID,
            "task_id":               task.id[:8],
            "phase":                 self._phase.name,
            "player_position":       {"x": pos.x, "y": pos.y},
            "skill_status":          self._skill.status().name,
            "charted_chunks":        wq.charted_chunks,
            "nearby_uncharted":      len(wq.nearby_uncharted_chunks),
            "needs_new_frontier":    self._needs_new_frontier,
            "chunks_approached":     self._chunks_approached,
        })
        return obs

    def progress(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> float:
        # Progress is measured externally by the coordinator via charted_chunks
        # delta. The agent reports a qualitative fraction based on phase.
        if self._phase == _Phase.APPROACH:
            skill_status = self._skill.status()
            if skill_status == SkillStatus.SUCCEEDED:
                return 0.3
            if skill_status == SkillStatus.RUNNING:
                return 0.1
            return 0.0
        # SCAN phase — report proportional to nearby_uncharted emptiness.
        nearby = len(wq.nearby_uncharted_chunks)
        if nearby == 0:
            return 0.9   # fully local; needs new frontier
        return 0.5

    def pending_patches(self) -> list[SelfModelPatch]:
        # Chunk charting is handled by the loop, not this agent.
        return []

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _tick_approach(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        """Navigate to frontier_position, then switch to SCAN."""
        skill_status = self._skill.status()

        # Skill done (arrived or stuck) — enter SCAN regardless.
        if skill_status in (SkillStatus.SUCCEEDED, SkillStatus.STUCK,
                            SkillStatus.FAILED, SkillStatus.IDLE):
            if skill_status == SkillStatus.STUCK:
                log.debug(
                    "ExplorationAgent: approach stalled at %s — switching to SCAN anyway",
                    wq.player_position(),
                )
            self._skill.reset()
            self._phase = _Phase.SCAN
            return self._tick_scan(task, blackboard, wq, ww, tick)

        actions = self._skill.tick(wq, ww, tick)
        self._write_position(blackboard, wq, tick)
        return actions

    def _tick_scan(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        """Walk toward the closest nearby uncharted chunk."""
        uncharted = wq.nearby_uncharted_chunks

        if not uncharted:
            # Surrounded by charted territory — signal the coordinator.
            if not self._needs_new_frontier:
                self._needs_new_frontier = True
                self._skill.reset()
                blackboard.write(
                    # GOAL scope: this observation must survive task
                    # resolution. The coordinator reads it in _handle_explore
                    # after the task completes; TASK scope would cause it to
                    # be cleared before the handler sees it.
                    category=EntryCategory.OBSERVATION,
                    scope=EntryScope.GOAL,
                    owner_agent=self.AGENT_ID,
                    created_at=tick,
                    data={
                        "type":           "exploration_needs_frontier",
                        "task_id":        task.id,
                        "position":       _pos_dict(wq.player_position()),
                        "charted_chunks": wq.charted_chunks,
                    },
                )
                log.debug(
                    "ExplorationAgent: no nearby uncharted chunks — "
                    "needs new frontier (charted=%d)",
                    wq.charted_chunks,
                )
            self._write_position(blackboard, wq, tick)
            return []

        # We have nearby uncharted chunks — pick the closest.
        self._needs_new_frontier = False
        player_pos = wq.player_position()
        target_chunk = _nearest_chunk(uncharted, player_pos)
        target_pos = _chunk_centre(target_chunk.cx, target_chunk.cy)

        # Start or redirect skill only when target changes significantly.
        if self._skill.status() == SkillStatus.IDLE or (
            self._current_uncharted_target is None
            or _dist(target_pos, self._current_uncharted_target) > _CHUNK_SIZE / 2
        ):
            self._skill.reset()
            self._skill.start(target_position=target_pos)
            self._current_uncharted_target = target_pos
            self._chunks_approached += 1
            log.debug(
                "ExplorationAgent: targeting uncharted chunk (%d,%d) → %s",
                target_chunk.cx, target_chunk.cy, target_pos,
            )

        # If skill is stuck trying to reach this chunk, try the next one.
        if self._skill.status() == SkillStatus.STUCK:
            log.debug(
                "ExplorationAgent: stuck on chunk (%d,%d), resetting skill",
                target_chunk.cx, target_chunk.cy,
            )
            self._skill.reset()
            self._current_uncharted_target = None
            self._write_position(blackboard, wq, tick)
            return []

        actions = self._skill.tick(wq, ww, tick)
        self._write_position(blackboard, wq, tick)
        return actions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_position(
        self,
        blackboard: "Blackboard",
        wq: "WorldQuery",
        tick: int,
    ) -> None:
        pos = wq.player_position()
        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.TASK,
            owner_agent=self.AGENT_ID,
            created_at=tick,
            data={"type": "player_position", "position": _pos_dict(pos)},
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _pos_dict(pos: Position) -> dict:
    return {"x": pos.x, "y": pos.y}


def _chunk_centre(cx: int, cy: int) -> Position:
    """Tile-space centre of chunk (cx, cy)."""
    return Position(
        x=cx * _CHUNK_SIZE + _CHUNK_SIZE / 2.0,
        y=cy * _CHUNK_SIZE + _CHUNK_SIZE / 2.0,
    )


def _dist(a: Position, b: Position) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def _nearest_chunk(chunks, player_pos: Position):
    """Return the ChunkCoord from chunks closest to player_pos."""
    def _key(c):
        cx_tile = c.cx * _CHUNK_SIZE + _CHUNK_SIZE / 2.0
        cy_tile = c.cy * _CHUNK_SIZE + _CHUNK_SIZE / 2.0
        return (cx_tile - player_pos.x) ** 2 + (cy_tile - player_pos.y) ** 2
    return min(chunks, key=_key)