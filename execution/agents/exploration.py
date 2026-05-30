"""
execution/agents/exploration.py

ExplorationAgent — charts new territory by navigating toward uncharted chunks.

Task types handled
------------------
  explore_region   Explore until the task's success condition is met.

Task attributes (set by coordinator)
-------------------------------------
  task.target_chunks : int (optional)
      The coordinator's exploration target, for logging only. The task's
      success_condition is the authoritative completion signal.

Exploration loop
----------------
The agent runs a fully autonomous loop — no coordinator involvement between
task activation and task completion:

  Phase 1 — APPROACH
    Pick a frontier from wq.chunk_map (the accumulated charted-chunk set in
    WorldState). Navigate toward it using NavigateSkill. On SUCCEEDED or STUCK,
    transition to SCAN regardless — the player may be close enough to start
    revealing nearby uncharted chunks.

  Phase 2 — SCAN
    Read wq.nearby_uncharted_chunks each tick. Pick a target from the nearest
    _SCAN_CANDIDATE_POOL candidates (random selection breaks deterministic
    retry loops). Navigate toward it. As the player moves, new chunks are
    charted and appear in wq.chunk_map.

    When nearby_uncharted_chunks is empty: the player is surrounded by charted
    territory within the scan radius. Return to APPROACH with a freshly-chosen
    frontier from wq.chunk_map. This inner loop repeats until the task's
    success_condition fires or a failure_condition is hit.

    If chunk_map is also empty (session just started, no deltas yet) and no
    nearby uncharted chunks exist, fall back to pushing outward from the
    player's current position.

Coordinator interaction
-----------------------
The coordinator pushes a single explore_region task and does not manage
frontier selection. The agent loops internally — APPROACH → SCAN → APPROACH →
... — until the task's success_condition is evaluated true by the coordinator,
or until the failure_condition fires.

No blackboard signals are needed between the agent and coordinator during
normal operation. The "exploration_needs_frontier" observation is removed;
frontier management is internal.

Self-model patches
------------------
ExplorationAgent does not write factory graph patches. Chunk accumulation
happens in WorldWriter.integrate_snapshot() — the agent does not need to
do anything for this.

Rules
-----
- No LLM calls. No RCON. No KnowledgeBase access.
- All state cleared on activate().
"""

from __future__ import annotations

import logging
import math
import random
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

from execution.agents.base import AgentProtocol
from execution.blackboard import EntryCategory, EntryScope
from execution.skills.navigate import NavigateSkill
from execution.skills.base import SkillStatus
from bridge import Action, StopMovement
from world import Position
from world.model.patch import SelfModelPatch

if TYPE_CHECKING:
    from execution.blackboard import Blackboard
    from planning.tasks.task import Task
    from world import KnowledgeBase, WorldQuery, WorldWriter

log = logging.getLogger(__name__)

# Tile-space size of one chunk (32×32 tiles).
_CHUNK_SIZE = 32

# Number of nearest frontier candidates to draw from randomly.
# Prevents the same unreachable frontier being retried deterministically.
_SCAN_CANDIDATE_POOL = 10


class _Phase(Enum):
    APPROACH = auto()   # navigating toward a chosen frontier
    SCAN     = auto()   # walking toward nearby uncharted chunks


class ExplorationAgent(AgentProtocol):
    """
    Charts new territory by navigating toward uncharted chunks.

    Fully autonomous: selects its own frontiers from wq.chunk_map and
    wq.nearby_uncharted_chunks without coordinator involvement.
    """

    AGENT_ID = "exploration"
    TASK_TYPES = frozenset({"explore_region"})

    def __init__(self) -> None:
        self._skill = NavigateSkill()
        self._phase: _Phase = _Phase.APPROACH
        self._home_position: Optional[Position] = None
        self._current_target: Optional[Position] = None
        self._chunks_approached: int = 0

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
        self._current_target = None
        self._chunks_approached = 0

        player_pos = wq.player_position() if wq is not None else None
        self._home_position = player_pos

        blackboard.write(
            category=EntryCategory.OBSERVATION,
            scope=EntryScope.TASK,
            owner_agent=self.AGENT_ID,
            created_at=task.created_at if wq is None else wq.tick,
            data={
                "type":                    "exploration_started",
                "task_id":                 task.id,
                "charted_chunks_at_start": wq.charted_chunks if wq is not None else 0,
            },
        )
        log.debug("ExplorationAgent activated: task=%s", task.id[:8])

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
            "agent":             self.AGENT_ID,
            "task_id":           task.id[:8],
            "phase":             self._phase.name,
            "player_position":   {"x": pos.x, "y": pos.y},
            "skill_status":      self._skill.status().name,
            "charted_chunks":    wq.charted_chunks,
            "nearby_uncharted":  len(wq.nearby_uncharted_chunks),
            "chunks_approached": self._chunks_approached,
        })
        return obs

    def progress(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        kb: "KnowledgeBase",
    ) -> float:
        if self._phase == _Phase.APPROACH:
            s = self._skill.status()
            if s == SkillStatus.SUCCEEDED:
                return 0.3
            if s == SkillStatus.RUNNING:
                return 0.1
            return 0.0
        nearby = len(wq.nearby_uncharted_chunks)
        return 0.5 if nearby > 0 else 0.8

    def teardown(self) -> list[Action]:
        """Halt any in-progress Lua movement when the task ends."""
        self._skill.reset()
        return [StopMovement()]

    def pending_patches(self) -> list[SelfModelPatch]:
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
        """
        Navigate toward a chosen frontier, then switch to SCAN.

        On first entry (skill IDLE), choose a frontier from wq.chunk_map or
        wq.nearby_uncharted_chunks. If nothing is available, push outward.
        On nav SUCCEEDED or STUCK, transition to SCAN regardless.
        """
        skill_status = self._skill.status()

        if skill_status == SkillStatus.IDLE:
            # Choose a frontier target.
            target = self._pick_frontier(wq)
            if target is not None:
                self._current_target = target
                self._skill.start(target_position=target)
                self._chunks_approached += 1
                log.debug(
                    "ExplorationAgent: APPROACH → %s (task=%s)",
                    target, task.id[:8],
                )
            else:
                # Nowhere to go — drop into SCAN and see if nearby chunks appear.
                log.debug(
                    "ExplorationAgent: no frontier available, falling back to SCAN"
                )
                self._phase = _Phase.SCAN
                return self._tick_scan(task, blackboard, wq, ww, tick)

        skill_status = self._skill.status()

        if skill_status in (SkillStatus.SUCCEEDED, SkillStatus.STUCK,
                            SkillStatus.FAILED):
            if skill_status == SkillStatus.STUCK:
                log.debug(
                    "ExplorationAgent: approach stalled at %s — switching to SCAN",
                    wq.player_position(),
                )
            self._skill.reset()
            self._current_target = None
            self._phase = _Phase.SCAN
            return self._tick_scan(task, blackboard, wq, ww, tick)

        self._write_position(blackboard, wq, tick)
        return self._skill.tick(wq, ww, tick)

    def _tick_scan(
        self,
        task: "Task",
        blackboard: "Blackboard",
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        """
        Walk toward nearby uncharted chunks until surrounded.

        Picks randomly from the nearest _SCAN_CANDIDATE_POOL candidates so
        that a STUCK attempt targets a different chunk on the next try.
        When no nearby uncharted chunks remain, returns to APPROACH to pick
        a new frontier from wq.chunk_map.
        """
        uncharted = wq.nearby_uncharted_chunks

        if not uncharted:
            # Locally surrounded — pick a new distant frontier.
            log.debug(
                "ExplorationAgent: locally surrounded, returning to APPROACH "
                "(charted=%d)", wq.charted_chunks,
            )
            self._skill.reset()
            self._current_target = None
            self._phase = _Phase.APPROACH
            return self._tick_approach(task, blackboard, wq, ww, tick)

        # Pick a target from the nearest candidates.
        player_pos = wq.player_position()
        target_pos = _pick_scan_target(uncharted, player_pos)

        # Redirect skill when target changes significantly or skill is idle.
        if (self._skill.status() == SkillStatus.IDLE
                or self._current_target is None
                or _dist(target_pos, self._current_target) > _CHUNK_SIZE / 2):
            self._skill.reset()
            self._skill.start(target_position=target_pos)
            self._current_target = target_pos
            self._chunks_approached += 1

        # On STUCK, reset and let the next tick pick a fresh target.
        if self._skill.status() == SkillStatus.STUCK:
            self._skill.reset()
            self._current_target = None
            self._write_position(blackboard, wq, tick)
            return []

        self._write_position(blackboard, wq, tick)
        return self._skill.tick(wq, ww, tick)

    # ------------------------------------------------------------------
    # Frontier selection
    # ------------------------------------------------------------------

    def _pick_frontier(self, wq: "WorldQuery") -> Optional[Position]:
        """
        Choose a frontier position for the APPROACH phase.

        Sources, in priority order:
          1. wq.nearby_uncharted_chunks — closest _SCAN_CANDIDATE_POOL entries,
             pick one randomly (already near the frontier edge).
          2. wq.chunk_map.frontiers() — accumulated charted-chunk set; pick
             from the nearest _SCAN_CANDIDATE_POOL frontier chunks randomly.
          3. Outward push — no chunk data yet; push outward from player.
        """
        player_pos = wq.player_position()

        # 1. Nearby uncharted chunks (PROXIMAL, most reliable).
        nearby = wq.nearby_uncharted_chunks
        if nearby:
            return _pick_scan_target(nearby, player_pos)

        # 2. Accumulated chunk_map frontiers (NON-PROXIMAL).
        chunk_map = wq.chunk_map
        if chunk_map:
            frontier_pos = _pick_chunk_map_frontier(chunk_map, player_pos)
            if frontier_pos is not None:
                return frontier_pos

        # 3. Fallback: push outward.
        import math as _math
        radius = max(64.0, _math.sqrt(wq.charted_chunks) * 32.0)
        return Position(x=player_pos.x + radius, y=player_pos.y)

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
            data={"type": "player_position", "position": {"x": pos.x, "y": pos.y}},
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _chunk_centre(cx: int, cy: int) -> Position:
    return Position(
        x=cx * _CHUNK_SIZE + _CHUNK_SIZE / 2.0,
        y=cy * _CHUNK_SIZE + _CHUNK_SIZE / 2.0,
    )


def _dist(a: Position, b: Position) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def _pick_scan_target(chunks, player_pos: Position) -> Position:
    """
    Pick a tile-space target from nearby uncharted chunks.
    Sorts by distance, selects randomly from the nearest _SCAN_CANDIDATE_POOL.
    """
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (
            (c.cx * _CHUNK_SIZE + _CHUNK_SIZE / 2.0 - player_pos.x) ** 2
            + (c.cy * _CHUNK_SIZE + _CHUNK_SIZE / 2.0 - player_pos.y) ** 2
        ),
    )
    candidate = random.choice(sorted_chunks[:_SCAN_CANDIDATE_POOL])
    return _chunk_centre(candidate.cx, candidate.cy)


def _pick_chunk_map_frontier(chunk_map, player_pos: Position) -> Optional[Position]:
    """
    Pick a frontier position from the accumulated chunk_map.
    Sorts frontiers by distance, selects randomly from the nearest pool.
    """
    frontiers = chunk_map.frontiers()
    if not frontiers:
        return None
    cx0 = int(player_pos.x // _CHUNK_SIZE)
    cy0 = int(player_pos.y // _CHUNK_SIZE)
    sorted_fronts = sorted(
        frontiers,
        key=lambda c: (c[0] - cx0) ** 2 + (c[1] - cy0) ** 2,
    )
    candidate = random.choice(sorted_fronts[:_SCAN_CANDIDATE_POOL])
    return _chunk_centre(*candidate)