"""
agent/loop.py

FactorioLoop — master orchestration for the four-timescale data flow.

This is the top-level controller that wires every subsystem together and
drives the per-tick poll cycle. It is the only place in the codebase that
knows about all subsystems simultaneously.

Four timescales
---------------
Per-tick (every TICK_INTERVAL game ticks):
  1. Poll Factorio → parse WorldState snapshot → integrate into live state
  2. Evaluate active goal conditions via RewardEvaluator
  3. coordinator.tick() → ExecutionResult
  4. Dispatch actions via ActionExecutor
  5. Handle ExecutionStatus (PROGRESSING, WAITING, STUCK, COMPLETE)

Per-goal:
  - On goal start: coordinator.reset(), activate agents
  - On goal complete/fail: record outcome to behavioral memory, request next goal
  - On STUCK: pass StuckContext to GoalSource.handle_stuck(), re-inject seeds

Per-run:
  - On startup: load KB, behavioral memory, construct all components
  - On shutdown: persist behavioral memory, flush logs

Rules
-----
- No LLM calls directly — the loop delegates to GoalSource.
- No WorldState reads for reasoning — the loop uses WorldQuery exclusively.
- The loop does not interpret goal conditions — RewardEvaluator does that.
- The loop does not interpret agent actions — ActionExecutor does that.
- All dependencies are injected at construction; no global state.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import config
from agent.execution_protocol import ExecutionStatus
from planning.goal import Goal, GoalStatus
from planning.reward_evaluator import RewardEvaluator
from world.state import WorldState
from world.query import WorldQuery
from world.writer import WorldWriter

if TYPE_CHECKING:
    from agent.memory.behavioral import BehavioralMemoryProtocol
    from agent.network.coordinator import CoordinatorProtocol
    from agent.self_model import SelfModel
    from bridge.action_executor import ActionExecutor
    from bridge.rcon_client import RconClient
    from bridge.state_parser import StateParser
    from bridge.world_poller import WorldPoller
    from llm.goal_source import GoalSource

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoopConfig
# ---------------------------------------------------------------------------

@dataclass
class LoopConfig:
    """
    Runtime configuration for the loop.

    Mirrors the relevant fields from config.py but can be overridden per-run
    without modifying the config file (useful for tests and experiments).
    """
    tick_interval: int   = config.TICK_INTERVAL
    local_scan_radius: int          = config.LOCAL_SCAN_RADIUS
    resource_scan_radius: int       = config.RESOURCE_SCAN_RADIUS
    ground_item_scan_radius: int    = config.GROUND_ITEM_SCAN_RADIUS
    max_stuck_retries: int          = 3    # STUCK events before giving up on a goal
    shutdown_on_empty_queue: bool   = True # exit cleanly when GoalQueue exhausted


# ---------------------------------------------------------------------------
# LoopStats — lightweight per-run telemetry
# ---------------------------------------------------------------------------

@dataclass
class LoopStats:
    goals_attempted: int = 0
    goals_completed: int = 0
    goals_failed: int    = 0
    goals_stuck: int     = 0
    total_ticks: int     = 0
    stuck_events: int    = 0


# ---------------------------------------------------------------------------
# FactorioLoop
# ---------------------------------------------------------------------------

class FactorioLoop:
    """
    Master orchestration loop.

    All dependencies are injected at construction. The loop holds no
    references to concrete implementation classes — only protocols and
    abstract interfaces, so every subsystem can be swapped independently.

    Parameters
    ----------
    client          : Connected RconClient.
    parser          : StateParser for converting raw RCON output to WorldState.
    poller          : WorldPoller for sending fa.get_state and returning JSON.
    executor        : ActionExecutor for dispatching Action objects to Factorio.
    coordinator     : CoordinatorProtocol implementation.
    goal_source     : GoalSource implementation (GoalQueue or real LLM).
    behavioral_mem  : BehavioralMemoryProtocol for recording outcomes.
    self_model      : SelfModel — passed to coordinator; starts empty each run.
    evaluator       : RewardEvaluator — evaluates goal conditions.
    cfg             : LoopConfig — runtime parameters.
    """

    def __init__(
        self,
        client: "RconClient",
        parser: "StateParser",
        poller: "WorldPoller",
        executor: "ActionExecutor",
        coordinator: "CoordinatorProtocol",
        goal_source: "GoalSource",
        behavioral_mem: "BehavioralMemoryProtocol",
        self_model: "SelfModel",
        evaluator: Optional[RewardEvaluator] = None,
        cfg: Optional[LoopConfig] = None,
    ) -> None:
        self._client       = client
        self._parser       = parser
        self._poller       = poller
        self._executor     = executor
        self._coordinator  = coordinator
        self._goal_source  = goal_source
        self._mem          = behavioral_mem
        self._sm           = self_model
        self._evaluator    = evaluator or RewardEvaluator()
        self._cfg          = cfg or LoopConfig()

        # Live world state — single shared instance mutated by WorldWriter.
        self._state  = WorldState()
        self._ww     = WorldWriter(self._state)
        self._wq     = WorldQuery(self._state)

        # Per-goal tracking
        self._active_goal: Optional[Goal] = None
        self._goal_start_tick: int = 0
        self._stuck_count: int = 0

        self._stats = LoopStats()
        self._running = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> LoopStats:
        """
        Run the loop until the GoalSource is exhausted or stop() is called.

        Returns a LoopStats summary of the run.
        """
        log.info("FactorioLoop: starting")
        self._running = True

        try:
            self._startup()
            while self._running:
                self._tick()
        except KeyboardInterrupt:
            log.info("FactorioLoop: interrupted by user")
        except Exception:
            log.exception("FactorioLoop: unhandled exception — shutting down")
            raise
        finally:
            self._shutdown()

        log.info(
            "FactorioLoop: done. goals=%d completed=%d failed=%d stuck_events=%d",
            self._stats.goals_attempted,
            self._stats.goals_completed,
            self._stats.goals_failed,
            self._stats.stuck_events,
        )
        return self._stats

    def stop(self) -> None:
        """Signal the loop to exit cleanly after the current tick."""
        self._running = False

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    def _startup(self) -> None:
        """
        Perform one initial world state poll and request the first goal.
        """
        log.info("FactorioLoop: performing initial world state poll")
        self._poll_world()
        self._request_next_goal()

    def _shutdown(self) -> None:
        """Flush behavioral memory and close the RCON connection."""
        try:
            self._mem.close()
            log.info("FactorioLoop: behavioral memory flushed")
        except Exception:
            log.exception("FactorioLoop: error closing behavioral memory")
        try:
            self._client.close()
            log.info("FactorioLoop: RCON connection closed")
        except Exception:
            log.exception("FactorioLoop: error closing RCON client")

    # ------------------------------------------------------------------
    # Per-tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """One poll cycle — the innermost loop body."""
        # 1. Poll Factorio and integrate snapshot.
        self._poll_world()
        tick = self._state.tick
        self._stats.total_ticks += 1

        # 2. If no active goal, we're done.
        if self._active_goal is None:
            if self._cfg.shutdown_on_empty_queue:
                log.info("FactorioLoop: no active goal and queue exhausted — stopping")
                self._running = False
            return

        goal = self._active_goal

        # 3. Evaluate goal conditions.
        eval_result = self._evaluator.evaluate(
            goal, self._wq, tick, self._goal_start_tick
        )

        if eval_result.success:
            self._on_goal_complete(goal, eval_result.reward, tick)
            return

        if eval_result.failure:
            self._on_goal_failed(goal, eval_result.reward, tick)
            return

        # 4. Tick coordinator.
        exec_result = self._coordinator.tick(goal, self._wq, self._ww, tick)

        # 5. Dispatch actions.
        for action in exec_result.actions:
            try:
                self._executor.execute(action)
            except Exception:
                log.exception(
                    "FactorioLoop: action executor raised on %s", action.kind
                )

        # 6. Handle execution status.
        if exec_result.status == ExecutionStatus.STUCK:
            self._on_stuck(goal, exec_result.stuck_context, tick)

        # 7. Sleep until next poll.
        time.sleep(self._cfg.tick_interval / 60.0)

    def _poll_world(self) -> None:
        """Poll Factorio for a world state snapshot and integrate it."""
        raw = self._poller.poll()
        if raw:
            snapshot = self._parser.parse(raw, self._state.tick)
            self._ww.integrate_snapshot(snapshot)

    # ------------------------------------------------------------------
    # Goal lifecycle
    # ------------------------------------------------------------------

    def _request_next_goal(self) -> None:
        """
        Ask the GoalSource for the next goal and reset the coordinator.

        If the source returns None, the queue is exhausted and the loop
        will exit on the next tick.
        """
        context = self._build_context()
        goal = self._goal_source.next_goal(context)

        if goal is None:
            self._active_goal = None
            return

        goal.activate(tick=self._state.tick)
        self._active_goal = goal
        self._goal_start_tick = self._state.tick
        self._stuck_count = 0
        self._stats.goals_attempted += 1

        self._coordinator.reset(goal, self._wq)
        log.info(
            "FactorioLoop: goal activated — %s [%s]",
            goal.description, getattr(goal, "type", "?"),
        )

    def _on_goal_complete(self, goal: Goal, reward: float, tick: int) -> None:
        """Handle a successfully completed goal."""
        goal.complete(tick)
        self._stats.goals_completed += 1
        log.info(
            "FactorioLoop: goal COMPLETE — %s (reward=%.3f, ticks=%s)",
            goal.description, reward, goal.ticks_elapsed,
        )
        self._record_outcome(goal, reward, success=True)
        self._goal_source.record_outcome(goal, reward)
        self._request_next_goal()

    def _on_goal_failed(self, goal: Goal, reward: float, tick: int) -> None:
        """Handle a goal whose failure condition was met."""
        goal.fail(tick)
        self._stats.goals_failed += 1
        log.warning(
            "FactorioLoop: goal FAILED — %s (reward=%.3f, ticks=%s)",
            goal.description, reward, goal.ticks_elapsed,
        )
        self._record_outcome(goal, reward, success=False)
        self._goal_source.record_outcome(goal, reward)
        self._request_next_goal()

    def _on_stuck(self, goal: Goal, stuck_context, tick: int) -> None:
        """
        Handle STUCK from the execution network.

        Passes the StuckContext to the GoalSource. If it returns seed
        subtasks, reset the coordinator with them. If it returns [] (GoalQueue
        stub), increment the stuck counter and let the failure condition
        handle it naturally.
        """
        self._stats.stuck_events += 1
        self._stuck_count += 1

        log.warning(
            "FactorioLoop: STUCK on goal %s (event %d/%d) — %s",
            goal.id[:8],
            self._stuck_count,
            self._cfg.max_stuck_retries,
            (
                stuck_context.immediate_failure.description
                if stuck_context and stuck_context.immediate_failure
                else "goal level"
            ),
        )

        seed_subtasks = self._goal_source.handle_stuck(stuck_context)

        if seed_subtasks:
            log.info(
                "FactorioLoop: GoalSource provided %d seed subtasks — resetting coordinator",
                len(seed_subtasks),
            )
            self._coordinator.reset(goal, self._wq, seed_subtasks=seed_subtasks)
        elif self._stuck_count >= self._cfg.max_stuck_retries:
            log.warning(
                "FactorioLoop: max stuck retries reached — failing goal %s",
                goal.id[:8],
            )
            self._on_goal_failed(goal, 0.0, tick)

    def _record_outcome(self, goal: Goal, reward: float, success: bool) -> None:
        """Write the outcome to behavioral memory."""
        from agent.memory.behavioral import GoalOutcome
        try:
            self._mem.record_outcome(
                goal_type=getattr(goal, "type", "unknown"),
                context_summary=self._build_context(),
                outcome=GoalOutcome(
                    success=success,
                    reward=reward,
                    ticks_elapsed=goal.ticks_elapsed or 0,
                ),
                ticks_elapsed=goal.ticks_elapsed or 0,
            )
        except Exception:
            log.exception("FactorioLoop: failed to record outcome to behavioral memory")

    def _build_context(self) -> dict:
        """
        Build a lightweight world-state summary for the GoalSource.

        In Phase 11 the examination layer will produce a richer structured
        summary from the self-model. For now this is a minimal snapshot of
        the most useful planning signals.
        """
        return {
            "tick": self._state.tick,
            "charted_chunks": self._wq.charted_chunks,
            "inventory": {
                slot.item: slot.count
                for slot in self._state.player.inventory.slots
            },
            "resources_known": [
                {
                    "type": patch.resource_type,
                    "position": {
                        "x": patch.position.x,
                        "y": patch.position.y,
                    },
                    "amount": patch.amount,
                }
                for patch in self._state.resource_map
            ],
            "research_unlocked": list(self._wq.research.unlocked),
        }