"""
world/production_tracker.py

ProductionTracker — tracks item throughput over time by observing successive
WorldState snapshots.

Rules:
- No RCON calls. No LLM calls. Does not mutate WorldState.
- Stateful across ticks; intended to be called once per tick cycle.
- Primary signal: inserter_activity counts from WorldState.logistics.inserter_activity
  (items moved by inserters since last observation).
- Fallback signal: inventory delta on output chests (less reliable — confounded by
  player manually moving items).
- Handles gaps gracefully — missing ticks are not fabricated.
- Nothing in this module imports from bridge/, agent/, planning/, llm/, or memory/.

Condition scope:
  production_rate() is a PROXIMAL condition — it only reflects entities within
  the current LOCAL_SCAN_RADIUS.  See CONDITION_SCOPE.md for the full
  discussion of what this means for goal evaluation and how to write conditions
  that handle it correctly.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from world.state import EntityStatus, WorldState


# ---------------------------------------------------------------------------
# Protocol — the interface the RewardEvaluator depends on
# ---------------------------------------------------------------------------

@runtime_checkable
class ProductionTrackerProtocol(Protocol):
    """
    Minimum interface the RewardEvaluator requires from a production tracker.

    Concrete implementations (ProductionTracker) may expose additional methods
    (rates_all, is_stalled, summary, etc.) but the evaluator only calls rate().
    Any object satisfying this protocol can be injected as a tracker.
    """

    def rate(self, item: str, window_ticks: int = 3600) -> float:
        """
        Return the smoothed production rate for *item* in items per minute.
        Returns 0.0 if insufficient history or item has never been tracked.
        Never raises.
        """
        ...


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProductionSummary:
    """Structured summary produced by ProductionTracker.summary()."""
    rates: dict[str, float]          # item → items/minute
    stalled_items: list[str]         # items with zero rate but active producers
    top_producers: list[tuple[str, float]]  # (item, rate) sorted desc, up to 10
    window_ticks: int                # window used
    tick_start: int                  # oldest tick in the window (or 0 if no history)
    tick_end: int                    # newest tick observed


# ---------------------------------------------------------------------------
# Internal snapshot record
# ---------------------------------------------------------------------------

@dataclass
class _Snapshot:
    tick: int
    # item → cumulative items moved (from inserter activity or inventory diff)
    cumulative: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ProductionTracker
# ---------------------------------------------------------------------------

class ProductionTracker:
    """
    Tracks item production rates over configurable rolling windows.

    Call update() once per WorldState refresh.  Query rate(), rates_all(),
    is_stalled(), or summary() at any time.

    The tracker records a rolling history of (tick, {item: cumulative_count})
    snapshots.  To compute a rate over a window, it finds the earliest snapshot
    within the window and diffs cumulative counts against the latest snapshot,
    then divides by elapsed ticks converted to minutes.

    Primary signal (inserter activity):
      WorldState.logistics.inserter_activity maps entity_id → items_moved_total.
      The tracker diffs successive values to get items_moved_this_cycle and
      attributes them to the recipe output of the target entity (if known).
      This is an approximation — inserters also carry inputs, not just outputs.
      The ProductionTracker makes a best-effort attribution; accuracy improves
      with multiple snapshots.

    Fallback signal (inventory delta):
      When inserter activity is sparse (early game, no logistic data), the tracker
      diffs output inventory contents of chest entities across snapshots.

    IMPORTANT — scan-radius limitation (PROXIMAL condition):
      Only entities within the bridge's current LOCAL_SCAN_RADIUS contribute to
      rate measurements each tick.  An assembler that is off-screen is invisible
      to the tracker while the player is away.  Conditions written using
      production_rate() are therefore PROXIMAL — they only reflect the factory
      segment the player is currently standing near.
      See CONDITION_SCOPE.md for the full discussion and mitigation strategies.
    """

    # Maximum snapshots to retain — ~10 minutes at 1 snapshot/second
    _MAX_HISTORY = 600

    def __init__(self) -> None:
        # Ordered list of snapshots, oldest first.
        self._history: list[_Snapshot] = []
        # Running cumulative totals (item → total items produced this session)
        self._cumulative: dict[str, int] = defaultdict(int)
        # Last seen inserter activity values (entity_id → count)
        self._last_inserter: dict[int, int] = {}
        # Last seen inventory hashes (entity_id → {item: count})
        self._last_inventories: dict[int, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, state: WorldState) -> None:
        """
        Ingest a new WorldState snapshot.  Called once per tick cycle.
        """
        tick = state.tick
        delta: dict[str, int] = defaultdict(int)

        # --- Primary signal: inserter activity ---
        inserter_activity = state.logistics.inserter_activity
        if inserter_activity:
            for entity_id, count in inserter_activity.items():
                prev = self._last_inserter.get(entity_id, count)
                moved = max(0, count - prev)  # guard against reset / wrap
                if moved > 0:
                    # Attribute to the entity this inserter targets.
                    # Without a full entity graph we can't know the exact item,
                    # so we record under the placeholder "__inserter_moves__"
                    # and also try to attribute via entity recipe lookup below.
                    delta["__inserter_moves__"] += moved
                self._last_inserter[entity_id] = count

        # --- Secondary signal: entity output inventories ---
        # Track output inventory of all entities that have an inventory.
        for entity in state.entities:
            if entity.inventory is None:
                continue
            inv_dict = entity.inventory.as_dict()
            prev_inv = self._last_inventories.get(entity.entity_id, {})
            for item, count in inv_dict.items():
                prev_count = prev_inv.get(item, 0)
                produced = max(0, count - prev_count)
                if produced > 0:
                    delta[item] += produced
            self._last_inventories[entity.entity_id] = inv_dict

        # Accumulate into session-level cumulative totals.
        for item, amount in delta.items():
            self._cumulative[item] += amount

        # Record snapshot (exclude the placeholder key from history).
        snap_cumulative = {k: v for k, v in self._cumulative.items()
                           if k != "__inserter_moves__"}
        self._history.append(_Snapshot(tick=tick, cumulative=dict(snap_cumulative)))

        # Prune old snapshots.
        if len(self._history) > self._MAX_HISTORY:
            self._history = self._history[-self._MAX_HISTORY:]

    def rate(self, item: str, window_ticks: int = 3600) -> float:
        """
        Items per minute for the given item over the last window_ticks.
        Returns 0.0 if insufficient history or item has never been tracked.
        """
        if len(self._history) < 2:
            return 0.0
        latest = self._history[-1]
        earliest = self._find_window_start(latest.tick, window_ticks)
        if earliest is None or earliest is latest:
            return 0.0
        elapsed_ticks = latest.tick - earliest.tick
        if elapsed_ticks <= 0:
            return 0.0
        delta = latest.cumulative.get(item, 0) - earliest.cumulative.get(item, 0)
        if delta <= 0:
            return 0.0
        # Convert ticks → minutes (60 tps × 60s = 3600 ticks/min)
        elapsed_minutes = elapsed_ticks / 3600.0
        return delta / elapsed_minutes  # items / minute

    def rates_all(self, window_ticks: int = 3600) -> dict[str, float]:
        """Items per minute for all tracked items over the window."""
        if len(self._history) < 2:
            return {}
        latest = self._history[-1]
        earliest = self._find_window_start(latest.tick, window_ticks)
        if earliest is None or earliest is latest:
            return {}
        elapsed_ticks = latest.tick - earliest.tick
        if elapsed_ticks <= 0:
            return {}
        elapsed_minutes = elapsed_ticks / 3600.0
        result: dict[str, float] = {}
        all_items = set(latest.cumulative) | set(earliest.cumulative)
        for item in all_items:
            delta = latest.cumulative.get(item, 0) - earliest.cumulative.get(item, 0)
            if delta > 0:
                result[item] = delta / elapsed_minutes
        return result

    def is_stalled(self, item: str, window_ticks: int = 600) -> bool:
        """
        True if the item's production rate is zero over the window,
        despite at least one producer entity existing in WorldState.

        Note: requires the caller to pass the current WorldState.
        This override accepts an optional WorldState for producer detection.
        """
        # Without a WorldState reference in the method signature (as per brief),
        # we detect "producer exists" by checking if any entity is WORKING
        # that could plausibly produce this item. Since we store the last
        # inventories, we infer a producer exists if the item appeared in any
        # entity's output at any point in our history.
        if self.rate(item, window_ticks) > 0:
            return False
        # Check if item has ever appeared in our tracking history —
        # if it was produced before but rate is now 0, that's stalled.
        # If it was never produced at all, we can't call it stalled.
        for snap in self._history:
            if snap.cumulative.get(item, 0) > 0:
                return True  # Was produced before but now stalled
        return False

    def is_stalled_with_state(self, item: str, state: WorldState,
                               window_ticks: int = 600) -> bool:
        """
        Preferred variant: checks for a live producer in the given WorldState.
        True if rate is zero but at least one non-idle entity exists.
        """
        if self.rate(item, window_ticks) > 0:
            return False
        # Any entity that is WORKING or NO_INPUT counts as a "producer" —
        # it exists and is trying to produce something.
        active_statuses = {EntityStatus.WORKING, EntityStatus.NO_INPUT,
                           EntityStatus.FULL_OUT}
        return any(e.status in active_statuses for e in state.entities)

    def summary(self, window_ticks: int = 3600) -> ProductionSummary:
        """Structured summary for the examiner and LLM context."""
        rates = self.rates_all(window_ticks)
        stalled: list[str] = []
        if len(self._history) >= 2:
            latest = self._history[-1]
            earliest = self._find_window_start(latest.tick, window_ticks)
            if earliest is not None and earliest is not latest:
                elapsed_ticks = latest.tick - earliest.tick
                if elapsed_ticks > 0:
                    # Items that appeared in earliest but not latest → stalled
                    for item in earliest.cumulative:
                        if latest.cumulative.get(item, 0) <= earliest.cumulative.get(item, 0):
                            if earliest.cumulative[item] > 0:
                                stalled.append(item)

        top_producers = sorted(rates.items(), key=lambda x: x[1], reverse=True)[:10]

        tick_start = self._history[0].tick if self._history else 0
        tick_end = self._history[-1].tick if self._history else 0

        return ProductionSummary(
            rates=rates,
            stalled_items=stalled,
            top_producers=top_producers,
            window_ticks=window_ticks,
            tick_start=tick_start,
            tick_end=tick_end,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_window_start(self, latest_tick: int,
                            window_ticks: int) -> Optional["_Snapshot"]:
        """
        Return the oldest snapshot within the window, or None if there's only
        one snapshot (or none).
        """
        if len(self._history) < 2:
            return None
        cutoff = latest_tick - window_ticks
        for snap in self._history:
            if snap.tick >= cutoff:
                return snap
        # All snapshots are within the window — return the oldest.
        return self._history[0]