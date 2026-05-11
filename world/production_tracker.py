"""
world/production_tracker.py

ProductionTracker — tracks item throughput over time by observing successive
WorldQuery snapshots.

Rules:
- No RCON calls. No LLM calls. Does not mutate WorldState.
- Stateful across ticks; intended to be called once per tick cycle.
- Primary signal: inserter_activity counts from WorldQuery.logistics.inserter_activity
  (items moved by inserters since last observation).
- Fallback signal: inventory delta on output chests (less reliable).
- Handles gaps gracefully — missing ticks are not fabricated.
- Nothing in this module imports from bridge/, agent/, planning/, llm/, or memory/.

Condition scope:
  production_rate() is a PROXIMAL condition — it only reflects entities within
  the current LOCAL_SCAN_RADIUS.  See CONDITION_SCOPE.md.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable, TYPE_CHECKING

from world.state import EntityStatus

if TYPE_CHECKING:
    from world.query import WorldQuery


# ---------------------------------------------------------------------------
# Protocol — the interface the RewardEvaluator depends on
# ---------------------------------------------------------------------------

@runtime_checkable
class ProductionTrackerProtocol(Protocol):
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
    rates: dict[str, float]
    stalled_items: list[str]
    top_producers: list[tuple[str, float]]
    window_ticks: int
    tick_start: int
    tick_end: int


# ---------------------------------------------------------------------------
# Internal snapshot record
# ---------------------------------------------------------------------------

@dataclass
class _Snapshot:
    tick: int
    cumulative: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ProductionTracker
# ---------------------------------------------------------------------------

class ProductionTracker:
    """
    Tracks item production rates over configurable rolling windows.

    Call update(wq) once per WorldQuery refresh.
    Query rate(), rates_all(), is_stalled(), or summary() at any time.

    IMPORTANT — scan-radius limitation (PROXIMAL condition):
      Only entities within the bridge's current LOCAL_SCAN_RADIUS contribute
      to rate measurements each tick.  See CONDITION_SCOPE.md.
    """

    _MAX_HISTORY = 600  # ~10 minutes at 1 snapshot/second

    def __init__(self) -> None:
        self._history: list[_Snapshot] = []
        self._cumulative: dict[str, int] = defaultdict(int)
        self._last_inserter: dict[int, int] = {}
        self._last_inventories: dict[int, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, wq: "WorldQuery") -> None:
        """
        Ingest a new WorldQuery snapshot.  Called once per tick cycle.

        Parameters
        ----------
        wq : WorldQuery
            The current world query interface. All reads go through this.
        """
        tick = wq.tick
        delta: dict[str, int] = defaultdict(int)

        # --- Primary signal: inserter activity ---
        inserter_activity = wq.logistics.inserter_activity
        if inserter_activity:
            for entity_id, count in inserter_activity.items():
                prev = self._last_inserter.get(entity_id, count)
                moved = max(0, count - prev)
                if moved > 0:
                    delta["__inserter_moves__"] += moved
                self._last_inserter[entity_id] = count

        # --- Secondary signal: entity output inventories ---
        for entity in wq.all_entities():
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

        for item, amount in delta.items():
            self._cumulative[item] += amount

        snap_cumulative = {k: v for k, v in self._cumulative.items()
                           if k != "__inserter_moves__"}
        self._history.append(_Snapshot(tick=tick, cumulative=dict(snap_cumulative)))

        if len(self._history) > self._MAX_HISTORY:
            self._history = self._history[-self._MAX_HISTORY:]

    def rate(self, item: str, window_ticks: int = 3600) -> float:
        """Items per minute for the given item over the last window_ticks."""
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
        elapsed_minutes = elapsed_ticks / 3600.0
        return delta / elapsed_minutes

    def rates_all(self, window_ticks: int = 3600) -> dict[str, float]:
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
        if self.rate(item, window_ticks) > 0:
            return False
        for snap in self._history:
            if snap.cumulative.get(item, 0) > 0:
                return True
        return False

    def is_stalled_with_query(self, item: str, wq: "WorldQuery",
                               window_ticks: int = 600) -> bool:
        """
        Preferred variant: checks for a live producer via WorldQuery.
        True if rate is zero but at least one non-idle entity exists.
        """
        if self.rate(item, window_ticks) > 0:
            return False
        active_statuses = {EntityStatus.WORKING, EntityStatus.NO_INPUT,
                           EntityStatus.FULL_OUT}
        return any(e.status in active_statuses for e in wq.all_entities())

    def summary(self, window_ticks: int = 3600) -> ProductionSummary:
        rates = self.rates_all(window_ticks)
        stalled: list[str] = []
        if len(self._history) >= 2:
            latest = self._history[-1]
            earliest = self._find_window_start(latest.tick, window_ticks)
            if earliest is not None and earliest is not latest:
                elapsed_ticks = latest.tick - earliest.tick
                if elapsed_ticks > 0:
                    for item in earliest.cumulative:
                        if latest.cumulative.get(item, 0) <= earliest.cumulative.get(item, 0):
                            if earliest.cumulative[item] > 0:
                                stalled.append(item)

        top_producers = sorted(rates.items(), key=lambda x: x[1], reverse=True)[:10]
        tick_start = self._history[0].tick if self._history else 0
        tick_end   = self._history[-1].tick if self._history else 0

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
        if len(self._history) < 2:
            return None
        cutoff = latest_tick - window_ticks
        for snap in self._history:
            if snap.tick >= cutoff:
                return snap
        return self._history[0]