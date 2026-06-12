"""
planning/evaluation/condition_namespace.py

Shared condition evaluation namespace for both RewardEvaluator and the
coordinator's subtask condition evaluator.

Design
------
Goal conditions (evaluated by RewardEvaluator) and subtask conditions
(evaluated by the coordinator) share a large common core of namespace
entries: inventory, charted_chunks, tick, elapsed_ticks, new, wq, state,
research, resources_of_type, and the safe builtins.

Each context adds its own extras on top:
  RewardEvaluator adds: production_rate, staleness, logistics, power,
                        threat, inserters_from/to, entities, entity_by_id
  Coordinator adds:     is_at, is_reachable, Position

Keeping the shared core here means:
  - Adding a new world-state concept touches one place
  - Both contexts get it automatically
  - The synchronization failure that motivated this refactor cannot recur
    for core entries — only context-specific extras can drift
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from world import WorldQuery

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BBoxQuery re-exported from world.observable.query where it is defined.
# Importing here makes it available to callers of build_core_namespace
# without them needing to know where BBoxQuery lives.
# ---------------------------------------------------------------------------
from world.observable.query import BBoxQuery  # noqa: F401 (re-export)

BLOCKED_NAMES = frozenset({
    "os", "sys", "subprocess", "importlib", "builtins",
    "open", "exec", "compile", "__import__",
})


class _DeltaView:
    """
    Provides delta access to world state relative to a goal-activation snapshot.

    Exposed in condition strings as the ``new`` namespace name::

        new.tick >= 1200                    # ticks since goal activation
        new.charted_chunks >= 5            # chunks revealed this goal
        new.charted_tiles >= 5120          # auto via __getattr__
        new.inventory('iron-ore') >= 10    # ore collected this goal
        new.resource_patches('coal') >= 1  # patches found this goal

    The snapshot is a shallow copy of WorldState taken at goal activation.
    Since WorldWriter replaces whole sub-objects (never mutates them in place),
    the snapshot's sub-objects remain stable after it is taken.

    Scalar WorldQuery properties get automatic delta support via __getattr__ —
    adding a new property to WorldQuery requires no changes here. Only
    parameterised lookups (inventory, resource_patches) need explicit methods.
    """

    def __init__(
        self,
        wq: "WorldQuery",
        start_wq: "WorldQuery",
        elapsed_ticks: int = 0,
    ) -> None:
        object.__setattr__(self, '_wq', wq)
        object.__setattr__(self, '_start_wq', start_wq)
        object.__setattr__(self, '_elapsed_ticks', elapsed_ticks)

    def __getattr__(self, name: str):
        """
        Automatic delta for scalar WorldQuery properties.
        new.charted_chunks → wq.charted_chunks - start_wq.charted_chunks
        Clamped to 0. Raises AttributeError if wq doesn't have the property.
        """
        wq       = object.__getattribute__(self, '_wq')
        start_wq = object.__getattribute__(self, '_start_wq')
        try:
            current = getattr(wq, name)
        except AttributeError:
            raise AttributeError(f"_DeltaView: no property {name!r} on WorldQuery")
        if callable(current):
            raise AttributeError(
                f"_DeltaView: {name!r} is callable — use an explicit method instead"
            )
        try:
            baseline = getattr(start_wq, name)
            return max(0, current - baseline)
        except (TypeError, AttributeError):
            return current  # non-numeric or missing on start_wq — return raw

    @property
    def tick(self) -> int:
        """Ticks elapsed since goal activation (current_tick - start_tick).
        Mirrors the top-level `tick` name: new.tick is the delta of tick.
        """
        return object.__getattribute__(self, '_elapsed_ticks')

    def inventory(self, item: str) -> int:
        """Net items of *item* collected since goal activation (clamped to 0)."""
        wq       = object.__getattribute__(self, '_wq')
        start_wq = object.__getattribute__(self, '_start_wq')
        return max(0, wq.inventory_count(item) - start_wq.inventory_count(item))

    def resource_patches(self, resource_type: str) -> int:
        """Resource patches of *resource_type* discovered since goal activation."""
        wq       = object.__getattribute__(self, '_wq')
        start_wq = object.__getattribute__(self, '_start_wq')
        return max(0, len(wq.resources_of_type(resource_type))
                      - len(start_wq.resources_of_type(resource_type)))

# ---------------------------------------------------------------------------
# WorldQuery
# ---------------------------------------------------------------------------


def build_core_namespace(
    wq: "WorldQuery",
    tick: int,
    start_tick: int = 0,
    start_wq: Optional["WorldQuery"] = None,
) -> dict:
    """
    Build the shared core namespace used by all condition evaluators.

    Parameters
    ----------
    wq            : WorldQuery for the current world state.
    tick          : Current game tick.
    start_tick    : Game tick when the current goal was activated.
    start_wq: World-state snapshot captured at goal activation,
                    used to populate the `new` delta object.

    Returns a dict suitable for passing to eval(). Does NOT include
    ``__builtins__`` — callers are responsible for setting that.

    Context-specific entries (production_rate, staleness, is_at, etc.)
    are added by the caller after receiving this dict.
    """
    elapsed_ticks = max(0, tick - start_tick)
    # If no start_wq provided, use wq itself as baseline → all deltas = 0
    effective_start_wq = start_wq if start_wq is not None else wq

    return {
        # ----------------------------------------------------------------
        # WorldQuery and raw state — available everywhere
        # ----------------------------------------------------------------
        "wq":    wq,
        "state": wq.state,

        # ----------------------------------------------------------------
        # Time — NON-PROXIMAL
        # ----------------------------------------------------------------
        "tick":         tick,
        # elapsed_ticks: ticks since goal activation.
        # Alias for new.tick; kept as a top-level name for readability.
        # Prefer new.tick for consistency with the delta framework.
        "elapsed_ticks": elapsed_ticks,

        # ----------------------------------------------------------------
        # Delta object — NON-PROXIMAL
        # new.tick            → elapsed_ticks (delta of tick)
        # new.charted_chunks  → chunks revealed this goal
        # new.charted_tiles   → auto via __getattr__
        # new.inventory(item) → items collected this goal
        # new.resource_patches(type) → patches found this goal
        # ----------------------------------------------------------------
        "new": _DeltaView(wq, effective_start_wq, elapsed_ticks),

        # ----------------------------------------------------------------
        # Inventory — NON-PROXIMAL
        # ----------------------------------------------------------------
        "inventory": wq.inventory_count,

        # ----------------------------------------------------------------
        # Exploration — NON-PROXIMAL
        # ----------------------------------------------------------------
        "charted_chunks":   wq.charted_chunks,
        "charted_tiles":    wq.charted_tiles,
        "charted_area_km2": wq.charted_area_km2,

        # ----------------------------------------------------------------
        # Research — NON-PROXIMAL
        # ----------------------------------------------------------------
        "research":      wq.research,
        "tech_unlocked": wq.tech_unlocked,

        # ----------------------------------------------------------------
        # Resource patches — NON-PROXIMAL
        # ----------------------------------------------------------------
        "resources_of_type": wq.resources_of_type,

        # ----------------------------------------------------------------
        # Entities — PROXIMAL (scan radius only)
        # ----------------------------------------------------------------
        "entities":           wq.entities_by_name,
        "entity_by_id":       wq.entity_by_id,
        "entities_by_status": wq.entities_by_status,

        # ----------------------------------------------------------------
        # Navigation helpers
        # ----------------------------------------------------------------
        # navigate_to(x, y) — True when player is within 1.5 tiles of target.
        # Used in task success_conditions for navigate goal type.
        "navigate_to": lambda x, y: (
            abs(wq.player_position().x - float(x)) < 1.5
            and abs(wq.player_position().y - float(y)) < 1.5
        ),

        # ----------------------------------------------------------------
        # Spatial queries — PROXIMAL (scan radius only)
        # ----------------------------------------------------------------
        # bbox(x_min, y_min, x_max, y_max) -> BBoxQuery
        # Use for clear_region conditions and region-based success checks:
        #   bbox(-16,-16,16,16).is_clear
        #   bbox(-16,-16,16,16).natural_count == 0
        #
        # The bbox() function is the bridge between condition strings and
        # WorldQuery.natural_objects_in_bbox(). condition_parser extracts the
        # coordinates to pass as coordinator params; the evaluator resolves
        # bbox() as a callable at eval() time.
        "bbox": lambda x_min, y_min, x_max, y_max: BBoxQuery(
            wq, float(x_min), float(y_min), float(x_max), float(y_max)
        ),

        # ----------------------------------------------------------------
        # Safe builtins subset
        # ----------------------------------------------------------------
        "True":  True,
        "False": False,
        "len":   len,
        "any":   any,
        "all":   all,
        "sum":   sum,
        "min":   min,
        "max":   max,
        "abs":   abs,
        "int":   int,
        "float": float,
        "str":   str,
        "bool":  bool,
        "list":  list,
        "dict":  dict,
        "set":   set,
        "range": range,
        "round": round,
    }


def build_full_namespace(
    wq: "WorldQuery",
    tick: int,
    start_tick: int = 0,
    start_wq: Optional["WorldQuery"] = None,
    tracker=None,
) -> dict:
    """
    Build the complete namespace for RewardEvaluator conditions.

    Includes everything from build_core_namespace plus the richer
    world-state access that goal conditions may use: production_rate,
    staleness, logistics, power, threat, and inserter queries.

    This is the single authoritative source for all eval() namespaces.
    The coordinator's task condition evaluator uses build_core_namespace
    (no tracker, no logistics); the RewardEvaluator uses this function.

    Parameters
    ----------
    wq          : WorldQuery for the current world state.
    tick        : Current game tick.
    start_tick  : Game tick when the current goal was activated.
    start_wq    : Snapshot at goal activation for delta conditions.
    tracker     : Optional ProductionTrackerProtocol for production_rate().
                  When None, production_rate() always returns 0.0.
    """
    ns = build_core_namespace(wq, tick, start_tick, start_wq)
    ns["__builtins__"] = safe_builtins()

    if tracker is not None:
        production_rate = tracker.rate
    else:
        import logging as _logging
        _prl = _logging.getLogger(__name__)
        def production_rate(item: str) -> float:  # type: ignore[misc]
            _prl.warning(
                "production_rate(%r) called but no ProductionTracker attached "
                "— returning 0.0. See CONDITION_SCOPE.md.", item
            )
            return 0.0

    def staleness(section: str):
        return wq.section_staleness(section, tick)

    ns.update({
        # Production rates — PROXIMAL (scan radius + tracker window)
        "production_rate": production_rate,
        # Staleness guard — META
        "staleness": staleness,
        # Logistics sub-objects — PROXIMAL
        "logistics": wq.logistics,
        "power":     wq.power,
        "threat":    wq.threat,
        # Connectivity queries — PROXIMAL
        "inserters_from":      wq.inserters_taking_from,
        "inserters_to":        wq.inserters_delivering_to,
        "inserters_from_type": wq.inserters_taking_from_type,
        "inserters_to_type":   wq.inserters_delivering_to_type,
    })
    return ns


def safe_builtins() -> dict:
    """
    Return a filtered ``__builtins__`` dict that excludes dangerous names.
    Pass as ``ns["__builtins__"]`` when calling eval().
    """
    import builtins as _builtins
    raw = vars(_builtins)
    return {k: v for k, v in raw.items() if k not in BLOCKED_NAMES}