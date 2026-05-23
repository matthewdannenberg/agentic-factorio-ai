"""
planning/condition_namespace.py

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
    from world.query import WorldQuery

log = logging.getLogger(__name__)

# Names that must never appear in eval() namespaces.
BLOCKED_NAMES = frozenset({
    "os", "sys", "subprocess", "importlib", "builtins",
    "open", "exec", "compile", "__import__",
})


class _DeltaView:
    """
    Provides delta access to world state relative to a goal-activation snapshot.

    Exposed in condition strings as the ``new`` namespace name:

        new.charted_chunks >= 5          # scalar property delta via __getattr__
        new.charted_tiles >= 5120        # any wq scalar property works automatically
        new.inventory('iron-ore') >= 10  # parameterized — explicit method
        new.resource_patches('coal') >= 1  # parameterized — explicit method
        new.elapsed_ticks > 1200         # ticks since goal activation

    Scalar wq properties get automatic delta support via __getattr__ — adding
    a new property to WorldQuery requires no changes here. Only parameterized
    lookups (inventory, resource_patches) need explicit methods.
    """

    def __init__(self, wq: "WorldQuery", snapshot: dict, elapsed_ticks: int = 0) -> None:
        # Use object.__setattr__ to avoid triggering our own __getattr__
        object.__setattr__(self, '_wq', wq)
        object.__setattr__(self, '_snapshot', snapshot)
        object.__setattr__(self, '_elapsed_ticks', elapsed_ticks)

    def __getattr__(self, name: str):
        """
        Automatic delta for scalar WorldQuery properties.
        new.charted_chunks → wq.charted_chunks - snapshot['charted_chunks']
        Clamped to 0. Raises AttributeError if the wq property doesn't exist.
        """
        wq       = object.__getattribute__(self, '_wq')
        snapshot = object.__getattribute__(self, '_snapshot')
        try:
            current = getattr(wq, name)
        except AttributeError:
            raise AttributeError(f"_DeltaView: no property {name!r} on WorldQuery")
        if callable(current):
            raise AttributeError(
                f"_DeltaView: {name!r} is callable — use an explicit method instead"
            )
        baseline = snapshot.get(name, current)
        try:
            return max(0, current - baseline)
        except TypeError:
            # Non-numeric property — return raw value, delta doesn't apply
            return current

    @property
    def tick(self) -> int:
        """Ticks elapsed since goal activation (current_tick - start_tick).
        Mirrors the top-level `tick` name: new.tick is the delta of tick.
        """
        return object.__getattribute__(self, '_elapsed_ticks')

    def inventory(self, item: str) -> int:
        """Net items of *item* collected since goal activation (clamped to 0)."""
        wq       = object.__getattribute__(self, '_wq')
        snapshot = object.__getattribute__(self, '_snapshot')
        snap_inv = snapshot.get("inventory", {})
        return max(0, wq.inventory_count(item)
                   - snap_inv.get(item, wq.inventory_count(item)))

    def resource_patches(self, resource_type: str) -> int:
        """Resource patches of *resource_type* discovered since goal activation."""
        wq       = object.__getattribute__(self, '_wq')
        snapshot = object.__getattribute__(self, '_snapshot')
        snap_rc  = snapshot.get("resource_counts", {})
        current  = len(wq.resources_of_type(resource_type))
        return max(0, current - snap_rc.get(resource_type, current))



def build_core_namespace(
    wq: "WorldQuery",
    tick: int,
    start_tick: int = 0,
    start_snapshot: Optional[dict] = None,
) -> dict:
    """
    Build the shared core namespace used by all condition evaluators.

    Parameters
    ----------
    wq            : WorldQuery for the current world state.
    tick          : Current game tick.
    start_tick    : Game tick when the current goal was activated.
    start_snapshot: World-state snapshot captured at goal activation,
                    used to populate the `new` delta object.

    Returns a dict suitable for passing to eval(). Does NOT include
    ``__builtins__`` — callers are responsible for setting that.

    Context-specific entries (production_rate, staleness, is_at, etc.)
    are added by the caller after receiving this dict.
    """
    elapsed_ticks = max(0, tick - start_tick)
    snapshot = start_snapshot or {}

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
        "new": _DeltaView(wq, snapshot, elapsed_ticks),

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


def safe_builtins() -> dict:
    """
    Return a filtered ``__builtins__`` dict that excludes dangerous names.
    Pass as ``ns["__builtins__"]`` when calling eval().
    """
    import builtins as _builtins
    raw = vars(_builtins)
    return {k: v for k, v in raw.items() if k not in BLOCKED_NAMES}