"""
world/query.py

WorldQuery — the sole read interface for WorldState.

Rules:
- No LLM calls. No RCON. No mutations to WorldState.
- Every file that reads game state must do so through a WorldQuery instance.
- WorldQuery holds a reference to the shared WorldState object; it never
  copies or owns the data.
- WorldQuery also holds an optional KnowledgeBase reference so that queries
  involving entity tile dimensions (spatial joins for large entities) can
  resolve those dimensions without the caller needing to do it.

Construction
------------
    state  = WorldState(...)          # bridge or WorldWriter produces this
    wq     = WorldQuery(state, kb)    # main loop wires these together
    # consumers receive wq, never state directly

Composable entity queries
--------------------------
The EntityQuery builder returned by WorldQuery.entities() supports
multi-predicate filtering in a single, readable expression:

    wq.entities()
      .with_name("assembling-machine-1")
      .with_recipe("electronic-circuit")
      .with_inserter_input()
      .with_inserter_output()
      .get()          # → list[EntityState]

Each .with_*() method narrows the working set.  Predicates that can use an
index do so; the rest filter the already-reduced list.  The builder is lazy —
nothing executes until .get() or .count() is called.

Convenience methods
-------------------
For the most common patterns, named methods wrap the builder:

    wq.entity_by_id(entity_id)
    wq.entities_by_name(name)
    wq.entities_by_status(status)
    wq.entities_by_recipe(recipe)
    wq.inserters_taking_from(entity_id, tile_width, tile_height)
    wq.inserters_delivering_to(entity_id, tile_width, tile_height)
    wq.inserters_taking_from_type(entity_name)
    wq.inserters_delivering_to_type(entity_name)
    wq.resources_of_type(resource_type)
    wq.inventory_count(item)
    wq.section_staleness(section, current_tick)

All methods return values or empty collections; none raise.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from world.state import (
    EntityState,
    EntityStatus,
    InserterState,
    Position,
    ResourcePatch,
    ResourceType,
    WorldState,
)

if TYPE_CHECKING:
    from world.knowledge import KnowledgeBase

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EntityQuery — composable filter builder
# ---------------------------------------------------------------------------

class EntityQuery:
    """
    Lazy, composable filter chain over EntityState objects.

    Constructed by WorldQuery.entities(); not instantiated directly.

    Each .with_*() call narrows self._working.  The narrowing is applied
    immediately (not truly lazy) because the working set is already a small
    in-memory list and true lazy evaluation would add complexity without
    measurable benefit at Factorio scan-radius scales (~10–200 entities).
    """

    def __init__(
        self,
        working: list[EntityState],
        wq: "WorldQuery",
    ) -> None:
        self._working = list(working)   # copy so mutations don't affect caller
        self._wq = wq

    # ------------------------------------------------------------------
    # Predicate filters
    # ------------------------------------------------------------------

    def with_name(self, name: str) -> "EntityQuery":
        """Keep only entities with the given prototype name."""
        # Use the index if the full set is still loaded; otherwise filter.
        if len(self._working) == len(self._wq._state.entities):
            self._working = self._wq._state._by_name.get(name, [])
        else:
            self._working = [e for e in self._working if e.name == name]
        return self

    def with_recipe(self, recipe: str) -> "EntityQuery":
        """Keep only entities currently set to the given recipe."""
        self._working = [e for e in self._working if e.recipe == recipe]
        return self

    def with_status(self, status: EntityStatus) -> "EntityQuery":
        """Keep only entities with the given status."""
        self._working = [e for e in self._working if e.status == status]
        return self

    def with_inserter_input(self) -> "EntityQuery":
        """Keep only entities that have at least one inserter delivering to them."""
        result = []
        for e in self._working:
            w, h = self._wq._tile_dims(e.name)
            if self._wq.inserters_delivering_to(e.entity_id, w, h):
                result.append(e)
        self._working = result
        return self

    def with_inserter_output(self) -> "EntityQuery":
        """Keep only entities that have at least one inserter taking from them."""
        result = []
        for e in self._working:
            w, h = self._wq._tile_dims(e.name)
            if self._wq.inserters_taking_from(e.entity_id, w, h):
                result.append(e)
        self._working = result
        return self

    def with_predicate(self, fn) -> "EntityQuery":
        """
        Keep only entities for which fn(entity) is truthy.
        Escape hatch for predicates not covered by the named methods.
        """
        self._working = [e for e in self._working if fn(e)]
        return self

    # ------------------------------------------------------------------
    # Terminal operations
    # ------------------------------------------------------------------

    def get(self) -> list[EntityState]:
        """Return the filtered entity list."""
        return list(self._working)

    def count(self) -> int:
        """Return the count of entities satisfying all applied predicates."""
        return len(self._working)

    def first(self) -> Optional[EntityState]:
        """Return the first matching entity, or None."""
        return self._working[0] if self._working else None


# ---------------------------------------------------------------------------
# WorldQuery
# ---------------------------------------------------------------------------

class WorldQuery:
    """
    Read-only interface over a WorldState snapshot.

    Parameters
    ----------
    state : WorldState
        The shared world state object.  WorldQuery holds a reference, not a
        copy; queries always reflect the current state of the object.
    kb : KnowledgeBase, optional
        Used to resolve entity tile dimensions for spatial join queries.
        If None, all spatial queries fall back to 1×1 tile dimensions, which
        is correct for small entities and a conservative approximation for
        larger ones.
    """

    def __init__(
        self,
        state: WorldState,
        kb: Optional["KnowledgeBase"] = None,
    ) -> None:
        self._state = state
        self._kb = kb

    # ------------------------------------------------------------------
    # Composable builder entry point
    # ------------------------------------------------------------------

    def entities(self) -> EntityQuery:
        """
        Return an EntityQuery builder over all current entities.

        Example:
            wq.entities()
              .with_name("assembling-machine-1")
              .with_recipe("electronic-circuit")
              .with_inserter_input()
              .with_inserter_output()
              .count()
        """
        return EntityQuery(self._state.entities, self)

    # ------------------------------------------------------------------
    # Named convenience queries — entity lookups
    # ------------------------------------------------------------------

    def entity_by_id(self, entity_id: int) -> Optional[EntityState]:
        """O(1) lookup by entity_id. Returns None if not found."""
        return self._state._by_id.get(entity_id)

    def entities_by_name(self, name: str) -> list[EntityState]:
        """Return all entities with the given prototype name."""
        return list(self._state._by_name.get(name, []))

    def entities_by_status(self, status: EntityStatus) -> list[EntityState]:
        """Return all entities with the given status."""
        return list(self._state._by_status.get(status, []))

    def entities_by_recipe(self, recipe: str) -> list[EntityState]:
        """Return all entities currently set to the given recipe."""
        return list(self._state._by_recipe.get(recipe, []))

    def all_entities(self) -> list[EntityState]:
        """Return all entities in the current scan radius."""
        return list(self._state.entities)

    # ------------------------------------------------------------------
    # Connectivity queries — inserters
    # ------------------------------------------------------------------

    def inserters_taking_from(
        self,
        entity_id: int,
        tile_width: int = 1,
        tile_height: int = 1,
    ) -> list[InserterState]:
        """
        Return inserters whose pickup_position falls within the bounding box
        of the entity with the given entity_id.

        tile_width / tile_height should come from KnowledgeBase for non-1×1
        entities (assemblers, furnaces, etc.).  If WorldQuery was constructed
        with a KnowledgeBase, pass tile_width=0 and tile_height=0 to signal
        "resolve from KB automatically" — otherwise pass explicit dimensions.

        Returns an empty list if the entity is not found.
        """
        target = self._state._by_id.get(entity_id)
        if target is None:
            return []

        # If the spatial index is available and we're using 1×1 dimensions,
        # use the pre-built index for O(1) lookup.
        if (self._state._inserters_from is not None
                and tile_width == 1 and tile_height == 1):
            return list(self._state._inserters_from.get(entity_id, []))

        # Fall back to bounding-box scan (required for non-1×1 or stale index).
        return self._inserters_taking_from_scan(target, tile_width, tile_height)

    def inserters_delivering_to(
        self,
        entity_id: int,
        tile_width: int = 1,
        tile_height: int = 1,
    ) -> list[InserterState]:
        """
        Return inserters whose drop_position falls within the bounding box
        of the entity with the given entity_id.
        """
        target = self._state._by_id.get(entity_id)
        if target is None:
            return []

        if (self._state._inserters_to is not None
                and tile_width == 1 and tile_height == 1):
            return list(self._state._inserters_to.get(entity_id, []))

        return self._inserters_delivering_to_scan(target, tile_width, tile_height)

    def inserters_taking_from_type(self, entity_name: str) -> list[InserterState]:
        """
        Return all inserters taking from any entity of the given type.
        Uses KB-resolved tile dimensions if available, else 1×1.
        """
        targets = self._state._by_name.get(entity_name, [])
        if not targets:
            return []
        w, h = self._tile_dims(entity_name)
        result: list[InserterState] = []
        seen: set[int] = set()
        for entity in targets:
            for ins in self._inserters_taking_from_scan(entity, w, h):
                if ins.entity_id not in seen:
                    seen.add(ins.entity_id)
                    result.append(ins)
        return result

    def inserters_delivering_to_type(self, entity_name: str) -> list[InserterState]:
        """
        Return all inserters delivering to any entity of the given type.
        """
        targets = self._state._by_name.get(entity_name, [])
        if not targets:
            return []
        w, h = self._tile_dims(entity_name)
        result: list[InserterState] = []
        seen: set[int] = set()
        for entity in targets:
            for ins in self._inserters_delivering_to_scan(entity, w, h):
                if ins.entity_id not in seen:
                    seen.add(ins.entity_id)
                    result.append(ins)
        return result

    # ------------------------------------------------------------------
    # Named convenience queries — compound / high-level
    # ------------------------------------------------------------------

    def fully_connected_entities(self, recipe: str) -> list[EntityState]:
        """
        Return entities set to *recipe* that have both at least one inserter
        delivering to them AND at least one inserter taking from them.

        This is the most common compound query: "assemblers actually wired up."
        Equivalent to:
            wq.entities().with_recipe(recipe)
                         .with_inserter_input()
                         .with_inserter_output()
                         .get()
        """
        return (
            self.entities()
            .with_recipe(recipe)
            .with_inserter_input()
            .with_inserter_output()
            .get()
        )

    # ------------------------------------------------------------------
    # Resource map
    # ------------------------------------------------------------------

    def resources_of_type(self, resource_type: ResourceType) -> list[ResourcePatch]:
        """NON-PROXIMAL. Return all known patches of the given resource type."""
        return [r for r in self._state.resource_map if r.resource_type == resource_type]

    # ------------------------------------------------------------------
    # Player / inventory
    # ------------------------------------------------------------------

    def inventory_count(self, item: str) -> int:
        """NON-PROXIMAL. Player item count, summed across all inventory slots."""
        return self._state.player.inventory.count(item)

    def player_position(self) -> Position:
        return self._state.player.position

    def player_health(self) -> float:
        return self._state.player.health

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    @property
    def charted_chunks(self) -> int:
        """NON-PROXIMAL. Total chunks the force has revealed."""
        return self._state.player.exploration.charted_chunks

    @property
    def charted_tiles(self) -> int:
        return self._state.player.exploration.charted_tiles

    @property
    def charted_area_km2(self) -> float:
        return self._state.player.exploration.charted_area_km2

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    @property
    def research(self):
        """NON-PROXIMAL. Current ResearchState."""
        return self._state.research

    def tech_unlocked(self, tech: str) -> bool:
        return self._state.research.is_unlocked(tech)

    # ------------------------------------------------------------------
    # Logistics / power
    # ------------------------------------------------------------------

    @property
    def logistics(self):
        """PROXIMAL. Current LogisticsState (scan radius only)."""
        return self._state.logistics

    @property
    def power(self):
        """PROXIMAL. PowerGrid of the nearest electric network."""
        return self._state.logistics.power

    # ------------------------------------------------------------------
    # Damage / destruction
    # ------------------------------------------------------------------

    @property
    def has_damage(self) -> bool:
        return self._state.has_damage

    @property
    def damaged_entities(self):
        return self._state.damaged_entities

    @property
    def recent_losses(self):
        return self._state.recent_losses

    # ------------------------------------------------------------------
    # Ground items
    # ------------------------------------------------------------------

    @property
    def ground_items(self):
        """PROXIMAL. Items on the ground within scan radius."""
        return self._state.ground_items

    # ------------------------------------------------------------------
    # Threat
    # ------------------------------------------------------------------

    @property
    def threat(self):
        return self._state.threat

    # ------------------------------------------------------------------
    # Time / tick
    # ------------------------------------------------------------------

    @property
    def tick(self) -> int:
        return self._state.tick

    @property
    def game_time_seconds(self) -> float:
        return self._state.game_time_seconds

    # ------------------------------------------------------------------
    # Staleness
    # ------------------------------------------------------------------

    def section_staleness(self, section: str, current_tick: int) -> Optional[int]:
        """
        Ticks since *section* was last observed, or None if never observed.

        At 60 tps: 60=1s, 300=5s, 1800=30s, 3600=1min.
        """
        last = self._state.observed_at.get(section)
        if last is None:
            return None
        return max(0, current_tick - last)

    # ------------------------------------------------------------------
    # Raw state access — escape hatch for the reward evaluator namespace
    # ------------------------------------------------------------------

    @property
    def state(self) -> WorldState:
        """
        Direct reference to the underlying WorldState.

        Exposed for the reward evaluator namespace so that condition strings
        like ``state.game_time_seconds`` and ``state.ground_items`` continue
        to work without requiring every attribute to be individually proxied.

        Consumers of WorldQuery in application code should prefer the named
        methods above rather than reaching through .state.
        """
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tile_dims(self, entity_name: str) -> tuple[int, int]:
        """Return (tile_width, tile_height) for an entity, defaulting to (1, 1)."""
        if self._kb is None:
            return (1, 1)
        record = self._kb.get_entity(entity_name)
        if record is None or record.is_placeholder:
            return (1, 1)
        return (record.tile_width, record.tile_height)

    def _entity_contains_position(
        self,
        entity: EntityState,
        pos: Position,
        tile_width: int = 1,
        tile_height: int = 1,
        eps: float = 0.1,
    ) -> bool:
        half_w = tile_width  / 2.0 + eps
        half_h = tile_height / 2.0 + eps
        return (
            abs(pos.x - entity.position.x) <= half_w and
            abs(pos.y - entity.position.y) <= half_h
        )

    def _inserters_taking_from_scan(
        self,
        entity: EntityState,
        tile_width: int,
        tile_height: int,
    ) -> list[InserterState]:
        return [
            ins for ins in self._state.logistics.inserters.values()
            if ins.pickup_position is not None
            and self._entity_contains_position(
                entity, ins.pickup_position, tile_width, tile_height
            )
        ]

    def _inserters_delivering_to_scan(
        self,
        entity: EntityState,
        tile_width: int,
        tile_height: int,
    ) -> list[InserterState]:
        return [
            ins for ins in self._state.logistics.inserters.values()
            if ins.drop_position is not None
            and self._entity_contains_position(
                entity, ins.drop_position, tile_width, tile_height
            )
        ]
