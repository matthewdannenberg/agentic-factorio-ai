"""
world/observable/writer.py

WorldWriter — the sole mutation interface for WorldState.

Rules:
- No LLM calls. No RCON. No reads of game state for reasoning purposes.
- Every mutation to WorldState must go through WorldWriter.
- WorldWriter keeps WorldState's internal indices consistent after each write.
- WorldWriter is injected into bridge/state_parser (via integrate_snapshot)
  and the execution layer. Nothing else receives a WorldWriter.

Two categories of write
-----------------------
1. Section replacement (bridge / integrate_snapshot):
   The StateParser produces a fresh snapshot WorldState from the Lua mod.
   WorldWriter.integrate_snapshot() merges it into the live global state
   section by section, updating observed_at for each section present and
   rebuilding indices.

   Staleness logic: if a snapshot section's tick is older than the
   corresponding observed_at in the live state, that section is skipped
   (we never overwrite fresh data with stale data).

   Entity identity
   ---------------
   integrate_snapshot() assigns stable system-side IDs (sys_ids) to every
   entity before it enters the live WorldState. Raw Factorio unit_numbers
   never appear above this layer.

   Placed entities (is_natural=False): a Factorio unit_number → sys_id
   mapping is maintained in _factorio_to_sys. Each unit_number is assigned
   a sys_id on first encounter and reused on every subsequent snapshot.

   Natural objects (is_natural=True): identity is resolved by name +
   position proximity matching against _natural_registry. Each distinct
   (name, position) combination is assigned a sys_id on first encounter.
   Epsilon for proximity matching is ~1.0 tile; conflicts (two same-named
   objects within epsilon) take the nearest match.

   Reachability
   ------------
   The snapshot player.reachable list contains raw Factorio unit_numbers.
   integrate_snapshot() translates these to sys_ids for placed entities and
   additionally adds sys_ids for natural objects within
   player.reach_distance tiles of the player position. The live
   player.reachable is always a set of sys_ids after integration.

2. Fine-grained mutation (execution layer):
   The agent places an entity, changes a recipe, picks up an item, etc.
   WorldWriter provides per-field methods that update the live state and
   keep indices consistent without rebuilding everything.

Construction
------------
    state  = WorldState()
    wq     = WorldQuery(state, kb)   # read interface
    ww     = WorldWriter(state)      # write interface
    # The main loop holds state, wq, ww.
    # Downstream components receive either wq (readers) or ww (writers).
    # Call ww.reset_identity_registry() at the start of each run.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from world.observable.state import (
    DamagedEntity,
    DestroyedEntity,
    EntityState,
    EntityStatus,
    ExplorationState,
    GroundItem,
    Inventory,
    LogisticsState,
    PlayerState,
    Position,
    ResearchState,
    ResourcePatch,
    ThreatState,
    WorldState,
)

log = logging.getLogger(__name__)


class WorldWriter:
    """
    Write interface for a shared WorldState object.

    Holds a reference to the same WorldState that WorldQuery wraps.
    All mutations go through this class; WorldState fields are never
    written to directly from outside world/.

    Parameters
    ----------
    state : WorldState
        The shared world state object to mutate.
    """

    def __init__(self, state: WorldState) -> None:
        self._state = state
        # ID registry — reset between runs via reset_identity_registry().
        self._sys_id_counter: int = 1
        # Factorio unit_number → sys_id (placed entities only).
        self._factorio_to_sys: dict[int, int] = {}
        # sys_id → (name, x, y) for natural objects; used for identity matching.
        self._natural_registry: dict[int, tuple[str, float, float]] = {}
        # Reverse map: sys_id → Factorio unit_number (0 for natural objects).
        # Exposed via factorio_id_for() so the bridge can dispatch actions.
        self._sys_to_factorio: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Section replacement — bridge / integrate_snapshot
    # ------------------------------------------------------------------

    def integrate_snapshot(self, snapshot: WorldState) -> None:
        """
        Merge a fresh StateParser snapshot into the live global WorldState.

        For each section present in the snapshot (i.e. where observed_at
        is set), the live state is updated only if the snapshot data is
        at least as fresh as the current live data (tick-based guard).

        Entities and natural objects are unified into a single list. Each
        entity is assigned a stable sys_id before entering the live state;
        raw Factorio unit_numbers do not appear above this method.

        The entity and inserter indices are rebuilt once at the end, after
        all sections have been merged.

        Parameters
        ----------
        snapshot : WorldState
            A WorldState produced by StateParser.parse(). This object is
            consumed by this call; callers should not reuse it.
        """
        live = self._state
        snap_tick = snapshot.tick

        # Always advance the global tick to the latest observed value.
        if snap_tick >= live.tick:
            live.tick = snap_tick

        def _is_fresh(section: str) -> bool:
            """True if snapshot has this section and it's not stale."""
            snap_obs = snapshot.observed_at.get(section)
            if snap_obs is None:
                return False
            live_obs = live.observed_at.get(section, -1)
            return snap_obs >= live_obs

        if _is_fresh("player"):
            # Carry accumulated charted_chunk_coords forward.
            accumulated = live.player.exploration.charted_chunk_coords
            for c in snapshot.player.exploration.newly_charted_chunks:
                accumulated.add((c.cx, c.cy))
            live.player = snapshot.player
            live.player.exploration.charted_chunk_coords = accumulated
            live.observed_at["player"] = snapshot.observed_at["player"]

        # Unify placed entities and natural objects into one list, assigning
        # sys_ids. Use the fresher of the two timestamps; in practice they
        # arrive together and are equal.
        entities_fresh = _is_fresh("entities")
        naturals_fresh = _is_fresh("natural_objects")

        if entities_fresh or naturals_fresh:
            unified: list[EntityState] = []

            if entities_fresh:
                for e in snapshot.entities:
                    sys_id = self._resolve_sys_id_placed(e.entity_id)
                    e.entity_id = sys_id
                    unified.append(e)

            if naturals_fresh:
                for e in snapshot.natural_objects:
                    sys_id = self._resolve_sys_id_natural(
                        e.name, e.position.x, e.position.y
                    )
                    e.entity_id = sys_id
                    unified.append(e)

            live.entities = unified
            # Use the fresher of the two observed_at values.
            obs = max(
                snapshot.observed_at.get("entities", -1),
                snapshot.observed_at.get("natural_objects", -1),
            )
            live.observed_at["entities"] = obs

        # Translate player.reachable from Factorio unit_numbers to sys_ids,
        # then add natural objects within reach_distance geometrically.
        # This runs whenever either the player or the entity list was updated.
        if _is_fresh("player") or entities_fresh or naturals_fresh:
            reach = live.player.reach_distance
            player_pos = live.player.position
            reachable_sys: set[int] = set()

            # Placed entities: translate Factorio IDs → sys_ids.
            for factorio_id in snapshot.player.reachable:
                sys_id = self._factorio_to_sys.get(factorio_id)
                if sys_id is not None:
                    reachable_sys.add(sys_id)

            # Natural objects: geometric reach check.
            if reach > 0.0:
                for e in live.entities:
                    if e.is_natural:
                        dx = e.position.x - player_pos.x
                        dy = e.position.y - player_pos.y
                        if math.sqrt(dx * dx + dy * dy) <= reach:
                            reachable_sys.add(e.entity_id)

            live.player.reachable = list(reachable_sys)

        if _is_fresh("resource_map"):
            live.resource_map = snapshot.resource_map
            live.observed_at["resource_map"] = snapshot.observed_at["resource_map"]

        if _is_fresh("ground_items"):
            live.ground_items = snapshot.ground_items
            live.observed_at["ground_items"] = snapshot.observed_at["ground_items"]

        if _is_fresh("research"):
            live.research = snapshot.research
            live.observed_at["research"] = snapshot.observed_at["research"]

        if _is_fresh("logistics"):
            live.logistics = snapshot.logistics
            live.observed_at["logistics"] = snapshot.observed_at["logistics"]

        if _is_fresh("damaged_entities"):
            live.damaged_entities = snapshot.damaged_entities
            live.observed_at["damaged_entities"] = snapshot.observed_at["damaged_entities"]

        if _is_fresh("destroyed_entities"):
            combined = live.destroyed_entities + snapshot.destroyed_entities
            cutoff = live.tick - live.destroyed_ttl_ticks
            live.destroyed_entities = [e for e in combined if e.destroyed_at >= cutoff]
            seen: set[tuple] = set()
            deduped: list[DestroyedEntity] = []
            for e in live.destroyed_entities:
                key = (e.name, e.position, e.destroyed_at)
                if key not in seen:
                    seen.add(key)
                    deduped.append(e)
            live.destroyed_entities = deduped
            live.observed_at["destroyed_entities"] = snapshot.observed_at["destroyed_entities"]

        if _is_fresh("threat"):
            live.threat = snapshot.threat
            live.observed_at["threat"] = snapshot.observed_at["threat"]

        if _is_fresh("tile_map"):
            live.tile_map.update(snapshot.tile_map)
            live.observed_at["tile_map"] = snapshot.observed_at["tile_map"]

        # Rebuild indices once after all sections are merged.
        self._rebuild_all_indices()

    # ------------------------------------------------------------------
    # Fine-grained entity mutations
    # ------------------------------------------------------------------

    def add_entity(self, entity: EntityState) -> None:
        """Add a new entity. Rebuilds entity indices."""
        self._state.entities.append(entity)
        self._state._rebuild_entity_indices()
        self._state._rebuild_inserter_indices()

    def remove_entity(self, entity_id: int) -> bool:
        """
        Remove the entity with the given entity_id.
        Returns True if found and removed, False if not found.
        Rebuilds entity indices.
        """
        before = len(self._state.entities)
        self._state.entities = [
            e for e in self._state.entities if e.entity_id != entity_id
        ]
        if len(self._state.entities) < before:
            self._state._rebuild_entity_indices()
            self._state._rebuild_inserter_indices()
            return True
        return False

    def update_entity_status(self, entity_id: int, status: EntityStatus) -> bool:
        """
        Update the status of a single entity.
        Returns True if the entity was found, False otherwise.
        Rebuilds status index only (cheaper than full rebuild).
        """
        entity = self._state._by_id.get(entity_id)
        if entity is None:
            return False
        # Remove from old status bucket.
        old_bucket = self._state._by_status.get(entity.status, [])
        if entity in old_bucket:
            old_bucket.remove(entity)
        # Update field.
        entity.status = status
        # Add to new status bucket.
        self._state._by_status.setdefault(status, []).append(entity)
        return True

    def update_entity_recipe(self, entity_id: int, recipe: Optional[str]) -> bool:
        """
        Update the recipe of a single entity.
        Returns True if the entity was found, False otherwise.
        Rebuilds recipe index only.
        """
        entity = self._state._by_id.get(entity_id)
        if entity is None:
            return False
        # Remove from old recipe bucket.
        if entity.recipe is not None:
            old_bucket = self._state._by_recipe.get(entity.recipe, [])
            if entity in old_bucket:
                old_bucket.remove(entity)
        # Update field.
        entity.recipe = recipe
        # Add to new recipe bucket.
        if recipe is not None:
            self._state._by_recipe.setdefault(recipe, []).append(entity)
        return True

    def update_entity_inventory(self, entity_id: int, inventory: Inventory) -> bool:
        """Update the inventory of a single entity. No index changes needed."""
        entity = self._state._by_id.get(entity_id)
        if entity is None:
            return False
        entity.inventory = inventory
        return True

    # ------------------------------------------------------------------
    # Fine-grained player mutations
    # ------------------------------------------------------------------

    def update_player_position(self, position: Position) -> None:
        self._state.player.position = position

    def update_player_inventory(self, inventory: Inventory) -> None:
        self._state.player.inventory = inventory

    def update_player_health(self, health: float) -> None:
        self._state.player.health = health

    def update_exploration(self, exploration: ExplorationState) -> None:
        self._state.player.exploration = exploration

    # ------------------------------------------------------------------
    # Section-level replacements (for partial state updates outside bridge)
    # ------------------------------------------------------------------

    def replace_entities(self, entities: list[EntityState], tick: int) -> None:
        self._state.entities = entities
        self._state.observed_at["entities"] = tick
        self._state._rebuild_entity_indices()
        self._state._rebuild_inserter_indices()

    def replace_logistics(self, logistics: LogisticsState, tick: int) -> None:
        self._state.logistics = logistics
        self._state.observed_at["logistics"] = tick
        # Inserter spatial index must be rebuilt when logistics changes.
        self._state._rebuild_inserter_indices()

    def replace_resource_map(self, patches: list[ResourcePatch], tick: int) -> None:
        self._state.resource_map = patches
        self._state.observed_at["resource_map"] = tick

    def replace_research(self, research: ResearchState, tick: int) -> None:
        self._state.research = research
        self._state.observed_at["research"] = tick

    def replace_ground_items(self, items: list[GroundItem], tick: int) -> None:
        self._state.ground_items = items
        self._state.observed_at["ground_items"] = tick

    def replace_player(self, player: PlayerState, tick: int) -> None:
        self._state.player = player
        self._state.observed_at["player"] = tick

    def replace_damaged_entities(
        self, damaged: list[DamagedEntity], tick: int
    ) -> None:
        self._state.damaged_entities = damaged
        self._state.observed_at["damaged_entities"] = tick

    def replace_threat(self, threat: ThreatState, tick: int) -> None:
        self._state.threat = threat
        self._state.observed_at["threat"] = tick

    # ------------------------------------------------------------------
    # Identity registry
    # ------------------------------------------------------------------

    def reset_identity_registry(self) -> None:
        """
        Clear the entity identity registry and reset the sys_id counter.

        Must be called at the start of each run. Factorio entity IDs are only
        stable within a session; carrying them across runs would cause
        mis-identification.
        """
        self._sys_id_counter = 1
        self._factorio_to_sys.clear()
        self._natural_registry.clear()
        self._sys_to_factorio.clear()

    def factorio_id_for(self, sys_id: int) -> int:
        """
        Return the raw Factorio unit_number for the given sys_id.

        Returns 0 for natural objects (which have no Factorio unit_number).
        Returns 0 if the sys_id is not in the registry (unknown entity).

        This is the sole controlled escape hatch for code that must dispatch
        a Factorio action (e.g. MineEntity) using a raw unit_number. It must
        not be used for any other purpose.
        """
        return self._sys_to_factorio.get(sys_id, 0)

    def _resolve_sys_id_placed(self, factorio_id: int) -> int:
        """
        Return the sys_id for a placed entity with the given Factorio unit_number.

        Allocates a new sys_id on first encounter and caches it.
        """
        sys_id = self._factorio_to_sys.get(factorio_id)
        if sys_id is None:
            sys_id = self._sys_id_counter
            self._sys_id_counter += 1
            self._factorio_to_sys[factorio_id] = sys_id
            self._sys_to_factorio[sys_id] = factorio_id
        return sys_id

    def _resolve_sys_id_natural(
        self, name: str, x: float, y: float, epsilon: float = 1.0
    ) -> int:
        """
        Return the sys_id for a natural object identified by name and position.

        Scans _natural_registry for an existing entry with the same name within
        epsilon tiles. If found, returns that sys_id (nearest match on conflict).
        If not found, allocates a new sys_id and registers it.
        """
        best_sys_id: Optional[int] = None
        best_dist = float("inf")
        for sid, (rname, rx, ry) in self._natural_registry.items():
            if rname != name:
                continue
            dist = math.sqrt((rx - x) ** 2 + (ry - y) ** 2)
            if dist < epsilon and dist < best_dist:
                best_dist = dist
                best_sys_id = sid

        if best_sys_id is not None:
            return best_sys_id

        sys_id = self._sys_id_counter
        self._sys_id_counter += 1
        self._natural_registry[sys_id] = (name, x, y)
        self._sys_to_factorio[sys_id] = 0
        return sys_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_all_indices(self) -> None:
        """Rebuild both entity and inserter indices."""
        self._state._rebuild_entity_indices()
        self._state._rebuild_inserter_indices()