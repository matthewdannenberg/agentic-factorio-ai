"""
world/state.py

WorldState — the agent's belief-state snapshot of the game at a point in time.

Produced by:  bridge/state_parser.py  (constructs fresh snapshots)
Mutated by:   world/writer.py         (WorldWriter — the ONLY permitted mutation path)
Read by:      world/query.py          (WorldQuery — the ONLY permitted read path)

Rules:
- Pure data. No LLM calls. No RCON. No side effects.
- All fields have safe defaults so partial state never requires special-casing.
- The nested dataclass tree is trivially serialisable to/from JSON.
- __post_init__ builds internal indices for O(1) / O(k) lookups; these are
  maintained by WorldWriter on every mutation.

IMPORTANT — access discipline
------------------------------
No file outside world/ may import WorldState directly. All reads go through
WorldQuery; all writes go through WorldWriter. This boundary allows the
backing implementation to be swapped without touching any consumer.

  world/__init__.py exports: WorldQuery, WorldWriter, and the dataclass types
  (EntityState, Position, etc.) needed in method signatures.
  WorldState itself is NOT exported from world/__init__.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntityStatus(Enum):
    WORKING    = "working"
    IDLE       = "idle"
    NO_INPUT   = "no_input"
    NO_POWER   = "no_power"
    FULL_OUT   = "full_output"
    UNKNOWN    = "unknown"


class ResourceName:
    IRON_ORE    = "iron-ore"
    COPPER_ORE  = "copper-ore"
    COAL        = "coal"
    STONE       = "stone"
    CRUDE_OIL   = "crude-oil"
    URANIUM_ORE = "uranium-ore"
    WATER       = "water"
    WOOD        = "wood"

ResourceType = str


class Direction(Enum):
    NORTH = 0
    EAST  = 2
    SOUTH = 4
    WEST  = 6


# ---------------------------------------------------------------------------
# Coordinate primitive
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Position:
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def __repr__(self) -> str:
        return f"({self.x:.1f}, {self.y:.1f})"


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

@dataclass
class InventorySlot:
    item: str
    count: int


@dataclass
class Inventory:
    slots: list[InventorySlot] = field(default_factory=list)

    def count(self, item: str) -> int:
        return sum(s.count for s in self.slots if s.item == item)

    def as_dict(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for slot in self.slots:
            result[slot.item] = result.get(slot.item, 0) + slot.count
        return result

    def is_empty(self) -> bool:
        return not self.slots or all(s.count == 0 for s in self.slots)


# ---------------------------------------------------------------------------
# Entities (placed buildings)
# ---------------------------------------------------------------------------

@dataclass
class EntityState:
    entity_id: int
    name: str
    position: Position
    direction: Direction = Direction.NORTH
    status: EntityStatus = EntityStatus.UNKNOWN
    recipe: Optional[str] = None
    inventory: Optional[Inventory] = None
    energy: float = 0.0


# ---------------------------------------------------------------------------
# Resource map
# ---------------------------------------------------------------------------

@dataclass
class ResourcePatch:
    resource_type: ResourceType
    position: Position
    amount: int
    size: int
    observed_at: int = 0


# ---------------------------------------------------------------------------
# Ground items
# ---------------------------------------------------------------------------

@dataclass
class GroundItem:
    item: str
    position: Position
    count: int
    observed_at: int = 0
    age_ticks: int = 0


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

@dataclass
class ResearchState:
    unlocked: list[str] = field(default_factory=list)
    in_progress: Optional[str] = None
    queued: list[str] = field(default_factory=list)
    science_per_minute: dict[str, float] = field(default_factory=dict)

    def is_unlocked(self, tech: str) -> bool:
        return tech in self.unlocked


# ---------------------------------------------------------------------------
# Logistics
# ---------------------------------------------------------------------------

@dataclass
class BeltLane:
    """
    Contents of one transport line lane on a belt tile.

    congested : True if this lane is backed-up.
    items     : {item_name: count}
    """
    congested: bool = False
    items: dict[str, int] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.items

    def carries(self, item: str) -> bool:
        return self.items.get(item, 0) > 0

    def total_items(self) -> int:
        return sum(self.items.values())


@dataclass
class BeltSegment:
    """
    One belt tile with two independent transport lanes.

    lane1 : Left transport line  (Factorio get_transport_line(1)).
    lane2 : Right transport line (Factorio get_transport_line(2)).
    """
    segment_id: int
    positions: list[Position]
    lane1: BeltLane = field(default_factory=BeltLane)
    lane2: BeltLane = field(default_factory=BeltLane)

    @property
    def congested(self) -> bool:
        return self.lane1.congested or self.lane2.congested

    @property
    def items(self) -> dict[str, int]:
        merged: dict[str, int] = dict(self.lane1.items)
        for item, count in self.lane2.items.items():
            merged[item] = merged.get(item, 0) + count
        return merged

    def carries(self, item: str) -> bool:
        return self.lane1.carries(item) or self.lane2.carries(item)


@dataclass
class PowerGrid:
    produced_kw: float = 0.0
    consumed_kw: float = 0.0
    accumulated_kj: float = 0.0
    satisfaction: float = 1.0

    @property
    def headroom_kw(self) -> float:
        return self.produced_kw - self.consumed_kw

    @property
    def is_brownout(self) -> bool:
        return self.satisfaction < 1.0


@dataclass
class InserterState:
    """
    A single inserter's current activity and reach positions.

    pickup_position : World coordinates where the arm picks items up from.
    drop_position   : World coordinates where the arm places items down.

    Either position may be None (pcall-guarded in Lua).
    """
    entity_id: int
    position: Position
    active: bool = False
    pickup_position: Optional[Position] = None
    drop_position: Optional[Position] = None


@dataclass
class LogisticsState:
    belts: list[BeltSegment] = field(default_factory=list)
    power: PowerGrid = field(default_factory=PowerGrid)
    inserters: dict[int, InserterState] = field(default_factory=dict)
    # Legacy activity shorthand: entity_id → 1 (active) or 0 (idle).
    inserter_activity: dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

@dataclass
class ExplorationState:
    """
    Force-level map exploration statistics. NON-PROXIMAL.

    charted_chunks : Chunks revealed by this force. Monotonically increasing.
                     Sourced from LuaForce::get_chart_size(surface).
    """
    charted_chunks: int = 0

    @property
    def charted_tiles(self) -> int:
        return self.charted_chunks * 1024

    @property
    def charted_area_km2(self) -> float:
        return self.charted_tiles / 1_000_000.0


# ---------------------------------------------------------------------------
# Structural damage
# ---------------------------------------------------------------------------

@dataclass
class DamagedEntity:
    entity_id: int
    name: str
    position: Position
    health_fraction: float
    observed_at: int = 0


@dataclass
class DestroyedEntity:
    name: str
    position: Position
    destroyed_at: int
    cause: str = "unknown"


# ---------------------------------------------------------------------------
# Threat (biter-specific)
# ---------------------------------------------------------------------------

@dataclass
class BiterBase:
    base_id: int
    position: Position
    size: int
    evolution: float


@dataclass
class ThreatState:
    biter_bases: list[BiterBase] = field(default_factory=list)
    pollution_cloud: list[Position] = field(default_factory=list)
    attack_timers: dict[int, float] = field(default_factory=dict)
    evolution_factor: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.biter_bases


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    position: Position = field(default_factory=lambda: Position(0.0, 0.0))
    health: float = 100.0
    inventory: Inventory = field(default_factory=Inventory)
    reachable: list[int] = field(default_factory=list)
    exploration: ExplorationState = field(default_factory=ExplorationState)


# ---------------------------------------------------------------------------
# WorldState — top-level snapshot
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    """
    The agent's current belief state about the game world.

    NOT ground truth — a cached, partially-observed snapshot assembled by
    bridge/state_parser.py. Different sections have different staleness;
    consumers should check observed_at via WorldQuery.staleness().

    Access discipline
    -----------------
    All reads must go through WorldQuery.
    All mutations must go through WorldWriter.
    Neither class is imported here; WorldState is a pure data container.

    Internal indices
    ----------------
    __post_init__ builds lookup dicts from the initial field values.
    WorldWriter is responsible for keeping these indices consistent after
    any mutation. Never mutate the index dicts directly.

    Section names used in observed_at
    ----------------------------------
    "player", "entities", "resource_map", "ground_items", "research",
    "logistics", "damaged_entities", "destroyed_entities", "threat"
    """

    tick: int = 0
    observed_at: dict[str, int] = field(default_factory=dict)

    player: PlayerState = field(default_factory=PlayerState)
    entities: list[EntityState] = field(default_factory=list)
    resource_map: list[ResourcePatch] = field(default_factory=list)
    ground_items: list[GroundItem] = field(default_factory=list)
    research: ResearchState = field(default_factory=ResearchState)
    logistics: LogisticsState = field(default_factory=LogisticsState)
    threat: ThreatState = field(default_factory=ThreatState)

    damaged_entities: list[DamagedEntity] = field(default_factory=list)
    destroyed_entities: list[DestroyedEntity] = field(default_factory=list)
    destroyed_ttl_ticks: int = 18_000

    # ------------------------------------------------------------------
    # Internal indices — built by __post_init__, maintained by WorldWriter.
    # Named with leading underscore to signal "do not touch from outside".
    # ------------------------------------------------------------------
    _by_id:     dict[int, EntityState] = field(default_factory=dict, init=False, repr=False)
    _by_name:   dict[str, list[EntityState]] = field(default_factory=dict, init=False, repr=False)
    _by_recipe: dict[str, list[EntityState]] = field(default_factory=dict, init=False, repr=False)
    _by_status: dict[EntityStatus, list[EntityState]] = field(default_factory=dict, init=False, repr=False)

    # Spatial inserter indices — keyed by entity_id of the source/target entity.
    # Built lazily by WorldWriter after entities + inserters are both populated.
    # None signals "not yet built"; WorldQuery rebuilds on demand if None.
    _inserters_from: Optional[dict[int, list[InserterState]]] = field(
        default=None, init=False, repr=False
    )
    _inserters_to: Optional[dict[int, list[InserterState]]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._rebuild_entity_indices()
        # Spatial inserter indices are not built here because the inserters
        # and entities lists may not both be populated at construction time
        # (e.g. snapshot WorldStates from StateParser).  WorldWriter rebuilds
        # them after integrate_snapshot(); WorldQuery falls back to a scan if
        # they are None.

    # ------------------------------------------------------------------
    # Index management — called by WorldWriter, not by external consumers
    # ------------------------------------------------------------------

    def _rebuild_entity_indices(self) -> None:
        """Rebuild all entity lookup dicts from self.entities."""
        by_id: dict[int, EntityState] = {}
        by_name: dict[str, list[EntityState]] = {}
        by_recipe: dict[str, list[EntityState]] = {}
        by_status: dict[EntityStatus, list[EntityState]] = {}

        for e in self.entities:
            by_id[e.entity_id] = e

            by_name.setdefault(e.name, []).append(e)

            if e.recipe is not None:
                by_recipe.setdefault(e.recipe, []).append(e)

            by_status.setdefault(e.status, []).append(e)

        self._by_id = by_id
        self._by_name = by_name
        self._by_recipe = by_recipe
        self._by_status = by_status
        # Invalidate spatial indices — must be rebuilt after entity changes.
        self._inserters_from = None
        self._inserters_to = None

    def _rebuild_inserter_indices(self, eps: float = 0.1) -> None:
        """
        Rebuild spatial inserter indices.

        For each inserter, we find which entity (if any) its pickup_position
        falls within (→ _inserters_from) and which entity its drop_position
        falls within (→ _inserters_to).

        This is O(n_inserters × n_entities) but runs only when WorldWriter
        calls it after a mutation, not on every query.

        tile_width / tile_height are not available here (they live in
        KnowledgeBase, which WorldState does not import).  We use a 1×1
        bounding box for the index, which is correct for chests and other
        small entities.  WorldQuery corrects for large entities (assemblers,
        furnaces) when it detects a mismatch via the KB.
        """
        from_map: dict[int, list[InserterState]] = {e.entity_id: [] for e in self.entities}
        to_map:   dict[int, list[InserterState]] = {e.entity_id: [] for e in self.entities}

        for ins in self.logistics.inserters.values():
            for entity in self.entities:
                half = 0.5 + eps
                if ins.pickup_position is not None:
                    if (abs(ins.pickup_position.x - entity.position.x) <= half and
                            abs(ins.pickup_position.y - entity.position.y) <= half):
                        from_map[entity.entity_id].append(ins)
                if ins.drop_position is not None:
                    if (abs(ins.drop_position.x - entity.position.x) <= half and
                            abs(ins.drop_position.y - entity.position.y) <= half):
                        to_map[entity.entity_id].append(ins)

        self._inserters_from = from_map
        self._inserters_to = to_map

    # ------------------------------------------------------------------
    # Simple computed properties (no query logic — kept here for repr)
    # ------------------------------------------------------------------

    @property
    def game_time_seconds(self) -> float:
        return self.tick / 60.0

    @property
    def has_damage(self) -> bool:
        return bool(self.damaged_entities)

    @property
    def recent_losses(self) -> list[DestroyedEntity]:
        return self.destroyed_entities

    @property
    def charted_chunks(self) -> int:
        """NON-PROXIMAL shorthand for player.exploration.charted_chunks."""
        return self.player.exploration.charted_chunks

    def __repr__(self) -> str:
        return (
            f"WorldState(tick={self.tick}, "
            f"entities={len(self.entities)}, "
            f"resources={len(self.resource_map)}, "
            f"charted_chunks={self.charted_chunks}, "
            f"power_headroom={self.logistics.power.headroom_kw:.1f}kW)"
        )