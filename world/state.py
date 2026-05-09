"""
world/state.py

WorldState — the complete agent-readable snapshot of the game at a point in time.

Produced by:  bridge/state_parser.py
Consumed by:  planning, execution, and examination layers

Rules:
- Pure data. No LLM calls. No RCON. No side effects.
- All fields have safe defaults so partial state (early game, biters off) never
  requires special-casing in consumers.
- Nested types are also plain dataclasses or simple Python primitives so the
  whole tree is trivially serialisable to/from JSON (for logging and replay).
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
    FULL_OUT   = "full_output"   # output chest / belt saturated
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
    """Represents the contents of the player's inventory or a storage entity."""
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
    """A single placed entity on the map."""
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
    """
    Coarse strategic record of a resource deposit.

    resource_type   : Factorio internal resource name string, e.g. "iron-ore".
    position        : Approximate centre of the patch bounding box.
    amount          : Remaining resource units (stale between visits).
    size            : Approximate tile count.
    observed_at     : Game tick when this record was last updated.
    """
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
    """An item stack lying on the ground within the local scan radius."""
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

    Factorio belts have two independent lanes (left = line index 1,
    right = line index 2). Each lane can carry items of any type; a single
    tile can have multiple item types on the same lane (e.g. mixed inputs
    arriving at a splitter). Represented as a dict so the planner can ask
    "does lane 1 carry iron-plate?" in O(1) without iterating a list.

    congested : True if this lane is backed-up (at or above line_length).
    items     : {item_name: count} — empty dict when the lane is empty.
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
    One belt tile (transport-belt, underground-belt, or splitter) within
    the scan radius.

    lane1 : Left transport line  (Factorio get_transport_line(1)).
    lane2 : Right transport line (Factorio get_transport_line(2)).

    Queried per-tile rather than per-path because the Lua mod iterates
    individual belt entities; path-level aggregation is left to the planning
    layer.

    Convenience properties mirror the old single-item interface so existing
    code that checks `b.congested` keeps working:
      congested : True if either lane is congested.
      items     : Combined {item: count} across both lanes.
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
        """Combined item counts across both lanes."""
        merged: dict[str, int] = dict(self.lane1.items)
        for item, count in self.lane2.items.items():
            merged[item] = merged.get(item, 0) + count
        return merged

    def carries(self, item: str) -> bool:
        """True if either lane carries the named item."""
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

    active          : True if the inserter is currently holding an item
                      (i.e. mid-swing with something in its hand).
    pickup_position : World coordinates where the arm picks items up from.
    drop_position   : World coordinates where the arm places items down.

    These positions are fixed by the entity's direction and prototype offsets —
    they do not change while the inserter is stationary.  The WorldState
    connectivity queries use them to determine which entity each inserter is
    taking from or delivering to via bounding-box containment, without any
    additional RCON calls.

    Either position may be None if the Lua mod was unable to read it (pcall
    guard in control.lua) — consumers must handle None gracefully.
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
    # Full inserter records — replaces the old activity-only dict.
    # Keyed by entity_id for O(1) lookup.
    inserters: dict[int, InserterState] = field(default_factory=dict)
    # Legacy activity shorthand: entity_id → 1 (active) or 0 (idle).
    # Populated from inserters for backwards-compatibility; prefer inserters.
    inserter_activity: dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structural damage
# ---------------------------------------------------------------------------

@dataclass
class DamagedEntity:
    """
    A placed entity below full health.

    health_fraction : (0.0, 1.0) exclusive — never exactly 1.0 or 0.0.
    observed_at     : Game tick when this reading was taken.
    """
    entity_id: int
    name: str
    position: Position
    health_fraction: float
    observed_at: int = 0


@dataclass
class DestroyedEntity:
    """
    A record of an entity destroyed during the current run.

    cause : "biter" | "vehicle" | "deconstruct" | "unknown"
    """
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
    """
    Biter-specific threat information.
    All fields are empty / zero when BITERS_ENABLED=False.
    Structural damage lives on WorldState directly — it is cause-agnostic.
    """
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


# ---------------------------------------------------------------------------
# WorldState — top-level snapshot
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    """
    The agent's current belief state about the game world.

    This is NOT ground truth — it is a cached, partially-observed snapshot
    assembled by bridge/state_parser.py. Different sections have different
    staleness; consumers should check observed_at when freshness matters.

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
    # Standard accessors
    # ------------------------------------------------------------------

    def entity_by_id(self, entity_id: int) -> Optional[EntityState]:
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        return None

    def entities_by_name(self, name: str) -> list[EntityState]:
        return [e for e in self.entities if e.name == name]

    def entities_by_status(self, status: EntityStatus) -> list[EntityState]:
        return [e for e in self.entities if e.status == status]

    def resources_of_type(self, resource_type: ResourceType) -> list[ResourcePatch]:
        return [r for r in self.resource_map if r.resource_type == resource_type]

    def section_staleness(self, section: str, current_tick: int) -> Optional[int]:
        last = self.observed_at.get(section)
        if last is None:
            return None
        return max(0, current_tick - last)

    def inventory_count(self, item: str) -> int:
        return self.player.inventory.count(item)

    @property
    def game_time_seconds(self) -> float:
        return self.tick / 60.0

    @property
    def has_damage(self) -> bool:
        return bool(self.damaged_entities)

    @property
    def recent_losses(self) -> list[DestroyedEntity]:
        return self.destroyed_entities

    # ------------------------------------------------------------------
    # Connectivity queries
    # ------------------------------------------------------------------

    def _entity_contains_position(
        self, entity: EntityState, pos: Position,
        tile_width: int = 1, tile_height: int = 1,
    ) -> bool:
        """
        Return True if world-coordinate pos falls within the bounding box of
        entity, given its tile dimensions.

        Factorio entities are centred on their position; a 1×1 entity at
        (5, 5) occupies [4.5, 5.5) × [4.5, 5.5).  We add a small epsilon
        (0.1 tile) on each edge so that inserter reach positions that land
        exactly on an entity edge are still matched.
        """
        eps = 0.1
        half_w = tile_width  / 2.0 + eps
        half_h = tile_height / 2.0 + eps
        return (
            abs(pos.x - entity.position.x) <= half_w and
            abs(pos.y - entity.position.y) <= half_h
        )

    def inserters_taking_from(
        self,
        entity_id: int,
        tile_width: int = 1,
        tile_height: int = 1,
    ) -> list[InserterState]:
        """
        Return all inserters whose pickup_position falls within the bounding
        box of the entity with the given entity_id.

        tile_width / tile_height should come from KnowledgeBase.get_entity()
        for non-1×1 entities (assemblers, furnaces, etc.).  Defaults of 1×1
        work correctly for chests, accumulators, and other small entities.

        Returns an empty list if the entity is not found or no inserters are
        taking from it.
        """
        target = self.entity_by_id(entity_id)
        if target is None:
            return []
        return [
            ins for ins in self.logistics.inserters.values()
            if ins.pickup_position is not None
            and self._entity_contains_position(
                target, ins.pickup_position, tile_width, tile_height
            )
        ]

    def inserters_delivering_to(
        self,
        entity_id: int,
        tile_width: int = 1,
        tile_height: int = 1,
    ) -> list[InserterState]:
        """
        Return all inserters whose drop_position falls within the bounding
        box of the entity with the given entity_id.

        tile_width / tile_height should come from KnowledgeBase.get_entity()
        for non-1×1 entities.  Defaults of 1×1 work correctly for chests and
        other small entities.

        Returns an empty list if the entity is not found or no inserters are
        delivering to it.
        """
        target = self.entity_by_id(entity_id)
        if target is None:
            return []
        return [
            ins for ins in self.logistics.inserters.values()
            if ins.drop_position is not None
            and self._entity_contains_position(
                target, ins.drop_position, tile_width, tile_height
            )
        ]

    def inserters_taking_from_type(self, entity_name: str) -> list[InserterState]:
        """
        Return all inserters taking from any entity of the given type.
        Useful when the exact entity_id is not known in advance.
        Assumes 1×1 tile footprint — use inserters_taking_from() with explicit
        tile dimensions for larger entities.
        """
        targets = {e.entity_id: e for e in self.entities_by_name(entity_name)}
        if not targets:
            return []
        result = []
        for ins in self.logistics.inserters.values():
            if ins.pickup_position is None:
                continue
            for entity in targets.values():
                if self._entity_contains_position(entity, ins.pickup_position):
                    result.append(ins)
                    break
        return result

    def inserters_delivering_to_type(self, entity_name: str) -> list[InserterState]:
        """
        Return all inserters delivering to any entity of the given type.
        Assumes 1×1 tile footprint.
        """
        targets = {e.entity_id: e for e in self.entities_by_name(entity_name)}
        if not targets:
            return []
        result = []
        for ins in self.logistics.inserters.values():
            if ins.drop_position is None:
                continue
            for entity in targets.values():
                if self._entity_contains_position(entity, ins.drop_position):
                    result.append(ins)
                    break
        return result

    def __repr__(self) -> str:
        return (
            f"WorldState(tick={self.tick}, "
            f"entities={len(self.entities)}, "
            f"resources={len(self.resource_map)}, "
            f"power_headroom={self.logistics.power.headroom_kw:.1f}kW)"
        )