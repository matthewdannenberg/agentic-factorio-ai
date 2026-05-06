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
from enum import Enum, auto
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


# Resource names are plain strings matching Factorio's internal item/fluid names,
# e.g. "iron-ore", "crude-oil", "se-cryonite" (Space Age), any mod resource.
# String constants below cover vanilla and avoid magic strings in core code —
# they are not exhaustive. The ResourceRegistry (world/entities.py) is the
# authoritative store of known resource metadata and is extended at runtime
# when the bridge parser encounters an unfamiliar resource name.
class ResourceName:
    IRON_ORE    = "iron-ore"
    COPPER_ORE  = "copper-ore"
    COAL        = "coal"
    STONE       = "stone"
    CRUDE_OIL   = "crude-oil"
    URANIUM_ORE = "uranium-ore"
    WATER       = "water"
    WOOD        = "wood"

# Type alias — resource types are plain strings throughout the codebase.
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

    @property
    def has_damage(self) -> bool:
        """True if any placed entity is currently below full health."""
        return bool(self.damaged_entities)

    @property
    def recent_losses(self) -> list[DestroyedEntity]:
        """All destroyed entities within the TTL window (pruning done by bridge)."""
        return self.destroyed_entities

    def __repr__(self) -> str:
        return f"({self.x:.1f}, {self.y:.1f})"


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

@dataclass
class InventorySlot:
    item: str          # Factorio internal item name, e.g. "iron-plate"
    count: int


@dataclass
class Inventory:
    """Represents the contents of the player's inventory or a storage entity."""
    slots: list[InventorySlot] = field(default_factory=list)

    def count(self, item: str) -> int:
        """Return total count of a named item across all slots."""
        return sum(s.count for s in self.slots if s.item == item)

    def as_dict(self) -> dict[str, int]:
        """Flatten to {item_name: total_count}."""
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
    entity_id: int              # Factorio unit_number
    name: str                   # Factorio internal name, e.g. "assembling-machine-2"
    position: Position
    direction: Direction = Direction.NORTH
    status: EntityStatus = EntityStatus.UNKNOWN
    recipe: Optional[str] = None          # Currently-set recipe (assemblers, furnaces)
    inventory: Optional[Inventory] = None # Input/output buffer, if exposed
    energy: float = 0.0                   # Current energy consumption in kW


# ---------------------------------------------------------------------------
# Resource map
# ---------------------------------------------------------------------------

@dataclass
class ResourcePatch:
    """
    Coarse strategic record of a resource deposit.

    This is intentionally low-resolution — it tells the planner "there is
    iron to the northeast, roughly 200k units remaining." It does NOT store
    individual tile positions; those are obtained by the bridge doing a local
    scan when the agent is physically at the patch (see bridge/state_parser.py).

    Fields
    ------
    resource_type   : Factorio internal resource name string, e.g. "iron-ore".
                      May be an unrecognised mod resource — consumers should
                      not assume it is a ResourceName constant.
    position        : Approximate centre of the patch bounding box.
    amount          : Remaining resource units. Updated on local scan; may be
                      stale between visits.
    size            : Approximate tile count (patch footprint, not yield).
    observed_at     : Game tick when this record was last updated by the bridge.
                      Consumers can compute staleness as (current_tick - observed_at).
    """
    resource_type: ResourceType       # str — Factorio internal name
    position: Position                # Approximate centre
    amount: int                       # Remaining units (stale between visits)
    size: int                         # Approximate tile count
    observed_at: int = 0              # Tick of last bridge update



# ---------------------------------------------------------------------------
# Ground items
# ---------------------------------------------------------------------------

@dataclass
class GroundItem:
    """
    An item stack lying on the ground (dropped from destroyed buildings,
    inventory overflow, player deaths, manually thrown items, etc.).

    Populated only within the bridge's local scan radius — the full map is
    never searched. Stale outside that radius.

    Fields
    ------
    item        : Factorio internal item name, e.g. "iron-plate".
    position    : Tile position of the item entity.
    count       : Stack size on the ground.
    observed_at : Game tick when this record was last seen by the bridge.
    age_ticks   : How many ticks the item has been on the ground (Factorio
                  tracks this; items despawn after a configurable duration).
    """
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
    unlocked: list[str] = field(default_factory=list)   # Technology names
    in_progress: Optional[str] = None                    # Currently researching
    queued: list[str] = field(default_factory=list)      # Ordered queue
    science_per_minute: dict[str, float] = field(default_factory=dict)  # pack → rate

    def is_unlocked(self, tech: str) -> bool:
        return tech in self.unlocked


# ---------------------------------------------------------------------------
# Logistics
# ---------------------------------------------------------------------------

@dataclass
class BeltSegment:
    segment_id: int
    positions: list[Position]
    congested: bool = False   # True if belt is backed-up / not moving
    item: Optional[str] = None  # Item currently on belt (if uniform)


@dataclass
class PowerGrid:
    produced_kw: float = 0.0
    consumed_kw: float = 0.0
    accumulated_kj: float = 0.0    # Battery / accumulator charge
    satisfaction: float = 1.0      # 1.0 = fully powered, <1.0 = brownout

    @property
    def headroom_kw(self) -> float:
        return self.produced_kw - self.consumed_kw

    @property
    def is_brownout(self) -> bool:
        return self.satisfaction < 1.0


@dataclass
class LogisticsState:
    belts: list[BeltSegment] = field(default_factory=list)
    power: PowerGrid = field(default_factory=PowerGrid)
    # Inserter activity: entity_id → items moved last N ticks (for throughput heuristics)
    inserter_activity: dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structural damage
# (cause-agnostic: biters, vehicles, player error, deconstruction, etc.)
# ---------------------------------------------------------------------------

@dataclass
class DamagedEntity:
    """
    A placed entity that has taken damage and is currently below full health.

    Populated whenever damage is detected regardless of cause — biter attack,
    vehicle collision, player error, or anything else. Always a world-level
    observation; not tied to biter activity.

    Fields
    ------
    entity_id       : Factorio unit_number.
    name            : Factorio internal entity name.
    position        : Map position.
    health_fraction : Current health as a fraction of maximum. In the range
                      (0.0, 1.0) — never exactly 1.0 (undamaged entities are
                      excluded) and never 0.0 (that would be destroyed).
    observed_at     : Game tick when this reading was taken.
    """
    entity_id: int
    name: str
    position: Position
    health_fraction: float   # (0.0, 1.0) exclusive on both ends
    observed_at: int = 0


@dataclass
class DestroyedEntity:
    """
    A record of an entity destroyed during the current run.

    Kept in a rolling window (see WorldState.destroyed_ttl_ticks) so the agent
    can plan recovery without the list growing unboundedly.  The bridge prunes
    records older than destroyed_ttl_ticks on each update.

    cause is best-effort — the Lua mod may not always be able to determine why
    an entity was destroyed.

    Fields
    ------
    name         : Factorio internal entity name (unit_number is gone at destruction).
    position     : Last known map position.
    destroyed_at : Game tick of destruction.
    cause        : "biter" | "vehicle" | "deconstruct" | "unknown"
    """
    name: str
    position: Position
    destroyed_at: int
    cause: str = "unknown"   # "biter" | "vehicle" | "deconstruct" | "unknown"


# ---------------------------------------------------------------------------
# Threat (biter-specific — empty when BITERS_ENABLED=False)
# ---------------------------------------------------------------------------

@dataclass
class BiterBase:
    base_id: int
    position: Position
    size: int        # Approximate unit count
    evolution: float # 0.0–1.0


@dataclass
class ThreatState:
    """
    Biter-specific threat information.

    All fields are empty / zero when BITERS_ENABLED=False. The bridge stub
    populates nothing here; consumers should check is_empty before acting.

    Structural damage (DamagedEntity, DestroyedEntity) lives on WorldState
    directly — it is cause-agnostic and present regardless of biter config.
    """
    biter_bases: list[BiterBase] = field(default_factory=list)
    pollution_cloud: list[Position] = field(default_factory=list)  # Sparse sample
    # Game ticks until next predicted attack per structure (entity_id → ticks)
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
    # Reached entities (within interaction range) — entity_ids
    reachable: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# WorldState — top-level snapshot
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    """
    The agent's current belief state about the game world.

    This is NOT ground truth. It is a cached, partially-observed snapshot
    assembled by bridge/state_parser.py from whatever the Lua mod could see
    at query time. Different sections have different staleness:

      - player, logistics.power  — refreshed every tick cycle (fresh)
      - entities near player     — refreshed on local scan (moderately fresh)
      - entities far from player — may be many ticks stale
      - resource_map             — updated only when agent visits a patch (very stale)
      - ground_items             — populated within local scan radius only; empty outside it
      - damaged_entities         — refreshed on each bridge scan; empty when nothing is damaged
      - destroyed_entities       — rolling window (destroyed_ttl_ticks); pruned by bridge
      - threat                   — only meaningful when biters are enabled and
                                   the scanner is actively polling biter territory

    Staleness is tracked per section via `observed_at`, a dict mapping section
    names to the game tick of their last bridge update. Consumers that care about
    freshness should check `observed_at.get("resource_map", 0)` etc.

    All sub-fields default to safe empty values so consumers never need to
    guard against None — only check .is_empty / .count / etc. as appropriate.

    Section names used in observed_at
    ----------------------------------
    "player", "entities", "resource_map", "ground_items", "research", "logistics",
    "damaged_entities", "destroyed_entities", "threat"
    """

    # Game clock — tick of the most recent bridge poll
    tick: int = 0   # Factorio game tick (60 ticks = 1 second)

    # Per-section staleness tracking.
    # Key: section name (see docstring). Value: tick of last bridge update.
    # Missing key means the section has never been observed this run.
    observed_at: dict[str, int] = field(default_factory=dict)

    # Top-level state sections
    player: PlayerState = field(default_factory=PlayerState)
    entities: list[EntityState] = field(default_factory=list)
    resource_map: list[ResourcePatch] = field(default_factory=list)
    ground_items: list[GroundItem] = field(default_factory=list)
    research: ResearchState = field(default_factory=ResearchState)
    logistics: LogisticsState = field(default_factory=LogisticsState)
    threat: ThreatState = field(default_factory=ThreatState)

    # Structural damage — cause-agnostic, always active regardless of biter config.
    # Populated by the bridge whenever damage or destruction is observed; not
    # gated on BITERS_ENABLED.
    damaged_entities: list[DamagedEntity] = field(default_factory=list)
    destroyed_entities: list[DestroyedEntity] = field(default_factory=list)
    # Bridge prunes destroyed_entities older than this many ticks on each update.
    destroyed_ttl_ticks: int = 18_000   # ~5 min at 60 tps

    # Convenience accessors ------------------------------------------------

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
        """Return all patches of a given resource type (string name)."""
        return [r for r in self.resource_map if r.resource_type == resource_type]

    def section_staleness(self, section: str, current_tick: int) -> Optional[int]:
        """
        Ticks since a named section was last observed, or None if never observed.
        Useful for planning layer to decide whether to request a fresh scan.
        """
        last = self.observed_at.get(section)
        if last is None:
            return None
        return max(0, current_tick - last)

    def inventory_count(self, item: str) -> int:
        """Player inventory count for a named item."""
        return self.player.inventory.count(item)

    @property
    def game_time_seconds(self) -> float:
        return self.tick / 60.0

    @property
    def has_damage(self) -> bool:
        """True if any placed entity is currently below full health."""
        return bool(self.damaged_entities)

    @property
    def recent_losses(self) -> list[DestroyedEntity]:
        """All destroyed entities within the TTL window (pruning done by bridge)."""
        return self.destroyed_entities

    def __repr__(self) -> str:
        return (
            f"WorldState(tick={self.tick}, "
            f"entities={len(self.entities)}, "
            f"resources={len(self.resource_map)}, "
            f"power_headroom={self.logistics.power.headroom_kw:.1f}kW)"
        )