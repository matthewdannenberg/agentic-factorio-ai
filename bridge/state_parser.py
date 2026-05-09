"""
bridge/state_parser.py

StateParser — translates raw JSON output from the Lua mod into WorldState objects.

Rules:
- No RCON calls. No LLM calls. No side effects beyond ResourceRegistry updates.
- Unknown resource names are registered in ResourceRegistry, not rejected.
- Every populated section sets WorldState.observed_at[section] = current_tick.
- Partial state is normal — missing keys → safe defaults, no exceptions.
- destroyed_entities is pruned to WorldState.destroyed_ttl_ticks on each update.
- DestroyedEntity.cause is normalised to the four canonical strings; anything else
  becomes "unknown".

Expected top-level JSON structure from the Lua mod
---------------------------------------------------
{
  "tick": 3600,
  "player": { ... },
  "entities": [ ... ],
  "resource_map": [ ... ],
  "ground_items": [ ... ],
  "research": { ... },
  "logistics": { ... },
  "damaged_entities": [ ... ],
  "destroyed_entities": [ ... ],   // new events from circular buffer
  "threat": { ... }
}

Any subset of these keys may be present in a single response. Missing keys leave
the corresponding WorldState section at its current value (partial update mode).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from world.state import (
    BeltLane,
    BeltSegment,
    BiterBase,
    DamagedEntity,
    DestroyedEntity,
    Direction,
    EntityState,
    EntityStatus,
    GroundItem,
    Inventory,
    InventorySlot,
    InserterState,
    LogisticsState,
    PlayerState,
    Position,
    PowerGrid,
    ResearchState,
    ResourcePatch,
    ThreatState,
    WorldState,
)

logger = logging.getLogger(__name__)

# Canonical cause values; anything else is normalised to "unknown".
_VALID_CAUSES = frozenset({"biter", "vehicle", "deconstruct", "unknown"})

# Mapping from Lua entity status strings to EntityStatus enum.
_STATUS_MAP: dict[str, EntityStatus] = {
    "working":         EntityStatus.WORKING,
    "idle":            EntityStatus.IDLE,
    "no_minable_resources": EntityStatus.NO_INPUT,
    "no_input_fluid":  EntityStatus.NO_INPUT,
    "item_ingredient_shortage": EntityStatus.NO_INPUT,
    "fluid_ingredient_shortage": EntityStatus.NO_INPUT,
    "no_power":        EntityStatus.NO_POWER,
    "no_fuel":         EntityStatus.NO_POWER,
    "full_output":     EntityStatus.FULL_OUT,
    "output_full":     EntityStatus.FULL_OUT,
    "not_plugged_in_electric_network": EntityStatus.NO_POWER,
}

_DIRECTION_MAP: dict[int, Direction] = {
    0: Direction.NORTH,
    2: Direction.EAST,
    4: Direction.SOUTH,
    6: Direction.WEST,
}


class StateParser:
    """
    Translates raw Lua/JSON strings from the bridge mod into WorldState objects.

    parse()         — full parse; produces a fresh WorldState each call.
    parse_partial() — merges a section update into an existing WorldState.
    """

    def __init__(self, resource_registry: Optional[Any] = None) -> None:
        """
        Parameters
        ----------
        resource_registry : Optional ResourceRegistry instance (world/entities.py).
                            If provided, unknown resource names are registered there.
                            If None, unknown names are still accepted — just not registered.
        """
        self._registry = resource_registry

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parse(self, raw: str, current_tick: int) -> WorldState:
        """
        Parse a (potentially partial) JSON response into a fresh WorldState.

        Sections present in *raw* are populated and stamped in observed_at.
        Sections absent from *raw* receive safe defaults and are not stamped.
        """
        state = WorldState(tick=current_tick)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("StateParser.parse: invalid JSON — %s. Raw: %r", exc, raw[:200])
            return state

        if not isinstance(data, dict):
            logger.error("StateParser.parse: expected JSON object, got %s", type(data))
            return state

        # Game tick from payload takes precedence if present.
        if "tick" in data:
            state.tick = int(data["tick"])

        self._populate_all(state, data)
        return state

    def parse_partial(self, raw: str, section: str, into: WorldState) -> WorldState:
        """
        Merge a single-section JSON update into an existing WorldState.

        Returns the same *into* object (mutated in place) for convenience.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(
                "StateParser.parse_partial[%s]: invalid JSON — %s", section, exc
            )
            return into

        if not isinstance(data, dict):
            logger.error(
                "StateParser.parse_partial[%s]: expected JSON object", section
            )
            return into

        current_tick = data.get("tick", into.tick)
        if "tick" in data:
            into.tick = int(current_tick)

        # Wrap section data under its key if the payload is just the section body.
        if section not in data:
            data = {section: data}

        self._populate_all(into, data, current_tick=int(current_tick))
        return into

    # ------------------------------------------------------------------
    # Internal dispatcher
    # ------------------------------------------------------------------

    def _populate_all(
        self,
        state: WorldState,
        data: dict[str, Any],
        current_tick: Optional[int] = None,
    ) -> None:
        tick = current_tick if current_tick is not None else state.tick

        if "player" in data:
            state.player = self._parse_player(data["player"])
            state.observed_at["player"] = tick

        if "entities" in data:
            state.entities = self._parse_entities(data["entities"])
            state.observed_at["entities"] = tick

        if "resource_map" in data:
            state.resource_map = self._parse_resource_map(data["resource_map"], tick)
            state.observed_at["resource_map"] = tick

        if "ground_items" in data:
            state.ground_items = self._parse_ground_items(data["ground_items"], tick)
            state.observed_at["ground_items"] = tick

        if "research" in data:
            state.research = self._parse_research(data["research"])
            state.observed_at["research"] = tick

        if "logistics" in data:
            state.logistics = self._parse_logistics(data["logistics"])
            state.observed_at["logistics"] = tick

        if "damaged_entities" in data:
            state.damaged_entities = self._parse_damaged_entities(data["damaged_entities"], tick)
            state.observed_at["damaged_entities"] = tick

        if "destroyed_entities" in data:
            new_events = self._parse_destroyed_entities(data["destroyed_entities"])
            # Merge with existing rolling window, then prune.
            combined = state.destroyed_entities + new_events
            cutoff = tick - state.destroyed_ttl_ticks
            state.destroyed_entities = [
                e for e in combined if e.destroyed_at >= cutoff
            ]
            state.observed_at["destroyed_entities"] = tick

        if "threat" in data:
            state.threat = self._parse_threat(data["threat"])
            state.observed_at["threat"] = tick

    # ------------------------------------------------------------------
    # Section parsers
    # ------------------------------------------------------------------

    def _parse_player(self, d: Any) -> PlayerState:
        if not isinstance(d, dict):
            return PlayerState()
        pos = self._parse_position(d.get("position", {}))
        health = float(d.get("health", 100.0))
        inventory = self._parse_inventory(d.get("inventory", []))
        reachable = [int(x) for x in d.get("reachable", [])]
        return PlayerState(
            position=pos,
            health=health,
            inventory=inventory,
            reachable=reachable,
        )

    def _parse_entities(self, lst: Any) -> list[EntityState]:
        if not isinstance(lst, list):
            return []
        result = []
        for item in lst:
            if not isinstance(item, dict):
                continue
            try:
                entity_id = int(item["unit_number"])
                name = str(item["name"])
                pos = self._parse_position(item.get("position", {}))
                direction = _DIRECTION_MAP.get(int(item.get("direction", 0)), Direction.NORTH)
                status_str = str(item.get("status", "unknown")).lower()
                status = _STATUS_MAP.get(status_str, EntityStatus.UNKNOWN)
                recipe = item.get("recipe") or None
                inventory = (
                    self._parse_inventory(item["inventory"])
                    if "inventory" in item
                    else None
                )
                energy = float(item.get("energy", 0.0))
                result.append(EntityState(
                    entity_id=entity_id,
                    name=name,
                    position=pos,
                    direction=direction,
                    status=status,
                    recipe=recipe,
                    inventory=inventory,
                    energy=energy,
                ))
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping malformed entity: %s — %s", item, exc)
        return result

    def _parse_resource_map(self, lst: Any, tick: int) -> list[ResourcePatch]:
        if not isinstance(lst, list):
            return []
        result = []
        for item in lst:
            if not isinstance(item, dict):
                continue
            try:
                resource_type = str(item["resource_type"])
                # Register unknown resource types in the registry.
                if self._registry is not None:
                    self._registry.ensure(resource_type)
                pos = self._parse_position(item.get("position", {}))
                amount = int(item.get("amount", 0))
                size = int(item.get("size", 0))
                observed_at = int(item.get("observed_at", tick))
                result.append(ResourcePatch(
                    resource_type=resource_type,
                    position=pos,
                    amount=amount,
                    size=size,
                    observed_at=observed_at,
                ))
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping malformed resource patch: %s — %s", item, exc)
        return result

    def _parse_ground_items(self, lst: Any, tick: int) -> list[GroundItem]:
        if not isinstance(lst, list):
            return []
        result = []
        for item in lst:
            if not isinstance(item, dict):
                continue
            try:
                result.append(GroundItem(
                    item=str(item["item"]),
                    position=self._parse_position(item.get("position", {})),
                    count=int(item.get("count", 1)),
                    observed_at=int(item.get("observed_at", tick)),
                    age_ticks=int(item.get("age_ticks", 0)),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping malformed ground item: %s — %s", item, exc)
        return result

    def _parse_research(self, d: Any) -> ResearchState:
        if not isinstance(d, dict):
            return ResearchState()
        unlocked = [str(t) for t in d.get("unlocked", [])]
        in_progress = d.get("in_progress") or None
        queued = [str(t) for t in d.get("queued", [])]
        science_per_minute: dict[str, float] = {}
        for k, v in d.get("science_per_minute", {}).items():
            science_per_minute[str(k)] = float(v)
        return ResearchState(
            unlocked=unlocked,
            in_progress=in_progress,
            queued=queued,
            science_per_minute=science_per_minute,
        )

    def _parse_logistics(self, d: Any) -> LogisticsState:
        if not isinstance(d, dict):
            return LogisticsState()

        belts: list[BeltSegment] = []
        for seg in d.get("belts", []):
            if not isinstance(seg, dict):
                continue
            try:
                positions = [self._parse_position(p) for p in seg.get("positions", [])]
                belts.append(BeltSegment(
                    segment_id=int(seg["segment_id"]),
                    positions=positions,
                    lane1=self._parse_belt_lane(seg.get("lane1", {})),
                    lane2=self._parse_belt_lane(seg.get("lane2", {})),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping malformed belt segment: %s — %s", seg, exc)

        power_raw = d.get("power", {})
        power = PowerGrid(
            produced_kw=float(power_raw.get("produced_kw", 0.0)),
            consumed_kw=float(power_raw.get("consumed_kw", 0.0)),
            accumulated_kj=float(power_raw.get("accumulated_kj", 0.0)),
            satisfaction=float(power_raw.get("satisfaction", 1.0)),
        )

        inserters: dict[int, InserterState] = {}
        inserter_activity: dict[int, int] = {}
        for k, v in d.get("inserter_activity", {}).items():
            try:
                entity_id = int(k)
                if isinstance(v, dict):
                    # New format: {active, pickup_position, drop_position}
                    active = bool(v.get("active", False))
                    pickup_position = self._parse_optional_position(
                        v.get("pickup_position")
                    )
                    drop_position = self._parse_optional_position(
                        v.get("drop_position")
                    )
                    ins = InserterState(
                        entity_id=entity_id,
                        position=Position(0.0, 0.0),  # not in logistics payload
                        active=active,
                        pickup_position=pickup_position,
                        drop_position=drop_position,
                    )
                    inserters[entity_id] = ins
                    inserter_activity[entity_id] = 1 if active else 0
                else:
                    # Legacy format: bare 0/1 integer
                    activity = int(v)
                    inserter_activity[entity_id] = activity
            except (ValueError, TypeError):
                pass

        return LogisticsState(
            belts=belts,
            power=power,
            inserters=inserters,
            inserter_activity=inserter_activity,
        )

    def _parse_damaged_entities(self, lst: Any, tick: int) -> list[DamagedEntity]:
        if not isinstance(lst, list):
            return []
        result = []
        for item in lst:
            if not isinstance(item, dict):
                continue
            try:
                health_fraction = float(item.get("health_fraction", 0.5))
                # Clamp to the valid open interval (0.0, 1.0).
                health_fraction = max(0.001, min(0.999, health_fraction))
                result.append(DamagedEntity(
                    entity_id=int(item["entity_id"]),
                    name=str(item["name"]),
                    position=self._parse_position(item.get("position", {})),
                    health_fraction=health_fraction,
                    observed_at=int(item.get("observed_at", tick)),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping malformed damaged entity: %s — %s", item, exc)
        return result

    def _parse_destroyed_entities(self, lst: Any) -> list[DestroyedEntity]:
        if not isinstance(lst, list):
            return []
        result = []
        for item in lst:
            if not isinstance(item, dict):
                continue
            try:
                cause = str(item.get("cause", "unknown")).lower()
                if cause not in _VALID_CAUSES:
                    cause = "unknown"
                result.append(DestroyedEntity(
                    name=str(item["name"]),
                    position=self._parse_position(item.get("position", {})),
                    destroyed_at=int(item["destroyed_at"]),
                    cause=cause,
                ))
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping malformed destroyed entity: %s — %s", item, exc)
        return result

    def _parse_threat(self, d: Any) -> ThreatState:
        if not isinstance(d, dict):
            return ThreatState()

        bases: list[BiterBase] = []
        for b in d.get("biter_bases", []):
            if not isinstance(b, dict):
                continue
            try:
                bases.append(BiterBase(
                    base_id=int(b["base_id"]),
                    position=self._parse_position(b.get("position", {})),
                    size=int(b.get("size", 0)),
                    evolution=float(b.get("evolution", 0.0)),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping malformed biter base: %s — %s", b, exc)

        pollution_cloud = [
            self._parse_position(p)
            for p in d.get("pollution_cloud", [])
            if isinstance(p, dict)
        ]

        attack_timers: dict[int, float] = {}
        for k, v in d.get("attack_timers", {}).items():
            try:
                attack_timers[int(k)] = float(v)
            except (ValueError, TypeError):
                pass

        return ThreatState(
            biter_bases=bases,
            pollution_cloud=pollution_cloud,
            attack_timers=attack_timers,
            evolution_factor=float(d.get("evolution_factor", 0.0)),
        )

    # ------------------------------------------------------------------
    # Primitive helpers
    # ------------------------------------------------------------------

    def _parse_belt_lane(self, d: Any) -> BeltLane:
        """Parse a single belt transport line lane from Lua output."""
        if not isinstance(d, dict):
            return BeltLane()
        congested = bool(d.get("congested", False))
        raw_items = d.get("items", {})
        items: dict[str, int] = {}
        if isinstance(raw_items, dict):
            for name, count in raw_items.items():
                try:
                    items[str(name)] = int(count)
                except (ValueError, TypeError):
                    pass
        return BeltLane(congested=congested, items=items)

    def _parse_optional_position(self, d: Any) -> Optional[Position]:
        """Parse a position dict, returning None if absent or malformed."""
        if not isinstance(d, dict):
            return None
        try:
            return Position(x=float(d.get("x", 0.0)), y=float(d.get("y", 0.0)))
        except (ValueError, TypeError):
            return None

    def _parse_position(self, d: Any) -> Position:
        if isinstance(d, dict):
            try:
                return Position(x=float(d.get("x", 0.0)), y=float(d.get("y", 0.0)))
            except (ValueError, TypeError):
                pass
        return Position(0.0, 0.0)

    def _parse_inventory(self, lst: Any) -> Inventory:
        if not isinstance(lst, list):
            return Inventory()
        slots = []
        for slot in lst:
            if not isinstance(slot, dict):
                continue
            try:
                slots.append(InventorySlot(
                    item=str(slot["item"]),
                    count=int(slot["count"]),
                ))
            except (KeyError, ValueError, TypeError):
                pass
        return Inventory(slots=slots)