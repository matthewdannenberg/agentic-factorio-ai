"""
bridge/actions.py

Action — the structured command type that separates the agent from the bridge.

Nothing outside bridge/ speaks RCON or knows about Lua. The execution layer
(agent/execution.py) and primitives (agent/primitives/) construct Action objects.
bridge/action_executor.py translates them into RCON commands.

Rules:
- Pure data. No LLM calls. No RCON.
- Every concrete action is a frozen dataclass that inherits from Action.
- The discriminator field `kind` is set automatically via __init_subclass__
  so pattern-matching in the executor is straightforward.
- All coordinate types use Position from world/state.py to keep units consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from world.state import Direction, Position


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    """Abstract base — never instantiate directly."""

    @property
    def kind(self) -> str:
        return type(self).__name__


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveTo(Action):
    """Walk the player character to a map position."""
    position: Position
    # If True, pathfind around obstacles; if False, walk direct (risky but fast)
    pathfind: bool = True


@dataclass(frozen=True)
class StopMovement(Action):
    """Halt all movement immediately."""


# ---------------------------------------------------------------------------
# Mining / resource collection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MineResource(Action):
    """Start mining a resource tile at the given position."""
    position: Position
    resource: str   # Factorio item name, e.g. "iron-ore"
    count: int = 1  # How many to mine before stopping (0 = mine until full)


@dataclass(frozen=True)
class MineEntity(Action):
    """Deconstruct (mine) a placed entity."""
    entity_id: int


# ---------------------------------------------------------------------------
# Crafting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CraftItem(Action):
    """Hand-craft an item in the player's crafting queue."""
    recipe: str     # Factorio recipe name
    count: int = 1


# ---------------------------------------------------------------------------
# Building / placing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlaceEntity(Action):
    """Place an item from inventory onto the map."""
    item: str           # Item to place (must be in inventory)
    position: Position
    direction: Direction = Direction.NORTH


@dataclass(frozen=True)
class SetRecipe(Action):
    """Set the active recipe of an assembler, chemical plant, etc."""
    entity_id: int
    recipe: str


@dataclass(frozen=True)
class SetFilter(Action):
    """Set a filter slot on an inserter, splitter, or filter inserter."""
    entity_id: int
    slot: int
    item: str


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApplyBlueprint(Action):
    """
    Paste a blueprint string at a map position.
    The bridge translates this into the Lua API call.
    """
    blueprint_string: str   # Full Factorio blueprint exchange string
    position: Position       # Top-left anchor or centre (bridge normalises)
    direction: Direction = Direction.NORTH
    force_build: bool = False  # If True, deconstruct overlapping entities first


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SetResearchQueue(Action):
    """Replace the current research queue with the given ordered list."""
    technologies: list[str]   # Ordered; first = research next


# ---------------------------------------------------------------------------
# Inventory / item management
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransferItems(Action):
    """Move items between player inventory and a nearby container/machine."""
    entity_id: int
    item: str
    count: int
    direction: str = "to_entity"   # "to_entity" | "from_entity"


# ---------------------------------------------------------------------------
# Meta / control
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Wait(Action):
    """
    Do nothing for a number of game ticks.
    The execution layer emits this when it is waiting on a crafting timer,
    belt fill, etc., rather than busy-looping.
    """
    ticks: int


@dataclass(frozen=True)
class NoOp(Action):
    """Explicit no-operation — used by stub modules and tests."""


# ---------------------------------------------------------------------------
# Action type registry — all concrete types in one place
# ---------------------------------------------------------------------------

ALL_ACTION_TYPES: tuple[type[Action], ...] = (
    MoveTo,
    StopMovement,
    MineResource,
    MineEntity,
    CraftItem,
    PlaceEntity,
    SetRecipe,
    SetFilter,
    ApplyBlueprint,
    SetResearchQueue,
    TransferItems,
    Wait,
    NoOp,
)
