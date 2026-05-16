"""
bridge/actions.py

Action — the structured command type that separates the agent from the bridge.

Nothing outside bridge/ speaks RCON or knows about Lua. The execution layer
(agent/execution.py) and primitives (agent/primitives/) construct Action objects.
bridge/action_executor.py translates them into RCON commands.

Rules
-----
- Pure data. No LLM calls. No RCON.
- Every concrete action is a frozen dataclass that inherits from Action.
- Every concrete action declares a CLASS-LEVEL `category: ActionCategory` so the
  execution layer can filter by context (e.g. no COMBAT actions when biters off,
  no VEHICLE actions when not in a vehicle).
- The `kind` property is the class name — used as a discriminator by the executor.
- All coordinate types use Position from world/state.py to keep units consistent.

Scope of this file
------------------
These are PRIMITIVE actions — single atomic operations the bridge can execute in
one RCON round-trip. Multi-step sequences (walk to patch, mine N ore, return to
base) are composed by agent/primitives/ and agent/execution.py, not here.

Known omissions
---------------
Circuit networks: Factorio's circuit network is a full embedded programming
language (wire connections, combinator logic, signal vocabularies, memory cells).
Modelling it requires its own type hierarchy and is intentionally deferred. The
agent can play a complete game — including defence and Space Age content — without
circuit networks. Adding them later means adding new Action subclasses (ConnectWire,
SetCombinatorCondition, etc.) and extending the Lua mod; no existing code changes.

Railroad networks: placing rails, signals, train stops, and locomotives is handled
by the existing PlaceEntity + RotateEntity actions. What is missing is train
*operation*: setting schedules, configuring station departure conditions, and
sending commands to specific trains. The planned action types are:

  SetTrainSchedule(train_id, schedule: list[StopCondition])
      Set a locomotive's full schedule — ordered list of station names and the
      conditions under which the train departs each stop (item count, circuit
      signal, inactivity timer, etc.).

  SetStationCondition(entity_id, condition: StationCondition)
      Set the departure or wait condition on a train stop entity independently
      of any specific train's schedule. Used for globally configuring stops.

These require a WorldState.trains section (new Lua polling surface for train
identity, schedule, cargo, current stop, and speed) and a new NON-PROXIMAL
reward namespace entry for train state conditions. The self-model already
anticipates this: TRAIN_STATION is a NodeType and CONNECTED_BY_RAIL is an
EdgeType. Implementation is deferred to a dedicated phase after the
spatial-logistics agent (Phase 9) is stable.

Combat and vehicle actions are present as stubs (see COMBAT and VEHICLE sections).
They are included in ALL_ACTION_TYPES but the execution layer gates them by
category, so they are never emitted when biters are disabled or no vehicle is
occupied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from world.state import Direction, Position


# ---------------------------------------------------------------------------
# ActionCategory
# ---------------------------------------------------------------------------

class ActionCategory(Enum):
    """
    Broad category for each action type.

    Used by the execution layer to decide which actions are valid in the
    current context:

      MOVEMENT  — always available (on foot)
      MINING    — always available
      CRAFTING  — always available (player crafting queue)
      BUILDING  — always available
      INVENTORY — always available
      RESEARCH  — always available
      PLAYER    — always available (equipment, consumables)
      VEHICLE   — only when player is currently occupying a vehicle
      COMBAT    — intended for use when BITERS_ENABLED=True, but the category
                  exists unconditionally; the execution layer enforces the gate
      META      — always available (Wait, NoOp, etc.)
    """
    MOVEMENT  = "movement"
    MINING    = "mining"
    CRAFTING  = "crafting"
    BUILDING  = "building"
    INVENTORY = "inventory"
    RESEARCH  = "research"
    PLAYER    = "player"
    VEHICLE   = "vehicle"
    COMBAT    = "combat"
    META      = "meta"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    """
    Abstract base — never instantiate directly.

    Every subclass must declare a class-level `category: ActionCategory`.
    The `kind` property returns the class name for use as an executor
    discriminator.
    """

    @property
    def kind(self) -> str:
        return type(self).__name__

    @property
    def category(self) -> ActionCategory:
        # Subclasses override this at the class level; this fallback exists
        # only so the base class is valid Python.
        raise NotImplementedError(f"{type(self).__name__} must declare category")


# ---------------------------------------------------------------------------
# MOVEMENT
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveTo(Action):
    """
    Walk the player character (on foot) to a map position.

    Not valid inside a vehicle — use DriveVehicle instead, which has
    different movement physics and collision behaviour.
    """
    position: Position
    pathfind: bool = True   # True: route around obstacles; False: walk direct
    category: ActionCategory = field(default=ActionCategory.MOVEMENT, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class StopMovement(Action):
    """Halt all on-foot movement immediately."""
    category: ActionCategory = field(default=ActionCategory.MOVEMENT, init=False, repr=False, compare=False)


# ---------------------------------------------------------------------------
# MINING
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MineResource(Action):
    """
    Mine a resource tile at the given position.

    count=0 means mine until inventory is full or the patch is exhausted.
    The bridge issues repeated mining commands; the execution layer monitors
    progress via WorldState.
    """
    position: Position
    resource: str        # Factorio internal resource name, e.g. "iron-ore"
    count: int = 1       # 0 = mine until full / exhausted
    category: ActionCategory = field(default=ActionCategory.MINING, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class MineEntity(Action):
    """
    Deconstruct (mine) a placed entity, returning its items to inventory.

    The entity must be within the player's reach radius.
    """
    entity_id: int
    category: ActionCategory = field(default=ActionCategory.MINING, init=False, repr=False, compare=False)


# ---------------------------------------------------------------------------
# CRAFTING
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CraftItem(Action):
    """
    Queue hand-crafting of an item.

    Uses the player's crafting queue, not an assembler. All ingredient items
    must already be in the player's inventory (or recursively hand-craftable).
    """
    recipe: str      # Factorio recipe name
    count: int = 1
    category: ActionCategory = field(default=ActionCategory.CRAFTING, init=False, repr=False, compare=False)


# ---------------------------------------------------------------------------
# BUILDING
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlaceEntity(Action):
    """Place an item from inventory onto the map as a building."""
    item: str            # Item to place (must be in player inventory)
    position: Position
    direction: Direction = Direction.NORTH
    category: ActionCategory = field(default=ActionCategory.BUILDING, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class SetRecipe(Action):
    """Set the active recipe of an assembler, chemical plant, oil refinery, etc."""
    entity_id: int
    recipe: str
    category: ActionCategory = field(default=ActionCategory.BUILDING, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class SetFilter(Action):
    """
    Set a filter slot on a filter inserter, splitter, or loader.

    slot is 0-indexed. item is the Factorio internal item name to filter for,
    or "" to clear the slot.
    """
    entity_id: int
    slot: int
    item: str            # "" to clear
    category: ActionCategory = field(default=ActionCategory.BUILDING, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class SetSplitterPriority(Action):
    """
    Set the input and/or output priority lanes of a splitter.

    Factorio splitters have two independently configurable priority settings:

      input_priority  — which input belt is preferentially consumed when both
                        are active. "left", "right", or "none" (balanced).
      output_priority — which output belt is preferentially filled before the
                        other receives items. "left", "right", or "none" (balanced).

    Either field may be None to leave that priority unchanged. Both may be set
    in one call.

    Output priority is the primary mechanism for allocating a node's output to
    a specific downstream consumer without circuit networks — the spatial-logistics
    agent uses it when wiring a production node to multiple consumers with
    different priority claims (e.g. preferentially feed a downstream assembler
    over a buffer chest).

    Input and output priority interact with SetFilter: a filtered output lane
    with output priority set will receive priority items-of-that-type first.
    The bridge sets both properties independently via the Lua API
    (entity.input_priority / entity.output_priority).

    Valid values: "left", "right", "none". None (Python) means leave unchanged.
    """
    entity_id: int
    input_priority:  Optional[str] = None   # "left" | "right" | "none" | None
    output_priority: Optional[str] = None   # "left" | "right" | "none" | None
    category: ActionCategory = field(default=ActionCategory.BUILDING, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        valid = {"left", "right", "none", None}
        if self.input_priority not in valid:
            raise ValueError(
                f"input_priority must be 'left', 'right', 'none', or None; "
                f"got {self.input_priority!r}"
            )
        if self.output_priority not in valid:
            raise ValueError(
                f"output_priority must be 'left', 'right', 'none', or None; "
                f"got {self.output_priority!r}"
            )
        if self.input_priority is None and self.output_priority is None:
            raise ValueError(
                "SetSplitterPriority requires at least one of input_priority "
                "or output_priority to be set"
            )


@dataclass(frozen=True)
class RotateEntity(Action):
    """
    Rotate a placed entity by 90 degrees clockwise (default) or
    counter-clockwise.

    Factorio's entity.rotate() advances the entity's direction by one step
    in the Direction enum: NORTH → EAST → SOUTH → WEST → NORTH.
    reverse=True rotates counter-clockwise instead.

    The entity must be within the player's reach radius and must support
    rotation (not all entities do — e.g. chests are directionless).
    The bridge validates rotatability before dispatching.
    """
    entity_id: int
    reverse: bool = False    # False = clockwise; True = counter-clockwise
    category: ActionCategory = field(default=ActionCategory.BUILDING, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class FlipEntity(Action):
    """
    Flip (mirror) a placed entity about a horizontal or vertical axis.

    Produces an orientation not achievable through rotation alone. Primarily
    useful for fluid-handling buildings with asymmetric pipe connections:
    oil refineries, chemical plants, and similar structures where the pipe
    layout on one side differs from the other.

    Not all entity types support flip. The bridge returns ok=false with
    reason="flip_not_supported" when the entity cannot be mirrored. The
    execution layer should treat this as a non-fatal failure and fall back to
    rotation if an alternative orientation is acceptable.

    Parameters
    ----------
    entity_id  : unit_number of the entity to flip.
    horizontal : True = flip horizontally (mirror left↔right across the
                 vertical axis). False = flip vertically (mirror top↔bottom
                 across the horizontal axis). Defaults to True as this is the
                 more common case for pipe layout adjustment.
    """
    entity_id: int
    horizontal: bool = True
    category: ActionCategory = field(default=ActionCategory.BUILDING, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class ApplyBlueprint(Action):
    """
    Paste a blueprint string at a map position.

    position is the top-left anchor tile; the bridge normalises to Factorio's
    coordinate convention. force_build=True deconstructs any overlapping
    entities before placing — use with caution outside safe test environments.

    Blueprint strings are the standard Factorio exchange format (base64-encoded
    zlib-compressed JSON). The agent obtains them from
    memory/blueprint_library/library.json or from the rich examiner's extraction
    output.
    """
    blueprint_string: str    # Factorio blueprint exchange string
    position: Position        # Top-left anchor tile
    direction: Direction = Direction.NORTH
    force_build: bool = False
    category: ActionCategory = field(default=ActionCategory.BUILDING, init=False, repr=False, compare=False)


# ---------------------------------------------------------------------------
# INVENTORY
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransferItems(Action):
    """
    Move items between the player's inventory and a nearby container or machine.

    direction controls which way items flow:
      "to_entity"   — player inventory → entity
      "from_entity" — entity → player inventory

    count=-1 means transfer all available.
    """
    entity_id: int
    item: str
    count: int
    direction: str = "to_entity"    # "to_entity" | "from_entity"
    category: ActionCategory = field(default=ActionCategory.INVENTORY, init=False, repr=False, compare=False)


# ---------------------------------------------------------------------------
# RESEARCH
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SetResearchQueue(Action):
    """
    Replace the current research queue with the given ordered technology list.

    First entry is researched next. An empty list cancels current research.
    Technologies must be unlockable given the current research state; the
    execution layer is responsible for validating prerequisites.
    """
    technologies: list[str]    # Ordered; first = research next
    category: ActionCategory = field(default=ActionCategory.RESEARCH, init=False, repr=False, compare=False)


# ---------------------------------------------------------------------------
# PLAYER  (equipment, consumables, character state)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EquipArmor(Action):
    """
    Equip an armor item from inventory into the player's armor slot.

    If the slot is already occupied the existing armor is swapped to inventory.
    The item must be an armor type; the bridge validates this before sending.
    """
    item: str    # Factorio internal armor item name, e.g. "heavy-armor"
    category: ActionCategory = field(default=ActionCategory.PLAYER, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class UseItem(Action):
    """
    Use a consumable or throwable item.

    Covers a broad set of one-shot uses:
      - Healing:    "raw-fish", "medical-pack" (modded)
      - Capsules:   "poison-capsule", "slowdown-capsule", etc.
      - Grenades:   "grenade", "cluster-grenade"
      - Other:      any item with a use-on-self or use-on-ground activation

    target_position is required for thrown/targeted items (capsules, grenades).
    Leave as None for self-use items (fish).

    Note: grenades and offensive capsules are combat actions in intent but
    categorised as PLAYER because they can be used without biters being
    present (e.g. clearing trees). The execution layer may apply additional
    context gates.
    """
    item: str
    target_position: Optional[Position] = None
    category: ActionCategory = field(default=ActionCategory.PLAYER, init=False, repr=False, compare=False)


# ---------------------------------------------------------------------------
# VEHICLE  (only valid when player is occupying a vehicle)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnterVehicle(Action):
    """
    Board a nearby vehicle (car, tank, spidertron, train).

    The vehicle entity must be within reach. Once boarded, MOVEMENT actions
    are replaced by DriveVehicle; standard MoveTo is no longer valid until
    ExitVehicle is issued.
    """
    entity_id: int    # Vehicle entity to board
    category: ActionCategory = field(default=ActionCategory.VEHICLE, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class ExitVehicle(Action):
    """
    Dismount the currently occupied vehicle.

    The player is placed adjacent to the vehicle. After this action, standard
    on-foot MOVEMENT actions become valid again.
    """
    category: ActionCategory = field(default=ActionCategory.VEHICLE, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class DriveVehicle(Action):
    """
    Drive the currently occupied vehicle to a map position.

    Semantically equivalent to MoveTo but uses vehicle movement physics:
    - Vehicles have momentum and turning radius (cars, tanks).
    - Spidertrons pathfind differently and can traverse water.
    - Trains follow rails and cannot freeform navigate.
    - Collision with entities causes damage to both vehicle and obstacle.

    The bridge implementation for each vehicle type is distinct. The execution
    layer is responsible for knowing which vehicle is occupied and emitting the
    appropriate sub-command.

    pathfind=False is risky for vehicles with momentum — prefer True except in
    open terrain or for spidertrons.
    """
    position: Position
    pathfind: bool = True
    category: ActionCategory = field(default=ActionCategory.VEHICLE, init=False, repr=False, compare=False)


# ---------------------------------------------------------------------------
# COMBAT  (intended for BITERS_ENABLED=True; category gates emission)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectWeapon(Action):
    """
    Switch to a specific weapon in the player's weapon slots.

    slot is 0-indexed (slot 0 = primary, slot 1 = secondary, etc.).
    The weapon item must already be in the corresponding equipment slot.
    """
    slot: int    # 0-indexed weapon slot
    category: ActionCategory = field(default=ActionCategory.COMBAT, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class ShootAt(Action):
    """
    Fire the currently selected weapon at a target.

    Exactly one of target_entity_id or target_position must be provided:
      target_entity_id — track and shoot a specific unit (biters, worms, etc.)
      target_position  — shoot at a fixed map coordinate (area denial, structures)

    The bridge maps this to Factorio's shooting_state and attack_target APIs.
    The execution layer is responsible for selecting the appropriate form based
    on whether a specific entity is being tracked or an area is being suppressed.
    """
    target_entity_id: Optional[int] = None
    target_position: Optional[Position] = None
    category: ActionCategory = field(default=ActionCategory.COMBAT, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (self.target_entity_id is None) == (self.target_position is None):
            raise ValueError(
                "ShootAt requires exactly one of target_entity_id or target_position, "
                f"got entity_id={self.target_entity_id!r}, position={self.target_position!r}"
            )


@dataclass(frozen=True)
class StopShooting(Action):
    """
    Stop firing. Issued when a combat engagement ends or is interrupted.
    """
    category: ActionCategory = field(default=ActionCategory.COMBAT, init=False, repr=False, compare=False)


# ---------------------------------------------------------------------------
# META
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Wait(Action):
    """
    Do nothing for a specified number of game ticks.

    The execution layer emits this when polling for a condition (crafting
    timer, belt fill, research completion) rather than busy-looping. The
    bridge does not send any RCON command — it simply sleeps for the
    equivalent real-world duration before the next poll.
    """
    ticks: int
    category: ActionCategory = field(default=ActionCategory.META, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class NoOp(Action):
    """Explicit no-operation. Used by stub modules, tests, and safe fallbacks."""
    category: ActionCategory = field(default=ActionCategory.META, init=False, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Registry and helpers
# ---------------------------------------------------------------------------

ALL_ACTION_TYPES: tuple[type[Action], ...] = (
    # MOVEMENT
    MoveTo,
    StopMovement,
    # MINING
    MineResource,
    MineEntity,
    # CRAFTING
    CraftItem,
    # BUILDING
    PlaceEntity,
    SetRecipe,
    SetFilter,
    SetSplitterPriority,
    RotateEntity,
    FlipEntity,
    ApplyBlueprint,
    # INVENTORY
    TransferItems,
    # RESEARCH
    SetResearchQueue,
    # PLAYER
    EquipArmor,
    UseItem,
    # VEHICLE
    EnterVehicle,
    ExitVehicle,
    DriveVehicle,
    # COMBAT
    SelectWeapon,
    ShootAt,
    StopShooting,
    # META
    Wait,
    NoOp,
)

# Map from ActionCategory to the action types that belong to it.
# Built once at import time; used by the execution layer for context filtering.
import dataclasses as _dc

ACTIONS_BY_CATEGORY: dict[ActionCategory, tuple[type[Action], ...]] = {}
for _action_type in ALL_ACTION_TYPES:
    # category is always the last field, declared with a fixed default.
    _cat = _dc.fields(_action_type)[-1].default  # ActionCategory instance
    ACTIONS_BY_CATEGORY.setdefault(_cat, ())
    ACTIONS_BY_CATEGORY[_cat] = ACTIONS_BY_CATEGORY[_cat] + (_action_type,)


def actions_for_context(
    in_vehicle: bool = False,
    biters_enabled: bool = False,
) -> tuple[type[Action], ...]:
    """
    Return the subset of action types valid in the current execution context.

    Parameters
    ----------
    in_vehicle      : True when the player is currently occupying a vehicle.
                      Gates VEHICLE category actions (valid) and excludes
                      on-foot MOVEMENT actions (MoveTo, StopMovement).
    biters_enabled  : True when BITERS_ENABLED=True in config.
                      COMBAT actions are included only when True.

    The execution layer calls this at goal-start and after any context change
    (boarding/exiting a vehicle, biter config toggle) to refresh its valid
    action set.
    """
    excluded: set[ActionCategory] = set()

    if in_vehicle:
        excluded.add(ActionCategory.MOVEMENT)   # use DriveVehicle instead
    else:
        excluded.add(ActionCategory.VEHICLE)    # not in a vehicle

    if not biters_enabled:
        excluded.add(ActionCategory.COMBAT)

    return tuple(
        action_type
        for action_type in ALL_ACTION_TYPES
        if action_type not in _get_excluded_types(excluded)
    )


def _get_excluded_types(
    excluded_categories: set[ActionCategory],
) -> set[type[Action]]:
    result: set[type[Action]] = set()
    for cat in excluded_categories:
        result.update(ACTIONS_BY_CATEGORY.get(cat, ()))
    return result