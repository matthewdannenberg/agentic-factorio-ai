"""
bridge/__init__.py

The bridge layer — the sole part of the codebase that speaks RCON or knows Lua.

Public surface
--------------
All imports from bridge/ must go through this file. Internal modules
(wire protocol, Lua command formatting, executor dispatch) are private
implementation details. Importing directly from bridge submodules is
not permitted outside of bridge/ itself.

Permitted consumers and what they use
--------------------------------------
  world/       — PrototypeQueryFn (type alias for KnowledgeBase.__init__)
  execution/   — Action types, ActionCategory helpers, ActionExecutor
  run.py       — RconClient, BridgeError, WorldPoller, StateParser,
                 make_prototype_query_fn, ActionExecutor (for wiring)

Layering note
-------------
bridge/actions.py imports Position and Direction from world/state.py.
This is the one permitted upward import: Position and Direction are
coordinate primitives with no operational logic, and Action objects are
meaningless without them. No other bridge module imports from world/.
"""

# --- Transport ----------------------------------------------------------------

from bridge.rcon_client import RconClient, BridgeError

# --- World interaction --------------------------------------------------------

from bridge.world_poller import WorldPoller
from bridge.state_parser import StateParser

# --- Prototype queries --------------------------------------------------------
# make_prototype_query_fn is called once at startup in run.py to produce the
# callable injected into KnowledgeBase. PrototypeQueryFn is the type alias
# used in KnowledgeBase.__init__'s signature.

from bridge.prototype_query import make_prototype_query_fn, PrototypeQueryFn

# --- Action execution ---------------------------------------------------------

from bridge.action_executor import ActionExecutor

# --- Actions — base and registry ----------------------------------------------

from bridge.actions import (
    Action,
    ActionCategory,
    ALL_ACTION_TYPES,
    ACTIONS_BY_CATEGORY,
    actions_for_context,
)

# --- Actions — concrete types -------------------------------------------------

from bridge.actions import (
    # MOVEMENT
    MoveTo,
    StopMovement,
    # MINING
    MineResource,
    MineEntity,
    StopMining,
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
    # VEHICLE (stubs — gated by actions_for_context)
    EnterVehicle,
    ExitVehicle,
    DriveVehicle,
    # COMBAT (stubs — gated by actions_for_context / BITERS_ENABLED)
    SelectWeapon,
    ShootAt,
    StopShooting,
    # META
    Wait,
    NoOp,
)

__all__ = [
    # Transport
    "RconClient",
    "BridgeError",
    # World interaction
    "WorldPoller",
    "StateParser",
    # Prototype queries
    "make_prototype_query_fn",
    "PrototypeQueryFn",
    # Execution
    "ActionExecutor",
    # Actions — base and registry
    "Action",
    "ActionCategory",
    "ALL_ACTION_TYPES",
    "ACTIONS_BY_CATEGORY",
    "actions_for_context",
    # Actions — movement
    "MoveTo",
    "StopMovement",
    # Actions — mining
    "MineResource",
    "MineEntity",
    "StopMining",
    # Actions — crafting
    "CraftItem",
    # Actions — building
    "PlaceEntity",
    "SetRecipe",
    "SetFilter",
    "SetSplitterPriority",
    "RotateEntity",
    "FlipEntity",
    "ApplyBlueprint",
    # Actions — inventory
    "TransferItems",
    # Actions — research
    "SetResearchQueue",
    # Actions — player
    "EquipArmor",
    "UseItem",
    # Actions — vehicle
    "EnterVehicle",
    "ExitVehicle",
    "DriveVehicle",
    # Actions — combat
    "SelectWeapon",
    "ShootAt",
    "StopShooting",
    # Actions — meta
    "Wait",
    "NoOp",
]
