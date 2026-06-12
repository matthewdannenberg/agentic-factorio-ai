"""
world/__init__.py

The world layer — everything the system knows about the game at any moment.

Three knowledge systems, each in its own subfolder:

  world/observable/   Scan-radius-limited game state. Updated every tick.
                      Read via WorldQuery; written via WorldWriter.

  world/knowledge/    Static Factorio content (entities, recipes, techs,
                      resources). Learned at runtime; persists across runs.

  world/model/        Coordinator's persistent factory graph (SelfModel).
                      Global, non-proximal. Updated via SelfModelPatch.

Public surface
--------------
All imports from world/ go through this file. Importing from submodules
directly is not permitted outside world/ itself — the one exception is
bridge/state_parser.py and bridge/actions.py, which must import from
world.observable.state for WorldState construction and Action coordinates.

WorldState is intentionally NOT exported here. External consumers read
through WorldQuery and write through WorldWriter only.
"""

# --- Observable state: read / write interfaces ------------------------------

from world.observable.query import WorldQuery, ChunkMapQuery, BBoxQuery
from world.observable.writer import WorldWriter
from world.observable.production_tracker import ProductionTracker, ProductionTrackerProtocol, ProductionSummary

# --- Game knowledge ---------------------------------------------------------

from world.knowledge.base import KnowledgeBase
from world.knowledge.tech_tree import TechTree

# --- Factory self-model -----------------------------------------------------

from world.model.self_model import (
    SelfModel,
    FactoryNode,
    FactoryEdge,
    NodeType,
    NodeStatus,
    EdgeType,
    ProcessType,
    BoundingBox,
    IOPoint,
    NodeId,
)
from world.model.patch import SelfModelPatch, PatchLayer, PatchAction

# --- Observable state dataclasses -------------------------------------------
# These appear in external method signatures and must be importable without
# going through WorldQuery. WorldState itself is NOT exported.

from world.observable.state import (
    # Primitives
    Position,
    Direction,
    ChunkCoord,
    # Entities
    EntityState,
    EntityStatus,
    # Resources
    ResourcePatch,
    ResourceType,
    # Inventory
    InventorySlot,
    Inventory,
    # Research
    ResearchState,
    # Logistics
    LogisticsState,
    BeltSegment,
    BeltLane,
    PowerGrid,
    InserterState,
    # Exploration
    ExplorationState,
    # Ground items
    GroundItem,
    # Damage / destruction
    DamagedEntity,
    DestroyedEntity,
    # Threat
    BiterBase,
    ThreatState,
    # Crafting queue
    CraftingQueueEntry,
    # Player
    PlayerState,
)

# --- Knowledge record types -------------------------------------------------
# Appear in signatures throughout execution and planning layers.

from world.knowledge.base import (
    EntityRecord,
    EntityCategory,
    ResourceRecord,
    FluidRecord,
    RecipeRecord,
    IngredientRecord,
    ItemRecord,
    ProductRecord,
    TechRecord,
)

# --- Bridge wiring helpers --------------------------------------------------
# ResourceRegistry is used by bridge/state_parser.py at construction time.
# Exported here so run.py has a single world import for wiring.

from world.knowledge.entities import ResourceRegistry

__all__ = [
    # Observable — interfaces
    "WorldQuery",
    "ChunkMapQuery",
    "BBoxQuery",
    "WorldWriter",
    "ProductionTracker",
    "ProductionTrackerProtocol",
    "ProductionSummary",
    # Knowledge
    "KnowledgeBase",
    "TechTree",
    # Self-model
    "SelfModel",
    "FactoryNode",
    "FactoryEdge",
    "NodeType",
    "NodeStatus",
    "EdgeType",
    "ProcessType",
    "BoundingBox",
    "IOPoint",
    "NodeId",
    "SelfModelPatch",
    "PatchLayer",
    "PatchAction",
    # Observable state dataclasses
    "Position",
    "Direction",
    "ChunkCoord",
    "CraftingQueueEntry",
    "EntityState",
    "EntityStatus",
    "ResourcePatch",
    "ResourceType",
    "InventorySlot",
    "Inventory",
    "ResearchState",
    "LogisticsState",
    "BeltSegment",
    "BeltLane",
    "PowerGrid",
    "InserterState",
    "ExplorationState",
    "GroundItem",
    "DamagedEntity",
    "DestroyedEntity",
    "BiterBase",
    "ThreatState",
    "PlayerState",
    # Knowledge record types
    "EntityRecord",
    "EntityCategory",
    "ResourceRecord",
    "FluidRecord",
    "RecipeRecord",
    "TechRecord",
    "IngredientRecord",
    "ItemRecord",
    "ProductRecord",
    # Bridge wiring
    "ResourceRegistry",
]