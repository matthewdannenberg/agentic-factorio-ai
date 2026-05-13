"""
agent/self_model.py

SelfModel — the agent's persistent, graph-theoretic model of what it has built
in the current run.

The self-model is the execution network's durable, globally accurate record of
factory infrastructure. It is not subject to scan-radius limitations — nodes are
added and updated by the coordinator and examination layer as structures are built
and verified, not by reading entity scan data directly.

Lifecycle
---------
- Starts EMPTY at the beginning of each run. WorldState observation does not
  automatically populate it.
- Agents write CANDIDATE nodes to the blackboard when they complete a
  construction subtask.
- The examination layer verifies candidates against WorldState and either
  promotes them to ACTIVE or discards them.
- The self-model persists until run end. What crosses the run boundary into
  behavioral memory (spatial patterns, subgraph summaries) is defined in
  Phase 10.

Node and edge semantics
-----------------------
Nodes represent logical factory units — not individual entities. A PRODUCTION_LINE
node might span a row of assemblers, inserters, and belt segments. A BELT_CORRIDOR
node represents a multi-tile belt run connecting two areas.

Edges represent relationships between nodes. They are directed (from_id → to_id)
but queries may traverse in either direction.

Rules
-----
- Pure data + graph operations. No LLM calls. No RCON. No WorldQuery reads.
- The concrete SelfModel implementation is injected; callers depend only on
  SelfModelProtocol.
- Thread safety is not a requirement at this stage.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from world.state import Position


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

NodeId = str   # UUID string


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(Enum):
    """Logical category of a self-model node."""
    PRODUCTION_LINE  = auto()   # assemblers/furnaces/refineries producing an item
    RESOURCE_SITE    = auto()   # miners on a resource patch
    BELT_CORRIDOR    = auto()   # belt run connecting two points / areas
    POWER_GRID       = auto()   # power generation + distribution network
    DEFENDED_REGION  = auto()   # region with active turret coverage (Phase 9+)
    TRAIN_STATION    = auto()   # train stop + associated infrastructure
    STORAGE          = auto()   # dedicated buffer/storage area


class NodeStatus(Enum):
    """Verification state of a self-model node."""
    CANDIDATE = auto()   # written by agents; not yet examination-verified
    ACTIVE    = auto()   # examination-confirmed and currently operating
    DEGRADED  = auto()   # examination found issues (low throughput, damage)
    INACTIVE  = auto()   # confirmed to exist but currently not operating


class EdgeType(Enum):
    """Relationship type between two self-model nodes."""
    FEEDS_INTO        = auto()   # node A provides output consumed by node B
    DEPENDS_ON        = auto()   # node A cannot operate without node B
    CONNECTED_BY_BELT = auto()   # items flow between nodes via belt
    CONNECTED_BY_RAIL = auto()   # items flow between nodes via train
    DEFENDS           = auto()   # node A provides defence coverage for node B
    SPATIALLY_ADJACENT = auto()  # nodes are geographically neighbouring


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """
    Axis-aligned bounding box in game-tile coordinates.

    Used to describe the spatial extent of a self-model node. Coordinates
    use the same system as Position (x increases east, y increases south
    in Factorio's convention).
    """
    top_left: Position
    bottom_right: Position

    def contains(self, position: Position) -> bool:
        """True if the given position falls within (or on) this box."""
        return (
            self.top_left.x <= position.x <= self.bottom_right.x
            and self.top_left.y <= position.y <= self.bottom_right.y
        )

    def overlaps(self, other: "BoundingBox") -> bool:
        """True if this box shares any area with another box."""
        return (
            self.top_left.x <= other.bottom_right.x
            and self.bottom_right.x >= other.top_left.x
            and self.top_left.y <= other.bottom_right.y
            and self.bottom_right.y >= other.top_left.y
        )

    @property
    def width(self) -> float:
        return self.bottom_right.x - self.top_left.x

    @property
    def height(self) -> float:
        return self.bottom_right.y - self.top_left.y

    def __repr__(self) -> str:
        return (
            f"BoundingBox({self.top_left} → {self.bottom_right}, "
            f"{self.width:.0f}×{self.height:.0f})"
        )


# ---------------------------------------------------------------------------
# Node and Edge dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SelfModelNode:
    """
    A logical factory unit in the self-model graph.

    Fields
    ------
    id              : NodeId (UUID string), auto-generated.
    type            : Logical category of this unit.
    status          : Verification state (CANDIDATE → ACTIVE/DEGRADED/INACTIVE).
    bounding_box    : Spatial extent in game tiles.
    label           : Human-readable name (e.g. "iron-plate smelter A").
    throughput      : Dict mapping item name → units per minute.
                      For production lines: items produced per minute.
                      For resource sites: ore extracted per minute.
                      Empty for belt corridors, power grids, etc.
    created_at      : Game tick at which this node was first written.
    last_verified_at: Game tick of the most recent examination verification.
                      0 if never verified (CANDIDATE nodes).
    """
    type: NodeType
    status: NodeStatus
    bounding_box: BoundingBox
    label: str
    throughput: dict[str, float]
    created_at: int
    last_verified_at: int = 0
    id: NodeId = field(default_factory=lambda: str(uuid.uuid4()))

    def __repr__(self) -> str:
        return (
            f"SelfModelNode({self.id[:8]}… type={self.type.name} "
            f"status={self.status.name} label={self.label!r})"
        )


@dataclass
class SelfModelEdge:
    """A directed relationship between two self-model nodes."""
    from_id: NodeId
    to_id: NodeId
    edge_type: EdgeType

    def __repr__(self) -> str:
        return (
            f"SelfModelEdge({self.from_id[:8]}… "
            f"→[{self.edge_type.name}]→ {self.to_id[:8]}…)"
        )


# ---------------------------------------------------------------------------
# SelfModelProtocol
# ---------------------------------------------------------------------------

class SelfModelProtocol:
    """
    Protocol for the factory self-model graph.

    Callers depend only on this interface. The concrete implementation is
    injected at startup. Swapping the backing store (e.g. from in-memory
    adjacency to a graph database) requires substituting the injected object,
    not restructuring callers.

    All methods that query nodes respect the current status of each node.
    Candidate nodes are included in query results unless explicitly filtered
    out — the examination layer needs to see them.
    """

    def add_node(self, node: SelfModelNode) -> NodeId:
        raise NotImplementedError

    def add_edge(
        self,
        from_id: NodeId,
        to_id: NodeId,
        edge_type: EdgeType,
    ) -> None:
        raise NotImplementedError

    def get_node(self, node_id: NodeId) -> Optional[SelfModelNode]:
        raise NotImplementedError

    def query_nodes(
        self,
        type: Optional[NodeType] = None,
        status: Optional[NodeStatus] = None,
    ) -> list[SelfModelNode]:
        raise NotImplementedError

    def query_path(
        self,
        from_id: NodeId,
        to_id: NodeId,
    ) -> Optional[list[NodeId]]:
        raise NotImplementedError

    def find_producers(self, item: str) -> list[SelfModelNode]:
        raise NotImplementedError

    def find_capacity(self, item: str) -> float:
        raise NotImplementedError

    def subgraph(self, node_ids: list[NodeId]) -> "SelfModel":
        raise NotImplementedError

    def promote_candidate(self, node_id: NodeId) -> None:
        raise NotImplementedError

    def discard_candidate(self, node_id: NodeId) -> None:
        raise NotImplementedError

    def overlapping_nodes(self, bbox: BoundingBox) -> list[SelfModelNode]:
        raise NotImplementedError

    def all_nodes(self) -> list[SelfModelNode]:
        raise NotImplementedError

    def all_edges(self) -> list[SelfModelEdge]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------

class SelfModel(SelfModelProtocol):
    """
    In-memory self-model graph backed by adjacency dicts.

    Nodes are stored in a dict keyed by NodeId. Edges are stored as a list
    and duplicated into forward and reverse adjacency dicts for efficient
    path queries.

    This is the production implementation for Phases 5-9. The examination
    layer (Phase 10) may extend it with richer indexing, but the interface
    remains the same.
    """

    def __init__(self) -> None:
        self._nodes: dict[NodeId, SelfModelNode] = {}
        self._edges: list[SelfModelEdge] = []
        # Forward adjacency: from_id -> list of (to_id, EdgeType)
        self._adj: dict[NodeId, list[tuple[NodeId, EdgeType]]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: SelfModelNode) -> NodeId:
        """
        Add a node to the graph and return its id.

        If a node with the same id already exists it is replaced. Edges
        referencing the old node are preserved.
        """
        self._nodes[node.id] = node
        if node.id not in self._adj:
            self._adj[node.id] = []
        return node.id

    def add_edge(
        self,
        from_id: NodeId,
        to_id: NodeId,
        edge_type: EdgeType,
    ) -> None:
        """
        Add a directed edge from_id → to_id.

        Both nodes must already exist. Duplicate edges (same from/to/type)
        are silently ignored.

        Raises ValueError if either node is not found.
        """
        if from_id not in self._nodes:
            raise ValueError(f"Source node {from_id!r} not found in self-model")
        if to_id not in self._nodes:
            raise ValueError(f"Target node {to_id!r} not found in self-model")
        # Dedup check
        for existing_to, existing_type in self._adj.get(from_id, []):
            if existing_to == to_id and existing_type == edge_type:
                return
        edge = SelfModelEdge(from_id=from_id, to_id=to_id, edge_type=edge_type)
        self._edges.append(edge)
        self._adj.setdefault(from_id, []).append((to_id, edge_type))

    def promote_candidate(self, node_id: NodeId) -> None:
        """
        Promote a CANDIDATE node to ACTIVE.

        Called by the examination layer after verifying the node against
        WorldState.

        Raises ValueError if the node is not found or not CANDIDATE.
        """
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} not found")
        if node.status != NodeStatus.CANDIDATE:
            raise ValueError(
                f"Node {node_id!r} has status {node.status.name}, expected CANDIDATE"
            )
        node.status = NodeStatus.ACTIVE

    def discard_candidate(self, node_id: NodeId) -> None:
        """
        Remove a CANDIDATE node that failed examination verification.

        Also removes all edges that reference this node.

        Raises ValueError if the node is not found or not CANDIDATE.
        """
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} not found")
        if node.status != NodeStatus.CANDIDATE:
            raise ValueError(
                f"Node {node_id!r} has status {node.status.name}, expected CANDIDATE"
            )
        del self._nodes[node_id]
        del self._adj[node_id]
        # Remove from adjacency of other nodes and from edge list
        for nid in self._adj:
            self._adj[nid] = [
                (to, et) for (to, et) in self._adj[nid] if to != node_id
            ]
        self._edges = [
            e for e in self._edges
            if e.from_id != node_id and e.to_id != node_id
        ]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_node(self, node_id: NodeId) -> Optional[SelfModelNode]:
        """Return the node with the given id, or None."""
        return self._nodes.get(node_id)

    def query_nodes(
        self,
        type: Optional[NodeType] = None,
        status: Optional[NodeStatus] = None,
    ) -> list[SelfModelNode]:
        """
        Return nodes matching all supplied filters.

        None for a filter means 'match all'. Both type and status can be
        filtered independently or together.
        """
        result = []
        for node in self._nodes.values():
            if type is not None and node.type != type:
                continue
            if status is not None and node.status != status:
                continue
            result.append(node)
        return result

    def query_path(
        self,
        from_id: NodeId,
        to_id: NodeId,
    ) -> Optional[list[NodeId]]:
        """
        Return the shortest path from from_id to to_id (BFS, directed edges).

        Returns None if no path exists or either node is not in the graph.
        The returned list includes both from_id and to_id.
        """
        if from_id not in self._nodes or to_id not in self._nodes:
            return None
        if from_id == to_id:
            return [from_id]

        visited: set[NodeId] = {from_id}
        queue: deque[list[NodeId]] = deque([[from_id]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            for neighbour, _ in self._adj.get(current, []):
                if neighbour == to_id:
                    return path + [to_id]
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(path + [neighbour])
        return None

    def find_producers(self, item: str) -> list[SelfModelNode]:
        """
        Return all nodes whose throughput dict contains the given item
        with a positive rate.

        Used by the coordinator to check whether a prerequisite item is
        already being produced before deriving a new production subtask.
        """
        return [
            node for node in self._nodes.values()
            if node.throughput.get(item, 0.0) > 0.0
        ]

    def find_capacity(self, item: str) -> float:
        """
        Return the total throughput (units per minute) for the given item
        across all ACTIVE nodes.

        CANDIDATE, DEGRADED, and INACTIVE nodes are excluded.
        """
        return sum(
            node.throughput.get(item, 0.0)
            for node in self._nodes.values()
            if node.status == NodeStatus.ACTIVE
        )

    def overlapping_nodes(self, bbox: BoundingBox) -> list[SelfModelNode]:
        """
        Return all nodes whose bounding box overlaps the given bbox.

        Used by the spatial-logistics agent to detect conflicts before
        placing new infrastructure, and by the examination layer to flag
        unexpected overlaps during reconciliation.

        Enforcement of non-overlap invariants is the caller's responsibility —
        this method only detects; it does not prevent or reject overlaps.
        """
        return [
            node for node in self._nodes.values()
            if node.bounding_box.overlaps(bbox)
        ]

    def all_nodes(self) -> list[SelfModelNode]:
        """Return all nodes in insertion order."""
        return list(self._nodes.values())

    def all_edges(self) -> list[SelfModelEdge]:
        """Return all edges in insertion order."""
        return list(self._edges)

    def subgraph(self, node_ids: list[NodeId]) -> "SelfModel":
        """
        Return a new SelfModel containing only the specified nodes and
        the edges between them.

        Nodes not found in this graph are silently skipped.
        """
        id_set = set(node_ids)
        sub = SelfModel()
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node is not None:
                sub.add_node(node)
        for edge in self._edges:
            if edge.from_id in id_set and edge.to_id in id_set:
                sub.add_edge(edge.from_id, edge.to_id, edge.edge_type)
        return sub

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return (
            f"SelfModel({len(self._nodes)} nodes, {len(self._edges)} edges)"
        )