"""
world/model/layers/factory_graph.py

FactoryGraph — the coordinator's persistent model of factory infrastructure.

A directed graph where nodes are logical factory components and edges are
item flows between them. Globally accurate and non-proximal — not subject to
scan-radius limitations. Updated by the coordinator applying SelfModelPatches
emitted by agents; verified by the examination layer.

Node semantics
--------------
Each FactoryNode represents a self-contained logical unit of the factory:
a smelting line, a production cell, a mining site, a power plant, a storage
buffer, a belt corridor, etc. A node is NOT a single entity — it spans all
the machines, inserters, belts, and chests that together perform one function.

Nodes go through a verification lifecycle:
  CANDIDATE  Written by agents after completing construction. Unverified.
  ACTIVE     Examination-confirmed: the component exists and is operating.
  DEGRADED   Examination found issues: low throughput, damage, missing inputs.
  INACTIVE   Component exists but is not currently running.

Edge semantics
--------------
Edges are directed and annotated with the item flowing and its rate. Power
is NOT modelled as point-to-point edges — a POWER_GRID node that other nodes
DEPEND_ON captures that relationship more accurately (power is a global
constraint, not a flow between two specific components).

IOPoints
--------
Each node carries a list of IOPoints describing where items physically enter
and leave the component. IOPoints start empty on CANDIDATE nodes and are
populated by the examination layer. The construction agent uses them for
precise belt routing; it falls back to bounding-box estimation when empty.

Design capacity vs observed throughput
---------------------------------------
  design_capacity : What this component *should* produce when fully supplied.
                    Derivable from KB (recipe throughput × machine count).
                    Set at node creation; does not change unless machines are
                    added or removed.
  throughput      : What this component *has been observed* to produce.
                    Updated by the examination layer from ProductionTracker.
                    Used to detect underperformance (throughput < capacity).

A node is considered "available" (its output can serve as input to new
construction) when:
  - status == ACTIVE
  - sum of outgoing edge rates for a given item < design_capacity[item]
  i.e. not all of the production is already spoken for downstream.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal, Optional

from world.model.types import BoundingBox, IOPoint, NodeId


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(Enum):
    """Logical category of a factory component node."""
    PRODUCTION_LINE = auto()   # assemblers/furnaces/refineries making an item
    RESOURCE_SITE   = auto()   # active mining operation on a resource patch
    BELT_CORRIDOR   = auto()   # belt run connecting two areas
    POWER_GRID      = auto()   # power generation + distribution network
    STORAGE         = auto()   # dedicated buffer / storage area
    DEFENDED_REGION = auto()   # area with active turret coverage (Phase 9+)
    TRAIN_STATION   = auto()   # train stop + associated infrastructure


class NodeStatus(Enum):
    """Verification state of a factory component node."""
    CANDIDATE = auto()   # agent-written; not yet examination-verified
    ACTIVE    = auto()   # examination-confirmed, currently operating
    DEGRADED  = auto()   # examination found issues (low throughput, damage)
    INACTIVE  = auto()   # confirmed to exist but currently not running


class EdgeType(Enum):
    """Relationship type between two factory component nodes."""
    ITEM_FLOW          = auto()   # items flow from A to B (belt, inserter, pipe)
    DEPENDS_ON         = auto()   # A cannot operate without B (e.g. needs power)
    CONNECTED_BY_RAIL  = auto()   # items move via train
    DEFENDS            = auto()   # A provides defence coverage for B
    SPATIALLY_ADJACENT = auto()   # nodes are geographically neighbouring


# ---------------------------------------------------------------------------
# ProcessType
# ---------------------------------------------------------------------------

class ProcessType(Enum):
    """
    Broad category of what a node is doing, used for LLM summaries and
    coordinator reasoning. More structured than a free-form string.
    """
    PRODUCTION  = auto()   # crafting items in assemblers / refineries
    SMELTING    = auto()   # ore → plate in furnaces
    MINING      = auto()   # extracting raw resources
    POWER       = auto()   # generating or distributing electricity
    RESEARCH    = auto()   # consuming science packs in labs
    STORAGE     = auto()   # holding items (chests, tanks)
    LOGISTICS   = auto()   # moving items (belt corridors, train routes)
    UNKNOWN     = auto()   # not yet classified


# ---------------------------------------------------------------------------
# FactoryNode
# ---------------------------------------------------------------------------

@dataclass
class FactoryNode:
    """
    A logical factory component in the self-model graph.

    Fields
    ------
    id               : NodeId (UUID string), auto-generated.
    node_type        : Logical category (PRODUCTION_LINE, RESOURCE_SITE, etc.).
    process_type     : What the component is doing (SMELTING, MINING, etc.).
    status           : Verification state (CANDIDATE → ACTIVE / DEGRADED / INACTIVE).
    bounding_box     : Spatial extent in game tiles.
    label            : Human-readable description, e.g. "iron plate smelter A".
    design_capacity  : {item: units/min} at theoretical full throughput.
                       Derived from KB; set at node creation.
    throughput       : {item: units/min} as last measured by examination.
                       0.0 until examination has run.
    io_points        : Connection points where items enter/leave this node.
                       Empty until examination populates them.
    created_at       : Game tick when this node was first registered.
    last_verified_at : Game tick of most recent examination verification.
                       0 for CANDIDATE nodes.
    """
    node_type: NodeType
    process_type: ProcessType
    status: NodeStatus
    bounding_box: BoundingBox
    label: str
    design_capacity: dict[str, float]
    throughput: dict[str, float] = field(default_factory=dict)
    io_points: list[IOPoint] = field(default_factory=list)
    created_at: int = 0
    last_verified_at: int = 0
    id: NodeId = field(default_factory=lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Derived queries
    # ------------------------------------------------------------------

    def available_capacity(self, item: str, committed: float) -> float:
        """
        Return the uncommitted design capacity for item in units/min.

        committed is the sum of outgoing edge rates for this item (how much
        is already spoken for by downstream connections). The coordinator
        uses this to decide whether this node's output can serve as input
        for new construction without additional production being needed.

        Returns 0.0 if this node does not produce item or is not ACTIVE.
        """
        if self.status != NodeStatus.ACTIVE:
            return 0.0
        cap = self.design_capacity.get(item, 0.0)
        return max(0.0, cap - committed)

    def inputs(self) -> list[IOPoint]:
        """Return only the input IOPoints (flow == 'in')."""
        return [p for p in self.io_points if p.flow == "in"]

    def outputs(self) -> list[IOPoint]:
        """Return only the output IOPoints (flow == 'out')."""
        return [p for p in self.io_points if p.flow == "out"]

    def output_for(self, item: str) -> Optional[IOPoint]:
        """Return the first output IOPoint for item, or None."""
        for p in self.io_points:
            if p.flow == "out" and p.item == item:
                return p
        return None

    def input_for(self, item: str) -> Optional[IOPoint]:
        """Return the first input IOPoint for item, or None."""
        for p in self.io_points:
            if p.flow == "in" and p.item == item:
                return p
        return None

    def __repr__(self) -> str:
        return (
            f"FactoryNode({self.id[:8]}… {self.node_type.name} "
            f"status={self.status.name} label={self.label!r})"
        )


# ---------------------------------------------------------------------------
# FactoryEdge
# ---------------------------------------------------------------------------

@dataclass
class FactoryEdge:
    """
    A directed relationship between two factory component nodes.

    For ITEM_FLOW edges, item and rate describe what flows and how fast.
    For structural edges (DEPENDS_ON, DEFENDS, etc.), item and rate are None/0.
    """
    from_id: NodeId
    to_id: NodeId
    edge_type: EdgeType
    item: Optional[str] = None       # item name for ITEM_FLOW edges
    rate: float = 0.0                # items/min for ITEM_FLOW; 0.0 otherwise
    transport: Optional[str] = None  # "belt", "inserter", "pipe", "train", etc.

    def __repr__(self) -> str:
        if self.edge_type == EdgeType.ITEM_FLOW and self.item:
            return (
                f"FactoryEdge({self.from_id[:8]}… →[{self.item} "
                f"{self.rate:.1f}/min]→ {self.to_id[:8]}…)"
            )
        return (
            f"FactoryEdge({self.from_id[:8]}… "
            f"→[{self.edge_type.name}]→ {self.to_id[:8]}…)"
        )


# ---------------------------------------------------------------------------
# FactoryGraph
# ---------------------------------------------------------------------------

class FactoryGraph:
    """
    In-memory directed graph of factory components.

    Backed by adjacency dicts for O(1) node lookup and efficient path queries.
    All mutations go through named methods; direct dict access from outside
    this class is not permitted.

    Queried by the coordinator to make construction decisions:
      - Is item X already being produced? (find_producers)
      - Is there enough uncommitted capacity of X to feed new construction?
        (available_capacity_for)
      - Which nodes haven't been examined recently? (stale_nodes)
      - Does any existing node overlap this proposed area? (overlapping_nodes)
      - What is the path from producer A to consumer B? (path)
    """

    def __init__(self) -> None:
        self._nodes: dict[NodeId, FactoryNode] = {}
        self._edges: list[FactoryEdge] = []
        # Forward adjacency: from_id → list of (to_id, EdgeType)
        self._adj: dict[NodeId, list[tuple[NodeId, EdgeType]]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: FactoryNode) -> NodeId:
        """
        Add a node and return its id. Replaces an existing node with the
        same id (edges referencing it are preserved).
        """
        self._nodes[node.id] = node
        self._adj.setdefault(node.id, [])
        return node.id

    def add_edge(
        self,
        from_id: NodeId,
        to_id: NodeId,
        edge_type: EdgeType,
        item: Optional[str] = None,
        rate: float = 0.0,
        transport: Optional[str] = None,
    ) -> None:
        """
        Add a directed edge from_id → to_id.

        Both nodes must exist. Duplicate edges (same from/to/type/item) are
        silently ignored. Raises ValueError if either node is missing.
        """
        if from_id not in self._nodes:
            raise ValueError(f"Source node {from_id!r} not found")
        if to_id not in self._nodes:
            raise ValueError(f"Target node {to_id!r} not found")
        for existing_to, existing_type in self._adj.get(from_id, []):
            if existing_to == to_id and existing_type == edge_type:
                return
        edge = FactoryEdge(
            from_id=from_id, to_id=to_id, edge_type=edge_type,
            item=item, rate=rate, transport=transport,
        )
        self._edges.append(edge)
        self._adj.setdefault(from_id, []).append((to_id, edge_type))

    def promote_candidate(self, node_id: NodeId) -> None:
        """
        Promote CANDIDATE → ACTIVE. Called by the examination layer.
        Raises ValueError if node not found or not CANDIDATE.
        """
        node = self._get_or_raise(node_id)
        if node.status != NodeStatus.CANDIDATE:
            raise ValueError(
                f"Node {node_id!r} has status {node.status.name}, expected CANDIDATE"
            )
        node.status = NodeStatus.ACTIVE

    def update_status(self, node_id: NodeId, status: NodeStatus) -> None:
        """
        Update a node's status directly. Used by the examination layer to
        mark nodes DEGRADED or INACTIVE after verification.
        Raises ValueError if node not found.
        """
        self._get_or_raise(node_id).status = status

    def update_throughput(
        self, node_id: NodeId, throughput: dict[str, float], verified_at: int
    ) -> None:
        """
        Record observed throughput and update last_verified_at.
        Called by the examination layer after measurement.
        Raises ValueError if node not found.
        """
        node = self._get_or_raise(node_id)
        node.throughput = dict(throughput)
        node.last_verified_at = verified_at

    def update_io_points(self, node_id: NodeId, io_points: list[IOPoint]) -> None:
        """
        Replace the IOPoints on a node. Called by the examination layer
        after observing the actual belt/inserter arrangement.
        Raises ValueError if node not found.
        """
        self._get_or_raise(node_id).io_points = list(io_points)

    def discard_candidate(self, node_id: NodeId) -> None:
        """
        Remove a CANDIDATE node and all edges referencing it.
        Raises ValueError if node not found or not CANDIDATE.
        """
        node = self._get_or_raise(node_id)
        if node.status != NodeStatus.CANDIDATE:
            raise ValueError(
                f"Node {node_id!r} has status {node.status.name}, expected CANDIDATE"
            )
        del self._nodes[node_id]
        del self._adj[node_id]
        for nid in self._adj:
            self._adj[nid] = [
                (to, et) for (to, et) in self._adj[nid] if to != node_id
            ]
        self._edges = [
            e for e in self._edges
            if e.from_id != node_id and e.to_id != node_id
        ]

    # ------------------------------------------------------------------
    # Query — nodes
    # ------------------------------------------------------------------

    def get_node(self, node_id: NodeId) -> Optional[FactoryNode]:
        """Return the node with the given id, or None."""
        return self._nodes.get(node_id)

    def all_nodes(self) -> list[FactoryNode]:
        return list(self._nodes.values())

    def all_edges(self) -> list[FactoryEdge]:
        return list(self._edges)

    def query_nodes(
        self,
        node_type: Optional[NodeType] = None,
        status: Optional[NodeStatus] = None,
        process_type: Optional[ProcessType] = None,
    ) -> list[FactoryNode]:
        """Return nodes matching all supplied filters (None = match all)."""
        result = []
        for node in self._nodes.values():
            if node_type is not None and node.node_type != node_type:
                continue
            if status is not None and node.status != status:
                continue
            if process_type is not None and node.process_type != process_type:
                continue
            result.append(node)
        return result

    def find_producers(self, item: str) -> list[FactoryNode]:
        """
        Return all nodes with positive design capacity for item.

        Includes CANDIDATE nodes — the coordinator needs to know about
        planned production even before examination confirms it.
        """
        return [
            n for n in self._nodes.values()
            if n.design_capacity.get(item, 0.0) > 0.0
        ]

    def active_capacity_for(self, item: str) -> float:
        """
        Total design capacity (units/min) for item across ACTIVE nodes only.
        """
        return sum(
            n.design_capacity.get(item, 0.0)
            for n in self._nodes.values()
            if n.status == NodeStatus.ACTIVE
        )

    def committed_rate_for(self, node_id: NodeId, item: str) -> float:
        """
        Sum of outgoing ITEM_FLOW edge rates for item leaving node_id.

        Represents how much of this node's production is already spoken for
        by downstream connections. Used with available_capacity() to determine
        whether a node's output can serve new construction.
        """
        return sum(
            e.rate for e in self._edges
            if e.from_id == node_id
            and e.edge_type == EdgeType.ITEM_FLOW
            and e.item == item
        )

    def available_capacity_for(self, node_id: NodeId, item: str) -> float:
        """
        Uncommitted design capacity for item at node_id (units/min).

        Returns 0.0 if the node is not ACTIVE or does not produce item.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return 0.0
        committed = self.committed_rate_for(node_id, item)
        return node.available_capacity(item, committed)

    def overlapping_nodes(self, bbox: BoundingBox) -> list[FactoryNode]:
        """
        Return all nodes whose bounding box overlaps bbox.

        Used by the coordinator before proposing a construction area, and
        by the examination layer to flag unexpected overlaps.
        """
        return [n for n in self._nodes.values() if n.bounding_box.overlaps(bbox)]

    def stale_nodes(self, current_tick: int, threshold_ticks: int) -> list[FactoryNode]:
        """
        Return ACTIVE and DEGRADED nodes that have not been examined within
        threshold_ticks. CANDIDATE and INACTIVE nodes are excluded.

        Used by the examination layer to prioritise which nodes to verify next.
        """
        return [
            n for n in self._nodes.values()
            if n.status in (NodeStatus.ACTIVE, NodeStatus.DEGRADED)
            and (current_tick - n.last_verified_at) >= threshold_ticks
        ]

    # ------------------------------------------------------------------
    # Query — paths
    # ------------------------------------------------------------------

    def path(self, from_id: NodeId, to_id: NodeId) -> Optional[list[NodeId]]:
        """
        Shortest path from from_id to to_id via directed edges (BFS).

        Returns None if no path exists or either node is not in the graph.
        The returned list includes both endpoints.
        """
        if from_id not in self._nodes or to_id not in self._nodes:
            return None
        if from_id == to_id:
            return [from_id]
        visited: set[NodeId] = {from_id}
        queue: deque[list[NodeId]] = deque([[from_id]])
        while queue:
            current_path = queue.popleft()
            current = current_path[-1]
            for neighbour, _ in self._adj.get(current, []):
                if neighbour == to_id:
                    return current_path + [to_id]
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(current_path + [neighbour])
        return None

    def subgraph(self, node_ids: list[NodeId]) -> "FactoryGraph":
        """
        Return a new FactoryGraph containing only the specified nodes and
        the edges between them. Unknown node ids are silently skipped.
        """
        id_set = set(node_ids)
        sub = FactoryGraph()
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node is not None:
                sub.add_node(node)
        for edge in self._edges:
            if edge.from_id in id_set and edge.to_id in id_set:
                sub.add_edge(
                    edge.from_id, edge.to_id, edge.edge_type,
                    item=edge.item, rate=edge.rate, transport=edge.transport,
                )
        return sub

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_raise(self, node_id: NodeId) -> FactoryNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} not found in FactoryGraph")
        return node

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"FactoryGraph({len(self._nodes)} nodes, {len(self._edges)} edges)"
