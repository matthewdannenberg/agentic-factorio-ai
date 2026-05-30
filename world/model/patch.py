"""
world/model/patch.py

SelfModelPatch — a structured update to the factory self-model.

Agents never read from or write to the SelfModel directly. Instead, when an
agent completes work that should be reflected in the self-model, it emits one
or more SelfModelPatch objects. The coordinator collects these and applies
them by calling SelfModel.apply(patch), which routes to the appropriate layer.

Information boundary
--------------------
  - Agents see only WorldState (via WorldQuery) and their active Task.
  - The coordinator owns the SelfModel; agents cannot query it.
  - SelfModelPatch is the sole channel through which agents affect the model.

Layer targeting
---------------
Every patch carries a `layer` string that routes it to the right layer:
  "factory"   FactoryGraph — component nodes and item-flow edges.

Patch actions
-------------
  add_node          : Register a new CANDIDATE FactoryNode. Requires `node`.
  add_edge          : Add a directed edge between two existing nodes.
                      Requires `from_id`, `to_id`, `edge_type`.
                      For ITEM_FLOW edges, also set `item`, `rate`, `transport`.
  promote           : Promote a CANDIDATE node to ACTIVE.
                      Requires `node_id`. Examination layer only.
  discard           : Remove a CANDIDATE node that failed verification.
                      Requires `node_id`. Examination layer only.
  update_status     : Change a node's status (e.g. ACTIVE -> DEGRADED).
                      Requires `node_id`, `new_status`.
  update_throughput : Record measured throughput on an ACTIVE node.
                      Requires `node_id`, `throughput`, `verified_at`.
  update_io_points  : Set IOPoints on an examined node.
                      Requires `node_id`, `io_points`.
Rules
-----
- Pure data. No LLM calls. No RCON. No WorldQuery reads.
- Invalid patches (wrong fields for the action) are detected and logged
  by the coordinator when applying; they do not raise at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from world.model.layers.factory_graph import EdgeType, FactoryNode, NodeStatus
from world.model.types import IOPoint, NodeId


PatchAction = Literal[
    "add_node",
    "add_edge",
    "promote",
    "discard",
    "update_status",
    "update_throughput",
    "update_io_points",
]

PatchLayer = Literal["factory"]


@dataclass
class SelfModelPatch:
    """
    A single proposed mutation to one layer of the factory self-model.

    Set only the fields relevant to `action`; leave others as None / default.
    The coordinator validates the combination before applying.

    Fields
    ------
    layer        : Which layer to apply this patch to ("factory" or "chunks").
    action       : What mutation to perform.
    node         : For "add_node" -- the FactoryNode to register as CANDIDATE.
    from_id      : For "add_edge" -- source node id.
    to_id        : For "add_edge" -- target node id.
    edge_type    : For "add_edge" -- relationship type.
    item         : For "add_edge" (ITEM_FLOW) -- item flowing along the edge.
    rate         : For "add_edge" (ITEM_FLOW) -- items/min.
    transport    : For "add_edge" (ITEM_FLOW) -- "belt", "pipe", "train", etc.
    node_id      : For "promote", "discard", "update_*" -- target node id.
    new_status   : For "update_status" -- the NodeStatus to set.
    throughput   : For "update_throughput" -- {item: units/min} measured.
    verified_at  : For "update_throughput" -- tick of measurement.
    io_points    : For "update_io_points" -- replacement IOPoint list.
    source_agent : Identifier of the agent that produced this patch.
    """
    layer: PatchLayer
    action: PatchAction

    # add_node
    node: Optional[FactoryNode] = None

    # add_edge
    from_id: Optional[NodeId] = None
    to_id: Optional[NodeId] = None
    edge_type: Optional[EdgeType] = None
    item: Optional[str] = None
    rate: float = 0.0
    transport: Optional[str] = None

    # promote / discard / update_*
    node_id: Optional[NodeId] = None

    # update_status
    new_status: Optional[NodeStatus] = None

    # update_throughput
    throughput: dict[str, float] = field(default_factory=dict)
    verified_at: int = 0

    # update_io_points
    io_points: list[IOPoint] = field(default_factory=list)

    # provenance
    source_agent: str = ""

    def __repr__(self) -> str:
        if self.action == "add_node" and self.node is not None:
            return (
                f"SelfModelPatch(layer={self.layer!r} add_node "
                f"label={self.node.label!r})"
            )
        if self.action == "add_edge":
            return (
                f"SelfModelPatch(layer={self.layer!r} add_edge "
                f"{str(self.from_id)[:8]}...->{str(self.to_id)[:8]}... "
                f"type={self.edge_type})"
            )
        if self.action in ("promote", "discard", "update_status",
                           "update_throughput", "update_io_points"):
            return (
                f"SelfModelPatch(layer={self.layer!r} {self.action} "
                f"node_id={str(self.node_id)[:8]}...)"
            )
        return f"SelfModelPatch(layer={self.layer!r} action={self.action!r})"