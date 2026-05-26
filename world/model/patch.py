"""
world/model/patch.py

SelfModelPatch — a structured update to the factory self-model.

Agents never read from or write to the SelfModel directly. Instead, when an
agent completes work that should be reflected in the self-model (e.g. placing
a production line, verifying a belt corridor), it emits one or more
SelfModelPatch objects as part of its tick() return value. The coordinator
collects these and applies them to the SelfModel.

This keeps the coordinator/agent information boundary clean:
  - Agents only know WorldState (via WorldQuery) and their active Task.
  - The coordinator owns the SelfModel; agents cannot query it.
  - The patch type is the sole channel through which agents affect the model.

Patch actions
-------------
  add_node    : Register a new CANDIDATE node. Requires `node`.
  add_edge    : Add a directed edge between two existing nodes.
                Requires `from_id`, `to_id`, `edge_type`.
  promote     : Promote a CANDIDATE node to ACTIVE (examination layer only).
                Requires `node_id`.
  discard     : Remove a CANDIDATE node that failed verification.
                Requires `node_id`.

Rules
-----
- Pure data. No LLM calls. No RCON. No WorldQuery reads.
- Invalid patches (wrong fields for the action) are detected and logged by
  the coordinator when applying; they do not raise at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from world.model.self_model import BoundingBox, EdgeType, NodeId, SelfModelNode


PatchAction = Literal["add_node", "add_edge", "promote", "discard"]


@dataclass
class SelfModelPatch:
    """
    A single proposed mutation to the factory self-model.

    Only the fields relevant to `action` need to be set. The coordinator
    validates the combination before applying.

    Fields
    ------
    action      : What mutation to perform.
    node        : Required for "add_node". The node to register as CANDIDATE.
    from_id     : Required for "add_edge". Source node id.
    to_id       : Required for "add_edge". Target node id.
    edge_type   : Required for "add_edge". Relationship type.
    node_id     : Required for "promote" and "discard". Target node id.
    source_agent: Identifier of the agent that produced this patch.
                  Used for logging and debugging.
    """
    action: PatchAction
    node: Optional[SelfModelNode] = None
    from_id: Optional[NodeId] = None
    to_id: Optional[NodeId] = None
    edge_type: Optional[EdgeType] = None
    node_id: Optional[NodeId] = None
    source_agent: str = ""

    def __repr__(self) -> str:
        if self.action == "add_node" and self.node is not None:
            return f"SelfModelPatch(add_node label={self.node.label!r})"
        if self.action == "add_edge":
            return (
                f"SelfModelPatch(add_edge "
                f"{str(self.from_id)[:8]}…→{str(self.to_id)[:8]}… "
                f"type={self.edge_type})"
            )
        if self.action in ("promote", "discard"):
            return f"SelfModelPatch({self.action} node_id={str(self.node_id)[:8]}…)"
        return f"SelfModelPatch(action={self.action!r})"