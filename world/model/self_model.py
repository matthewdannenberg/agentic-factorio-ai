"""
world/model/self_model.py

SelfModel -- the coordinator's persistent, layered model of the game world.

Contains one layer for now:
  factory  : FactoryGraph -- directed graph of factory components and flows.

The coordinator accesses layers directly (sm.factory.find_producers(...)) for
queries, and applies mutations via SelfModel.apply(patch), which routes each
SelfModelPatch to the correct layer's apply logic.

Lifecycle
---------
- Starts empty at the beginning of each run.
- Agents emit SelfModelPatch objects; the coordinator calls apply().
- The examination layer calls layer methods directly (promote_candidate,
  update_throughput, update_io_points) after verifying WorldState.
- What crosses the run boundary into behavioral memory is defined in Phase 10.

Access rules
------------
- SelfModel is owned and queried exclusively by the coordinator.
- Agents may not query SelfModel; they emit patches instead.
- The examination layer has direct write access (it IS the verification step).
- Nothing in this module imports from execution/, planning/, or llm/.
"""

from __future__ import annotations

import logging

from world.model.layers.factory_graph import (
    EdgeType,
    FactoryEdge,
    FactoryGraph,
    FactoryNode,
    NodeStatus,
    NodeType,
    ProcessType,
)
from world.model.patch import SelfModelPatch
from world.model.types import BoundingBox, IOPoint, NodeId

log = logging.getLogger(__name__)


class SelfModel:
    """
    Layered self-model container.

    Attributes
    ----------
    factory : FactoryGraph
        The factory component graph. Primary coordination data structure.
"""

    def __init__(self) -> None:
        self.factory = FactoryGraph()

    # ------------------------------------------------------------------
    # Patch application
    # ------------------------------------------------------------------

    def apply(self, patch: SelfModelPatch) -> None:
        """
        Apply a SelfModelPatch to the appropriate layer.

        Logs a warning and returns without raising for:
          - Unknown layer names
          - Actions with missing required fields
          - ValueErrors from the layer (e.g. node not found)

        This keeps the coordinator's tick loop safe from malformed patches
        emitted by agents.
        """
        try:
            if patch.layer == "factory":
                self._apply_factory(patch)
            else:
                log.warning(
                    "SelfModel.apply: unknown layer %r (action=%r source=%r)",
                    patch.layer, patch.action, patch.source_agent,
                )
        except (ValueError, TypeError) as exc:
            log.warning(
                "SelfModel.apply: failed to apply %r from %r: %s",
                patch.action, patch.source_agent, exc,
            )

    def _apply_factory(self, patch: SelfModelPatch) -> None:
        action = patch.action

        if action == "add_node":
            if patch.node is None:
                log.warning("add_node patch missing `node` field")
                return
            self.factory.add_node(patch.node)

        elif action == "add_edge":
            if None in (patch.from_id, patch.to_id, patch.edge_type):
                log.warning("add_edge patch missing from_id, to_id, or edge_type")
                return
            self.factory.add_edge(
                patch.from_id, patch.to_id, patch.edge_type,
                item=patch.item, rate=patch.rate, transport=patch.transport,
            )

        elif action == "promote":
            if patch.node_id is None:
                log.warning("promote patch missing `node_id` field")
                return
            self.factory.promote_candidate(patch.node_id)

        elif action == "discard":
            if patch.node_id is None:
                log.warning("discard patch missing `node_id` field")
                return
            self.factory.discard_candidate(patch.node_id)

        elif action == "update_status":
            if patch.node_id is None or patch.new_status is None:
                log.warning("update_status patch missing node_id or new_status")
                return
            self.factory.update_status(patch.node_id, patch.new_status)

        elif action == "update_throughput":
            if patch.node_id is None:
                log.warning("update_throughput patch missing `node_id` field")
                return
            self.factory.update_throughput(
                patch.node_id, patch.throughput, patch.verified_at
            )

        elif action == "update_io_points":
            if patch.node_id is None:
                log.warning("update_io_points patch missing `node_id` field")
                return
            self.factory.update_io_points(patch.node_id, patch.io_points)

        else:
            log.warning("Unknown factory patch action: %r", action)

    # ------------------------------------------------------------------
    # Convenience pass-throughs for the most common coordinator queries
    # ------------------------------------------------------------------

    def find_producers(self, item: str) -> list[FactoryNode]:
        """Shorthand for sm.factory.find_producers(item)."""
        return self.factory.find_producers(item)

    def active_capacity_for(self, item: str) -> float:
        """Shorthand for sm.factory.active_capacity_for(item)."""
        return self.factory.active_capacity_for(item)

    def overlapping_nodes(self, bbox: BoundingBox) -> list[FactoryNode]:
        """Shorthand for sm.factory.overlapping_nodes(bbox)."""
        return self.factory.overlapping_nodes(bbox)

    def stale_nodes(self, current_tick: int, threshold_ticks: int) -> list[FactoryNode]:
        """Shorthand for sm.factory.stale_nodes(current_tick, threshold_ticks)."""
        return self.factory.stale_nodes(current_tick, threshold_ticks)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"SelfModel(factory={self.factory!r})"


# ---------------------------------------------------------------------------
# Re-export key types so callers can do:
#   from world.model.self_model import SelfModel, FactoryNode, NodeType, ...
# without knowing the internal layer structure.
# ---------------------------------------------------------------------------

__all__ = [
    "SelfModel",
    # From factory_graph
    "FactoryNode",
    "FactoryEdge",
    "NodeType",
    "NodeStatus",
    "EdgeType",
    "ProcessType",
    # From types
    "BoundingBox",
    "IOPoint",
    "NodeId",
]