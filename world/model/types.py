"""
world/model/types.py

Shared geometric primitives used across self-model layers.

Kept separate so that chunk_grid.py, factory_graph.py, and patch.py can all
import these without creating circular or cross-layer dependencies.

Types defined here
------------------
NodeId      : str alias for UUID node identifiers.
BoundingBox : Axis-aligned bounding box in game-tile coordinates.
IOPoint     : A single input or output connection point on a factory node.
              Populated by the examination layer; starts empty on CANDIDATE nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from world.observable.state import Direction, Position


# ---------------------------------------------------------------------------
# NodeId
# ---------------------------------------------------------------------------

NodeId = str   # UUID string


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """
    Axis-aligned bounding box in game-tile coordinates.

    Coordinates follow Factorio's convention: x increases east, y increases
    south. top_left is the north-west corner; bottom_right is the south-east
    corner.
    """
    top_left: Position
    bottom_right: Position

    def contains(self, position: Position) -> bool:
        """True if position falls within (or on the boundary of) this box."""
        return (
            self.top_left.x <= position.x <= self.bottom_right.x
            and self.top_left.y <= position.y <= self.bottom_right.y
        )

    def overlaps(self, other: "BoundingBox") -> bool:
        """True if this box shares any area with other (touching edges count)."""
        return (
            self.top_left.x <= other.bottom_right.x
            and self.bottom_right.x >= other.top_left.x
            and self.top_left.y <= other.bottom_right.y
            and self.bottom_right.y >= other.top_left.y
        )

    def expanded(self, margin: float) -> "BoundingBox":
        """Return a new box expanded by margin tiles on every side."""
        return BoundingBox(
            top_left=Position(self.top_left.x - margin, self.top_left.y - margin),
            bottom_right=Position(self.bottom_right.x + margin, self.bottom_right.y + margin),
        )

    @property
    def width(self) -> float:
        return self.bottom_right.x - self.top_left.x

    @property
    def height(self) -> float:
        return self.bottom_right.y - self.top_left.y

    @property
    def centre(self) -> Position:
        return Position(
            (self.top_left.x + self.bottom_right.x) / 2.0,
            (self.top_left.y + self.bottom_right.y) / 2.0,
        )

    def __repr__(self) -> str:
        return (
            f"BoundingBox({self.top_left} → {self.bottom_right}, "
            f"{self.width:.0f}×{self.height:.0f})"
        )


# ---------------------------------------------------------------------------
# IOPoint
# ---------------------------------------------------------------------------

@dataclass
class IOPoint:
    """
    A single input or output connection point on a factory node.

    IOPoints describe where items enter or leave a factory component — the
    specific tile positions where belts connect, and what flows through them.
    They are used by the construction agent to route belts and inserters when
    linking one component to another.

    Lifecycle
    ---------
    IOPoints start empty (io_points=[]) when a node is first registered as
    CANDIDATE. The examination layer populates them after observing the actual
    belt and inserter arrangement around the component. The construction agent
    can still operate without populated IOPoints by estimating connection
    positions from the bounding box, but populated IOPoints enable precise
    routing.

    Fields
    ------
    position  : Tile where the belt connection is made (the tile the
                construction agent should place or connect a belt at).
    direction : Direction items flow *at this point*. For an output point,
                this is the direction items leave (e.g. Direction.EAST means
                items flow eastward out of the component). For an input point,
                this is the direction items arrive from (e.g. Direction.WEST
                means items arrive from the west, i.e. the belt faces east
                into the component).
    flow      : "in" if items enter the component here; "out" if they leave.
    item      : Factorio internal item name, or None if unknown/mixed.
    rate      : Design-capacity throughput in items per minute. 0.0 means
                unknown (set before examination has measured actual rate).
    label     : Optional human-readable description, e.g. "iron ore input".
    """
    position: Position
    direction: Direction
    flow: Literal["in", "out"]
    item: Optional[str] = None
    rate: float = 0.0
    label: str = ""

    def __repr__(self) -> str:
        item_str = self.item or "?"
        return (
            f"IOPoint({self.flow} {item_str} @ {self.position} "
            f"dir={self.direction.name} {self.rate:.1f}/min)"
        )
