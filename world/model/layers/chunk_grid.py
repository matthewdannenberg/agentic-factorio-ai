"""
world/model/layers/chunk_grid.py

ChunkGrid — spatial index of charted map chunks.

STUB — not yet implemented.

This layer will track which specific chunks have been charted, enabling the
coordinator to identify exploration frontiers (chunks adjacent to uncharted
territory) and reason spatially about where to build vs explore.

Why it's a stub
---------------
The bridge currently reports only a *count* of charted chunks via
ExplorationState.charted_chunks (sourced from LuaForce::get_chart_size /
the 2.x surface.get_chunks() iterator). It does not report *which* chunks
are charted. Populating the grid with real data requires a new bridge
polling surface:

  fa.get_charted_chunks() -> list of {cx, cy} chunk coordinates

This requires a Lua mod change (bridge/mod/control.lua) and a new StateParser
section. Until that is implemented, the ChunkGrid returns safe empty defaults
for all queries so the coordinator can be written against its interface.

Frontier definition
-------------------
A charted chunk is a "frontier" if at least one of its four cardinal
neighbours (north, south, east, west) is not in the charted set. These are
the chunks where exploration should be directed next.

The "degree < 4" shorthand is tempting but incorrect for map-edge chunks,
which will never have 4 neighbours regardless of how much is explored. The
"has an uncharted neighbour" definition is robust to map boundaries.

Phase
-----
Full implementation is paired with the bridge/mod changes that add
fa.get_charted_chunks(). See OPEN_DECISIONS for tracking.
"""

from __future__ import annotations

from world.model.types import Position


class ChunkGrid:
    """
    Spatial index of charted map chunks.

    A chunk is a 32×32-tile region. Chunk coordinates (cx, cy) are the
    chunk's position in chunk-space: tile (x, y) is in chunk
    (floor(x/32), floor(y/32)).

    All methods return safe empty defaults until the grid is populated.
    The coordinator may call any method at any time without risk of error.
    """

    CHUNK_SIZE: int = 32   # tiles per chunk side

    def __init__(self) -> None:
        # Set of (cx, cy) chunk coordinates that have been charted.
        self._charted: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Population (called by the coordinator when bridge data is available)
    # ------------------------------------------------------------------

    def mark_charted(self, cx: int, cy: int) -> None:
        """Mark chunk (cx, cy) as charted. Idempotent."""
        self._charted.add((cx, cy))

    def mark_charted_bulk(self, chunks: list[tuple[int, int]]) -> None:
        """Mark a batch of chunks as charted. More efficient than repeated calls."""
        self._charted.update(chunks)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def is_charted(self, cx: int, cy: int) -> bool:
        """True if chunk (cx, cy) has been charted."""
        return (cx, cy) in self._charted

    def charted_count(self) -> int:
        """Number of charted chunks. Matches ExplorationState.charted_chunks."""
        return len(self._charted)

    def frontiers(self) -> list[tuple[int, int]]:
        """
        Return all charted chunks that have at least one uncharted cardinal
        neighbour. These are the chunks at the edge of explored territory.

        Returns an empty list when the grid is unpopulated (stub state).
        """
        result = []
        for (cx, cy) in self._charted:
            neighbours = [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)]
            if any(n not in self._charted for n in neighbours):
                result.append((cx, cy))
        return result

    def chunk_for_position(self, position: Position) -> tuple[int, int]:
        """Return the chunk coordinates containing the given tile position."""
        return (int(position.x // self.CHUNK_SIZE), int(position.y // self.CHUNK_SIZE))

    def centre_of_chunk(self, cx: int, cy: int) -> Position:
        """Return the tile position of the centre of chunk (cx, cy)."""
        tile_x = cx * self.CHUNK_SIZE + self.CHUNK_SIZE / 2.0
        tile_y = cy * self.CHUNK_SIZE + self.CHUNK_SIZE / 2.0
        return Position(tile_x, tile_y)

    def nearest_frontier(self, position: Position) -> Optional[tuple[int, int]]:
        """
        Return the frontier chunk closest to position, or None if no
        frontiers exist (grid unpopulated or fully surrounded by charted
        territory on all sides).
        """
        fronts = self.frontiers()
        if not fronts:
            return None
        cx0, cy0 = self.chunk_for_position(position)
        return min(fronts, key=lambda c: (c[0] - cx0) ** 2 + (c[1] - cy0) ** 2)

    def all_charted(self) -> list[tuple[int, int]]:
        """Return all charted chunk coordinates."""
        return list(self._charted)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._charted)

    def __bool__(self) -> bool:
        return bool(self._charted)

    def __repr__(self) -> str:
        return f"ChunkGrid({len(self._charted)} chunks, {len(self.frontiers())} frontiers)"


# Deferred import to avoid circular reference at module level
from typing import Optional
