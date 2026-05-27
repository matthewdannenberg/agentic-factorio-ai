"""
world/model/layers/chunk_grid.py

ChunkGrid -- spatial index of charted map chunks.

Populated incrementally by the coordinator each tick from
WorldQuery.newly_charted_chunks (a delta list of ChunkCoord objects produced
by the Lua mod's delta-tracking logic). The mod emits only newly-charted
chunks each poll, so mark_charted_bulk is O(delta size) -- typically zero or
a handful of chunks per tick during active exploration.

Chunk coordinates
-----------------
A chunk is a 32x32-tile region. Chunk coordinate (cx, cy) maps to tile region:
  x in [cx*32, cx*32 + 31],  y in [cy*32, cy*32 + 31].
Tile (tx, ty) is in chunk (floor(tx/32), floor(ty/32)).

Frontier definition
-------------------
A charted chunk is a "frontier" if at least one of its four cardinal
neighbours (north, south, east, west) has NOT been charted. This is more
robust than "degree < 4" which breaks at map boundaries.

Thread safety
-------------
Not required at this stage -- the coordinator is single-threaded.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from world.observable.state import ChunkCoord

from world.observable.state import Position


class ChunkGrid:
    """
    Spatial index of charted map chunks.

    Populated from bridge data via mark_charted_bulk(). All methods return
    safe empty defaults when the grid is unpopulated.
    """

    CHUNK_SIZE: int = 32   # tiles per chunk side

    def __init__(self) -> None:
        # Set of (cx, cy) tuples for every chunk confirmed charted.
        self._charted: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Population -- called by the coordinator each tick
    # ------------------------------------------------------------------

    def mark_charted(self, cx: int, cy: int) -> None:
        """Mark chunk (cx, cy) as charted. Idempotent."""
        self._charted.add((cx, cy))

    def mark_charted_bulk(self, chunks: list["ChunkCoord"]) -> None:
        """
        Mark a batch of chunks as charted from a delta list of ChunkCoord.

        Called by the coordinator each tick with WorldQuery.newly_charted_chunks.
        Idempotent -- safe to call with the full set on reconnect.
        """
        for c in chunks:
            self._charted.add((c.cx, c.cy))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def is_charted(self, cx: int, cy: int) -> bool:
        """True if chunk (cx, cy) has been charted."""
        return (cx, cy) in self._charted

    def charted_count(self) -> int:
        """Number of charted chunks in this grid."""
        return len(self._charted)

    def frontiers(self) -> list[tuple[int, int]]:
        """
        Return all charted chunks that have at least one uncharted cardinal
        neighbour. These are the edge chunks of explored territory.

        Returns an empty list when the grid is unpopulated.
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
        return Position(
            x=cx * self.CHUNK_SIZE + self.CHUNK_SIZE / 2.0,
            y=cy * self.CHUNK_SIZE + self.CHUNK_SIZE / 2.0,
        )

    def nearest_frontier(self, position: Position) -> Optional[tuple[int, int]]:
        """
        Return the frontier chunk closest (in chunk-space) to position,
        or None if no frontiers exist or the grid is empty.
        """
        fronts = self.frontiers()
        if not fronts:
            return None
        cx0, cy0 = self.chunk_for_position(position)
        return min(fronts, key=lambda c: (c[0] - cx0) ** 2 + (c[1] - cy0) ** 2)

    def nearest_frontier_position(self, position: Position) -> Optional[Position]:
        """
        Return the tile-space centre of the nearest frontier chunk, or None.
        Convenience wrapper around nearest_frontier + centre_of_chunk.
        """
        frontier = self.nearest_frontier(position)
        if frontier is None:
            return None
        return self.centre_of_chunk(*frontier)

    def all_charted(self) -> list[tuple[int, int]]:
        """Return all charted chunk coordinates as (cx, cy) tuples."""
        return list(self._charted)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._charted)

    def __bool__(self) -> bool:
        return bool(self._charted)

    def __repr__(self) -> str:
        frontier_count = len(self.frontiers())
        return f"ChunkGrid({len(self._charted)} chunks, {frontier_count} frontiers)"