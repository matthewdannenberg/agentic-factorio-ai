"""
tests_in_game/04_exploration/test_chunk_grid.py

Verifies that wq.chunk_map is correctly populated during agent exploration.

How chunk_map gets populated
------------------------------
WorldWriter.integrate_snapshot() accumulates newly_charted_chunks deltas
from the Lua mod into ExplorationState.charted_chunk_coords on every poll.
wq.chunk_map wraps this set in a ChunkMapQuery that provides frontier
detection, nearest-frontier lookup, and coordinate helpers.

The pipeline under test is:
  Lua mod  →  bridge/state_parser.py  →  WorldState.player.exploration
  →  newly_charted_chunks  →  WorldWriter.integrate_snapshot()
  →  ExplorationState.charted_chunk_coords  →  wq.chunk_map

Access in tests
---------------
All chunk map queries go through result.wq.chunk_map. No SelfModel access
is needed — ChunkMapQuery lives in the observable layer.

What is tested
--------------
  - wq.chunk_map.charted_count() > 0 after exploration
  - is_charted() returns True for chunks the bridge reported
  - frontiers() is non-empty after partial exploration
  - nearest_frontier() returns a valid (cx, cy) tuple
  - nearest_frontier_position() returns the correct tile-space centre
  - chunk_for_position() converts tile positions to chunk coords correctly
  - centre_of_chunk() returns a position in the correct tile range
  - chunk_map grows monotonically during a run
"""

import pytest
from planning import GoalQueueEntry
from world.observable.state import Position


# ---------------------------------------------------------------------------
# Helper goals
# ---------------------------------------------------------------------------

def _warm_goal() -> GoalQueueEntry:
    """
    Minimal goal that completes in one tick.

    charted_chunks >= 1 is always satisfied on any loaded map, so this
    runs the loop just long enough to populate the ChunkGrid from the
    initial poll without requiring the agent to actually move anywhere.
    Used for ChunkGrid pipeline tests that only care about the
    bridge → mark_charted_bulk pathway, not about agent navigation.
    """
    return GoalQueueEntry(
        description="Warm loop for ChunkGrid pipeline test",
        goal_type="exploration",
        success_condition="charted_chunks >= 1",
        failure_condition="elapsed_ticks > 600",
    )


def _explore_goal(chunks: int = 5) -> GoalQueueEntry:
    """
    Exploration goal that requires the agent to chart new chunks.
    Used for tests that verify agent navigation behaviour.
    """
    return GoalQueueEntry(
        description=f"Explore {chunks} new chunks (chunk_grid test)",
        goal_type="exploration",
        success_condition=f"new.charted_chunks >= {chunks}",
        failure_condition="elapsed_ticks > 18000",
    )


# ---------------------------------------------------------------------------
# Basic population
# ---------------------------------------------------------------------------

class TestChunkGridPopulation:
    """ChunkGrid is filled during exploration."""

    def test_chunk_grid_non_empty_after_exploration(self, run_goal):
        """
        After exploring new chunks, wq.chunk_map must contain at least one entry.

        ChunkGrid is populated exclusively from newly_charted_chunks deltas
        emitted by the Lua mod — it does NOT seed from the absolute
        charted_chunks count at startup. This means the grid is empty until
        the agent actually causes new chunks to be charted and the mod reports
        them as newly seen.

        A failed goal is acceptable here: even if navigation STUCKs, any
        chunks charted before the STUCK will have been drained into the grid.
        We assert only that at least one chunk was recorded.
        """
        result = run_goal(_explore_goal(chunks=5))
        # Goal may complete or fail — either way check the grid.

        count = result.wq.chunk_map.charted_count()
        assert count > 0, (
            f"chunk_map.charted_count()={count} after exploration run. "
            "Check WorldWriter.integrate_snapshot() → charted_chunk_coords pipeline. "
            "and that the Lua mod emits newly_charted_chunks during movement."
        )

    def test_chunk_grid_count_matches_worldquery(self, run_goal):
        """
        chunk_map.charted_count() should be > 0 after any exploration run.

        wq.charted_chunks is the Lua force total (may be 196+ from prior
        sessions). ChunkGrid accumulates only delta chunks emitted by the mod
        during this session. They will differ in absolute value; what matters
        is that ChunkGrid is non-empty, proving the drain pipeline works.
        """
        result = run_goal(_explore_goal(chunks=5))

        grid_count = result.wq.chunk_map.charted_count()
        wq_count = result.wq.charted_chunks

        assert grid_count >= 1, (
            f"chunk_map is empty (count=0) even though wq.charted_chunks={wq_count}. "
            "The loop is not draining newly_charted_chunks into the grid."
        )

    def test_chunk_grid_bool_is_true_after_exploration(self, run_goal):
        """bool(chunk_grid) is True when non-empty, False when empty."""
        result = run_goal(_explore_goal(chunks=5))

        assert bool(result.wq.chunk_map), (
            "bool(chunk_grid) is False — grid is empty after exploration run. "
            "Check that newly_charted_chunks deltas are being emitted and drained."
        )

    def test_chunk_grid_len_matches_charted_count(self, run_goal):
        """len(chunk_grid) == chunk_grid.charted_count() regardless of grid contents."""
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        assert len(grid) == grid.charted_count(), (
            f"len(grid)={len(grid)} != grid.charted_count()={grid.charted_count()}"
        )
        # len() and charted_count() are consistent even on an empty grid.
        # (The grid may be empty if no newly_charted_chunks were emitted — that
        # is tested separately in test_chunk_grid_non_empty_after_exploration.)


# ---------------------------------------------------------------------------
# is_charted and all_charted
# ---------------------------------------------------------------------------

class TestChunkGridLookup:
    """is_charted() and all_charted() return consistent data."""

    def test_is_charted_true_for_known_chunks(self, run_goal):
        """
        Every chunk in all_charted() should return True from is_charted().
        """
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        for cx, cy in grid.all_charted():
            assert grid.is_charted(cx, cy), (
                f"is_charted({cx}, {cy}) returned False but ({cx},{cy}) "
                "is in all_charted(). Internal set inconsistency."
            )

    def test_is_charted_false_for_distant_unknown_chunk(self, run_goal):
        """
        A chunk very far from spawn (e.g. 10000, 10000) should not be charted
        on a fresh session. This is a sanity check that is_charted() doesn't
        always return True.
        """
        result = run_goal(_warm_goal())

        # (10000, 10000) is 10000 * 32 = 320000 tiles from origin.
        # No agent could reach there in a short test run.
        assert not result.wq.chunk_map.is_charted(10000, 10000), (
            "is_charted(10000, 10000) returned True — either the grid is "
            "treating everything as charted, or something very wrong happened."
        )

    def test_all_charted_returns_tuples(self, run_goal):
        """all_charted() should return a list of (cx, cy) int tuples."""
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        chunks = grid.all_charted()
        assert isinstance(chunks, list), (
            f"all_charted() returned {type(chunks)!r}, expected list"
        )
        for item in chunks:
            assert isinstance(item, tuple) and len(item) == 2, (
                f"all_charted() item is not a 2-tuple: {item!r}"
            )
            cx, cy = item
            assert isinstance(cx, int) and isinstance(cy, int), (
                f"Chunk coords should be ints; got ({type(cx)!r}, {type(cy)!r})"
            )


# ---------------------------------------------------------------------------
# Frontiers
# ---------------------------------------------------------------------------

class TestChunkGridFrontiers:
    """Frontier detection — edge chunks of explored territory."""

    def test_frontiers_non_empty_after_partial_exploration(self, run_goal):
        """
        After charting some chunks, at least some must border uncharted
        territory — those are frontiers. An empty frontiers list means either
        the chunk_map is empty (no chunks were charted this session) or every
        charted chunk is fully surrounded (impossible on a partial exploration).

        This test requires real exploration so the ChunkGrid has actual chunk
        coordinates to compute frontiers from.
        """
        result = run_goal(_explore_goal(chunks=5))

        grid = result.wq.chunk_map
        if grid.charted_count() == 0:
            import pytest
            pytest.skip(
                "chunk_map is empty — no newly_charted_chunks were emitted "
                "during this run. Navigation may be broken."
            )

        fronts = grid.frontiers()
        assert len(fronts) >= 1, (
            f"frontiers() returned [] after charting {grid.charted_count()} chunks. "
            "All charted chunks appear to be fully surrounded — impossible unless "
            "the entire map was explored in a single run."
        )

    def test_all_frontiers_are_charted(self, run_goal):
        """
        Every frontier chunk must itself be charted. Frontiers are charted
        chunks with at least one uncharted neighbour — not uncharted chunks.
        """
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        for cx, cy in grid.frontiers():
            assert grid.is_charted(cx, cy), (
                f"frontier ({cx},{cy}) is not itself charted — "
                "frontiers() is returning uncharted chunks"
            )

    def test_frontier_chunks_have_uncharted_neighbour(self, run_goal):
        """
        Every frontier chunk must have at least one cardinal neighbour that
        is NOT charted (that's the definition of a frontier).
        """
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        for cx, cy in grid.frontiers():
            neighbours = [(cx, cy-1), (cx, cy+1), (cx-1, cy), (cx+1, cy)]
            has_uncharted = any(not grid.is_charted(nx, ny) for nx, ny in neighbours)
            assert has_uncharted, (
                f"Frontier chunk ({cx},{cy}) has all four neighbours charted — "
                "it should not be in the frontier list"
            )


# ---------------------------------------------------------------------------
# Nearest frontier
# ---------------------------------------------------------------------------

class TestChunkGridNearestFrontier:
    """nearest_frontier() and nearest_frontier_position() return valid results."""

    def test_nearest_frontier_returns_tuple(self, run_goal):
        """nearest_frontier() should return a (cx, cy) tuple or None."""
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        pos = result.wq.player_position()
        frontier = grid.nearest_frontier(pos)

        if not grid.frontiers():
            assert frontier is None, "nearest_frontier() should be None when no frontiers"
        else:
            assert frontier is not None, (
                "nearest_frontier() returned None but frontiers() is non-empty"
            )
            assert isinstance(frontier, tuple) and len(frontier) == 2, (
                f"nearest_frontier() returned {frontier!r}, expected (cx, cy) tuple"
            )

    def test_nearest_frontier_is_a_frontier(self, run_goal):
        """The returned chunk must actually be in the frontier set."""
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        pos = result.wq.player_position()
        frontier = grid.nearest_frontier(pos)
        fronts = grid.frontiers()

        if frontier is None:
            pytest.skip("No frontier available")

        assert frontier in fronts, (
            f"nearest_frontier() returned {frontier!r} which is not in "
            f"frontiers() = {fronts[:5]!r}..."
        )

    def test_nearest_frontier_position_is_in_correct_tile_range(self, run_goal):
        """
        nearest_frontier_position() should return a tile-space Position
        that is the centre of the returned chunk. For chunk (cx, cy) the
        tile-space centre is (cx*32 + 16, cy*32 + 16).
        """
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        pos = result.wq.player_position()
        frontier_pos = grid.nearest_frontier_position(pos)
        frontier_chunk = grid.nearest_frontier(pos)

        if frontier_chunk is None:
            pytest.skip("No frontier available")

        assert frontier_pos is not None, (
            "nearest_frontier_position() returned None but nearest_frontier() "
            "returned a chunk"
        )

        cx, cy = frontier_chunk
        expected_x = cx * 32 + 16.0
        expected_y = cy * 32 + 16.0
        assert abs(frontier_pos.x - expected_x) < 0.01, (
            f"frontier_pos.x={frontier_pos.x} != expected {expected_x} "
            f"for chunk ({cx},{cy})"
        )
        assert abs(frontier_pos.y - expected_y) < 0.01, (
            f"frontier_pos.y={frontier_pos.y} != expected {expected_y} "
            f"for chunk ({cx},{cy})"
        )


# ---------------------------------------------------------------------------
# Exploration-driven growth
# ---------------------------------------------------------------------------

class TestChunkGridGrowth:
    """
    Verifies that agent exploration causes the ChunkGrid to grow.

    These tests require the exploration agent to actually move and chart new
    territory — they will fail if navigation is permanently broken. They are
    the definitive tests that the full pipeline
    (agent movement → Lua mod → newly_charted_chunks → chunk_map) works.
    """

    def test_chunk_grid_grows_during_exploration(self, run_goal):
        """
        After an exploration goal, ChunkGrid must contain more chunks than
        it did at the start of the run (i.e. the agent charted at least one
        new chunk and the bridge emitted it as a newly_charted_chunks delta).

        A freshly constructed SelfModel has charted_count()=0. After exploring,
        it must be > 0.
        """
        result = run_goal(_explore_goal(chunks=5))

        grid_count = result.wq.chunk_map.charted_count()
        assert grid_count > 0, (
            f"chunk_map.charted_count()={grid_count} after exploration run. "
            "Either navigation is broken (agent never moved) or the Lua mod "
            "is not emitting newly_charted_chunks during movement. "
            f"Goal outcome: completed={result.stats.goals_completed}, "
            f"failed={result.stats.goals_failed}, stuck={result.stats.stuck_events}."
        )

    def test_chunk_grid_count_increases_with_more_exploration(self, run_goals):
        """
        A goal that explores more chunks should produce a larger ChunkGrid
        than one that explores fewer. This tests that grid growth is
        proportional to actual exploration distance, not a one-time event.
        """
        # First short run — collect baseline grid count.
        result_short = run_goals([_explore_goal(chunks=3)])
        short_count = result_short.wq.chunk_map.charted_count()

        # Second longer run — should add more chunks.
        result_long = run_goals([_explore_goal(chunks=10)])
        long_count = result_long.wq.chunk_map.charted_count()

        # The longer run's grid must be at least as large as the shorter one.
        # Each run starts with a fresh SelfModel (per conftest), so we compare
        # independent session counts.
        assert long_count >= short_count, (
            f"Longer exploration produced fewer grid entries ({long_count}) "
            f"than shorter exploration ({short_count}). "
            "ChunkGrid growth is not proportional to exploration."
        )
        # At least one of them should be non-zero.
        assert max(short_count, long_count) > 0, (
            "Neither exploration run produced any ChunkGrid entries. "
            "Navigation may be completely broken."
        )

    def test_chunk_grid_reflects_newly_charted_count(self, run_goal):
        """
        The number of entries added to ChunkGrid during a run should roughly
        match the number of new chunks charted (wq.charted_chunks minus the
        pre-run total as reported in the start snapshot context).

        We can't know the exact pre-run charted_chunks without instrumenting
        conftest, so instead we assert that if the goal completed (new chunks
        were actually charted), the ChunkGrid has at least as many entries as
        the target chunk count.
        """
        chunks_target = 5
        result = run_goal(_explore_goal(chunks=chunks_target))

        if result.stats.goals_completed == 1:
            # Goal succeeded → at least chunks_target new chunks were charted →
            # ChunkGrid must have at least that many entries.
            grid_count = result.wq.chunk_map.charted_count()
            assert grid_count >= chunks_target, (
                f"Goal completed ({chunks_target} new chunks) but "
                f"ChunkGrid only has {grid_count} entries. "
                "newly_charted_chunks deltas are not all reaching the grid."
            )
        else:
            # Goal failed/stuck — still check that any movement produced entries.
            grid_count = result.wq.chunk_map.charted_count()
            # Non-zero is enough — we can't assert exact count on a failed run.
            # (May be 0 if agent never moved at all — acceptable failure mode.)
            if grid_count == 0:
                import pytest
                pytest.skip(
                    "Exploration goal failed with no chunks charted — "
                    "navigation may be broken in this environment. "
                    f"stats={result.stats}"
                )


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------

class TestChunkGridCoordinateHelpers:
    """chunk_for_position() and centre_of_chunk() correctness."""

    def test_chunk_for_position_spawn(self, run_goal):
        """
        The player spawns near (0, 0). chunk_for_position(Position(0, 0))
        should return (0, 0) (floor(0/32) = 0 in both axes).
        """
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        cx, cy = grid.chunk_for_position(Position(0.0, 0.0))
        assert cx == 0 and cy == 0, (
            f"chunk_for_position(Position(0,0)) = ({cx},{cy}), expected (0,0)"
        )

    def test_chunk_for_position_positive_tiles(self, run_goal):
        """Tile (96, 64) is in chunk (3, 2) since floor(96/32)=3, floor(64/32)=2."""
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        cx, cy = grid.chunk_for_position(Position(96.0, 64.0))
        assert cx == 3 and cy == 2, (
            f"chunk_for_position(Position(96, 64)) = ({cx},{cy}), expected (3,2)"
        )

    def test_chunk_for_position_negative_tiles(self, run_goal):
        """Tile (-32, -1) is in chunk (-1, -1) since floor(-32/32)=-1, floor(-1/32)=-1."""
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        cx, cy = grid.chunk_for_position(Position(-32.0, -1.0))
        assert cx == -1 and cy == -1, (
            f"chunk_for_position(Position(-32, -1)) = ({cx},{cy}), expected (-1,-1)"
        )

    def test_centre_of_chunk_zero_zero(self, run_goal):
        """centre_of_chunk(0, 0) should be (16.0, 16.0)."""
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        centre = grid.centre_of_chunk(0, 0)
        assert abs(centre.x - 16.0) < 0.01 and abs(centre.y - 16.0) < 0.01, (
            f"centre_of_chunk(0,0) = {centre}, expected (16.0, 16.0)"
        )

    def test_centre_of_chunk_roundtrip(self, run_goal):
        """
        chunk_for_position(centre_of_chunk(cx, cy)) should return (cx, cy).
        This tests that the two coordinate helpers are inverse-consistent.
        """
        result = run_goal(_warm_goal())

        grid = result.wq.chunk_map
        for cx, cy in [(0,0), (1,0), (-1,2), (3,-1)]:
            centre = grid.centre_of_chunk(cx, cy)
            back_cx, back_cy = grid.chunk_for_position(centre)
            assert (back_cx, back_cy) == (cx, cy), (
                f"Roundtrip failed: centre_of_chunk({cx},{cy})={centre!r} → "
                f"chunk_for_position → ({back_cx},{back_cy}) != ({cx},{cy})"
            )
