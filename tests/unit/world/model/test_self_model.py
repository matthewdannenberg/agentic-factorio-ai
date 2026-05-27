"""
tests/unit/world/test_self_model.py

Tests for world/model/ -- SelfModel, FactoryGraph, ChunkGrid, BoundingBox, IOPoint.

Organised in sections:
  1. BoundingBox geometry
  2. IOPoint
  3. FactoryGraph -- node lifecycle, queries, edges, paths, capacity
  4. ChunkGrid    -- charting, frontiers (stub layer)
  5. SelfModel    -- container, apply() routing, convenience pass-throughs

Run with:  python -m pytest tests/unit/world/test_self_model.py -v
"""

from __future__ import annotations

import unittest

from world import (
    BoundingBox,
    ChunkCoord,
    ChunkGrid,
    EdgeType,
    FactoryEdge,
    FactoryNode,
    IOPoint,
    NodeId,
    NodeStatus,
    NodeType,
    Position,
    ProcessType,
    SelfModel,
    SelfModelPatch,
    Direction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox(x1=0.0, y1=0.0, x2=10.0, y2=10.0) -> BoundingBox:
    return BoundingBox(
        top_left=Position(x1, y1),
        bottom_right=Position(x2, y2),
    )


def _node(
    label: str = "test",
    node_type: NodeType = NodeType.PRODUCTION_LINE,
    process_type: ProcessType = ProcessType.PRODUCTION,
    status: NodeStatus = NodeStatus.CANDIDATE,
    design_capacity: dict | None = None,
    bbox: BoundingBox | None = None,
    created_at: int = 0,
) -> FactoryNode:
    return FactoryNode(
        node_type=node_type,
        process_type=process_type,
        status=status,
        bounding_box=bbox or _bbox(),
        label=label,
        design_capacity=design_capacity or {},
        created_at=created_at,
    )


def _iopoint(
    x=0.0, y=0.0,
    flow="out",
    item="iron-plate",
    rate=30.0,
) -> IOPoint:
    return IOPoint(
        position=Position(x, y),
        direction=Direction.EAST,
        flow=flow,
        item=item,
        rate=rate,
    )


# ===========================================================================
# Section 1 -- BoundingBox
# ===========================================================================

class TestBoundingBox(unittest.TestCase):

    def test_contains_interior(self):
        self.assertTrue(_bbox(0, 0, 10, 10).contains(Position(5, 5)))

    def test_contains_corners(self):
        bb = _bbox(0, 0, 10, 10)
        self.assertTrue(bb.contains(Position(0, 0)))
        self.assertTrue(bb.contains(Position(10, 10)))

    def test_does_not_contain_exterior(self):
        bb = _bbox(0, 0, 10, 10)
        self.assertFalse(bb.contains(Position(11, 5)))
        self.assertFalse(bb.contains(Position(-1, 5)))

    def test_overlaps_sharing_area(self):
        a = _bbox(0, 0, 10, 10)
        b = _bbox(5, 5, 15, 15)
        self.assertTrue(a.overlaps(b))
        self.assertTrue(b.overlaps(a))

    def test_no_overlap_disjoint(self):
        self.assertFalse(_bbox(0, 0, 5, 5).overlaps(_bbox(10, 10, 20, 20)))

    def test_touching_edges_count_as_overlap(self):
        self.assertTrue(_bbox(0, 0, 10, 10).overlaps(_bbox(10, 0, 20, 10)))

    def test_dimensions(self):
        bb = _bbox(2, 3, 12, 8)
        self.assertAlmostEqual(bb.width, 10.0)
        self.assertAlmostEqual(bb.height, 5.0)

    def test_centre(self):
        bb = _bbox(0, 0, 10, 10)
        self.assertAlmostEqual(bb.centre.x, 5.0)
        self.assertAlmostEqual(bb.centre.y, 5.0)

    def test_expanded(self):
        bb = _bbox(2, 2, 8, 8)
        expanded = bb.expanded(1.0)
        self.assertAlmostEqual(expanded.top_left.x, 1.0)
        self.assertAlmostEqual(expanded.top_left.y, 1.0)
        self.assertAlmostEqual(expanded.bottom_right.x, 9.0)
        self.assertAlmostEqual(expanded.bottom_right.y, 9.0)


# ===========================================================================
# Section 2 -- IOPoint
# ===========================================================================

class TestIOPoint(unittest.TestCase):

    def test_fields_stored(self):
        p = _iopoint(x=5.0, y=3.0, flow="out", item="iron-plate", rate=30.0)
        self.assertEqual(p.position, Position(5.0, 3.0))
        self.assertEqual(p.flow, "out")
        self.assertEqual(p.item, "iron-plate")
        self.assertAlmostEqual(p.rate, 30.0)

    def test_defaults(self):
        p = IOPoint(position=Position(0, 0), direction=Direction.NORTH, flow="in")
        self.assertIsNone(p.item)
        self.assertAlmostEqual(p.rate, 0.0)
        self.assertEqual(p.label, "")

    def test_repr_contains_flow_and_item(self):
        r = repr(_iopoint())
        self.assertIn("out", r)
        self.assertIn("iron-plate", r)


# ===========================================================================
# Section 3 -- FactoryGraph
# ===========================================================================

class TestFactoryGraphNodes(unittest.TestCase):

    def setUp(self):
        from world.model.layers.factory_graph import FactoryGraph
        self.g = FactoryGraph()

    def test_add_node_returns_id(self):
        n = _node("furnace A")
        nid = self.g.add_node(n)
        self.assertEqual(nid, n.id)

    def test_get_node(self):
        n = _node("furnace A")
        self.g.add_node(n)
        self.assertEqual(self.g.get_node(n.id).label, "furnace A")

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.g.get_node("ghost"))

    def test_len(self):
        self.assertEqual(len(self.g), 0)
        self.g.add_node(_node("a"))
        self.assertEqual(len(self.g), 1)

    def test_all_nodes(self):
        n1 = _node("a")
        n2 = _node("b")
        self.g.add_node(n1)
        self.g.add_node(n2)
        ids = {n.id for n in self.g.all_nodes()}
        self.assertIn(n1.id, ids)
        self.assertIn(n2.id, ids)


class TestFactoryGraphQuery(unittest.TestCase):

    def setUp(self):
        from world.model.layers.factory_graph import FactoryGraph
        self.g = FactoryGraph()
        self.prod = _node("prod", NodeType.PRODUCTION_LINE, status=NodeStatus.ACTIVE,
                          design_capacity={"iron-plate": 30.0})
        self.res  = _node("res",  NodeType.RESOURCE_SITE,  status=NodeStatus.ACTIVE)
        self.cand = _node("cand", NodeType.PRODUCTION_LINE, status=NodeStatus.CANDIDATE,
                          design_capacity={"iron-plate": 15.0})
        for n in [self.prod, self.res, self.cand]:
            self.g.add_node(n)

    def test_query_all(self):
        self.assertEqual(len(self.g.query_nodes()), 3)

    def test_query_by_type(self):
        self.assertEqual(len(self.g.query_nodes(node_type=NodeType.PRODUCTION_LINE)), 2)

    def test_query_by_status(self):
        self.assertEqual(len(self.g.query_nodes(status=NodeStatus.ACTIVE)), 2)

    def test_query_by_type_and_status(self):
        results = self.g.query_nodes(
            node_type=NodeType.PRODUCTION_LINE, status=NodeStatus.CANDIDATE
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, self.cand.id)

    def test_find_producers_includes_candidate(self):
        # CANDIDATE nodes are included -- coordinator needs to know planned production
        producers = self.g.find_producers("iron-plate")
        ids = {p.id for p in producers}
        self.assertIn(self.prod.id, ids)
        self.assertIn(self.cand.id, ids)

    def test_find_producers_excludes_non_producer(self):
        self.assertNotIn(self.res, self.g.find_producers("iron-plate"))

    def test_active_capacity_for_sums_active_only(self):
        # prod: 30.0 ACTIVE, cand: 15.0 CANDIDATE -- only ACTIVE counts
        self.assertAlmostEqual(self.g.active_capacity_for("iron-plate"), 30.0)

    def test_active_capacity_for_unknown_item(self):
        self.assertAlmostEqual(self.g.active_capacity_for("copper-plate"), 0.0)


class TestFactoryGraphCapacity(unittest.TestCase):

    def setUp(self):
        from world.model.layers.factory_graph import FactoryGraph
        self.g = FactoryGraph()
        self.producer = _node(
            "iron smelter", status=NodeStatus.ACTIVE,
            design_capacity={"iron-plate": 60.0},
        )
        self.consumer = _node("gear maker", status=NodeStatus.ACTIVE)
        self.g.add_node(self.producer)
        self.g.add_node(self.consumer)

    def test_committed_rate_zero_when_no_edges(self):
        self.assertAlmostEqual(
            self.g.committed_rate_for(self.producer.id, "iron-plate"), 0.0
        )

    def test_committed_rate_sums_item_flow_edges(self):
        self.g.add_edge(
            self.producer.id, self.consumer.id,
            EdgeType.ITEM_FLOW, item="iron-plate", rate=30.0,
        )
        self.assertAlmostEqual(
            self.g.committed_rate_for(self.producer.id, "iron-plate"), 30.0
        )

    def test_available_capacity_subtracts_committed(self):
        self.g.add_edge(
            self.producer.id, self.consumer.id,
            EdgeType.ITEM_FLOW, item="iron-plate", rate=20.0,
        )
        # design 60 - committed 20 = 40
        self.assertAlmostEqual(
            self.g.available_capacity_for(self.producer.id, "iron-plate"), 40.0
        )

    def test_available_capacity_zero_for_non_active(self):
        cand = _node("cand", status=NodeStatus.CANDIDATE,
                     design_capacity={"iron-plate": 30.0})
        self.g.add_node(cand)
        self.assertAlmostEqual(
            self.g.available_capacity_for(cand.id, "iron-plate"), 0.0
        )

    def test_node_available_capacity_method(self):
        self.assertAlmostEqual(
            self.producer.available_capacity("iron-plate", committed=10.0), 50.0
        )


class TestFactoryGraphEdges(unittest.TestCase):

    def setUp(self):
        from world.model.layers.factory_graph import FactoryGraph
        self.g = FactoryGraph()
        self.a = _node("A")
        self.b = _node("B")
        self.c = _node("C")
        for n in [self.a, self.b, self.c]:
            self.g.add_node(n)

    def test_add_edge(self):
        self.g.add_edge(self.a.id, self.b.id, EdgeType.ITEM_FLOW,
                        item="iron-plate", rate=30.0)
        self.assertEqual(len(self.g.all_edges()), 1)
        e = self.g.all_edges()[0]
        self.assertEqual(e.item, "iron-plate")
        self.assertAlmostEqual(e.rate, 30.0)

    def test_duplicate_edge_ignored(self):
        self.g.add_edge(self.a.id, self.b.id, EdgeType.DEPENDS_ON)
        self.g.add_edge(self.a.id, self.b.id, EdgeType.DEPENDS_ON)
        self.assertEqual(len(self.g.all_edges()), 1)

    def test_different_edge_type_not_duplicate(self):
        self.g.add_edge(self.a.id, self.b.id, EdgeType.ITEM_FLOW)
        self.g.add_edge(self.a.id, self.b.id, EdgeType.DEPENDS_ON)
        self.assertEqual(len(self.g.all_edges()), 2)

    def test_missing_from_raises(self):
        with self.assertRaises(ValueError):
            self.g.add_edge("ghost", self.b.id, EdgeType.ITEM_FLOW)

    def test_missing_to_raises(self):
        with self.assertRaises(ValueError):
            self.g.add_edge(self.a.id, "ghost", EdgeType.ITEM_FLOW)


class TestFactoryGraphPath(unittest.TestCase):

    def setUp(self):
        from world.model.layers.factory_graph import FactoryGraph
        self.g = FactoryGraph()
        self.a = _node("A")
        self.b = _node("B")
        self.c = _node("C")
        self.d = _node("D")   # isolated
        for n in [self.a, self.b, self.c, self.d]:
            self.g.add_node(n)
        self.g.add_edge(self.a.id, self.b.id, EdgeType.ITEM_FLOW)
        self.g.add_edge(self.b.id, self.c.id, EdgeType.ITEM_FLOW)

    def test_direct_path(self):
        self.assertEqual(
            self.g.path(self.a.id, self.b.id), [self.a.id, self.b.id]
        )

    def test_two_hop_path(self):
        self.assertEqual(
            self.g.path(self.a.id, self.c.id),
            [self.a.id, self.b.id, self.c.id],
        )

    def test_no_path_returns_none(self):
        self.assertIsNone(self.g.path(self.a.id, self.d.id))

    def test_path_to_self(self):
        self.assertEqual(self.g.path(self.a.id, self.a.id), [self.a.id])

    def test_directed_no_backward_path(self):
        self.assertIsNone(self.g.path(self.c.id, self.a.id))

    def test_ghost_node_returns_none(self):
        self.assertIsNone(self.g.path("ghost", self.b.id))


class TestFactoryGraphMutations(unittest.TestCase):

    def setUp(self):
        from world.model.layers.factory_graph import FactoryGraph
        self.g = FactoryGraph()

    def test_promote_candidate_to_active(self):
        n = _node("prod", status=NodeStatus.CANDIDATE)
        self.g.add_node(n)
        self.g.promote_candidate(n.id)
        self.assertEqual(self.g.get_node(n.id).status, NodeStatus.ACTIVE)

    def test_promote_non_candidate_raises(self):
        n = _node("prod", status=NodeStatus.ACTIVE)
        self.g.add_node(n)
        with self.assertRaises(ValueError):
            self.g.promote_candidate(n.id)

    def test_discard_candidate_removes_node_and_edges(self):
        a = _node("A", status=NodeStatus.ACTIVE)
        b = _node("B", status=NodeStatus.CANDIDATE)
        self.g.add_node(a)
        self.g.add_node(b)
        self.g.add_edge(a.id, b.id, EdgeType.ITEM_FLOW)
        self.g.discard_candidate(b.id)
        self.assertIsNone(self.g.get_node(b.id))
        self.assertEqual(len(self.g.all_edges()), 0)

    def test_update_status(self):
        n = _node("prod", status=NodeStatus.ACTIVE)
        self.g.add_node(n)
        self.g.update_status(n.id, NodeStatus.DEGRADED)
        self.assertEqual(self.g.get_node(n.id).status, NodeStatus.DEGRADED)

    def test_update_throughput(self):
        n = _node("prod", status=NodeStatus.ACTIVE)
        self.g.add_node(n)
        self.g.update_throughput(n.id, {"iron-plate": 28.5}, verified_at=3600)
        node = self.g.get_node(n.id)
        self.assertAlmostEqual(node.throughput["iron-plate"], 28.5)
        self.assertEqual(node.last_verified_at, 3600)

    def test_update_io_points(self):
        n = _node("prod", status=NodeStatus.ACTIVE)
        self.g.add_node(n)
        pts = [_iopoint(x=5.0, y=0.0, flow="out")]
        self.g.update_io_points(n.id, pts)
        self.assertEqual(len(self.g.get_node(n.id).io_points), 1)
        self.assertEqual(self.g.get_node(n.id).io_points[0].position, Position(5.0, 0.0))

    def test_update_missing_node_raises(self):
        with self.assertRaises(ValueError):
            self.g.update_status("ghost", NodeStatus.ACTIVE)


class TestFactoryGraphSpatial(unittest.TestCase):

    def setUp(self):
        from world.model.layers.factory_graph import FactoryGraph
        self.g = FactoryGraph()
        self.left   = _node("left",   bbox=BoundingBox(Position(0,  0), Position(10, 10)))
        self.middle = _node("middle", bbox=BoundingBox(Position(20, 0), Position(30, 10)))
        self.right  = _node("right",  bbox=BoundingBox(Position(40, 0), Position(50, 10)))
        for n in [self.left, self.middle, self.right]:
            self.g.add_node(n)

    def test_overlapping_exact(self):
        hits = self.g.overlapping_nodes(BoundingBox(Position(0, 0), Position(10, 10)))
        self.assertEqual(len(hits), 1)
        self.assertIs(hits[0], self.left)

    def test_overlapping_partial(self):
        hits = self.g.overlapping_nodes(BoundingBox(Position(5, 0), Position(25, 10)))
        ids = {n.id for n in hits}
        self.assertIn(self.left.id, ids)
        self.assertIn(self.middle.id, ids)
        self.assertNotIn(self.right.id, ids)

    def test_no_overlap(self):
        self.assertEqual(
            self.g.overlapping_nodes(BoundingBox(Position(11, 0), Position(19, 10))), []
        )

    def test_stale_nodes(self):
        active = _node("a", status=NodeStatus.ACTIVE)
        active.last_verified_at = 100
        self.g.add_node(active)
        # Threshold of 500 ticks; current tick 1000 -> 900 ticks stale -> included
        stale = self.g.stale_nodes(current_tick=1000, threshold_ticks=500)
        self.assertIn(active, stale)

    def test_stale_nodes_excludes_recent(self):
        active = _node("a", status=NodeStatus.ACTIVE)
        active.last_verified_at = 900
        self.g.add_node(active)
        stale = self.g.stale_nodes(current_tick=1000, threshold_ticks=500)
        self.assertNotIn(active, stale)

    def test_stale_nodes_excludes_candidate(self):
        cand = _node("c", status=NodeStatus.CANDIDATE)
        cand.last_verified_at = 0
        self.g.add_node(cand)
        self.assertNotIn(cand, self.g.stale_nodes(current_tick=9999, threshold_ticks=1))


class TestFactoryNodeIOPoints(unittest.TestCase):

    def test_inputs_and_outputs_filtered(self):
        n = _node("prod")
        n.io_points = [
            _iopoint(flow="in",  item="iron-ore"),
            _iopoint(flow="out", item="iron-plate"),
        ]
        self.assertEqual(len(n.inputs()), 1)
        self.assertEqual(len(n.outputs()), 1)
        self.assertEqual(n.inputs()[0].item, "iron-ore")
        self.assertEqual(n.outputs()[0].item, "iron-plate")

    def test_output_for_item(self):
        n = _node("prod")
        n.io_points = [_iopoint(flow="out", item="iron-plate")]
        self.assertIsNotNone(n.output_for("iron-plate"))
        self.assertIsNone(n.output_for("copper-plate"))

    def test_input_for_item(self):
        n = _node("prod")
        n.io_points = [_iopoint(flow="in", item="coal")]
        self.assertIsNotNone(n.input_for("coal"))
        self.assertIsNone(n.input_for("iron-ore"))


# ===========================================================================
# Section 4 -- ChunkGrid
# ===========================================================================

class TestChunkGrid(unittest.TestCase):

    def test_empty_by_default(self):
        g = ChunkGrid()
        self.assertEqual(len(g), 0)
        self.assertFalse(bool(g))

    def test_mark_charted(self):
        g = ChunkGrid()
        g.mark_charted(0, 0)
        self.assertTrue(g.is_charted(0, 0))
        self.assertFalse(g.is_charted(1, 0))

    def test_mark_charted_idempotent(self):
        g = ChunkGrid()
        g.mark_charted(5, 5)
        g.mark_charted(5, 5)
        self.assertEqual(len(g), 1)

    def test_mark_charted_bulk(self):
        g = ChunkGrid()
        g.mark_charted_bulk([ChunkCoord(0, 0), ChunkCoord(1, 0), ChunkCoord(0, 1)])
        self.assertEqual(len(g), 3)

    def test_frontiers_single_chunk(self):
        g = ChunkGrid()
        g.mark_charted(0, 0)
        # A single chunk has 4 uncharted neighbours
        self.assertIn((0, 0), g.frontiers())

    def test_frontiers_surrounded_chunk_not_frontier(self):
        g = ChunkGrid()
        for cx, cy in [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]:
            g.mark_charted(cx, cy)
        # (1,1) is surrounded on all 4 sides
        self.assertNotIn((1, 1), g.frontiers())

    def test_frontiers_empty_grid(self):
        self.assertEqual(ChunkGrid().frontiers(), [])

    def test_chunk_for_position(self):
        g = ChunkGrid()
        self.assertEqual(g.chunk_for_position(Position(0, 0)), (0, 0))
        self.assertEqual(g.chunk_for_position(Position(32, 0)), (1, 0))
        self.assertEqual(g.chunk_for_position(Position(31.9, 31.9)), (0, 0))

    def test_centre_of_chunk(self):
        g = ChunkGrid()
        c = g.centre_of_chunk(0, 0)
        self.assertAlmostEqual(c.x, 16.0)
        self.assertAlmostEqual(c.y, 16.0)

    def test_nearest_frontier_returns_closest(self):
        g = ChunkGrid()
        g.mark_charted_bulk([ChunkCoord(0, 0), ChunkCoord(5, 5), ChunkCoord(10, 10)])
        pos = Position(32 * 5 + 16, 32 * 5 + 16)   # centre of chunk (5,5)
        nearest = g.nearest_frontier(pos)
        self.assertIsNotNone(nearest)

    def test_nearest_frontier_none_when_empty(self):
        self.assertIsNone(ChunkGrid().nearest_frontier(Position(0, 0)))


# ===========================================================================
# Section 5 -- SelfModel container and apply()
# ===========================================================================

class TestSelfModelContainer(unittest.TestCase):

    def setUp(self):
        self.sm = SelfModel()

    def test_has_factory_and_chunks(self):
        self.assertIsNotNone(self.sm.factory)
        self.assertIsNotNone(self.sm.chunks)

    def test_apply_add_node(self):
        n = _node("iron smelter", status=NodeStatus.CANDIDATE)
        patch = SelfModelPatch(layer="factory", action="add_node", node=n)
        self.sm.apply(patch)
        self.assertIsNotNone(self.sm.factory.get_node(n.id))

    def test_apply_promote(self):
        n = _node("prod", status=NodeStatus.CANDIDATE)
        self.sm.factory.add_node(n)
        patch = SelfModelPatch(layer="factory", action="promote", node_id=n.id)
        self.sm.apply(patch)
        self.assertEqual(self.sm.factory.get_node(n.id).status, NodeStatus.ACTIVE)

    def test_apply_discard(self):
        n = _node("prod", status=NodeStatus.CANDIDATE)
        self.sm.factory.add_node(n)
        patch = SelfModelPatch(layer="factory", action="discard", node_id=n.id)
        self.sm.apply(patch)
        self.assertIsNone(self.sm.factory.get_node(n.id))

    def test_apply_add_edge(self):
        a = _node("A", status=NodeStatus.ACTIVE)
        b = _node("B", status=NodeStatus.ACTIVE)
        self.sm.factory.add_node(a)
        self.sm.factory.add_node(b)
        patch = SelfModelPatch(
            layer="factory", action="add_edge",
            from_id=a.id, to_id=b.id, edge_type=EdgeType.ITEM_FLOW,
            item="iron-plate", rate=30.0,
        )
        self.sm.apply(patch)
        self.assertEqual(len(self.sm.factory.all_edges()), 1)
        self.assertEqual(self.sm.factory.all_edges()[0].item, "iron-plate")

    def test_apply_update_status(self):
        n = _node("prod", status=NodeStatus.ACTIVE)
        self.sm.factory.add_node(n)
        patch = SelfModelPatch(
            layer="factory", action="update_status",
            node_id=n.id, new_status=NodeStatus.DEGRADED,
        )
        self.sm.apply(patch)
        self.assertEqual(self.sm.factory.get_node(n.id).status, NodeStatus.DEGRADED)

    def test_apply_update_throughput(self):
        n = _node("prod", status=NodeStatus.ACTIVE)
        self.sm.factory.add_node(n)
        patch = SelfModelPatch(
            layer="factory", action="update_throughput",
            node_id=n.id, throughput={"iron-plate": 28.0}, verified_at=3600,
        )
        self.sm.apply(patch)
        self.assertAlmostEqual(
            self.sm.factory.get_node(n.id).throughput["iron-plate"], 28.0
        )

    def test_apply_update_io_points(self):
        n = _node("prod", status=NodeStatus.ACTIVE)
        self.sm.factory.add_node(n)
        pts = [_iopoint(x=5.0, y=0.0, flow="out")]
        patch = SelfModelPatch(
            layer="factory", action="update_io_points",
            node_id=n.id, io_points=pts,
        )
        self.sm.apply(patch)
        self.assertEqual(len(self.sm.factory.get_node(n.id).io_points), 1)

    def test_apply_unknown_layer_does_not_raise(self):
        patch = SelfModelPatch(layer="chunks", action="add_node")
        # Chunk patches are not yet implemented; should log warning, not raise
        try:
            self.sm.apply(patch)
        except Exception as exc:
            self.fail(f"apply() raised unexpectedly: {exc}")

    def test_apply_malformed_patch_does_not_raise(self):
        # Missing node field for add_node
        patch = SelfModelPatch(layer="factory", action="add_node", node=None)
        try:
            self.sm.apply(patch)
        except Exception as exc:
            self.fail(f"apply() raised on malformed patch: {exc}")

    def test_apply_nonexistent_node_does_not_raise(self):
        # Promote a node that doesn't exist
        patch = SelfModelPatch(layer="factory", action="promote", node_id="ghost-id")
        try:
            self.sm.apply(patch)
        except Exception as exc:
            self.fail(f"apply() raised on missing node: {exc}")


class TestSelfModelConvenienceMethods(unittest.TestCase):

    def setUp(self):
        self.sm = SelfModel()
        self.producer = _node(
            "iron smelter", status=NodeStatus.ACTIVE,
            design_capacity={"iron-plate": 60.0},
            bbox=BoundingBox(Position(0, 0), Position(20, 10)),
        )
        self.sm.factory.add_node(self.producer)

    def test_find_producers(self):
        results = self.sm.find_producers("iron-plate")
        self.assertEqual(len(results), 1)
        self.assertIs(results[0], self.producer)

    def test_active_capacity_for(self):
        self.assertAlmostEqual(self.sm.active_capacity_for("iron-plate"), 60.0)

    def test_overlapping_nodes(self):
        hits = self.sm.overlapping_nodes(BoundingBox(Position(0, 0), Position(5, 5)))
        self.assertIn(self.producer, hits)

    def test_stale_nodes(self):
        self.producer.last_verified_at = 0
        stale = self.sm.stale_nodes(current_tick=10000, threshold_ticks=100)
        self.assertIn(self.producer, stale)


if __name__ == "__main__":
    unittest.main(verbosity=2)