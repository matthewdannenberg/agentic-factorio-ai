"""
tests/unit/agent/test_self_model.py

Tests for agent/self_model.py

Run with:  python -m pytest tests/unit/agent/test_self_model.py -v
       or:  python -m unittest tests.unit.agent.test_self_model
"""

from __future__ import annotations

import unittest

from world.state import Position
from agent.self_model import (
    BoundingBox,
    EdgeType,
    NodeStatus,
    NodeType,
    SelfModel,
    SelfModelEdge,
    SelfModelNode,
)


def _bbox(x1=0.0, y1=0.0, x2=10.0, y2=10.0) -> BoundingBox:
    return BoundingBox(
        top_left=Position(x1, y1),
        bottom_right=Position(x2, y2),
    )


def _node(
    label: str = "test",
    node_type: NodeType = NodeType.PRODUCTION_LINE,
    status: NodeStatus = NodeStatus.CANDIDATE,
    throughput: dict | None = None,
    created_at: int = 0,
) -> SelfModelNode:
    return SelfModelNode(
        type=node_type,
        status=status,
        bounding_box=_bbox(),
        label=label,
        throughput=throughput or {},
        created_at=created_at,
    )


class TestBoundingBox(unittest.TestCase):
    def test_contains_interior_point(self):
        bb = _bbox(0, 0, 10, 10)
        self.assertTrue(bb.contains(Position(5, 5)))

    def test_contains_corner_point(self):
        bb = _bbox(0, 0, 10, 10)
        self.assertTrue(bb.contains(Position(0, 0)))
        self.assertTrue(bb.contains(Position(10, 10)))

    def test_does_not_contain_exterior_point(self):
        bb = _bbox(0, 0, 10, 10)
        self.assertFalse(bb.contains(Position(11, 5)))
        self.assertFalse(bb.contains(Position(-1, 5)))

    def test_overlaps_when_sharing_area(self):
        a = _bbox(0, 0, 10, 10)
        b = _bbox(5, 5, 15, 15)
        self.assertTrue(a.overlaps(b))
        self.assertTrue(b.overlaps(a))

    def test_no_overlap_when_disjoint(self):
        a = _bbox(0, 0, 5, 5)
        b = _bbox(10, 10, 20, 20)
        self.assertFalse(a.overlaps(b))

    def test_edge_touching_counts_as_overlap(self):
        a = _bbox(0, 0, 10, 10)
        b = _bbox(10, 0, 20, 10)
        self.assertTrue(a.overlaps(b))

    def test_dimensions(self):
        bb = _bbox(2, 3, 12, 8)
        self.assertAlmostEqual(bb.width, 10.0)
        self.assertAlmostEqual(bb.height, 5.0)


class TestSelfModelNodes(unittest.TestCase):
    def setUp(self):
        self.sm = SelfModel()

    def test_add_node_returns_id(self):
        n = _node("furnace A")
        node_id = self.sm.add_node(n)
        self.assertEqual(node_id, n.id)

    def test_get_node_by_id(self):
        n = _node("furnace A")
        self.sm.add_node(n)
        retrieved = self.sm.get_node(n.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.label, "furnace A")

    def test_get_nonexistent_node_returns_none(self):
        self.assertIsNone(self.sm.get_node("nonexistent-id"))

    def test_all_nodes_empty_graph(self):
        self.assertEqual(self.sm.all_nodes(), [])

    def test_all_nodes_contains_added_nodes(self):
        n1 = _node("a")
        n2 = _node("b")
        self.sm.add_node(n1)
        self.sm.add_node(n2)
        ids = {n.id for n in self.sm.all_nodes()}
        self.assertIn(n1.id, ids)
        self.assertIn(n2.id, ids)

    def test_len_counts_nodes(self):
        self.assertEqual(len(self.sm), 0)
        self.sm.add_node(_node("a"))
        self.assertEqual(len(self.sm), 1)
        self.sm.add_node(_node("b"))
        self.assertEqual(len(self.sm), 2)


class TestSelfModelQueryNodes(unittest.TestCase):
    def setUp(self):
        self.sm = SelfModel()
        self.prod = _node("prod", NodeType.PRODUCTION_LINE, NodeStatus.ACTIVE)
        self.res = _node("res", NodeType.RESOURCE_SITE, NodeStatus.ACTIVE)
        self.cand = _node("cand", NodeType.PRODUCTION_LINE, NodeStatus.CANDIDATE)
        for n in [self.prod, self.res, self.cand]:
            self.sm.add_node(n)

    def test_query_all_nodes(self):
        self.assertEqual(len(self.sm.query_nodes()), 3)

    def test_query_by_type(self):
        prod_nodes = self.sm.query_nodes(type=NodeType.PRODUCTION_LINE)
        self.assertEqual(len(prod_nodes), 2)

    def test_query_by_status(self):
        active = self.sm.query_nodes(status=NodeStatus.ACTIVE)
        self.assertEqual(len(active), 2)

    def test_query_by_type_and_status(self):
        results = self.sm.query_nodes(
            type=NodeType.PRODUCTION_LINE, status=NodeStatus.CANDIDATE
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, self.cand.id)

    def test_query_returns_empty_for_missing(self):
        results = self.sm.query_nodes(type=NodeType.TRAIN_STATION)
        self.assertEqual(results, [])


class TestSelfModelEdges(unittest.TestCase):
    def setUp(self):
        self.sm = SelfModel()
        self.a = _node("A")
        self.b = _node("B")
        self.c = _node("C")
        self.sm.add_node(self.a)
        self.sm.add_node(self.b)
        self.sm.add_node(self.c)

    def test_add_edge(self):
        self.sm.add_edge(self.a.id, self.b.id, EdgeType.FEEDS_INTO)
        edges = self.sm.all_edges()
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].from_id, self.a.id)
        self.assertEqual(edges[0].to_id, self.b.id)

    def test_duplicate_edge_ignored(self):
        self.sm.add_edge(self.a.id, self.b.id, EdgeType.FEEDS_INTO)
        self.sm.add_edge(self.a.id, self.b.id, EdgeType.FEEDS_INTO)
        self.assertEqual(len(self.sm.all_edges()), 1)

    def test_different_edge_type_is_not_duplicate(self):
        self.sm.add_edge(self.a.id, self.b.id, EdgeType.FEEDS_INTO)
        self.sm.add_edge(self.a.id, self.b.id, EdgeType.DEPENDS_ON)
        self.assertEqual(len(self.sm.all_edges()), 2)

    def test_edge_with_missing_from_raises(self):
        with self.assertRaises(ValueError):
            self.sm.add_edge("ghost-id", self.b.id, EdgeType.FEEDS_INTO)

    def test_edge_with_missing_to_raises(self):
        with self.assertRaises(ValueError):
            self.sm.add_edge(self.a.id, "ghost-id", EdgeType.FEEDS_INTO)


class TestSelfModelPath(unittest.TestCase):
    def setUp(self):
        self.sm = SelfModel()
        self.a = _node("A")
        self.b = _node("B")
        self.c = _node("C")
        self.d = _node("D")
        for n in [self.a, self.b, self.c, self.d]:
            self.sm.add_node(n)
        # A → B → C (D is isolated)
        self.sm.add_edge(self.a.id, self.b.id, EdgeType.FEEDS_INTO)
        self.sm.add_edge(self.b.id, self.c.id, EdgeType.FEEDS_INTO)

    def test_direct_path(self):
        path = self.sm.query_path(self.a.id, self.b.id)
        self.assertIsNotNone(path)
        self.assertEqual(path, [self.a.id, self.b.id])

    def test_two_hop_path(self):
        path = self.sm.query_path(self.a.id, self.c.id)
        self.assertIsNotNone(path)
        self.assertEqual(path, [self.a.id, self.b.id, self.c.id])

    def test_no_path_returns_none(self):
        path = self.sm.query_path(self.a.id, self.d.id)
        self.assertIsNone(path)

    def test_path_to_self(self):
        path = self.sm.query_path(self.a.id, self.a.id)
        self.assertEqual(path, [self.a.id])

    def test_directed_no_backward_path(self):
        # Edges go A→B→C; there's no path from C→A
        path = self.sm.query_path(self.c.id, self.a.id)
        self.assertIsNone(path)

    def test_path_with_nonexistent_node(self):
        path = self.sm.query_path("ghost", self.b.id)
        self.assertIsNone(path)


class TestSelfModelProducers(unittest.TestCase):
    def setUp(self):
        self.sm = SelfModel()

    def test_find_producers_returns_nodes_with_item(self):
        n1 = _node("iron smelter", throughput={"iron-plate": 30.0})
        n2 = _node("copper smelter", throughput={"copper-plate": 20.0})
        n3 = _node("iron smelter 2", throughput={"iron-plate": 15.0})
        for n in [n1, n2, n3]:
            self.sm.add_node(n)

        producers = self.sm.find_producers("iron-plate")
        self.assertEqual(len(producers), 2)
        ids = {p.id for p in producers}
        self.assertIn(n1.id, ids)
        self.assertIn(n3.id, ids)

    def test_find_producers_empty_when_none(self):
        n = _node("furnace", throughput={"iron-plate": 10.0})
        self.sm.add_node(n)
        self.assertEqual(self.sm.find_producers("copper-plate"), [])

    def test_find_capacity_sums_active_only(self):
        active = _node("a", status=NodeStatus.ACTIVE, throughput={"iron-plate": 30.0})
        candidate = _node("c", status=NodeStatus.CANDIDATE, throughput={"iron-plate": 10.0})
        degraded = _node("d", status=NodeStatus.DEGRADED, throughput={"iron-plate": 5.0})
        for n in [active, candidate, degraded]:
            self.sm.add_node(n)

        capacity = self.sm.find_capacity("iron-plate")
        self.assertAlmostEqual(capacity, 30.0)   # only ACTIVE counts

    def test_find_capacity_zero_when_no_active(self):
        n = _node("prod", status=NodeStatus.CANDIDATE, throughput={"coal": 100.0})
        self.sm.add_node(n)
        self.assertAlmostEqual(self.sm.find_capacity("coal"), 0.0)


class TestSelfModelCandidateLifecycle(unittest.TestCase):
    def setUp(self):
        self.sm = SelfModel()

    def test_promote_candidate_to_active(self):
        n = _node("prod", status=NodeStatus.CANDIDATE)
        self.sm.add_node(n)
        self.sm.promote_candidate(n.id)
        self.assertEqual(self.sm.get_node(n.id).status, NodeStatus.ACTIVE)

    def test_promote_non_candidate_raises(self):
        n = _node("prod", status=NodeStatus.ACTIVE)
        self.sm.add_node(n)
        with self.assertRaises(ValueError):
            self.sm.promote_candidate(n.id)

    def test_promote_nonexistent_raises(self):
        with self.assertRaises(ValueError):
            self.sm.promote_candidate("ghost-id")

    def test_discard_candidate_removes_node(self):
        n = _node("prod", status=NodeStatus.CANDIDATE)
        self.sm.add_node(n)
        self.sm.discard_candidate(n.id)
        self.assertIsNone(self.sm.get_node(n.id))
        self.assertEqual(len(self.sm), 0)

    def test_discard_candidate_removes_edges(self):
        a = _node("A", status=NodeStatus.ACTIVE)
        b = _node("B", status=NodeStatus.CANDIDATE)
        self.sm.add_node(a)
        self.sm.add_node(b)
        self.sm.add_edge(a.id, b.id, EdgeType.FEEDS_INTO)
        self.sm.discard_candidate(b.id)
        self.assertEqual(len(self.sm.all_edges()), 0)

    def test_discard_non_candidate_raises(self):
        n = _node("prod", status=NodeStatus.ACTIVE)
        self.sm.add_node(n)
        with self.assertRaises(ValueError):
            self.sm.discard_candidate(n.id)


class TestSelfModelSubgraph(unittest.TestCase):
    def setUp(self):
        self.sm = SelfModel()
        self.a = _node("A")
        self.b = _node("B")
        self.c = _node("C")
        for n in [self.a, self.b, self.c]:
            self.sm.add_node(n)
        self.sm.add_edge(self.a.id, self.b.id, EdgeType.FEEDS_INTO)
        self.sm.add_edge(self.b.id, self.c.id, EdgeType.FEEDS_INTO)

    def test_subgraph_contains_only_specified_nodes(self):
        sub = self.sm.subgraph([self.a.id, self.b.id])
        ids = {n.id for n in sub.all_nodes()}
        self.assertIn(self.a.id, ids)
        self.assertIn(self.b.id, ids)
        self.assertNotIn(self.c.id, ids)

    def test_subgraph_contains_only_internal_edges(self):
        sub = self.sm.subgraph([self.a.id, self.b.id])
        edges = sub.all_edges()
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].from_id, self.a.id)
        self.assertEqual(edges[0].to_id, self.b.id)

    def test_subgraph_skips_missing_node_ids(self):
        sub = self.sm.subgraph([self.a.id, "ghost-id"])
        self.assertEqual(len(sub), 1)

    def test_empty_subgraph(self):
        sub = self.sm.subgraph([])
        self.assertEqual(len(sub), 0)
        self.assertEqual(sub.all_edges(), [])


class TestEmptyGraph(unittest.TestCase):
    def setUp(self):
        self.sm = SelfModel()

    def test_query_nodes_empty(self):
        self.assertEqual(self.sm.query_nodes(), [])

    def test_query_path_both_missing(self):
        self.assertIsNone(self.sm.query_path("a", "b"))

    def test_find_producers_empty(self):
        self.assertEqual(self.sm.find_producers("iron-plate"), [])

    def test_find_capacity_zero(self):
        self.assertAlmostEqual(self.sm.find_capacity("iron-plate"), 0.0)

    def test_all_edges_empty(self):
        self.assertEqual(self.sm.all_edges(), [])

    def test_get_node_returns_none(self):
        self.assertIsNone(self.sm.get_node("any-id"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
