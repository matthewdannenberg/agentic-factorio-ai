"""
tests/unit/execution/test_predicates.py

Unit tests for execution/predicates.py

Covers: is_at, is_reachable, can_mine, player_has_item, can_destroy

Run with:  python -m pytest tests/unit/execution/test_predicates.py -v
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

from execution.predicates import is_at, is_reachable, can_mine, player_has_item, can_destroy
from world import Position, NaturalObject


# ===========================================================================
# Stubs
# ===========================================================================

class _WQ:
    def __init__(self, position=None, reachable=None, inventory=None):
        self._position  = position or Position(x=0.0, y=0.0)
        self._reachable = reachable or []
        self._inventory = dict(inventory or {})
        self.state      = MagicMock()
        self.state.player.reachable = self._reachable

    def player_position(self) -> Position:
        return self._position

    def entity_by_id(self, eid):
        return MagicMock() if eid in getattr(self, "_entities", {}) else None

    def inventory_count(self, item: str) -> int:
        return self._inventory.get(item, 0)

    def add_entity(self, eid):
        if not hasattr(self, "_entities"):
            self._entities = set()
        self._entities.add(eid)


def _nat(name="tree-01", eid=1, proto="tree"):
    return NaturalObject(entity_id=eid, name=name,
                        position=Position(0, 0), prototype_type=proto)


def _kb_with(name, minable=True, placeholder=False):
    """Build a minimal KnowledgeBase stub for one entity."""
    record = MagicMock()
    record.minable = minable
    record.is_placeholder = placeholder
    kb = MagicMock()
    kb.get_entity.return_value = record if not placeholder else _make_placeholder()
    return kb


def _make_placeholder():
    r = MagicMock()
    r.is_placeholder = True
    r.minable = False
    return r


def _kb_unknown():
    """KB that returns None for any entity."""
    kb = MagicMock()
    kb.get_entity.return_value = None
    return kb


# ===========================================================================
# is_at
# ===========================================================================

class TestIsAt(unittest.TestCase):

    def test_exact_position(self):
        wq = _WQ(position=Position(x=5.0, y=5.0))
        self.assertTrue(is_at(Position(x=5.0, y=5.0), wq, tolerance=1.0))

    def test_within_tolerance(self):
        wq = _WQ(position=Position(x=5.0, y=5.0))
        self.assertTrue(is_at(Position(x=5.5, y=5.0), wq, tolerance=1.0))

    def test_at_tolerance_boundary(self):
        # Exactly at tolerance distance — should be True (<=)
        wq = _WQ(position=Position(x=5.0, y=5.0))
        self.assertTrue(is_at(Position(x=6.0, y=5.0), wq, tolerance=1.0))

    def test_outside_tolerance(self):
        wq = _WQ(position=Position(x=0.0, y=0.0))
        self.assertFalse(is_at(Position(x=10.0, y=10.0), wq, tolerance=1.0))

    def test_default_tolerance_one(self):
        wq = _WQ(position=Position(x=0.0, y=0.0))
        self.assertTrue(is_at(Position(x=0.9, y=0.0), wq))
        self.assertFalse(is_at(Position(x=1.1, y=0.0), wq))


# ===========================================================================
# is_reachable
# ===========================================================================

class TestIsReachable(unittest.TestCase):

    def test_reachable_when_in_list(self):
        wq = _WQ(reachable=[1, 2, 3])
        self.assertTrue(is_reachable(2, wq))

    def test_not_reachable_when_absent(self):
        wq = _WQ(reachable=[1, 3])
        self.assertFalse(is_reachable(2, wq))

    def test_empty_reachable_list(self):
        wq = _WQ(reachable=[])
        self.assertFalse(is_reachable(1, wq))


# ===========================================================================
# can_mine
# ===========================================================================

class TestCanMine(unittest.TestCase):

    def test_can_mine_when_reachable_and_present(self):
        wq = _WQ(reachable=[5])
        wq.add_entity(5)
        self.assertTrue(can_mine(5, wq))

    def test_cannot_mine_when_not_reachable(self):
        wq = _WQ(reachable=[])
        wq.add_entity(5)
        self.assertFalse(can_mine(5, wq))

    def test_cannot_mine_when_entity_absent(self):
        wq = _WQ(reachable=[5])
        # entity not in wq — entity_by_id returns None
        self.assertFalse(can_mine(5, wq))


# ===========================================================================
# player_has_item
# ===========================================================================

class TestPlayerHasItem(unittest.TestCase):

    def test_has_item_at_count(self):
        wq = _WQ(inventory={"iron-ore": 10})
        self.assertTrue(player_has_item("iron-ore", 10, wq))

    def test_has_item_above_count(self):
        wq = _WQ(inventory={"iron-ore": 20})
        self.assertTrue(player_has_item("iron-ore", 10, wq))

    def test_has_item_below_count(self):
        wq = _WQ(inventory={"iron-ore": 5})
        self.assertFalse(player_has_item("iron-ore", 10, wq))

    def test_item_absent(self):
        wq = _WQ(inventory={})
        self.assertFalse(player_has_item("iron-ore", 1, wq))


# ===========================================================================
# can_destroy
# ===========================================================================

class TestCanDestroy(unittest.TestCase):
    """
    can_destroy checks NaturalObject.is_minable and EntityRecord.minable.

    Key insight from live testing: mineable_properties.minable=True for
    trees, rocks, machines, and chests (all MineEntity-destroyable).
    minable=False for cliffs (require cliff explosives, Phase 7).
    has_mining_trigger was removed — it fires for cosmetic effects only.
    """

    def test_tree_is_destroyable(self):
        obj = _nat("tree-01", eid=1, proto="tree")
        kb  = _kb_with("tree-01", minable=True)
        self.assertTrue(can_destroy(obj, kb))

    def test_rock_is_destroyable(self):
        obj = _nat("rock-huge", eid=2, proto="simple-entity")
        kb  = _kb_with("rock-huge", minable=True)
        self.assertTrue(can_destroy(obj, kb))

    def test_cliff_not_destroyable(self):
        # Vanilla cliffs have minable=False — require cliff explosives
        obj = _nat("cliff", eid=3, proto="cliff")
        kb  = _kb_with("cliff", minable=False)
        self.assertFalse(can_destroy(obj, kb))

    def test_modded_directly_minable_cliff_is_destroyable(self):
        # A mod that makes cliffs directly minable
        obj = _nat("small-cliff", eid=8, proto="cliff")
        kb  = _kb_with("small-cliff", minable=True)
        self.assertTrue(can_destroy(obj, kb))

    def test_modded_non_minable_obstacle_not_destroyable(self):
        # A modded entity with minable=False for any reason
        obj = _nat("modded-locked-obstacle", eid=7, proto="simple-entity")
        kb  = _kb_with("modded-locked-obstacle", minable=False)
        self.assertFalse(can_destroy(obj, kb))

    def test_zero_entity_id_not_destroyable(self):
        obj = NaturalObject(entity_id=0, name="cliff",
                            position=Position(0, 0), prototype_type="cliff")
        kb  = _kb_with("cliff", minable=False)
        self.assertFalse(can_destroy(obj, kb))

    def test_unknown_entity_not_destroyable(self):
        obj = _nat("modded-unknown-thing", eid=5)
        kb  = _kb_unknown()
        self.assertFalse(can_destroy(obj, kb))

    def test_placeholder_record_not_destroyable(self):
        obj = _nat("some-entity", eid=6)
        kb  = MagicMock()
        kb.get_entity.return_value = _make_placeholder()
        self.assertFalse(can_destroy(obj, kb))


if __name__ == "__main__":
    unittest.main(verbosity=2)
