"""
tests/integration/test_StateParser_WorldState.py

Integration tests for StateParser with WorldState + WorldQuery.

StateParser constructs WorldState snapshots directly — that is the bridge's
legitimate construction path. Tests verify the output through WorldQuery,
which is the correct consumer interface for all other layers.

Run with:  python tests/integration/test_StateParser_WorldState.py
"""

from __future__ import annotations

import json
import unittest

from bridge import StateParser
from world import WorldQuery, EntityStatus, ChunkCoord, CraftingQueueEntry


class TestStateParserWorldQueryAPI(unittest.TestCase):
    """Verify that parser output satisfies WorldQuery accessor contracts."""

    def setUp(self):
        self.parser = StateParser()

    def _full_state_json(self) -> str:
        return json.dumps({
            "tick": 7200,
            "player": {
                "position": {"x": 12.5, "y": -8.0},
                "health": 65.0,
                "inventory": [
                    {"item": "iron-plate", "count": 100},
                    {"item": "iron-plate", "count": 50},
                    {"item": "coal", "count": 30},
                ],
                "reachable": [3, 7],
            },
            "entities": [
                {"unit_number": 3, "name": "stone-furnace",
                 "position": {"x": 10, "y": -8}, "status": "working"},
                {"unit_number": 7, "name": "burner-mining-drill",
                 "position": {"x": 14, "y": -8}, "status": "no_minable_resources"},
            ],
            "resource_map": [
                {"resource_type": "iron-ore", "position": {"x": 50, "y": 0},
                 "amount": 150000, "size": 300, "observed_at": 7200},
                {"resource_type": "coal", "position": {"x": -20, "y": 10},
                 "amount": 80000, "size": 150, "observed_at": 7100},
            ],
            "research": {"unlocked": ["automation"], "queued": [], "in_progress": None},
            "logistics": {"power": {"produced_kw": 900.0, "consumed_kw": 750.0,
                                    "accumulated_kj": 0.0, "satisfaction": 1.0}},
            "damaged_entities": [],
            "destroyed_entities": [],
            "threat": {},
        })

    def _parse(self) -> WorldQuery:
        state = self.parser.parse(self._full_state_json(), current_tick=7200)
        return WorldQuery(state)

    def test_entity_by_id(self):
        wq = self._parse()
        e = wq.entity_by_id(3)
        self.assertIsNotNone(e)
        self.assertEqual(e.name, "stone-furnace")

    def test_entity_by_id_missing(self):
        wq = self._parse()
        self.assertIsNone(wq.entity_by_id(999))

    def test_entities_by_name(self):
        wq = self._parse()
        drills = wq.entities_by_name("burner-mining-drill")
        self.assertEqual(len(drills), 1)

    def test_entities_by_status(self):
        wq = self._parse()
        working = wq.entities_by_status(EntityStatus.WORKING)
        self.assertEqual(len(working), 1)
        self.assertEqual(working[0].name, "stone-furnace")

    def test_resources_of_type(self):
        wq = self._parse()
        iron = wq.resources_of_type("iron-ore")
        self.assertEqual(len(iron), 1)
        self.assertEqual(iron[0].amount, 150000)

    def test_inventory_count_multi_slot(self):
        """Counts across multiple slots of the same item are summed."""
        wq = self._parse()
        self.assertEqual(wq.inventory_count("iron-plate"), 150)  # 100+50

    def test_power_headroom(self):
        wq = self._parse()
        self.assertAlmostEqual(wq.power.headroom_kw, 150.0)

    def test_recent_losses_empty(self):
        wq = self._parse()
        self.assertEqual(wq.recent_losses, [])

    def test_has_damage_false(self):
        wq = self._parse()
        self.assertFalse(wq.has_damage)




class TestStateParserExplorationFields(unittest.TestCase):
    """
    Verify that StateParser correctly parses newly_charted_chunks and
    nearby_uncharted_chunks from bridge JSON into ExplorationState and
    exposes them via WorldQuery.
    """

    def setUp(self):
        self.parser = StateParser()

    def _parse_player(self, extra_player_fields: dict) -> WorldQuery:
        """Build a minimal state JSON with custom player fields and parse it."""
        player = {
            "position": {"x": 0.0, "y": 0.0},
            "health": 100.0,
            "inventory": [],
            "reachable": [],
            "charted_chunks": 42,
        }
        player.update(extra_player_fields)
        raw = json.dumps({"tick": 1000, "player": player})
        state = self.parser.parse(raw, current_tick=1000)
        return WorldQuery(state)

    # --- newly_charted_chunks ------------------------------------------------

    def test_newly_charted_chunks_parsed(self):
        wq = self._parse_player({"newly_charted_chunks": [
            {"cx": 1, "cy": 2},
            {"cx": -3, "cy": 0},
        ]})
        result = wq.newly_charted_chunks
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].cx, 1)
        self.assertEqual(result[0].cy, 2)
        self.assertEqual(result[1].cx, -3)
        self.assertEqual(result[1].cy, 0)

    def test_newly_charted_chunks_are_chunk_coords(self):
        wq = self._parse_player({"newly_charted_chunks": [{"cx": 5, "cy": 7}]})
        self.assertIsInstance(wq.newly_charted_chunks[0], ChunkCoord)

    def test_newly_charted_chunks_absent_gives_empty(self):
        """Older bridge payloads without the field should parse safely."""
        wq = self._parse_player({})
        self.assertEqual(wq.newly_charted_chunks, [])

    def test_newly_charted_chunks_empty_list(self):
        """Explicit empty list (normal case — no new chunks this poll)."""
        wq = self._parse_player({"newly_charted_chunks": []})
        self.assertEqual(wq.newly_charted_chunks, [])

    def test_newly_charted_chunks_malformed_entry_skipped(self):
        """Entries missing cx or cy are silently dropped."""
        wq = self._parse_player({"newly_charted_chunks": [
            {"cx": 1, "cy": 2},    # valid
            {"cx": 3},             # missing cy — skipped
            {"cy": 4},             # missing cx — skipped
            {"cx": 5, "cy": 6},    # valid
        ]})
        self.assertEqual(len(wq.newly_charted_chunks), 2)
        self.assertEqual(wq.newly_charted_chunks[0].cx, 1)
        self.assertEqual(wq.newly_charted_chunks[1].cx, 5)

    def test_newly_charted_chunks_negative_coords(self):
        """Negative chunk coordinates are valid (south-west of spawn)."""
        wq = self._parse_player({"newly_charted_chunks": [{"cx": -10, "cy": -7}]})
        c = wq.newly_charted_chunks[0]
        self.assertEqual(c.cx, -10)
        self.assertEqual(c.cy, -7)

    def test_newly_charted_chunks_large_batch(self):
        """A large initial delta (full map on reconnect) is parsed correctly."""
        batch = [{"cx": i, "cy": j} for i in range(10) for j in range(10)]
        wq = self._parse_player({"newly_charted_chunks": batch})
        self.assertEqual(len(wq.newly_charted_chunks), 100)

    # --- nearby_uncharted_chunks ---------------------------------------------

    def test_nearby_uncharted_chunks_parsed(self):
        wq = self._parse_player({"nearby_uncharted_chunks": [
            {"cx": 7, "cy": 3},
            {"cx": 7, "cy": 5},
            {"cx": 6, "cy": 4},
        ]})
        result = wq.nearby_uncharted_chunks
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].cx, 7)
        self.assertEqual(result[2].cx, 6)

    def test_nearby_uncharted_chunks_are_chunk_coords(self):
        wq = self._parse_player({"nearby_uncharted_chunks": [{"cx": 2, "cy": -1}]})
        self.assertIsInstance(wq.nearby_uncharted_chunks[0], ChunkCoord)

    def test_nearby_uncharted_chunks_absent_gives_empty(self):
        wq = self._parse_player({})
        self.assertEqual(wq.nearby_uncharted_chunks, [])

    def test_nearby_uncharted_chunks_empty_list(self):
        """Player surrounded by charted territory — empty list is valid."""
        wq = self._parse_player({"nearby_uncharted_chunks": []})
        self.assertEqual(wq.nearby_uncharted_chunks, [])

    def test_nearby_uncharted_chunks_malformed_entry_skipped(self):
        wq = self._parse_player({"nearby_uncharted_chunks": [
            {"cx": 1, "cy": 2},    # valid
            {"not_cx": 0},         # malformed — skipped
            {"cx": 3, "cy": 4},    # valid
        ]})
        self.assertEqual(len(wq.nearby_uncharted_chunks), 2)

    def test_nearby_uncharted_maximum_bounded_by_scan_radius(self):
        """At r=6 the scan is (2*6+1)^2 = 169 chunks; all uncharted gives 169 entries."""
        r = 6
        batch = [{"cx": dx, "cy": dy}
                 for dx in range(-r, r+1)
                 for dy in range(-r, r+1)]
        wq = self._parse_player({"nearby_uncharted_chunks": batch})
        self.assertEqual(len(wq.nearby_uncharted_chunks), (2*r+1)**2)

    # --- both fields together ------------------------------------------------

    def test_both_fields_parsed_independently(self):
        """newly_charted and nearby_uncharted are distinct and don't interfere."""
        wq = self._parse_player({
            "charted_chunks": 100,
            "newly_charted_chunks": [{"cx": 10, "cy": 10}],
            "nearby_uncharted_chunks": [{"cx": 15, "cy": 10}, {"cx": 10, "cy": 15}],
        })
        self.assertEqual(wq.charted_chunks, 100)
        self.assertEqual(len(wq.newly_charted_chunks), 1)
        self.assertEqual(len(wq.nearby_uncharted_chunks), 2)
        self.assertEqual(wq.newly_charted_chunks[0].cx, 10)
        self.assertEqual(wq.nearby_uncharted_chunks[1].cy, 15)

    def test_charted_chunks_count_unaffected_by_new_fields(self):
        """The count field is independent of the chunk lists."""
        wq = self._parse_player({
            "charted_chunks": 77,
            "newly_charted_chunks": [{"cx": 1, "cy": 1}, {"cx": 2, "cy": 2}],
            "nearby_uncharted_chunks": [{"cx": 5, "cy": 5}],
        })
        self.assertEqual(wq.charted_chunks, 77)


class TestStateParserExplorationFullRoundtrip(unittest.TestCase):
    """
    End-to-end: a realistic bridge payload with exploration data flows
    correctly through StateParser → WorldState → WorldQuery.
    """

    def setUp(self):
        self.parser = StateParser()

    def _realistic_state(self) -> str:
        return json.dumps({
            "tick": 3600,
            "player": {
                "position": {"x": 64.0, "y": 32.0},
                "health": 100.0,
                "inventory": [{"item": "iron-ore", "count": 10}],
                "reachable": [],
                "charted_chunks": 25,
                "newly_charted_chunks": [
                    {"cx": 3, "cy": 1},
                    {"cx": 4, "cy": 1},
                ],
                "nearby_uncharted_chunks": [
                    {"cx": 5, "cy": 1},
                    {"cx": 5, "cy": 0},
                    {"cx": 2, "cy": 1},
                ],
                "movement_status": "idle",
                "inventory_size": 80,
            },
            "entities": [],
            "resource_map": [],
            "research": {"unlocked": ["automation"], "queued": []},
        })

    def _parse(self) -> WorldQuery:
        return WorldQuery(
            self.parser.parse(self._realistic_state(), current_tick=3600)
        )

    def test_charted_chunks_roundtrip(self):
        self.assertEqual(self._parse().charted_chunks, 25)

    def test_newly_charted_count_and_values(self):
        wq = self._parse()
        self.assertEqual(len(wq.newly_charted_chunks), 2)
        cxs = {c.cx for c in wq.newly_charted_chunks}
        self.assertIn(3, cxs)
        self.assertIn(4, cxs)

    def test_nearby_uncharted_count_and_values(self):
        wq = self._parse()
        self.assertEqual(len(wq.nearby_uncharted_chunks), 3)
        cxs = {c.cx for c in wq.nearby_uncharted_chunks}
        self.assertIn(5, cxs)
        self.assertIn(2, cxs)

    def test_inventory_unaffected_by_exploration_fields(self):
        self.assertEqual(self._parse().inventory_count("iron-ore"), 10)

    def test_exploration_state_accessible_via_player(self):
        from world.observable.state import WorldState
        state = self.parser.parse(self._realistic_state(), current_tick=3600)
        exp = state.player.exploration
        self.assertEqual(exp.charted_chunks, 25)
        self.assertEqual(len(exp.newly_charted_chunks), 2)
        self.assertEqual(len(exp.nearby_uncharted_chunks), 3)



class TestStateParserCraftingQueue(unittest.TestCase):
    """
    Verify that StateParser correctly parses crafting_queue and
    crafting_queue_size from bridge JSON into PlayerState and exposes
    them via WorldQuery.
    """

    def setUp(self):
        self.parser = StateParser()

    def _parse_player(self, extra_player_fields: dict) -> WorldQuery:
        player = {
            "position": {"x": 0.0, "y": 0.0},
            "health": 100.0,
            "inventory": [],
            "reachable": [],
            "charted_chunks": 0,
        }
        player.update(extra_player_fields)
        raw = json.dumps({"tick": 100, "player": player})
        return WorldQuery(self.parser.parse(raw, current_tick=100))

    # --- crafting_queue entries ---

    def test_empty_queue_parsed(self):
        wq = self._parse_player({"crafting_queue": [], "crafting_queue_size": 0})
        self.assertEqual(wq.crafting_queue, [])
        self.assertEqual(wq.crafting_queue_size, 0)

    def test_single_entry_parsed(self):
        wq = self._parse_player({
            "crafting_queue": [
                {"recipe": "iron-gear-wheel", "count": 5, "progress": 0.4}
            ],
            "crafting_queue_size": 1,
        })
        self.assertEqual(len(wq.crafting_queue), 1)
        entry = wq.crafting_queue[0]
        self.assertIsInstance(entry, CraftingQueueEntry)
        self.assertEqual(entry.recipe, "iron-gear-wheel")
        self.assertEqual(entry.count, 5)
        self.assertAlmostEqual(entry.progress, 0.4)

    def test_multiple_entries_parsed(self):
        wq = self._parse_player({
            "crafting_queue": [
                {"recipe": "iron-gear-wheel", "count": 5, "progress": 0.75},
                {"recipe": "copper-cable",    "count": 20, "progress": 0.0},
                {"recipe": "electronic-circuit", "count": 3, "progress": 0.0},
            ],
            "crafting_queue_size": 3,
        })
        self.assertEqual(len(wq.crafting_queue), 3)
        self.assertEqual(wq.crafting_queue_size, 3)
        self.assertEqual(wq.crafting_queue[0].recipe, "iron-gear-wheel")
        self.assertAlmostEqual(wq.crafting_queue[0].progress, 0.75)
        self.assertEqual(wq.crafting_queue[1].recipe, "copper-cable")
        self.assertAlmostEqual(wq.crafting_queue[1].progress, 0.0)
        self.assertEqual(wq.crafting_queue[2].recipe, "electronic-circuit")

    def test_entries_are_crafting_queue_entry_instances(self):
        wq = self._parse_player({
            "crafting_queue": [{"recipe": "coal", "count": 1, "progress": 0.0}],
            "crafting_queue_size": 1,
        })
        self.assertIsInstance(wq.crafting_queue[0], CraftingQueueEntry)

    def test_absent_queue_defaults_to_empty(self):
        """Older bridge payloads without crafting_queue parse safely."""
        wq = self._parse_player({})
        self.assertEqual(wq.crafting_queue, [])
        self.assertEqual(wq.crafting_queue_size, 0)

    def test_absent_queue_size_defaults_to_zero(self):
        wq = self._parse_player({"crafting_queue": []})
        self.assertEqual(wq.crafting_queue_size, 0)

    def test_malformed_entry_skipped(self):
        """Entries missing recipe or count are silently dropped."""
        wq = self._parse_player({
            "crafting_queue": [
                {"recipe": "iron-gear-wheel", "count": 5, "progress": 0.5},  # valid
                {"count": 3, "progress": 0.0},                                # missing recipe
                {"recipe": "coal"},                                            # missing count
                {"recipe": "copper-cable", "count": 10, "progress": 0.0},    # valid
            ],
            "crafting_queue_size": 2,
        })
        self.assertEqual(len(wq.crafting_queue), 2)
        self.assertEqual(wq.crafting_queue[0].recipe, "iron-gear-wheel")
        self.assertEqual(wq.crafting_queue[1].recipe, "copper-cable")

    def test_progress_defaults_to_zero_when_absent(self):
        wq = self._parse_player({
            "crafting_queue": [{"recipe": "iron-gear-wheel", "count": 5}],
            "crafting_queue_size": 1,
        })
        self.assertAlmostEqual(wq.crafting_queue[0].progress, 0.0)

    def test_queue_size_independent_of_list_length(self):
        """crafting_queue_size is parsed as-is from the bridge, not derived
        from len(crafting_queue). They should match in practice but the parser
        does not enforce this."""
        wq = self._parse_player({
            "crafting_queue": [{"recipe": "iron-gear-wheel", "count": 5, "progress": 0.0}],
            "crafting_queue_size": 7,   # intentionally mismatched for this test
        })
        self.assertEqual(len(wq.crafting_queue), 1)
        self.assertEqual(wq.crafting_queue_size, 7)

    def test_other_player_fields_unaffected(self):
        wq = self._parse_player({
            "inventory": [{"item": "iron-ore", "count": 10}],
            "charted_chunks": 5,
            "crafting_queue": [{"recipe": "iron-gear-wheel", "count": 5, "progress": 0.3}],
            "crafting_queue_size": 1,
        })
        self.assertEqual(wq.inventory_count("iron-ore"), 10)
        self.assertEqual(wq.charted_chunks, 5)
        self.assertEqual(len(wq.crafting_queue), 1)


class TestStateParserCraftingQueueRoundtrip(unittest.TestCase):
    """
    End-to-end: a realistic bridge payload with crafting queue data flows
    correctly through StateParser -> WorldState -> WorldQuery.
    """

    def setUp(self):
        self.parser = StateParser()

    def _realistic_state(self, queue=None, queue_size=0) -> str:
        return json.dumps({
            "tick": 1200,
            "player": {
                "position": {"x": 16.0, "y": 16.0},
                "health": 100.0,
                "inventory": [
                    {"item": "iron-plate", "count": 100},
                    {"item": "copper-plate", "count": 50},
                ],
                "reachable": [],
                "charted_chunks": 10,
                "newly_charted_chunks": [],
                "nearby_uncharted_chunks": [],
                "movement_status": "idle",
                "inventory_size": 80,
                "crafting_queue": queue or [],
                "crafting_queue_size": queue_size,
            },
            "entities": [],
            "resource_map": [],
            "research": {"unlocked": [], "queued": []},
        })

    def test_idle_player_has_empty_queue(self):
        wq = WorldQuery(self.parser.parse(
            self._realistic_state(), current_tick=1200
        ))
        self.assertEqual(wq.crafting_queue, [])
        self.assertEqual(wq.crafting_queue_size, 0)

    def test_active_queue_roundtrip(self):
        queue = [
            {"recipe": "iron-gear-wheel", "count": 10, "progress": 0.6},
            {"recipe": "electronic-circuit", "count": 5, "progress": 0.0},
        ]
        wq = WorldQuery(self.parser.parse(
            self._realistic_state(queue=queue, queue_size=2), current_tick=1200
        ))
        self.assertEqual(len(wq.crafting_queue), 2)
        self.assertEqual(wq.crafting_queue_size, 2)
        self.assertEqual(wq.crafting_queue[0].recipe, "iron-gear-wheel")
        self.assertEqual(wq.crafting_queue[0].count, 10)
        self.assertAlmostEqual(wq.crafting_queue[0].progress, 0.6)
        self.assertEqual(wq.crafting_queue[1].recipe, "electronic-circuit")
        self.assertAlmostEqual(wq.crafting_queue[1].progress, 0.0)

    def test_first_entry_has_progress_others_zero(self):
        queue = [
            {"recipe": "iron-gear-wheel", "count": 5, "progress": 0.8},
            {"recipe": "copper-cable",    "count": 10, "progress": 0.0},
            {"recipe": "iron-chest",      "count": 2,  "progress": 0.0},
        ]
        wq = WorldQuery(self.parser.parse(
            self._realistic_state(queue=queue, queue_size=3), current_tick=1200
        ))
        self.assertGreater(wq.crafting_queue[0].progress, 0.0)
        self.assertAlmostEqual(wq.crafting_queue[1].progress, 0.0)
        self.assertAlmostEqual(wq.crafting_queue[2].progress, 0.0)

    def test_inventory_and_queue_coexist(self):
        queue = [{"recipe": "iron-gear-wheel", "count": 5, "progress": 0.0}]
        wq = WorldQuery(self.parser.parse(
            self._realistic_state(queue=queue, queue_size=1), current_tick=1200
        ))
        self.assertEqual(wq.inventory_count("iron-plate"), 100)
        self.assertEqual(wq.inventory_count("copper-plate"), 50)
        self.assertEqual(len(wq.crafting_queue), 1)

    def test_state_player_crafting_queue_accessible_directly(self):
        """White-box: verify crafting_queue is on state.player, not just wq."""
        from world.observable.state import WorldState
        queue = [{"recipe": "coal", "count": 1, "progress": 0.1}]
        state = self.parser.parse(
            self._realistic_state(queue=queue, queue_size=1), current_tick=1200
        )
        self.assertEqual(len(state.player.crafting_queue), 1)
        self.assertEqual(state.player.crafting_queue[0].recipe, "coal")
        self.assertAlmostEqual(state.player.crafting_queue[0].progress, 0.1)

class TestStateParserCharterChunkCoords(unittest.TestCase):
    """
    charted_chunk_coords is accumulated by WorldWriter.integrate_snapshot()
    from newly_charted_chunks deltas, not populated by StateParser directly.
    """

    def setUp(self):
        self.parser = StateParser()

    def test_fresh_parse_charted_chunk_coords_empty(self):
        """StateParser produces ExplorationState with empty charted_chunk_coords."""
        raw = json.dumps({"tick": 100, "player": {
            "position": {"x": 0, "y": 0}, "inventory": [],
            "newly_charted_chunks": [{"cx": 1, "cy": 2}],
        }})
        state = self.parser.parse(raw, current_tick=100)
        # Parser does NOT accumulate into charted_chunk_coords — WorldWriter does
        self.assertEqual(state.player.exploration.charted_chunk_coords, set())

    def test_writer_accumulates_charted_coords_from_newly_charted(self):
        """WorldWriter.integrate_snapshot accumulates newly_charted into coords."""
        from world.observable.state import WorldState
        from world.observable.writer import WorldWriter

        live = WorldState(tick=0)
        snap_raw = json.dumps({"tick": 100, "player": {
            "position": {"x": 0, "y": 0}, "inventory": [],
            "newly_charted_chunks": [{"cx": 3, "cy": 1}, {"cx": 4, "cy": 2}],
        }})
        snap = self.parser.parse(snap_raw, current_tick=100)
        WorldWriter(live).integrate_snapshot(snap)
        self.assertIn((3, 1), live.player.exploration.charted_chunk_coords)
        self.assertIn((4, 2), live.player.exploration.charted_chunk_coords)

    def test_writer_accumulates_across_multiple_snapshots(self):
        """Successive integrate_snapshot calls union-merge charted_chunk_coords."""
        from world.observable.state import WorldState
        from world.observable.writer import WorldWriter

        live = WorldState(tick=0)
        ww = WorldWriter(live)

        snap1 = self.parser.parse(json.dumps({"tick": 100, "player": {
            "position": {"x": 0, "y": 0}, "inventory": [],
            "newly_charted_chunks": [{"cx": 1, "cy": 0}],
        }}), current_tick=100)
        ww.integrate_snapshot(snap1)

        snap2 = self.parser.parse(json.dumps({"tick": 101, "player": {
            "position": {"x": 0, "y": 0}, "inventory": [],
            "newly_charted_chunks": [{"cx": 2, "cy": 0}],
        }}), current_tick=101)
        ww.integrate_snapshot(snap2)

        coords = live.player.exploration.charted_chunk_coords
        self.assertIn((1, 0), coords)
        self.assertIn((2, 0), coords)

    def test_writer_does_not_lose_old_coords_on_player_replace(self):
        """Prior accumulated coords are preserved when player section is replaced."""
        from world.observable.state import WorldState
        from world.observable.writer import WorldWriter

        live = WorldState(tick=0)
        ww = WorldWriter(live)

        # First snap adds (0,0)
        snap1 = self.parser.parse(json.dumps({"tick": 100, "player": {
            "position": {"x": 0, "y": 0}, "inventory": [],
            "newly_charted_chunks": [{"cx": 0, "cy": 0}],
        }}), current_tick=100)
        ww.integrate_snapshot(snap1)
        self.assertIn((0, 0), live.player.exploration.charted_chunk_coords)

        # Second snap (player replaced) — old coords must survive
        snap2 = self.parser.parse(json.dumps({"tick": 101, "player": {
            "position": {"x": 5, "y": 5}, "inventory": [],
            "newly_charted_chunks": [{"cx": 1, "cy": 0}],
        }}), current_tick=101)
        ww.integrate_snapshot(snap2)
        # Both old and new coords present
        self.assertIn((0, 0), live.player.exploration.charted_chunk_coords)
        self.assertIn((1, 0), live.player.exploration.charted_chunk_coords)

    def test_chunk_map_query_via_wq_reflects_accumulated_coords(self):
        """wq.chunk_map shows the accumulated charted_chunk_coords."""
        from world.observable.state import WorldState
        from world.observable.writer import WorldWriter

        live = WorldState(tick=0)
        ww = WorldWriter(live)
        snap = self.parser.parse(json.dumps({"tick": 100, "player": {
            "position": {"x": 0, "y": 0}, "inventory": [],
            "newly_charted_chunks": [{"cx": 5, "cy": 3}],
        }}), current_tick=100)
        ww.integrate_snapshot(snap)

        wq = WorldQuery(live)
        self.assertTrue(wq.chunk_map.is_charted(5, 3))
        self.assertFalse(wq.chunk_map.is_charted(0, 0))


class TestStateParserTileMapIntegration(unittest.TestCase):
    """
    Integration tests: tile_map flows through StateParser → WorldState →
    WorldWriter accumulation → WorldQuery accessors.
    """

    def setUp(self):
        self.parser = StateParser()

    def _state_with_tiles(self, entries):
        raw = json.dumps({"tick": 100, "tile_map": entries})
        return self.parser.parse(raw, current_tick=100)

    def test_tile_map_accessible_via_wq(self):
        state = self._state_with_tiles([{"x": 3, "y": 5, "tile": "water"}])
        wq = WorldQuery(state)
        self.assertEqual(wq.tile_at(3, 5), "water")

    def test_is_water_via_wq(self):
        state = self._state_with_tiles([{"x": 0, "y": 0, "tile": "deepwater"}])
        wq = WorldQuery(state)
        self.assertTrue(wq.is_water(0, 0))

    def test_is_buildable_via_wq(self):
        state = self._state_with_tiles([{"x": 1, "y": 1, "tile": "water"}])
        wq = WorldQuery(state)
        self.assertFalse(wq.is_buildable(1, 1))
        self.assertTrue(wq.is_buildable(0, 0))  # unobserved → buildable

    def test_tile_map_writer_accumulation(self):
        """Successive snapshots accumulate tile_map (union merge)."""
        from world.observable.state import WorldState
        from world.observable.writer import WorldWriter
        live = WorldState(tick=0)
        snap1 = self._state_with_tiles([{"x": 0, "y": 0, "tile": "water"}])
        snap2 = self._state_with_tiles([{"x": 1, "y": 0, "tile": "deepwater"}])
        ww = WorldWriter(live)
        ww.integrate_snapshot(snap1)
        ww.integrate_snapshot(snap2)
        self.assertIn((0, 0), live.tile_map)
        self.assertIn((1, 0), live.tile_map)

    def test_tile_map_section_staleness(self):
        state = self._state_with_tiles([])
        wq = WorldQuery(state)
        # Observed at tick 100; queried at tick 100 → staleness = 0
        self.assertEqual(wq.section_staleness("tile_map", 100), 0)
        # Queried at tick 200 → staleness = 100
        self.assertEqual(wq.section_staleness("tile_map", 200), 100)
        # Section that was never stamped → None
        self.assertIsNone(wq.section_staleness("logistics", 100))


if __name__ == "__main__":
    unittest.main(verbosity=2)