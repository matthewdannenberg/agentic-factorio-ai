"""
tests_in_game/01_knowledge/test_entity_discovery.py

Verifies that the KnowledgeBase correctly learns entity, resource, and item
prototypes from Factorio, and that cross-domain queries work correctly once
the data is populated.

Runs in 01_knowledge/ (after 01_knowledge recipe/tech tests) so the KB
connection is already warm. Each test calls ensure_*() directly — no
in-game actions are required beyond the warm-up goal that establishes the
RCON link.

What is tested
--------------
  Entities:
    - ensure_entity returns a non-placeholder for known game entities
    - proto_type, category, tile dimensions, has_recipe_slot populated
    - minable field correctly reflects prototype data
    - Placeholder returned for truly unknown names (no crash)

  Resources:
    - ensure_resource returns a non-placeholder for standard ore types
    - is_fluid / is_infinite fields correctly populated (False for solid ores)
    - display_name populated

  Items:
    - ensure_item returns a non-placeholder for standard items
    - stack_size is a sensible positive integer (not the placeholder default of 1
      for high-stack-size items like iron-plate)

  Cross-domain queries:
    - recipes_for_product finds recipes that produce the item
    - recipes_for_ingredient finds recipes that consume the item
    - recipes_made_in returns recipes for a given machine type
    - production_chain returns transitive ingredient closure
"""

import pytest
from planning import GoalQueueEntry


# ---------------------------------------------------------------------------
# Warm-up goal — establishes RCON link before direct KB queries
# ---------------------------------------------------------------------------

_WARM_KB_GOAL = GoalQueueEntry(
    description="Warm KB connection (entity tests)",
    goal_type="exploration",
    success_condition="charted_chunks >= 1",
    failure_condition="elapsed_ticks > 600",
)


# ---------------------------------------------------------------------------
# Entity prototype tests
# ---------------------------------------------------------------------------

class TestEntityDiscovery:
    """KnowledgeBase entity prototype learning."""

    def test_assembler_not_placeholder(self, run_goal, knowledge_base):
        """assembling-machine-1 is the most fundamental assembly entity."""
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_entity("assembling-machine-1")
        assert record is not None
        assert not record.is_placeholder, (
            "assembling-machine-1 is still a placeholder after ensure_entity. "
            "Check fa.get_entity_prototype and bridge/prototype_query.py."
        )

    def test_assembler_proto_type(self, run_goal, knowledge_base):
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_entity("assembling-machine-1")
        if record.is_placeholder:
            pytest.skip("assembling-machine-1 not learned")
        assert record.proto_type == "assembling-machine", (
            f"Expected proto_type='assembling-machine', got {record.proto_type!r}"
        )

    def test_assembler_category(self, run_goal, knowledge_base):
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_entity("assembling-machine-1")
        if record.is_placeholder:
            pytest.skip("assembling-machine-1 not learned")
        assert record.category == "assembly", (
            f"Expected category='assembly', got {record.category!r}"
        )

    def test_assembler_has_recipe_slot(self, run_goal, knowledge_base):
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_entity("assembling-machine-1")
        if record.is_placeholder:
            pytest.skip("assembling-machine-1 not learned")
        assert record.has_recipe_slot is True, (
            "assembling-machine-1 should have a recipe slot"
        )

    def test_assembler_tile_dimensions(self, run_goal, knowledge_base):
        """assembling-machine-1 occupies a 3×3 tile footprint."""
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_entity("assembling-machine-1")
        if record.is_placeholder:
            pytest.skip("assembling-machine-1 not learned")
        assert record.tile_width == 3 and record.tile_height == 3, (
            f"Expected 3×3 tile dimensions, got "
            f"{record.tile_width}×{record.tile_height}"
        )

    def test_stone_furnace_category(self, run_goal, knowledge_base):
        """stone-furnace has category 'smelting'."""
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_entity("stone-furnace")
        if record.is_placeholder:
            pytest.skip("stone-furnace not learned")
        assert record.category == "smelting", (
            f"Expected category='smelting', got {record.category!r}"
        )

    def test_chest_category(self, run_goal, knowledge_base):
        """wooden-chest has category 'storage'."""
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_entity("wooden-chest")
        if record.is_placeholder:
            pytest.skip("wooden-chest not learned")
        assert record.category == "storage", (
            f"Expected category='storage', got {record.category!r}"
        )

    def test_entity_is_minable(self, run_goal, knowledge_base):
        """
        Machines and chests should be minable (player can pick them up).
        minable=True means MineEntity can be used on them.
        """
        run_goal(_WARM_KB_GOAL)
        for entity_name in ("stone-furnace", "wooden-chest"):
            record = knowledge_base.ensure_entity(entity_name)
            if record.is_placeholder:
                continue
            assert record.minable is True, (
                f"{entity_name}: expected minable=True (can be picked up)"
            )

    def test_unknown_entity_gives_placeholder(self, run_goal, knowledge_base):
        """
        ensure_entity on a nonexistent name must not raise; it must return a
        placeholder rather than crashing. This validates the safe-fallback path.
        """
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_entity(
            "absolutely-not-a-real-factorio-entity-xyz-12345"
        )
        assert record is not None, "ensure_entity must never return None"
        assert record.is_placeholder is True, (
            "Unknown entity names should produce a placeholder, not a real record"
        )

    def test_entity_learned_by_warmed_kb(self, run_goal, knowledge_base):
        """
        The conftest warm-up loop calls kb.ensure_entity for every entity in the
        final world scan. After a warm-up exploration run, the KB should contain
        at least one real (non-placeholder) entity.
        """
        run_goal(_WARM_KB_GOAL)
        all_entities = knowledge_base.all_entities()
        real_entities = [e for e in all_entities.values() if not e.is_placeholder]
        assert len(real_entities) >= 1, (
            f"No real entities in KB after warm-up run. "
            f"all_entities count={len(all_entities)}. "
            f"Check conftest KB warm-up loop and bridge entity scanning."
        )


# ---------------------------------------------------------------------------
# Resource prototype tests
# ---------------------------------------------------------------------------

class TestResourceDiscovery:
    """KnowledgeBase resource prototype learning."""

    def test_iron_ore_not_placeholder(self, run_goal, knowledge_base):
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_resource("iron-ore")
        assert record is not None
        assert not record.is_placeholder, (
            "iron-ore is still a placeholder. "
            "Check fa.get_resource_prototype and bridge/prototype_query.py."
        )

    def test_iron_ore_is_not_fluid(self, run_goal, knowledge_base):
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_resource("iron-ore")
        if record.is_placeholder:
            pytest.skip("iron-ore not learned")
        assert record.is_fluid is False, (
            f"iron-ore should not be a fluid resource; got is_fluid={record.is_fluid}"
        )

    def test_iron_ore_is_not_infinite(self, run_goal, knowledge_base):
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_resource("iron-ore")
        if record.is_placeholder:
            pytest.skip("iron-ore not learned")
        assert record.is_infinite is False, (
            "iron-ore is a finite resource; expected is_infinite=False"
        )

    def test_iron_ore_has_display_name(self, run_goal, knowledge_base):
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_resource("iron-ore")
        if record.is_placeholder:
            pytest.skip("iron-ore not learned")
        assert record.display_name, (
            "iron-ore display_name should not be empty"
        )

    def test_coal_not_placeholder(self, run_goal, knowledge_base):
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_resource("coal")
        assert record is not None
        assert not record.is_placeholder, "coal resource should not be a placeholder"

    def test_unknown_resource_gives_placeholder(self, run_goal, knowledge_base):
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_resource("not-a-real-resource-xyz")
        assert record is not None
        assert record.is_placeholder is True


# ---------------------------------------------------------------------------
# Item prototype tests
# ---------------------------------------------------------------------------

class TestItemDiscovery:
    """KnowledgeBase item prototype learning (stack sizes, etc.)."""

    def test_iron_plate_not_placeholder(self, run_goal, knowledge_base):
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_item("iron-plate")
        assert record is not None
        assert not record.is_placeholder, (
            "iron-plate item is still a placeholder. "
            "Check fa.get_item_prototype and the 'item' domain in prototype_query."
        )

    def test_iron_plate_stack_size(self, run_goal, knowledge_base):
        """iron-plate has stack_size=100 in vanilla Factorio."""
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_item("iron-plate")
        if record.is_placeholder:
            pytest.skip("iron-plate item not learned")
        assert record.stack_size > 1, (
            f"iron-plate stack_size should be > 1 (vanilla=100); "
            f"got {record.stack_size}. May indicate placeholder fallback."
        )

    def test_item_stack_size_method(self, run_goal, knowledge_base):
        """
        KnowledgeBase.item_stack_size() is the canonical accessor.
        It calls ensure_item internally and returns a positive int.
        """
        run_goal(_WARM_KB_GOAL)
        size = knowledge_base.item_stack_size("iron-plate")
        assert isinstance(size, int) and size >= 1, (
            f"item_stack_size('iron-plate') returned {size!r}; expected int >= 1"
        )

    def test_unknown_item_gives_placeholder(self, run_goal, knowledge_base):
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_item("not-a-real-item-xyz-99999")
        assert record is not None
        assert record.is_placeholder is True

    def test_unknown_item_stack_size_returns_one(self, run_goal, knowledge_base):
        """
        The placeholder stack_size is 1 — the conservative safe default
        for slot-count arithmetic. This is contractual: code that calls
        item_stack_size() for unknown items must get 1, not 0 or an error.
        """
        run_goal(_WARM_KB_GOAL)
        size = knowledge_base.item_stack_size("not-a-real-item-xyz-99999")
        assert size == 1, (
            f"Unknown item stack size should be 1 (placeholder); got {size}"
        )


# ---------------------------------------------------------------------------
# Cross-domain query tests
# ---------------------------------------------------------------------------

class TestCrossDomainQueries:
    """
    Tests for multi-table queries that depend on recipes being populated.
    These call ensure_recipe first to guarantee the data is in the DB.
    """

    def test_recipes_for_product_finds_iron_plate(self, run_goal, knowledge_base):
        """
        After learning the iron-plate recipe, recipes_for_product should find it.
        """
        run_goal(_WARM_KB_GOAL)
        knowledge_base.ensure_recipe("iron-plate")
        recipes = knowledge_base.recipes_for_product("iron-plate")
        recipe_names = [r.name for r in recipes]
        assert "iron-plate" in recipe_names, (
            f"recipes_for_product('iron-plate') returned {recipe_names!r}; "
            "expected 'iron-plate' to appear."
        )

    def test_recipes_for_ingredient_iron_plate(self, run_goal, knowledge_base):
        """
        iron-plate is an ingredient of iron-gear-wheel. After learning both
        recipes, recipes_for_ingredient('iron-plate') should include
        iron-gear-wheel.
        """
        run_goal(_WARM_KB_GOAL)
        knowledge_base.ensure_recipe("iron-plate")
        knowledge_base.ensure_recipe("iron-gear-wheel")
        recipes = knowledge_base.recipes_for_ingredient("iron-plate")
        recipe_names = [r.name for r in recipes]
        assert "iron-gear-wheel" in recipe_names, (
            f"recipes_for_ingredient('iron-plate') = {recipe_names!r}; "
            "expected 'iron-gear-wheel' since it uses iron-plate."
        )

    def test_recipes_made_in_assembler(self, run_goal, knowledge_base):
        """
        After learning iron-gear-wheel (made in assembling-machine), 
        recipes_made_in('assembling-machine-1') should include it.
        """
        run_goal(_WARM_KB_GOAL)
        knowledge_base.ensure_recipe("iron-gear-wheel")
        recipes = knowledge_base.recipes_made_in("assembling-machine-1")
        recipe_names = [r.name for r in recipes]
        assert len(recipes) >= 1, (
            f"recipes_made_in('assembling-machine-1') returned empty list. "
            f"Ensure 'iron-gear-wheel' recipe was learned with made_in populated."
        )
        assert "iron-gear-wheel" in recipe_names, (
            f"recipes_made_in returned {recipe_names!r}; "
            "expected 'iron-gear-wheel'."
        )

    def test_production_chain_electronic_circuit(self, run_goal, knowledge_base):
        """
        production_chain('electronic-circuit') should transitively include
        iron-plate and copper-cable (which itself needs copper-plate).
        This verifies the recursive CTE in the SQLite query.
        """
        run_goal(_WARM_KB_GOAL)
        knowledge_base.ensure_recipe("electronic-circuit")
        knowledge_base.ensure_recipe("copper-cable")
        chain = knowledge_base.production_chain("electronic-circuit")
        # chain is a set of ingredient names; electronic-circuit itself excluded.
        assert isinstance(chain, set), "production_chain should return a set"
        assert "iron-plate" in chain, (
            f"production_chain('electronic-circuit') = {chain!r}; "
            "expected 'iron-plate' to appear (direct ingredient)."
        )
        assert "copper-cable" in chain, (
            f"production_chain missing 'copper-cable': {chain!r}"
        )

    def test_production_chain_empty_for_unknown(self, run_goal, knowledge_base):
        """
        production_chain on a name with no known recipe returns an empty set,
        not an error.
        """
        run_goal(_WARM_KB_GOAL)
        chain = knowledge_base.production_chain("not-a-real-item-xyz")
        assert chain == set(), (
            f"Expected empty set for unknown item; got {chain!r}"
        )


# ---------------------------------------------------------------------------
# Harvest pipeline tests — mining_products and entities_that_produce
# ---------------------------------------------------------------------------

class TestMiningProducts:
    """
    Verifies that entity prototypes include the mining_products field
    introduced for the harvest_natural pipeline (Phase 7).

    mining_products populates kb.entities_that_produce(item), used by the
    coordinator to find harvestable sources for items like wood that are not
    resource patches. If any of these tests fail, harvest_natural goals STUCK.
    """

    def test_tree_entity_has_mining_products(self, run_goal, knowledge_base):
        """
        After ensure_entity on a tree, mining_products must contain 'wood'.
        This is the single most important prerequisite for harvest_natural goals.
        """
        run_goal(_WARM_KB_GOAL)
        tree_names = [
            "tree-01", "tree-02", "tree-04", "tree-06",
            "tree-01-red", "tree-02-red",
        ]
        learned = None
        for name in tree_names:
            record = knowledge_base.ensure_entity(name)
            if not record.is_placeholder:
                learned = record
                break
        if learned is None:
            pytest.skip(
                "No common tree entity names found in KB after ensure_entity. "
                "Map may use modded tree names."
            )
        assert learned.mining_products, (
            f"{learned.name}.mining_products is empty after ensure_entity. "
            "Check that control.lua fa.get_entity_prototype() returns "
            "mining_products from mineable_properties.products."
        )
        assert "wood" in learned.mining_products, (
            f"{learned.name}.mining_products = {learned.mining_products!r}; "
            "expected 'wood' to be present."
        )
        assert learned.mining_products["wood"] > 0, (
            f"{learned.name}.mining_products['wood'] must be positive."
        )

    def test_tree_minable_is_true(self, run_goal, knowledge_base):
        """
        Trees must have minable=True — this is the filter gate in
        entities_that_produce. Non-minable entities are excluded by the SQL.
        """
        run_goal(_WARM_KB_GOAL)
        tree_names = ["tree-01", "tree-02", "tree-04", "tree-06"]
        for name in tree_names:
            record = knowledge_base.ensure_entity(name)
            if record.is_placeholder:
                continue
            assert record.minable is True, (
                f"{name}.minable = {record.minable}; "
                "trees must be minable=True for entities_that_produce to find them."
            )
            return
        pytest.skip("No tree entities learned — map may use modded names")

    def test_entities_that_produce_wood_non_empty(self, run_goal, knowledge_base):
        """
        After the KB warm-up (which runs ensure_entity for all natural objects
        in scan radius), entities_that_produce('wood') must return at least one
        entity. This is the direct prerequisite for harvest_natural goals.
        If this fails, the coordinator will always STUCK on wood collection.
        """
        run_goal(_WARM_KB_GOAL)
        results = knowledge_base.entities_that_produce("wood")
        assert len(results) > 0, (
            "entities_that_produce('wood') returned [] after KB warm-up. "
            "Possible causes: no trees in scan radius during warm-up; "
            "control.lua not returning mining_products; "
            "entity_mining_products table missing (check _migrate()); "
            "or loop not passing kb= to FactorioLoop (check conftest.py)."
        )
        for rec in results:
            assert rec.minable is True, (
                f"entities_that_produce('wood') returned {rec.name} "
                f"with minable={rec.minable}; all results must be minable."
            )
            assert "wood" in rec.mining_products, (
                f"entities_that_produce('wood') returned {rec.name} but "
                f"mining_products has no 'wood' key: {rec.mining_products!r}"
            )

    def test_entities_that_produce_returns_list(self, run_goal, knowledge_base):
        """
        entities_that_produce must always return a list, never raise.
        Covers the case where the entity_mining_products table is missing
        (the query has a try/except that should return [] not raise).
        """
        run_goal(_WARM_KB_GOAL)
        result = knowledge_base.entities_that_produce("iron-ore")
        assert isinstance(result, list), (
            "entities_that_produce must return a list, not raise"
        )
        result2 = knowledge_base.entities_that_produce("not-a-real-item-xyz")
        assert isinstance(result2, list)
        assert result2 == []

    def test_non_harvestable_entity_no_wood_in_products(
        self, run_goal, knowledge_base
    ):
        """
        Entities that don't drop wood (e.g. steam-engine) must not appear
        in entities_that_produce('wood'). Verifies the DB join is correct.
        """
        run_goal(_WARM_KB_GOAL)
        record = knowledge_base.ensure_entity("steam-engine")
        if record.is_placeholder:
            pytest.skip("steam-engine not learned")
        assert "wood" not in record.mining_products, (
            f"steam-engine.mining_products contains 'wood': "
            f"{record.mining_products!r}"
        )

    def test_mining_products_populated_by_loop_kb_warmup(
        self, run_goals, knowledge_base
    ):
        """
        The loop's _poll_world KB warm-up (added to FactorioLoop when kb= is
        passed) calls ensure_entity for every natural_object in scan radius.
        After a warm-up goal run, entities_that_produce('wood') must be
        non-empty — this confirms the full pipeline from loop to KB to query.
        """
        run_goals([_WARM_KB_GOAL])
        results = knowledge_base.entities_that_produce("wood")
        assert len(results) > 0, (
            "KB warm-up ran but entities_that_produce('wood') is still empty. "
            "Check that FactorioLoop receives kb= in conftest _execute_goals, "
            "that _poll_world calls kb.ensure_entity for natural_objects, "
            "and that ensure_entity persists mining_products to the DB."
        )
    def test_entities_that_produce_wood_full_record_details(
        self, run_goals, knowledge_base
    ):
        """
        Forensic test: dumps the full KB contents for wood harvesting so we can
        see exactly what was learned. Always passes — purely diagnostic.
        """
        import sys
        run_goals([_WARM_KB_GOAL])

        results = knowledge_base.entities_that_produce("wood")
        sys.stderr.write(
            "\n=== entities_that_produce('wood') => {} results ===\n".format(
                len(results))
        )
        for rec in results:
            sys.stderr.write(
                "  name={!r} minable={} proto_type={!r} "
                "is_placeholder={} mining_products={!r}\n".format(
                    rec.name, rec.minable, rec.proto_type,
                    rec.is_placeholder, rec.mining_products)
            )

        try:
            rows = knowledge_base._conn.execute(
                "SELECT entity_name, item_name, amount "
                "FROM entity_mining_products "
                "ORDER BY entity_name, item_name"
            ).fetchall()
            sys.stderr.write(
                "\n=== entity_mining_products table ({} rows) ===\n".format(
                    len(rows))
            )
            for row in rows[:30]:
                sys.stderr.write(
                    "  {!r} -> {!r} x{}\n".format(row[0], row[1], row[2])
                )
        except Exception as exc:
            sys.stderr.write(
                "\n=== entity_mining_products ERROR: {} ===\n".format(exc)
            )

        try:
            minable_rows = knowledge_base._conn.execute(
                "SELECT name, proto_type, minable, is_placeholder "
                "FROM entities WHERE minable=1 AND is_placeholder=0 "
                "ORDER BY proto_type, name LIMIT 30"
            ).fetchall()
            sys.stderr.write(
                "\n=== minable non-placeholder entities ({} shown) ===\n".format(
                    len(minable_rows))
            )
            for row in minable_rows:
                sys.stderr.write(
                    "  {!r} type={!r}\n".format(row[0], row[1])
                )
        except Exception as exc:
            sys.stderr.write(
                "\n=== entities query ERROR: {} ===\n".format(exc)
            )

        tree_like = {
            name: rec
            for name, rec in knowledge_base._entities.items()
            if not rec.is_placeholder
            and rec.proto_type in ("tree", "simple-entity")
        }
        sys.stderr.write(
            "\n=== in-memory tree/simple-entity records ({}) ===\n".format(
                len(tree_like))
        )
        for name, rec in list(tree_like.items())[:20]:
            sys.stderr.write(
                "  {!r}: minable={} mining_products={!r}\n".format(
                    name, rec.minable, rec.mining_products)
            )

        try:
            sys.stderr.write(
                "\n=== KB summary: {} ===\n".format(knowledge_base.summary())
            )
        except Exception as exc:
            sys.stderr.write(
                "\n=== summary error: {} ===\n".format(exc)
            )

        assert True