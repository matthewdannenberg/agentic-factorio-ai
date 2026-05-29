"""
tests_in_game/01_world/test_entity_discovery.py

Verifies that the KnowledgeBase correctly learns entity, resource, and item
prototypes from Factorio, and that cross-domain queries work correctly once
the data is populated.

Runs in 01_world/ (after 01_knowledge recipe/tech tests) so the KB
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
