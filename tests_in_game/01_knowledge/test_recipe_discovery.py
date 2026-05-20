"""
tests_in_game/01_knowledge/test_recipe_discovery.py

Verifies that the KnowledgeBase correctly learns recipe prototypes from
Factorio when queried during normal agent operation.

These tests run first (01_) so the KB is populated before collection and
exploration tests run. They also validate the KB independently so a failure
here signals a KB/bridge problem rather than an agent problem.
"""

import pytest
from llm.goal_source import GoalQueueEntry


# A short exploration goal is enough to trigger several state polls and
# give the KB a chance to learn recipes from the observed entities.
_WARM_KB_GOAL = GoalQueueEntry(
    description="Warm KB — explore briefly",
    success_condition="charted_chunks >= 2",
    failure_condition="tick > 3600",
    goal_type="exploration",
)


def test_kb_learns_iron_plate_recipe(run_goal, knowledge_base):
    """
    After a state poll, the KB should be able to look up the iron-plate recipe.
    iron-plate is a core recipe present in every default Factorio game.
    """
    run_goal(_WARM_KB_GOAL)
    recipe = knowledge_base.get_recipe("iron-plate")
    assert recipe is not None, (
        "KB did not learn the iron-plate recipe after polling game state. "
        "Check that StateParser populates KnowledgeBase on each snapshot."
    )


def test_kb_learns_iron_gear_wheel_recipe(run_goal, knowledge_base):
    """
    iron-gear-wheel requires iron-plate as an ingredient. Verifying its
    recipe is in the KB confirms multi-ingredient recipes are stored correctly.
    """
    run_goal(_WARM_KB_GOAL)
    recipe = knowledge_base.get_recipe("iron-gear-wheel")
    assert recipe is not None
    ingredient_names = [i.name for i in (recipe.ingredients or [])]
    assert "iron-plate" in ingredient_names, (
        f"iron-gear-wheel recipe missing iron-plate ingredient; got {ingredient_names}"
    )


def test_kb_recipe_has_expected_fields(run_goal, knowledge_base):
    """
    A learned recipe should have non-empty ingredients and products lists
    and a non-empty category string.
    """
    run_goal(_WARM_KB_GOAL)
    recipe = knowledge_base.get_recipe("electronic-circuit")
    if recipe is None:
        pytest.skip("electronic-circuit not yet in KB — run after more polling")

    assert recipe.category, "recipe.category should not be empty"
    assert recipe.ingredients, "recipe.ingredients should not be empty"
    assert recipe.products, "recipe.products should not be empty"
    assert not recipe.is_placeholder, (
        "recipe should not be a placeholder after KB has learned it"
    )
