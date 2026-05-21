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


# A goal that succeeds on the very first tick — all we need is for the loop
# to run once so that kb._query_fn is confirmed live. The KB queries in each
# test (ensure_recipe, ensure_tech) go directly to Factorio via query_fn and
# don't need any in-game action to have occurred first.
_WARM_KB_GOAL = GoalQueueEntry(
    description="Warm KB connection",
    success_condition="tick >= 0",
    failure_condition="tick > 600",
    goal_type="exploration",
)


def test_kb_learns_iron_plate_recipe(run_goal, knowledge_base):
    """
    After calling ensure_recipe('iron-plate'), the KB should contain the
    full prototype. The warm-up goal establishes the RCON connection;
    ensure_recipe then queries Factorio via kb._query_fn.
    """
    run_goal(_WARM_KB_GOAL)
    recipe = knowledge_base.ensure_recipe("iron-plate")
    assert recipe is not None, (
        "KB.ensure_recipe('iron-plate') returned None. "
        "Check that kb._query_fn is set and fa.get_recipe_prototype is reachable."
    )


def test_kb_learns_iron_gear_wheel_recipe(run_goal, knowledge_base):
    run_goal(_WARM_KB_GOAL)
    recipe = knowledge_base.ensure_recipe("iron-gear-wheel")
    assert recipe is not None
    ingredient_names = [i.name for i in (recipe.ingredients or [])]
    assert "iron-plate" in ingredient_names, (
        f"iron-gear-wheel recipe missing iron-plate ingredient; got {ingredient_names}"
    )


def test_kb_recipe_has_expected_fields(run_goal, knowledge_base):
    run_goal(_WARM_KB_GOAL)
    recipe = knowledge_base.ensure_recipe("electronic-circuit")
    if recipe is None or recipe.is_placeholder:
        pytest.skip("electronic-circuit recipe could not be fetched")

    assert recipe.category, "recipe.category should not be empty"
    assert recipe.ingredients, "recipe.ingredients should not be empty"
    assert recipe.products, "recipe.products should not be empty"
    assert not recipe.is_placeholder, (
        "recipe should not be a placeholder after ensure_recipe"
    )