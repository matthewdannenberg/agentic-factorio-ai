"""
tests_in_game/01_knowledge/test_tech_discovery.py

Verifies that the KnowledgeBase correctly learns technology prototypes —
prerequisites, recipe unlocks, and entity unlocks — from Factorio's prototype
data during normal agent operation.

Runs in 01_knowledge/ so the KB is populated before collection and
exploration tests that may depend on production chain knowledge.
"""

import pytest
from llm.goal_source import GoalQueueEntry


_WARM_KB_GOAL = GoalQueueEntry(
    description="Warm KB — explore briefly",
    success_condition="charted_chunks >= 2",
    failure_condition="tick > 3600",
    goal_type="exploration",
)


def test_kb_learns_automation_tech(run_goal, knowledge_base):
    """
    'automation' is the first research technology in every vanilla Factorio
    game. The KB should learn it during the first state poll that queries
    the research prototype table.
    """
    run_goal(_WARM_KB_GOAL)
    tech = knowledge_base.get_tech("automation")
    assert tech is not None, (
        "KB did not learn the 'automation' technology. "
        "Check that StateParser populates KnowledgeBase.ensure_tech() "
        "when the research state is polled."
    )


def test_automation_tech_not_placeholder(run_goal, knowledge_base):
    """
    A properly learned tech should not be a placeholder — it should have
    the full prototype data including unlocks and prerequisites.
    """
    run_goal(_WARM_KB_GOAL)
    tech = knowledge_base.get_tech("automation")
    if tech is None:
        pytest.skip("automation tech not yet in KB")
    assert not tech.is_placeholder, (
        "automation tech is still a placeholder — "
        "KB learned the name but not the prototype data"
    )


def test_automation_unlocks_assembling_machine(run_goal, knowledge_base):
    """
    The 'automation' tech unlocks the assembling-machine-1 recipe.
    Verifying the unlock list confirms that recipe-unlock effects are
    parsed and stored correctly.
    """
    run_goal(_WARM_KB_GOAL)
    tech = knowledge_base.get_tech("automation")
    if tech is None or tech.is_placeholder:
        pytest.skip("automation tech not fully learned yet")
    assert "assembling-machine-1" in tech.unlocks_recipes, (
        f"automation.unlocks_recipes = {tech.unlocks_recipes!r}; "
        "expected 'assembling-machine-1'"
    )


def test_logistics_tech_has_automation_prerequisite(run_goal, knowledge_base):
    """
    'logistics' requires 'automation' as a prerequisite.
    Verifies that prerequisite edges are stored correctly.
    """
    run_goal(_WARM_KB_GOAL)
    tech = knowledge_base.get_tech("logistics")
    if tech is None or tech.is_placeholder:
        pytest.skip("logistics tech not fully learned yet")
    assert "automation" in tech.prerequisites, (
        f"logistics.prerequisites = {tech.prerequisites!r}; "
        "expected 'automation'"
    )


def test_techs_unlocking_recipe_roundtrip(run_goal, knowledge_base):
    """
    KnowledgeBase.techs_unlocking_recipe() should return the tech that
    unlocks a given recipe. 'automation' unlocks 'assembling-machine-1',
    so techs_unlocking_recipe('assembling-machine-1') should include it.
    """
    run_goal(_WARM_KB_GOAL)
    tech = knowledge_base.get_tech("automation")
    if tech is None or tech.is_placeholder:
        pytest.skip("automation tech not fully learned yet")

    techs = knowledge_base.techs_unlocking_recipe("assembling-machine-1")
    tech_names = [t.name for t in techs]
    assert "automation" in tech_names, (
        f"techs_unlocking_recipe('assembling-machine-1') = {tech_names!r}; "
        "expected 'automation'"
    )
