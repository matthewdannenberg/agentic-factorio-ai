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
    After calling ensure_tech('automation'), the KB should contain the full
    prototype. The warm-up exploration goal runs first to establish the RCON
    connection; the KB's query_fn is then used to fetch the tech prototype.
    """
    run_goal(_WARM_KB_GOAL)
    # Explicitly fetch the tech — the KB won't have it unless we ask for it,
    # since StateParser doesn't call ensure_tech during snapshot parsing.
    tech = knowledge_base.ensure_tech("automation")
    assert tech is not None, (
        "KB.ensure_tech('automation') returned None. "
        "Check that kb._query_fn is set and fa.get_technology is reachable."
    )


def test_automation_tech_not_placeholder(run_goal, knowledge_base):
    run_goal(_WARM_KB_GOAL)
    tech = knowledge_base.ensure_tech("automation")
    assert not tech.is_placeholder, (
        "automation tech is still a placeholder — "
        "ensure_tech queried Factorio but parsing failed"
    )


def test_automation_unlocks_assembling_machine(run_goal, knowledge_base):
    run_goal(_WARM_KB_GOAL)
    tech = knowledge_base.ensure_tech("automation")
    if tech is None or tech.is_placeholder:
        pytest.skip("automation tech not fully learned yet")
    assert "assembling-machine-1" in tech.unlocks_recipes, (
        f"automation.unlocks_recipes = {tech.unlocks_recipes!r}; "
        "expected 'assembling-machine-1'"
    )


def test_logistics_tech_has_automation_prerequisite(run_goal, knowledge_base):
    run_goal(_WARM_KB_GOAL)
    knowledge_base.ensure_tech("automation")  # fetch prerequisite first
    tech = knowledge_base.ensure_tech("logistics")
    if tech is None or tech.is_placeholder:
        pytest.skip("logistics tech not fully learned yet")
    assert "automation" in tech.prerequisites, (
        f"logistics.prerequisites = {tech.prerequisites!r}; "
        "expected 'automation'"
    )


def test_techs_unlocking_recipe_roundtrip(run_goal, knowledge_base):
    run_goal(_WARM_KB_GOAL)
    knowledge_base.ensure_tech("automation")
    tech = knowledge_base.get_tech("automation")
    if tech is None or tech.is_placeholder:
        pytest.skip("automation tech not fully learned yet")

    techs = knowledge_base.techs_unlocking_recipe("assembling-machine-1")
    tech_names = [t.name for t in techs]
    assert "automation" in tech_names, (
        f"techs_unlocking_recipe('assembling-machine-1') = {tech_names!r}; "
        "expected 'automation'"
    )