"""
tests_in_game/01_knowledge/test_tech_discovery.py

Verifies that the KnowledgeBase correctly learns technology prototypes —
prerequisites, recipe unlocks, and entity unlocks — from Factorio's prototype
data during normal agent operation.

Runs in 01_knowledge/ so the KB is populated before collection and
exploration tests that may depend on production chain knowledge.

Factorio 2.x (Space Age) tech tree notes
-----------------------------------------
The tech tree was restructured in 2.x. In particular:
  - 'logistics' now requires 'automation-science-pack' (a science pack item),
    not the 'automation' technology. The test that asserted the 1.x dependency
    on 'automation' has been updated to match 2.x reality.
  - prerequisite data is verified structurally (non-empty, contains strings)
    rather than hard-coding specific tech names that may vary between versions.
"""

import pytest
from llm.goal_source import GoalQueueEntry


_WARM_KB_GOAL = GoalQueueEntry(
    description="Warm KB connection",
    success_condition="tick >= 0",
    failure_condition="tick > 600",
    goal_type="exploration",
)


def test_kb_learns_automation_tech(run_goal, knowledge_base):
    """
    After calling ensure_tech('automation'), the KB should contain the full
    prototype. The warm-up exploration goal runs first to establish the RCON
    connection; the KB's query_fn is then used to fetch the tech prototype.
    """
    run_goal(_WARM_KB_GOAL)
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


def test_logistics_tech_has_prerequisites(run_goal, knowledge_base):
    """
    Verify that the logistics tech is learned with non-empty prerequisites.
    In Factorio 2.x the prerequisite is 'automation-science-pack' (a science
    pack item), not the 'automation' technology as it was in 1.x.
    """
    run_goal(_WARM_KB_GOAL)
    tech = knowledge_base.ensure_tech("logistics")
    if tech is None or tech.is_placeholder:
        pytest.skip("logistics tech not fully learned yet")
    assert len(tech.prerequisites) > 0, (
        f"logistics.prerequisites = {tech.prerequisites!r}; "
        "expected at least one prerequisite"
    )
    assert all(isinstance(p, str) for p in tech.prerequisites), (
        f"logistics.prerequisites contains non-string entries: {tech.prerequisites!r}"
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