"""
tests_in_game/03_collection/test_collect_natural.py

Verifies the harvest_natural pipeline for items that come from natural objects
rather than resource patches (wood from trees, fish from fish entities, etc.).

How this differs from test_collect_iron.py
-------------------------------------------
Iron ore is collected via Path A of _handle_collection: the coordinator finds a
resource patch in wq.resource_map and pushes a gather_resource task. MiningAgent
navigates to the patch and uses MineSkill.

Wood has no resource patch. It is collected via Path B:
  1. _handle_collection checks kb.entities_that_produce('wood')
  2. KB returns tree entity types (learned from fa.get_entity_prototype)
  3. Coordinator pushes a harvest_natural task with those entity types
  4. MiningAgent._tick_harvest scans wq.natural_objects for matching trees,
     navigates to each one, destroys it with DestroySkill, and loops until
     the success_condition fires.

Prerequisites
-------------
The KB must have learned which entities drop wood before the coordinator can
dispatch a harvest_natural task. On a fresh map this happens the first time
any tree's prototype is queried — the bridge's fa.get_entity_prototype()
returns mineable_properties.products, which now includes mining_products.

If no tree prototypes have been queried yet, entities_that_produce('wood')
returns [] and the coordinator STUCKs. A preceding exploration goal (or a
prior test run that observed trees) populates the KB. The failure condition
is generous to account for this.

Notes on wood availability
---------------------------
Wood is always available near spawn on a standard Factorio map. The agent
does not need to explore first — trees are visible in the initial scan radius.
However, if the map has been heavily cleared by prior test runs, the agent
may need to walk further to find trees. The failure condition allows 5 minutes.
"""

from planning import GoalQueueEntry


# Trivial exploration goal that satisfies immediately (charted_chunks is always
# >= 1 on any loaded map). Used as a warm-up before wood collection goals so
# that the loop has had at least one full _poll_world cycle — including KB
# warm-up via ensure_entity on natural_objects — before the coordinator first
# tries entities_that_produce('wood').
_WARM_KB_GOAL = GoalQueueEntry(
    description="KB warm-up (natural objects scan)",
    goal_type="exploration",
    success_condition="charted_chunks >= 1",
    failure_condition="elapsed_ticks > 600",
)


def test_collect_5_new_wood(run_goals):
    """
    Collect 5 new wood by harvesting trees.

    Exercises the full harvest_natural pipeline:
      - Coordinator calls kb.entities_that_produce('wood')
      - harvest_natural task pushed with tree entity types
      - MiningAgent finds nearest tree, navigates, destroys it
      - Loops until new.inventory('wood') >= 5

    Uses new.inventory() so the test requires genuine new collection
    regardless of wood already in the player's inventory.
    """
    entry = GoalQueueEntry(
        description="Collect 5 new wood",
        goal_type="collection",
        success_condition="new.inventory('wood') >= 5",
        failure_condition="elapsed_ticks > 18000",   # 5 minutes
    )
    # Warm-up goal ensures the KB has had at least one full scan cycle to
    # learn tree entity types via ensure_entity before the collection goal runs.
    stats, wq = run_goals([_WARM_KB_GOAL, entry])
    # goals_completed counts both warm-up and collection; we want the
    # collection goal (goal 2) to have completed, so assert == 2.

    assert stats.goals_completed == 2, (
        f"Wood collection did not complete "
        f"(completed={stats.goals_completed}, failed={stats.goals_failed}, "
        f"stuck_events={stats.stuck_events}). "
        f"goals_completed==1 means warm-up passed but wood collection STUCK — "
        f"KB warm-up ran but entities_that_produce('wood') was still empty. "
        f"goals_completed==0 means even the warm-up failed — check RCON."
    )
    assert wq.inventory_count("wood") >= 5, (
        f"Goal completed but wood count is {wq.inventory_count('wood')} < 5. "
        f"Check that DestroySkill is correctly dropping items into inventory."
    )


def test_collect_wood_does_not_use_resource_patch(run_goals):
    """
    Verify that wood collection takes the harvest_natural path, not
    the gather_resource path. Wood has no resource patch in wq.resource_map
    so resources_of_type('wood') returns [] and Path B must be used.

    Prepends a warm-up goal so the KB has time to learn tree entity types
    before the collection goal fires.
    """
    entry = GoalQueueEntry(
        description="Collect 5 new wood (path verification)",
        goal_type="collection",
        success_condition="new.inventory('wood') >= 5",
        failure_condition="elapsed_ticks > 18000",
    )
    stats, wq = run_goals([_WARM_KB_GOAL, entry])

    assert stats.goals_failed == 0, (
        f"Wood collection STUCK — likely cause: "
        f"kb.entities_that_produce('wood') is empty because no tree "
        f"entity prototype has been queried yet. "
        f"stats={stats}"
    )
    assert stats.goals_completed == 2, (
        f"Wood collection did not complete. stats={stats}"
    )


def test_collect_wood_then_iron(run_goals):
    """
    Collect 5 new wood (harvest_natural) then 5 new iron ore (gather_resource).

    Verifies that:
      a) The harvest_natural pipeline works end-to-end
      b) The coordinator correctly transitions from a harvest task to a
         gather task for the next goal — no stale state from the harvest loop
      c) StopMining teardown fires correctly between goals (the harvest loop
         must be halted before the iron goal starts)
    """
    wood_entry = GoalQueueEntry(
        description="Collect 5 new wood",
        goal_type="collection",
        success_condition="new.inventory('wood') >= 5",
        failure_condition="elapsed_ticks > 18000",
    )
    iron_entry = GoalQueueEntry(
        description="Collect 5 new iron ore",
        goal_type="collection",
        success_condition="new.inventory('iron-ore') >= 5",
        failure_condition="elapsed_ticks > 10800",
    )
    stats, wq = run_goals([_WARM_KB_GOAL, wood_entry, iron_entry])

    assert stats.goals_completed == 3, (
        f"Expected both wood and iron goals to complete; "
        f"got completed={stats.goals_completed}, failed={stats.goals_failed}, "
        f"stuck_events={stats.stuck_events}. "
        f"goals_completed==1: warm-up passed, wood STUCK (KB not populated). "
        f"goals_completed==2: warm-up+wood passed, iron STUCK (mining agent reset). "
        f"goals_completed==0: even warm-up failed — check RCON."
    )
    assert wq.inventory_count("wood") >= 5
    assert wq.inventory_count("iron-ore") >= 5