"""
execution/skills/

Stateful, tick-driven capability units. Each skill encapsulates one thing
the player character can do in Factorio, including internal state management,
success/stuck/failure detection, and action generation.

Skills have no Blackboard, no Task, no coordinator — they know only the game
world (WorldQuery) and produce Actions. All coordination stays in the agent.

Implemented (Phase 6)
---------------------
NavigateSkill  Walk to a position or entity.  start(target_position, target_entity_id)
MineSkill      Mine a resource patch for N.   start(position, resource, count)
CraftSkill     Hand-craft a list of items.    start(targets, expected_post_inv)
DestroySkill   Deconstruct one entity.        start(entity_id)

Stubs (Phase 7)
---------------
PlaceSkill     Place an entity.               start(item, position, direction)
InteractSkill  Set recipe/filter/rotate/flip. start(entity_id, action, **params)

"""

from execution.skills.base import SkillProtocol, SkillStatus
from execution.skills.navigate import NavigateSkill
from execution.skills.mine import MineSkill
from execution.skills.craft import CraftSkill, CraftTarget
from execution.skills.place import PlaceSkill
from execution.skills.interact import InteractSkill
from execution.skills.destroy import DestroySkill

__all__ = [
    "SkillProtocol",
    "SkillStatus",
    "NavigateSkill",
    "MineSkill",
    "CraftSkill",
    "CraftTarget",
    "DestroySkill",
    "PlaceSkill",
    "InteractSkill",
]