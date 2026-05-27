"""
execution/skills/interact.py

InteractSkill — interact with a placed entity (set recipe, set filter,
rotate, flip).

Phase 7 stub. Interface is complete; implementation deferred to Phase 7.

These interactions all share the same structure: navigate to reach the
entity (handled by the agent using NavigateSkill before this), issue the
command, confirm the change via entity scan. Grouping them in one skill
avoids four nearly-identical skills with trivial differences.

start() parameters
------------------
entity_id : Id of the entity to interact with.
action    : One of: "set_recipe", "set_filter", "rotate", "flip".
params    : Action-specific keyword arguments:
            set_recipe : recipe (str)
            set_filter : slot (int), item (str)
            rotate     : (no extra params needed)
            flip       : (no extra params needed)

Status transitions (stub)
--------------------------
IDLE    → RUNNING : start() called.
RUNNING → FAILED  : immediately on first tick (not yet implemented).

Implementation notes (Phase 7)
-------------------------------
- Map action string to the appropriate bridge Action subclass.
- Issue the action.
- Confirm by polling entity state on the next tick (recipe changed,
  filter set, orientation updated).
- SUCCEEDED once confirmed. FAILED if entity not found in scan.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from execution.skills.base import SkillProtocol, SkillStatus
from bridge import Action

if TYPE_CHECKING:
    from world import WorldQuery, WorldWriter

log = logging.getLogger(__name__)

_VALID_ACTIONS = frozenset({"set_recipe", "set_filter", "rotate", "flip"})


class InteractSkill(SkillProtocol):
    """Interact with a placed entity. Phase 7 stub."""

    def __init__(self) -> None:
        self._status: SkillStatus = SkillStatus.IDLE
        self._entity_id: Optional[int] = None
        self._action: str = ""
        self._params: dict[str, Any] = {}

    def start(
        self,
        entity_id: int,
        action: str,
        **params: Any,
    ) -> None:
        """
        Initialise for an interaction job.

        Parameters
        ----------
        entity_id : Target entity id.
        action    : "set_recipe" | "set_filter" | "rotate" | "flip".
        **params  : Action-specific parameters (see module docstring).
        """
        if action not in _VALID_ACTIONS:
            raise ValueError(
                f"InteractSkill: unknown action {action!r}. "
                f"Valid: {sorted(_VALID_ACTIONS)}"
            )
        self._entity_id = entity_id
        self._action    = action
        self._params    = params
        self._status    = SkillStatus.RUNNING
        log.debug(
            "InteractSkill started: entity=%d action=%s params=%s",
            entity_id, action, params,
        )

    def tick(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> list[Action]:
        if self._status != SkillStatus.RUNNING:
            return []
        log.warning("InteractSkill: not yet implemented — FAILED")
        self._status = SkillStatus.FAILED
        return []

    def status(self) -> SkillStatus:
        return self._status

    def reset(self) -> None:
        self._status    = SkillStatus.IDLE
        self._entity_id = None
        self._action    = ""
        self._params    = {}

    def observe(self) -> dict:
        return {
            "interact_status":    self._status.name,
            "interact_entity_id": self._entity_id,
            "interact_action":    self._action,
        }
