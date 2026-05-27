"""
execution/skills/base.py

SkillProtocol and SkillStatus — the interface every skill must satisfy.

A Skill is a stateful, tick-driven capability unit. It encapsulates one thing
a player can do in Factorio — navigate, mine, place, craft — including all the
internal state needed to do it reliably across multiple ticks, the success/
stuck/failure detection specific to that action, and the action generation each
tick.

Design rules
------------
- Skills have no Blackboard, no Task, no KnowledgeBase, no coordinator.
  They know only the game world (WorldQuery) and produce Actions.
- Each concrete skill defines its own typed start() method. The base class
  defines start() as taking no parameters; subclasses override with the
  signature appropriate to their job.
- tick() is called every poll cycle while the skill is RUNNING. It returns
  the list of Actions to dispatch (may be empty).
- A skill transitions to SUCCEEDED, FAILED, or STUCK on its own — the agent
  reads skill.status() after each tick and decides what to do next.
- reset() returns the skill to IDLE, clearing all internal state. The agent
  can then call start() again with new parameters.
- Agents hold skill instances across task lifetimes and reuse them. Skills
  are never constructed per-tick.

Status semantics
----------------
IDLE      Not started or reset. start() has not been called since last reset.
RUNNING   Actively working. tick() is producing meaningful actions or
          waiting on a game condition (e.g. pathfinder computing, mining swing
          in progress). The agent should keep ticking.
SUCCEEDED The skill completed its job successfully. The agent should read
          observe() for any useful output, then reset() or move on.
FAILED    The skill encountered an irrecoverable error (e.g. target entity
          no longer exists, resource exhausted before count reached). The agent
          should escalate or choose a different approach.
STUCK     Progress has stalled past the grace period. The skill cannot advance
          on its own. The agent should decide whether to retry (call start()
          again) or escalate to the coordinator.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bridge import Action
    from world import WorldQuery, WorldWriter


class SkillStatus(Enum):
    IDLE      = auto()
    RUNNING   = auto()
    SUCCEEDED = auto()
    FAILED    = auto()
    STUCK     = auto()

    @property
    def is_terminal(self) -> bool:
        """True if the skill has finished (SUCCEEDED, FAILED, or STUCK)."""
        return self in (SkillStatus.SUCCEEDED, SkillStatus.FAILED, SkillStatus.STUCK)


class SkillProtocol:
    """
    Base interface for all skills.

    Concrete skills inherit from this class and override:
      - start()  with typed parameters for their specific job
      - tick()   with the action-generation and state-update logic
      - observe() with skill-specific observation keys

    tick(), status(), reset(), and observe() are defined here with safe
    defaults so that stub skills (Phase 7+) can inherit and only override
    what they implement.
    """

    def start(self) -> None:
        """
        Initialise the skill for a new job.

        Concrete skills override this with typed parameters:
            def start(self, target_position: Position, ...) -> None

        Calling start() on a RUNNING skill resets and restarts it.
        Calling start() on a terminal skill restarts it with new parameters.
        """
        raise NotImplementedError

    def tick(
        self,
        wq: "WorldQuery",
        ww: "WorldWriter",
        tick: int,
    ) -> "list[Action]":
        """
        Advance by one poll cycle.

        Called every tick while the skill is RUNNING. Returns the list of
        Actions to dispatch this tick (may be empty). Must update internal
        status as appropriate.

        Must not be called on an IDLE or terminal skill without first calling
        start() (or reset() + start()).
        """
        raise NotImplementedError

    def status(self) -> SkillStatus:
        """Current status of the skill."""
        raise NotImplementedError

    def reset(self) -> None:
        """
        Return to IDLE, clearing all internal state.

        Safe to call at any time. After reset(), the skill is ready for a
        new start() call.
        """
        raise NotImplementedError

    def observe(self) -> dict:
        """
        Return a flat dict of named observations from the skill's current state.

        Called by the agent's observe() method to build its combined output.
        Keys should be prefixed with the skill name to avoid collision when
        agents hold multiple skills, e.g. {"navigate_target": ..., ...}.

        Returns {} by default; concrete skills override as needed.
        """
        return {}
