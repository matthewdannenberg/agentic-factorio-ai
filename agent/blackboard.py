"""
agent/blackboard.py

Blackboard — shared working memory for agents within a goal's lifetime.

The blackboard is the communication substrate for the multi-agent network. It
is partitioned by scope and cleared at defined lifecycle boundaries:

  - SUBTASK-scoped entries are cleared when a subtask resolves (success or
    failure), allowing the next subtask to start with a clean slate for
    transient intentions and reservations.
  - GOAL-scoped entries persist for the full duration of the goal, allowing
    accumulation of durable observations and cross-subtask coordination.
  - All entries are cleared on goal reset (Blackboard.clear_all()).

Entry categories
----------------
  INTENTION   : Something an agent plans to do but hasn't done yet.
                Written by planning-adjacent agents; read by spatial and
                construction agents to avoid redundant or conflicting work.
  OBSERVATION : Something an agent has noticed that others should know.
                Written and read by any agent.
  RESERVATION : A claim on a resource, tile region, or entity.
                Written by the spatial-logistics agent; read by all agents
                to avoid conflicts.

Entry scopes
------------
  GOAL    : Persists for the full goal lifetime. Not cleared on subtask
            resolution.
  SUBTASK : Cleared when the current subtask resolves. Used for transient
            working data specific to one subtask's execution.

Entry expiry
------------
  An entry may carry an optional expires_at tick. Expired entries are not
  returned by read operations and are pruned lazily on the next read or
  explicit prune call.

Rules
-----
- No LLM calls. No RCON. No game state reads.
- Pure in-memory data structure. The coordinator holds the blackboard; agents
  receive it by reference on every tick.
- Thread safety is not a requirement at this stage — the coordinator is
  single-threaded within a tick.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntryCategory(Enum):
    """Semantic role of a blackboard entry."""
    INTENTION   = auto()   # planned but not yet executed
    OBSERVATION = auto()   # noticed fact shared with all agents
    RESERVATION = auto()   # claim on resource / region / entity


class EntryScope(Enum):
    """Lifetime of a blackboard entry relative to goal/subtask boundaries."""
    GOAL    = auto()   # persists across subtasks within the goal
    SUBTASK = auto()   # cleared when current subtask resolves


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

@dataclass
class BlackboardEntry:
    """
    A single entry in the blackboard.

    Fields
    ------
    id          : UUID string, auto-generated.
    category    : Semantic role (INTENTION, OBSERVATION, RESERVATION).
    scope       : Lifetime (GOAL or SUBTASK).
    owner_agent : Identifier string of the agent that wrote this entry.
    created_at  : Game tick at which the entry was written.
    data        : Arbitrary dict payload — the actual content of the entry.
    expires_at  : Optional tick after which this entry should no longer be
                  returned. None means the entry does not expire (it will
                  be cleared only by scope clearing or clear_all).
    """
    category: EntryCategory
    scope: EntryScope
    owner_agent: str
    created_at: int
    data: dict
    expires_at: Optional[int] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def is_expired(self, current_tick: int) -> bool:
        """True if this entry has passed its expiry tick."""
        return self.expires_at is not None and current_tick >= self.expires_at


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------

class Blackboard:
    """
    Shared working memory for agents within a goal's lifetime.

    Typical usage
    -------------
    The coordinator calls clear_all() on goal reset, clear_scope(SUBTASK) on
    each subtask resolution, and passes the blackboard to each agent on every
    tick. Agents call write() to post entries and read() to consume them.

    The main loop and examination layer may call snapshot() to capture the
    current state for StuckContext or examination summaries.
    """

    def __init__(self) -> None:
        # Master store: entry_id -> BlackboardEntry
        self._entries: dict[str, BlackboardEntry] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        category: EntryCategory,
        scope: EntryScope,
        owner_agent: str,
        created_at: int,
        data: dict,
        expires_at: Optional[int] = None,
    ) -> BlackboardEntry:
        """
        Write a new entry to the blackboard and return it.

        Parameters
        ----------
        category    : Semantic role.
        scope       : GOAL or SUBTASK lifetime.
        owner_agent : Identifier of the writing agent.
        created_at  : Current game tick.
        data        : Content dict — arbitrary structure, owned by the caller.
        expires_at  : Optional tick after which the entry is considered stale.

        Returns
        -------
        The newly created BlackboardEntry.
        """
        entry = BlackboardEntry(
            category=category,
            scope=scope,
            owner_agent=owner_agent,
            created_at=created_at,
            data=data,
            expires_at=expires_at,
        )
        self._entries[entry.id] = entry
        return entry

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(
        self,
        *,
        category: Optional[EntryCategory] = None,
        scope: Optional[EntryScope] = None,
        owner_agent: Optional[str] = None,
        current_tick: int = 0,
    ) -> list[BlackboardEntry]:
        """
        Return entries matching all supplied filters, excluding expired entries.

        Any filter parameter set to None is treated as a wildcard (matches all).

        Parameters
        ----------
        category     : Filter by entry category. None = all categories.
        scope        : Filter by scope. None = both scopes.
        owner_agent  : Filter by owning agent id. None = all agents.
        current_tick : Used to evaluate expiry. Entries expired at this tick
                       are excluded. Defaults to 0 (no expiry filtering if
                       all expires_at values are > 0).

        Returns
        -------
        Matching, non-expired entries in insertion order.
        """
        result = []
        for entry in self._entries.values():
            if entry.is_expired(current_tick):
                continue
            if category is not None and entry.category != category:
                continue
            if scope is not None and entry.scope != scope:
                continue
            if owner_agent is not None and entry.owner_agent != owner_agent:
                continue
            result.append(entry)
        return result

    def get(self, entry_id: str) -> Optional[BlackboardEntry]:
        """Return a specific entry by id, or None if not found or expired."""
        return self._entries.get(entry_id)

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear_scope(self, scope: EntryScope) -> int:
        """
        Remove all entries with the given scope.

        Called by the coordinator on subtask resolution (scope=SUBTASK) to
        clean up transient working data while preserving goal-scoped entries.

        Returns the number of entries removed.
        """
        to_remove = [
            eid for eid, e in self._entries.items() if e.scope == scope
        ]
        for eid in to_remove:
            del self._entries[eid]
        return len(to_remove)

    def clear_all(self) -> int:
        """
        Remove all entries regardless of scope.

        Called by the coordinator on goal reset.

        Returns the number of entries removed.
        """
        count = len(self._entries)
        self._entries.clear()
        return count

    def prune_expired(self, current_tick: int) -> int:
        """
        Remove all expired entries.

        Expiry is evaluated lazily on read() calls; this method allows
        explicit pruning to keep memory usage bounded on long-running goals.

        Returns the number of entries removed.
        """
        to_remove = [
            eid for eid, e in self._entries.items()
            if e.is_expired(current_tick)
        ]
        for eid in to_remove:
            del self._entries[eid]
        return len(to_remove)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, current_tick: int = 0) -> dict:
        """
        Return a plain-dict representation of all current (non-expired) entries.

        Used by StuckContext to capture working memory at the moment the
        execution network declares it is stuck. Also used by the examination
        layer for structured summaries.

        The snapshot is a shallow copy — modifications to it do not affect
        the blackboard.

        Format
        ------
        {
            entry_id: {
                "id": str,
                "category": str,          # EntryCategory.name
                "scope": str,             # EntryScope.name
                "owner_agent": str,
                "created_at": int,
                "expires_at": int | None,
                "data": dict,
            },
            ...
        }
        """
        return {
            eid: {
                "id": e.id,
                "category": e.category.name,
                "scope": e.scope.name,
                "owner_agent": e.owner_agent,
                "created_at": e.created_at,
                "expires_at": e.expires_at,
                "data": dict(e.data),
            }
            for eid, e in self._entries.items()
            if not e.is_expired(current_tick)
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Blackboard({len(self._entries)} entries)"
