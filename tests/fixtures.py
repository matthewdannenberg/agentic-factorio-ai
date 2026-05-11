"""
tests/fixtures.py

Shared test fixtures and helpers used across multiple test files.

WorldState construction helpers
--------------------------------
make_world_state(tick, entities, inserter_activity) -> WorldState
make_world_query(tick, entities, inserter_activity) -> WorldQuery

    Use these in any test file that needs a WorldState/WorldQuery purely as
    scaffolding (e.g. test_production_tracker, test_reward_evaluator).
    Centralising construction here means a WorldState signature change only
    requires updating one place, rather than every test file that happened to
    build one.

    test_state.py is the exception: it tests WorldState internals directly and
    constructs WorldState objects explicitly by design.

MockRconClient
--------------
Fake RCON client that records commands and returns canned responses.
"""

from __future__ import annotations

import json
from typing import Optional

from world.state import (
    EntityState,
    EntityStatus,
    Inventory,
    InventorySlot,
    LogisticsState,
    Position,
    WorldState,
)
from world.query import WorldQuery


# ---------------------------------------------------------------------------
# MockRconClient
# ---------------------------------------------------------------------------

class MockRconClient:
    """Fake RCON client that records commands sent and returns canned responses."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self.sent_commands: list[str] = []
        self._default_response = json.dumps({"ok": True})

    def send(self, command: str) -> str:
        self.sent_commands.append(command)
        for prefix, response in self._responses.items():
            if prefix in command:
                return response
        return self._default_response

    def is_connected(self) -> bool:
        return True

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# WorldState / WorldQuery construction helpers
# ---------------------------------------------------------------------------

def make_world_state(
    tick: int = 0,
    entities: Optional[list[EntityState]] = None,
    inserter_activity: Optional[dict[int, int]] = None,
) -> WorldState:
    """
    Build a minimal WorldState for use as test scaffolding.

    Use this in test files that need a WorldState purely to drive something
    else (e.g. ProductionTracker, RewardEvaluator).  Do not use in
    test_state.py, which constructs WorldState explicitly to test its internals.
    """
    logistics = LogisticsState()
    if inserter_activity:
        logistics = LogisticsState(inserter_activity=inserter_activity)
    return WorldState(
        tick=tick,
        entities=entities or [],
        logistics=logistics,
    )


def make_world_query(
    tick: int = 0,
    entities: Optional[list[EntityState]] = None,
    inserter_activity: Optional[dict[int, int]] = None,
) -> WorldQuery:
    """
    Build a WorldQuery wrapping a minimal WorldState.

    Use this wherever a test needs a WorldQuery as input scaffolding.
    """
    return WorldQuery(make_world_state(tick, entities, inserter_activity))


def make_inventory_entity(
    entity_id: int,
    name: str,
    items: dict[str, int],
    status: EntityStatus = EntityStatus.WORKING,
) -> EntityState:
    """
    Build an EntityState with a populated inventory.  Convenience for
    production tracker tests that need entities with known output counts.
    """
    slots = [InventorySlot(item=k, count=v) for k, v in items.items()]
    return EntityState(
        entity_id=entity_id,
        name=name,
        position=Position(0, 0),
        status=status,
        inventory=Inventory(slots=slots),
    )