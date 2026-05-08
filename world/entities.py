"""
world/entities.py

Thin facade over KnowledgeBase for entity and resource queries.

Preserves the public API used by bridge/state_parser.py and the rest of the
codebase, while delegating all storage and discovery to KnowledgeBase.

The agent starts with empty knowledge files and accumulates entries at runtime.

Public API 
--------------------------------------------
  EntityCategory          — enum of building categories
  EntityRecord            — metadata for a placed entity prototype (re-exported)
  ResourceRecord          — metadata for a resource patch type (re-exported)
  get_entity_metadata(name, kb) → EntityRecord
  ResourceRegistry(kb)    — legacy-compatible wrapper; used by state_parser
"""

from __future__ import annotations

from world.knowledge import (
    EntityCategory,
    EntityRecord,
    FluidRecord,
    KnowledgeBase,
    ResourceRecord,
)

__all__ = [
    "EntityCategory",
    "EntityRecord",
    "FluidRecord",
    "ResourceRecord",
    "ResourceRegistry",
    "get_entity_metadata",
]


def get_entity_metadata(name: str, kb: KnowledgeBase) -> EntityRecord:
    """
    Return the EntityRecord for a named entity, learning it if necessary.

    Delegates to kb.ensure_entity() — if the entity is unknown it will be
    queried from Factorio (or get a placeholder if the query function is
    unavailable). Never raises.
    """
    return kb.ensure_entity(name)


class ResourceRegistry:
    """
    Legacy-compatible wrapper around KnowledgeBase for resource queries.

    bridge/state_parser.py calls registry.ensure(resource_type) whenever it
    encounters a resource name. This class preserves that interface while
    routing all actual storage through KnowledgeBase.

    Note: the 'is_fluid' and 'is_infinite' properties on ResourceRecord are
    populated from Factorio prototype data rather than hard-coded assumptions,
    so they start as False (placeholder) until the game is queried.
    """

    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb

    def ensure(self, resource_type: str) -> ResourceRecord:
        """
        Register resource_type if unknown and return its record.
        Always succeeds — never raises for unknown names.
        """
        return self._kb.ensure_resource(resource_type)

    def get(self, resource_type: str) -> ResourceRecord | None:
        return self._kb.get_resource(resource_type)

    def all_known(self) -> list[str]:
        return list(self._kb.all_resources().keys())

    def is_fluid(self, resource_type: str) -> bool:
        record = self._kb.get_resource(resource_type)
        return record.is_fluid if record is not None else False

    def is_infinite(self, resource_type: str) -> bool:
        record = self._kb.get_resource(resource_type)
        return record.is_infinite if record is not None else False