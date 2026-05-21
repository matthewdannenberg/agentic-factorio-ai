"""
bridge/prototype_query.py

Factory for the PrototypeQueryFn used by world/knowledge.py.

This is the sole place in the codebase that knows:
  - The Factorio mod namespace prefix (__agent__)
  - The fa.get_*_prototype / fa.get_technology function names
  - That results come back via rcon.print() as JSON strings

world/knowledge.py receives a PrototypeQueryFn and calls it as:
    query_fn("recipe", "iron-gear-wheel")  ->  dict | None

It never sees Lua, RCON commands, or mod namespacing.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from bridge.rcon_client import RconClient

logger = logging.getLogger(__name__)

# The callable type injected into KnowledgeBase.
# (domain: str, name: str) -> parsed prototype dict, or None on failure.
PrototypeQueryFn = Callable[[str, str], Optional[dict]]

# Maps knowledge domain names to the fa.* Lua function that fetches them.
_DOMAIN_TO_LUA_FN: dict[str, str] = {
    "entity":   "fa.get_entity_prototype",
    "resource": "fa.get_resource_prototype",
    "fluid":    "fa.get_fluid_prototype",
    "recipe":   "fa.get_recipe_prototype",
    "tech":     "fa.get_technology",
}


def make_prototype_query_fn(client: RconClient) -> PrototypeQueryFn:
    """
    Return a PrototypeQueryFn backed by the given RconClient.

    The returned function is safe to call from any thread that already holds
    any necessary coordination — it delegates to RconClient.send() which is
    itself thread-safe.

    Usage::

        query_fn = make_prototype_query_fn(rcon_client)
        kb = KnowledgeBase(data_dir=..., query_fn=query_fn)
    """
    def _query(domain: str, name: str) -> Optional[dict]:
        lua_fn = _DOMAIN_TO_LUA_FN.get(domain)
        if lua_fn is None:
            logger.warning("prototype_query: unknown domain %r (name=%r)", domain, name)
            return None

        # Build the full Factorio console command.
        # __agent__ scopes the call to the agent mod's Lua environment,
        # where the fa table is defined.
        command = f"/c __agent__ rcon.print({lua_fn}({json.dumps(name)}))"

        try:
            raw = client.send(command)
        except Exception as exc:
            logger.debug("prototype_query: RCON error for %s(%r): %s", lua_fn, name, exc)
            return None

        raw = raw.strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug(
                "prototype_query: non-JSON response for %s(%r): %r", lua_fn, name, raw
            )
            return None

        if isinstance(data, dict) and data.get("ok") is False:
            logger.debug(
                "prototype_query: Lua error for %s(%r): %s",
                lua_fn, name, data.get("reason"),
            )
            return None

        return data

    return _query
