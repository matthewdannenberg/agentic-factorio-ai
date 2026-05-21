"""
bridge/world_poller.py

WorldPoller — the sole place in the codebase that constructs the fa.get_state
Lua call and knows the RCON command format for world state queries.

This is the bridge-layer counterpart to bridge/prototype_query.py. Together
they ensure that no Lua strings, /c prefixes, or __agent__ namespace markers
appear anywhere outside bridge/.

The loop calls poller.poll() and receives a raw JSON string. It knows nothing
about how that string was obtained.
"""

from __future__ import annotations

import logging

from bridge.rcon_client import RconClient

log = logging.getLogger(__name__)


class WorldPoller:
    """
    Sends fa.get_state to Factorio via RCON and returns the raw JSON string.

    Parameters
    ----------
    client          : Connected RconClient.
    local_scan      : Radius in tiles for entity / ground-item scans.
    resource_scan   : Radius in tiles for resource patch scans.
    item_scan       : Radius in tiles for ground item scans.
    """

    def __init__(
        self,
        client: RconClient,
        local_scan: int,
        resource_scan: int,
        item_scan: int,
    ) -> None:
        self._client        = client
        self._local_scan    = local_scan
        self._resource_scan = resource_scan
        self._item_scan     = item_scan

        # Pre-build the command string — it never changes between polls.
        # /c __agent__ scopes the call into the agent mod's Lua environment
        # where the fa table is defined. rcon.print() sends the return value
        # back over the RCON socket as a string.
        self._cmd = (
            f"/c __agent__ rcon.print(fa.get_state({{"
            f"radius={self._local_scan}, "
            f"resource_radius={self._resource_scan}, "
            f"item_radius={self._item_scan}"
            f"}}))"
        )

    def poll(self) -> str:
        """
        Send fa.get_state to Factorio and return the raw JSON response string.

        Returns an empty string on failure (caller should treat as a skippable
        poll — the previous WorldState remains valid).
        """
        try:
            return self._client.send(self._cmd)
        except Exception:
            log.exception("WorldPoller: RCON poll failed")
            return ""
