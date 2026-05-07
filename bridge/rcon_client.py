"""
bridge/rcon_client.py

RconClient — maintains a TCP connection to Factorio's RCON server.

Responsibilities:
- Open and maintain a persistent RCON connection.
- Reconnect automatically on disconnect with exponential backoff.
- Send a Lua command string, return the raw response string.
- Thread-safe (lock around socket I/O).

This module knows nothing about WorldState, Action, or Lua semantics.
It is a thin reliable transport layer.

RCON protocol reference
-----------------------
Each packet is: [4-byte LE length][4-byte LE request-id][4-byte LE type][payload bytes][00 00]
Types used:
  3 = SERVERDATA_AUTH
  2 = SERVERDATA_AUTH_RESPONSE / SERVERDATA_EXECCOMMAND
  0 = SERVERDATA_RESPONSE_VALUE
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from typing import Optional

import config

logger = logging.getLogger(__name__)

# RCON packet type constants
_TYPE_AUTH         = 3
_TYPE_EXECCOMMAND  = 2
_TYPE_RESPONSE     = 0
_AUTH_FAIL_ID      = -1


class BridgeError(Exception):
    """Raised for unrecoverable bridge failures (disconnected, auth failure, Lua error)."""


class RconClient:
    """
    Thread-safe RCON client for Factorio.

    Usage::

        client = RconClient()
        client.connect()
        response = client.send("/c game.print('hello')")
        client.close()

    Or use as a context manager::

        with RconClient() as client:
            response = client.send("...")
    """

    def __init__(
        self,
        host: str = config.RCON_HOST,
        port: int = config.RCON_PORT,
        password: str = config.RCON_PASSWORD,
        timeout_s: float = config.RCON_TIMEOUT_S,
        reconnect_attempts: int = config.RCON_RECONNECT_ATTEMPTS,
        reconnect_backoff_s: float = config.RCON_RECONNECT_BACKOFF_S,
    ) -> None:
        self._host = host
        self._port = port
        self._password = password
        self._timeout_s = timeout_s
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_backoff_s = reconnect_backoff_s

        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._request_id = 0
        self._connected = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the connection and authenticate. Raises BridgeError on failure."""
        with self._lock:
            self._connect_locked()

    def send(self, command: str) -> str:
        """
        Send a Lua command to Factorio and return the response string.

        Automatically reconnects once on transient failure.
        Raises BridgeError for unrecoverable failures.
        """
        with self._lock:
            if not self._connected:
                self._connect_locked()
            try:
                return self._send_locked(command)
            except (OSError, BridgeError) as exc:
                logger.warning("RCON send failed (%s); attempting reconnect.", exc)
                self._close_locked()
                self._connect_locked()
                return self._send_locked(command)

    def is_connected(self) -> bool:
        return self._connected

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RconClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers (all called with self._lock held)
    # ------------------------------------------------------------------

    def _connect_locked(self) -> None:
        backoff = self._reconnect_backoff_s
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._reconnect_attempts + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self._timeout_s)
                sock.connect((self._host, self._port))
                self._sock = sock
                self._authenticate_locked()
                self._connected = True
                logger.info(
                    "RCON connected to %s:%d (attempt %d).",
                    self._host, self._port, attempt,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "RCON connect attempt %d/%d failed: %s",
                    attempt, self._reconnect_attempts, exc,
                )
                if self._sock:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                    self._sock = None
                if attempt < self._reconnect_attempts:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)

        raise BridgeError(
            f"Failed to connect to RCON at {self._host}:{self._port} "
            f"after {self._reconnect_attempts} attempts."
        ) from last_exc

    def _authenticate_locked(self) -> None:
        req_id = self._next_id()
        self._send_packet_locked(req_id, _TYPE_AUTH, self._password)
        resp_id, resp_type, _ = self._recv_packet_locked()
        if resp_id == _AUTH_FAIL_ID or resp_type == _AUTH_FAIL_ID:
            raise BridgeError("RCON authentication failed — wrong password?")
        logger.debug("RCON authenticated (req_id=%d, resp_id=%d).", req_id, resp_id)

    def _send_locked(self, command: str) -> str:
        req_id = self._next_id()
        self._send_packet_locked(req_id, _TYPE_EXECCOMMAND, command)
        resp_id, _resp_type, payload = self._recv_packet_locked()
        if resp_id != req_id:
            logger.debug(
                "RCON response id mismatch: expected %d got %d (non-fatal).",
                req_id, resp_id,
            )
        return payload

    def _send_packet_locked(self, req_id: int, packet_type: int, body: str) -> None:
        body_bytes = body.encode("utf-8")
        # Packet format: length (4B LE) | id (4B LE) | type (4B LE) | body | \x00\x00
        payload = struct.pack("<ii", req_id, packet_type) + body_bytes + b"\x00\x00"
        length_prefix = struct.pack("<i", len(payload))
        assert self._sock is not None
        self._sock.sendall(length_prefix + payload)

    def _recv_packet_locked(self) -> tuple[int, int, str]:
        assert self._sock is not None
        # Read 4-byte length prefix
        raw_len = self._recvexact_locked(4)
        length = struct.unpack("<i", raw_len)[0]
        if length < 10:
            raise BridgeError(f"RCON packet too short: length={length}")
        # Read rest of packet
        data = self._recvexact_locked(length)
        req_id, packet_type = struct.unpack("<ii", data[:8])
        # Body is between bytes 8 and len-2 (strip two null terminators)
        body = data[8:-2].decode("utf-8", errors="replace")
        return req_id, packet_type, body

    def _recvexact_locked(self, n: int) -> bytes:
        assert self._sock is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise BridgeError("RCON connection closed by server.")
            buf.extend(chunk)
        return bytes(buf)

    def _close_locked(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._connected = False

    def _next_id(self) -> int:
        self._request_id = (self._request_id % 0x7FFF_FFFF) + 1
        return self._request_id
