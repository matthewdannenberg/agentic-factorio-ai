"""
tests/unit/bridge/rcon_client.py

Tests for bridge/rcon_client.py

Run with:  python tests/unit/bridge/rcon_client.py
"""

from __future__ import annotations

import struct
import threading
import unittest
from unittest.mock import patch

from bridge.action_executor import ActionExecutor
from bridge.actions import NoOp
from bridge.rcon_client import BridgeError, RconClient
from tests.fixtures import MockRconClient


def _make_rcon_packet(req_id: int, packet_type: int, body: str) -> bytes:
    """Build a valid RCON response packet."""
    body_bytes = body.encode("utf-8")
    payload = struct.pack("<ii", req_id, packet_type) + body_bytes + b"\x00\x00"
    return struct.pack("<i", len(payload)) + payload


class FakeSocket:
    """Simulates a socket for RCON testing."""

    def __init__(self, recv_data: bytes) -> None:
        self._recv_buf = bytearray(recv_data)
        self.sent_data = bytearray()
        self._closed = False

    def sendall(self, data: bytes) -> None:
        self.sent_data.extend(data)

    def recv(self, n: int) -> bytes:
        chunk = bytes(self._recv_buf[:n])
        del self._recv_buf[:n]
        return chunk

    def close(self) -> None:
        self._closed = True

    def settimeout(self, t: float) -> None:
        pass

    def connect(self, addr: tuple) -> None:
        pass


class TestRconClient(unittest.TestCase):
    def _make_auth_response(self, req_id: int) -> bytes:
        return _make_rcon_packet(req_id, 2, "")

    def _make_exec_response(self, req_id: int, body: str) -> bytes:
        return _make_rcon_packet(req_id, 0, body)

    def _make_client_with_socket(self, sock: FakeSocket) -> RconClient:
        client = RconClient(
            host="localhost", port=25575, password="test",
            timeout_s=1.0, reconnect_attempts=1, reconnect_backoff_s=0.0,
        )
        with patch("socket.socket") as mock_socket_cls:
            mock_socket_cls.return_value = sock
            client.connect()
        return client

    def _make_response_sequence(self, *packets: bytes) -> bytes:
        return b"".join(packets)

    def test_connect_and_is_connected(self):
        # Auth: client sends req_id=1 type 3; server replies with req_id=1 type 2.
        auth_resp = self._make_auth_response(1)
        sock = FakeSocket(auth_resp)
        client = self._make_client_with_socket(sock)
        self.assertTrue(client.is_connected())

    def test_send_returns_response_body(self):
        # Sequence: auth(req=1) then exec response(req=2, body='{"ok":true}')
        auth_resp = self._make_auth_response(1)
        exec_resp = self._make_exec_response(2, '{"ok":true}')
        sock = FakeSocket(auth_resp + exec_resp)
        client = self._make_client_with_socket(sock)
        result = client.send("/c fa.get_tick()")
        self.assertEqual(result.strip(), '{"ok":true}')

    def test_auth_failure_raises_bridge_error(self):
        # Server responds with id=-1 to signal auth failure.
        auth_fail = _make_rcon_packet(-1, 2, "")
        sock = FakeSocket(auth_fail)
        with patch("socket.socket") as mock_socket_cls:
            mock_socket_cls.return_value = sock
            client = RconClient(
                host="localhost", port=25575, password="wrong",
                timeout_s=1.0, reconnect_attempts=1, reconnect_backoff_s=0.0,
            )
            with self.assertRaises(BridgeError):
                client.connect()

    def test_close_marks_disconnected(self):
        auth_resp = self._make_auth_response(1)
        sock = FakeSocket(auth_resp)
        client = self._make_client_with_socket(sock)
        self.assertTrue(client.is_connected())
        client.close()
        self.assertFalse(client.is_connected())

    def test_thread_safety_multiple_sends(self):
        """Multiple threads can call send() without corruption."""
        # This test just checks no exceptions are raised — true socket
        # interleaving can't be simulated without a real server.
        results: list[bool] = []
        errors: list[Exception] = []

        def send_worker(executor: ActionExecutor) -> None:
            try:
                r = executor.execute(NoOp())
                results.append(r)
            except Exception as e:
                errors.append(e)

        mock_client = MockRconClient()
        executor = ActionExecutor(mock_client)
        threads = [threading.Thread(target=send_worker, args=(executor,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)