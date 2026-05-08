"""
    Support fixtures necessary for multiple test files.
"""

import json 

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