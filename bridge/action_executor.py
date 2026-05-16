"""
bridge/action_executor.py

ActionExecutor — translates Action objects into RCON commands sent to Factorio.

Rules:
- One RCON call per action (actions are primitive by design).
- Returns False for recoverable failures (entity out of reach, inventory full).
- Raises BridgeError for unrecoverable failures (disconnected, Lua error).
- VEHICLE and COMBAT category actions raise NotImplementedError (stubs until needed).
- Does NOT validate whether an action is appropriate for context — that is the
  execution layer's job via actions_for_context().

Lua command convention
----------------------
All commands are Factorio console commands of the form:
    /c <lua expression>

The Lua mod exposes a global `fa` table with helper functions. For simple actions
(movement, crafting) we call these helpers. The return value is a JSON string
with at minimum {"ok": true} or {"ok": false, "reason": "..."}.

Recoverable failure: Lua returns {"ok": false, "reason": "out_of_reach"} etc.
Unrecoverable failure: Lua returns a raw error string (not valid JSON), or the
RCON connection drops.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from bridge.actions import (
    Action,
    ApplyBlueprint,
    CraftItem,
    DriveVehicle,
    EnterVehicle,
    EquipArmor,
    ExitVehicle,
    FlipEntity,
    MineEntity,
    MineResource,
    MoveTo,
    NoOp,
    PlaceEntity,
    RotateEntity,
    SelectWeapon,
    SetFilter,
    SetRecipe,
    SetResearchQueue,
    SetSplitterPriority,
    ShootAt,
    StopMovement,
    StopShooting,
    TransferItems,
    UseItem,
    Wait,
)
from bridge.rcon_client import BridgeError, RconClient

logger = logging.getLogger(__name__)


def _lua(expr: str) -> str:
    """Wrap a Lua expression as a Factorio console command."""
    return f"/c __agent__ {expr}"


def _pos(position: Any) -> str:
    """Serialise a Position as a Lua table literal."""
    return f"{{x={position.x}, y={position.y}}}"


def _parse_response(raw: str) -> dict[str, Any]:
    """
    Parse a Lua response string into a dict.

    Raises BridgeError if the response is not valid JSON (indicates a Lua
    error or unexpected output from Factorio).
    """
    raw = raw.strip()
    if not raw:
        # Empty response is OK for fire-and-forget commands.
        return {"ok": True}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise BridgeError(f"Lua returned non-JSON response: {raw!r}")


def _is_ok(resp: dict[str, Any]) -> bool:
    return bool(resp.get("ok", True))  # Absence of "ok" → assume success.


class ActionExecutor:
    """
    Translates Action objects into RCON commands.

    Parameters
    ----------
    client : RconClient
        An already-connected RCON client. The executor does not manage the
        connection lifecycle — the caller (bridge or main loop) is responsible.
    """

    def __init__(self, client: RconClient) -> None:
        self._client = client

    def execute(self, action: Action) -> bool:
        """
        Execute a single primitive action.

        Returns True on success, False on recoverable failure.
        Raises BridgeError on unrecoverable failure.
        Raises NotImplementedError for stub actions (VEHICLE, COMBAT).
        """
        kind = action.kind
        handler = self._dispatch.get(kind)
        if handler is None:
            raise BridgeError(f"ActionExecutor: unknown action kind '{kind}'")
        return handler(self, action)

    # ------------------------------------------------------------------
    # Internal helper: send command and interpret response
    # ------------------------------------------------------------------

    def _send(self, lua_expr: str) -> bool:
        """Send a Lua command, return True on success, False on recoverable failure."""
        raw = self._client.send(_lua(lua_expr))
        resp = _parse_response(raw)
        ok = _is_ok(resp)
        if not ok:
            reason = resp.get("reason", "unknown")
            logger.debug("Action failed (recoverable): %s", reason)
        return ok

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _execute_move_to(self, action: MoveTo) -> bool:  # type: ignore[override]
        return self._send(
            f"fa.move_to({_pos(action.position)}, {str(action.pathfind).lower()})"
        )

    def _execute_stop_movement(self, action: StopMovement) -> bool:
        return self._send("fa.stop_movement()")

    def _execute_mine_resource(self, action: MineResource) -> bool:
        return self._send(
            f"fa.mine_resource({_pos(action.position)}, "
            f'"{action.resource}", {action.count})'
        )

    def _execute_mine_entity(self, action: MineEntity) -> bool:
        return self._send(f"fa.mine_entity({action.entity_id})")

    def _execute_craft_item(self, action: CraftItem) -> bool:
        return self._send(f'fa.craft_item("{action.recipe}", {action.count})')

    def _execute_place_entity(self, action: PlaceEntity) -> bool:
        return self._send(
            f'fa.place_entity("{action.item}", {_pos(action.position)}, '
            f"{action.direction.value})"
        )

    def _execute_set_recipe(self, action: SetRecipe) -> bool:
        return self._send(f'fa.set_recipe({action.entity_id}, "{action.recipe}")')

    def _execute_set_filter(self, action: SetFilter) -> bool:
        item_lua = f'"{action.item}"' if action.item else "nil"
        return self._send(f"fa.set_filter({action.entity_id}, {action.slot}, {item_lua})")

    def _execute_apply_blueprint(self, action: ApplyBlueprint) -> bool:
        force = str(action.force_build).lower()
        return self._send(
            f'fa.apply_blueprint("{action.blueprint_string}", '
            f"{_pos(action.position)}, {action.direction.value}, {force})"
        )

    def _execute_transfer_items(self, action: TransferItems) -> bool:
        to_player = "true" if action.direction == "from_entity" else "false"
        item_lua = f'"{action.item}"' if action.item else "nil"
        return self._send(
            f"fa.transfer_items({action.entity_id}, {to_player}, "
            f"{item_lua}, {action.count})"
        )

    def _execute_set_research_queue(self, action: SetResearchQueue) -> bool:
        techs_lua = (
            "{" + ", ".join(f'"{t}"' for t in action.technologies) + "}"
        )
        return self._send(f"fa.set_research_queue({techs_lua})")

    def _execute_equip_armor(self, action: EquipArmor) -> bool:
        return self._send(f'fa.equip_armor("{action.item}")')

    def _execute_use_item(self, action: UseItem) -> bool:
        if action.target_position is not None:
            return self._send(
                f'fa.use_item("{action.item}", {_pos(action.target_position)})'
            )
        return self._send(f'fa.use_item("{action.item}", nil)')

    def _execute_wait(self, action: Wait) -> bool:
        # Wait is handled by the execution layer sleeping; no RCON command needed.
        import time
        time.sleep(action.ticks / 60.0)
        return True

    def _execute_noop(self, action: NoOp) -> bool:
        return True

    def _execute_set_splitter_priority(self, action: SetSplitterPriority) -> bool:
        input_lua = f'"{action.input_priority}"' if action.input_priority is not None else "nil"
        output_lua = f'"{action.output_priority}"' if action.output_priority is not None else "nil"
        return self._send(
            f"fa.set_splitter_priority({action.entity_id}, {input_lua}, {output_lua})"
        )

    def _execute_rotate_entity(self, action: RotateEntity) -> bool:
        return self._send(
            f"fa.rotate_entity({action.entity_id}, {str(action.reverse).lower()})"
        )

    def _execute_flip_entity(self, action: FlipEntity) -> bool:
        return self._send(
            f"fa.flip_entity({action.entity_id}, {str(action.horizontal).lower()})"
        )

    # ------------------------------------------------------------------
    # VEHICLE stubs (NotImplementedError until vehicles are needed)
    # ------------------------------------------------------------------

    def _execute_enter_vehicle(self, action: EnterVehicle) -> bool:
        raise NotImplementedError(
            "EnterVehicle is a stub — vehicle support is not yet implemented. "
            "Enable when the agent reaches vehicle-relevant content."
        )

    def _execute_exit_vehicle(self, action: ExitVehicle) -> bool:
        raise NotImplementedError(
            "ExitVehicle is a stub — vehicle support is not yet implemented."
        )

    def _execute_drive_vehicle(self, action: DriveVehicle) -> bool:
        raise NotImplementedError(
            "DriveVehicle is a stub — vehicle support is not yet implemented."
        )

    # ------------------------------------------------------------------
    # COMBAT stubs (NotImplementedError until BITERS_ENABLED)
    # ------------------------------------------------------------------

    def _execute_select_weapon(self, action: SelectWeapon) -> bool:
        raise NotImplementedError(
            "SelectWeapon is a stub — BITERS_ENABLED is False. "
            "Enable when biters are activated."
        )

    def _execute_shoot_at(self, action: ShootAt) -> bool:
        raise NotImplementedError(
            "ShootAt is a stub — BITERS_ENABLED is False. "
            "Enable when biters are activated."
        )

    def _execute_stop_shooting(self, action: StopShooting) -> bool:
        raise NotImplementedError(
            "StopShooting is a stub — BITERS_ENABLED is False. "
            "Enable when biters are activated."
        )

    # ------------------------------------------------------------------
    # Dispatch table — built at class definition time
    # ------------------------------------------------------------------

    _dispatch: dict[str, Any] = {
        "MoveTo":            _execute_move_to,
        "StopMovement":      _execute_stop_movement,
        "MineResource":      _execute_mine_resource,
        "MineEntity":        _execute_mine_entity,
        "CraftItem":         _execute_craft_item,
        "PlaceEntity":       _execute_place_entity,
        "SetRecipe":         _execute_set_recipe,
        "SetFilter":         _execute_set_filter,
        "SetSplitterPriority": _execute_set_splitter_priority,
        "RotateEntity":      _execute_rotate_entity,
        "FlipEntity":        _execute_flip_entity,
        "ApplyBlueprint":    _execute_apply_blueprint,
        "TransferItems":     _execute_transfer_items,
        "SetResearchQueue":  _execute_set_research_queue,
        "EquipArmor":        _execute_equip_armor,
        "UseItem":           _execute_use_item,
        "Wait":              _execute_wait,
        "NoOp":              _execute_noop,
        # VEHICLE stubs
        "EnterVehicle":      _execute_enter_vehicle,
        "ExitVehicle":       _execute_exit_vehicle,
        "DriveVehicle":      _execute_drive_vehicle,
        # COMBAT stubs
        "SelectWeapon":      _execute_select_weapon,
        "ShootAt":           _execute_shoot_at,
        "StopShooting":      _execute_stop_shooting,
    }