# Factorio Agent

An agentic system that plays Factorio autonomously using an LLM for strategic reasoning
and pure Python for execution. The agent sets its own goals, decomposes them into tasks,
executes them, and learns across runs.

Target game version: **Factorio 2.x (Space Age)**.

## Status

**Phase 2 complete — bridge layer.** The RCON client, Lua mod, state parser, and action
executor are implemented and unit tested. In-game integration testing is the next step
before moving to the world model.

```
[✅] Core dataclasses     128 tests passing
[✅] Bridge / Lua mod      80 unit tests passing; in-game testing pending
[ ] World model
[ ] Planning layer
[ ] Primitives
[ ] Execution layer
[ ] Examination
[ ] Tactical LLM layer
[ ] Strategic LLM layer
[ ] Memory
[ ] Main loop
[ ] Threat module (biters)
```

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full design documentation including:
- Key design decisions and their rationale
- All dataclass interfaces (WorldState, Goal, AuditReport, Action types)
- Bridge layer component contracts and 2.x API notes
- State machine transitions
- Build order and testing strategy
- Component conversation brief template for handoff between conversations

## Project Structure

```
factorio-agent/
├── bridge/
│   ├── actions.py          ✅ 21 action types, ActionCategory, actions_for_context()
│   ├── rcon_client.py      ✅ RCON TCP connection, reconnect, thread-safe
│   ├── state_parser.py     ✅ JSON from Lua mod → WorldState (full + partial)
│   ├── action_executor.py  ✅ Action objects → RCON commands
│   └── mod/
│       ├── info.json       ✅ Factorio mod metadata (factorio_version: "2.0")
│       └── control.lua     ✅ Lua mod — exposes state, accepts commands
├── world/
│   └── state.py            ✅ WorldState and all sub-dataclasses
├── planning/
│   └── goal.py             ✅ Goal, RewardSpec, Priority, GoalStatus, make_goal()
├── agent/
│   ├── state_machine.py    ✅ AgentState, ExamineMode, assert_valid_transition()
│   └── examiner/
│       └── audit_report.py ✅ AuditReport, BoundingBox, BlueprintCandidate, damage types
├── config.py               ✅ All tunable parameters (RCON, scan radii, agent behaviour)
└── tests/
    ├── test_core_dataclasses.py         ✅ 128 tests, pure stdlib, no Factorio needed
    ├── unit/bridge/
    │   ├── test_state_parser.py         ✅ 80 tests against mock RCON responses
    │   └── test_action_executor.py      ✅ Round-trip command generation tests
    └── integration/
        └── test_bridge_live.lua         In-game test suite (requires running game)
```

## Running the Tests

### Unit tests — no Factorio required

No dependencies beyond Python 3.11+.

```bash
python -m unittest tests.test_core_dataclasses
python -m unittest tests.unit.bridge.test_state_parser
python -m unittest tests.unit.bridge.test_action_executor
```

### In-game integration tests

Once the mod is installed and Factorio is running with RCON enabled, open the in-game
console (`` ` `` key) and run:

```lua
-- Quick smoke test
/c rcon.print(fa.get_state({radius=32, resource_radius=128, item_radius=16}))

-- Full integration suite (prints PASS/FAIL per test)
/c require("test_bridge_live")

-- Run a single suite
/c T.run_suite("state_queries")   -- or "action_commands", "edge_cases"
```

## Installing the Mod and Connecting

### 1. Install the Factorio mod

Copy `bridge/mod/` into Factorio's mods directory, named exactly `factorio-agent_0.1.0`:

**Linux / Mac:**
```bash
cp -r bridge/mod ~/.factorio/mods/factorio-agent_0.1.0
```

**Windows:**
```
xcopy bridge\mod "%APPDATA%\Factorio\mods\factorio-agent_0.1.0" /E /I
```

Enable the mod in Factorio's mod manager before starting a game.

### 2. Start Factorio with RCON enabled

```bash
factorio --start-server-load-latest \
         --rcon-port 25575 \
         --rcon-password factorio
```

### 3. Configure the agent

Edit `config.py` if your setup differs from the defaults:

```python
RCON_HOST     = "localhost"
RCON_PORT     = 25575
RCON_PASSWORD = "factorio"
```

### Configuration reference

| Parameter | Default | Description |
|---|---|---|
| `RCON_HOST` | `"localhost"` | Factorio RCON host |
| `RCON_PORT` | `25575` | Factorio RCON port |
| `RCON_PASSWORD` | `"factorio"` | RCON password |
| `RCON_TIMEOUT_S` | `5.0` | Socket timeout in seconds |
| `RCON_RECONNECT_ATTEMPTS` | `5` | Max reconnect attempts before raising |
| `RCON_RECONNECT_BACKOFF_S` | `1.0` | Initial backoff; doubles each attempt |
| `LOCAL_SCAN_RADIUS` | `32` | Entity scan radius in tiles |
| `RESOURCE_SCAN_RADIUS` | `128` | Resource patch scan radius in tiles |
| `GROUND_ITEM_SCAN_RADIUS` | `16` | Ground item scan radius in tiles |
| `BITERS_ENABLED` | `False` | Enables COMBAT actions and threat module |
| `TICK_INTERVAL` | `10` | Poll every N game ticks |
| `LLM_CALL_BUDGET` | `30` | Max LLM calls per real-world hour |

## Design Principles

**The bridge is a hard boundary.** Nothing outside `bridge/` speaks RCON or knows about Lua.
The entire agent stack runs against mock `WorldState` objects without Factorio installed.

**LLM calls are expensive.** Target: 10–30 per real-world hour. The execution layer is pure
code. The LLM is called for strategic goal-setting, tactical decomposition, and examination
— never during primitive execution.

**WorldState is a belief state.** It is a cached, partially-observed snapshot assembled
by the bridge. Different sections have different staleness. The `observed_at` dict tracks
per-section freshness. Consumers must not treat it as ground truth.

**Resource types are strings.** Factorio internal names (`"iron-ore"`, `"crude-oil"`) are
used throughout. `ResourceName` provides named constants for vanilla resources. The
`ResourceRegistry` (to be implemented in `world/entities.py`) extends itself at runtime
when unfamiliar resource names appear — making the system compatible with Space Age and
modded content without code changes.

**Structural damage is cause-agnostic.** `DamagedEntity` and `DestroyedEntity` live on
`WorldState` directly. They are populated for any cause: biter attack, vehicle collision,
player error, deconstruction. `ThreatState` is strictly biter-specific.

**Actions are primitive and categorised.** `bridge/actions.py` contains only single-RCON-
call operations. Composition is the execution layer's job. Every action declares an
`ActionCategory`; `actions_for_context()` returns the valid set for the current situation
(on foot vs in vehicle, biters on vs off).

**Entity scanning is mod-compatible by design.** The Lua mod uses Factorio's `unit_number`
property rather than a whitelist of entity type strings. Any building added by any mod is
automatically visible to the agent. Trees, rocks, resources, and decoratives do not receive
unit numbers and are automatically excluded.

**Blueprints are agent-discovered.** `memory/blueprint_library/library.json` starts empty.
The rich examiner nominates regions via `BlueprintCandidate`; the curator extracts and
stores them. Evaluation (throughput/tile, tech tier, improvability) is in `BlueprintRecord`
in the memory layer, keeping nomination and evaluation separate.

**Circuit networks are deferred.** The agent can play a complete game without them. Adding
them later requires new action subclasses and Lua mod extensions; no existing code changes.

**Biters are off by default.** All COMBAT and biter-related code exists but is gated behind
`BITERS_ENABLED`. Enabling it switches the threat stub for the real monitor without
structural changes elsewhere.

## Passing Context for the Next Phase

When implementing the next component:

1. Include `ARCHITECTURE.md` — the full design and all interface specifications
2. Include the actual source files for everything the new component consumes or produces
3. Include the relevant test file(s) — the implementer should run them to verify their
   changes don't break existing contracts
4. Write a component brief using the template in `ARCHITECTURE.md`

For the world model (next phase), the files to include are:
- `ARCHITECTURE.md`
- `world/state.py` — ResourceRegistry must integrate with WorldState
- `bridge/state_parser.py` — the parser calls `registry.ensure(resource_type)`
- `tests/test_core_dataclasses.py` and `tests/unit/bridge/test_state_parser.py` — must
  still pass after world model is added