# Factorio Agent

A multi-agent reinforcement learning system that plays Factorio autonomously.
The system receives coarse strategic goals from an LLM, decomposes and executes
them through a learned agent network, and accumulates knowledge and strategy
across runs.

Target game version: **Factorio 2.x (Space Age)**.

## Design Philosophy

**Learning over programming.** The agent begins each fresh install knowing nothing
about Factorio's content beyond the schema needed to store what it learns. Game
mechanics, production relationships, spatial strategies, and logistics patterns
emerge from runtime experience rather than hard-coded rules.

**Execution intelligence is local.** The LLM handles long-horizon strategic
reasoning over novel situations. Everything below coarse goal setting —
decomposition, planning, spatial reasoning, logistics, navigation — is the agent
network's responsibility. LLM calls are expensive, infrequent, and narrowly scoped.

**Clean segmentation.** Every major component exposes a narrow, versioned interface.
Implementations can be replaced — including wholesale replacement of the RL system —
without restructuring surrounding code.

**Agents own subtasks, not goals.** Goals are the coordinator's concern. The
coordinator translates a Goal into Subtasks and hands each Subtask to the
appropriate agent. Agents never receive or inspect the Goal object.

## Status

**Phases 1-6 complete.**
The full end-to-end pipeline is implemented and running against a live Factorio
instance. The agent can navigate, gather resources, explore, and clear terrain.
All in-game tests pass, including Factorio 2.x compatibility fixes and delta-based
condition evaluation (`new.inventory()`, `new.charted_chunks()`).
Phase 7 (construction agent and observation/reward infrastructure) is next.

```
[✅] Core dataclasses       128 tests passing
[✅] Bridge / Lua mod        80+ tests passing (4 Lua test suites: T/TM/TS/TP)
[✅] World model            200+ tests passing
[✅] Planning layer         ~200 tests passing (condition_namespace, DeltaView)
[✅] WorldQuery/Writer refactor — all tests passing
[✅] Phase 5  — Execution layer foundation   179 tests passing
[✅] Phase 6  — Navigation, mining, rule-based coordinator, run loop
[✅] Phase 6+ — Factorio 2.x compatibility; delta conditions; all in-game tests pass
[ ] Phase 7  — Construction agent, observation and reward infrastructure
[ ] Phase 8  — Production agent (RL)
[ ] Phase 9  — Spatial-logistics agent (RL)
[ ] Phase 10 — Examination layer revision
[ ] Phase 11 — LLM layer revision
```

## Architecture Overview

The system has four major layers:

**LLM planning layer** — sets coarse goals (e.g. "establish iron plate production
at 30 plates/second", "explore and find a nearby oil patch"). Fires infrequently.
Receives structured factory summaries, not raw game state. Also handles escalation
when the execution network gets stuck. Currently stubbed by `GoalQueue` in
`llm/goal_source.py`, which accepts manually-authored or pre-recorded goal sequences.

**Agent execution network** — a coordinated set of agents operating behind
a clean protocol interface. Communicates internally via a shared blackboard and
subtask ledger. The coordinator translates Goals into Subtasks and routes each
Subtask to the appropriate agent by `agent_hint`. Agents interact with Subtasks
only — never Goals.

**Knowledge and memory system** — three distinct layers:
- *Game knowledge* (KnowledgeBase) — what the game contains, learned at runtime
- *Factory self-model* — a graph of logical factory units and their relationships,
  built incrementally during execution
- *Behavioral memory* — strategies and patterns accumulated across runs

**Game interaction** — the Factorio mod (`bridge/mod/control.lua`) exposes a
persistent-state movement and mining system driven by `on_tick` handlers. The
`fa.move_to()` function uses Factorio's built-in pathfinder (`surface.request_path`)
with a collision mask covering `object` and `water_tile` layers so the character
routes around trees, walls, and water. `fa.mine_resource()` and `fa.mine_entity()`
similarly maintain `mining_state` every tick. The Python bridge speaks to this mod
exclusively via RCON.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full design documentation.

## Writing Goal Conditions

Goal conditions should use **relative conditions** rather than absolute game state
where possible. The `new` object and `elapsed_ticks` alias give access to deltas
since goal activation:

```python
# Collect 5 new iron ore (works regardless of prior inventory)
success_condition="new.inventory('iron-ore') >= 5"

# 3 minutes since this goal activated (not absolute game clock)
failure_condition="elapsed_ticks > 10800"   # or: new.tick > 10800

# Explore 5 new chunks (works regardless of prior charted count)
success_condition="new.charted_chunks >= 5"
```

See [`REWARD_NAMESPACE.md`](REWARD_NAMESPACE.md) for the complete namespace
reference and [`CONDITION_SCOPE.md`](CONDITION_SCOPE.md) for proximal vs
non-proximal guidance.

## Running the Agent

```bash
# Default smoke-test sequence (collect iron, then explore)
python run.py

# Single inline goal
python run.py --goal-type collection \
              --success "new.inventory('iron-ore') >= 10" \
              --failure "elapsed_ticks > 18000" \
              --description "Collect 10 new iron ore"

# Load a recorded goal sequence
python run.py --goals runs/my_goals.json

# Save goals and outcomes after the run
python run.py --save-goals runs/session_2026.json
```

## Running the Tests

### Unit and integration tests — no Factorio required

```bash
# All unit tests
pytest tests/

# Specific layer
pytest tests/unit/agent/ -v
pytest tests/unit/planning/ -v   # includes condition_namespace tests
```

### In-game tests — Factorio required

Start Factorio with RCON enabled (see Installation below), then:

```bash
# All in-game tests
pytest tests_in_game/ -v

# One category at a time (recommended — tests run in numbered order)
pytest tests_in_game/01_knowledge/ -v
pytest tests_in_game/02_collection/ -v
pytest tests_in_game/03_exploration/ -v
```

In-game tests skip automatically if RCON is unreachable. Live log streaming is
configured in `pytest.ini` — add `--log-cli-level DEBUG` for full coordinator
and agent output.

### Lua-side tests — Factorio console required

Four test suites can be run directly from the Factorio in-game console. These test
the bridge interface without any Python involvement and are the fastest way to
diagnose Factorio API issues:

```
/c __agent__ T.run_all()           # state queries, action commands, edge cases
/c __agent__ TM.run_suite("status_api")    # movement status API
/c __agent__ TM.async_start("player_actually_moves")
# wait 5 seconds
/c __agent__ TM.async_finish("player_actually_moves")
/c __agent__ TS.run_all()          # exploration (charted_chunks), mining status
/c __agent__ TP.run_all()          # prototype queries (entities, recipes, tech)
```

Results are written to `script-output/` in the Factorio data directory.

## Project Structure

```
factorio-agent/
├── bridge/          # RCON interface, Lua mod, state parser, action executor
├── world/           # WorldState, WorldQuery, WorldWriter, KnowledgeBase
├── planning/        # GoalTree, RewardEvaluator, condition_namespace, ResourceAllocator
├── agent/
│   ├── execution_protocol.py  # ExecutionLayerProtocol, ExecutionResult, StuckContext
│   ├── blackboard.py          # Shared working memory for agents
│   ├── subtask.py             # Subtask, SubtaskLedger (live stack + history log)
│   ├── self_model.py          # Factory self-model graph
│   ├── preconditions.py       # is_at, is_reachable, can_reach_count, valid_actions
│   ├── loop.py                # FactorioLoop — master tick loop
│   ├── network/
│   │   ├── agent_protocol.py  # AgentProtocol — subtask-first interface
│   │   ├── coordinator.py     # RuleBasedCoordinator + StubCoordinator
│   │   ├── registry.py        # AgentRegistry — no goal-type coupling
│   │   └── agents/
│   │       ├── navigation.py  # NavigationAgent — movement only
│   │       └── mining.py      # MiningAgent — gathering and clearing
│   ├── memory/
│   │   └── behavioral.py      # BehavioralMemoryProtocol + SQLiteBehavioralMemory
│   └── examiner/              # Rich and mechanical examination (Phase 10)
├── llm/
│   └── goal_source.py         # GoalSource protocol, GoalQueue implementation
├── data/            # Runtime databases (gitignored)
├── config.py        # All tunable parameters
├── run.py           # CLI entry point
├── tests/           # Unit and integration tests (no Factorio required)
└── tests_in_game/   # In-game tests (Factorio required)
    ├── conftest.py  # Shared fixtures: rcon_client, knowledge_base, run_goal
    ├── 01_knowledge/
    ├── 02_collection/
    └── 03_exploration/
```

## Installation

### 1. Install the Factorio mod

```bash
# Linux / Mac
cp -r bridge/mod ~/.factorio/mods/agent_0.1.0

# Windows
xcopy bridge\mod "%APPDATA%\Factorio\mods\agent_0.1.0" /E /I
```

Enable the mod in Factorio's mod manager before starting a game, or add it
to `%APPDATA%\Factorio\mods\mod-list.json`.

### 2. Start Factorio with RCON enabled

In our tests, this has been done through Steam, where one edits the game launch
via Properties -> Launch Options, entering `--rcon-port 25575 --rcon-password factorio`.
Alternatively, these commands can be added to the executable launching the game.

#### Visualized GUI Option

Upon launching the game, on the main menu hold down "Ctrl-Alt" and click "Settings".
Select "The Rest" and find the "local-rcon-socket" and "local-rcon-password" options.
Enter `127.0.0.1:25575` and `factorio`, respectively.

#### Headless Option (No GUI)

On Windows, the most reliable approach is the `--start-server` flag (the
standalone `factorio.exe`, not the Steam-launched version):

```powershell
& "C:\Program Files (x86)\Steam\steamapps\common\Factorio\bin\x64\factorio.exe" `
  --start-server "$env:APPDATA\Factorio\saves\agent-test.zip" `
  --rcon-port 25575 --rcon-password factorio `
  --mod-directory "$env:APPDATA\Factorio\mods"
```

### 3. Verify the connection

```bash
python -c "
from bridge.rcon_client import RconClient
c = RconClient('localhost', 25575, 'factorio', timeout_s=5.0)
print(repr(c.send('/c rcon.print(type(fa))')))
c.close()
"
```

Should print `'table\n'`. If it prints `'nil\n'`, the mod is not loaded —
check that the mod folder is correctly installed and the save was created
with the mod enabled.

### 4. Configure

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
| `RCON_RECONNECT_ATTEMPTS` | `5` | Max reconnect attempts |
| `RCON_RECONNECT_BACKOFF_S` | `1.0` | Initial backoff; doubles each attempt |
| `LOCAL_SCAN_RADIUS` | `32` | Entity scan radius in tiles |
| `RESOURCE_SCAN_RADIUS` | `128` | Resource patch scan radius in tiles |
| `GROUND_ITEM_SCAN_RADIUS` | `16` | Ground item scan radius in tiles |
| `BITERS_ENABLED` | `False` | Enables combat agents and threat module |
| `TICK_INTERVAL` | `10` | Poll every N game ticks |

## Key Documents

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — full design, interfaces, data flow, phase plan
- [`CONDITION_SCOPE.md`](CONDITION_SCOPE.md) — proximal vs non-proximal condition reference
- [`REWARD_NAMESPACE.md`](REWARD_NAMESPACE.md) — complete reward evaluator namespace reference
- [`OPEN_DECISIONS.md`](OPEN_DECISIONS.md) — deferred architectural questions
- [`PHASE_7_BRIEF.md`](PHASE_7_BRIEF.md) — component brief for the next phase
- [`PRE_PHASE_7_DECISIONS.md`](PRE_PHASE_7_DECISIONS.md) — decisions required before Phase 7 begins