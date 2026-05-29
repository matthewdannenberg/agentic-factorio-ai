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

**Agents own tasks, not goals.** Goals are the coordinator's concern. The coordinator
translates a Goal into a sequence of Tasks and hands each Task to the appropriate
agent. Agents never receive or inspect the Goal object.

## Status

**Phases 1–6 complete. Major refactor complete.**

The full end-to-end pipeline is implemented and running against a live Factorio
instance. The agent can navigate, gather resources, explore, and clear terrain.
All unit tests pass. In-game tests are being updated to the new structure.

```
[✅] Core dataclasses / bridge / Lua mod
[✅] World model (observable, knowledge, self-model layers)
[✅] Planning layer (Goals, Tasks, RewardEvaluator, condition namespace)
[✅] Execution layer (coordinator, agents, skills, blackboard, memory)
[✅] Examination layer (AuditReport stub)
[✅] Phase 6 — Navigation, mining, rule-based coordinator, run loop
[✅] Refactor — world/, planning/, execution/, examination/, memory/ layout
[ ] Phase 7 — Construction agent
[ ] Phase 8 — Production agent (RL)
[ ] Phase 9 — Spatial-logistics agent (RL)
[ ] Phase 10 — Examination layer revision
[ ] Phase 11 — LLM layer revision
```

## Architecture Overview

The system has five top-level layers plus the bridge:

**Bridge** (`bridge/`) — the sole RCON boundary. Produces `WorldState` snapshots,
dispatches `Action` objects. Nothing outside `bridge/` speaks RCON or knows Lua.

**World** (`world/`) — three sub-layers:
- *Observable* — `WorldState` (data), `WorldQuery` (read), `WorldWriter` (write)
- *Knowledge* — `KnowledgeBase`: game content learned at runtime (recipes, entities, tech)
- *Model* — `SelfModel`: layered self-model of what the agent has built. Currently
  two layers: `FactoryGraph` (logical factory components) and `ChunkGrid` (explored terrain)

**Planning** (`planning/`) — `Goal`, `Task`, `GoalTree`, `RewardEvaluator`,
`condition_namespace`. Goals are long-horizon objectives; Tasks are concrete
work items the coordinator assigns to agents. `RewardEvaluator` evaluates
condition strings against a `WorldQuery` namespace.

**Execution** (`execution/`) — the agent network. Contains the coordinator,
agents, skills, blackboard, preconditions, and the main loop. The coordinator
translates Goals into Tasks via a hierarchical, depth-first state machine and
dispatches them to agents. Agents use skills to produce actions.

**Examination** (`examination/`) — `AuditReport` and auditor stubs.
Grows significantly in Phase 10.

**Memory** (`memory/`) — `BehavioralMemory`: strategy records and performance
history persisted across runs via SQLite.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full design documentation.

## Key Concepts

### Goals and Tasks

**Goals** are coarse, long-horizon objectives set by the LLM or the coordinator's
hierarchical planner (e.g. "collect 200 iron ore", "clear this region", "produce
iron plates at 30/min"). Goals have `success_condition` and `failure_condition`
strings evaluated against `WorldQuery`.

**Tasks** are concrete work items derived by the coordinator and assigned to a
single agent (e.g. "navigate to position X", "gather 10 iron ore", "clear natural
objects in bbox"). Tasks also carry condition strings, evaluated by the coordinator
each tick.

The coordinator never exposes Goals to agents. Agents receive Tasks only.

### Skills

Skills encode multi-step but well-determined sequences of primitive actions, along
with success/failure detection. They contain no decision logic — that belongs to
agents. Examples: `NavigateSkill`, `MineSkill`, `DestroySkill`, `CraftSkill`.
Each agent holds references to the skills relevant to its task types.

### Self-Model Patches

Agents do not read or write the self-model directly. Instead, they emit
`SelfModelPatch` objects describing changes resulting from their actions
(e.g. "I placed an assembler at this location"). The coordinator collects
patches from agents each tick and applies them to the `SelfModel`.

### Coordinator Architecture

The coordinator maintains a **goal stack** (pending goals in depth-first order)
and an **active task**. Each tick it either ticks the active task or runs the
top goal's handler. Goal handlers advance a `step` counter and may push sub-goals
or tasks. Completed sub-goals pop from the stack and advance their parent's step.
This gives the appearance of recursive decomposition while keeping the tick loop flat.

### Writing Goal Conditions

Use **relative conditions** rather than absolute game state where possible:

```python
# Collect 5 new iron ore (works regardless of prior inventory)
success_condition="new.inventory('iron-ore') >= 5"

# 3 minutes since this goal activated (not absolute game clock)
failure_condition="elapsed_ticks > 10800"

# Explore 5 new chunks (works regardless of prior charted count)
success_condition="new.charted_chunks >= 5"
```

See [`REWARD_NAMESPACE.md`](REWARD_NAMESPACE.md) for the complete namespace
reference and [`CONDITION_SCOPE.md`](CONDITION_SCOPE.md) for proximal vs
non-proximal guidance.

## Running the Agent

```bash
# Default smoke-test sequence
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

### Unit tests — no Factorio required

```bash
pytest tests/
pytest tests/unit/execution/ -v
pytest tests/unit/world/ -v
pytest tests/unit/planning/ -v
```

### In-game tests — Factorio required

Start Factorio with RCON enabled, then:

```bash
pytest tests_in_game/ -v
pytest tests_in_game/01_knowledge/ -v
pytest tests_in_game/02_collection/ -v
pytest tests_in_game/03_exploration/ -v
```

In-game tests skip automatically if RCON is unreachable.

### Lua-side tests — Factorio console required

```
/c __agent__ T.run_all()     # bridge state queries and actions
/c __agent__ TM.run_all()    # movement and pathfinding
/c __agent__ TS.run_all()    # exploration and mining status
/c __agent__ TP.run_all()    # prototype queries (KB)
```

## Project Structure

```
factorio-agent/
├── bridge/          # RCON boundary — nothing outside speaks RCON
├── world/
│   ├── observable/  # WorldState, WorldQuery, WorldWriter
│   ├── knowledge/   # KnowledgeBase, TechTree, ProductionTracker
│   └── model/       # SelfModel, FactoryGraph, ChunkGrid
├── planning/        # Goal, Task, GoalTree, RewardEvaluator, condition_namespace
├── execution/
│   ├── agents/      # NavigationAgent, MiningAgent, ExplorationAgent, CraftingAgent
│   ├── skills/      # NavigateSkill, MineSkill, DestroySkill, CraftSkill, ...
│   ├── coordinator/ # RuleBasedCoordinator, GoalFrame, goal handlers
│   └── memory/      # BehavioralMemory (SQLite-backed)
├── examination/     # AuditReport, auditor stubs (grows in Phase 10)
├── memory/          # Cross-run behavioral memory (may gain more in future phases)
├── llm/             # GoalSource, GoalQueue (LLM client in Phase 11)
├── data/            # Runtime databases (gitignored)
├── config.py
├── run.py
├── tests/           # Unit and integration tests
└── tests_in_game/   # In-game integration tests
```

## Installation

### 1. Install the Factorio mod

```bash
# Linux / Mac
cp -r bridge/mod ~/.factorio/mods/agent_0.1.0

# Windows
xcopy bridge\mod "%APPDATA%\Factorio\mods\agent_0.1.0" /E /I
```

Enable the mod in Factorio's mod manager before starting a game.

### 2. Start Factorio with RCON enabled

Via Steam launch options: `--rcon-port 25575 --rcon-password factorio` then 
upon launching the game, on the main menu hold down "Ctrl-Alt" and click "Settings".
Select "The Rest" and find the "local-rcon-socket" and "local-rcon-password" options.
Enter `127.0.0.1:25575` and `factorio`, respectively.

Or headless on Windows:

```powershell
& "C:\...\factorio.exe" --start-server "$env:APPDATA\Factorio\saves\agent-test.zip" `
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

Should print `'table\n'`.

### 4. Configure

Edit `config.py` if your setup differs from the defaults:

```python
RCON_HOST     = "localhost"
RCON_PORT     = 25575
RCON_PASSWORD = "factorio"
```

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