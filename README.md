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

## Status

**Phases 1-5 complete.**
The bridge, world model, planning, and execution layer foundation are fully
implemented and tested. Phase 6 (navigation agent and rule-based coordinator) is next.

```
[✅] Core dataclasses       128 tests passing
[✅] Bridge / Lua mod        80+ tests passing
[✅] World model            200+ tests passing
[✅] Planning layer         ~170 tests passing
[✅] WorldQuery/Writer refactor — all tests passing
[✅] Phase 5  — Execution layer foundation   179 tests passing
[ ] Phase 6  — Navigation agent and rule-based coordinator
[ ] Phase 7  — Observation and reward infrastructure
[ ] Phase 8  — Production agent (RL)
[ ] Phase 9  — Spatial-logistics agent (RL)
[ ] Phase 10 — Examination layer revision
[ ] Phase 11 — LLM layer revision
[ ] Phase 12 — Main loop and state machine
```

## Architecture Overview

The system has four major layers:

**LLM planning layer** — sets coarse goals (e.g. "establish iron plate production
at 30 plates/second", "explore and find a nearby oil patch"). Fires infrequently.
Receives structured factory summaries, not raw game state. Also handles escalation
when the execution network gets stuck.

**Agent execution network** — a coordinated set of learned agents operating behind
a clean protocol interface. Communicates internally via a shared blackboard and
subtask ledger. Responsible for everything between receiving a goal and the game
state changing.

**Knowledge and memory system** — three distinct layers:
- *Game knowledge* (KnowledgeBase) — what the game contains, learned at runtime
- *Factory self-model* — a graph of logical factory units and their relationships,
  built incrementally during execution
- *Behavioral memory* — strategies and patterns accumulated across runs

**Game Interaction** — a dedicated interface to reading the state of the game at
any given moment and carrying out actions selected by the execution network in-game.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full design documentation.

## Project Structure

```
factorio-agent/
├── bridge/          # RCON interface, Lua mod, state parser, action executor
├── world/           # WorldState, WorldQuery, WorldWriter, KnowledgeBase
├── planning/        # GoalTree, RewardEvaluator, ResourceAllocator
├── agent/
│   ├── execution_protocol.py  # ExecutionLayerProtocol, ExecutionResult, StuckContext
│   ├── blackboard.py          # Shared working memory for agents
│   ├── subtask.py             # Subtask, SubtaskLedger (live stack + history log)
│   ├── self_model.py          # Factory self-model graph
│   ├── network/     # Coordinator, agent registry, individual agents
│   ├── memory/      # Behavioral memory (SQLite-backed)
│   └── examiner/    # Rich and mechanical examination
├── llm/             # LLM client, prompts, rate limiting
├── data/            # Runtime knowledge database (gitignored)
├── config.py        # All tunable parameters
├── tests/           # Unit and integration tests
└── tests_in_game/   # Integration tests run with an active Factorio instance
```

## Running the Tests

### Unit and integration tests — no Factorio required

```bash
# All tests
python -m unittest discover -s tests -v

# By layer
python -m unittest tests.unit.bridge.test_actions
python -m unittest tests.unit.world.test_state
python -m unittest tests.unit.planning.test_goal_tree
python -m unittest tests.unit.agent.test_blackboard
python -m unittest tests.unit.agent.test_subtask
python -m unittest tests.unit.agent.test_self_model
python -m unittest tests.unit.agent.test_behavioral_memory
python -m unittest tests.unit.agent.test_execution_protocol
python -m unittest tests.unit.agent.test_registry
python -m unittest tests.integration.test_evaluator_capabilities
```

### In-game mod tests

Once the mod is installed and Factorio is running with RCON enabled:

```lua
/c require("test_bridge_live")
```

### In-Game Tests

Tests in this directory require a live Factorio instance with RCON enabled.
They are kept separate from `tests/` so that `pytest tests/` runs cleanly
without a game connection.

#### Running

Start Factorio with RCON enabled (see README.md in the project root for the
full setup), then:

```bash
# All in-game tests
pytest tests_in_game/ -v

# One category
pytest tests_in_game/02_collection/ -v

# One file
pytest tests_in_game/02_collection/test_collect_iron.py -v
```

If Factorio is not reachable the entire session is skipped with a clear message.

#### Directory structure

Subdirectories of tests_in_game/ are numbered to express dependencies and natural execution order:

```
01_knowledge/   KB population — should run first; other tests benefit from a warm KB
02_collection/  Resource gathering goals
03_exploration/ Exploration goals
```

Within a directory, files run in alphabetical order (pytest default).

#### Design principles

**Tests run through the real loop.** Every test feeds a `GoalQueueEntry` to
`FactorioLoop` via the `run_goal` or `run_goals` fixture in `conftest.py`.
No test calls bridge, coordinator, or agent code directly. This ensures the
test exercises the same code path as production.

**Short failure conditions.** Test goals use tight failure conditions (≤ 3600
ticks, ≈ 1 minute) so a broken agent fails fast rather than timing out after 5+
minutes.

**Isolation.** Each `run_goal` / `run_goals` call gets a fresh Blackboard,
SubtaskLedger, SelfModel, and BehavioralMemory. Only the KnowledgeBase persists
across the session — it accumulates entity and recipe knowledge as tests run,
matching production behaviour.

**Multi-goal sequences.** Use `run_goals([entry_a, entry_b])` when two goals
should share accumulated world knowledge (KB state) but still run in a clean
coordinator context. Prefer this over per-test file ordering for tightly
coupled sequences.

#### Adding new tests

1. Create a file in the appropriate numbered directory (or add a new numbered
   directory if the category is new).
2. Use the `run_goal` or `run_goals` fixture — never build the loop manually.
3. Keep failure conditions at ≤ 3600 ticks unless the task genuinely requires
   longer (document why if so).
4. Assert on `stats.goals_completed`, `stats.goals_failed`, and
   `wq.inventory_count(...)` / `wq.charted_chunks` / etc. as appropriate.

## Installing the Mod and Connecting

### 1. Install the Factorio mod

```bash
# Linux / Mac
cp -r bridge/mod ~/.factorio/mods/factorio-agent_0.1.0

# Windows
xcopy bridge\mod "%APPDATA%\Factorio\mods\factorio-agent_0.1.0" /E /I
```

Enable the mod in Factorio's mod manager before starting a game.

### 2. Start Factorio with RCON enabled

```bash
factorio --start-server-load-latest \
         --rcon-port 25575 \
         --rcon-password factorio
```

### 3. Configure

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