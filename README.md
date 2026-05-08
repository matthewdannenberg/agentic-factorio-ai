# Factorio Agent

An agentic system that plays Factorio autonomously using an LLM for strategic reasoning
and pure Python for execution. The agent sets its own goals, decomposes them into tasks,
executes them, and learns across runs.

Target game version: **Factorio 2.x (Space Age)**.

## Status

**Phase 3 complete — world model.** The knowledge base, tech tree, production tracker,
and entity/resource facades are implemented and tested. The knowledge base uses SQLite
for persistent, queryable storage of all learned game knowledge.

```
[✅] Core dataclasses     128 tests passing
[✅] Bridge / Lua mod      80 unit tests passing; in-game testing pending
[✅] World model          see breakdown below
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

### World model test breakdown

```
[✅] world/knowledge.py        75 tests — SQLite-backed KnowledgeBase
[✅] world/entities.py         36 tests — ResourceRegistry and entity metadata facade
[✅] world/tech_tree.py        64 tests — KnowledgeBase-backed TechTree
[✅] world/production_tracker.py  (existing tests passing)
```

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full design documentation including:
- Key design decisions and their rationale
- All dataclass interfaces (WorldState, Goal, AuditReport, Action types)
- Bridge layer component contracts and 2.x API notes
- World model: KnowledgeBase schema, discovery flow, enrichment lifecycle
- State machine transitions
- Build order and testing strategy
- Component conversation brief template for handoff between conversations

## Project Structure

```
factorio-agent/
├── bridge/
│   ├── actions.py              ✅ 21 action types, ActionCategory, actions_for_context()
│   ├── rcon_client.py          ✅ RCON TCP connection, reconnect, thread-safe
│   ├── state_parser.py         ✅ JSON from Lua mod → WorldState (full + partial)
│   ├── action_executor.py      ✅ Action objects → RCON commands
│   └── mod/
│       ├── info.json           ✅ Factorio mod metadata (factorio_version: "2.0")
│       └── control.lua         ✅ Lua mod — exposes state, accepts commands,
│                                    includes prototype query functions
├── world/
│   ├── state.py                ✅ WorldState and all sub-dataclasses
│   ├── knowledge.py            ✅ KnowledgeBase — SQLite-backed, runtime-extensible
│   ├── entities.py             ✅ Facade: ResourceRegistry, get_entity_metadata()
│   ├── tech_tree.py            ✅ TechTree — KB-backed research graph queries
│   └── production_tracker.py  ✅ Throughput tracking over successive WorldState snapshots
├── planning/
│   └── goal.py                 ✅ Goal, RewardSpec, Priority, GoalStatus, make_goal()
├── agent/
│   ├── state_machine.py        ✅ AgentState, ExamineMode, assert_valid_transition()
│   └── examiner/
│       └── audit_report.py     ✅ AuditReport, BoundingBox, BlueprintCandidate
├── data/
│   └── knowledge/
│       └── knowledge.db        (created at runtime — gitignored)
├── config.py                   ✅ All tunable parameters
└── tests/
    ├── test_core_dataclasses.py             ✅ 128 tests
    ├── unit/bridge/
    │   ├── test_state_parser.py             ✅ 80 tests
    │   └── test_action_executor.py          ✅
    ├── unit/world/
    │   ├── test_knowledge.py                ✅ 75 tests
    │   ├── test_entities.py                 ✅ 36 tests
    │   ├── test_tech_tree.py                ✅ 64 tests
    │   └── test_production_tracker.py       ✅
    └── integration/
        └── test_bridge_live.lua             In-game test suite (requires running game)
```

## Running the Tests

### Unit tests — no Factorio required

```bash
# Core dataclasses
python -m unittest tests.test_core_dataclasses

# Bridge
python -m unittest tests.unit.bridge.test_state_parser
python -m unittest tests.unit.bridge.test_action_executor

# World model
python -m unittest tests.unit.world.test_knowledge
python -m unittest tests.unit.world.test_entities
python -m unittest tests.unit.world.test_tech_tree
python -m unittest tests.unit.world.test_production_tracker
```

### In-game integration tests

Once the mod is installed and Factorio is running with RCON enabled:

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

**The world model starts empty.** The agent begins each fresh install knowing nothing about
specific game content. `KnowledgeBase` is populated entirely at runtime by querying Factorio
prototype data via RCON. Knowledge persists across runs in `data/knowledge/knowledge.db`.
Mods and Space Age content are handled automatically — any entity, resource, fluid, recipe,
or technology the agent encounters is queried, stored, and available for future reasoning
without code changes.

**LLM calls are expensive.** Target: 10–30 per real-world hour. The execution layer is pure
code. The LLM is called for strategic goal-setting, tactical decomposition, and examination
— never during primitive execution.

**WorldState is a belief state.** It is a cached, partially-observed snapshot assembled
by the bridge. Different sections have different staleness. The `observed_at` dict tracks
per-section freshness. Consumers must not treat it as ground truth.

**Knowledge is queryable, not just retrievable.** The SQLite backing store supports
cross-domain queries the planning layer will need: all recipes that produce X, all recipes
that run in entity Y, techs that unlock a given recipe, recursive production chain closure.
These are indexed SQL queries, not O(n) Python scans.

**Placeholders degrade gracefully.** When the agent encounters an unknown name offline
(no RCON connection), a placeholder record is stored immediately. Once a connection is
available, `KnowledgeBase.enrich_placeholders()` re-queries all placeholders and replaces
them with real data. Enriched records persist to the DB so subsequent runs start fully
informed.

**Actions are primitive and categorised.** `bridge/actions.py` contains only single-RCON-
call operations. Composition is the execution layer's job. Every action declares an
`ActionCategory`; `actions_for_context()` returns the valid set for the current situation.

**Biters are off by default.** All COMBAT and biter-related code exists but is gated behind
`BITERS_ENABLED`. Enabling it switches the threat stub for the real monitor without
structural changes elsewhere.

## Passing Context for the Next Phase

When implementing the next component:

1. Include `ARCHITECTURE.md` — the full design and all interface specifications
2. Include the actual source files for everything the new component consumes or produces
3. Include the relevant test file(s) — the implementer should run them to verify changes
   don't break existing contracts
4. Write a component brief using the template in `ARCHITECTURE.md`

For the planning layer (next phase), the files to include are:
- `ARCHITECTURE.md`
- `world/knowledge.py` — the planning layer queries the KB directly
- `world/tech_tree.py` — goal-setting uses TechTree.path_to() and next_researchable()
- `world/state.py` — RewardSpec is evaluated against WorldState
- `planning/goal.py` — Goal, RewardSpec, Priority (already implemented, extend here)
- `tests/unit/world/test_knowledge.py` — must still pass