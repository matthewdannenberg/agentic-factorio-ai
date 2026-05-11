# Factorio Agent

An agentic system that plays Factorio autonomously using an LLM for strategic reasoning
and pure Python for execution. The agent sets its own goals, decomposes them into tasks,
executes them, and learns across runs.

Target game version: **Factorio 2.x (Space Age)**.

## Status

**Phase 4 complete — planning layer, plus world-state access refactor.**
The goal tree, reward evaluator, and resource allocator are implemented and tested. The
evaluator exposes a rich condition namespace covering inventory, exploration, production
rates, connectivity, research, logistics, and more. A `WorldQuery` / `WorldWriter`
access boundary was introduced over world state, enforcing a clean read/write interface
across the codebase. See `REWARD_NAMESPACE.md` for the complete reference and
`CONDITION_SCOPE.md` for the critical proximal/non-proximal distinction that governs how
goals must be written.

```
[✅] Core dataclasses     128 tests passing
[✅] Bridge / Lua mod      80+ unit tests passing; basic in-game testing passed
[✅] World model          200+ tests
[✅] Planning layer        ~80 unit tests + ~90 integration tests passing
[✅] WorldQuery/WorldWriter access refactor — all tests passing
[ ] Primitives
[ ] Execution layer
[ ] Examination
[ ] Tactical LLM layer
[ ] Strategic LLM layer
[ ] Memory
[ ] Main loop
[ ] Threat module (biters)
```

### Unit Test Breakdown - by source file

```
[✅] agent/examiner/              25 tests — AuditReport
[✅] agent/state_machine.py        9 tests — StateMachine
[✅] bridge/action_executor.py    28 tests — ActionExecutor
[✅] bridge/actions.py            40 tests — Action, ActionCategory, and actions_for_context
[✅] bridge/rcon_client.py         5 tests — RconClient
[✅] bridge/state_parser.py       50 tests — two-lane belts, inserter objects, charted_chunks
[✅] planning/goal.py             20 tests — Goal, Priority, RewardSpec
[✅] planning/goal_tree.py        40 tests — GoalTree, LIFO preemption, subgoal completion
[✅] planning/resource_allocator  10 tests — pass-through interface
[✅] planning/reward_evaluator    30 tests — conditions, discounting, milestones, safety
[✅] world/entities.py            36 tests — ResourceRegistry and entity metadata facade
[✅] world/knowledge.py           75 tests — SQLite-backed KnowledgeBase
[✅] world/production_tracker.py  25 tests — ProductionSummary, ProductionTracker
[✅] world/query.py + writer.py   ~40 tests — WorldQuery (lookups, connectivity, builder),
                                             WorldWriter (section replacement, fine-grained
                                             mutation, integrate_snapshot) — in test_state.py
[✅] world/state.py               87 tests — WorldState indices, BeltLane/BeltSegment,
                                             InserterState, ExplorationState
[✅] world/tech_tree.py           64 tests — KnowledgeBase-backed TechTree
```

### Integration Test Breakdown - by test file in tests/integration/

```
[✅] test_StateParser_WorldState  9 tests  — StateParser with WorldQuery
[✅] test_evaluator_capabilities  ~91 tests — capability matrix across 14 categories:
                                   IV (inventory), EN (entity placement),
                                   PR (production/logistics, two-lane belts),
                                   RS (research), RM (resource map), TM (time),
                                   DM (damage), CN (connectivity), GI (ground items),
                                   EX (exploration / charted_chunks — NON-PROXIMAL),
                                   PT (production_rate — PROXIMAL),
                                   ST (staleness guards), SC (scope boundary),
                                   XC (compound conditions, including wq builder)
```

### In Game Test Breakdown

```
[✅] test_bridge_live.lua         ~35 tests - basic in-game functionality of state reading
```

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full design documentation including:
- Key design decisions and their rationale
- All dataclass interfaces (WorldState, Goal, AuditReport, Action types, GoalTree,
  RewardEvaluator, ExplorationState, BeltLane/BeltSegment, InserterState)
- WorldQuery / WorldWriter access boundary design
- Bridge layer: two-lane belt reading, inserter positions, charted_chunks, fa.get_exploration
- World model: KnowledgeBase schema, discovery flow, enrichment lifecycle
- Planning layer: GoalTree, RewardEvaluator namespace, ResourceAllocator
- State machine transitions
- Build order and testing strategy
- Component conversation brief template for handoff between conversations

See [`CONDITION_SCOPE.md`](CONDITION_SCOPE.md) for the critical PROXIMAL vs NON-PROXIMAL
condition distinction — required reading before implementing any LLM layer.

See [`REWARD_NAMESPACE.md`](REWARD_NAMESPACE.md) for the complete eval namespace reference
covering every name available in goal condition expressions.

## Project Structure

```
factorio-agent/
├── agent/
│   ├── state_machine.py        ✅ AgentState, ExamineMode, assert_valid_transition()
│   └── examiner/
│       └── audit_report.py     ✅ AuditReport, BoundingBox, BlueprintCandidate
├── bridge/
│   ├── actions.py              ✅ 21 action types, ActionCategory, actions_for_context()
│   ├── rcon_client.py          ✅ RCON TCP connection, reconnect, thread-safe
│   ├── state_parser.py         ✅ JSON from Lua mod → WorldState snapshot;
│   │                                two-lane belts, inserter objects, charted_chunks;
│   │                                rebuilds WorldState indices after each parse
│   ├── action_executor.py      ✅ Action objects → RCON commands
│   └── mod/
│       ├── info.json           ✅ Factorio mod metadata (factorio_version: "2.0")
│       ├── control.lua         ✅ Lua mod — two-lane belt reading, inserter pickup/drop
│       │                             positions, charted_chunks via get_chart_size,
│       │                             fa.get_exploration(), prototype query functions
│       └── test_bridge_live.lua✅ In-game test suite (requires running game)
├── planning/
│   ├── goal.py                 ✅ Goal, RewardSpec, Priority, GoalStatus, make_goal()
│   ├── goal_tree.py            ✅ GoalTree — LIFO preemption, subgoal auto-completion
│   ├── reward_evaluator.py     ✅ RewardEvaluator — receives WorldQuery; full condition
│   │                                namespace including wq composable builder,
│   │                                ProductionTracker integration, staleness guards
│   └── resource_allocator.py   ✅ ResourceAllocator — pass-through (biters pending)
├── world/
│   ├── state.py                ✅ WorldState — pure data container with internal indices;
│   │                                BeltLane/BeltSegment (two-lane), InserterState
│   │                                (pickup/drop positions), ExplorationState
│   │                                (charted_chunks); indices rebuilt by __post_init__
│   │                                and maintained by WorldWriter
│   ├── query.py                ✅ WorldQuery — sole read interface for WorldState;
│   │                                named lookups (entity_by_id, entities_by_name,
│   │                                entities_by_status, entities_by_recipe,
│   │                                inserters_taking_from, inserters_delivering_to,
│   │                                resources_of_type, inventory_count, staleness);
│   │                                composable EntityQuery builder
│   │                                (.with_name/.with_recipe/.with_status/
│   │                                 .with_inserter_input/.with_inserter_output)
│   ├── writer.py               ✅ WorldWriter — sole write interface for WorldState;
│   │                                integrate_snapshot() for bridge updates,
│   │                                fine-grained mutation for execution layer
│   ├── knowledge.py            ✅ KnowledgeBase — SQLite-backed, runtime-extensible
│   ├── entities.py             ✅ Facade: ResourceRegistry, get_entity_metadata()
│   ├── tech_tree.py            ✅ TechTree — KB-backed research graph queries
│   └── production_tracker.py   ✅ ProductionTrackerProtocol + ProductionTracker
│                                    PROXIMAL — scan-radius scoped; update() takes WorldQuery
├── data/
│   └── knowledge/
│       └── knowledge.db        (created at runtime — gitignored)
├── config.py                   ✅ All tunable parameters
├── CONDITION_SCOPE.md          ✅ PROXIMAL/NON-PROXIMAL reference — include in LLM convos
├── REWARD_NAMESPACE.md         ✅ Complete eval namespace reference
└── tests/
    ├── fixtures.py                          ✅ Shared helpers: MockRconClient,
    │                                            make_world_state(), make_world_query(),
    │                                            make_inventory_entity()
    ├── unit/agent/
    │   ├── test_examiner.py                 ✅ 25 tests
    │   └── test_state_machine.py            ✅ 9 tests
    ├── unit/bridge/
    │   ├── test_action_executor.py          ✅ 28 tests
    │   ├── test_actions.py                  ✅ 40 tests
    │   ├── test_rcon_client.py              ✅ 5 tests
    │   └── test_state_parser.py             ✅ ~50 tests
    ├── unit/planning/
    │   ├── test_goal_tree.py                ✅ ~40 tests
    │   ├── test_reward_evaluator.py         ✅ ~30 tests (WorldQuery interface)
    │   └── test_resource_allocator.py       ✅ ~10 tests
    ├── unit/world/
    │   ├── test_state.py                    ✅ ~120 tests — WorldState indices (Section 2),
    │   │                                        WorldQuery (Section 3), WorldWriter (Section 4),
    │   │                                        plus all original dataclass tests
    │   ├── test_knowledge.py                ✅ 75 tests
    │   ├── test_entities.py                 ✅ 36 tests
    │   ├── test_tech_tree.py                ✅ 64 tests
    │   └── test_production_tracker.py       ✅ ~25 tests (WorldQuery interface)
    └── integration/
        ├── test_evaluator_capabilities.py   ✅ ~91 tests — capability matrix
        └── test_StateParser_WorldState.py   ✅ 9 tests (WorldQuery interface)
```

## Running the Tests

### Unit tests — no Factorio required

```bash
# Bridge
python -m unittest tests.unit.bridge.test_state_parser
python -m unittest tests.unit.bridge.test_action_executor

# World model (includes WorldQuery and WorldWriter)
python -m unittest tests.unit.world.test_state
python -m unittest tests.unit.world.test_knowledge
python -m unittest tests.unit.world.test_entities
python -m unittest tests.unit.world.test_tech_tree
python -m unittest tests.unit.world.test_production_tracker

# Planning layer
python -m unittest tests.unit.planning.test_goal_tree
python -m unittest tests.unit.planning.test_reward_evaluator
python -m unittest tests.unit.planning.test_resource_allocator

# Everything at once
python -m unittest discover -s tests -v
```

### Integration tests — no Factorio required

```bash
# Capability matrix — evaluator × WorldQuery
python -m unittest tests.integration.test_evaluator_capabilities

# StateParser → WorldQuery
python -m unittest tests.integration.test_StateParser_WorldState
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

**WorldState is accessed only through WorldQuery and WorldWriter.** `WorldState` is a
pure data container with internal indices. All reads go through `WorldQuery`; all
mutations go through `WorldWriter`. This boundary isolates consumers from the backing
implementation — the storage format can be changed without touching any consumer.

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

**Condition scope matters.** Proximal conditions (entities, logistics, production rates,
connectivity) are only reliable when the player is within scan radius. Non-proximal
conditions (inventory, research, exploration, time, resource patches) are globally accurate.
Write strategic goals using non-proximal conditions. See `CONDITION_SCOPE.md`.

**Belt lanes are independent.** Each belt tile has two transport lines. `BeltSegment` models
both via `BeltLane` objects with per-lane item dicts and congestion flags.

**Inserter connectivity is spatial.** `WorldQuery.inserters_taking_from(entity_id)` and
`inserters_delivering_to(entity_id)` use pickup/drop world coordinates for bounding-box
containment checks, enabling connectivity queries without additional RCON calls.

**Exploration is non-proximal.** `charted_chunks` reflects the force's global chart —
monotonically increasing, accurate anywhere. Use it in goals to incentivise exploration.

**Knowledge is queryable, not just retrievable.** The SQLite backing store supports
cross-domain queries the planning layer needs: recipes producing X, recipes running in
entity Y, techs unlocking a given recipe, recursive production chain closure.

**Placeholders degrade gracefully.** Unknown names encountered offline are stored as
placeholders and enriched automatically when a live connection becomes available.

**Actions are primitive and categorised.** `bridge/actions.py` contains only single-RCON-
call operations. Composition is the execution layer's job. Every action declares an
`ActionCategory`; `actions_for_context()` returns the valid set for the current situation.

**Biters are off by default.** All COMBAT and biter-related code exists but is gated behind
`BITERS_ENABLED`. Enabling it switches the threat stub for the real monitor without
structural changes elsewhere.

## Passing Context for the Next Phase

When implementing the next component:

1. Include `ARCHITECTURE.md` — the full design and all interface specifications
2. Include `CONDITION_SCOPE.md` — if the component generates or evaluates goal conditions
3. Include `REWARD_NAMESPACE.md` — if the component writes condition strings
4. Include the actual source files for everything the new component consumes or produces
5. Include the relevant test file(s) — the implementer should run them to verify changes
   don't break existing contracts
6. Write a component brief using the template in `ARCHITECTURE.md`

For the **primitives and execution layer** (next phase), the files to include are:
- `ARCHITECTURE.md`
- `EXECUTION_BRIEF.md` — the component brief for this phase
- `bridge/actions.py` — the Action types primitives must produce
- `world/state.py` — WorldState data container
- `world/query.py` — WorldQuery, the read interface the execution layer uses
- `world/writer.py` — WorldWriter, the write interface the execution layer uses
- `planning/goal.py` — Goal the execution layer pursues
- `agent/state_machine.py` — execution runs within EXECUTING state
- `world/knowledge.py` — KnowledgeBase the execution layer queries
- `config.py` — scan radii and other tunable parameters