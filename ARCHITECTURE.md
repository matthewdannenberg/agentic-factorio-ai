# Factorio Agent — Architecture Document

## Project Goal

Build an agentic system capable of playing Factorio autonomously. The system should set its
own long-term goals, develop short-term plans to pursue them, execute those plans, and learn
from outcomes across runs. The paradigm is reinforcement learning with minimal seeding — the
agent should figure out how to play rather than being told.

The system must demonstrate two distinct capabilities eventually:
- **Factory building**: long-term planning, production chain reasoning, logistics
- **Biter defense**: priority interruption, emergency response, competing resource demands

Biters are disabled initially. The architecture must support enabling them without structural
rewrites.

Target game version: **Factorio 2.x (Space Age)**. The Lua mod and all API assumptions are
written against the 2.x runtime API. The mod is not compatible with 1.x.

---

## Key Design Decisions

### LLM Calls Are Expensive — Treat Them As Such

All LLM interaction routes through `llm/client.py`. No other module calls the API directly.
The execution layer is pure code. The LLM is only invoked when genuine reasoning is needed:

- Strategic layer: every ~5-10 minutes of game time
- Tactical layer: on subgoal completion, failure, or unexpected state change
- Examination layer: on goal completion, rate-limit recovery, or periodic trigger
- Never during primitive execution

Target: 10-30 LLM calls per real-world hour of play. If rate limited, the system transitions
to mechanical examination mode rather than halting.

### The Bridge Is a Hard Interface Boundary

Nothing outside `bridge/` speaks RCON or knows about Lua. The rest of the system operates
on `WorldState` objects and structured `Action` objects exclusively. This means the full
agent stack can be tested against a mock game state without Factorio running.

### Rewards Are Self-Designed

When the strategic LLM sets a goal, it simultaneously produces a `RewardSpec` — a structured
definition of success conditions, failure conditions, milestone rewards, and time discounting.
The `RewardSpec` is evaluated mechanically by `planning/reward_evaluator.py` against live
game state. No LLM is involved in reward evaluation during execution.

After goal resolution, a reflection call reviews whether the reward spec was well-calibrated.
That reflection is stored in memory and informs future goal-setting.

The only externally fixed reward is a meta-reward (e.g. "launch the rocket", "survive N
hours") that the agent cannot redefine. Everything beneath that is agent-designed.

### Biter Support Via Stub Pattern

`agent/threat/stub.py` is a drop-in no-op that satisfies the same interface as
`agent/threat/monitor.py`. Config flag `BITERS_ENABLED` determines which is loaded.

The following are built from day one even though they go unused until biters are enabled:
- `Priority` enum in `planning/goal.py` (EMERGENCY / URGENT / NORMAL / BACKGROUND)
- Goal preemption and resumption in `planning/goal_tree.py`
- `planning/resource_allocator.py` (trivial pass-through until resource contention exists)
- `ActionCategory.COMBAT` and `ActionCategory.VEHICLE` in `bridge/actions.py`
- Combat action stubs: `ShootAt`, `SelectWeapon`, `StopShooting`
- Vehicle action stubs: `EnterVehicle`, `ExitVehicle`, `DriveVehicle`

### Self-Examination Is a First-Class State

`EXAMINING` is a peer state in the state machine, not a fallback. It has two modes:

- **Rich examination** (LLM available): full reflection, bottleneck identification, blueprint
  curation trigger, goal validity check
- **Mechanical examination** (LLM unavailable): pure-code audit of throughput, idle entities,
  starved inputs, power headroom, belt congestion

Both modes produce an `AuditReport`. When the LLM becomes available after rate limiting, it
receives the accumulated audit report as context before resuming planning. The agent is never
blind on resumption.

### Blueprints Are Agent-Discovered, Not Pre-Seeded

`memory/blueprint_library/library.json` starts empty. The agent builds factories ad-hoc,
and the rich examiner extracts reusable designs when production metrics cross a performance
threshold. Blueprints accumulate across runs, representing genuine learned competence.

The nomination/evaluation split is explicit:
- `BlueprintCandidate` in `audit_report.py` is a **nomination** — the examiner flags a region
- `BlueprintRecord` in `memory/blueprint_library/` is an **evaluation** — stored performance
  data, tech tier, improvability estimate. Not yet implemented; designed when memory layer
  is built.

### Resource Types Are Strings, Not Enums

Resource names are plain strings matching Factorio's internal item names (`"iron-ore"`,
`"crude-oil"`, etc.). A `ResourceName` class provides named constants for vanilla resources
but is not exhaustive. The `ResourceRegistry` in `world/entities.py` is the authoritative
store of known resource metadata and is extended at runtime when the bridge parser encounters
an unfamiliar name. This makes the system robust to Space Age and modded content without
code changes.

### WorldState Is a Belief State, Not Ground Truth

`WorldState` is a cached, partially-observed snapshot. Different sections have different
staleness characteristics. Staleness is tracked per-section via `observed_at: dict[str, int]`
(section name → game tick of last bridge update). The `section_staleness(section, tick)`
method returns ticks since last observation, or `None` if never observed.

Sections and their typical freshness:
- `player`, `logistics.power` — refreshed every tick cycle
- `entities` near player — refreshed on local scan
- `entities` far from player — may be many ticks stale
- `resource_map` — updated only when agent physically visits a patch
- `ground_items` — populated within local scan radius only
- `damaged_entities` — refreshed on each bridge scan
- `destroyed_entities` — rolling window, pruned by bridge
- `threat` — only meaningful when `BITERS_ENABLED=True`

### Structural Damage Is Cause-Agnostic

`DamagedEntity` and `DestroyedEntity` live on `WorldState` directly, not inside `ThreatState`.
They are populated regardless of cause — biter attack, vehicle collision, player error,
deconstruction. `ThreatState` is strictly biter-specific. `DestroyedEntity.cause` is a
string with known values: `"biter"` | `"vehicle"` | `"deconstruct"` | `"unknown"`.

### Actions Are Categorised and Context-Filtered

Every `Action` subclass declares an `ActionCategory`. The function `actions_for_context(
in_vehicle, biters_enabled)` returns the valid action subset for the current execution
context. `VEHICLE` actions are only valid while in a vehicle; `MOVEMENT` actions are only
valid on foot; `COMBAT` actions are only emitted when `BITERS_ENABLED=True`.

### Circuit Networks Are Explicitly Deferred

Factorio's circuit network (wire connections, combinator logic, signal vocabularies, memory
cells) is a full embedded programming language. It is intentionally out of scope for the
initial implementation. The agent can play a complete game — including Space Age content and
biter defence — without circuit networks. Adding them later means new `Action` subclasses
and Lua mod extensions; no existing code changes.

### Primitive vs Composite Actions

`bridge/actions.py` contains only **primitive** actions — single atomic operations executed
in one RCON round-trip. Multi-step sequences (walk to patch → mine N ore → return to base)
are composed by `agent/primitives/` and `agent/execution.py`. Nothing in `actions.py`
sequences or loops.

### Entity Scanning Is Mod-Compatible by Design

The Lua mod's entity scanner uses Factorio's `unit_number` property rather than a whitelist
of entity type strings. `unit_number` is assigned by the engine to all persistent,
player-interactable entities — assemblers, inserters, poles, chests, modded machines, turrets,
trains — and is not assigned to trees, rocks, resources, decoratives, projectiles, or other
cosmetic/transient objects. This means any building added by any mod is automatically visible
to the agent without code changes. Player characters (which also have unit numbers) are
excluded by an explicit type check.

---

## Project Structure

```
factorio-agent/
│
├── bridge/                          # Hard interface boundary — nothing outside speaks RCON
│   ├── actions.py                   # Action dataclasses — primitive commands to the bridge
│   ├── rcon_client.py               # RCON connection, reconnection logic
│   ├── state_parser.py              # Raw Lua output → WorldState
│   ├── action_executor.py           # Action objects → RCON commands
│   └── mod/
│       ├── info.json                # Factorio mod metadata (factorio_version: "2.0")
│       └── control.lua              # Lua mod — exposes state, accepts commands
│
├── world/                           # Pure data — no LLM, no RCON
│   ├── state.py                     # WorldState and all sub-dataclasses
│   ├── entities.py                  # Entity type definitions, ResourceRegistry
│   ├── tech_tree.py                 # Research graph, unlock dependencies
│   └── production_tracker.py        # Throughput over time — feeds reward + curation
│
├── agent/
│   ├── loop.py                      # Master orchestration
│   ├── state_machine.py             # PLANNING / EXECUTING / EXAMINING / WAITING
│   │
│   ├── strategic.py                 # Long-horizon goal setting (LLM)
│   ├── tactical.py                  # Subgoal decomposition (LLM)
│   ├── execution.py                 # Task execution (pure code)
│   │
│   ├── examiner/
│   │   ├── rich_examiner.py         # LLM-available: reflection + curation trigger
│   │   ├── mechanical_auditor.py    # LLM-unavailable: pure-code health checks
│   │   └── audit_report.py          # Shared dataclass — passed to LLM on resumption
│   │
│   ├── threat/
│   │   ├── monitor.py               # Real threat detection (biters-on)
│   │   ├── defense_planner.py       # Defensive goal generation (biters-on)
│   │   └── stub.py                  # Drop-in no-op (biters-off default)
│   │
│   └── primitives/                  # Atomic operations composed by execution layer
│       ├── movement.py
│       ├── crafting.py
│       ├── building.py
│       ├── mining.py
│       └── blueprint.py             # Apply blueprint string at location
│
├── planning/
│   ├── goal.py                      # Goal, RewardSpec, Priority enum, make_goal()
│   ├── goal_tree.py                 # Hierarchy, preemption, suspension, resumption
│   ├── reward_evaluator.py          # Mechanical RewardSpec evaluation vs WorldState
│   └── resource_allocator.py        # Priority-weighted allocation (trivial until biters)
│
├── memory/
│   ├── episodic.py                  # Per-run summaries
│   ├── semantic.py                  # Factorio mechanics the agent has learned
│   ├── plan_library.py              # Successful plan templates + metadata
│   └── blueprint_library/
│       ├── library.json             # Empty at start, agent-populated across runs
│       └── blueprint_curator.py     # Extracts + annotates blueprints from examination
│
├── llm/
│   ├── client.py                    # Single API chokepoint — all calls go here
│   ├── budget.py                    # Rate limit detection, pause logic, call tracking
│   └── prompts/
│       ├── strategic.md
│       ├── tactical.md
│       ├── examination.md
│       └── reflection.md
│
├── config.py                        # All tunable parameters
│                                    # BITERS_ENABLED, LLM_CALL_BUDGET, TICK_INTERVAL
│                                    # RCON_HOST/PORT/PASSWORD, scan radii
│
└── tests/
    ├── unit/
    │   └── bridge/
    │       ├── test_rcon_client.py
    │       ├── test_state_parser.py  # 80 tests — parser, partial updates, edge cases
    │       └── test_action_executor.py
    ├── mock_game_state.py            # Fake WorldState — no Factorio needed
    ├── test_core_dataclasses.py      # 128 tests covering all core types
    └── integration/
        └── test_bridge_live.lua      # In-game console test suite (requires running game)
```

---

## Core Dataclasses

These are the interfaces everything else must satisfy. All are implemented and tested.

### WorldState (`world/state.py`)

A belief state snapshot of the game, not ground truth. Produced by `bridge/state_parser.py`.

Key fields:
- `tick`: game tick (60 ticks = 1 second)
- `observed_at`: `dict[str, int]` — per-section staleness tracking
- `player`: `PlayerState` — position, health, inventory, reachable entity ids
- `entities`: `list[EntityState]` — placed buildings, status, recipe, energy
- `resource_map`: `list[ResourcePatch]` — coarse strategic resource records
- `ground_items`: `list[GroundItem]` — items on the ground within scan radius
- `research`: `ResearchState` — unlocked techs, queue, science rates
- `logistics`: `LogisticsState` — belt segments, power grid, inserter activity
- `damaged_entities`: `list[DamagedEntity]` — entities below full health (any cause)
- `destroyed_entities`: `list[DestroyedEntity]` — rolling window of destroyed entities
- `destroyed_ttl_ticks`: pruning window for destroyed_entities (default 18 000)
- `threat`: `ThreatState` — biter bases, pollution, attack timers (empty when biters off)

Key methods: `entity_by_id()`, `entities_by_name()`, `entities_by_status()`,
`resources_of_type()`, `inventory_count()`, `section_staleness()`, `has_damage`,
`recent_losses`, `game_time_seconds`.

Sub-types: `Position` (frozen), `Direction` (enum), `EntityStatus` (enum),
`ResourceName` (string constants), `ResourceType = str`, `InventorySlot`, `Inventory`,
`EntityState`, `ResourcePatch`, `GroundItem`, `ResearchState`, `BeltSegment`,
`PowerGrid`, `LogisticsState`, `DamagedEntity`, `DestroyedEntity`, `BiterBase`,
`ThreatState`, `PlayerState`.

### Goal and RewardSpec (`planning/goal.py`)

Produced by the strategic or tactical LLM. Contains its own `RewardSpec`.

Goal key fields:
- `id`: UUID4 string (auto-generated)
- `description`: human-readable
- `priority`: `Priority` IntEnum — BACKGROUND(0) / NORMAL(1) / URGENT(2) / EMERGENCY(3)
- `success_condition`: evaluable expression string against WorldState
- `failure_condition`: evaluable expression string against WorldState
- `reward_spec`: `RewardSpec` instance
- `parent_id`: for subgoals (str | None)
- `status`: `GoalStatus` — PENDING / ACTIVE / SUSPENDED / COMPLETE / FAILED
- `created_at`, `resolved_at`: game ticks

Goal lifecycle: `activate(tick)` → `suspend()` → `activate(tick)` → `complete(tick)` or
`fail(tick)`. Illegal transitions raise `RuntimeError`.

RewardSpec key fields:
- `success_reward`, `failure_penalty`: floats
- `milestone_rewards`: `dict[str, float]` — condition expression → partial reward
- `time_discount`: float in (0.0, 1.0] — reward decay per tick (1.0 = no decay)
- `calibration_notes`: str — LLM's forward assessment for reflection call

Factory function: `make_goal(description, success_condition, failure_condition, ...)` —
flat constructor that handles RewardSpec nesting.

### AuditReport (`agent/examiner/audit_report.py`)

Produced by either examiner mode. Passed to LLM as context on resumption after rate
limiting. `merge()` combines reports accumulated during blackout periods.

Key fields:
- `tick`, `mode` (RICH | MECHANICAL), `observation_ticks`
- `starved_entities`: `list[StarvedEntity]` — machines with missing inputs
- `idle_entities`: `list[IdleEntity]` — non-working machines
- `power_headroom_kw`, `power_satisfaction`
- `belt_congestion`: `list[CongestionSegment]`
- `production_rates`: `dict[str, float]` — item → items/minute
- `anomalies`: `list[Anomaly]` — severity-sorted (CRITICAL / WARNING / INFO)
- `damaged_entities`: `list[DamagedEntityRecord]` — cause-agnostic, not biter-gated
- `destroyed_entities`: `list[DestroyedEntityRecord]` — cause-agnostic, not biter-gated
- `blueprint_candidates`: `list[BlueprintCandidate]` — rich mode only; nominations only
- `llm_observations`: str — rich mode only

`BoundingBox` (frozen): `min_x, min_y, max_x, max_y` (int tile coords). Matches Factorio's
blueprint capture API. Properties: `width`, `height`, `area`, `center_x/y`, `contains()`.

`BlueprintCandidate`: nomination only — `region_description`, `bounds: BoundingBox`,
`performance_metric`, `rationale`. Evaluation (throughput/tile, tech tier, improvability)
lives in `BlueprintRecord` in the memory layer (not yet implemented).

### Action types (`bridge/actions.py`)

Frozen dataclasses. Every action declares an `ActionCategory`. 21 concrete types across
10 categories.

Categories and types:
- `MOVEMENT`: `MoveTo`, `StopMovement`
- `MINING`: `MineResource`, `MineEntity`
- `CRAFTING`: `CraftItem`
- `BUILDING`: `PlaceEntity`, `SetRecipe`, `SetFilter`, `ApplyBlueprint`
- `INVENTORY`: `TransferItems`
- `RESEARCH`: `SetResearchQueue`
- `PLAYER`: `EquipArmor`, `UseItem` (fish, capsules, grenades — target_position optional)
- `VEHICLE`: `EnterVehicle`, `ExitVehicle`, `DriveVehicle` (stub — unused until vehicles needed)
- `COMBAT`: `SelectWeapon`, `ShootAt`, `StopShooting` (stub — unused until `BITERS_ENABLED`)
- `META`: `Wait`, `NoOp`

`ShootAt` validates that exactly one of `target_entity_id` or `target_position` is provided.

`ACTIONS_BY_CATEGORY`: `dict[ActionCategory, tuple[type[Action], ...]]` — built at import.

`actions_for_context(in_vehicle=False, biters_enabled=False)` — returns valid action types
for the current execution context. Called by execution layer at goal-start and on context
change (board/exit vehicle, biter config toggle).

### AgentState (`agent/state_machine.py`)

Four states: `PLANNING`, `EXECUTING`, `EXAMINING`, `WAITING`.
`ExamineMode`: `RICH`, `MECHANICAL`.
`assert_valid_transition(from, to)` — raises `RuntimeError` on illegal moves.

Valid transitions:
- PLANNING → EXECUTING
- EXECUTING → EXAMINING, PLANNING (emergency preempt)
- EXAMINING → PLANNING, WAITING
- WAITING → EXAMINING

---

## Bridge Layer

### Overview

The bridge layer is the only part of the system that speaks RCON or knows about Lua. It has
three Python components and one embedded Lua component.

```
Python agent                          Factorio game process
─────────────────                     ─────────────────────
ActionExecutor  ──── RCON /c cmd ───► control.lua (fa.*)
StateParser     ◄─── JSON string ──── control.lua (fa.*)
RconClient          (TCP socket)
```

### `bridge/rcon_client.py` — RconClient

Manages the TCP connection to Factorio's RCON server. Implements the RCON binary protocol
(4-byte little-endian length-prefixed packets, type-3 auth, type-2 exec, type-0 response).
Thread-safe via a single lock. Reconnects with exponential backoff on transient failures.

Key methods: `connect()`, `send(command) → str`, `is_connected() → bool`, `close()`.
Also usable as a context manager. Raises `BridgeError` on unrecoverable failures.

### `bridge/state_parser.py` — StateParser

Translates JSON strings from the Lua mod into `WorldState` objects. Accepts either a full
snapshot or a single-section partial update.

Key design properties:
- Unknown resource names (e.g. mod resources) are registered in `ResourceRegistry` rather
  than rejected. The parser is the first point where mod resources enter the Python model.
- Every populated section stamps `WorldState.observed_at[section] = current_tick`. Absent
  sections are not stamped, preserving staleness information for consumers.
- `destroyed_entities` is a rolling window: new events from the Lua circular buffer are
  merged with the existing list, then pruned to `WorldState.destroyed_ttl_ticks`.
- `DestroyedEntity.cause` is normalised to one of the four canonical strings; anything
  unrecognised becomes `"unknown"`.
- All section parsers are defensively typed — wrong types and missing fields produce safe
  defaults, never exceptions.

Key methods:
- `parse(raw, current_tick) → WorldState` — full parse into a fresh WorldState.
- `parse_partial(raw, section, into) → WorldState` — merge one section into existing state.

### `bridge/action_executor.py` — ActionExecutor

Translates `Action` objects into RCON Lua commands. Dispatches on `action.kind` (the class
name string) via a class-level dictionary. One RCON call per action.

Failure modes:
- Returns `False` for recoverable failures (entity out of reach, item missing, etc.) — the
  Lua mod returns `{"ok": false, "reason": "..."}`.
- Raises `BridgeError` when the Lua side returns non-JSON (indicates a crash or mod error).
- Raises `NotImplementedError` for VEHICLE and COMBAT stub actions.

Does not validate contextual appropriateness — that is `actions_for_context()`'s job.

### `bridge/mod/control.lua` — Lua mod

Runs inside Factorio. Exposes a global `fa` table callable from RCON `/c` commands.

**State queries** (return JSON strings via `rcon.print()`):
- `fa.get_state(opts)` — full snapshot, all sections, drains destruction buffer
- `fa.get_player()`, `fa.get_entities(r)`, `fa.get_resource_map(r)` — individual sections
- `fa.get_ground_items(r)`, `fa.get_research()`, `fa.get_logistics(r)` — individual sections
- `fa.get_damaged_entities(r)`, `fa.drain_destruction_events()`, `fa.get_threat()` — individual sections
- `fa.get_tick()` — current game tick as a string

**Action commands** (return `{"ok":true}` or `{"ok":false,"reason":"..."}`):
- Movement: `fa.move_to(pos, pathfind)`, `fa.stop_movement()`
- Mining: `fa.mine_resource(pos, name, count)`, `fa.mine_entity(id)`
- Crafting: `fa.craft_item(recipe, count)`
- Building: `fa.place_entity(item, pos, dir)`, `fa.set_recipe(id, recipe)`,
  `fa.set_filter(id, slot, item)`, `fa.apply_blueprint(string, pos, dir, force)`
- Inventory: `fa.transfer_items(id, to_player, item, count)`
- Research: `fa.set_research_queue(techs)`
- Player: `fa.equip_armor(item)`, `fa.use_item(item, pos)`
- Stubs: `fa.enter_vehicle`, `fa.exit_vehicle`, `fa.drive_vehicle` (vehicle)
- Stubs: `fa.select_weapon`, `fa.shoot_at`, `fa.stop_shooting` (combat)

**Notable implementation details:**
- Entity scanning uses `unit_number` presence rather than a type whitelist. Any modded
  building is automatically included. Player characters are excluded by type check.
- `fa.move_to` supports all 8 compass directions including diagonals. Uses a 2:1 axis
  ratio threshold to choose between cardinal and diagonal movement.
- The destruction event circular buffer (512 entries) is populated by `on_entity_died`
  and drained atomically by `fa.get_state()` or `fa.drain_destruction_events()`.
- Power statistics use the 2.x `get_flow_count{name, category, precision_index, count}`
  API, summing across all producer and consumer prototypes in the network.
- Health ratios use `entity.health / entity.max_health` (2.x; `get_health_ratio()` removed).
- Research queue management uses `force.add_research()` / `force.cancel_current_research()`
  (2.x; direct table assignment removed).

**2.x API changes from 1.x** (summary for future reference):
- Logistic chest type names: `logistic-chest-*` → `passive-provider-chest`, etc.
- `player.reach_distance` → `player.character.reach_distance`
- `player.walking_state` → `player.character.walking_state`
- `entity.get_health_ratio()` → `entity.health / entity.max_health`
- `get_flow_count("output", true)` → `get_flow_count{name=..., category=..., ...}`
- `force.research_queue = {...}` → `force.add_research()` / `force.cancel_current_research()`
- `spawner.spawner_data.max_count` → `spawner.unit_count`
- `entity.get_recipe()` returns two values in 2.x; capture only the first

---

## State Machine

```
         ┌──────────────────────────────────────┐
         ▼                                      │
      PLANNING ──── LLM sets Goal + RewardSpec ─┘
         │
         ▼
      EXECUTING ──── pure code, no LLM
         │
    ┌────┴──────────────────────────┐
    │                               │
    ▼                               ▼
goal complete               rate limited / scheduled
    │                       anomaly detected
    ▼                               │
EXAMINING (rich)            EXAMINING (mechanical)
    │                               │
    ▼                               ▼
 PLANNING                       WAITING
                                    │
                            LLM available again
                                    │
                                    ▼
                            EXAMINING (rich)
                            with accumulated AuditReport
                                    │
                                    ▼
                                PLANNING
```

---

## Build Order

Each layer is independently testable before the next is built.

1. **Core dataclasses** ✅ — `WorldState`, `Goal`, `RewardSpec`, `AuditReport`, `Priority`,
   `Action` types, `AgentState` — 128 tests passing
2. **Bridge** ✅ — RCON client, Lua mod, state parser, action executor — 80 tests passing
   (unit tests against mock RCON; in-game integration tests pending live game)
3. **World** — `ResourceRegistry`, tech tree, production tracker
4. **Planning** — goal tree, reward evaluator, resource allocator (stub)
5. **Primitives** — movement, crafting, building, mining
6. **Execution layer** — composes primitives against active goal
7. **Examination** — mechanical auditor first, rich examiner second
8. **Tactical layer** — subgoal decomposition via LLM
9. **Strategic layer** — long-horizon goal setting via LLM
10. **Memory** — episodic, semantic, plan library, blueprint curation
11. **Main loop + state machine** — ties everything together
12. **Threat module** — replaces stub when `BITERS_ENABLED=True`

---

## Testing Strategy

### Unit tests (no Factorio required)

`tests/unit/bridge/test_state_parser.py` — 80 tests covering full parse, partial parse,
safe defaults, unknown resource registration, destroyed entity cause normalisation, TTL
pruning, and all WorldState accessor methods.

`tests/unit/bridge/test_action_executor.py` — round-trip tests asserting the correct Lua
command string is produced for each action type; recoverable failure returns False;
unrecoverable failure raises BridgeError; VEHICLE/COMBAT stubs raise NotImplementedError.

`tests/test_core_dataclasses.py` — 128 tests for WorldState, Goal, RewardSpec, AuditReport,
and all sub-types.

### In-game integration tests (requires live Factorio + mod)

`tests/integration/test_bridge_live.lua` — a Lua script loadable from the Factorio console
that exercises each `fa.*` function against the live game and prints PASS/FAIL results.
Tests are grouped into suites: state queries, action commands, and edge cases. Each test
is independently runnable. See the script's header for usage instructions.

The test script deliberately avoids relying on specific map state — it creates its own
known conditions (e.g. drops an item, then checks ground_items) and cleans up after itself.

---

## Component Conversation Brief Template

When opening a new conversation to implement a component, include:

1. `ARCHITECTURE.md` (this file) — full or relevant sections
2. `CONVERSATION_BRIEF.md` — the specific brief for this component (see below)
3. The actual source files for all interfaces the component must satisfy or consume
4. The test file — so the implementer can run and extend it

The brief template:

```
## Component: <name>

### What it receives
<input types, where they come from>

### What it must return
<output types>

### What it must NOT do
<e.g. no LLM calls, no RCON, no side effects>

### Adjacent interfaces it must satisfy
<list of files + the specific types/functions it must be compatible with>

### Files provided in this conversation
<list>

### What this conversation must produce
<list of files to create/modify>
```

---

## Current Status

- [x] Architecture designed
- [x] Core dataclasses (`world/state.py`, `planning/goal.py`,
      `agent/examiner/audit_report.py`, `agent/state_machine.py`,
      `bridge/actions.py`) — 128 tests
- [x] Bridge — `bridge/rcon_client.py`, `bridge/state_parser.py`,
      `bridge/action_executor.py`, `bridge/mod/control.lua`,
      `bridge/mod/info.json` — 80 unit tests; in-game integration tests pending
- [ ] World model (`entities.py` ResourceRegistry, `tech_tree.py`, `production_tracker.py`)
- [ ] Planning layer
- [ ] Primitives
- [ ] Execution layer
- [ ] Examination layer
- [ ] Tactical layer
- [ ] Strategic layer
- [ ] Memory layer
- [ ] Main loop
- [ ] Threat module