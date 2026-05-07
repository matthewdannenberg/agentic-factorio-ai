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
│
└── tests/
    ├── mock_game_state.py           # Fake WorldState — no Factorio needed
    ├── test_core_dataclasses.py     # 128 tests covering all core types
    ├── test_bridge.py
    ├── test_planning.py
    ├── test_examiner.py
    └── test_goal_tree.py
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
2. **Bridge** — RCON client + Lua mod + state parser + action executor
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
- [ ] Bridge / Lua mod
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