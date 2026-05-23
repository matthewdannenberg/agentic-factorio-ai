# Factorio Agent — Architecture Document

## Project Goal

Build a multi-agent reinforcement learning system capable of playing Factorio
autonomously. The system receives coarse strategic goals from an LLM, decomposes
them internally through a learned agent network, executes them via direct game
interaction, and accumulates knowledge across runs.

The system has two distinguishing design commitments:

**Learning over programming.** The agent begins each fresh install knowing nothing
about Factorio's content beyond the schema needed to store what it learns. Game
mechanics, production relationships, spatial strategies, and logistics patterns all
emerge from runtime experience. Hard-coded Factorio knowledge is a code smell.

**Execution intelligence is local.** The LLM handles what learning handles poorly:
long-horizon strategic reasoning over novel situations. Everything below coarse goal
setting — decomposition, planning, spatial reasoning, logistics, navigation — is the
agent network's responsibility. LLM calls are expensive, infrequent, and narrowly
scoped.

Target game version: **Factorio 2.x (Space Age)**.

---

## Core Design Principles

### Clean segmentation

Every major component exposes a narrow, versioned interface. Implementations can be
replaced without touching consumers. This principle governs the world state layer
(WorldQuery/WorldWriter), the agent protocol (AgentProtocol), and every other
component boundary.

When a component's interface is defined, it should be defined as a Protocol or
abstract base. The concrete implementation is injected at startup. Swapping
implementations — including wholesale replacement of the RL system — requires
substituting the injected object, not restructuring surrounding code.

### Agents own subtasks, not goals

Goals are the coordinator's concern. The coordinator translates a Goal into Subtasks
and routes each Subtask to the appropriate agent via `agent_hint`. Agents interact
with Subtasks only — they never receive or inspect the Goal object. This boundary
keeps agents focused on concrete executable work and lets the coordinator change
its decomposition strategy without touching agent implementations.

### Single active agent per subtask

At any given time exactly one agent owns the active subtask. The coordinator selects
the owning agent when a subtask is pushed (by matching `agent_hint` against each
agent's `AGENT_ID` class attribute). Only that agent is ticked until the subtask
resolves. This replaces the earlier poll-all approach and correctly reflects that
subtasks have clear single owners.

### Four timescales

The system operates at four distinct timescales, each with its own data flow and
component responsibilities:

- **Per-tick** — action selection, observation, blackboard reads/writes
- **Per-subtask** — locally-derived prerequisite tasks, blackboard lifecycle
- **Per-goal** — goal lifecycle, examination, behavioral memory recording
- **Per-run / cross-run** — policy checkpointing, KB and behavioral memory persistence

### Three knowledge layers

- **Game knowledge** (`world/knowledge.py`) — what the game contains. Recipes,
  entities, tech tree. Learned from Factorio at runtime, persists across runs.
- **Factory self-model** (`agent/self_model.py`) — what the agent has built in the
  current run. A graph of logical factory units and their relationships. Built
  incrementally during execution, starts empty each run.
- **Behavioral memory** (`agent/memory/behavioral.py`) — what the agent has learned
  to do across runs. Strategy records, performance history, and spatial pattern
  foundations. Persists and grows across runs.

---

## Component Inventory

### Stable — Phases 1-4 complete

**Bridge layer** (`bridge/`)
Hard interface boundary. Nothing outside speaks RCON or knows Lua. Produces
WorldState snapshots via StateParser, dispatches Action objects via ActionExecutor.

Action types (24 total across 10 categories): movement, mining, crafting, building
(including `RotateEntity`, `SetSplitterPriority`, `FlipEntity`), inventory,
research, player, vehicle, combat, and meta. Circuit networks and railroad
scheduling operations are explicitly deferred — see `bridge/actions.py`.

**World state layer** (`world/`)
`WorldState` is a pure data container. `WorldQuery` is the sole read interface.
`WorldWriter` is the sole write interface. `KnowledgeBase` stores game knowledge
learned at runtime via injected `query_fn`. `TechTree`, `ProductionTracker` unchanged.

**Planning layer** (`planning/`)
`GoalTree`, `RewardEvaluator`, `ResourceAllocator`. Goal lifecycle management and
reward evaluation against WorldQuery. `RewardEvaluator` evaluates condition strings
against a namespace built by `planning/condition_namespace.py` — see below.

---

### Phase 5 — Execution layer foundation ✅

**ExecutionLayerProtocol** (`agent/execution_protocol.py`)

The sole interface between the planning layer and the agent execution network.

```python
class ExecutionStatus(Enum):
    PROGRESSING = auto()
    WAITING     = auto()
    STUCK       = auto()
    COMPLETE    = auto()

@dataclass
class ExecutionResult:
    actions: list[Action]
    status: ExecutionStatus
    stuck_context: Optional[StuckContext] = None

class ExecutionLayerProtocol:
    def reset(self, goal: Goal, wq: WorldQuery,
              seed_subtasks: Optional[list[Subtask]] = None) -> None: ...
    def tick(self, goal: Goal, wq: WorldQuery, ww: WorldWriter,
             tick: int) -> ExecutionResult: ...
    def progress(self, goal: Goal, wq: WorldQuery) -> float: ...
    def observe(self, wq: WorldQuery) -> dict: ...
```

**StuckContext**, **Blackboard**, **SubtaskLedger**, **SelfModel**, **BehavioralMemory**
— all implemented and stable. See Phase 5 documentation for full detail.

---

### Phase 6 — Navigation, mining, rule-based coordinator, run loop ✅

#### AgentProtocol (revised)

Agents receive **Subtasks**, not Goals. All four methods take `subtask` as their
first parameter:

```python
class AgentProtocol:
    def activate(self, subtask: Subtask, blackboard: Blackboard,
                 wq: WorldQuery) -> None: ...
    def tick(self, subtask: Subtask, blackboard: Blackboard,
             wq: WorldQuery, ww: WorldWriter, tick: int) -> list[Action]: ...
    def observe(self, subtask: Subtask, blackboard: Blackboard,
                wq: WorldQuery) -> dict: ...
    def progress(self, subtask: Subtask, blackboard: Blackboard,
                 wq: WorldQuery) -> float: ...
```

`activate()` is called by the coordinator each time a new subtask is assigned —
not once per goal at reset time. Each new subtask triggers a fresh `activate()`
call, giving the agent a clean slate and precise knowledge of what it's working on.

#### AgentRegistry (revised)

Agents are registered directly without goal-type annotations:

```python
registry.register(nav_agent)
registry.register(mine_agent)
```

Selection is via `registry.agent_by_id(hint)` where `hint` matches the agent's
`AGENT_ID` class attribute. The registry has no knowledge of goals or subtask types.

#### NavigationAgent (`agent/network/agents/navigation.py`)

Rule-based movement agent. Sole responsibility: walk the player to a target
position or entity. Issues `MoveTo` once per waypoint, relying on the Factorio
mod's `on_tick` handler to drive `walking_state` continuously between RCON polls.

Suppresses redundant `MoveTo` commands within `_REDUNDANT_THRESHOLD` tiles.
Stall detection fires after `_STALL_GRACE_TICKS` (full grace) or
`_UNREACHABLE_GRACE_TICKS` (30 poll ticks, used when the player has not moved
at all since activation — fast detection of pathfinder "unreachable" results).
On stall, writes a `navigation_stalled` OBSERVATION to the blackboard; the
coordinator reads this and escalates immediately rather than waiting for the
subtask failure timeout.

Arrival is detected by evaluating the subtask's `success_condition` directly
(via the coordinator's condition namespace). No side-channel blackboard signal.

#### MiningAgent (`agent/network/agents/mining.py`)

Handles resource gathering and obstacle clearing. Two subtask types:

- `gather_resource` — mine N of a named resource from a known patch. Issues
  `MineResource` once and re-issues only on inventory stall after a grace period.
- `clear_region` — remove all (or all natural) entities from a bounding box.
  Handles its own local positioning within the box. Natural objects identified
  by entity name heuristics (tree/rock/cliff/fish) — KB-free.

#### RuleBasedCoordinator (`agent/network/coordinator.py`)

Manages SubtaskLedger, Blackboard, AgentRegistry, and SelfModel. Key behaviours:

**Goal type routing** — derivable: `collection`, `exploration`, `clear_region`.
Returns STUCK immediately for all other types.

**Subtask boundary** — the coordinator is the sole consumer of Goals. It
constructs Subtasks and hands them to agents. The `agent_hint` dynamic attribute
on each Subtask drives `registry.agent_by_id()` selection.

**Subtask activation** — `activate(subtask, ...)` is called on the selected agent
at each subtask transition, not at goal reset time.

**Success evaluation** — arrival and completion detected by evaluating
`subtask.success_condition` against `_build_condition_namespace()`, which calls
`build_core_namespace()` from `planning/condition_namespace.py` and adds
positional predicates (`is_at`, `is_reachable`, `Position`). Navigation stall
detected via `navigation_stalled` blackboard observation — escalates immediately
without waiting for timeout.

**Exploration resilience** — exploration subtask stalls return `WAITING` (not
`STUCK`) so the goal stays alive. The coordinator re-derives a new waypoint in
a rotated cardinal direction, tracked by `_exploration_attempt`. Waypoints are
absolute positions from world origin sized to push past the charted frontier.

**Collection derivation** — two subtasks: approach (navigation) then gather
(mining). The gather subtask's success condition is an absolute inventory
threshold (`inventory('type') >= current + target`) rather than the goal's raw
condition string, so it works correctly with both `inventory()` and
`new.inventory()` goal conditions.

**Goal start snapshot** — the loop calls `coordinator.set_start_snapshot(start_wq)`
immediately after `reset()`, providing a `WorldQuery` snapshot of world state at
goal activation. This populates the `new` delta object in subtask condition
namespaces.

**Timeout** — `_SUBTASK_TIMEOUT_TICKS` (5 minutes) for general subtasks;
`_NAV_SUBTASK_TIMEOUT_TICKS` (30 seconds) for navigation subtasks.

#### Condition namespace (`planning/condition_namespace.py`)

Shared condition evaluation infrastructure used by both `RewardEvaluator` and
the coordinator's subtask evaluator.

`build_core_namespace(wq, tick, start_tick, start_wq)` returns the dict used by
`eval()` for condition strings. Contains: `inventory`, `charted_chunks`,
`charted_tiles`, `charted_area_km2`, `tick`, `elapsed_ticks`, `new`, `wq`,
`state`, `research`, `tech_unlocked`, `resources_of_type`, `entities`,
`entity_by_id`, and safe builtins.

`_DeltaView` is the object exposed as `new` in condition strings. It wraps the
current `WorldQuery` and a `start_wq` snapshot taken at goal activation:

```python
new.tick                    # ticks since goal activation (== elapsed_ticks)
new.charted_chunks          # chunks charted this goal
new.charted_tiles           # auto via __getattr__ — any scalar wq property
new.inventory('iron-ore')   # ore collected this goal
new.resource_patches('coal') # patches discovered this goal
```

`__getattr__` automatically delegates any scalar `WorldQuery` property — adding
a new `wq` property requires no changes to `_DeltaView`. Parameterised lookups
(`inventory`, `resource_patches`) have explicit methods.

`elapsed_ticks` is kept as a top-level alias for `new.tick` for readability in
timeout conditions (`elapsed_ticks > 1200`).

#### RewardEvaluator (`planning/reward_evaluator.py`)

Evaluates goal `success_condition` and `failure_condition` strings against a
`WorldQuery`. Calls `build_core_namespace()` for the shared core and adds
evaluator-specific entries: `production_rate`, `staleness`, `logistics`, `power`,
`threat`, `inserters_from/to`.

`evaluate(goal, wq, tick, start_tick, start_wq)` — `start_wq` is a `WorldQuery`
snapshot taken at goal activation, used to populate `new`. When `None`, `new`
has zero deltas.

**Avoid absolute `tick` values in conditions.** Use `elapsed_ticks > N` or
`new.tick > N` instead — absolute tick values cause conditions to fire
immediately if the game has been running before the test.

#### FactorioLoop (`agent/loop.py`)

Master tick loop implementing the four-timescale data flow. At goal activation,
calls `_snapshot_world_query()` which takes a `copy.copy(WorldState)` wrapped in
a `WorldQuery`. Shallow copy is safe because `WorldWriter.integrate_snapshot()`
replaces whole sub-objects rather than mutating them in place. The snapshot is
passed to `RewardEvaluator.evaluate()` and `coordinator.set_start_snapshot()`.

#### Factorio mod (`bridge/mod/control.lua`) — Factorio 2.x

Updated for Factorio 2.x throughout this phase:

- `game.entity_prototypes` → `prototypes.entity` (same for fluid/recipe)
- `LuaForce.get_chart_size()` removed → `surface.get_chunks()` +
  `force.is_chunk_charted(surface, chunk)` (pass chunk directly, not
  `chunk.position`)
- `tech.effects`/`tech.prerequisites` → `tech.prototype.effects/prerequisites`
- `proto.infinite_resource` replaces `mineable_properties.infinite`
- `request_path` collision mask: `{layers={object=true, water_tile=true},
  consider_tile_transitions=true}` — `object` layer covers trees/walls/buildings;
  `water_tile` covers water. Passing the prototype's `CollisionMask` userdata
  directly causes "unreachable" because the character prototype includes layers
  that mark ground positions as blocked.
- `movement_status` included in `_player_table()` output
  (`"idle"`, `"pathing"`, `"walking"`, `"unreachable"`)
- `StateParser` exposes `movement_status` property

**Lua-side test suite** — four test files loadable from the in-game console:

```
/c __agent__ T.run_all()     # test_bridge_live.lua — state queries, actions
/c __agent__ TM.run_all()    # test_movement_live.lua — movement, pathfinding
/c __agent__ TS.run_all()    # test_state_live.lua — exploration, mining status
/c __agent__ TP.run_all()    # test_prototypes_live.lua — KB query functions
```

Async movement tests use start/finish pairs:
```
/c __agent__ TM.async_start("player_actually_moves")
# wait 5 seconds
/c __agent__ TM.async_finish("player_actually_moves")
```

#### GoalSource and GoalQueue (`llm/goal_source.py`)

`GoalSource` is the abstract interface the loop uses to obtain goals. `GoalQueue`
is the Phase 6 implementation: an ordered list of `GoalQueueEntry` objects
dispensed one at a time.

`handle_stuck()` returns `[]` (no seed subtasks) — the real LLM client in Phase 11
will return decomposition subtasks here.

#### In-game test framework (`tests_in_game/`)

`tests_in_game/conftest.py` provides `run_goal`/`run_goals` fixtures that build a
complete component stack, run goals through `FactorioLoop`, and return
`(LoopStats, WorldQuery)`.

In-game tests use `new.` delta conditions and `elapsed_ticks`/`new.tick` for
timeouts so they are independent of absolute game clock:

```python
success_condition="new.inventory('iron-ore') >= 5"  # collected this goal
failure_condition="new.tick > 10800"                 # 3 minutes since activation
success_condition="new.charted_chunks >= 5"          # charted this goal
```

#### Phase 6 status (all resolved)

All Phase 6 in-game tests pass. Issues resolved during extended Phase 6 work:

- Collection/exploration test timing: replaced absolute `tick` thresholds with
  `elapsed_ticks`/`new.tick` relative conditions
- Exploration chunk counting: `get_chart_size` removed in 2.x; replaced with
  `get_chunks()`/`is_chunk_charted()` iterator
- Navigation pathfinding: Factorio 2.x `request_path` collision mask format
  corrected; player now routes around trees, walls, and water
- Exploration completeness: waypoints are now absolute coordinates from world
  origin sized to push past the charted frontier; exploration stalls rotate
  direction and continue rather than failing the goal
- `new.inventory()` delta conditions: collection goals using `new.inventory()`
  now parse correctly and the gather subtask uses an absolute threshold built
  from current inventory + target count

---

### Phase 7 — Construction agent (next)

The construction agent handles placing entities, connecting inserters and belts,
and verifying that placed infrastructure is functional. It is the first agent
that directly modifies the factory rather than gathering raw materials.

See `PRE_PHASE_7_DECISIONS.md` for the decisions that must be made, and
`PHASE_7_BRIEF.md` for the component brief once those decisions are recorded.

---

### Phase 8 — Production agent (RL)

First learned agent. Handles recipe selection, machine configuration, and
production chain management. Chosen first because production goals are the most
common and the KB production chain queries provide strong structural guidance.

---

### Phase 9 — Spatial-logistics agent (RL)

Spatial reasoning and logistics. Whether this becomes one agent or two coordinating
agents is an implementation decision for Phase 9 — see OD-5.

---

### Phase 10 — Examination layer revision

Examination gains self-model reconciliation, blackboard promotion, and structured
factory summaries. RewardEvaluator namespace extended with STRUCTURAL conditions
backed by the self-model — see OD-6.

---

### Phase 11 — LLM layer revision

Real LLM client replaces GoalQueue. Receives self-model summary and behavioral
memory statistics. Handles escalation via StuckContext.

---

### Deferred — no phase assigned

- Biter defense — combat agents, threat module activation
- Blueprint system — curation of spatial patterns from behavioral memory
- Coordinator learning — transition from rule-based to learned coordination
- Railroad networks — train state, scheduling, train-station agent
- Circuit networks — wire connections, combinator conditions

---

## Data Flow

### Per-tick

1. Bridge produces a fresh WorldState snapshot. `WorldWriter.integrate_snapshot()`
   merges it into live global state.

2. `RewardEvaluator` checks active goal success/failure conditions against
   `WorldQuery` and `start_wq` snapshot. If triggered, main loop transitions state.

3. `coordinator.tick()` is called. Inside:
   - Coordinator evaluates active subtask success/failure conditions using
     `_build_condition_namespace()` (which includes `new`, `elapsed_ticks`)
   - Checks for `navigation_stalled` observation → immediate escalation
   - On success: calls `complete(tick)`, pops subtask, selects and activates
     agent for next subtask, writes blackboard intention entry
   - `_tick_active_agent()` calls `agent.tick(subtask, ...)` on the single
     owning agent
   - Actions returned, ExecutionResult assembled

4. Main loop dispatches `ExecutionResult.actions` to `bridge/action_executor.py`.

5. `ExecutionResult.status` is inspected:
   - `PROGRESSING` / `WAITING` — continue
   - `STUCK` — pass `StuckContext` to GoalSource for decomposition
   - `COMPLETE` — trigger examination (Phase 10)

### Per-subtask

**Push:** coordinator derives a subtask, sets `agent_hint`, stores metadata as
dynamic attributes (e.g. `target_position`, `resource_type`), pushes to ledger,
calls `agent.activate(subtask, ...)` on the selected agent, writes blackboard
INTENTION entry.

**Tick:** `agent.tick(subtask, ...)` called each poll cycle. Agent reads INTENTION
entries (waypoints, mining tasks) and returns actions.

**Completion:** coordinator evaluates `subtask.success_condition` against its
namespace. On True: calls `complete(tick)`, pops ledger, clears subtask-scoped
blackboard entries, selects and activates agent for next subtask.

**Failure/Escalation:** `failure_condition` fires → `fail(tick)` → escalation
sequence → STUCK returned to main loop. For exploration goals, escalation returns
WAITING instead — the coordinator re-derives a new waypoint in a rotated direction.

### Per-goal

**Initiation:** `coordinator.reset(goal, wq)` called. Blackboard and ledger
cleared. Loop calls `_snapshot_world_query()` and passes result to both
`coordinator.set_start_snapshot()` and the evaluator. Derivation starts from
scratch (or from `seed_subtasks` if post-escalation).

**Resolution:** `RewardEvaluator` detects success or failure. Behavioral memory
records outcome. Blackboard cleared. GoalSource requests next goal.

---

## Open Implementation Decisions

See `OPEN_DECISIONS.md` for full discussion of each.

- **OD-1** — Observation space construction
- **OD-2** — Coordinator learning
- **OD-3** — Self-model cross-run persistence
- **OD-4** — RL algorithm family
- **OD-5** — Spatial-logistics internal structure
- **OD-6** — RewardEvaluator namespace extension for self-model (STRUCTURAL conditions)
- **OD-7** — NodeType-specific SelfModelNode subclasses
- **OD-8** — Construction agent design (new — see PRE_PHASE_7_DECISIONS.md)

---

## Project Structure

```
factorio-agent/
│
├── bridge/                          # Hard boundary — nothing outside speaks RCON
│   ├── actions.py                   ✅ 24 action types across 10 categories
│   ├── rcon_client.py               ✅
│   ├── state_parser.py              ✅ movement_status exposed on StateParser
│   ├── action_executor.py           ✅
│   └── mod/
│       ├── info.json                ✅
│       ├── control.lua              ✅ persistent movement + mining; 2.x fixes
│       ├── test_bridge_live.lua     ✅ global T — state/action/edge-case suites
│       ├── test_movement_live.lua   ✅ global TM — pathfinding, obstacle routing
│       ├── test_state_live.lua      ✅ global TS — exploration, mining status
│       └── test_prototypes_live.lua ✅ global TP — all KB query functions
│
├── world/                           # Pure data and computation
│   ├── state.py                     ✅ WorldState pure data container
│   ├── query.py                     ✅ WorldQuery — sole read interface
│   ├── writer.py                    ✅ WorldWriter — sole write interface
│   ├── knowledge.py                 ✅ KnowledgeBase — SQLite-backed, query_fn injected
│   ├── entities.py                  ✅
│   ├── tech_tree.py                 ✅
│   └── production_tracker.py        ✅
│
├── planning/                        # Goal lifecycle and reward evaluation
│   ├── goal.py                      ✅
│   ├── goal_tree.py                 ✅
│   ├── condition_namespace.py       ✅ _DeltaView, build_core_namespace, safe_builtins
│   ├── reward_evaluator.py          ✅ uses build_core_namespace; start_wq snapshot
│   └── resource_allocator.py        ✅
│
├── agent/
│   ├── execution_protocol.py        ✅ ExecutionLayerProtocol, ExecutionResult,
│   │                                   ExecutionStatus, StuckContext
│   ├── blackboard.py                ✅ Blackboard, BlackboardEntry,
│   │                                   EntryCategory, EntryScope
│   ├── subtask.py                   ✅ Subtask, SubtaskStatus, SubtaskRecord,
│   │                                   SubtaskLedger
│   ├── self_model.py                ✅ SelfModel, SelfModelProtocol,
│   │                                   node/edge types, BoundingBox
│   ├── preconditions.py             ✅ is_at, is_reachable, can_reach_count,
│   │                                   can_place, can_mine, valid_actions
│   ├── loop.py                      ✅ FactorioLoop, LoopConfig, LoopStats;
│   │                                   _snapshot_world_query() at goal activation
│   ├── observation.py               [ Phase 7 ] Goal-conditioned observation
│   │                                            constructors
│   ├── reward.py                    [ Phase 7 ] Dense intermediate reward signals
│   ├── state_machine.py             ✅
│   │
│   ├── network/
│   │   ├── agent_protocol.py        ✅ AgentProtocol — subtask-first signatures
│   │   ├── coordinator.py           ✅ RuleBasedCoordinator + StubCoordinator;
│   │   │                               navigation_stalled escalation;
│   │   │                               exploration direction rotation;
│   │   │                               set_start_snapshot(start_wq)
│   │   ├── registry.py              ✅ AgentRegistry — no goal-type coupling
│   │   └── agents/
│   │       ├── navigation.py        ✅ NavigationAgent — movement only;
│   │       │                           navigation_stalled observation;
│   │       │                           _UNREACHABLE_GRACE_TICKS fast detection
│   │       ├── mining.py            ✅ MiningAgent — gathering + clearing
│   │       ├── construction.py      [ Phase 7 ] Construction agent
│   │       ├── production.py        [ Phase 8 ] RL production agent
│   │       └── spatial_logistics.py [ Phase 9 ] RL spatial-logistics agent
│   │
│   ├── memory/
│   │   └── behavioral.py            ✅ BehavioralMemoryProtocol +
│   │                                   SQLiteBehavioralMemory stub
│   │
│   └── examiner/
│       ├── audit_report.py          ✅
│       ├── rich_examiner.py         [ Phase 10 ] Revised with self-model
│       └── mechanical_auditor.py    [ Phase 10 ]
│
├── llm/
│   ├── goal_source.py               ✅ GoalSource protocol, GoalQueue,
│   │                                   GoalQueueEntry (with bbox + clear_mode)
│   ├── client.py                    [ Phase 11 ]
│   ├── budget.py                    [ Phase 11 ]
│   └── prompts/
│       ├── strategic.md             [ Phase 11 ]
│       ├── escalation.md            [ Phase 11 ]
│       ├── examination.md           [ Phase 10 ]
│       └── reflection.md            [ Phase 11 ]
│
├── data/
│   ├── knowledge/
│   │   └── knowledge.db             (runtime, gitignored)
│   └── behavioral.db                (runtime, gitignored)
│
├── config.py                        ✅
├── run.py                           ✅ CLI entry point
├── ARCHITECTURE.md                  (this file)
├── CONDITION_SCOPE.md               ✅
├── REWARD_NAMESPACE.md              ✅
├── OPEN_DECISIONS.md                ✅
├── PHASE_7_BRIEF.md                 (to be written after PRE_PHASE_7_DECISIONS)
├── PRE_PHASE_7_DECISIONS.md         ✅
│
├── tests/
│   ├── fixtures.py                  ✅
│   ├── unit/
│   │   ├── bridge/                  ✅
│   │   ├── world/                   ✅
│   │   ├── planning/                ✅ test_condition_namespace.py (new)
│   │   ├── agent/                   ✅ Phase 6 complete
│   │   └── llm/                     ✅ test_goal_source.py
│   └── integration/                 ✅ test_evaluator_capabilities.py
│
└── tests_in_game/
    ├── conftest.py                  ✅ rcon_client, knowledge_base, run_goal fixtures
    ├── README.md                    ✅
    ├── 01_knowledge/                ✅ recipe and tech discovery
    ├── 02_collection/               ✅ new.inventory() delta conditions
    └── 03_exploration/              ✅ new.charted_chunks() delta conditions
```

---

## Build Order and Testing Strategy

Each phase produces interfaces before implementations, and tests before or alongside
code. No phase begins until the previous phase's tests pass. In-game tests for the
current phase are the final gate before moving to the next.

**Phase 6** — first runnable system. In-game tests verify collection, exploration,
and clearing against a live Factorio instance. Subtask-first agent protocol
established. Rule-based coordinator handles three goal types. All in-game tests
fully passing including delta conditions and Factorio 2.x compatibility.

**Phase 7** — construction agent. The first agent that modifies the factory
structure. Design is being specified before implementation begins. See
`PRE_PHASE_7_DECISIONS.md` and `PHASE_7_BRIEF.md`.

**Phase 8** — production agent (RL). First learned agent. KB production chain
queries provide strong structural guidance for subtask derivation.

**Phase 9** — spatial-logistics agent (RL). Layout, routing, inserter placement.
Internal structure (one agent or two) decided at phase start — see OD-5.

**Phases 10-11** — examination and LLM layers. Full loop closure.

---

## Passing Context for the Next Phase

When opening a new conversation to implement a component, include:

1. `ARCHITECTURE.md` — full or relevant sections
2. `CONDITION_SCOPE.md` — if the component generates or evaluates goal conditions
3. `REWARD_NAMESPACE.md` — if the component writes or interprets condition strings
4. `OPEN_DECISIONS.md` — if the component touches any open decision
5. `PRE_PHASE_7_DECISIONS.md` — decisions recorded before Phase 7
6. `PHASE_7_BRIEF.md` — the component brief for the phase
7. The actual source files for all interfaces the component must satisfy or consume
8. The relevant test files