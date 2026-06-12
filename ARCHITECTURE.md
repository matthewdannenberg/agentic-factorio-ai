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
replaced without touching consumers. This governs the world state layer
(WorldQuery/WorldWriter), the agent protocol (AgentProtocol), and every other
component boundary. Concrete implementations are injected at startup; swapping them
requires substituting the injected object, not restructuring surrounding code.

### Unified planning stack

Goals and Tasks are both `PlanningItem` subclasses and live on a single unified
stack (`coordinator._stack: list[PlanningItem]`). The top of the stack is always
the current unit of work: if it is a `Task`, tick its agent; if it is a `Goal`,
run its handler. This replaces the previous split between a goal stack and an
active-task slot, and allows the failure pipeline (sub-goal failure, diagnosis,
retry) to be expressed naturally as stack operations.

### Agents own tasks, not goals

Goals are the coordinator's concern. The coordinator translates a Goal into a
sequence of Tasks and routes each Task to the appropriate agent via `agent_hint`.
Agents interact with Tasks only — they never receive or inspect the Goal object.
This keeps agents focused on concrete executable work and lets the coordinator
change its decomposition strategy without touching agent implementations.

### Single active agent per task

At any given time exactly one agent owns the active task (the topmost Task on the
stack). The coordinator selects the owning agent by matching `agent_hint` against
each agent's `AGENT_ID`. Only that agent is ticked until the task resolves.

### Four timescales

- **Per-tick** — action selection, observation, blackboard reads/writes
- **Per-task** — locally-derived prerequisite tasks, blackboard lifecycle
- **Per-goal** — goal lifecycle, examination, behavioral memory recording
- **Per-run / cross-run** — policy checkpointing, KB and behavioral memory persistence

### Three knowledge layers

- **Game knowledge** (`world/knowledge/`) — what the game contains. Recipes,
  entities, tech tree. Learned from Factorio at runtime, persists across runs.
- **Factory self-model** (`world/model/`) — what the agent has built in the
  current run. A layered graph structure built incrementally during execution,
  starts empty each run.
- **Behavioral memory** (`memory/`) — what the agent has learned to do across
  runs. Strategy records, performance history, and spatial pattern foundations.
  Persists and grows across runs.

---

## Directory Layout

```
factorio-agent/
├── bridge/          # RCON boundary — nothing outside speaks RCON or knows Lua
│   └── mod/         # All Factorio mod files — must be added to Factorio as a local mod
├── world/
│   ├── observable/  # WorldState (data), WorldQuery (read), WorldWriter (write)
│   ├── knowledge/   # KnowledgeBase, TechTree, ProductionTracker
│   └── model/       # SelfModel, FactoryGraph, ChunkGrid, types
├── planning/
│   ├── planning_item.py  # PlanningItem base class, ItemStatus
│   ├── goals/            # Goal, GoalTree, GoalSource, GoalQueueEntry
│   ├── tasks/            # Task, TaskLedger, TaskRecord
│   └── evaluation/       # RewardEvaluator, condition_namespace, condition_parser
├── execution/
│   ├── agents/      # NavigationAgent, MiningAgent, ExplorationAgent, CraftingAgent
│   ├── skills/      # NavigateSkill, MineSkill, DestroySkill, CraftSkill, ...
│   ├── coordinator/ # RuleBasedCoordinator, StubCoordinator
│   ├── predicates.py    # Pure world-state observation predicates
│   └── preconditions.py # Richer pre-action checks (KB + inventory)
├── examination/     # AuditReport, auditor stubs (grows in Phase 10)
├── memory/          # Cross-run behavioral memory (may gain more in future phases)
├── llm/             # GoalSource, GoalQueue (LLM client in Phase 11)
├── data/            # Runtime databases (gitignored)
├── config.py
├── run.py
├── tests/           # Unit and integration tests
└── tests_in_game/   # In-game integration tests
```

---

## Component Reference

### Bridge (`bridge/`)

Hard interface boundary. Produces `WorldState` snapshots via `StateParser`,
dispatches `Action` objects via `ActionExecutor`. Nothing outside `bridge/`
speaks RCON or inspects Lua tables directly.

Action types (24 total across 10 categories): movement, mining, crafting,
building (`RotateEntity`, `SetSplitterPriority`, `FlipEntity`), inventory,
research, player, vehicle, combat, and meta. Circuit networks and railroad
scheduling are explicitly deferred — see `bridge/actions.py`.

The Factorio mod (`bridge/mod/control.lua`) exposes persistent-state movement
and mining driven by `on_tick` handlers. Key Factorio 2.x adaptations:

- `game.entity_prototypes` → `prototypes.entity`
- `LuaForce.get_chart_size()` removed → `surface.get_chunks()` + `force.is_chunk_charted()`
- `request_path` collision mask: `{layers={object=true, water_tile=true}}` —
  `object` covers trees/walls/buildings; `water_tile` covers water
- `movement_status` in `_player_table()`: `"idle"`, `"pathing"`, `"walking"`, `"unreachable"`
- Natural objects (trees, rocks) have no unit number in Factorio 2.x, so their
  `entity_id` is 0. This is an implementation detail of the bridge layer; code
  above the bridge should use `predicates.is_present()` and
  `predicates.is_reachable()` rather than inspecting `entity_id` directly.

---

### World (`world/`)

#### Observable layer (`world/observable/`)

`WorldState` is a pure data container — a `@dataclass` with no methods beyond
index maintenance. `WorldQuery` is the sole read interface; `WorldWriter` is
the sole write interface. External code never touches `WorldState` fields
directly.

Notable fields and additions:

- `WorldState.natural_objects: list[NaturalObject]` — scanned from neutral-force,
  non-resource entities in the scan radius. Used by `MiningAgent` for obstacle
  clearing. Identified by `force="neutral"` and `prototype_type != "resource"`,
  not by name — no hardcoded entity names.
- `NaturalObject.is_minable` — reflects whether the Lua mod returned a non-zero
  entity_id. **Do not use this field as the sole authority for whether an object
  can be cleared.** In Factorio 2.x trees have no unit number (entity_id=0) so
  `is_minable` returns False even though they are perfectly minable. Use
  `can_destroy(obj, kb)` from `execution/predicates.py` instead, which consults
  the KnowledgeBase record.
- `EntityState.force` and `EntityState.prototype_type` — populated from
  the Lua prototype table, used for factory-intersection detection in the coordinator.
- `ExplorationState.nearby_uncharted_chunks: list[ChunkCoord]` — adjacent uncharted
  chunks from the current scan, giving the exploration agent a concrete frontier
  target without requiring full map iteration.
- `CraftingQueueEntry` and `PlayerState.crafting_queue` — used by `CraftSkill`
  to confirm that a crafting order was accepted.
- `PlayerState.reachable: set[int]` — entity IDs currently within game-engine
  reach, as reported by the Lua mod. Used by `predicates.is_reachable()`.

**Known gap (Phase 10):** WorldState has no scan-coverage map. There is no record
of which sub-regions of a bbox have ever been within a scan radius, and no mechanism
to generate navigation targets to fill coverage gaps. Any PROXIMAL query over a
region larger than the scan radius is unreliable without prior navigation. A scan
coverage layer should be added to WorldState in Phase 10.

#### Knowledge layer (`world/knowledge/`)

`KnowledgeBase` stores game content learned at runtime via an injected
`query_fn`. Persists across runs in SQLite. No hard-coded Factorio names.

`EntityRecord` includes `minable: bool` — populated from
`proto.mineable_properties.minable` at prototype query time. Used by
`can_destroy()` in `execution/predicates.py` to determine whether a natural
object can be cleared with `MineEntity`. See the cliff gap note in
`execution/coordinator/coordinator.py` for the known limitation around
technology-dependent minability changes.

#### Model layer (`world/model/`)

The factory self-model is a layered container. `SelfModel` owns the layers and
routes `SelfModelPatch` objects to the appropriate one.

**`FactoryGraph`** (`world/model/layers/factory_graph.py`) — a directed graph of
`FactoryNode` objects representing logical factory components (production lines,
resource sites, power grids, etc.) connected by `FactoryEdge` objects representing
material flow.

**`ChunkGrid`** (`world/model/layers/chunk_grid.py`) — spatial index of explored
terrain. Stub implementation pending fuller integration.

**`SelfModelPatch`** — the mutation protocol. Agents never write to `SelfModel`
directly; they emit `SelfModelPatch` objects via `pending_patches()`. The
coordinator collects and applies them each tick via `SelfModel.apply(patch)`.

---

### Planning (`planning/`)

#### PlanningItem, Goal, and Task

**`PlanningItem`** (`planning/planning_item.py`) — base class for both `Goal` and
`Task`. Carries: `id`, `description`, `success_condition`, `failure_condition`,
`status` (unified `ItemStatus` enum), `created_at`, `resolved_at`, `parent_id`.
All lifecycle transitions live here: `activate`, `suspend`, `complete`, `fail`,
`escalate`. `parent_id` points to any `PlanningItem` — Goal or Task — supporting
the failure pipeline.

**`Goal`** (`planning/goals/goal.py`) — a strategic or tactical planning objective.
Adds `priority`, `reward_spec`, `step` (coordinator handler step counter), and
`context` (handler carry-along state). `GoalStatus` is a backward-compatibility
alias for `ItemStatus`. The `step` and `context` fields replace the former
`GoalFrame` dataclass — they are now first-class fields on `Goal` itself.

**`Task`** (`planning/tasks/task.py`) — a concrete work item routed to a specific
agent. Adds `agent_hint`, `task_type`, `params: dict`, and `derived_locally`.
`TaskStatus` is a backward-compatibility alias for `ItemStatus`. The constructor
accepts deprecated `parent_goal_id` and `parent_task_id` kwargs and resolves them
to `parent_id` for backward compatibility with existing call sites.

**`GoalTree`** (`planning/goals/goal_tree.py`) — runtime manager for the goal
hierarchy, including priority-based preemption and parent/child completion
propagation. Currently not wired into the execution path; reserved for the Phase 11
LLM layer.

**`TaskLedger`** (`planning/tasks/task_ledger.py`) — live stack + history log for
task lifecycle. Populated when tasks pop off the unified coordinator stack.

#### GoalQueueEntry and GoalQueue

**`GoalQueueEntry`** (`planning/goals/goal_source.py`) — serialisable
representation of a queued planning item. Has a `goal_type` field; `to_goal()`
constructs a `Goal` for coordinator goal handlers.

`condition_parser` extracts structured params (bbox coords, item names, navigate
targets) from `success_condition` strings at goal activation time, populating
`goal_params` passed to `coordinator.reset()`.

#### Condition namespace and evaluator

**`build_core_namespace(wq, tick, start_tick, start_wq)`** in
`planning/evaluation/condition_namespace.py` returns the `eval()` namespace for
all condition strings. Includes `inventory`, `charted_chunks`, `charted_area_km2`,
`tick`, `elapsed_ticks`, `new` (delta view), `wq`, `state`, `research`,
`tech_unlocked`, `resources_of_type`, `entities`, `entity_by_id`, `bbox`,
`navigate_to`, and safe builtins.

`navigate_to(x, y)` — True when player is within 1.5 tiles of (x, y). Used as
the `success_condition` for navigate tasks.

`bbox(x_min, y_min, x_max, y_max)` — returns a `BBoxQuery` with an `.is_clear`
property (True when `wq.natural_objects_in_bbox(...)` is empty). PROXIMAL —
must be staleness-guarded for goals larger than the scan radius.

**`RewardEvaluator`** (`planning/evaluation/reward_evaluator.py`) — evaluates
`success_condition` and `failure_condition` strings each tick via `eval()` against
the condition namespace. Also evaluates task conditions via `eval_condition()`,
called by the coordinator.

---

### Execution (`execution/`)

#### Predicates and Preconditions

**`execution/predicates.py`** — pure, cheap observation predicates over
`WorldQuery` only. No KnowledgeBase. No side effects. Key predicates:

- `is_at(target, wq, tolerance)` — True if player is within tolerance tiles of target.
- `is_reachable(entity_id, wq)` — True if entity_id is in `wq.state.player.reachable`
  (the actual game-engine reach set from the Lua mod, not a hardcoded distance).
- `can_mine(entity_id, wq)` — True if entity is present in scan AND reachable.
- `is_present(entity_id, position, wq, name, natural_radius)` — True if the target
  still exists. For `entity_id != 0`: uses entity scan. For `entity_id == 0`
  (natural objects without a unit number): proximity scan of `wq.natural_objects`.
- `can_destroy(obj, kb)` — True if KB confirms the natural object is minable.
  **This is the authoritative check for natural object clearability** — do not use
  `NaturalObject.is_minable` directly.
- `player_has_item(item, count, wq)` — True if player inventory has at least count.

**`execution/preconditions.py`** — richer pre-action checks that may involve
KnowledgeBase, inventory arithmetic, or recipe lookups. Called before committing
to a task, not in tight tick loops.

#### Skills (`execution/skills/`)

Skills encode multi-step, well-determined action sequences with built-in
success/failure detection. They contain **no decision logic** — that belongs to
agents and the coordinator. Skills delegate world-state questions to predicates.

- `NavigateSkill` — walks the player to a target position or entity.
- `MineSkill` — gathers resources from a resource patch.
- `DestroySkill` — deconstructs a single entity. Uses `predicates.is_present()`
  for target-gone detection (handles `entity_id=0` natural objects correctly) and
  `predicates.is_reachable()` for reach checks. Exposes `is_target_present(wq)`
  and `is_target_reachable(wq)` for agents.
- `CraftSkill` — hand-crafts items.

#### Agents (`execution/agents/`)

Agents read `WorldQuery` and call skills. They do not inspect Factorio
implementation details directly (entity_id values, unit numbers, etc.) — those
concerns belong in predicates.py and skills.

Current agents: `NavigationAgent`, `MiningAgent`, `ExplorationAgent`, `CraftingAgent`.

#### Coordinator (`execution/coordinator/`)

**Unified planning stack.** The coordinator maintains `_stack: list[PlanningItem]`
(replaces the former `_goal_stack: list[GoalFrame]` + `_active_task` split). The
top of the stack is the current unit of work:

- **Task on top**: tick its agent, evaluate task success/failure, pop on resolution.
- **Goal on top**: run its handler, which may push Tasks or sub-Goals onto the stack.

**Goal handlers** — one method per goal type. Each handler is called repeatedly
across ticks; `goal.step` (a field on `Goal`) tracks which decision point the
handler has reached. Handlers push sub-goals via `_push_goal()` or tasks via
`_push_task()`.

**`TASK_GOAL_TYPES`** — a module-level dict mapping goal_type strings that bypass
the handler machinery and push a `Task` directly. Currently: `"navigate"` →
`("navigation", "navigate_to")`. Adding a new task-backed goal type requires one
line here. The routing decision lives in the coordinator because it is
execution-layer knowledge, not planning-layer knowledge.

**`_dispatch_as_task(frame, wq, tick)`** — handles goal types in `TASK_GOAL_TYPES`.
Step 0 calls `_push_task`; step 1 completes the goal frame when the task resolves.
`condition_parser` extracts structured params (including a `Position` object for
navigate targets) from the condition string before the coordinator sees them.

**`_bbox_empty_condition(bbox)`** — generates the task `success_condition` for
`clear_region` goals. Checks `wq.natural_objects_in_bbox(...)` (not
`state.entities`) with a staleness guard on the `natural_objects` section.

**Goal type vocabulary:**

| Type | Description |
|---|---|
| `navigate` | Walk to a position — task-backed, bypasses goal handler |
| `collection` | Gather N of a resource from known patches |
| `acquire` | Get N of an item by any means (mine → produce) |
| `crafting` | Hand-craft N of an item |
| `exploration` | Chart N new chunks |
| `clear_region` | Remove natural obstacles from a bounding box |
| `prep_region` | Factory-aware region clearing (checks for infrastructure, belts) |
| `construction` | Build infrastructure in a region |
| `production` | Establish item production at a rate (Phase 8 stub) |
| `logistics` | Connect factory nodes with belts/inserters (Phase 9 stub) |
| `byproduct` | Route/consume all output from a node (Phase 8 stub) |
| `research` | Unlock a technology |
| `noop` | Idle — ask LLM for the next goal (Phase 11 stub) |

**Known capability gaps** (documented in `coordinator.py`):

- *Wood/harvest gap* — items obtained by clearing natural objects (e.g. wood
  from trees) cannot be acquired directly. Fix requires `EntityRecord.mining_products`,
  `kb.entities_that_produce(item)`, and a `harvest_natural` task type (Phase 7).
- *Cliff gap* — undestroyable objects (cliffs) cause `clear_region` to fail.
  Fix requires technology-change detection for `minable`, `UseItemOnEntity`
  bridge action, and coordinator retry after unlock (Phase 7/10).
- *Scan coverage gap* — `clear_region` and other PROXIMAL bbox queries are
  unreliable without prior navigation across the full bbox. A scan coverage map
  is needed in WorldState (Phase 10). Current bandaid: navigate to bbox corners
  before issuing clear_region goals.

#### Blackboard (`execution/blackboard.py`)

Shared working memory for the coordinator and agents. Entries have a category
(`INTENTION`, `OBSERVATION`, `RESERVATION`) and a scope (`GOAL` or `TASK`).
`TASK`-scoped entries are cleared when the active task resolves.
`GOAL`-scoped entries persist until the goal completes.

---

### Examination (`examination/`)

`AuditReport` — a data container for mechanical and rich examination results.
Carries anomalies, production rates, power state, damage records, and (in rich
mode) blueprint candidates and LLM observations.

Richer examination (self-model reconciliation, blackboard promotion) is Phase 10.

---

### Memory (`memory/`)

`SQLiteBehavioralMemory` — stores strategy records (what worked for a goal type),
performance history (per-goal-type statistics), and spatial pattern metadata
(factory subgraph summaries). Persists across runs.

---

## Data Flow

### Per-tick

1. Bridge produces a `WorldState` snapshot. `WorldWriter.integrate_snapshot()`
   merges it into live global state.

2. `RewardEvaluator` checks active goal `success_condition` and `failure_condition`
   against `WorldQuery` and `start_wq` snapshot. If triggered, main loop transitions.

3. `coordinator.tick(wq, ww, tick)` is called:
   - If top of `_stack` is a `Task`: evaluate task conditions, tick the owning
     agent, collect `SelfModelPatch` objects. Pop on resolution.
   - If top of `_stack` is a `Goal`: run its handler, which may push Tasks or
     sub-Goals. Completed/failed items propagate upward.

4. Main loop dispatches returned actions to `bridge/action_executor.py`.

5. `coordinator.drain_patches()` returns accumulated self-model patches; the
   loop applies them to `SelfModel`.

### Per-task

**Push:** `_push_task(...)` constructs a `Task`, sets it ACTIVE, looks up the
agent by `agent_hint`, calls `agent.activate(task, bb, wq, kb)`, and appends the
task to `_stack`.

**Tick:** `agent.tick(task, ...)` called each poll cycle. Agent runs skills and
returns actions.

**Completion:** coordinator evaluates `task.success_condition` via `_eval`. On
True: pops the task, calls `_bb.clear_scope(EntryScope.TASK)`, advances the
parent goal's `step`.

**Failure/Escalation:** `failure_condition` fires → task popped → parent goal's
`status` set to `ItemStatus.FAILED` → propagates upward to root → `STUCK`.

### Per-goal

**Initiation:** `coordinator.reset(goal_type, params, wq)` called. Stack and
blackboard cleared. A `Goal` with `goal_type` and `params` is pushed as ACTIVE.
Loop snapshots `WorldQuery` as `start_wq` for `new` delta conditions.

**Resolution:** `RewardEvaluator` detects success or failure. Behavioral memory
records outcome. Blackboard cleared. `GoalSource` requests next goal.

---

## Phase Plan

### Phases 1–6 ✅

Full end-to-end pipeline implemented: bridge, world model, planning, execution
layer (coordinator, agents, skills), examination stub, behavioral memory.
Rule-based coordinator handles collection, exploration, clearing, crafting,
and research goals. All unit tests pass.

### Phase 6.5 ✅ 

Major architectural refactors completed:

- **Unified planning stack**: `GoalFrame` eliminated; `Goal` carries `step` and
  `context` directly; `_goal_stack + _active_task` replaced by `_stack: list[PlanningItem]`.
- **`PlanningItem` base class**: `Goal` and `Task` share a unified lifecycle enum
  (`ItemStatus`), `parent_id` linkage, and lifecycle methods.
- **`navigate` goal type**: Task-backed goal type that walks the player to a
  position. Used by in-game tests for scan coverage before `clear_region`.
- **`clear_region` working end-to-end**: trees/rocks now cleared correctly.
  Fixed: `is_minable` guard removed from clear_natural path; `_bbox_empty_condition`
  now checks `natural_objects_in_bbox`; `DestroySkill` uses `predicates.is_present`
  and `predicates.is_reachable` instead of reimplementing game mechanics.
- **Predicate layer**: `is_present` added to `predicates.py`. Skills delegate all
  world-state questions to predicates rather than implementing them inline.
- **`condition_parser`**: extracts structured params (`bbox`, `navigate_to` coords)
  from condition strings; produces `Position` objects for navigate targets.
- **`TASK_GOAL_TYPES`**: routing table in coordinator for task-backed goal types.

### Phase 7 — Construction agent (RL)

The construction agent places entities, connects inserters and belts, and
verifies that placed infrastructure is functional. Also planned for Phase 7:
`UseItemOnEntity` bridge action and `harvest_natural` task type (wood gap fix).

### Phase 8 — Production agent

First learned agent. The `production` and `byproduct` coordinator stubs become real.

### Phase 9 — Spatial-logistics agent

Spatial reasoning and logistics: belt routing, inserter placement, layout.
The `logistics` coordinator stub becomes real.

### Phase 10 — Examination layer revision + WorldState refactor

- Examination gains self-model reconciliation, blackboard promotion, structured
  factory summaries.
- `RewardEvaluator` namespace extended with STRUCTURAL conditions (OD-6).
- **WorldState scan coverage map** — track which regions have been observed;
  generate navigation targets for unobserved sub-regions of a bbox query.
- Technology-change detection (cliff gap fix).

### Phase 11 — LLM layer revision

Real LLM client replacing `GoalQueue`. Receives self-model summary and behavioral
memory statistics. The `noop` coordinator stub becomes real.

### Deferred

- Biter defense — combat agents, threat module activation
- Blueprint system — curation of spatial patterns from behavioral memory
- Coordinator learning — transition from rule-based to learned coordination
- Railroad networks — train state, scheduling, train-station agent
- Circuit networks — wire connections, combinator conditions
- Ocean-aware `ChunkGrid` — adjacency edges only where walking is possible

---

## Open Implementation Decisions

See `OPEN_DECISIONS.md` for full discussion.

- **OD-1** — Observation space construction
- **OD-2** — Coordinator learning
- **OD-3** — Self-model cross-run persistence
- **OD-4** — RL algorithm family
- **OD-5** — Spatial-logistics internal structure
- **OD-6** — RewardEvaluator namespace extension for self-model (STRUCTURAL conditions)
- **OD-7** — NodeType-specific FactoryNode subclasses
- **OD-8** — Agent architecture: thin protocol vs behavioral composition
- **OD-9** — WorldState dialect: translate Factorio internals into a convenient
  agent-facing representation rather than exposing raw game concepts (entity_id=0,
  unit numbers, etc.) to the planning and execution layers.

---

## Passing Context for the Next Phase

When opening a new conversation to implement a component, include:

1. `ARCHITECTURE.md` — full or relevant sections
2. `CONDITION_SCOPE.md` — if the component generates or evaluates goal conditions
3. `REWARD_NAMESPACE.md` — if the component writes or interprets condition strings
4. `OPEN_DECISIONS.md` — if the component touches any open decision
5. The actual source files for all interfaces the component must satisfy or consume
6. The relevant test files