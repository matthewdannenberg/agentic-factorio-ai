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

### Agents own tasks, not goals

Goals are the coordinator's concern. The coordinator translates a Goal into a
sequence of Tasks and routes each Task to the appropriate agent via `agent_hint`.
Agents interact with Tasks only — they never receive or inspect the Goal object.
This keeps agents focused on concrete executable work and lets the coordinator
change its decomposition strategy without touching agent implementations.

### Single active agent per task

At any given time exactly one agent owns the active task. The coordinator selects
the owning agent by matching `agent_hint` against each agent's `AGENT_ID`. Only
that agent is ticked until the task resolves.

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
├── world/
│   ├── observable/  # WorldState (data), WorldQuery (read), WorldWriter (write)
│   ├── knowledge/   # KnowledgeBase, TechTree, ProductionTracker
│   └── model/       # SelfModel, FactoryGraph, ChunkGrid, types
├── planning/        
│   ├── goals/       # Goal, GoalTree, GoalSource
│   ├── tasks/       # Task, TaskLedger
│   └── evaluation/  # Reward_evaluator, condition_namespace
├── execution/
│   ├── agents/      # NavigationAgent, MiningAgent, ExplorationAgent, CraftingAgent
│   ├── skills/      # NavigateSkill, MineSkill, DestroySkill, CraftSkill, ...
│   ├── coordinator/ # RuleBasedCoordinator, GoalFrame, StubCoordinator
│   ├── memory/      # SQLiteBehavioralMemory
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

---

### World (`world/`)

#### Observable layer (`world/observable/`)

`WorldState` is a pure data container — a `@dataclass` with no methods beyond
index maintenance. `WorldQuery` is the sole read interface; `WorldWriter` is
the sole write interface. External code never touches `WorldState` fields
directly.

Notable additions over the original design:

- `WorldState.natural_objects: list[NaturalObject]` — scanned from neutral-force,
  non-resource entities in the scan radius. Used by `MiningAgent` for obstacle
  clearing. Identified by `force="neutral"` and `prototype_type != "resource"`,
  not by name — no hardcoded entity names.
- `NaturalObject.is_minable` — True when `entity_id != 0` (i.e. the object has
  a unit number and can be targeted by `MineEntity`).
- `EntityState.force` and `EntityState.prototype_type` — populated from
  the Lua prototype table, used for factory-intersection detection in the
  coordinator.
- `ExplorationState.nearby_uncharted_chunks: list[ChunkCoord]` — the bridge
  returns adjacent uncharted chunks from the current scan, giving the exploration
  agent a concrete frontier target without requiring full map iteration.
- `CraftingQueueEntry` and `PlayerState.crafting_queue` — used by `CraftSkill`
  to confirm that a crafting order was accepted.

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

**`FactoryGraph`** (`world/model/layers/factory_graph.py`) — the primary
coordination data structure. A directed graph of `FactoryNode` objects
representing logical factory components (production lines, resource sites,
power grids, etc.) connected by `FactoryEdge` objects representing material
flow. Nodes carry `node_type`, `process_type`, `status`, `bounding_box`,
`design_capacity`, and `throughput`. Edges carry `item`, `rate`, and
`transport` type.

**`ChunkGrid`** (`world/model/layers/chunk_grid.py`) — spatial index of
explored terrain. Stub implementation pending fuller integration with the
bridge's chunk scanning. Future note: geographically adjacent chunks should
be connected by an edge in `ChunkGrid` only if it is possible to walk directly
between them (ruling out ocean tiles). This adjacency model is not yet
implemented; the current stub treats all charted chunks as reachable.

**`SelfModelPatch`** — the mutation protocol. Agents never write to `SelfModel`
directly; they emit `SelfModelPatch` objects via `pending_patches()`. The
coordinator collects and applies them each tick via `SelfModel.apply(patch)`.
This keeps agents fully decoupled from the self-model's internal structure.

---

### Planning (`planning/`)

#### Goals and Tasks

**`Goal`** — a coarse, long-horizon objective. Has `success_condition`,
`failure_condition`, and optional `milestone_rewards` — all Python expression
strings evaluated by `RewardEvaluator` against a `WorldQuery` namespace.

**`Task`** — a concrete work item derived by the coordinator and assigned to
a single agent. Also carries condition strings, evaluated by the coordinator
each tick. Tasks are coordinator-internal; the LLM and `GoalTree` never see them.

The distinction matters: goals express what we want, tasks express what we're
doing right now to get there.

#### Condition namespace (`planning/condition_namespace.py`)

`build_core_namespace(wq, tick, start_tick, start_wq)` returns the `eval()`
namespace for condition strings. Includes `inventory`, `charted_chunks`,
`charted_tiles`, `charted_area_km2`, `tick`, `elapsed_ticks`, `new`, `wq`,
`state`, `research`, `tech_unlocked`, `resources_of_type`, `entities`,
`entity_by_id`, and safe builtins.

`_DeltaView` (exposed as `new`) wraps the current `WorldQuery` and a `start_wq`
snapshot taken at goal activation:

```python
new.tick                    # ticks since goal activation (== elapsed_ticks)
new.charted_chunks          # chunks charted this goal
new.inventory('iron-ore')   # ore collected this goal
```

**Use `elapsed_ticks > N` (not `tick > N`) in conditions** — absolute tick
values cause conditions to fire immediately if the game ran before the test.

#### RewardEvaluator (`planning/reward_evaluator.py`)

Evaluates goal `success_condition` and `failure_condition` strings. Adds to the
core namespace: `production_rate`, `staleness`, `logistics`, `power`, `threat`,
`inserters_from/to`.

---

### Execution (`execution/`)

#### Skills (`execution/skills/`)

Skills encode multi-step but well-determined sequences of primitive actions
with built-in success/failure detection. They contain **no decision logic** —
that belongs to agents and the coordinator.

Each skill has a `status()` → `SkillStatus` (`IDLE`, `RUNNING`, `SUCCEEDED`,
`STUCK`) and produces a list of `Action` objects from `tick(wq, ww, tick)`.
Current skills: `NavigateSkill`, `MineSkill`, `DestroySkill`, `CraftSkill`.

#### Agents (`execution/agents/`)

Each agent holds references to the skills relevant to its task types. Agents
read `WorldQuery` (observable world state) but do **not** have access to the
`SelfModel`. Changes to the self-model are communicated upward via
`SelfModelPatch` objects accumulated in `pending_patches()`.

The agent protocol:

```python
class AgentProtocol:
    def activate(self, task: Task, blackboard: Blackboard,
                 wq: WorldQuery, kb: KnowledgeBase) -> None: ...
    def tick(self, task: Task, blackboard: Blackboard,
             wq: WorldQuery, ww: WorldWriter, tick: int,
             kb: KnowledgeBase) -> list[Action]: ...
    def observe(self, task: Task, blackboard: Blackboard,
                wq: WorldQuery, kb: KnowledgeBase) -> dict: ...
    def progress(self, task: Task, blackboard: Blackboard,
                 wq: WorldQuery, kb: KnowledgeBase) -> float: ...
    def pending_patches(self) -> list[SelfModelPatch]: ...
```

Current agents:

- **`NavigationAgent`** — walks the player to a target position or entity.
  Uses `NavigateSkill`. Writes `navigation_stalled` observation on stall.
- **`MiningAgent`** — gathers resources and clears regions. Uses `MineSkill`,
  `DestroySkill`, `NavigateSkill`. Determines clearable objects via
  `can_destroy()` (KB-driven, no name hardcoding). See the harvest gap comment
  in `coordinator.py` for the wood/natural-object limitation.
- **`ExplorationAgent`** — navigates to frontier positions and scans. Uses
  `NavigateSkill`. Writes `exploration_needs_frontier` (GOAL-scoped) when the
  local area is exhausted.
- **`CraftingAgent`** — hand-crafts items. Uses `CraftSkill`.

#### Coordinator (`execution/coordinator/`)

The coordinator has access to the `SelfModel` and the player inventory from
`WorldQuery`. It does **not** make use of the rest of `WorldState` — entity
scans, logistics, and power state are the concern of agents and skills.

The coordinator's primary responsibility is translating Goals into Tasks via
a hierarchical, depth-first, quasi-recursive state machine.

**Goal stack** — a `list[GoalFrame]`, top = active. Each `GoalFrame` carries
the goal type, params, a `step` counter, completed/failed flags, and a `context`
dict for state that must persist across ticks within a handler.

**Goal handlers** — one method per goal type. Each handler is called repeatedly
across ticks; `frame.step` tracks which decision point the handler has reached.
Handlers push sub-goals (onto the goal stack) or tasks (to the active task slot).

**Task activation** — `_push_task(...)` creates a `Task`, selects the owning
agent via `registry.agent_by_id(hint)`, calls `agent.activate(...)`, and writes
a blackboard INTENTION entry. Only one task is active at a time.

**Sub-goal recursion** — handlers call `_push_goal(GOAL_TYPE, params)` to push
a `GoalFrame`. When the sub-goal completes, it pops from the stack and the
parent's step advances. This gives the appearance of recursive decomposition
while keeping the tick loop flat.

**Goal type vocabulary:**

| Type | Description |
|---|---|
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

#### Blackboard (`execution/blackboard.py`)

Shared working memory for the coordinator and agents. Entries have a category
(`INTENTION`, `OBSERVATION`, `RESERVATION`) and a scope (`GOAL` or `TASK`).
`TASK`-scoped entries are cleared when the active task resolves.
`GOAL`-scoped entries persist until the goal completes. The coordinator calls
`clear_scope(EntryScope.TASK)` on each task resolution and `clear_all()` on
goal reset.

---

### Examination (`examination/`)

`AuditReport` — a data container for mechanical and rich examination results.
Carries anomalies, production rates, power state, damage records, and (in rich
mode) blueprint candidates and LLM observations. The merge protocol lets
reports accumulated across ticks be collapsed into a single summary.

Richer examination (self-model reconciliation, blackboard promotion) is Phase 10.

---

### Memory (`memory/`)

`SQLiteBehavioralMemory` — stores strategy records (what worked for a goal type),
performance history (per-goal-type statistics), and spatial pattern metadata
(factory subgraph summaries). Persists across runs. Full strategy matching and
pattern deduplication are Phase 8 concerns.

---

## Data Flow

### Per-tick

1. Bridge produces a `WorldState` snapshot. `WorldWriter.integrate_snapshot()`
   merges it into live global state.

2. `RewardEvaluator` checks active goal success/failure conditions against
   `WorldQuery` and `start_wq` snapshot. If triggered, main loop transitions.

3. `coordinator.tick(wq, ww, tick)` is called:
   - If a task is active: evaluate task success/failure conditions, tick the
     owning agent, collect its `SelfModelPatch` objects.
   - If no task is active: run the top `GoalFrame`'s handler, which either
     pushes a new task or a new sub-goal.
   - Completed/failed sub-goals propagate upward in the goal stack.

4. Main loop dispatches returned actions to `bridge/action_executor.py`.

5. `coordinator.drain_patches()` returns accumulated self-model patches; the
   loop applies them to `SelfModel`.

### Per-task

**Push:** coordinator calls `_push_task(...)`, creates `Task`, calls
`agent.activate(task, ...)`, writes blackboard INTENTION entry.

**Tick:** `agent.tick(task, ...)` called each poll cycle. Agent runs skills
and returns actions.

**Completion:** coordinator evaluates `task.success_condition`. On True:
clears the active task, calls `_bb.clear_scope(EntryScope.TASK)`, runs the
goal handler again (advancing `frame.step`) on the same tick.

**Failure/Escalation:** `failure_condition` fires → goal marked failed →
STUCK propagates to the main loop.

### Per-goal

**Initiation:** `coordinator.reset(goal_type, params, wq)` called. Blackboard
and goal stack cleared. Loop snapshots `WorldQuery` and stores it as `start_wq`
for `new` delta conditions.

**Resolution:** `RewardEvaluator` detects success or failure. Behavioral memory
records outcome. Blackboard cleared. `GoalSource` requests next goal.

---

## Phase Plan

### Phases 1–6 ✅

Full end-to-end pipeline implemented: bridge, world model, planning, execution
layer (coordinator, agents, skills), examination stub, behavioral memory.
Rule-based coordinator handles collection, exploration, clearing, crafting,
and research goals. All unit tests pass.

### Phase 7 — Construction agent (RL)

The construction agent places entities, connects inserters and belts, and
verifies that placed infrastructure is functional. Also planned for Phase 7:
`UseItemOnEntity` bridge action and `harvest_natural` task type (wood gap fix).

### Phase 8 — Production agent

First learned agent. Handles recipe selection, machine configuration, and
production chain management. The `production` and `byproduct` coordinator stubs
become real.

### Phase 9 — Spatial-logistics agent

Spatial reasoning and logistics: belt routing, inserter placement, layout.
The `logistics` coordinator stub becomes real. Internal structure (one agent
or two) decided at phase start — see OD-5.

### Phase 10 — Examination layer revision

Examination gains self-model reconciliation, blackboard promotion, and structured
factory summaries. `RewardEvaluator` namespace extended with STRUCTURAL conditions
backed by the self-model — see OD-6. Technology-change detection (cliff gap fix)
planned here.

### Phase 11 — LLM layer revision

Real LLM client capable of replacing `GoalQueue`. Receives self-model summary and behavioral
memory statistics. Handles escalation via `StuckContext`. The `noop` coordinator
stub becomes real.

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

---

## Passing Context for the Next Phase

When opening a new conversation to implement a component, include:

1. `ARCHITECTURE.md` — full or relevant sections
2. `CONDITION_SCOPE.md` — if the component generates or evaluates goal conditions
3. `REWARD_NAMESPACE.md` — if the component writes or interprets condition strings
4. `OPEN_DECISIONS.md` — if the component touches any open decision
5. The actual source files for all interfaces the component must satisfy or consume
6. The relevant test files