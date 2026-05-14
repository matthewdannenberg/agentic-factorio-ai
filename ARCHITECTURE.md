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
replaced without touching consumers. This principle already governs the world state
layer (WorldQuery/WorldWriter) and extends to every new component.

When a component's interface is defined, it should be defined as a Protocol or
abstract base. The concrete implementation is injected at startup. Swapping
implementations — including wholesale replacement of the RL system — requires
substituting the injected object, not restructuring surrounding code.

### Multi-agent execution is the core

The agent network responsible for executing goals is a coordinated set of learned
agents communicating through a shared blackboard. This is not a scripted bot with
occasional LLM assistance. The learning system does the real work.

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

Action types (23 total across 10 categories): movement, mining, crafting, building
(including `RotateEntity` and `SetSplitterPriority`), inventory, research, player,
vehicle, combat, and meta. Circuit networks and railroad scheduling operations are
explicitly deferred — see `bridge/actions.py` known omissions.

**World state layer** (`world/`)
`WorldState` is a pure data container. `WorldQuery` is the sole read interface.
`WorldWriter` is the sole write interface. `KnowledgeBase` stores game knowledge
learned at runtime. `TechTree`, `ProductionTracker` unchanged.

A `WorldState.trains` section for railroad network state is anticipated but
deferred — see `world/state.py` deferred sections note.

**Planning layer** (`planning/`)
`GoalTree`, `RewardEvaluator`, `ResourceAllocator`. Goal lifecycle management and
reward evaluation against WorldQuery.

The `RewardEvaluator` currently evaluates condition strings against a namespace
derived from `WorldQuery`. In Phase 10, the evaluator's namespace will be extended
to include STRUCTURAL conditions backed by the self-model — covering infrastructure
existence, throughput, and connectivity in ways that WorldQuery's scan-radius-limited
snapshot cannot reliably express. Until Phase 10, the evaluator is unchanged.
See `OPEN_DECISIONS.md` OD-6 and `CONDITION_SCOPE.md` for the design.

---

### Phase 5 — Execution layer foundation ✅

**ExecutionLayerProtocol** (`agent/execution_protocol.py`)

The sole interface between the planning layer and the agent execution network. The
main loop, goal tree, and examination layer interact with execution exclusively
through this protocol. The multi-agent system behind it is invisible to all callers.

```python
class ExecutionStatus(Enum):
    PROGRESSING = auto()   # making progress toward goal
    WAITING     = auto()   # blocked on a game condition (crafting timer, etc.)
    STUCK       = auto()   # cannot advance; escalate to LLM
    COMPLETE    = auto()   # execution network believes goal is achieved

@dataclass
class ExecutionResult:
    actions: list[Action]
    status: ExecutionStatus
    stuck_context: Optional[StuckContext] = None  # populated when STUCK

class ExecutionLayerProtocol:
    def reset(self, goal: Goal, wq: WorldQuery,
              seed_subtasks: Optional[list[Subtask]] = None) -> None: ...
    def tick(self, goal: Goal, wq: WorldQuery, ww: WorldWriter,
             tick: int) -> ExecutionResult: ...
    def progress(self, goal: Goal, wq: WorldQuery) -> float: ...
    def observe(self, wq: WorldQuery) -> dict: ...
```

`reset()` accepts an optional `seed_subtasks` list for the post-escalation path:
when the LLM decomposes a stuck goal and injects subtasks, the coordinator
pre-populates its ledger from these rather than starting derivation from scratch.

**StuckContext** carries the full execution context needed for LLM escalation:

```python
@dataclass
class StuckContext:
    goal: Goal
    failure_chain: list[Subtask]          # outermost → innermost nesting
    sibling_history: dict[str, list[SubtaskRecord]]  # resolved siblings per level
    blackboard_snapshot: dict
```

This gives the LLM a complete picture of what was attempted at every level —
not just the immediate failure point. The LLM can then decide whether to
decompose the leaf failure, discard the parent's approach, or revise further up.

**Blackboard** (`agent/blackboard.py`)

Working memory shared between agents within a goal's lifetime. Cleared on `reset()`.
Partitioned into goal-scoped and subtask-scoped entries; subtask entries are cleared
on subtask resolution while goal entries persist.

Three entry categories:

- *Intentions* — things an agent plans to do but hasn't done yet.
- *Observations* — things an agent has noticed that others should know.
- *Reservations* — claims on resources or regions. Written by the spatial-logistics
  agent, read by all agents to avoid conflicts.

Each entry carries: category, owning agent identifier, creation tick, optional
expiry tick, and scope (goal or subtask).

**Subtask and SubtaskLedger** (`agent/subtask.py`)

A locally-derived prerequisite task. Structurally similar to `Goal` but created
by the execution network rather than the LLM, without LLM involvement unless
derivation fails (`derived_locally=False`).

```python
class SubtaskStatus(Enum):
    PENDING | ACTIVE | COMPLETE | FAILED | ESCALATED

@dataclass
class Subtask:
    id: str
    description: str
    success_condition: str
    failure_condition: str
    parent_goal_id: str
    parent_subtask_id: Optional[str]
    created_at: int
    status: SubtaskStatus
    derived_locally: bool
    resolved_at: Optional[int]
```

The `SubtaskLedger` replaces the simple stack design. It combines a live LIFO
stack (the active nesting chain) with a history log of resolved `SubtaskRecord`
objects, keyed by parent id. The stack uses call-stack semantics: pushing a new
subtask activates it immediately (it is a prerequisite that must run first) and
suspends the parent; popping a completed subtask resumes the parent.

This design ensures the LLM receives complete sibling context at escalation time —
completed predecessors at every nesting level are preserved in the history log and
surfaced via `failure_chain()` and `sibling_history()`.

**Self-model graph** (`agent/self_model.py`)

The agent's persistent graph-theoretic model of what it has built. Starts empty each
run. Built incrementally as the examination layer verifies constructed structures.

Node types: `PRODUCTION_LINE`, `RESOURCE_SITE`, `BELT_CORRIDOR`, `POWER_GRID`,
`DEFENDED_REGION`, `TRAIN_STATION`, `STORAGE`

Node status: `CANDIDATE` → `ACTIVE` / `DEGRADED` / `INACTIVE`

Edge types: `FEEDS_INTO`, `DEPENDS_ON`, `CONNECTED_BY_BELT`, `CONNECTED_BY_RAIL`,
`DEFENDS`, `SPATIALLY_ADJACENT`

Each node carries: type, bounding box, status, throughput dict (item → units/min),
creation tick, last verified tick.

Key queries: `find_producers(item)`, `find_capacity(item)`, `query_path(a, b)`,
`overlapping_nodes(bbox)`, `subgraph(node_ids)`, `promote_candidate(id)`,
`discard_candidate(id)`.

`overlapping_nodes()` detects spatial conflicts between nodes but does not enforce
non-overlap — enforcement is the spatial-logistics agent's responsibility.

See `OPEN_DECISIONS.md` OD-7 for the question of NodeType-specific subclasses.

**Behavioral memory** (`agent/memory/behavioral.py`)

Persistent across runs. SQLite-backed. Implements `BehavioralMemoryProtocol` and
supports use as a context manager (`with SQLiteBehavioralMemory(...) as mem:`).

Three content categories:

- *Strategy records* — for a given goal type and context, what approach worked.
- *Spatial patterns* — recurring self-model subgraphs that proved effective.
  Foundation for the eventual blueprint system. Full pattern extraction deferred
  to Phase 10; stub stores node/edge counts only.
- *Performance history* — per-goal-type statistics across runs.

**Agent network interfaces** (`agent/network/`)

```python
class AgentProtocol:
    def activate(self, goal: Goal, blackboard: Blackboard,
                 wq: WorldQuery) -> None: ...
    def tick(self, blackboard: Blackboard, wq: WorldQuery,
             ww: WorldWriter, tick: int) -> list[Action]: ...
    def observe(self, blackboard: Blackboard, wq: WorldQuery) -> dict: ...
    def progress(self, blackboard: Blackboard, wq: WorldQuery) -> float: ...

class CoordinatorProtocol:
    def reset(self, goal: Goal, wq: WorldQuery,
              seed_subtasks: Optional[list] = None) -> None: ...
    def tick(self, goal: Goal, wq: WorldQuery, ww: WorldWriter,
             tick: int) -> ExecutionResult: ...

class AgentRegistry:
    def register(self, agent: AgentProtocol,
                 goal_types: list[str]) -> None: ...
    def agents_for_goal(self, goal_type: str) -> list[AgentProtocol]: ...
    def all_agents(self) -> list[AgentProtocol]: ...
```

`StubCoordinator` satisfies `CoordinatorProtocol` and returns
`ExecutionResult(actions=[], status=WAITING)` on every tick. Replaced in Phase 6.

**Action preconditions** (`agent/preconditions.py`) — Phase 6

---

### Phase 6 — Navigation agent and rule-based coordinator

A working end-to-end system capable of executing simple goals before the RL agents
exist. Proves the plumbing. The navigation agent is rule-based by design — movement
is well-understood and adds little value from learning at this stage.

- `agent/network/agents/navigation.py` — rule-based movement, pathfinding, waypoint
  following. Reads waypoints from blackboard, updates player position via WorldWriter.
- `agent/preconditions.py` — action validity predicates
- Rule-based coordinator implementation — routes goals to agents by goal type, derives
  simple subtask trees from KB production chains and self-model queries. Manages
  SubtaskLedger lifecycle: pushes, pops, calls complete()/fail()/escalate() before
  popping, clears ledger on goal reset.
- Basic self-model writes — coordinator writes candidate nodes when construction
  subtasks complete.
- `bridge/action_executor.py` — action dispatch handlers for all 23 action types.
- End-to-end smoke test: "collect 50 iron ore" executes successfully.

---

### Phase 7 — Observation and reward infrastructure

The goal-conditioned observation and reward machinery that all RL agents depend on.
Built and validated before any agent is trained to avoid costly representation
mistakes.

**Observation** (`agent/observation.py`)

Goal-conditioned observation constructors. State is expressed relative to goal
structure — as progress fractions toward production chain nodes, spatial coverage
fractions, research completion fractions — rather than in absolute game terms.

This normalisation is what makes a "produce red circuits" goal and a "produce green
circuits" goal structurally similar to a learned policy, despite using different
items. The observation space shape is consistent within a goal type; the content
is goal-specific.

Observation constructors are defined per goal type:
- `ProductionObservation` — inventory fractions, machine status, belt congestion,
  production chain node completion
- `ExplorationObservation` — charted fraction of target, resource patch discovery
  rate, distance to unexplored frontier
- `ResearchObservation` — tech tree path completion, science rate, queue state
- `ConstructionObservation` — placement completion fraction, connectivity status,
  entity status distribution

**Reward** (`agent/reward.py`)

Dense intermediate reward signals beyond what `RewardEvaluator` provides. Derived
from production chain progress, subtask completion, self-model graph growth, and
examination layer feedback.

Reward constructors are goal-type-specific and complement the sparse goal-level
reward from `RewardEvaluator`. Together they provide a learning signal at every
timescale.

Observation and reward infrastructure is validated against constructed WorldQuery
fixtures before any agent uses it. Key invariant: structurally similar goals with
different items should produce structurally similar observations.

---

### Phase 8 — Production agent

First learned agent. Handles recipe selection, machine configuration, and production
chain management. Chosen first because production goals are the most common and the
KB production chain queries provide strong structural guidance.

- `agent/network/agents/production.py` — RL production agent
- Training harness — episode collection, experience replay, policy update
- Policy checkpointing and loading via behavioral memory
- Subtask derivation — production agent identifies prerequisite gaps from KB
  production chains and constructs subtask trees locally
- Behavioral memory integration — strategy records written after goal resolution,
  warm-starting on subsequent similar goals

The specific RL algorithm is an implementation decision, not an architecture
commitment. The agent satisfies `AgentProtocol` regardless of algorithm.

---

### Phase 9 — Spatial-logistics agent

Spatial reasoning and logistics are treated as one phase with explicitly open
internal structure. Whether this becomes one agent or two coordinating agents is
an implementation decision made when the phase begins, informed by research into
MARL approaches for spatially-entangled concerns.

The rationale: placement decisions and routing decisions are deeply interdependent.
"Place furnace here" and "route belt from here to there" cannot be cleanly separated
without creating more coordination overhead than the separation saves.

- `agent/network/agents/spatial_logistics.py` — layout planning, region designation,
  belt routing, inserter placement, splitter priority management (one agent or two
  coordinating, TBD)
- Self-model graph grows richer — region nodes, belt corridor edges, connectivity
  relationships
- `overlapping_nodes()` used for placement conflict detection
- Coordinator updated to route spatial and logistics concerns appropriately

---

### Phase 10 — Examination layer revision

Examination gains new responsibilities under the revised architecture.

- **Self-model reconciliation** — verifies candidate nodes against WorldState,
  promotes confirmed nodes, discards stale ones
- **Blackboard promotion** — converts verified intentions to self-model nodes
- **Behavioral memory updates** — extracts spatial patterns from end-of-goal
  self-model state, records outcomes; full `record_spatial_pattern()` implementation
  replacing the Phase 5 stub
- **Structured factory summary** — self-model-derived description consumed by
  the LLM layer instead of raw WorldState
- **RewardEvaluator namespace extension** — STRUCTURAL conditions backed by
  self-model (`production_line`, `production_capacity`, `has_infrastructure`,
  `connected`, `sm_staleness`). See `OPEN_DECISIONS.md` OD-6.
- Rich and mechanical examination modes preserved

---

### Phase 11 — LLM layer revision

Updated to work with the revised architecture. The LLM never receives raw WorldState.

- Revised strategic prompt — receives self-model summary and behavioral memory
  statistics, not WorldState
- Escalation handling — receives `StuckContext` from `ExecutionResult` (including
  `failure_chain` and `sibling_history`), produces targeted subgoals injected into
  `GoalTree` as children of the current goal; coordinator receives them via
  `reset(seed_subtasks=...)` to re-enter the execution network at the right level
- Reflection — post-goal call assesses reward spec calibration, result written to
  behavioral memory

---

### Phase 12 — Main loop and state machine

Wires everything together.

- `agent/loop.py` — master orchestration implementing the four-timescale data flow
- Revised state machine — execution states include subtask lifecycle transitions
- End-to-end integration tests against a running Factorio instance

---

### Deferred — no phase assigned

These are explicitly designed for but not implemented in the first functional version.
Adding them should require new components, not restructuring existing ones.

- **Biter defense** — combat agents, threat module activation, defensive planning
- **Blueprint system** — curation of spatial patterns from behavioral memory,
  deployment of learned blueprints, occasional annealing for innovation
- **Coordinator learning** — transition from rule-based to learned coordination
- **Higher-order compositional learning** — learning to compose previously learned
  strategies
- **Railroad networks** — `SetTrainSchedule`, `SetStationCondition` action types;
  `WorldState.trains` section; train-state reward namespace entries. Self-model
  already has `TRAIN_STATION` (NodeType) and `CONNECTED_BY_RAIL` (EdgeType) in
  anticipation. Likely warrants a dedicated train-scheduler agent.
- **Circuit networks** — `ConnectWire`, `SetCombinatorCondition`, and related
  action types; significant Lua mod extension required. Likely warrants a dedicated
  circuit-network agent.

---

## Data Flow

### Per-tick

1. Bridge produces a fresh WorldState snapshot. `WorldWriter.integrate_snapshot()`
   merges it into live global state.

2. `RewardEvaluator` checks active goal success/failure conditions against
   `WorldQuery`. If triggered, main loop transitions state.

3. `ExecutionLayerProtocol.tick()` is called. Inside the protocol boundary:
   - Coordinator polls each active agent via `AgentProtocol.tick()`
   - Each agent reads from `WorldQuery` and blackboard, writes observations and
     reservations, returns candidate actions
   - Coordinator resolves conflicts, assembles `ExecutionResult`

4. Main loop dispatches `ExecutionResult.actions` to `bridge/action_executor.py`.

5. `ExecutionResult.status` is inspected:
   - `PROGRESSING` / `WAITING` — continue
   - `STUCK` — pass `StuckContext` to LLM layer for decomposition
   - `COMPLETE` — trigger examination

### Per-subtask

**Initiation:**
1. An agent identifies a prerequisite gap via self-model (`find_producers(item)`)
   and KB (`production_chain(item)`).
2. Coordinator constructs a `Subtask` and pushes it onto the ledger. The new subtask
   is activated immediately (call-stack semantics); the parent is suspended.
3. Blackboard is partitioned — subtask-scoped entries created alongside goal-scoped
   entries. Agents reconfigure observation spaces relative to new subtask context.

**Resolution:**
1. Subtask success condition met — coordinator calls `complete(tick)`, then pops
   from the ledger. The record is written to the ledger's history log under the
   subtask's parent id.
2. The parent subtask (or goal) is resumed.
3. If subtask produced a persistent structure, coordinator writes a candidate
   `SelfModelNode` to the blackboard.
4. Examination layer (next cycle) verifies and promotes or discards the candidate.
5. Subtask-scoped blackboard entries cleared. Goal-scoped entries persist.

**Failure and escalation:**
1. Subtask failure condition triggers, or coordinator cannot derive decomposition.
2. Coordinator calls `escalate(tick)` on the stuck subtask (and any ancestors that
   can no longer proceed), then pops them, building the history log.
3. `ExecutionResult.status = STUCK`. `StuckContext` is constructed from the ledger:
   - `failure_chain` = `ledger.failure_chain()` — the live nesting at failure time
   - `sibling_history` = `ledger.sibling_history(chain, goal.id)` — resolved
     siblings at each level, so the LLM sees what was already completed or attempted
4. Main loop passes `StuckContext` to LLM layer. LLM produces subgoals injected
   into `GoalTree` as children of the current goal, and returns them for
   `reset(seed_subtasks=...)`. Execution network resets with the LLM's decomposition
   pre-loaded into the ledger.

### Per-goal

**Initiation:**
1. `ExecutionLayerProtocol.reset(goal, wq, seed_subtasks)` called.
2. Coordinator clears blackboard and subtask ledger.
3. If `seed_subtasks` provided (post-escalation), coordinator pushes them onto the
   ledger in order. Otherwise, derivation starts from scratch.
4. Self-model queried for prerequisite feasibility. If infeasible, `STUCK` returned
   immediately before any ticks are spent.
5. Behavioral memory queried for relevant strategy records. Matching agents
   warm-started with prior policy state.
6. Agents activated; observation spaces configured relative to goal structure.

**Progression:**
1. Subtask tree derived incrementally — only as far ahead as current state makes
   visible.
2. `progress()` available for main loop and examination layer at any time.
3. Examination layer runs periodically, reconciling WorldState with self-model.
   Significant divergences written as blackboard observations.

**Resolution:**
1. `RewardEvaluator` detects success or failure.
2. Examination layer produces structured summary.
3. Behavioral memory records outcome: goal type, context summary, strategy, reward,
   ticks elapsed.
4. Self-model updated: candidate nodes confirmed or discarded.
5. Blackboard cleared.

### Per-run and cross-run

**End of run:**
1. Effective self-model subgraphs extracted as spatial pattern candidates and written
   to behavioral memory.
2. Behavioral memory, KB, and agent policy weights persisted to disk.

**Run start:**
1. KB loaded — all game knowledge immediately available.
2. Behavioral memory loaded — strategy records and performance history available.
3. Agent policies loaded from most recent checkpoint.
4. Self-model and blackboard start empty.

---

## Open Implementation Decisions

These questions are explicitly deferred pending research or experimentation.
See `OPEN_DECISIONS.md` for full discussion of each.

- **OD-1** — Observation space construction
- **OD-2** — Coordinator learning
- **OD-3** — Self-model cross-run persistence
- **OD-4** — RL algorithm family
- **OD-5** — Spatial-logistics internal structure
- **OD-6** — RewardEvaluator namespace extension for self-model (STRUCTURAL conditions)
- **OD-7** — NodeType-specific SelfModelNode subclasses

---

## Project Structure

```
factorio-agent/
│
├── bridge/                          # Hard boundary — nothing outside speaks RCON
│   ├── actions.py                   ✅ 23 action types across 10 categories
│   ├── rcon_client.py               ✅
│   ├── state_parser.py              ✅
│   ├── action_executor.py           ✅
│   └── mod/
│       ├── info.json                ✅
│       ├── control.lua              ✅
│       └── test_bridge_live.lua     ✅
│
├── world/                           # Pure data and computation
│   ├── state.py                     ✅ WorldState pure data container
│   ├── query.py                     ✅ WorldQuery — sole read interface
│   ├── writer.py                    ✅ WorldWriter — sole write interface
│   ├── knowledge.py                 ✅ KnowledgeBase — SQLite-backed
│   ├── entities.py                  ✅
│   ├── tech_tree.py                 ✅
│   └── production_tracker.py        ✅
│
├── planning/                        # Goal lifecycle and reward evaluation
│   ├── goal.py                      ✅
│   ├── goal_tree.py                 ✅
│   ├── reward_evaluator.py          ✅
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
│   ├── preconditions.py             [ Phase 6 ] Action validity predicates
│   ├── observation.py               [ Phase 7 ] Goal-conditioned observation
│   │                                            constructors
│   ├── reward.py                    [ Phase 7 ] Dense intermediate reward
│   │                                            signals
│   ├── state_machine.py             ✅
│   ├── loop.py                      [ Phase 12 ]
│   │
│   ├── network/
│   │   ├── agent_protocol.py        ✅ AgentProtocol
│   │   ├── coordinator.py           ✅ CoordinatorProtocol + StubCoordinator
│   │   ├── registry.py              ✅ AgentRegistry
│   │   └── agents/
│   │       ├── navigation.py        [ Phase 6 ] Rule-based movement
│   │       ├── production.py        [ Phase 8 ] RL production agent
│   │       └── spatial_logistics.py [ Phase 9 ] RL spatial-logistics agent
│   │                                            (internal structure TBD)
│   │
│   ├── memory/
│   │   └── behavioral.py            ✅ BehavioralMemoryProtocol +
│   │                                   SQLiteBehavioralMemory stub
│   │
│   └── examiner/
│       ├── audit_report.py          ✅
│       ├── rich_examiner.py         [ Phase 10 ] Revised with self-model
│       │                                         reconciliation
│       └── mechanical_auditor.py    [ Phase 10 ]
│
├── llm/
│   ├── client.py                    [ Phase 11 ]
│   ├── budget.py                    [ Phase 11 ]
│   └── prompts/
│       ├── strategic.md             [ Phase 11 ] Receives self-model summary
│       ├── escalation.md            [ Phase 11 ] Receives StuckContext
│       ├── examination.md           [ Phase 10 ]
│       └── reflection.md            [ Phase 11 ]
│
├── data/
│   └── knowledge/
│       └── knowledge.db             (runtime, gitignored)
│
├── config.py                        ✅
├── ARCHITECTURE.md                  (this file)
├── CONDITION_SCOPE.md               ✅
├── REWARD_NAMESPACE.md              ✅
├── OPEN_DECISIONS.md                ✅
│
└── tests/
    ├── fixtures.py                  ✅
    ├── unit/
    │   ├── bridge/                  ✅
    │   ├── world/                   ✅
    │   ├── planning/                ✅
    │   └── agent/                   ✅ Phase 5 complete (179 tests)
    └── integration/                 ✅ existing; extended each phase
```

---

## Build Order and Testing Strategy

Each phase produces interfaces before implementations, and tests before or alongside
code. No phase begins until the previous phase's tests pass.

**Phase 5** produced only interfaces, protocols, and stubs. 179 tests verify that
the protocol surface is complete, stubs satisfy their protocols, and the SubtaskLedger
correctly maintains call-stack semantics and history. No game interaction required.

**Phase 6** produces the first runnable system. Tests verify end-to-end execution
of simple goals without a learned policy.

**Phase 7** produces observation and reward machinery validated entirely against
constructed fixtures. Key test invariant: structurally similar goals with different
items produce structurally similar observations.

**Phases 8-9** each produce a learned agent. Tests verify that the agent improves
over episodes and that behavioral memory warm-starting reduces time-to-goal on
repeated similar goals.

**Phases 10-12** integrate and close the loop. Full end-to-end tests require a
running Factorio instance.

---

## Passing Context for the Next Phase

When opening a new conversation to implement a component, include:

1. `ARCHITECTURE.md` — full or relevant sections
2. `CONDITION_SCOPE.md` — if the component generates or evaluates goal conditions
3. `REWARD_NAMESPACE.md` — if the component writes or interprets condition strings
4. `OPEN_DECISIONS.md` — if the component touches any open decision
5. The actual source files for all interfaces the component must satisfy or consume
6. The relevant test files

The component brief template (from the original architecture document) remains valid.