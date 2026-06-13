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

**Unified planning stack.** Goals and Tasks are both `PlanningItem` subclasses and
live on a single `_stack`. The coordinator dispatches based on what is on top: Tasks
go to agents, Goals go to handlers. This replaces the former split between a goal
stack and an active-task slot.

**Predicates own world-state logic.** Skills and agents delegate all world-state
questions (is this entity present? is it reachable? can it be mined?) to
`execution/predicates.py`, which provides pure, cheap, side-effect-free checks
over `WorldQuery`. Factorio implementation details (raw unit numbers, the old
`entity_id=0` pattern for natural objects, etc.) are encapsulated in the bridge
and `WorldWriter` and never leak into agent or coordinator logic.

## Status

**Phases 1–6 complete. Phase 6.5 refactors complete. Phase 6.6 refactors complete.**

The full end-to-end pipeline is implemented and running against a live Factorio
instance. The agent can navigate, gather resources, explore, clear terrain, and
craft. All unit tests pass. `clear_region` works end-to-end including trees and
rocks.

```
[✅] Core dataclasses / bridge / Lua mod
[✅] World model (observable, knowledge, self-model layers)
[✅] Planning layer (Goals, Tasks, RewardEvaluator, condition namespace)
[✅] Execution layer (coordinator, agents, skills, blackboard, memory)
[✅] Examination layer (AuditReport stub)
[✅] Phase 6 — Navigation, mining, rule-based coordinator, run loop
[✅] Phase 6.5 — Unified planning stack (PlanningItem), navigate goal type,
                 clear_region end-to-end, predicate layer, condition_parser
[✅] Phase 6.6 — Unified entity identity (sys_ids), entity accumulation,
                 tile coverage map, NaturalObject removed, scan coverage resolved
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
- *Model* — `SelfModel`: layered self-model of what the agent has built

**Planning** (`planning/`) — `PlanningItem` (base), `Goal`, `Task`, `GoalTree`,
`RewardEvaluator`, `condition_namespace`, `condition_parser`. Goals and Tasks share a
unified lifecycle (`ItemStatus`). `RewardEvaluator` evaluates condition strings against
a `WorldQuery` namespace each tick.

**Execution** (`execution/`) — the agent network. Contains the coordinator,
agents, skills, blackboard, predicates, preconditions, and the main loop. The
coordinator maintains a unified `_stack: list[PlanningItem]` and dispatches to
goal handlers or agents depending on what is on top.

**Examination** (`examination/`) — `AuditReport` and auditor stubs. Grows in Phase 10.

**Memory** (`memory/`) — `BehavioralMemory`: strategy records and performance
history persisted across runs via SQLite.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full design documentation.

## Key Concepts

### PlanningItem, Goal, and Task

`PlanningItem` is the base class for both `Goal` and `Task`. Both live on the
coordinator's unified `_stack`. The top of the stack is always the current unit of
work: if it is a `Task`, tick its agent; if it is a `Goal`, run its handler.

**Goals** are coarse, long-horizon objectives (e.g. "collect 200 iron ore", "clear
this region"). They carry `priority`, `reward_spec`, `step`, and `context`. The
`step` counter and `context` dict replace the former `GoalFrame` — they are now
first-class fields on `Goal`.

**Tasks** are concrete work items assigned to a specific agent (e.g. "navigate to
position X", "gather 10 iron ore", "clear natural objects in bbox"). They carry
`agent_hint`, `task_type`, and `params`.

### TASK_GOAL_TYPES

Some goal_type strings bypass coordinator goal handlers entirely and push a `Task`
directly onto the stack. This routing table lives in `coordinator.py` because it
is execution knowledge. Currently: `"navigate"` routes to NavigationAgent. Adding
a new task-backed goal type requires one line in `TASK_GOAL_TYPES`.

### Predicates

`execution/predicates.py` provides pure observation functions over `WorldQuery`:

- `is_present(sys_id, wq)` — True if the entity with the given sys_id exists in
  the unified entity list. Works uniformly for placed entities and natural objects.
- `is_reachable(sys_id, wq)` — checks the sys_id-keyed reachable set populated by
  `WorldWriter`. Do not use hardcoded distance constants as a proxy.
- `can_destroy(entity, kb)` — the authoritative check for natural object clearability.
  Takes an `EntityState` with `is_natural=True`; KB is the sole authority.

### Skills

Skills encode multi-step action sequences with success/failure detection. They
contain no decision logic and delegate all world-state questions to predicates.
`DestroySkill` uses `predicates.is_present(sys_id, wq)` for target-gone detection
and `predicates.is_reachable(sys_id, wq)` for reach checks. All entities — placed
and natural — are identified by stable sys_ids assigned by `WorldWriter`.

### Writing Goal Conditions

Use **relative conditions** rather than absolute game state where possible:

```python
# Collect 5 new iron ore (works regardless of prior inventory)
success_condition="new.inventory('iron-ore') >= 5"

# 3 minutes since this goal activated (not absolute game clock)
failure_condition="elapsed_ticks > 10800"

# Navigate to a specific position
success_condition="navigate_to(95.0, 95.0)"

# Clear a region (PROXIMAL — navigate to corners first for scan coverage)
success_condition="bbox(-50,-50,50,50).is_clear"
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

```bash
# Unit tests — no Factorio required
pytest tests/
pytest tests/unit/execution/ -v
pytest tests/unit/world/ -v
pytest tests/unit/planning/ -v

# In-game tests — Factorio required
pytest tests_in_game/ -v
pytest tests_in_game/03_collection/ -v
```

## Key Documents

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — full design, interfaces, data flow, phase plan
- [`CONDITION_SCOPE.md`](CONDITION_SCOPE.md) — proximal vs non-proximal condition reference
- [`REWARD_NAMESPACE.md`](REWARD_NAMESPACE.md) — complete reward evaluator namespace reference
- [`OPEN_DECISIONS.md`](OPEN_DECISIONS.md) — deferred architectural questions