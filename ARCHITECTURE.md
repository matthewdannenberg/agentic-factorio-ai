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

### The World Model Starts Empty

The agent begins each fresh install knowing nothing about specific game content. There are
no hard-coded entity lists, tech trees, or recipe tables. `KnowledgeBase` is populated
entirely at runtime by querying Factorio's prototype API via RCON. Knowledge persists across
runs in `data/knowledge/knowledge.db` (gitignored).

This makes the system inherently mod-compatible: a mod that adds new buildings, resources,
or technologies is handled automatically. The agent discovers them the first time they appear
in the world, queries their prototype data, stores the results, and reasons about them
identically to vanilla content — without any code changes.

### KnowledgeBase Uses SQLite for Queryable Persistence

Five knowledge domains (entities, resources, fluids, recipes, techs) are stored in eleven
normalised tables. Recipes and techs are split into header + child tables because their
ingredients, products, prerequisites, and unlock effects are variable-length lists that must
be queryable by content. Scalar domains (entities, resources, fluids) use single flat tables.

The planning layer needs cross-domain queries that are expensive as dict scans:
- "All recipes that produce X" — indexed on `recipe_products(item_name)`
- "All recipes that run in entity Y" — indexed on `recipe_made_in(entity_name)`
- "Techs that unlock recipe X" — indexed on `tech_unlocks_recipes(recipe_name)`
- "Full production chain for X" — recursive CTE over `recipe_ingredients`/`recipe_products`

In-memory dicts act as a write-through cache so reads never hit the DB during normal
operation. The SQLite connection is opened once per `KnowledgeBase` lifetime and must be
explicitly closed (context manager or `.close()`) before the data directory is deleted —
important on Windows where SQLite holds a file lock.

### Placeholder / Enrichment Lifecycle

When an unknown name is encountered offline (no RCON connection), a placeholder record is
stored immediately with safe defaults and `is_placeholder=True`. Once a live connection
becomes available, `KnowledgeBase.enrich_placeholders()` re-queries every placeholder
across all five registries and replaces them with real data. Each `ensure_*` method also
re-queries automatically if it encounters a placeholder and a `query_fn` is available,
so enrichment happens opportunistically during normal operation as well as on explicit call.

This means:
- The agent never crashes or refuses to operate due to missing knowledge
- Knowledge gaps are filled in automatically as the game runs
- Subsequent sessions start with all previously-learned knowledge intact

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

### Actions Are Categorised and Context-Filtered

Every `Action` subclass declares an `ActionCategory`. The function `actions_for_context(
in_vehicle, biters_enabled)` returns the valid action subset for the current execution
context. `VEHICLE` actions are only valid while in a vehicle; `MOVEMENT` actions are only
valid on foot; `COMBAT` actions are only emitted when `BITERS_ENABLED=True`.

### Circuit Networks Are Explicitly Deferred

Factorio's circuit network is intentionally out of scope for the initial implementation.
Adding them later means new `Action` subclasses and Lua mod extensions; no existing code
changes.

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
│       └── control.lua              # Lua mod — exposes state, accepts commands,
│                                    #   includes fa.get_entity_prototype(),
│                                    #   fa.get_recipe_prototype(), fa.get_technology(),
│                                    #   fa.get_fluid_prototype(), fa.get_resource_prototype()
│
├── world/                           # Pure data and computation — no LLM, no RCON
│   ├── state.py                     # WorldState and all sub-dataclasses
│   ├── knowledge.py                 # KnowledgeBase — SQLite-backed, runtime-extensible
│   │                                #   Five registries: entities, resources, fluids,
│   │                                #   recipes, techs. Eleven normalised tables.
│   │                                #   Placeholder/enrichment lifecycle.
│   │                                #   Cross-domain queries: recipes_for_product(),
│   │                                #   recipes_for_ingredient(), recipes_made_in(),
│   │                                #   techs_unlocking_recipe(), production_chain().
│   ├── entities.py                  # Thin facade: ResourceRegistry, get_entity_metadata()
│   ├── tech_tree.py                 # TechTree — KB-backed research graph
│   │                                #   is_unlocked(), is_reachable(), path_to(),
│   │                                #   next_researchable(), absorb_research_state()
│   ├── production_tracker.py        # Throughput tracking over WorldState snapshots
│   └── entities.py                  # (legacy API preserved — delegates to KnowledgeBase)
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
│   │   └── audit_report.py         # Shared dataclass — passed to LLM on resumption
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
│       └── blueprint.py
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
├── data/
│   └── knowledge/
│       └── knowledge.db             # SQLite — created at runtime, gitignored
│                                    # Tables: entities, resources, fluids,
│                                    #   recipes, recipe_ingredients, recipe_products,
│                                    #   recipe_made_in, techs, tech_prerequisites,
│                                    #   tech_unlocks_recipes, tech_unlocks_entities
│
├── config.py                        # All tunable parameters
│
└── tests/
    ├── test_core_dataclasses.py              ✅ 128 tests
    ├── unit/
    │   ├── bridge/
    │   │   ├── test_state_parser.py          ✅ 80 tests
    │   │   └── test_action_executor.py       ✅
    │   └── world/
    │       ├── test_knowledge.py             ✅ 75 tests
    │       ├── test_entities.py              ✅ 36 tests
    │       ├── test_tech_tree.py             ✅ 64 tests
    │       └── test_production_tracker.py   ✅
    └── integration/
        └── test_bridge_live.lua              In-game console test suite
```

---

## Core Dataclasses

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
- `threat`: `ThreatState` — biter bases, pollution, attack timers (empty when biters off)

Key methods: `entity_by_id()`, `entities_by_name()`, `entities_by_status()`,
`resources_of_type()`, `inventory_count()`, `section_staleness()`, `has_damage`,
`recent_losses`, `game_time_seconds`.

### KnowledgeBase (`world/knowledge.py`)

The agent's persistent store of learned game knowledge. Backed by SQLite.

**Registry methods** (all five domains follow the same pattern):
- `ensure_*(name)` — return record, querying Factorio if unknown or if placeholder + query_fn
- `get_*(name)` — return record or None, never queries
- `all_*()` — return dict snapshot of in-memory cache

**Cross-domain query methods**:
- `recipes_for_product(item)` — recipes that produce a given item
- `recipes_for_ingredient(item)` — recipes that consume a given item
- `recipes_made_in(entity)` — recipes that can run in a given entity type
- `techs_unlocking_recipe(recipe)` — techs whose effects unlock a given recipe
- `production_chain(item)` — recursive ingredient closure (SQL recursive CTE)

**Lifecycle methods**:
- `enrich_placeholders()` — re-query all placeholder records; call once RCON connects
- `close()` / context manager — must be called before deleting the data directory

**query_fn injection**: `KnowledgeBase(data_dir, query_fn)` where `query_fn` is a
`Callable[[str], str]` that sends a Lua expression and returns raw JSON. Injected by
the main loop; `None` in tests and offline mode.

### TechTree (`world/tech_tree.py`)

Research dependency graph backed by KnowledgeBase. No static data — all knowledge
is learned at runtime.

Key methods:
- `is_unlocked(tech, research)` — pure ResearchState lookup, no KB needed
- `is_reachable(tech, research)` — True if all direct prerequisites are unlocked
- `prerequisites(tech)` / `all_prerequisites(tech)` — direct and transitive prereqs
- `unlocks_recipe(tech)` / `unlocks_entity(tech)` — what a tech grants
- `path_to(tech, research)` — ordered list of techs to research; topological sort
- `next_researchable(research)` — all techs with prerequisites met, sorted by depth
- `absorb_research_state(research)` — ensure all mentioned techs are in KB; call
  each tick cycle so the KB grows to match the agent's actual tech state

### Goal and RewardSpec (`planning/goal.py`)

Produced by the strategic or tactical LLM. See original ARCHITECTURE documentation.

### AuditReport (`agent/examiner/audit_report.py`)

Produced by either examiner mode. See original ARCHITECTURE documentation.

### Action types (`bridge/actions.py`)

21 concrete types across 10 categories. See original ARCHITECTURE documentation.

### AgentState (`agent/state_machine.py`)

Four states: `PLANNING`, `EXECUTING`, `EXAMINING`, `WAITING`.

---

## Bridge Layer

### `bridge/mod/control.lua` — Prototype Query Functions

In addition to the state query and action command functions documented previously,
the Lua mod exposes five prototype query functions used by `KnowledgeBase` for discovery:

- `fa.get_entity_prototype(name)` — tile dimensions, type, recipe slot, ingredient/output
  slot counts. Used to populate `EntityRecord` on first encounter.
- `fa.get_resource_prototype(name)` — whether the resource yields a fluid, whether it is
  infinite, display name.
- `fa.get_fluid_prototype(name)` — default and max temperature, fuel value in joules,
  emissions multiplier.
- `fa.get_recipe_prototype(name)` — ingredients, products (with amounts and probabilities),
  crafting time, category, and which entity types can craft it (derived by scanning
  `game.entity_prototypes` for matching crafting categories).
- `fa.get_technology(name)` — prerequisites, effects (unlock-recipe and give-item),
  whether it is currently researched and enabled.

All five return `{"ok": false, "reason": "..."}` if the prototype does not exist.

---

## World Model Layer

### Design principles

**Starts empty, grows at runtime.** No vanilla content is hard-coded. Every entity,
resource, fluid, recipe, and technology the agent encounters is queried from Factorio,
stored in SQLite, and available for future reasoning. This makes the system inherently
compatible with mods and Space Age content.

**EntityCategory is agent vocabulary, not game data.** Factorio has prototype type strings
(`"furnace"`, `"assembling-machine"`). The agent maps these to its own category enum
(`SMELTING`, `ASSEMBLY`) for reasoning purposes. The mapping is minimal — only types where
the category meaningfully changes what actions or reasoning apply. Unknown types fall through
to `OTHER` safely.

**Placeholders are first-class.** A placeholder record (`is_placeholder=True`) represents
"we know this thing exists but don't yet know its properties." The agent can operate on
placeholders — it just can't make category-specific decisions. Placeholders are enriched
automatically when a live connection is available.

**Normalisation enables planning queries.** Recipes are split into header + ingredient,
product, and made-in child tables so the planning layer can ask "what produces X?" and
"what can this machine make?" as indexed SQL queries rather than Python scans. The
`production_chain()` recursive CTE computes the full ingredient closure of any item in
a single SQL statement.

### Knowledge file location

`data/knowledge/knowledge.db` — single SQLite file. Add `data/` to `.gitignore`.
Missing on first run: `KnowledgeBase.__init__` creates the directory and schema automatically.

### Main loop integration points

```python
# At startup — create shared KB and inject query_fn once RCON connects
kb = KnowledgeBase(data_dir=Path("data/knowledge"), query_fn=rcon_client.send)
kb.enrich_placeholders()   # re-query anything learned offline in a previous session

# Shared instances
resource_registry = ResourceRegistry(kb)   # passed to StateParser
tech_tree = TechTree(kb)
production_tracker = ProductionTracker()

# Each tick cycle — after bridge returns a new WorldState
tech_tree.absorb_research_state(world_state.research)
production_tracker.update(world_state)
```

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
3. **World model** ✅ — `KnowledgeBase` (SQLite), `ResourceRegistry`, `TechTree`,
   `ProductionTracker` — 175+ tests passing
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

`tests/unit/world/test_knowledge.py` — 75 tests covering all five registries, placeholder
lifecycle, enrichment, persistence across simulated restarts, SQL query methods
(`recipes_for_product`, `recipes_for_ingredient`, `recipes_made_in`,
`techs_unlocking_recipe`, `production_chain`), and Windows SQLite file-lock safety
(context manager usage throughout).

`tests/unit/world/test_entities.py` — 36 tests covering the `ResourceRegistry` facade and
`get_entity_metadata()` delegation, placeholder defaults, safety guarantees, and correct
passthrough of real records injected directly into the KB cache.

`tests/unit/world/test_tech_tree.py` — 64 tests covering empty-KB behaviour, prerequisite
queries, reachability, unlock queries, path planning, next-researchable, absorb lifecycle,
persistence across restarts, and mod compatibility.

`tests/unit/world/test_production_tracker.py` — throughput tracking, gap handling,
stall detection, and summary generation.

### Windows compatibility note

SQLite holds a file lock for the lifetime of the connection. All tests use `KnowledgeBase`
as a context manager (`with KnowledgeBase(...) as kb:`) or call `kb.close()` explicitly
in `tearDown` before `TemporaryDirectory.cleanup()`. This prevents `PermissionError` on
Windows when the temp directory is deleted while the connection is still open.

### In-game integration tests (requires live Factorio + mod)

`tests/integration/test_bridge_live.lua` — exercises each `fa.*` function against the live
game and prints PASS/FAIL results. See the script's header for usage instructions.

---

## Component Conversation Brief Template

When opening a new conversation to implement a component, include:

1. `ARCHITECTURE.md` (this file) — full or relevant sections
2. `CONVERSATION_BRIEF.md` — the specific brief for this component
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
- [x] World model — `world/knowledge.py`, `world/entities.py`, `world/tech_tree.py`,
      `world/production_tracker.py` — 175+ tests
- [ ] Planning layer (`goal_tree.py`, `reward_evaluator.py`, `resource_allocator.py`)
- [ ] Primitives
- [ ] Execution layer
- [ ] Examination layer
- [ ] Tactical layer
- [ ] Strategic layer
- [ ] Memory layer
- [ ] Main loop
- [ ] Threat module