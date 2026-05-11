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
on `WorldState` objects (accessed via `WorldQuery`) and structured `Action` objects
exclusively. This means the full agent stack can be tested against a mock game state
without Factorio running.

### WorldState Is Accessed Only Through WorldQuery and WorldWriter

`WorldState` is a pure data container with internal lookup indices. No file outside
`world/` may access `WorldState` fields directly. All reads go through `WorldQuery`;
all mutations go through `WorldWriter`. This boundary means:

- The backing store can be changed (e.g. from Python dicts to an in-memory database)
  without touching any consumer file.
- The evaluator, execution layer, and examiner all receive `WorldQuery`; they cannot
  accidentally mutate game state.
- The bridge and execution layer receive `WorldWriter`; they are the only legitimate
  mutation paths.

`world/__init__.py` exports `WorldQuery` and `WorldWriter` (plus the dataclass types
needed in signatures). `WorldState` itself is not exported from `world/`.

### WorldQuery Provides Both Named Lookups and Composable Filtering

`WorldQuery` exposes named convenience methods for the most common patterns
(`entity_by_id`, `entities_by_name`, `entities_by_status`, `entities_by_recipe`,
`inserters_taking_from`, `inserters_delivering_to`, `resources_of_type`,
`inventory_count`, `section_staleness`) and a composable `EntityQuery` builder for
multi-predicate joins:

```python
wq.entities()
  .with_name("assembling-machine-1")
  .with_recipe("electronic-circuit")
  .with_inserter_input()
  .with_inserter_output()
  .count()
```

The builder is also exposed in the reward evaluator namespace as `wq`, so condition
strings can use it directly:

```python
success_condition = (
    "wq.entities()"
    "   .with_recipe('electronic-circuit')"
    "   .with_inserter_input()"
    "   .with_inserter_output()"
    "   .count() >= 4"
)
```

### WorldWriter Handles Both Bulk and Fine-Grained Mutations

Two mutation categories:

- **Section replacement** (`integrate_snapshot`, `replace_entities`, `replace_logistics`,
  etc.): the bridge use case. `WorldWriter.integrate_snapshot(snapshot)` merges a
  `StateParser`-produced snapshot into the live global state, section by section, with
  staleness guards and index rebuilds.
- **Fine-grained mutation** (`add_entity`, `remove_entity`, `update_entity_status`,
  `update_entity_recipe`, `update_player_inventory`, etc.): the execution layer use case.
  Each method updates only the affected indices rather than rebuilding everything.

### StateParser Produces Snapshots; WorldWriter Integrates Them

`StateParser.parse()` constructs a fresh `WorldState` snapshot from raw Lua/JSON output.
This is the one legitimate place that builds `WorldState` via its dataclass constructor
directly. After populating all sections, `_populate_all` calls
`state._rebuild_entity_indices()` and `state._rebuild_inserter_indices()` so the snapshot
has valid indices immediately.

The main loop then calls `world_writer.integrate_snapshot(snapshot)` to merge the snapshot
into the live global state. Consumers never receive the raw `WorldState` — they receive a
`WorldQuery` wrapping it.

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

### Rewards Are Self-Designed

When the strategic LLM sets a goal, it simultaneously produces a `RewardSpec` — a structured
definition of success conditions, failure conditions, milestone rewards, and time discounting.
The `RewardSpec` is evaluated mechanically by `planning/reward_evaluator.py` against a live
`WorldQuery`. No LLM is involved in reward evaluation during execution.

After goal resolution, a reflection call reviews whether the reward spec was well-calibrated.
That reflection is stored in memory and informs future goal-setting.

The only externally fixed reward is a meta-reward (e.g. "launch the rocket", "survive N
hours") that the agent cannot redefine. Everything beneath that is agent-designed.

### Condition Scope Is a First-Class Design Constraint

Not all reward conditions are equally reliable. The system distinguishes:

- **PROXIMAL** conditions: only accurate when the player is within `LOCAL_SCAN_RADIUS` of
  relevant entities. Includes `entities()`, `production_rate()`, `logistics`, `power`, and
  connectivity queries. Evaluating these while the player is far away may produce false
  negatives.
- **NON-PROXIMAL** conditions: globally accurate at all times. Includes `inventory()`,
  `tech_unlocked()`, `resources_of_type()`, `charted_chunks`, and time.

The `staleness(section)` function is available in condition expressions to guard proximal
conditions against stale data. See `CONDITION_SCOPE.md` for the authoritative breakdown
and the rules the LLM must follow when writing goal conditions. See `REWARD_NAMESPACE.md`
for the complete eval namespace reference. Both documents must be included in any
conversation implementing the strategic or tactical LLM layers.

### Belt Transport Is Two-Lane

Factorio belts have two independent transport lanes. `BeltSegment` models both via
`BeltLane` objects — each carrying an `{item: count}` dict and a per-lane `congested`
flag. The Lua mod reads both lanes fully via `get_transport_line(1)` and
`get_transport_line(2)`. The old single-item approximation has been replaced.

### Inserters Carry Spatial Context

`InserterState` records `pickup_position` and `drop_position` as world coordinates.
`WorldQuery` exposes connectivity queries — `inserters_taking_from(entity_id)` and
`inserters_delivering_to(entity_id)` — that determine which inserters are connected to
which entities via bounding-box containment, without additional RCON calls. `WorldState`
maintains pre-built spatial inserter indices (keyed by entity_id) for O(1) lookup on 1×1
entities; `WorldQuery` falls back to a bounding-box scan for larger entities using
tile dimensions from `KnowledgeBase`.

### Exploration Is Tracked as a Non-Proximal Accumulator

`ExplorationState` on `PlayerState` records `charted_chunks` — the total 32×32-tile
chunks the force has revealed, sourced from `LuaForce::get_chart_size(surface)`. Global,
monotonically increasing, and accurate anywhere. Exposed in the reward namespace as
`charted_chunks`, `charted_tiles`, and `charted_area_km2`.

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
│   ├── state_parser.py              # Raw Lua output → WorldState snapshot;
│   │                                #   rebuilds entity + inserter indices after parse
│   ├── action_executor.py           # Action objects → RCON commands
│   └── mod/
│       ├── info.json                # Factorio mod metadata (factorio_version: "2.0")
│       └── control.lua              # Lua mod — exposes state, accepts commands,
│                                    #   two-lane belt reading, inserter pickup/drop
│                                    #   positions, charted_chunks via get_chart_size,
│                                    #   fa.get_exploration(), prototype query functions
│
├── world/                           # Pure data and computation — no LLM, no RCON
│   ├── state.py                     # WorldState — pure data container.
│   │                                #   Internal indices: _by_id, _by_name, _by_recipe,
│   │                                #   _by_status (built by __post_init__);
│   │                                #   _inserters_from, _inserters_to (built by
│   │                                #   WorldWriter after logistics section is populated).
│   │                                #   BeltLane + BeltSegment (two independent lanes),
│   │                                #   InserterState (pickup/drop world positions),
│   │                                #   ExplorationState (charted_chunks),
│   │                                #   LogisticsState, PlayerState.
│   │                                #   NOT exported from world/__init__.py.
│   ├── query.py                     # WorldQuery — sole read interface.
│   │                                #   Named lookups: entity_by_id(), entities_by_name(),
│   │                                #   entities_by_status(), entities_by_recipe(),
│   │                                #   inserters_taking_from(), inserters_delivering_to(),
│   │                                #   inserters_taking_from_type(),
│   │                                #   inserters_delivering_to_type(),
│   │                                #   resources_of_type(), inventory_count(),
│   │                                #   section_staleness(), fully_connected_entities().
│   │                                #   Composable builder: entities().with_name()
│   │                                #     .with_recipe().with_status()
│   │                                #     .with_inserter_input().with_inserter_output()
│   │                                #     .with_predicate().get()/.count()/.first()
│   │                                #   Properties: charted_chunks, charted_tiles,
│   │                                #     charted_area_km2, tick, game_time_seconds,
│   │                                #     research, logistics, power, threat,
│   │                                #     has_damage, damaged_entities, recent_losses,
│   │                                #     ground_items.
│   │                                #   .state property: escape hatch to raw WorldState
│   │                                #     (for reward evaluator namespace compatibility).
│   ├── writer.py                    # WorldWriter — sole write interface.
│   │                                #   integrate_snapshot(snapshot): bridge use case;
│   │                                #     merges StateParser snapshot into live state
│   │                                #     with staleness guards and index rebuild.
│   │                                #   Section replacement: replace_entities(),
│   │                                #     replace_logistics(), replace_research(), etc.
│   │                                #   Fine-grained mutation: add_entity(),
│   │                                #     remove_entity(), update_entity_status(),
│   │                                #     update_entity_recipe(), update_entity_inventory(),
│   │                                #     update_player_position(), update_player_inventory(),
│   │                                #     update_player_health(), update_exploration().
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
│   └── production_tracker.py        # Throughput tracking — update() takes WorldQuery
│                                    #   (PROXIMAL — scan-radius scoped)
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
│   ├── goal_tree.py                 # ✅ GoalTree — LIFO preemption, subgoal completion
│   ├── reward_evaluator.py          # ✅ RewardEvaluator — eval() against WorldQuery;
│   │                                #   namespace includes wq (composable builder),
│   │                                #   state (WorldState escape hatch for compatibility),
│   │                                #   and all named condition helpers.
│   │                                #   See REWARD_NAMESPACE.md for full reference.
│   └── resource_allocator.py        # ✅ ResourceAllocator — pass-through until biters
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
│
├── config.py                        # All tunable parameters
│
└── tests/
    ├── fixtures.py                           # Shared: MockRconClient,
    │                                         #   make_world_state(), make_world_query(),
    │                                         #   make_inventory_entity()
    ├── test_core_dataclasses.py              ✅ 128 tests
    ├── unit/
    │   ├── bridge/
    │   │   ├── test_state_parser.py          ✅ ~50 tests (two-lane belts, inserter objects,
    │   │   │                                              charted_chunks; WorldQuery for
    │   │   │                                              staleness/inventory helpers)
    │   │   └── test_action_executor.py       ✅
    │   ├── world/
    │   │   ├── test_state.py                 ✅ ~120 tests
    │   │   │   # Section 1: core dataclass tests (Position, Inventory, WorldState,
    │   │   │   #   BeltLane/BeltSegment, ExplorationState, InserterState)
    │   │   │   # Section 2: WorldState index tests (direct internal field inspection)
    │   │   │   # Section 3: WorldQuery tests (all named lookups, connectivity,
    │   │   │   #   EntityQuery builder, fully_connected_entities)
    │   │   │   # Section 4: WorldWriter tests (section replacement, fine-grained
    │   │   │   #   mutation, integrate_snapshot)
    │   │   ├── test_knowledge.py             ✅ 75 tests
    │   │   ├── test_entities.py              ✅ 36 tests
    │   │   ├── test_tech_tree.py             ✅ 64 tests
    │   │   └── test_production_tracker.py   ✅ ~25 tests (uses make_world_query fixture)
    │   └── planning/
    │       ├── test_goal_tree.py             ✅ ~40 tests
    │       ├── test_reward_evaluator.py      ✅ ~30 tests (uses make_world_query fixture)
    │       └── test_resource_allocator.py   ✅ ~10 tests
    └── integration/
        ├── test_evaluator_capabilities.py   ✅ ~91 tests — capability matrix across
        │                                        14 condition categories (IV, EN, PR, RS,
        │                                        RM, TM, DM, CN, GI, EX, PT, ST, SC, XC)
        │                                        XC now includes wq builder compound query
        └── test_StateParser_WorldState.py   ✅ 9 tests (WorldQuery interface)
```

---

## Core Dataclasses

### WorldState (`world/state.py`)

A belief state snapshot of the game, not ground truth. Produced by `bridge/state_parser.py`.
**Not exported from `world/`; consumed only through `WorldQuery` and `WorldWriter`.**

Key fields:
- `tick`: game tick (60 ticks = 1 second)
- `observed_at`: `dict[str, int]` — per-section staleness tracking
- `player`: `PlayerState` — position, health, inventory, reachable entity ids,
  **`exploration: ExplorationState`** (charted_chunks, charted_tiles, charted_area_km2)
- `entities`: `list[EntityState]` — placed buildings, status, recipe, energy
- `resource_map`: `list[ResourcePatch]` — coarse strategic resource records
- `ground_items`: `list[GroundItem]` — items on the ground within scan radius
- `research`: `ResearchState` — unlocked techs, queue, science rates
- `logistics`: `LogisticsState` — **two-lane `BeltSegment` list**, power grid,
  **`dict[int, InserterState]`** with pickup/drop positions
- `damaged_entities`: `list[DamagedEntity]` — entities below full health (any cause)
- `destroyed_entities`: `list[DestroyedEntity]` — rolling window of destroyed entities
- `threat`: `ThreatState` — biter bases, pollution, attack timers (empty when biters off)

Internal indices (built by `__post_init__`, maintained by `WorldWriter`):
- `_by_id: dict[int, EntityState]`
- `_by_name: dict[str, list[EntityState]]`
- `_by_recipe: dict[str, list[EntityState]]`
- `_by_status: dict[EntityStatus, list[EntityState]]`
- `_inserters_from: dict[int, list[InserterState]] | None` — built lazily after integrate
- `_inserters_to: dict[int, list[InserterState]] | None` — built lazily after integrate

Convenience properties (still on WorldState, used by WorldQuery): `game_time_seconds`,
`has_damage`, `recent_losses`, `charted_chunks`.

### WorldQuery (`world/query.py`)

The sole read interface. Wraps a `WorldState` reference. Optionally wraps a
`KnowledgeBase` reference for tile-dimension-aware spatial queries.

```python
class WorldQuery:
    def __init__(self, state: WorldState, kb: KnowledgeBase | None = None)

    # Entity lookups (O(1) via index)
    def entity_by_id(self, entity_id: int) -> EntityState | None
    def entities_by_name(self, name: str) -> list[EntityState]
    def entities_by_status(self, status: EntityStatus) -> list[EntityState]
    def entities_by_recipe(self, recipe: str) -> list[EntityState]
    def all_entities(self) -> list[EntityState]

    # Connectivity queries
    def inserters_taking_from(self, entity_id, tile_width=1, tile_height=1) -> list[InserterState]
    def inserters_delivering_to(self, entity_id, tile_width=1, tile_height=1) -> list[InserterState]
    def inserters_taking_from_type(self, entity_name: str) -> list[InserterState]
    def inserters_delivering_to_type(self, entity_name: str) -> list[InserterState]

    # Compound convenience
    def fully_connected_entities(self, recipe: str) -> list[EntityState]

    # Composable builder
    def entities(self) -> EntityQuery  # → .with_name/.with_recipe/.with_status/
                                       #    .with_inserter_input/.with_inserter_output/
                                       #    .with_predicate → .get()/.count()/.first()

    # Resource / player / exploration
    def resources_of_type(self, resource_type: str) -> list[ResourcePatch]
    def inventory_count(self, item: str) -> int
    def player_position(self) -> Position
    def player_health(self) -> float

    # Properties
    charted_chunks: int           # NON-PROXIMAL
    charted_tiles: int            # NON-PROXIMAL
    charted_area_km2: float       # NON-PROXIMAL
    tick: int                     # NON-PROXIMAL
    game_time_seconds: float      # NON-PROXIMAL
    research: ResearchState       # NON-PROXIMAL
    logistics: LogisticsState     # PROXIMAL
    power: PowerGrid              # PROXIMAL
    threat: ThreatState
    has_damage: bool
    damaged_entities: list
    recent_losses: list
    ground_items: list

    def tech_unlocked(self, tech: str) -> bool
    def section_staleness(self, section: str, current_tick: int) -> int | None

    # Escape hatch for reward evaluator namespace compatibility
    @property
    def state(self) -> WorldState
```

### WorldWriter (`world/writer.py`)

The sole write interface. Wraps the same `WorldState` reference as `WorldQuery`.

```python
class WorldWriter:
    def __init__(self, state: WorldState)

    # Bridge use case — merges snapshot, staleness-guarded, rebuilds indices
    def integrate_snapshot(self, snapshot: WorldState) -> None

    # Section replacement (bridge or bulk updates)
    def replace_entities(self, entities: list[EntityState], tick: int) -> None
    def replace_logistics(self, logistics: LogisticsState, tick: int) -> None
    def replace_research(self, research: ResearchState, tick: int) -> None
    def replace_resource_map(self, patches: list[ResourcePatch], tick: int) -> None
    def replace_ground_items(self, items: list[GroundItem], tick: int) -> None
    def replace_player(self, player: PlayerState, tick: int) -> None
    def replace_damaged_entities(self, damaged: list[DamagedEntity], tick: int) -> None
    def replace_threat(self, threat: ThreatState, tick: int) -> None

    # Fine-grained mutation (execution layer)
    def add_entity(self, entity: EntityState) -> None
    def remove_entity(self, entity_id: int) -> bool
    def update_entity_status(self, entity_id: int, status: EntityStatus) -> bool
    def update_entity_recipe(self, entity_id: int, recipe: str | None) -> bool
    def update_entity_inventory(self, entity_id: int, inventory: Inventory) -> bool
    def update_player_position(self, position: Position) -> None
    def update_player_inventory(self, inventory: Inventory) -> None
    def update_player_health(self, health: float) -> None
    def update_exploration(self, exploration: ExplorationState) -> None
```

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

### GoalTree (`planning/goal_tree.py`)

Runtime manager for the agent's goal hierarchy. LIFO preemption stack: when a
higher-priority goal is added, the active goal is suspended and pushed onto the stack.
On resolution, suspended goals resume in most-recently-suspended order. Parent goals
auto-complete when all their children are complete. Single active goal at any time.

```python
class GoalTree:
    def add_goal(self, goal: Goal) -> None          # activates or preempts
    def active_goal(self) -> Goal | None
    def complete_active(self, tick: int) -> Goal | None
    def fail_active(self, tick: int, reason: str = "") -> Goal | None
    def pending_goals(self) -> list[Goal]           # sorted priority-descending
    def goal_by_id(self, goal_id: str) -> Goal | None
    def all_goals(self) -> list[Goal]
```

### RewardEvaluator (`planning/reward_evaluator.py`)

Evaluates `Goal.success_condition`, `Goal.failure_condition`, and
`RewardSpec.milestone_rewards` as Python expression strings against a controlled
namespace derived from the current `WorldQuery`. No LLM involved.

**Interface change from Phase 4:** `evaluate()` and `evaluate_conditions()` now accept
a `WorldQuery` argument (not `WorldState`). This enforces the access boundary.

See **`REWARD_NAMESPACE.md`** for the complete namespace reference. See **`CONDITION_SCOPE.md`**
for the PROXIMAL vs NON-PROXIMAL distinction.

```python
class RewardEvaluator:
    def __init__(self, tracker: Optional[ProductionTrackerProtocol] = None)
    def evaluate(self, goal: Goal, wq: WorldQuery, tick: int, start_tick: int) -> EvaluationResult
    def evaluate_conditions(self, success_condition, failure_condition, spec,
                            wq: WorldQuery, tick, start_tick) -> EvaluationResult
```

Key namespace entries (see `REWARD_NAMESPACE.md` for full list):

| Name | Scope | Description |
|---|---|---|
| `inventory(item)` | NON-PROXIMAL | Player item count |
| `charted_chunks` | NON-PROXIMAL | Force chart size (exploration) |
| `charted_area_km2` | NON-PROXIMAL | `charted_chunks × 1024 ÷ 1,000,000` |
| `tech_unlocked(tech)` | NON-PROXIMAL | Research state |
| `resources_of_type(t)` | NON-PROXIMAL | Accumulated resource patches |
| `tick` | NON-PROXIMAL | Current game tick |
| `entities(name)` | **PROXIMAL** | Entities in scan radius |
| `production_rate(item)` | **PROXIMAL** | Items/min via ProductionTracker |
| `power` | **PROXIMAL** | Nearest electric network |
| `inserters_from(id)` | **PROXIMAL** | Inserters taking from entity |
| `inserters_to(id)` | **PROXIMAL** | Inserters delivering to entity |
| `staleness(section)` | META | Ticks since section last observed |
| `wq` | — | The WorldQuery object; enables composable builder in conditions |
| `state` | — | The raw WorldState; backwards-compat escape hatch |

### ExplorationState (`world/state.py`)

```python
@dataclass
class ExplorationState:
    charted_chunks: int = 0          # LuaForce::get_chart_size(surface)
    charted_tiles: int               # computed: charted_chunks * 1024
    charted_area_km2: float          # computed: charted_tiles / 1_000_000
```

NON-PROXIMAL. Monotonically increasing. Lives on `PlayerState.exploration`.
Accessible via `WorldQuery.charted_chunks` (and `.charted_tiles`, `.charted_area_km2`).

### BeltLane and BeltSegment (`world/state.py`)

Each belt tile exposes two independent transport lanes. Mixed-item belts are correctly
modelled.

```python
@dataclass
class BeltLane:
    congested: bool; items: dict[str, int]
    def carries(self, item) -> bool
    def is_empty(self) -> bool
    def total_items(self) -> int

@dataclass
class BeltSegment:
    segment_id: int; positions: list[Position]
    lane1: BeltLane    # left  — get_transport_line(1)
    lane2: BeltLane    # right — get_transport_line(2)
    congested: bool    # computed: either lane congested
    items: dict[str, int]  # computed: merged across both lanes
    def carries(self, item) -> bool
```

### InserterState (`world/state.py`)

```python
@dataclass
class InserterState:
    entity_id: int; position: Position; active: bool
    pickup_position: Optional[Position]  # world coords of pickup arm
    drop_position: Optional[Position]    # world coords of drop arm
```

Enables connectivity queries via bounding-box containment. Either position may be
`None` (pcall-guarded in Lua).

### AuditReport (`agent/examiner/audit_report.py`)

Produced by either examiner mode. See original ARCHITECTURE documentation.

### Action types (`bridge/actions.py`)

21 concrete types across 10 categories. See original ARCHITECTURE documentation.

### AgentState (`agent/state_machine.py`)

Four states: `PLANNING`, `EXECUTING`, `EXAMINING`, `WAITING`.

---

## Bridge Layer

### `bridge/mod/control.lua` — State Query and Prototype Functions

The Lua mod exposes state query functions, action command handlers, and prototype
query functions.

**State queries added or updated in Phase 4:**

- `fa._player_table(player)` — now includes **`charted_chunks`** from
  `player.force.get_chart_size(player.surface)`. Global to the force; pcall-guarded.
- Belt reading — both transport lanes are now read in full via `get_transport_line(1)`
  and `get_transport_line(2)`. Each lane emits `{congested, items: {name: count}}`.
- Inserter records — each inserter now emits `{active, pickup_position, drop_position}`
  with world coordinates sourced from `ins.pickup_position` and `ins.drop_position`.
- `fa.get_exploration()` — new partial-state query returning `{tick, charted_chunks}`
  without a full state query, for lightweight polling.

**Prototype query functions** (unchanged, used by KnowledgeBase):
- `fa.get_entity_prototype(name)`, `fa.get_resource_prototype(name)`,
  `fa.get_fluid_prototype(name)`, `fa.get_recipe_prototype(name)`,
  `fa.get_technology(name)`

All five prototype functions return `{"ok": false, "reason": "..."}` if the prototype
does not exist.

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

# Shared world state wiring (done once, references shared for the run)
world_state = WorldState()
world_query = WorldQuery(world_state, kb)
world_writer = WorldWriter(world_state)

# Each tick cycle — bridge produces a snapshot, writer integrates it
snapshot = state_parser.parse(raw_json, current_tick=tick)
world_writer.integrate_snapshot(snapshot)

# Then downstream consumers use world_query (never world_state directly)
tech_tree.absorb_research_state(world_query.research)
production_tracker.update(world_query)
reward = evaluator.evaluate(goal, world_query, tick=tick, start_tick=start_tick)
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
2. **Bridge** ✅ — RCON client, Lua mod (two-lane belts, inserter positions, charted_chunks,
   fa.get_exploration), state parser, action executor — 80+ tests passing
3. **World model** ✅ — `KnowledgeBase` (SQLite), `ResourceRegistry`, `TechTree`,
   `ProductionTracker`, `ExplorationState`, `BeltLane/BeltSegment`, `InserterState`
   connectivity queries — 200+ tests passing
4. **Planning** ✅ — `GoalTree` (LIFO preemption), `RewardEvaluator` (full condition
   namespace + ProductionTracker + staleness guards), `ResourceAllocator` (pass-through)
   — ~80 unit tests + ~91 integration tests (capability matrix) passing
4a. **WorldQuery / WorldWriter refactor** ✅ — `WorldState` reduced to pure data container
   with internal indices; `WorldQuery` (read) and `WorldWriter` (write) enforce access
   boundary across codebase; `RewardEvaluator` and `ProductionTracker` updated to accept
   `WorldQuery`; `StateParser` rebuilds indices after parse — all tests passing
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

`tests/unit/world/test_state.py` — ~120 tests across four sections:
- Section 1: core dataclass tests (Position, Inventory, WorldState basics,
  BeltLane/BeltSegment, ExplorationState, InserterState)
- Section 2: WorldState index tests (direct internal field inspection — intentional
  for this file; other test files use fixtures helpers instead)
- Section 3: WorldQuery tests (all named lookups, connectivity queries, EntityQuery
  composable builder, fully_connected_entities)
- Section 4: WorldWriter tests (section replacement, fine-grained mutation,
  integrate_snapshot with staleness guards and deduplication)

`tests/unit/world/test_knowledge.py` — 75 tests covering all five registries, placeholder
lifecycle, enrichment, persistence across simulated restarts, SQL query methods, and
Windows SQLite file-lock safety.

`tests/unit/world/test_entities.py` — 36 tests covering the `ResourceRegistry` facade and
`get_entity_metadata()` delegation, placeholder defaults, and safety guarantees.

`tests/unit/world/test_tech_tree.py` — 64 tests covering empty-KB behaviour, prerequisite
queries, reachability, unlock queries, path planning, next-researchable, absorb lifecycle.

`tests/unit/world/test_production_tracker.py` — ~25 tests. Uses `make_world_query()` from
`tests/fixtures.py` so WorldState construction changes do not require edits here.

`tests/unit/bridge/test_state_parser.py` — ~50 tests covering two-lane belt parsing,
inserter object format, charted_chunks, and legacy format compatibility. The four tests
that check `section_staleness` and `inventory_count` use `WorldQuery(state).method()`
since these moved from `WorldState` to `WorldQuery`.

`tests/unit/planning/test_goal_tree.py` — ~40 tests covering LIFO preemption, subgoal
completion, pending goal promotion, and all status lifecycle transitions.

`tests/unit/planning/test_reward_evaluator.py` — ~30 tests. Uses `make_world_query()`
from `tests/fixtures.py`. Tests include `wq` namespace availability.

`tests/unit/planning/test_resource_allocator.py` — ~10 tests covering the pass-through
interface across all Priority levels.

### Shared fixtures (`tests/fixtures.py`)

`MockRconClient` — fake RCON client for bridge tests.

`make_world_state(tick, entities, inserter_activity)` — builds a minimal `WorldState`
for use as test scaffolding in files that don't test `WorldState` internals.

`make_world_query(tick, entities, inserter_activity)` — wraps `make_world_state` in a
`WorldQuery`. Use in `test_production_tracker.py`, `test_reward_evaluator.py`, and any
future test that needs a `WorldQuery` purely as input scaffolding.

`make_inventory_entity(entity_id, name, items, status)` — builds an `EntityState` with
a populated inventory for production tracker tests.

**Design rule:** `test_state.py` is the exception — it tests `WorldState` internals
directly (Section 2) because that is its purpose. All other test files use `make_world_query`
and never import `WorldState` for scaffolding purposes.

### Integration tests (no Factorio required)

`tests/integration/test_evaluator_capabilities.py` — **the capability matrix**. ~91 tests
proving that every supported condition category can be expressed and evaluated correctly
end-to-end against a constructed `WorldQuery`. Categories: IV (inventory), EN (entity
placement), PR (production/logistics including two-lane belts), RS (research), RM
(resource map), TM (time), DM (damage/destruction), CN (connectivity), GI (ground items),
EX (exploration / charted_chunks — NON-PROXIMAL), PT (production_rate via
ProductionTracker — PROXIMAL), ST (staleness guards), SC (proximal/non-proximal boundary),
XC (compound multi-category conditions, including `wq` composable builder).

`tests/integration/test_StateParser_WorldState.py` — 9 tests. Parses a JSON fixture
through `StateParser`, wraps the result in `WorldQuery`, and asserts all query methods
return correct results.

Adding a new condition type means adding a test to the capability matrix file first.

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
2. `CONDITION_SCOPE.md` — if the component generates or evaluates goal conditions
3. `REWARD_NAMESPACE.md` — if the component writes or interprets condition strings
4. `CONVERSATION_BRIEF.md` — the specific brief for this component
5. The actual source files for all interfaces the component must satisfy or consume
6. The test file — so the implementer can run and extend it

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
      `bridge/action_executor.py`, `bridge/mod/control.lua` (two-lane belts,
      inserter positions, charted_chunks, fa.get_exploration),
      `bridge/mod/info.json` — 80+ unit tests; in-game integration tests pending
- [x] World model — `world/knowledge.py`, `world/entities.py`, `world/tech_tree.py`,
      `world/production_tracker.py`, `world/state.py` (BeltLane/BeltSegment,
      InserterState, ExplorationState) — 200+ tests
- [x] Planning layer — `planning/goal_tree.py` (LIFO preemption, subgoal completion),
      `planning/reward_evaluator.py` (full namespace, ProductionTracker integration,
      staleness guards), `planning/resource_allocator.py` (pass-through)
      — ~80 unit tests + ~91 capability matrix integration tests
- [x] WorldQuery / WorldWriter access refactor — `world/query.py`, `world/writer.py`;
      `WorldState` reduced to pure data container with internal indices;
      `RewardEvaluator` and `ProductionTracker` updated to `WorldQuery` interface;
      `StateParser` rebuilds indices after parse; `tests/fixtures.py` shared helpers
      — all tests passing
- [x] `CONDITION_SCOPE.md` — proximal/non-proximal reference
- [x] `REWARD_NAMESPACE.md` — complete eval namespace reference
- [ ] Primitives
- [ ] Execution layer
- [ ] Examination layer
- [ ] Tactical layer
- [ ] Strategic layer
- [ ] Memory layer
- [ ] Main loop
- [ ] Threat module