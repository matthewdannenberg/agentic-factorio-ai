# Reward Evaluator Namespace Reference

This document is the complete reference for every name available inside goal condition
expressions evaluated by `planning/reward_evaluator.py`.

**Include this document when:**
- Implementing the strategic LLM layer (`agent/strategic.py`)
- Implementing the tactical LLM layer (`agent/tactical.py`)
- Writing or reviewing LLM prompts in `llm/prompts/`
- Adding new condition types to the evaluator

Also read `CONDITION_SCOPE.md` alongside this document — it explains which conditions
are PROXIMAL (only reliable near the player) and which are NON-PROXIMAL (globally
accurate), and gives rules for how to use them correctly in goals.

---

## How Conditions Work

Goal conditions are Python expression strings stored on `Goal` objects:

```python
goal.success_condition  = "inventory('iron-plate') >= 200"
goal.failure_condition  = "tick > 7200 and inventory('iron-plate') < 50"
```

Milestone conditions are keys in `RewardSpec.milestone_rewards`:

```python
spec.milestone_rewards = {
    "inventory('iron-plate') >= 50":  0.1,
    "inventory('iron-plate') >= 100": 0.2,
}
```

All three are evaluated by `RewardEvaluator.evaluate_conditions()` using Python's `eval()`
against the namespace defined here. Any expression that evaluates to a truthy value
triggers its associated effect. Any exception is caught, logged, and treated as
non-triggering — the goal is not failed due to a bad expression.

---

## Top-Level Objects

### `wq`
Type: `WorldQuery`

The `WorldQuery` interface over the current belief state. Provides O(1) indexed lookups,
connectivity queries, composable filtering, and all named helper methods. **This is the
preferred way to query game state in conditions.** Both proximal and non-proximal
conditions are available through `wq`.

```python
# Named lookups
wq.entity_by_id(42)
wq.entities_by_name('assembling-machine-1')
wq.entities_by_recipe('electronic-circuit')
wq.inventory_count('iron-plate')
wq.resources_of_type('iron-ore')
wq.tech_unlocked('automation')
wq.section_staleness('entities', tick)

# Composable builder
wq.entities().with_recipe('electronic-circuit').with_inserter_input().with_inserter_output().count()

# Convenience compound query
wq.fully_connected_entities('electronic-circuit')

# Properties
wq.charted_chunks
wq.charted_tiles
wq.charted_area_km2
wq.tick
wq.research
wq.logistics
wq.power
```

### `state`
Type: `WorldState`

The underlying `WorldState` data object. Exposed for backwards compatibility with
condition strings that use `state.` prefix. Prefer named namespace entries or `wq`
methods over `state.` where possible, but `state.` patterns remain fully supported.

```python
state.tick                     # current game tick (also available as tick)
state.game_time_seconds        # tick / 60.0
state.has_damage               # True if any damaged_entities present
state.recent_losses            # list[DestroyedEntity]
state.damaged_entities         # list[DamagedEntity]
state.destroyed_entities       # list[DestroyedEntity]
state.entities                 # list[EntityState] — scan radius only
state.ground_items             # list[GroundItem] — scan radius only
state.resource_map             # list[ResourcePatch] — accumulated
```

### `tick`
Type: `int`
Scope: NON-PROXIMAL

Current game tick. 60 ticks = 1 second.

```python
tick >= 3600          # at least 1 minute has passed
tick > 7200           # more than 2 minutes
```

---

## Inventory

### `inventory(item: str) -> int`
Scope: **NON-PROXIMAL**

Returns the player's current count of the named item. Aggregates across all inventory
slots. Returns 0 if the item is not present. Equivalent to `wq.inventory_count(item)`.

```python
inventory('iron-plate') >= 200
inventory('iron-gear-wheel') >= 50
inventory('coal') == 0
inventory('iron-plate') >= 100 and inventory('copper-plate') >= 100
```

---

## Exploration

### `charted_chunks`
Type: `int`
Scope: **NON-PROXIMAL**

Total number of 32×32-tile chunks revealed by this force. Sourced from
`LuaForce::get_chart_size(surface)`. Monotonically increasing — never decreases.
Accurate anywhere, regardless of player position. Equivalent to `wq.charted_chunks`.

### `charted_tiles`
Type: `int`
Scope: **NON-PROXIMAL**

`charted_chunks × 1024`. Total revealed tiles (32² = 1024 tiles per chunk).
Equivalent to `wq.charted_tiles`.

### `charted_area_km2`
Type: `float`
Scope: **NON-PROXIMAL**

`charted_tiles ÷ 1,000,000`. Factorio tiles are nominally 1m × 1m in lore.
Equivalent to `wq.charted_area_km2`.

```python
charted_chunks >= 50            # explored at least 50 chunks
charted_area_km2 >= 1.0         # explored at least 1 km²
charted_tiles >= 51200          # 50 chunks * 1024

# Failure: time expired without enough exploration
tick > 7200 and charted_chunks < 20

# Progressive milestones
# milestone_rewards: {"charted_chunks >= 25": 0.1, "charted_chunks >= 100": 0.3}

# Combined with resource discovery
charted_chunks >= 50 and len(resources_of_type('iron-ore')) >= 3
```

---

## Research

### `tech_unlocked(tech: str) -> bool`
Scope: **NON-PROXIMAL**

Returns True if the named technology is in the force's researched set.
Equivalent to `wq.tech_unlocked(tech)`.

### `research`
Type: `ResearchState`
Scope: **NON-PROXIMAL**

Equivalent to `wq.research`.

```python
research.in_progress            # str | None — currently researching
research.queued                 # list[str] — queued techs
research.unlocked               # list[str] — all unlocked
research.science_per_minute     # dict[str, float]
```

```python
tech_unlocked('steel-processing')
tech_unlocked('automation') and tech_unlocked('logistics')
research.in_progress == 'advanced-electronics'
len(research.queued) >= 2
```

---

## Resource Map

### `resources_of_type(resource_type: str) -> list[ResourcePatch]`
Scope: **NON-PROXIMAL**

Returns all resource patches of the named type accumulated across all visits.
Equivalent to `wq.resources_of_type(resource_type)`.
Each patch has `.amount`, `.size`, `.position`, `.observed_at`.

```python
len(resources_of_type('iron-ore')) >= 1
len(resources_of_type('coal')) == 0
any(p.amount >= 50000 for p in resources_of_type('iron-ore'))
any(p.size >= 100 for p in resources_of_type('iron-ore'))

# All required types found
(len(resources_of_type('iron-ore')) >= 1 and
 len(resources_of_type('copper-ore')) >= 1 and
 len(resources_of_type('coal')) >= 1)
```

---

## Entities

### `entities(name: str) -> list[EntityState]`
Scope: **PROXIMAL** — scan radius only

Returns all placed entities of the named type currently within `LOCAL_SCAN_RADIUS`.
Equivalent to `wq.entities_by_name(name)`.
Each `EntityState` has `.entity_id`, `.name`, `.position`, `.status`, `.recipe`,
`.inventory`, `.energy`.

`EntityStatus` values (use `.status.value` for string comparison):
`"working"`, `"idle"`, `"no_input"`, `"no_power"`, `"full_output"`, `"unknown"`

### `entity_by_id(entity_id: int) -> EntityState | None`
Scope: **PROXIMAL** — scan radius only

Equivalent to `wq.entity_by_id(entity_id)`.

### `entities_by_status(status: EntityStatus) -> list[EntityState]`
Scope: **PROXIMAL** — scan radius only

Equivalent to `wq.entities_by_status(status)`.

```python
len(entities('assembling-machine-1')) >= 3
len(entities('iron-chest')) >= 5

any(e.recipe == 'iron-gear-wheel' for e in entities('assembling-machine-1'))

any(e.status.value == 'working' and e.recipe == 'electronic-circuit'
    for e in entities('assembling-machine-2'))

# Count working entities
sum(1 for e in state.entities if e.status.value == 'working') >= 5

# Failure: any entity is starved
any(e.status.value == 'no_input' for e in state.entities)
```

### `wq` composable builder for entity conditions
Scope: **PROXIMAL** for inserter predicates; **same as entities()** otherwise

The `wq.entities()` builder chains multiple predicates and executes them efficiently
using pre-built indices. Prefer this over ad-hoc comprehensions for compound conditions:

```python
# Assemblers with recipe AND both input and output inserters
wq.entities().with_recipe('electronic-circuit').with_inserter_input().with_inserter_output().count() >= 4

# Via convenience method (equivalent)
len(wq.fully_connected_entities('electronic-circuit')) >= 4

# With status filter
wq.entities().with_name('assembling-machine-1').with_status(EntityStatus.WORKING).count() >= 3

# Custom predicate
wq.entities().with_predicate(lambda e: e.energy > 50000).count() >= 2
```

---

## Production Rates

### `production_rate(item: str) -> float`
Scope: **PROXIMAL** — scan radius only, tracker-backed

Returns smoothed items-per-minute rate for the named item over the tracker's rolling
window (default 3600 ticks = 60 seconds). Returns 0.0 if:
- No `ProductionTracker` is attached to the evaluator
- Fewer than two snapshots have been recorded
- The item has never been observed

**Units: items per minute.** 60.0 = one per second.

```python
production_rate('iron-plate') >= 60.0     # at least 60/min
production_rate('copper-plate') >= 30.0

# Failure: throughput dropped below minimum
production_rate('iron-plate') < 30.0

# Milestone: first throughput threshold
# milestone_rewards: {"production_rate('iron-plate') >= 30.0": 0.2}
```

**Important:** this is PROXIMAL. For strategic goals, prefer inventory accumulation:
```python
# BETTER for a top-level goal than production_rate()
inventory('iron-plate') >= 500
```

See `CONDITION_SCOPE.md` Rule 4.

---

## Logistics

### `logistics`
Type: `LogisticsState`
Scope: **PROXIMAL** — scan radius only

Equivalent to `wq.logistics`.

```python
logistics.belts          # list[BeltSegment]
logistics.inserters      # dict[int, InserterState]
logistics.power          # PowerGrid
logistics.inserter_activity  # dict[int, int] — legacy, prefer inserters
```

### `power`
Type: `PowerGrid`
Scope: **PROXIMAL** — nearest electric pole's network

Equivalent to `wq.power`.

```python
power.produced_kw        # float
power.consumed_kw        # float
power.accumulated_kj     # float
power.headroom_kw        # produced - consumed
power.is_brownout        # satisfaction < 1.0
power.satisfaction       # float in [0.0, 1.0]
```

```python
power.produced_kw >= 500
power.headroom_kw >= 100
not power.is_brownout

# Failure
power.is_brownout
power.satisfaction < 0.8
```

### Belt conditions

`logistics.belts` is a `list[BeltSegment]`. Each segment has:

```python
seg.lane1              # BeltLane — left transport line
seg.lane2              # BeltLane — right transport line
seg.congested          # bool — either lane congested
seg.items              # dict[str, int] — merged across both lanes
seg.carries(item)      # True if either lane carries item

# Per-lane:
seg.lane1.congested
seg.lane1.carries('iron-plate')
seg.lane1.items        # dict[str, int]
seg.lane1.total_items()
seg.lane1.is_empty()
```

```python
any(b.congested for b in logistics.belts)
not any(b.congested for b in logistics.belts)
any(b.lane1.carries('iron-plate') for b in logistics.belts)
any(b.lane1.carries('iron-plate') and b.lane2.carries('copper-plate')
    for b in logistics.belts)
any(b.items.get('iron-plate', 0) >= 6 for b in logistics.belts)
```

### Inserter activity

```python
# Count active inserters
sum(1 for i in logistics.inserters.values() if i.active) >= 5
```

---

## Connectivity

### `inserters_from(entity_id: int) -> list[InserterState]`
Scope: **PROXIMAL** — scan radius only

Inserters whose `pickup_position` falls within the bounding box of the entity.
Equivalent to `wq.inserters_taking_from(entity_id)`. For non-1×1 entities, pass
tile dimensions explicitly: `wq.inserters_taking_from(entity_id, tile_width=3, tile_height=3)`.

### `inserters_to(entity_id: int) -> list[InserterState]`
Scope: **PROXIMAL** — scan radius only

Inserters whose `drop_position` falls within the bounding box of the entity.
Equivalent to `wq.inserters_delivering_to(entity_id)`.

### `inserters_from_type(entity_name: str) -> list[InserterState]`
Scope: **PROXIMAL** — scan radius only

All inserters taking from any entity of the named type.
Equivalent to `wq.inserters_taking_from_type(entity_name)`.

### `inserters_to_type(entity_name: str) -> list[InserterState]`
Scope: **PROXIMAL** — scan radius only

All inserters delivering to any entity of the named type.
Equivalent to `wq.inserters_delivering_to_type(entity_name)`.

```python
len(inserters_from(entity_id)) >= 1     # at least one inserter taking from entity
len(inserters_to(entity_id)) >= 1       # at least one inserter delivering to entity

# Both ends connected
len(inserters_to(1)) >= 1 and len(inserters_from(1)) >= 1

# Active inserter feeding
any(i.active for i in inserters_to(entity_id))

# By type
len(inserters_from_type('iron-chest')) >= 2
len(inserters_to_type('assembling-machine-1')) >= 1
```

---

## Damage and Destruction

All accessed via `state.`. Scope: **NON-PROXIMAL** (global event buffer).

```python
state.has_damage                        # True if damaged_entities non-empty
state.damaged_entities                  # list[DamagedEntity]
state.recent_losses                     # list[DestroyedEntity]
state.destroyed_entities                # same as recent_losses

# DamagedEntity fields: entity_id, name, position, health_fraction, observed_at
# DestroyedEntity fields: name, position, destroyed_at, cause

# Failure conditions
state.has_damage
any(e.health_fraction < 0.3 for e in state.damaged_entities)

# Biter activity
any(e.cause == 'biter' for e in state.recent_losses)
len([e for e in state.recent_losses if e.cause == 'biter']) >= 3
```

---

## Ground Items

Accessed via `state.ground_items`. Scope: **PROXIMAL** — scan radius only.

```python
# GroundItem fields: item, position, count, observed_at, age_ticks

len(state.ground_items) == 0
any(g.item == 'iron-plate' for g in state.ground_items)
sum(g.count for g in state.ground_items if g.item == 'iron-plate') >= 20
```

---

## Staleness Guard

### `staleness(section: str) -> int | None`
Scope: META

Returns ticks since `section` was last observed, or `None` if never observed this session.
Use to guard PROXIMAL conditions against stale data.
Equivalent to `wq.section_staleness(section, tick)`.

Section names: `'player'`, `'entities'`, `'resource_map'`, `'ground_items'`,
`'research'`, `'logistics'`, `'damaged_entities'`, `'destroyed_entities'`, `'threat'`

At 60 tps: 60 = 1s, 300 = 5s, 600 = 10s, 1800 = 30s, 3600 = 1 min.

```python
# Guard pattern: only evaluate if fresh
(staleness('entities') is not None and
 staleness('entities') < 300 and
 len(entities('assembling-machine-1')) >= 3)

# In a failure condition: don't falsely fail if data is stale
(staleness('entities') is not None and
 staleness('entities') < 600 and
 len(entities('stone-furnace')) == 0)
```

`None` means the section has never been observed. An integer means it was observed
that many ticks ago. Distinguish them explicitly:

```python
staleness('entities') is None      # never seen
staleness('entities') is not None  # seen at least once
staleness('entities') == 0         # observed this exact tick
```

---

## Threat

### `threat`
Type: `ThreatState`
Scope: NON-PROXIMAL (populated from destruction event buffer; empty when biters off)

Equivalent to `wq.threat`.

```python
threat.is_empty           # True when BITERS_ENABLED=False
threat.biter_bases        # list[BiterBase]
threat.evolution_factor   # float
```

---

## Common Patterns

### Strategic goal (non-proximal success, time limit failure)

```python
success_condition = "inventory('iron-plate') >= 300"
failure_condition = "tick > 14400"    # 4 minutes
```

### Explore then gather

```python
success_condition = (
    "charted_chunks >= 50 and "
    "len(resources_of_type('iron-ore')) >= 2 and "
    "len(resources_of_type('copper-ore')) >= 1"
)
failure_condition = "tick > 18000"   # 5 minutes
```

### Research gate

```python
success_condition = (
    "tech_unlocked('steel-processing') and "
    "inventory('steel-plate') >= 50"
)
```

### Guarded factory check (proximal, with staleness)

```python
success_condition = (
    "staleness('entities') is not None and "
    "staleness('entities') < 300 and "
    "sum(1 for e in state.entities if e.status.value == 'working') >= 5"
)
```

### Compound entity condition using wq builder (proximal)

```python
success_condition = (
    "staleness('entities') is not None and "
    "staleness('entities') < 300 and "
    "wq.entities()"
    "   .with_recipe('electronic-circuit')"
    "   .with_inserter_input()"
    "   .with_inserter_output()"
    "   .count() >= 4"
)
```

### Progressive milestones

```python
milestone_rewards = {
    "charted_chunks >= 10":  0.05,
    "charted_chunks >= 25":  0.10,
    "charted_chunks >= 50":  0.20,
    "inventory('iron-plate') >= 100": 0.15,
}
```

### Both-sides connectivity check

```python
success_condition = (
    "staleness('entities') is not None and "
    "staleness('entities') < 300 and "
    "len(inserters_to(1)) >= 1 and "
    "len(inserters_from(1)) >= 1 and "
    "any(e.status.value == 'working' for e in entities('assembling-machine-1'))"
)
```

---

## Adding a New Condition Type

1. Add the new function or value to `_build_namespace()` in `planning/reward_evaluator.py`
2. Add a test to the appropriate category in `tests/integration/test_evaluator_capabilities.py`
3. Classify it as PROXIMAL or NON-PROXIMAL and add it to the summary table in `CONDITION_SCOPE.md`
4. Update this document with the new entry
5. Update `CONDITION_SCOPE.md`'s "Files to update" list if needed