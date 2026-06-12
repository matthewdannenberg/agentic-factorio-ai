# Condition Scope — Proximal vs Non-Proximal

**This document must be included in every LLM conversation that implements or
modifies the strategic layer, tactical layer, or goal-generation prompts.**

It describes a fundamental constraint on goal condition expressions: not all
conditions have the same evidence quality, and the LLM must understand the
difference when writing `success_condition`, `failure_condition`, and
`milestone_rewards` strings.

---

## The Core Problem

The `RewardEvaluator` evaluates condition expressions against a `WorldQuery`
object (which wraps the current `WorldState` belief state). But `WorldState` is
not ground truth about the entire game — it is a **partially-observed,
locally-scoped snapshot** assembled by the bridge from whatever the Lua mod
could see during the last scan.

Specifically, the bridge only queries entities, belts, inserters, and
production data **within `LOCAL_SCAN_RADIUS` tiles of the player** (default
32 tiles). Anything outside that radius is invisible to the current snapshot.

This means a factory with two smelting lines 200 tiles apart cannot be fully
observed at once. If the player is standing at line A, line B is stale or
absent from `state.entities`. A condition that checks whether line B is
working will silently evaluate against stale data — and may return False
even when line B is running fine.

---

## The Two Scopes

### PROXIMAL conditions

Evidence depends on **player proximity**. The condition is only reliable when
the player is standing near the relevant entities (within scan radius).

Evaluating a proximal condition while the player is far away may produce
**false negatives** — the goal appears incomplete when it is actually done.

Proximal namespace names:
- `entities(name)` — only entities in current scan radius
- `entities_by_status(status)` — same
- `entity_by_id(id)` — same
- `production_rate(item)` — only counts output from entities in scan radius
- `logistics` / `logistics.belts` / `logistics.inserters` — scan radius only
- `power` — scoped to the nearest electric pole's network
- `inserters_from(id)` / `inserters_to(id)` / `inserters_from_type(name)` / `inserters_to_type(name)` — scan radius
- `wq.entities().with_inserter_input()` / `.with_inserter_output()` — scan radius

**How to guard proximal conditions:**

Use the `staleness(section)` function, which returns the number of ticks since
a section was last observed, or `None` if it has never been observed:

```python
# Only evaluate if entities section is fresh (observed within last 5 seconds)
staleness('entities') is not None and staleness('entities') < 300 and len(entities('assembling-machine-1')) >= 3
```

Or write the goal so that **completion requires the player to be nearby** —
which is often the right design for factory-building tasks anyway (the agent
needs to walk to a site to verify it is working).

### NON-PROXIMAL conditions

Evidence is **global and accumulates across the run**. These are safe to
evaluate at any time regardless of where the player is standing.

Non-proximal namespace names:
- `inventory(item)` — player inventory travels with the player
- `tech_unlocked(tech)` — research state is global; unlocks persist
- `research.in_progress` / `research.queued` — global
- `resources_of_type(type)` — the resource map accumulates across all visits;
  once a patch is found it stays in the map permanently (patches don't move)
- `charted_chunks` — force chart size; grows monotonically as the player
  explores; sourced from `LuaForce::get_chart_size(surface)`; never decreases
- `charted_tiles` — `charted_chunks × 1024` (32×32 tiles per chunk)
- `charted_area_km2` — `charted_tiles ÷ 1,000,000` (Factorio tiles are 1m×1m)
- `tick` / `state.game_time_seconds` — always current
- `state.recent_losses` / `state.damaged_entities` — populated from the
  destruction event buffer, which is global (not scan-radius scoped)

### NAVIGATE conditions

`navigate_to(x, y)` — True when the player is within 1.5 tiles of `(x, y)`.
Used as the `success_condition` for `navigate` task-backed goals. Scope: PROXIMAL
(depends on player position), but used specifically as a completion trigger when
the player has arrived — so the staleness issue does not apply in practice.

```python
# Navigate to a specific position
success_condition="navigate_to(95.0, 95.0)"
```

This is generated automatically by `condition_parser` when `goal_type="navigate"`
is used in a `GoalQueueEntry`. It is also the `success_condition` written by
`_dispatch_as_task` for `TASK_GOAL_TYPES["navigate"]`.

### STRUCTURAL conditions

**Phase 10+ only. Requires a populated self-model.**

Evidence comes from the **factory self-model graph** — the agent's persistent,
graph-theoretic model of what it has built. STRUCTURAL conditions are globally
accurate (not scan-radius limited) and persistent (not subject to snapshot
staleness), but they can lag behind game reality if the examination layer has not
run recently.

STRUCTURAL conditions are the natural form for goals that describe factory
infrastructure rather than directly observable game state:

```python
# Does an active iron plate production line exist?
production_line('iron-plate') is not None

# Is throughput sufficient?
production_capacity('iron-plate') >= 30.0

# Is infrastructure present?
has_infrastructure('BELT_CORRIDOR')

# Self-model freshness guard (ticks since examination layer last reconciled)
sm_staleness() < 1800 and production_capacity('iron-plate') >= 30.0
```

STRUCTURAL conditions are evaluated by `RewardEvaluator` when a self-model is
provided alongside `WorldQuery`. When no self-model is available (early phases,
or before examination has run), structural conditions evaluate to `False` rather
than raising.

The `sm_staleness()` function returns the number of ticks since the examination
layer last reconciled the self-model against WorldState. Use it to guard structural
conditions the same way `staleness(section)` guards proximal conditions.

**These namespace entries are not yet implemented.** See `OPEN_DECISIONS.md` OD-6
for the design and the list of files to update when they are added (Phase 10).

---

## Rules for the Strategic/Tactical LLM

When generating a `success_condition` or `failure_condition` string:

1. **Prefer non-proximal conditions** for top-level strategic goals where
   possible. "Research automation", "collect 200 iron plates", or "explore
   50 chunks" are better top-level conditions than "have 3 working assemblers"
   because they don't depend on where the player is standing when checked.

2. **Use proximal conditions for subgoals** where the execution layer will
   naturally be nearby. "Place 3 assemblers and verify they are working" is a
   subgoal that the execution layer resolves locally; it is fine to use
   proximal conditions here.

3. **Guard proximal conditions with staleness** when the condition is written
   into a long-running goal that may be checked while the player is far away:
   ```python
   staleness('entities') is not None and staleness('entities') < 600 and \
   sum(1 for e in state.entities if e.status.value == 'working') >= 5
   ```

4. **Use `production_rate()` only for subgoals**, not top-level strategic
   conditions. Production rate is PROXIMAL — it only reflects the factory
   segment currently in scan radius. For strategic throughput goals, prefer
   inventory accumulation conditions instead:
   ```python
   # WRONG for a strategic goal (proximal, requires standing at the factory)
   production_rate('iron-plate') >= 60.0

   # BETTER (non-proximal, counts what has actually reached the player)
   inventory('iron-plate') >= 300
   ```

5. **`production_rate()` units are items per minute.** A rate of 60.0 means
   one item per second.

6. **Use `charted_chunks` to incentivise exploration.** This is the cleanest
   exploration metric — it is NON-PROXIMAL, monotonically increasing, and
   requires no proximity to evaluate. Useful for goals like:
   ```python
   # Success: explore at least 100 chunks
   charted_chunks >= 100

   # Failure: time limit expired without exploring enough
   tick > 7200 and charted_chunks < 20

   # Progressive milestones at exploration thresholds
   # milestone_rewards: {"charted_chunks >= 25": 0.1, "charted_chunks >= 100": 0.3}

   # Combined: must explore AND find resources
   charted_chunks >= 50 and len(resources_of_type('iron-ore')) >= 2
   ```

7. **`bbox().is_clear` requires prior navigation for large bboxes.** `bbox(x_min,
   y_min, x_max, y_max).is_clear` is PROXIMAL — it checks `wq.natural_objects_in_bbox()`
   which only sees objects within the current scan radius. If the bbox is larger than
   the scan radius, unobserved areas appear clear spuriously.

   The correct pattern is to navigate to the corners of the bbox before issuing a
   `clear_region` goal, ensuring the full region has been scanned:

   ```python
   # In test files: issue navigate goals to each corner first
   # success_condition for each: navigate_to(x, y)
   # Then issue clear_region goal:
   success_condition="bbox(-50,-50,50,50).is_clear"
   ```

   A scan coverage map (Phase 10) will eventually provide a principled solution.
   Until then, the staleness guard is also recommended:

   ```python
   # Guard so condition only fires when scan is fresh
   "staleness('natural_objects') is not None and "
   "staleness('natural_objects') < 300 and "
   "len(wq.natural_objects_in_bbox(-50,-50,50,50)) == 0"
   ```

   `charted_area_km2` is useful for human-readable goal descriptions:
   ```python
   charted_area_km2 >= 1.0   # explored at least 1 square kilometre
   ```

7. **Use the `wq` composable builder for compound entity conditions.** The
   `wq` name in the namespace gives access to the full `WorldQuery` interface,
   including the `EntityQuery` builder. This is more reliable than ad-hoc
   Python comprehensions over raw lists because it uses pre-built indices and
   correctly handles tile dimensions:
   ```python
   # Compound: assemblers with electronic-circuit recipe AND both ends wired
   wq.entities().with_recipe('electronic-circuit').with_inserter_input().with_inserter_output().count() >= 4

   # Or via the convenience method
   len(wq.fully_connected_entities('electronic-circuit')) >= 4
   ```
   Note that `with_inserter_input()` / `with_inserter_output()` are still
   PROXIMAL — they only see inserters within scan radius.

---

## The `staleness()` Function

Available in all condition expressions as `staleness(section_name)`.

```python
staleness('entities')       # ticks since entity scan, or None if never scanned
staleness('logistics')      # ticks since logistics (belts/inserters) scan
staleness('resource_map')   # ticks since resource scan
staleness('player')         # ticks since player state updated (always fresh)
staleness('research')       # ticks since research state updated
```

At 60 ticks per second: 60 = 1s, 300 = 5s, 1800 = 30s, 3600 = 1 min.

Returns `None` if the section has never been observed this session.
Always returns a non-negative integer if observed.

---

## Summary Table

| Namespace name | Scope | Notes |
|---|---|---|
| `inventory(item)` | NON-PROXIMAL | Always accurate |
| `tech_unlocked(tech)` | NON-PROXIMAL | Always accurate |
| `research.*` | NON-PROXIMAL | Always accurate |
| `resources_of_type(t)` | NON-PROXIMAL | Accumulates across visits |
| `charted_chunks` | NON-PROXIMAL | Global force chart, monotonically increasing |
| `charted_tiles` | NON-PROXIMAL | `charted_chunks × 1024` |
| `charted_area_km2` | NON-PROXIMAL | `charted_tiles ÷ 1,000,000` |
| `tick` / `game_time_seconds` | NON-PROXIMAL | Always accurate |
| `state.recent_losses` | NON-PROXIMAL | Global event buffer |
| `state.damaged_entities` | NON-PROXIMAL | Global scan |
| `entities(name)` | **PROXIMAL** | Scan radius only |
| `entity_by_id(id)` | **PROXIMAL** | Scan radius only |
| `production_rate(item)` | **PROXIMAL** | Scan radius + tracker window |
| `logistics.*` | **PROXIMAL** | Scan radius only |
| `power.*` | **PROXIMAL** | Nearest pole's network |
| `inserters_from/to*` | **PROXIMAL** | Scan radius only |
| `wq.entities().with_inserter_*` | **PROXIMAL** | Scan radius only |
| `staleness(section)` | META | Use to guard proximal conditions |
| `production_line(item)` | **STRUCTURAL** | Self-model: active producer node for item (Phase 10+) |
| `production_capacity(item)` | **STRUCTURAL** | Self-model: throughput in units/min (Phase 10+) |
| `has_infrastructure(type)` | **STRUCTURAL** | Self-model: any active node of type (Phase 10+) |
| `connected(node_a, node_b)` | **STRUCTURAL** | Self-model: path exists between nodes (Phase 10+) |
| `sm_staleness()` | META | Use to guard structural conditions (Phase 10+) |
| `navigate_to(x, y)` | PROXIMAL | True when player within 1.5 tiles; used as navigate task success_condition |
| `bbox(x,y,x,y).is_clear` | **PROXIMAL** | True when natural_objects_in_bbox is empty; requires prior navigation for large bboxes |
| `wq` | — | WorldQuery object; composable builder; non-proximal methods safe anywhere |
| `state` | — | Raw WorldState; backwards-compat escape hatch |

---

## Files to update when this changes

- `planning/reward_evaluator.py` — `_build_namespace()` comments and namespace dict
- `world/observable/query.py` — `WorldQuery` method docstrings
- `world/observable/state.py` — `ExplorationState` and `WorldState.charted_chunks`
- `bridge/mod/control.lua` — `fa._player_table()` and `fa.get_exploration()`
- `bridge/state_parser.py` — `_parse_player()`
- `llm/prompts/strategic.md` — condition-writing guidelines
- `llm/prompts/tactical.md` — condition-writing guidelines
- `tests/integration/test_evaluator_capabilities.py` — EX and XC categories

**Additional files to update when STRUCTURAL conditions are implemented (Phase 7-10):**
- `world/model/self_model.py` — self-model query interface exposed to evaluator
- `REWARD_NAMESPACE.md` — new STRUCTURAL namespace entries
- `OPEN_DECISIONS.md` — close out OD-6