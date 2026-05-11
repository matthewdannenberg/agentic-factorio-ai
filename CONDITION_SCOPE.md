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
| `wq` | — | WorldQuery object; composable builder; non-proximal methods safe anywhere |
| `state` | — | Raw WorldState; backwards-compat escape hatch |

---

## Files to update when this changes

- `planning/reward_evaluator.py` — `_build_namespace()` comments and namespace dict
- `world/query.py` — `WorldQuery` method docstrings
- `world/state.py` — `ExplorationState` and `WorldState.charted_chunks`
- `bridge/mod/control.lua` — `fa._player_table()` and `fa.get_exploration()`
- `bridge/state_parser.py` — `_parse_player()`
- `llm/prompts/strategic.md` — condition-writing guidelines
- `llm/prompts/tactical.md` — condition-writing guidelines
- `tests/integration/test_evaluator_capabilities.py` — EX and XC categories