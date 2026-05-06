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

In the long run, blueprint selection and orchestration (which blueprint, where, when, in what
sequence) is a meaningful planning problem even with a populated library.

---

## Project Structure

```
factorio-agent/
│
├── bridge/                          # Hard interface boundary — nothing outside speaks RCON
│   ├── rcon_client.py               # RCON connection, reconnection logic
│   ├── state_parser.py              # Raw Lua output → WorldState
│   ├── action_executor.py           # Action objects → RCON commands
│   └── mod/
│       └── control.lua              # Lua mod — exposes state, accepts commands
│
├── world/                           # Pure data — no LLM, no RCON
│   ├── state.py                     # WorldState dataclass
│   ├── entities.py                  # Entity type definitions and properties
│   ├── tech_tree.py                 # Research graph, unlock dependencies
│   └── production_tracker.py        # Throughput over time — feeds reward + curation
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
│   │   └── audit_report.py          # Shared dataclass — passed to LLM on resumption
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
│       └── blueprint.py             # Apply blueprint string at location
│
├── planning/
│   ├── goal.py                      # Goal, RewardSpec, Priority enum
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
├── config.py                        # All tunable parameters
│                                    # BITERS_ENABLED, LLM_CALL_BUDGET, TICK_INTERVAL
│
└── tests/
    ├── mock_game_state.py           # Fake WorldState — no Factorio needed
    ├── test_bridge.py
    ├── test_planning.py
    ├── test_examiner.py
    └── test_goal_tree.py
```

---

## Core Dataclasses

These are the interfaces everything else must satisfy. Defined in full before any other
component is built.

### WorldState (`world/state.py`)

The complete agent-readable snapshot of the game at a point in time. Produced by
`bridge/state_parser.py`. Consumed by planning, execution, and examination layers.

Key fields:
- `inventory`: player and chest contents
- `entities`: placed buildings, their recipes, status (working / idle / no_input / no_power)
- `resource_map`: ore patches, water, trees (sparse list)
- `research`: tech tree progress, unlocked technologies, current queue
- `logistics`: belt network status, power grid health, inserter activity
- `threat`: biter base locations, pollution extent, attack timers (empty when biters off)
- `timestamp`: game tick

### Goal (`planning/goal.py`)

Produced by the strategic or tactical LLM. Contains its own `RewardSpec`.

Key fields:
- `id`: unique identifier
- `description`: human-readable
- `priority`: Priority enum value
- `success_condition`: evaluable expression against WorldState
- `failure_condition`: evaluable expression against WorldState
- `reward_spec`: RewardSpec instance
- `parent_id`: for subgoals
- `status`: ACTIVE / SUSPENDED / COMPLETE / FAILED
- `created_at`, `resolved_at`

### RewardSpec (`planning/goal.py`)

Produced by the LLM alongside every Goal. Evaluated mechanically — no LLM during execution.

Key fields:
- `success_reward`: float
- `failure_penalty`: float
- `milestone_rewards`: dict mapping condition expressions to float rewards
- `time_discount`: float — reward decays the longer a goal takes
- `calibration_notes`: LLM's own notes on how to assess this spec post-resolution

### AuditReport (`agent/examiner/audit_report.py`)

Produced by either examiner mode. Passed to LLM as context on resumption.

Key fields:
- `timestamp`
- `mode`: RICH or MECHANICAL
- `starved_entities`: list of entities with missing inputs
- `idle_entities`: list of non-working machines
- `power_headroom`: current surplus/deficit
- `belt_congestion`: congested segments
- `production_rates`: throughput per item type over observation window
- `anomalies`: anything unexpected observed during the audit period
- `blueprint_candidates`: (rich mode only) systems flagged as extraction candidates
- `llm_observations`: (rich mode only) LLM's own summary of what it found

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

1. **Core dataclasses** — `WorldState`, `Goal`, `RewardSpec`, `AuditReport`, `Priority`
2. **Bridge** — RCON client + Lua mod + state parser + action executor
3. **World** — entities, tech tree, production tracker
4. **Planning** — goal tree, reward evaluator, resource allocator (stub)
5. **Primitives** — movement, crafting, building, mining
6. **Execution layer** — composes primitives against active goal
7. **Examination** — mechanical auditor first, rich examiner second
8. **Tactical layer** — subgoal decomposition via LLM
9. **Strategic layer** — long-horizon goal setting via LLM
10. **Memory** — episodic, semantic, plan library, blueprint curation
11. **Main loop + state machine** — ties everything together
12. **Threat module** — replaces stub when BITERS_ENABLED=True

---

## Component Conversation Brief Template

When opening a new conversation to implement a component, include:

1. This architecture document (or the relevant sections)
2. The component's contract:
   - What it receives (input types)
   - What it must return (output types)
   - What it must not do (e.g. no LLM calls, no RCON)
3. The interfaces of adjacent components it must satisfy
4. Any existing code that defines those interfaces
5. Current build status — what exists, what this conversation must produce

---

## Current Status

*Updated as components are completed.*

- [x] Architecture designed
- [ ] Core dataclasses
- [ ] Bridge / Lua mod
- [ ] World model
- [ ] Planning layer
- [ ] Primitives
- [ ] Execution layer
- [ ] Examination layer
- [ ] Tactical layer
- [ ] Strategic layer
- [ ] Memory layer
- [ ] Main loop
- [ ] Threat module
