# Factorio Agent

An agentic system that plays Factorio autonomously — setting its own goals, building
production chains, managing logistics, and eventually defending against biters.

The system is built around a reinforcement learning paradigm with minimal seeding. It is
not told how to play. It figures that out.

---

## What This Is

Factorio is a game about building factories. You mine ore, smelt it into plates, craft
those into components, use components to build machines that automate the mining and
smelting, then use those machines to research better machines. The loop compounds until
you launch a rocket.

It's also an excellent testbed for agentic AI systems. The game has:

- **Long time horizons** — a full run takes many hours, with decisions made early
  having consequences much later
- **Hierarchical planning** — every high-level goal ("research automation") decomposes
  into a tree of subgoals, each of which decomposes further
- **Rich feedback signals** — production rates, research progress, power levels, and
  eventually combat outcomes all provide continuous measurable signal
- **Competing priorities** — the resources that build your factory are the same ones
  that defend it from attack

This project is an attempt to build a system that handles all of that.

---

## Architecture Overview

The system is structured in layers, each with a clear responsibility:

### Game Bridge
A Lua mod running inside Factorio exposes game state and accepts commands via RCON.
A Python bridge translates between the game and the rest of the system. Nothing outside
the bridge layer speaks RCON or Lua — the rest of the system operates on structured
Python objects.

### World Model
A structured, agent-readable snapshot of the game state at any point in time: inventory,
placed entities and their status, resource locations, research progress, logistics health,
and (eventually) threat state. Updated continuously from the bridge.

### Planning Layer
A three-tier hierarchy:

- **Strategic**: long-horizon goal setting. Runs infrequently. Powered by an LLM.
- **Tactical**: subgoal decomposition. Runs when goals are set or conditions change. LLM.
- **Execution**: task execution. Runs continuously. Pure code — no LLM.

The LLM is treated as an expensive resource. The execution layer never calls it.
Target: 10–30 LLM calls per real-world hour of play.

### Self-Designed Rewards
When the LLM sets a goal, it simultaneously defines a reward specification for that goal —
success conditions, failure conditions, milestone rewards, and time discounting. Reward
evaluation during execution is entirely mechanical, requiring no further LLM calls.

After a goal resolves, a reflection call reviews whether the reward spec was well-calibrated.
That reflection is stored in memory and influences how future reward specs are designed.

### Self-Examination
The agent has a dedicated examination state — not a fallback, but a scheduled activity.
When the LLM is available, examination involves rich reflection: identifying bottlenecks,
assessing whether current goals are still the right ones, and flagging production systems
worth preserving as blueprints. When the LLM is rate-limited, a mechanical auditor runs
instead — tracking throughput, idle machines, starved inputs, and power headroom in pure
code. On LLM resumption, the accumulated audit report is passed as context before planning
resumes.

### Memory Across Runs
The system maintains three kinds of memory that persist across runs:

- **Episodic**: what happened each run, and why
- **Semantic**: Factorio mechanics the agent has discovered
- **Blueprint library**: production designs that worked well, extracted automatically
  when the examiner identifies a stable, high-performing system

The blueprint library starts empty. The agent populates it through play.

### Biter Defense
Biters are disabled in early development. The architecture supports enabling them without
structural changes — the threat module is a stub that satisfies the same interface as the
real implementation. The goal tree supports priority-based preemption and goal suspension
from day one. Enabling biters means implementing the threat module, not restructuring
the system.

---

## Current Status

The project is in early development. The architecture is designed; implementation has
not yet begun.

**Completed**
- Architecture design
- Project structure

**In Progress**
- Nothing yet

**Planned (in order)**
- Core dataclasses (`WorldState`, `Goal`, `RewardSpec`, `AuditReport`)
- Game bridge (RCON client, Lua mod, state parser, action executor)
- World model (entities, tech tree, production tracker)
- Planning layer (goal tree, reward evaluator, resource allocator)
- Action primitives (movement, crafting, building, mining)
- Execution layer
- Examination layer (mechanical auditor, rich examiner)
- Tactical LLM layer
- Strategic LLM layer
- Memory layer (episodic, semantic, blueprint curation)
- Main loop and state machine
- Threat module (biter defense)

---

## Project Structure

```
factorio-agent/
│
├── bridge/                  # Game interface — RCON, Lua mod, state parsing
├── world/                   # World model — WorldState and supporting data
├── agent/                   # Agent logic — loop, state machine, planning layers
│   ├── examiner/            # Self-examination (rich and mechanical modes)
│   ├── threat/              # Biter defense (stubbed until enabled)
│   └── primitives/          # Atomic game actions
├── planning/                # Goal tree, reward specs, resource allocation
├── memory/                  # Persistent memory across runs
│   └── blueprint_library/   # Agent-discovered production blueprints
├── llm/                     # LLM client, budget tracking, prompt templates
├── config.py                # All tunable parameters including BITERS_ENABLED
└── tests/                   # Full stack testable without Factorio running
```

Full detail in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Design Principles

**The bridge is a hard boundary.** Nothing outside `bridge/` speaks RCON. This means
the entire agent stack can be tested without Factorio running.

**The LLM is a sparse resource.** All API calls route through a single client with
budget tracking and rate limit handling. The execution layer is pure code.

**Rewards are intrinsic to goals.** The agent designs its own reward structures. There
is one externally fixed meta-reward it cannot redefine; everything else is self-specified.

**Examination is productive downtime.** Rate limiting doesn't halt the system — it
shifts it into mechanical audit mode. The agent is never blind on resumption.

**Learning is cumulative.** Memory, plan templates, and blueprints persist across runs.
Early runs are slow and exploratory. Later runs build on what worked.

---

## Requirements

*To be filled in as the stack is defined.*

- Python 3.11+
- Factorio (version TBD)
- Anthropic API access (claude.ai subscription — no additional API spend required)

---

## Getting Started

*To be written once initial implementation exists.*

---

## Notes

This is an experimental research project. The goal is to demonstrate genuine agentic
capability — long-horizon planning, self-directed learning, and priority management —
in a rich, unforgiving environment. Factorio doesn't forgive bad planning. Neither does
this system's reward function.
