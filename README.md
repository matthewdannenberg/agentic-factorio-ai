# Factorio Agent

An agentic system that plays Factorio autonomously using an LLM for strategic reasoning
and pure Python for execution. The agent sets its own goals, decomposes them into tasks,
executes them, and learns across runs.

## Status

**Phase 1 complete — core dataclasses.** All interfaces are defined and tested. No game
connection exists yet; the next phase is the bridge layer.

```
[✅] Core dataclasses     128 tests passing
[ ] Bridge / Lua mod
[ ] World model
[ ] Planning layer
[ ] Primitives
[ ] Execution layer
[ ] Examination
[ ] Tactical LLM layer
[ ] Strategic LLM layer
[ ] Memory
[ ] Main loop
[ ] Threat module (biters)
```

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full design documentation including:
- Key design decisions and their rationale
- All dataclass interfaces (WorldState, Goal, AuditReport, Action types)
- State machine transitions
- Build order
- Component conversation brief template for handoff between conversations

## Project Structure

```
factorio-agent/
├── bridge/
│   └── actions.py          ✅ 21 action types, ActionCategory, actions_for_context()
├── world/
│   └── state.py            ✅ WorldState and all sub-dataclasses
├── planning/
│   └── goal.py             ✅ Goal, RewardSpec, Priority, GoalStatus, make_goal()
├── agent/
│   ├── state_machine.py    ✅ AgentState, ExamineMode, assert_valid_transition()
│   └── examiner/
│       └── audit_report.py ✅ AuditReport, BoundingBox, BlueprintCandidate, damage types
└── tests/
    └── test_core_dataclasses.py  ✅ 128 tests, pure stdlib, no Factorio needed
```

## Running the Tests

No dependencies beyond Python 3.11+.

```bash
python tests/test_core_dataclasses.py
```

## Design Principles

**The bridge is a hard boundary.** Nothing outside `bridge/` speaks RCON or knows about Lua.
The entire agent stack runs against mock `WorldState` objects without Factorio installed.

**LLM calls are expensive.** Target: 10–30 per real-world hour. The execution layer is pure
code. The LLM is called for strategic goal-setting, tactical decomposition, and examination
— never during primitive execution.

**WorldState is a belief state.** It is a cached, partially-observed snapshot assembled
by the bridge. Different sections have different staleness. The `observed_at` dict tracks
per-section freshness. Consumers must not treat it as ground truth.

**Resource types are strings.** Factorio internal names (`"iron-ore"`, `"crude-oil"`) are
used throughout. `ResourceName` provides named constants for vanilla resources. The
`ResourceRegistry` (to be implemented in `world/entities.py`) extends itself at runtime
when unfamiliar resource names appear — making the system compatible with Space Age and
modded content without code changes.

**Structural damage is cause-agnostic.** `DamagedEntity` and `DestroyedEntity` live on
`WorldState` directly. They are populated for any cause: biter attack, vehicle collision,
player error, deconstruction. `ThreatState` is strictly biter-specific.

**Actions are primitive and categorised.** `bridge/actions.py` contains only single-RCON-
call operations. Composition is the execution layer's job. Every action declares an
`ActionCategory`; `actions_for_context()` returns the valid set for the current situation
(on foot vs in vehicle, biters on vs off).

**Blueprints are agent-discovered.** `memory/blueprint_library/library.json` starts empty.
The rich examiner nominates regions via `BlueprintCandidate`; the curator extracts and
stores them. Evaluation (throughput/tile, tech tier, improvability) is in `BlueprintRecord`
in the memory layer, keeping nomination and evaluation separate.

**Circuit networks are deferred.** The agent can play a complete game without them. Adding
them later requires new action subclasses and Lua mod extensions; no existing code changes.

## Passing Context to the Next Conversation

When starting a new conversation to implement the next component:

1. Include `ARCHITECTURE.md` — the full design and all interface specifications
2. Include the actual source files for everything the new component consumes or produces
3. Include `tests/test_core_dataclasses.py` — implementer should run it to verify
   their changes don't break existing contracts
4. Write a component brief using the template in `ARCHITECTURE.md`

For the bridge (next phase), the files to include are:
- `ARCHITECTURE.md`
- `world/state.py` — the parser must produce this
- `bridge/actions.py` — the executor must consume this
- `tests/test_core_dataclasses.py` — must still pass after bridge is added

The brief should specify: RCON protocol details, what the Lua mod must expose, what
`state_parser.py` must handle (partial state, unknown resource names, staleness tracking),
and what `action_executor.py` must not do (no LLM calls, no state mutation, one RCON
call per action).