# Open Implementation Decisions

This document tracks architectural questions that are explicitly deferred pending
research or experimentation. Each decision is isolated behind a protocol boundary
so that answering it requires changing an implementation, not restructuring
surrounding code.

When a decision is made, update this document with the decision and rationale,
then update the relevant section of `ARCHITECTURE.md`.

---

## OD-1 — Observation Space Construction

**Question:** How are agent observation spaces constructed and structured?

**What we know:**
- Observations should be goal-conditioned — state expressed relative to goal
  structure rather than in absolute game terms
- The KnowledgeBase production chain queries make goal-relative construction tractable

**What is undecided:**
- Whether observation spaces are fixed per goal type or evolve over time
- Whether a single unified observation space or per-agent spaces
- Whether the representation should be high-dimensional and fixed, or lower-dimensional and adaptive

**Where the decision lives:** `execution/observation.py` and individual agent implementations.

---

## OD-2 — Coordinator Learning

**Question:** Does the coordinator transition from rule-based to learned, and if so, how and when?

**What we know:**
- The coordinator starts rule-based
- A learned coordinator could handle more complex goal-agent routing and subtask sequencing
- The coordinator satisfies `CoordinatorProtocol` regardless of implementation

**What is undecided:**
- Whether coordinator learning is necessary or whether rule-based is sufficient
- If learned: which MARL coordination approach
- When to transition

**Where the decision lives:** `execution/coordinator/coordinator.py`.

---

## OD-3 — Self-Model Cross-Run Persistence

**Question:** What self-model information survives a run and in what form?

**What we know:**
- The self-model graph starts empty each run
- Something useful should cross the run boundary into behavioral memory
- Candidates: subgraph summaries, spatial patterns, throughput benchmarks

**What is undecided:**
- Exactly what is extracted at end-of-run
- How spatial patterns are represented in behavioral memory
- Whether pattern similarity matching is needed

**Where the decision lives:** `world/model/self_model.py` and `memory/behavioral.py`.

---

## OD-4 — RL Algorithm Family

**Question:** Which RL algorithm family do individual agents use?

**What we know:**
- The system needs to run in approximate real-time — the agent makes decisions
  every `TICK_INTERVAL` ticks, giving more compute budget than it might appear
- Goals are coarse and long-horizon; dense intermediate rewards from `agent/reward.py`
  are needed to make learning tractable
- Behavioral memory warm-starting means agents don't start from scratch on familiar
  goal types

**What is undecided:**
- Whether policy gradient (PPO, A3C), value-based (DQN variants), or actor-critic
  approaches are most suitable
- Whether to use a single shared network across agents or per-agent networks
- How to handle the partial observability introduced by scan radius limits
- Online vs offline learning (learning during play vs between runs)

**Where the decision lives:** individual agent implementations.

---

## OD-5 — Spatial-Logistics Internal Structure

**Question:** Is the spatial-logistics concern handled by one agent or two coordinating agents?

**What we know:**
- Spatial reasoning (layout, placement, region designation) and logistics reasoning
  (belt routing, inserter placement, throughput) are deeply interdependent
- Separating them creates coordination overhead that may exceed the benefit

**What is undecided:**
- Whether a single agent with richer internal structure handles both
- Or two agents with heavy blackboard coordination

**Where the decision lives:** `execution/agents/spatial_logistics.py`.

---

## OD-6 — RewardEvaluator Namespace Extension for Self-Model

**Question:** Should `RewardEvaluator` be extended to evaluate conditions against
the self-model in addition to `WorldQuery`, and what does that namespace look like?

**What we know:**
- Many goals the LLM will set are structural ("establish iron plate production")
  and are more naturally evaluated against the self-model
- The self-model is non-proximal and not subject to scan-radius staleness

**Proposed namespace entries (tentative):**
```python
production_line(item)       # -> FactoryNode | None
production_capacity(item)   # -> float — throughput in units per minute
has_infrastructure(type)    # -> bool
connected(node_a, node_b)   # -> bool
sm_staleness()              # -> int — ticks since examination last reconciled
```

**Where the decision lives:** `planning/reward_evaluator.py` and `CONDITION_SCOPE.md`.

**When relevant:** Phase 7, when the self-model is reliably populated.

**Files to update when implemented:**
- `planning/reward_evaluator.py`
- `CONDITION_SCOPE.md`
- `REWARD_NAMESPACE.md`
- `ARCHITECTURE.md`
- `world/model/self_model.py`
- `tests/integration/test_evaluator_capabilities.py`

---

## OD-7 — NodeType-Specific FactoryNode Attributes

**Question:** Should different NodeType values have different attributes, modelled
through `FactoryNode` subclasses?

**What we know:**
- The current `FactoryNode` carries a `throughput: dict[str, float]` field
  that is the primary structured attribute beyond geometry and status. This is
  sufficient for production-related queries (`find_producers`, `find_capacity`)
  and for the examination layer through Phase 9.
- Different node types plausibly need type-specific data. A `POWER_GRID` node
  would benefit from `produced_kw` / `consumed_kw` fields. A `TRAIN_STATION`
  might carry `scheduled_resources`. A `DEFENDED_REGION` might carry
  `turret_coverage_fraction`. None of these fit cleanly into `throughput`.
- Subclassing `FactoryNode` per NodeType is the natural Python pattern for
  this, and is localized to `world/model/self_model.py` and the examination layer.

**What is undecided:**
- Whether `throughput` plus ad-hoc extra fields is sufficient through Phase 10
- If subclasses: whether graph queries need to return typed subclasses

**When relevant:** Phase 10.

**Where the decision lives:** `world/model/self_model.py`.

---

## OD-8 — Agent Architecture: Thin Protocol vs Behavioral Composition

**Question:** Is the thin `AgentProtocol` + single-responsibility design the right
long-term architecture, or should agents compose reusable behavioral units internally?

**Conclusion:**
The `execution/skills/` directory was established to contain reusable behavioral units.
The behavior within skills is largely logic-free - actual decisions are made by agents,
the skills serve as replicable patterns which can execute extended sequences of commands
in order to achieve a particular goal. These can be used by multiple agents.
Examples include NavigateSkill, DestroySkill, CraftSkill, MineSkill, etc.

---

## OD-9 — WorldState Dialect

**Question:** Should WorldState translate Factorio internals into a convenient
agent-facing representation, rather than exposing raw game concepts directly?

**The problem:**
The current WorldState exposes Factorio implementation details that leak into the
planning and execution layers:
- `entity_id=0` for natural objects (trees, rocks have no unit number in Factorio 2.x)
- `NaturalObject.is_minable` is unreliable (False for trees even though they are minable)
- `PlayerState.reachable` exposes a raw set of integer entity IDs

These have caused repeated bugs (trees skipped because `is_minable` is False;
entities considered reachable because they're in the scan radius rather than within
mining reach; etc.). Each fix has required special-casing in predicates, skills, and
agents. The root cause is that WorldState speaks Factorio rather than speaking to the
agent's needs.

**Proposed direction:**
The bridge/parser layer should translate Factorio internals into stable, semantically
correct agent concepts before they reach WorldState:
- Natural objects should carry an explicit `is_minable: bool` that reflects KB knowledge,
  not raw `entity_id != 0`
- Reachability should be expressed as a distance or a flag that reflects actual
  game-engine mining reach, not a raw entity ID set
- The `entity_id=0` pattern should be invisible above the bridge; natural objects
  should have a stable identity scheme that doesn't require callers to special-case 0

**What is undecided:**
- The exact representation (richer NaturalObject fields? A separate natural object
  identity scheme? A post-parse enrichment step using the KB?)
- Whether this belongs in the bridge parser, the WorldWriter, or a new enrichment
  layer between them
- How to handle the bootstrap problem (KB needs to be populated before enrichment
  can be correct)

**Where the decision lives:** `bridge/state_parser.py`, `world/observable/state.py`,
`world/observable/writer.py`.

**When relevant:** Phase 10, in conjunction with the WorldState refactor already
planned for that phase.

---

## Decision log

| ID | Decision | Date | Notes |
|---|---|---|---|
| — | Unified planning stack | Session 4 | GoalFrame eliminated; Goal carries step/context; _stack replaces _goal_stack + _active_task |
| — | navigate as TASK_GOAL_TYPE | Session 4 | task-backed goal type; no coordinator handler needed |
| — | Predicates own world-state logic | Session 4 | is_present, is_reachable, can_destroy in predicates.py; skills delegate to them |