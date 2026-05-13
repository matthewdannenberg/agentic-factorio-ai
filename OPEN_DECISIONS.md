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
- This normalisation is what makes structurally similar goals (different recipes,
  same task type) produce structurally similar observations
- The KnowledgeBase production chain queries make goal-relative construction
  tractable

**What is undecided:**
- Whether observation spaces are fixed per goal type or evolve over time
- Whether a single unified observation space or per-agent spaces
- Whether the representation should be high-dimensional and fixed, or lower-
  dimensional and adaptive
- How to handle goals that don't map cleanly to a single goal type

**Where the decision lives:** `agent/observation.py` and individual agent
implementations. Nothing outside these files assumes anything about observation
shape.

**Research needed:** representation learning approaches for goal-conditioned RL;
universal value function approximators (UVFAs); whether adaptive representations
are worth the added complexity given the training stability cost.

---

## OD-2 — Coordinator Learning

**Question:** Does the coordinator transition from rule-based to learned, and if
so, how and when?

**What we know:**
- The coordinator starts rule-based — routes goals to agents by goal type, derives
  simple subtask trees from KB and self-model
- A learned coordinator could handle more complex goal-agent routing, conflict
  resolution, and subtask sequencing
- The coordinator satisfies `CoordinatorProtocol` regardless of implementation

**What is undecided:**
- Whether coordinator learning is necessary or whether a sophisticated rule-based
  coordinator is sufficient
- If learned: which MARL coordination approach (centralised training with
  decentralised execution, emergent communication, explicit communication channels)
- When to transition — after individual agents are stable, or jointly from the start

**Where the decision lives:** `agent/network/coordinator.py`. The protocol boundary
means this can be swapped without touching agents or the execution protocol.

**Research needed:** MARL coordination literature; whether the coordination problem
here is hard enough to warrant learned coordination or whether rule-based routing
with learned agents is sufficient.

---

## OD-3 — Self-Model Cross-Run Persistence

**Question:** What self-model information survives a run and in what form?

**What we know:**
- The self-model graph represents this factory — it starts empty each run
- Something useful should cross the run boundary into behavioral memory
- Candidates: full graph (probably too run-specific), subgraph summaries,
  spatial patterns that proved effective, throughput benchmarks

**What is undecided:**
- Exactly what is extracted at end-of-run
- How spatial patterns are represented in behavioral memory
- Whether pattern similarity matching is needed to avoid duplicates across runs
- How this feeds the eventual blueprint system

**Where the decision lives:** `agent/self_model.py` and `agent/memory/behavioral.py`.
The interface between them (what gets passed at end-of-run) is the key boundary.

**Research needed:** graph summarisation approaches; whether learned spatial patterns
need explicit representation or emerge implicitly from policy structure.

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

**Where the decision lives:** individual agent implementations. Each satisfies
`AgentProtocol` regardless of algorithm. Training harness is internal to each agent.

**Research needed:** RL approaches for real-time strategy games; handling of
partial observability; whether the goal-conditioned observation space design
(OD-1) constrains algorithm choice.

---

## OD-5 — Spatial-Logistics Internal Structure

**Question:** Is the spatial-logistics concern handled by one agent or two
coordinating agents?

**What we know:**
- Spatial reasoning (layout, placement, region designation) and logistics reasoning
  (belt routing, inserter placement, throughput) are deeply interdependent
- Separating them creates coordination overhead that may exceed the benefit
- Both will be doing significant work at essentially all times during factory building

**What is undecided:**
- Whether a single agent with richer internal structure handles both concerns
- Or two agents with heavy blackboard coordination
- Whether the MARL literature suggests a natural factoring for spatially-entangled
  concerns of this kind

**Where the decision lives:** `agent/network/agents/spatial_logistics.py` (or
split into two files if two agents). The coordinator routes to this subsystem;
the internal structure is invisible above that boundary.

**Research needed:** MARL approaches for spatially-coupled tasks; whether emergent
coordination between two agents can learn the entanglement or whether it needs to
be explicitly modelled in a single agent.

---

## OD-6 — RewardEvaluator Namespace Extension for Self-Model

**Question:** Should `RewardEvaluator` be extended to evaluate conditions against
the self-model in addition to `WorldQuery`, and what does that namespace look like?

**What we know:**
- The current evaluator evaluates condition strings against a namespace derived
  from `WorldQuery`. This works well for directly observable conditions: inventory
  counts, research state, exploration progress, time, damage events.
- Many goals the LLM will set are structural and persistent — "establish iron plate
  production at 30 plates/second", "connect ore site to smelting line" — and are
  more naturally and reliably evaluated against the self-model than against
  scan-radius-limited WorldQuery.
- The self-model is non-proximal by nature: it is persistent, globally accurate,
  and not subject to staleness in the way entity scans are.
- The proximal/non-proximal distinction in `CONDITION_SCOPE.md` should eventually
  gain a third category — STRUCTURAL — for self-model-backed conditions.

**What is undecided:**
- Exactly which self-model queries belong in the evaluator namespace
- Whether STRUCTURAL conditions should be guarded differently from NON-PROXIMAL
  ones (the self-model can lag behind game reality if the examination layer hasn't
  run recently — this is a different kind of staleness from scan radius limits)
- Whether the self-model namespace entries should be added to the existing evaluator
  or whether a separate structural evaluator is cleaner
- How to express self-model staleness in condition strings — the existing
  `staleness(section)` mechanism is WorldQuery-scoped and doesn't apply

**Proposed namespace entries (tentative):**
```python
production_line(item)       # -> SelfModelNode | None — active producer of item
production_capacity(item)   # -> float — total throughput in units per minute
has_infrastructure(type)    # -> bool — any active node of given type exists
connected(node_a, node_b)   # -> bool — path exists between nodes in self-model
sm_staleness()              # -> int — ticks since examination layer last reconciled
```

**Where the decision lives:** `planning/reward_evaluator.py` (namespace extension)
and `CONDITION_SCOPE.md` (new STRUCTURAL scope category). The self-model is passed
into the evaluator alongside `WorldQuery`; callers that don't have a self-model
pass `None` and structural conditions evaluate to `False` rather than raising.

**When this becomes relevant:** Phase 10, when the examination layer gains
self-model reconciliation responsibilities and the self-model is reliably populated.
Until then, structural conditions are not useful even if the namespace exists.

**Files to update when this is implemented:**
- `planning/reward_evaluator.py` — `_build_namespace()` and docstrings
- `CONDITION_SCOPE.md` — new STRUCTURAL scope category and summary table
- `REWARD_NAMESPACE.md` — new namespace entries
- `ARCHITECTURE.md` — RewardEvaluator description
- `tests/integration/test_evaluator_capabilities.py` — new SM category

---

## OD-7 — NodeType-Specific SelfModelNode Attributes

**Question:** Should different NodeType values have different attributes, and if
so, should that be modelled through subclasses of SelfModelNode?

**What we know:**
- The current `SelfModelNode` carries a `throughput: dict[str, float]` field
  that is the primary structured attribute beyond geometry and status. This is
  sufficient for production-related queries (`find_producers`, `find_capacity`)
  and for the examination layer through Phase 9.
- Different node types plausibly need type-specific data. A `POWER_GRID` node
  would benefit from `produced_kw` / `consumed_kw` fields. A `TRAIN_STATION`
  might carry `scheduled_resources`. A `DEFENDED_REGION` might carry
  `turret_coverage_fraction`. None of these fit cleanly into `throughput`.
- Subclassing `SelfModelNode` per NodeType is the natural Python pattern for
  this, and is localized to `agent/self_model.py` and the examination layer.

**What is undecided:**
- Whether `throughput` plus ad-hoc extra fields in the examination layer is
  sufficient through Phase 10, or whether type-specific subclasses are needed
  before that.
- If subclasses: whether the graph queries (`query_nodes`, `find_producers`,
  etc.) need to be updated to return typed subclasses, or whether a generic
  `extra: dict` field is a simpler interim.

**Where the decision lives:** `agent/self_model.py`. The graph interface
(`SelfModelProtocol`) does not need to change — only the node dataclass and
the examination layer code that writes to it.

**When this becomes relevant:** Phase 10, when the examination layer begins
writing type-specific verification data to nodes (throughput metrics for
production lines, wattage for power grids, etc.).

**Proposed approach when implemented:**
- Introduce typed subclasses (`ProductionLineNode`, `PowerGridNode`, etc.)
  each adding their own fields
- `SelfModel.add_node()` accepts any `SelfModelNode` subclass
- `query_nodes(type=NodeType.POWER_GRID)` returns `list[PowerGridNode]`
  (caller casts); or add a typed variant `query_nodes_typed(type, cls)`
- Keep `throughput` on the base class; it remains the common currency for
  cross-type capacity queries

---

## Decision log

| ID | Decision | Date | Notes |
|---|---|---|---|
| — | — | — | No decisions recorded yet |