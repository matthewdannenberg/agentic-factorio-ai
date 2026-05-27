"""
execution/

The execution layer — everything that runs goals and produces actions.

No LLM calls. Reads WorldQuery and KnowledgeBase. Writes SelfModelPatch.
The coordinator additionally reads SelfModel directly.

Subpackages
-----------
coordinator/    RuleBasedCoordinator + AgentRegistry. Translates Goals into
                Tasks and routes each Task to the appropriate agent.
agents/         AgentProtocol (base) + concrete agent implementations.
                Each agent owns one active Task, emits Actions + SelfModelPatch.
memory/         BehavioralMemoryProtocol + SQLite implementation.

Top-level modules
-----------------
loop.py          FactorioLoop — master tick loop.
protocol.py      ExecutionLayerProtocol, ExecutionResult, ExecutionStatus,
                 StuckContext.
blackboard.py    Blackboard — shared working memory within a goal's lifetime.
preconditions.py Pure predicate functions over WorldQuery + KnowledgeBase.
state_machine.py AgentState enum and valid transitions.

Information boundary
--------------------
  - Coordinator sees SelfModel + player inventory (via WorldQuery).
  - Agents see WorldQuery (scan-radius-limited) + their active Task.
  - Agents emit SelfModelPatch objects; coordinator applies them.
  - Nothing in execution/ imports from llm/. GoalSource is the interface.
"""

from execution.protocol import (
    ExecutionLayerProtocol,
    ExecutionResult,
    ExecutionStatus,
    StuckContext,
)
from execution.blackboard import (
    Blackboard,
    BlackboardEntry,
    EntryCategory,
    EntryScope,
)
from execution.loop import FactorioLoop, LoopConfig, LoopStats
from execution.state_machine import AgentState, ExamineMode
from execution.predicates import is_at, is_reachable, can_mine, player_has_item

__all__ = [
    "ExecutionLayerProtocol",
    "ExecutionResult",
    "ExecutionStatus",
    "StuckContext",
    "Blackboard",
    "BlackboardEntry",
    "EntryCategory",
    "EntryScope",
    "FactorioLoop",
    "LoopConfig",
    "LoopStats",
    "AgentState",
    "ExamineMode",
    "is_at",
    "is_reachable",
    "can_mine",
    "player_has_item",
]