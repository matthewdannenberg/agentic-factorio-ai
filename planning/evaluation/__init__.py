"""
planning/evaluation/

Mechanical evaluation of goal conditions and reward shaping.

Contents
--------
condition_namespace.py   build_core_namespace, _DeltaView, safe_builtins.
                         Shared between RewardEvaluator and the coordinator's
                         subtask condition evaluator. Adding a new world-state
                         concept here gives both contexts access automatically.
condition_parser.py      params_from_condition - Extract coordinator handler 
                         params from a goal_type + success_condition.
reward_evaluator.py      RewardEvaluator — evaluates success/failure/milestone
                         condition strings against a WorldQuery snapshot.
                         Pure computation; no LLM calls.
resource_allocator.py    ResourceAllocator — priority-weighted allocation of
                         shared agent resources (action slots, LLM call budget).
                         Currently a pass-through; designed for future contention
                         when BITERS_ENABLED=True.
"""
