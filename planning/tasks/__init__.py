"""
planning/tasks/

Task lifecycle: the coordinator's derived work units handed to agents.

A Task (previously called Subtask) is the coordinator's translation of a Goal
into concrete executable work. Agents interact only with Tasks — they never
see the Goal object. The coordinator derives Tasks from Goals and routes each
Task to the appropriate agent via agent_hint.

Contents
--------
task.py        Task, TaskStatus, TaskRecord — the data type.
               Separated from the ledger: Task is a value object; the
               ledger is an operational structure. Mirrors the
               goal.py / goal_tree.py separation.
task_ledger.py TaskLedger — active stack and historical log.
               The coordinator pushes and pops Tasks here; the ledger
               tracks the full history for examination and debugging.
"""
