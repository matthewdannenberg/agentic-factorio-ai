"""
world/knowledge/

Static game content learned at runtime from Factorio.

Grows across runs as the agent encounters new entities, recipes, techs,
and resources. Persists to SQLite between sessions.

Contents
--------
base.py       KnowledgeBase — SQLite-backed store. Five domains, eleven tables.
tech_tree.py  TechTree — research dependency graph backed by KnowledgeBase.

Access rules
------------
- KnowledgeBase is read by the execution layer (agents + coordinator) and the
  planning layer. Never written to at tick time — only at startup/discovery.
- query_fn is injected by run.py; this package never calls RCON directly.
"""
