"""
world/model/

The factory self-model — the coordinator's persistent, globally-accurate
graph of what has been built in the current run.

Contents
--------
self_model.py   SelfModel, SelfModelProtocol, node/edge types, BoundingBox.
patch.py        SelfModelPatch — structured update objects emitted by agents
                and applied by the coordinator.

Access rules
------------
- SelfModel is read and written by the coordinator only.
- Agents emit SelfModelPatch objects; the coordinator applies them.
- The examination layer may promote CANDIDATE nodes to ACTIVE directly.
- Nothing in world/model/ imports from execution/, planning/, or llm/.
"""
