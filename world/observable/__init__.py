"""
world/observable/

Scan-radius-limited game state: WorldState, WorldQuery, WorldWriter.

Updated every tick by the bridge via StateParser + WorldWriter.integrate_snapshot().
Different sections have different staleness — consumers should check
WorldQuery.section_staleness() before relying on proximal data.

Contents
--------
state.py              WorldState pure data container + all observable dataclasses.
query.py              WorldQuery — sole read interface.
writer.py             WorldWriter — sole write interface.
production_tracker.py ProductionTracker — rolling throughput measurement.

Access rules
------------
- WorldState is NOT exported from world/__init__.py. Use WorldQuery / WorldWriter.
- bridge/state_parser.py constructs WorldState snapshots directly — that is its
  legitimate role. Nothing else constructs WorldState.
- bridge/state_parser.py and bridge/actions.py import from world.observable.state
  directly — the one permitted upward import from bridge into world.
"""