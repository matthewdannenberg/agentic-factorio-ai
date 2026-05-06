"""
agent/examiner/audit_report.py

AuditReport — the shared output type of both examiner modes.

Produced by:
  agent/examiner/mechanical_auditor.py  (LLM unavailable)
  agent/examiner/rich_examiner.py       (LLM available)

Consumed by:
  llm/client.py — passed as context on resumption after rate limiting
  agent/strategic.py — informs next goal selection
  memory/episodic.py — stored per-run

Rules:
- Pure data. No LLM calls. No RCON.
- RICH and MECHANICAL modes share the same dataclass; RICH-only fields are
  Optional and default to None so MECHANICAL reports are always valid.
- Designed to serialise cleanly to JSON (all fields are primitives or
  dataclasses with the same property).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AuditMode(Enum):
    RICH       = "rich"        # LLM-assisted — full reflection
    MECHANICAL = "mechanical"  # Pure-code — LLM unavailable


class AnomalySeverity(Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Sub-types
# ---------------------------------------------------------------------------

@dataclass
class StarvedEntity:
    """An entity that cannot work because one or more inputs are absent."""
    entity_id: int
    name: str
    position_x: float
    position_y: float
    missing_inputs: list[str]   # Item names that are absent


@dataclass
class IdleEntity:
    """An entity that is not working, not starved, and not low on power."""
    entity_id: int
    name: str
    position_x: float
    position_y: float
    reason: str   # Best-effort human-readable reason (e.g. "no recipe set")


@dataclass
class CongestionSegment:
    """A belt or pipe segment where flow has stalled."""
    segment_id: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    item: Optional[str] = None   # Item causing congestion, if identifiable


@dataclass
class Anomaly:
    """
    Anything unexpected observed during the audit period.
    Both examiner modes can emit these; the rich examiner may emit richer ones.
    """
    severity: AnomalySeverity
    description: str
    # Optional structured payload — examiner fills what it can
    entity_id: Optional[int] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None


@dataclass(frozen=True)
class BoundingBox:
    """
    Axis-aligned rectangular region in Factorio tile coordinates.

    Matches the coordinate system used by Factorio's blueprint capture API
    (LuaSurface.create_script_inventory / get_blueprint_entities area param),
    so this value can be passed directly to the bridge when extracting a
    blueprint string.

    All coordinates are tile integers (Factorio's map uses integer tile positions
    even though entity positions are floats). min < max on both axes.
    """
    min_x: int
    min_y: int
    max_x: int
    max_y: int

    @property
    def width(self) -> int:
        return self.max_x - self.min_x

    @property
    def height(self) -> int:
        return self.max_y - self.min_y

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) / 2.0

    @property
    def center_y(self) -> float:
        return (self.min_y + self.max_y) / 2.0

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def __repr__(self) -> str:
        return f"BoundingBox(({self.min_x},{self.min_y})→({self.max_x},{self.max_y}) {self.width}×{self.height})"


@dataclass
class BlueprintCandidate:
    """
    (Rich mode only) A sub-factory region nominated for extraction as a
    reusable blueprint.

    This is a NOMINATION only — it records why a region looks worth capturing.
    Evaluation of a stored blueprint (throughput per tile, tech tier, estimated
    improvability, etc.) is handled by BlueprintRecord in
    memory/blueprint_library/ once the blueprint has actually been extracted
    and run in production. Do not add evaluation fields here.

    The curator (memory/blueprint_library/blueprint_curator.py) reads these
    and decides whether to act on them (extract, deduplicate, store).

    Fields
    ------
    region_description  : Human-readable description, e.g. "4-lane iron smelting column".
    bounds              : Tile-aligned bounding box. Passed directly to the
                          bridge's blueprint-capture call.
    performance_metric  : Observed throughput summary, e.g. "62 iron-plate/min
                          sustained for 300 ticks". Stored with the nomination
                          so the curator can filter low-quality candidates.
    rationale           : Why the rich examiner flagged this region (e.g.
                          "sustained output 20% above prior best for this item").
    """
    region_description: str
    bounds: BoundingBox
    performance_metric: str
    rationale: str



@dataclass
class DamagedEntityRecord:
    """
    An entity observed below full health during the audit period.

    Mirrors world.state.DamagedEntity but is a snapshot owned by the report —
    the examiner copies relevant fields rather than holding a live reference
    into the state tree. This keeps the report self-contained for serialisation
    and LLM context injection.

    Cause-agnostic: may result from biter attack, vehicle collision, player
    error, or anything else. Always populated when damage exists, regardless
    of BITERS_ENABLED.
    """
    entity_id: int
    name: str
    position_x: float
    position_y: float
    health_fraction: float   # (0.0, 1.0)


@dataclass
class DestroyedEntityRecord:
    """
    An entity destroyed during (or just before) the audit period.

    Pulled from WorldState.destroyed_entities within the observation window.
    Cause-agnostic: biter attack, vehicle collision, player deconstruction, etc.
    Always populated when destruction events exist, regardless of BITERS_ENABLED.
    """
    name: str
    position_x: float
    position_y: float
    destroyed_at: int   # Game tick
    cause: str          # "biter" | "vehicle" | "deconstruct" | "unknown"


# ---------------------------------------------------------------------------
# AuditReport
# ---------------------------------------------------------------------------

@dataclass
class AuditReport:
    """
    The unified output of both examiner modes.

    MECHANICAL fields are always populated.
    RICH-only fields (blueprint_candidates, llm_observations) are None when
    mode == AuditMode.MECHANICAL.

    Fields
    ------
    tick                : Game tick at the start of the audit period.
    mode                : Which examiner produced this report.
    observation_ticks   : How many ticks the audit covered.
    starved_entities    : Machines with missing inputs.
    idle_entities       : Machines that are not working for other reasons.
    power_headroom_kw   : Produced - consumed, in kW. Negative = brownout.
    power_satisfaction  : 0.0–1.0 fraction of demand being met.
    belt_congestion     : Belt segments where flow has stalled.
    production_rates    : item_name → items/minute over the observation window.
    anomalies           : Anything unusual, ordered by severity descending.
                          The examiner automatically promotes damaged/destroyed
                          entities into CRITICAL anomalies — callers should not
                          need to inspect damaged_entities separately for
                          alerting purposes, but the structured lists are there
                          for recovery planning.
    damaged_entities    : Entities observed below full health this period.
                          Populated whenever damage exists — cause-agnostic,
                          not gated on BITERS_ENABLED.
    destroyed_entities  : Entities destroyed during (or just before) this period.
                          Cause-agnostic, not gated on BITERS_ENABLED.
    blueprint_candidates: (rich only) Sub-factory regions nominated for extraction.
                          See BlueprintCandidate for the nomination/evaluation split.
    llm_observations    : (rich only) The LLM's own free-text summary.
    accumulated         : True when this report merges multiple prior reports
                          (used when handing a backlog to the LLM after rate-
                          limit recovery).
    """

    tick: int
    mode: AuditMode
    observation_ticks: int = 0

    # Health indicators — always populated
    starved_entities: list[StarvedEntity] = field(default_factory=list)
    idle_entities: list[IdleEntity] = field(default_factory=list)
    power_headroom_kw: float = 0.0
    power_satisfaction: float = 1.0
    belt_congestion: list[CongestionSegment] = field(default_factory=list)

    # Throughput — always populated (may be empty early game)
    production_rates: dict[str, float] = field(default_factory=dict)   # item → /min

    # Anomalies — always populated
    anomalies: list[Anomaly] = field(default_factory=list)

    # Structural damage — always present, empty when BITERS_ENABLED=False
    damaged_entities: list[DamagedEntityRecord] = field(default_factory=list)
    destroyed_entities: list[DestroyedEntityRecord] = field(default_factory=list)

    # Rich-only fields
    blueprint_candidates: Optional[list[BlueprintCandidate]] = None
    llm_observations: Optional[str] = None

    # Meta
    accumulated: bool = False   # True if this merges several audit periods

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def has_critical_anomalies(self) -> bool:
        return any(a.severity == AnomalySeverity.CRITICAL for a in self.anomalies)

    @property
    def is_brownout(self) -> bool:
        return self.power_satisfaction < 1.0

    @property
    def total_starved(self) -> int:
        return len(self.starved_entities)

    @property
    def total_idle(self) -> int:
        return len(self.idle_entities)

    @property
    def has_structural_damage(self) -> bool:
        """True if any entities are damaged or were destroyed this period."""
        return bool(self.damaged_entities or self.destroyed_entities)

    def production_rate(self, item: str) -> float:
        """Return items/minute for a specific item, 0.0 if not tracked."""
        return self.production_rates.get(item, 0.0)

    def merge(self, other: "AuditReport") -> "AuditReport":
        """
        Merge another AuditReport into this one, returning a new combined report.
        Used when accumulating MECHANICAL reports during a rate-limit blackout.

        Strategy:
        - Tick: keep the earlier (self) tick.
        - observation_ticks: sum.
        - Entity issues: union by entity_id.
        - Power: use the worse (lower satisfaction) values.
        - Belt congestion: union by segment_id.
        - Production rates: average across reports (equal weight).
        - Anomalies: concatenate, severity-sort.
        - Rich fields: use other's if self has none.
        - accumulated: True always.
        """
        # Entity union
        seen_starved = {e.entity_id for e in self.starved_entities}
        merged_starved = list(self.starved_entities) + [
            e for e in other.starved_entities if e.entity_id not in seen_starved
        ]

        seen_idle = {e.entity_id for e in self.idle_entities}
        merged_idle = list(self.idle_entities) + [
            e for e in other.idle_entities if e.entity_id not in seen_idle
        ]

        # Belt union
        seen_segs = {s.segment_id for s in self.belt_congestion}
        merged_belts = list(self.belt_congestion) + [
            s for s in other.belt_congestion if s.segment_id not in seen_segs
        ]

        # Production rates — simple average
        all_items = set(self.production_rates) | set(other.production_rates)
        merged_rates = {
            item: (
                (self.production_rates.get(item, 0.0) + other.production_rates.get(item, 0.0))
                / 2.0
            )
            for item in all_items
        }

        # Power — use the worse reading
        worse_sat = min(self.power_satisfaction, other.power_satisfaction)
        worse_head = min(self.power_headroom_kw, other.power_headroom_kw)

        # Anomalies — concatenate and sort critical first
        merged_anomalies = sorted(
            self.anomalies + other.anomalies,
            key=lambda a: a.severity.value,
            reverse=True,
        )

        # Damaged entities — union by entity_id
        seen_damaged = {e.entity_id for e in self.damaged_entities}
        merged_damaged = list(self.damaged_entities) + [
            e for e in other.damaged_entities if e.entity_id not in seen_damaged
        ]

        # Destroyed entities — union by (name, position_x, position_y, destroyed_at)
        seen_destroyed = {
            (e.name, e.position_x, e.position_y, e.destroyed_at)
            for e in self.destroyed_entities
        }
        merged_destroyed = list(self.destroyed_entities) + [
            e for e in other.destroyed_entities
            if (e.name, e.position_x, e.position_y, e.destroyed_at) not in seen_destroyed
        ]

        # Rich fields
        bp_candidates = (
            (self.blueprint_candidates or []) + (other.blueprint_candidates or [])
        ) or None
        llm_obs = "\n---\n".join(
            filter(None, [self.llm_observations, other.llm_observations])
        ) or None

        return AuditReport(
            tick=self.tick,
            mode=AuditMode.MECHANICAL if other.mode == AuditMode.MECHANICAL else self.mode,
            observation_ticks=self.observation_ticks + other.observation_ticks,
            starved_entities=merged_starved,
            idle_entities=merged_idle,
            power_headroom_kw=worse_head,
            power_satisfaction=worse_sat,
            belt_congestion=merged_belts,
            production_rates=merged_rates,
            anomalies=merged_anomalies,
            damaged_entities=merged_damaged,
            destroyed_entities=merged_destroyed,
            blueprint_candidates=bp_candidates,
            llm_observations=llm_obs,
            accumulated=True,
        )

    def summary(self) -> str:
        """Single-line human-readable summary for logging."""
        parts = [
            f"tick={self.tick}",
            f"mode={self.mode.name}",
            f"starved={self.total_starved}",
            f"idle={self.total_idle}",
            f"power={self.power_satisfaction:.0%}",
            f"congested_belts={len(self.belt_congestion)}",
            f"anomalies={len(self.anomalies)}",
            f"damaged={len(self.damaged_entities)}",
            f"destroyed={len(self.destroyed_entities)}",
        ]
        if self.accumulated:
            parts.append("(accumulated)")
        return "AuditReport(" + " ".join(parts) + ")"

    def __repr__(self) -> str:
        return self.summary()