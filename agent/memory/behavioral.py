"""
agent/memory/behavioral.py

BehavioralMemory — persistent cross-run strategy and performance storage.

The behavioral memory stores what the agent has learned to do across runs:
strategy records (what worked for a goal type), spatial patterns (recurring
factory subgraphs), and performance history (per-goal-type statistics).

This module provides:
  - GoalOutcome, StrategyRecord, PerformanceStats: data types
  - BehavioralMemoryProtocol: the interface
  - SQLiteBehavioralMemory: concrete SQLite-backed implementation

The SQLite stub stores and retrieves data correctly but uses minimal schema
design. Full strategy matching (fuzzy context similarity, pattern deduplication)
comes in Phase 8 alongside the production agent that writes meaningful records.

Rules
-----
- No LLM calls. No RCON. No WorldQuery reads.
- The protocol is injected; callers depend only on BehavioralMemoryProtocol.
- The SQLite connection is opened lazily on first write/read and closed by
  calling close(). The main loop is responsible for calling close() at run end.
- record_spatial_pattern accepts a SelfModel but only stores its metadata in
  the stub — full pattern extraction is a Phase 10 concern.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.self_model import SelfModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GoalOutcome:
    """
    Summary of a single goal resolution.

    Fields
    ------
    success       : True if the goal's success condition was met.
    reward        : Final reward value from RewardEvaluator.
    ticks_elapsed : Ticks from goal activation to resolution.
    """
    success: bool
    reward: float
    ticks_elapsed: int


@dataclass
class StrategyRecord:
    """
    A record of how a particular goal type was approached and how it went.

    Fields
    ------
    goal_type       : The goal type string (matches GoalTree conventions).
    context_summary : Snapshot of relevant game state at goal start. Produced
                      by the examination layer; structure is goal-type-specific.
    outcome         : The outcome of the attempt.
    recorded_at     : Unix timestamp at recording time.
    """
    goal_type: str
    context_summary: dict
    outcome: GoalOutcome
    recorded_at: int


@dataclass
class PerformanceStats:
    """
    Aggregate performance statistics for a goal type across all recorded runs.

    Fields
    ------
    goal_type      : The goal type these stats cover.
    total_attempts : Total number of recorded attempts (success + failure).
    success_rate   : Fraction of attempts that succeeded, in [0.0, 1.0].
    mean_ticks     : Mean ticks elapsed across all attempts.
    mean_reward    : Mean reward across all attempts.
    """
    goal_type: str
    total_attempts: int
    success_rate: float
    mean_ticks: float
    mean_reward: float


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class BehavioralMemoryProtocol:
    """
    Interface for the behavioral memory system.

    record_outcome()
        Record the outcome of a completed or failed goal. Called by the
        examination layer at goal resolution time.

    query_strategies()
        Retrieve strategy records for a given goal type. The context_summary
        is used for similarity matching; in the stub, all records for the
        goal type are returned regardless of context.

    record_spatial_pattern()
        Store a recurring factory subgraph that proved effective. The label
        is a human-readable name for the pattern. Full pattern logic is Phase 10.

    get_performance_history()
        Return aggregate performance statistics for a goal type.

    close()
        Flush and close the backing store. Called by the main loop at run end.
    """

    def record_outcome(
        self,
        goal_type: str,
        context_summary: dict,
        outcome: GoalOutcome,
        ticks_elapsed: int,
    ) -> None:
        raise NotImplementedError

    def query_strategies(
        self,
        goal_type: str,
        context_summary: dict,
    ) -> list[StrategyRecord]:
        raise NotImplementedError

    def record_spatial_pattern(
        self,
        subgraph: "SelfModel",
        label: str,
    ) -> None:
        raise NotImplementedError

    def get_performance_history(
        self,
        goal_type: str,
    ) -> PerformanceStats:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SQLite-backed implementation
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_type     TEXT    NOT NULL,
    context_json  TEXT    NOT NULL,
    success       INTEGER NOT NULL,   -- 0 or 1
    reward        REAL    NOT NULL,
    ticks_elapsed INTEGER NOT NULL,
    recorded_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS spatial_patterns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT    NOT NULL,
    node_count  INTEGER NOT NULL,
    edge_count  INTEGER NOT NULL,
    recorded_at INTEGER NOT NULL
);
"""


class SQLiteBehavioralMemory(BehavioralMemoryProtocol):
    """
    SQLite-backed behavioral memory.

    Phase 5 stub: stores and retrieves records correctly but performs no
    similarity matching, pattern deduplication, or strategy ranking. Full
    behavioral intelligence comes in Phase 8.

    Parameters
    ----------
    db_path : Path to the SQLite database file. Use ":memory:" for tests.
    """

    def __init__(self, db_path: str = "data/behavioral.db") -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # BehavioralMemoryProtocol implementation
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        goal_type: str,
        context_summary: dict,
        outcome: GoalOutcome,
        ticks_elapsed: int,
    ) -> None:
        """
        Persist a goal outcome record.

        context_summary is JSON-serialised as-is. Non-serialisable values
        are coerced to strings to avoid data loss.
        """
        conn = self._connect()
        try:
            context_json = json.dumps(context_summary, default=str)
        except Exception as exc:
            log.warning("Failed to serialise context_summary: %s", exc)
            context_json = "{}"
        conn.execute(
            """
            INSERT INTO strategy_records
                (goal_type, context_json, success, reward, ticks_elapsed, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                goal_type,
                context_json,
                1 if outcome.success else 0,
                outcome.reward,
                ticks_elapsed,
                int(time.time()),
            ),
        )
        conn.commit()

    def query_strategies(
        self,
        goal_type: str,
        context_summary: dict,
    ) -> list[StrategyRecord]:
        """
        Return all strategy records for the given goal type.

        Stub: ignores context_summary for similarity filtering. Returns all
        records for the goal type in recorded_at descending order (most recent
        first). Phase 8 will add context similarity ranking.
        """
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT goal_type, context_json, success, reward, ticks_elapsed, recorded_at
            FROM strategy_records
            WHERE goal_type = ?
            ORDER BY recorded_at DESC
            """,
            (goal_type,),
        ).fetchall()

        records = []
        for row in rows:
            try:
                ctx = json.loads(row["context_json"])
            except Exception:
                ctx = {}
            records.append(
                StrategyRecord(
                    goal_type=row["goal_type"],
                    context_summary=ctx,
                    outcome=GoalOutcome(
                        success=bool(row["success"]),
                        reward=row["reward"],
                        ticks_elapsed=row["ticks_elapsed"],
                    ),
                    recorded_at=row["recorded_at"],
                )
            )
        return records

    def record_spatial_pattern(
        self,
        subgraph: "SelfModel",
        label: str,
    ) -> None:
        """
        Store metadata about a recurring factory subgraph pattern.

        Stub: records only node count, edge count, and label. Full pattern
        extraction (serialising the graph structure for replay) is Phase 10.
        """
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO spatial_patterns (label, node_count, edge_count, recorded_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                label,
                len(subgraph.all_nodes()),
                len(subgraph.all_edges()),
                int(time.time()),
            ),
        )
        conn.commit()

    def get_performance_history(
        self,
        goal_type: str,
    ) -> PerformanceStats:
        """
        Return aggregate performance statistics for a goal type.

        Returns a PerformanceStats with zero attempts and safe defaults if no
        records exist for the goal type.
        """
        conn = self._connect()
        row = conn.execute(
            """
            SELECT
                COUNT(*)                AS total_attempts,
                AVG(CAST(success AS REAL)) AS success_rate,
                AVG(ticks_elapsed)      AS mean_ticks,
                AVG(reward)             AS mean_reward
            FROM strategy_records
            WHERE goal_type = ?
            """,
            (goal_type,),
        ).fetchone()

        if row is None or row["total_attempts"] == 0:
            return PerformanceStats(
                goal_type=goal_type,
                total_attempts=0,
                success_rate=0.0,
                mean_ticks=0.0,
                mean_reward=0.0,
            )

        return PerformanceStats(
            goal_type=goal_type,
            total_attempts=row["total_attempts"],
            success_rate=row["success_rate"] or 0.0,
            mean_ticks=row["mean_ticks"] or 0.0,
            mean_reward=row["mean_reward"] or 0.0,
        )
