"""
planning/evaluation/condition_parser.py

Extracts coordinator handler params from a goal_type + success_condition pair.

Purpose
-------
The loop's main responsibility is running the execution network. It should not
contain goal-type-specific parsing logic. This module centralises param
extraction so that:

  - Adding a new goal type touches only this file and the coordinator.
  - The loop stays generic: it calls params_from_condition() and passes the
    result to coordinator.reset() without knowing anything about goal types.
  - The regex patterns live next to the condition strings they serve (in the
    planning layer), not scattered in execution infrastructure.

Design
------
Coordinator handlers read from frame.params (a plain dict). GoalQueueEntry
has no params field — the coordinator's params are implicit in the
success_condition string, which is the authoritative source of truth for what
the goal is trying to achieve. This module makes that implicit structure
explicit.

Extraction is best-effort: if a condition string doesn't match any known
pattern for a goal type, the function returns {} and the coordinator's own
safe-fail path handles the missing params (logging a warning and returning
STUCK).

Condition patterns by goal type
---------------------------------
collection / acquire / crafting
    inventory('ITEM') >= COUNT
    new.inventory('ITEM') >= COUNT
    → {"item": ITEM, "count": COUNT}

exploration — chunk-count form
    charted_chunks >= N
    new.charted_chunks >= N
    → {"target_chunks": N}

exploration — resource-discovery form
    len(resources_of_type('TYPE')) >= COUNT
    new.resource_patches('TYPE') >= COUNT
    → {"target_chunks": 0, "success_condition": original_condition}
    (target_chunks=0 so the coordinator doesn't short-circuit on chunks;
    the custom success_condition is passed through to the task)

research
    tech_unlocked('TECH')
    → {"tech": TECH}

production
    production_rate('ITEM') >= RATE
    → {"item": ITEM, "rate_per_min": RATE}

byproduct
    Cannot be meaningfully extracted from condition strings alone — the
    coordinator handler reads item and source_node_id which are not
    expressed in standard condition forms. Returns {}.

All others
    Returns {} so the coordinator's per-handler safe-fail fires.

Extension
---------
When a new goal type is added to the coordinator:
  1. Add its condition pattern(s) to _PATTERNS below.
  2. Add a test case to tests/unit/planning/test_condition_parser.py.
  3. Verify the coordinator handler's params.get() calls match the keys.
"""

from __future__ import annotations

import re
import logging
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern table
# ---------------------------------------------------------------------------
# Each entry: (goal_type_or_set, compiled_regex, builder_fn)
# builder_fn(match) -> dict of coordinator params.
#
# Patterns are tried in order; the first match wins.

def _item_count(m: re.Match) -> dict:
    return {"item": m.group("item"), "count": int(m.group("count"))}

def _target_chunks(m: re.Match) -> dict:
    return {"target_chunks": int(m.group("n"))}

def _resource_discovery(m: re.Match) -> dict:
    # Pass the full condition through as a custom exploration condition.
    # target_chunks=0 prevents the coordinator from short-circuiting on
    # a chunk count that hasn't been met.
    return {
        "target_chunks": 0,
        "success_condition": m.string,   # original full condition string
        "description": f"Find {m.group('count')} {m.group('resource')} patch(es)",
    }

def _tech(m: re.Match) -> dict:
    return {"tech": m.group("tech")}

def _production_rate(m: re.Match) -> dict:
    return {"item": m.group("item"), "rate_per_min": float(m.group("rate"))}


_ITEM_COUNT_TYPES = frozenset({"collection", "acquire", "crafting"})

_PATTERNS: list[tuple[frozenset[str] | str, re.Pattern, Any]] = [

    # --- collection / acquire / crafting ---
    # inventory('item') >= N  or  new.inventory('item') >= N
    (
        _ITEM_COUNT_TYPES,
        re.compile(
            r"(?:new\.)?inventory\(['\"](?P<item>[^'\"]+)['\"]\)"
            r"\s*>=\s*(?P<count>\d+)"
        ),
        _item_count,
    ),

    # --- exploration: chunk-count form ---
    # (new.)charted_chunks >= N
    (
        frozenset({"exploration"}),
        re.compile(r"(?:new\.)?charted_chunks\s*>=\s*(?P<n>\d+)"),
        _target_chunks,
    ),

    # --- exploration: resource-discovery form ---
    # len(resources_of_type('type')) >= N
    (
        frozenset({"exploration"}),
        re.compile(
            r"len\(resources_of_type\(['\"](?P<resource>[^'\"]+)['\"]\)\)"
            r"\s*>=\s*(?P<count>\d+)"
        ),
        _resource_discovery,
    ),
    # new.resource_patches('type') >= N
    (
        frozenset({"exploration"}),
        re.compile(
            r"new\.resource_patches\(['\"](?P<resource>[^'\"]+)['\"]\)"
            r"\s*>=\s*(?P<count>\d+)"
        ),
        _resource_discovery,
    ),

    # --- research ---
    # tech_unlocked('tech')
    (
        frozenset({"research"}),
        re.compile(r"tech_unlocked\(['\"](?P<tech>[^'\"]+)['\"]\)"),
        _tech,
    ),

    # --- production ---
    # production_rate('item') >= N.N
    (
        frozenset({"production"}),
        re.compile(
            r"production_rate\(['\"](?P<item>[^'\"]+)['\"]\)"
            r"\s*>=\s*(?P<rate>[\d.]+)"
        ),
        _production_rate,
    ),

    # --- clear_region / prep_region ---
    # bbox(X_MIN, Y_MIN, X_MAX, Y_MAX)
    # Also matches bbox(...).is_clear and bbox(...).natural_count == 0 —
    # the suffix is ignored by the parser (handled at eval() by BBoxQuery).
    (
        frozenset({"clear_region", "prep_region"}),
        re.compile(
            r"bbox[(]\s*(?P<x_min>-?[\d.]+)\s*,"
            r"\s*(?P<y_min>-?[\d.]+)\s*,"
            r"\s*(?P<x_max>-?[\d.]+)\s*,"
            r"\s*(?P<y_max>-?[\d.]+)\s*[)]"
        ),
        lambda m: {
            "bbox": {
                "x_min": float(m.group("x_min")),
                "y_min": float(m.group("y_min")),
                "x_max": float(m.group("x_max")),
                "y_max": float(m.group("y_max")),
            }
        },
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def params_from_condition(goal_type: str, success_condition: str) -> dict:
    """
    Extract coordinator handler params from a goal_type + success_condition.

    Parameters
    ----------
    goal_type           : Goal type string (e.g. "collection", "exploration").
    success_condition   : The RewardEvaluator condition string for the goal.

    Returns
    -------
    dict    : Coordinator params extracted from the condition.
              Empty dict if no pattern matched — the coordinator's safe-fail
              path handles this (logs a warning, returns STUCK).

    Examples
    --------
    >>> params_from_condition("collection", "new.inventory('iron-ore') >= 5")
    {'item': 'iron-ore', 'count': 5}

    >>> params_from_condition("exploration", "new.charted_chunks >= 10")
    {'target_chunks': 10}

    >>> params_from_condition("exploration",
    ...     "len(resources_of_type('copper-ore')) >= 1")
    {'target_chunks': 0, 'success_condition': "len(resources_of_type('copper-ore')) >= 1",
     'description': 'Find 1 copper-ore patch(es)'}

    >>> params_from_condition("research", "tech_unlocked('automation')")
    {'tech': 'automation'}
    """
    sc = success_condition or ""
    for type_set, pattern, builder in _PATTERNS:
        if goal_type not in type_set:
            continue
        m = pattern.search(sc)
        if m:
            result = builder(m)
            log.debug(
                "condition_parser: %r matched pattern %r → %s",
                goal_type, pattern.pattern, result,
            )
            return result

    if sc:
        log.debug(
            "condition_parser: no pattern matched for goal_type=%r "
            "condition=%r — coordinator will use safe-fail",
            goal_type, sc,
        )
    return {}