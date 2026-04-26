"""Scenario 08 — drift: date format (events.ts iso_string → epoch_ms).

Baseline filters events inside a specific UTC day via ISO string
comparisons. When the drift fires, the `ts` column becomes a BIGINT of
epoch-ms; the agent must rewrite comparisons against the numeric value.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb

from ._fixtures import categorical_choices, seeded_rng
from .base import BuilderResult, DriftConfig, ScenarioSpec

# Fixed anchor day — comparisons are deterministic across seeds.
_ANCHOR = datetime(2026, 4, 21, tzinfo=UTC)
_ANCHOR_NEXT = _ANCHOR + timedelta(days=1)
_ANCHOR_ISO = _ANCHOR.isoformat().replace("+00:00", "Z")
_ANCHOR_NEXT_ISO = _ANCHOR_NEXT.isoformat().replace("+00:00", "Z")
_ANCHOR_MS = int(_ANCHOR.timestamp() * 1000)
_ANCHOR_NEXT_MS = int(_ANCHOR_NEXT.timestamp() * 1000)


def _build(spec: ScenarioSpec, seed: int, scale: int) -> BuilderResult:
    rng = seeded_rng(spec.scenario_id, seed, scale)
    n_events = scale * 8

    window_start = _ANCHOR - timedelta(days=3)
    window_span_s = 7 * 86_400
    event_dts = [
        (window_start + timedelta(seconds=rng.randrange(window_span_s))) for _ in range(n_events)
    ]
    event_iso = [dt.isoformat().replace("+00:00", "Z") for dt in event_dts]
    kinds = categorical_choices(
        rng, ["login", "action", "error"], n_events, weights=[0.6, 0.35, 0.05]
    )

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE events( id BIGINT PRIMARY KEY, ts VARCHAR, kind VARCHAR);")
    conn.executemany(
        "INSERT INTO events VALUES (?, ?, ?)",
        [(i, iso, k) for i, (iso, k) in enumerate(zip(event_iso, kinds, strict=False), 1)],
    )

    baseline_sql = (
        f"SELECT kind, COUNT(*) AS n FROM events "
        f"WHERE ts >= '{_ANCHOR_ISO}' AND ts < '{_ANCHOR_NEXT_ISO}' "
        f"GROUP BY kind ORDER BY kind"
    )
    gt_sql_predrift = baseline_sql
    gt_sql_postdrift = (
        f"SELECT kind, COUNT(*) AS n FROM events "
        f"WHERE ts >= {_ANCHOR_MS} AND ts < {_ANCHOR_NEXT_MS} "
        f"GROUP BY kind ORDER BY kind"
    )

    synopsis = (
        "events(id PK, ts VARCHAR(ISO-8601 UTC), kind). Under drift, `ts` "
        f"becomes BIGINT epoch-ms. Filter window is {_ANCHOR_ISO} – "
        f"{_ANCHOR_NEXT_ISO} (i.e. epoch-ms in "
        f"[{_ANCHOR_MS}, {_ANCHOR_NEXT_MS}))."
    )
    # Date-format drift keeps the ``ts`` identifier; what changes is the
    # literal shape (ISO string → epoch-ms integer). The rubric
    # therefore can't distinguish "adapted" from "not adapted" on
    # identifiers alone, so we expose the ISO anchor strings as the
    # pre-drift distinctive set and leave postdrift empty — the rubric
    # treats absence-of-predrift-markers as adaptation whenever
    # ``postdrift_identifiers`` is empty (DriftAdapt rubric case).
    return (
        conn,
        baseline_sql,
        gt_sql_predrift,
        gt_sql_postdrift,
        synopsis,
        frozenset(),
        frozenset({_ANCHOR_ISO, _ANCHOR_NEXT_ISO}),
    )


SPEC = ScenarioSpec(
    scenario_id="08_drift_date_format",
    family="events",
    tags=frozenset({"drift", "date_format", "iso_to_epoch", "events"}),
    drift_config=DriftConfig(
        kind="date_format",
        payload={"table": "events", "col": "ts", "from": "iso_string", "to": "epoch_ms"},
    ),
    builder=_build,
    base_scale=500,
)
