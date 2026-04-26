"""P13 — end-to-end :func:`training.eval.run_eval` sweep on a subset.

Runs the random-baseline agent across 2 scenarios × 1 seed (so the
test stays fast) and asserts the writer emits ``report.md`` and
``per_episode.csv`` with the expected rows.

The full "5 seeds × 10 scenarios" eval sweep is produced
as a real artifact outside pytest; this keeps CI time bounded.
"""

from __future__ import annotations

import csv
from pathlib import Path

from training.eval import run_eval


def test_run_eval_writes_expected_outputs(tmp_path: Path) -> None:
    scenarios = ["01_correlated_subquery", "07_drift_column_rename"]
    summary = run_eval(
        checkpoint="base",
        scenarios=scenarios,
        seeds_per_scenario=1,
        out_dir=tmp_path,
        max_steps=30,
        base_seed=7,
    )

    # Artifacts exist
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "per_episode.csv").exists()
    assert (tmp_path / "summary.json").exists()

    # summary shape
    assert summary["overall"]["n_episodes"] == 2
    assert set(summary["by_scenario"].keys()) == set(scenarios)

    # CSV row count matches
    with (tmp_path / "per_episode.csv").open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert {r["scenario_id"] for r in rows} == set(scenarios)

    # report.md mentions each scenario
    report = (tmp_path / "report.md").read_text()
    for sid in scenarios:
        assert sid in report
