"""Demo rollout — walk a hand-scripted agent through a drift scenario.

Run with:

    uv run python utilities/demo_rollout.py

The script instantiates :class:`SqlDriftEnvironment` in-process (no
HTTP), plays a scripted solve of ``07_drift_column_rename`` (drift
fires mid-episode; agent reads the changelog, adapts, submits), then
dumps per-step state so you can eyeball what a well-trained agent
would see at inference time.

Intentionally dependency-light — uses only stdlib + the env itself.
"""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path

# Make the repo root importable when the script is run as
# ``python utilities/demo_rollout.py`` from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from client import SqlDriftEnv
from models import (
    ConsultDBAResult,
    DescribeTableResult,
    ExplainQueryResult,
    ListTablesResult,
    ReadChangelogResult,
    RunQueryResult,
    SampleRowsResult,
    SqlDriftAction,
    SqlDriftObservation,
    SubmitRewriteResult,
    ToolError,
)
from server import SqlDriftEnvironment


def _fmt_tool_result(obs: SqlDriftObservation) -> str:
    result = obs.tool_result
    if result is None:
        return "∅"
    if isinstance(result, ListTablesResult):
        preview = ", ".join(result.tables)
    elif isinstance(result, DescribeTableResult):
        preview = " | ".join(f"{c['name']}:{c['type']}" for c in result.columns[:5])
    elif isinstance(result, SampleRowsResult):
        preview = f"{len(result.rows)} rows over {len(result.columns)} cols"
    elif isinstance(result, RunQueryResult):
        preview = f"{result.row_count} rows in {result.runtime_ms:.1f} ms"
    elif isinstance(result, ExplainQueryResult):
        preview = f"plan={result.plan[:60]!r}"
    elif isinstance(result, SubmitRewriteResult):
        preview = (
            f"accepted={result.accepted} match_gt={result.matches_ground_truth} "
            f"{result.runtime_ms:.1f}ms"
        )
    elif isinstance(result, ReadChangelogResult):
        preview = " || ".join(result.entries) or "(empty)"
    elif isinstance(result, ConsultDBAResult):
        preview = f"tier={result.tier} hint={result.hint!r}"
    elif isinstance(result, ToolError):
        preview = f"ERROR[{result.code}] {result.message}"
    else:
        preview = result.kind
    return f"{result.kind}: {preview}"


def _print_header() -> None:
    print("=" * 78)
    print("SQLDrift demo rollout — 07_drift_column_rename")
    print("=" * 78)


def _print_obs(step_idx: int, action: SqlDriftAction | None, obs: SqlDriftObservation) -> None:
    action_repr = (
        f"{action.tool.value}({action.payload.model_dump(exclude={'kind'})})"
        if action is not None
        else "reset"
    )
    hints = textwrap.shorten(obs.learned_hints or "(none)", width=70)
    print(f"\n[step {step_idx:02d}] action={action_repr}")
    print(
        f"         phase={obs.phase}  drift_fired={obs.drift_fired}  "
        f"budget={obs.budget_steps_remaining}  reward={obs.reward}"
    )
    print(f"         result: {_fmt_tool_result(obs)}")
    print(f"         hints: {hints}")
    if obs.reward_components:
        parts = ", ".join(f"{k}={v:+.2f}" for k, v in obs.reward_components.items())
        print(f"         components: {parts}")


def _scripted_actions() -> Iterable[SqlDriftAction]:
    """Hand-scripted trajectory for the column-rename drift scenario.

    We intentionally issue a post-drift submission that uses the
    *renamed* column so the rubric awards the full correctness +
    speedup bundle (baseline fails after drift → speedup = +∞).
    """
    yield SqlDriftEnv.action_list_tables()
    yield SqlDriftEnv.action_describe_table("orders")
    yield SqlDriftEnv.action_sample_rows("orders", limit=3)
    yield SqlDriftEnv.action_run_query("SELECT COUNT(*) FROM orders WHERE user_id IS NOT NULL")
    # Cross the drift trigger window.
    yield SqlDriftEnv.action_list_tables()
    yield SqlDriftEnv.action_describe_table("orders")
    yield SqlDriftEnv.action_read_changelog()
    yield SqlDriftEnv.action_run_query("SELECT COUNT(*) FROM orders WHERE account_id IS NOT NULL")
    yield SqlDriftEnv.action_submit_rewrite(
        "SELECT o.account_id, COUNT(*) AS n FROM orders o GROUP BY 1"
    )


def run() -> None:
    _print_header()
    env = SqlDriftEnvironment()
    try:
        obs = env.reset(seed=7, scenario_id="07_drift_column_rename")
        _print_obs(0, None, obs)
        for i, action in enumerate(_scripted_actions(), start=1):
            if obs.done:
                break
            obs = env.step(action)
            _print_obs(i, action, obs)
        print("\n" + "=" * 78)
        print(f"Episode done: reward={obs.reward} components={obs.reward_components}")
        print("Final state:", env.state.model_dump())
    finally:
        env.close()


if __name__ == "__main__":
    run()
