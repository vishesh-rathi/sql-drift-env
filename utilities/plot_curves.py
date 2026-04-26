"""Produce the demo curves from a completed GRPO run.

Reads the per-step JSONL written by ``training.grpo_train._FlushLogHistory``
and emits ``training/evidence/grpo_metrics.csv`` plus EMA-smoothed PNGs.

Per-component reward plots are produced *first* — they tell the actual
training story (correctness, drift adaptation, etc.). The summed
``reward`` curve is kept for completeness but published last because
``episode_return`` (sum of per-step shaping) tracks trajectory length
more than correctness; see the reward-balance notes in the hackathon audit.

Usage::

    python utilities/plot_curves.py [PATH_TO_log_history.jsonl]

If no path is given, defaults to ``outputs/grpo_run/log_history.jsonl``
(the location written by ``train()`` when ``output_dir=outputs/grpo_run``).

Requires: ``uv sync --extra evidence`` (or ``pip install -e .[evidence]``) for
``matplotlib`` and ``pandas``.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EVIDENCE = Path("training/evidence")
EVIDENCE.mkdir(parents=True, exist_ok=True)

# 1. Load the JSONL the _FlushLogHistory callback wrote per step.
log_jsonl = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/grpo_run/log_history.jsonl")
records = [json.loads(line) for line in log_jsonl.read_text().splitlines() if line.strip()]
df = pd.DataFrame(records)
if df.empty:
    raise SystemExit(f"No records in {log_jsonl}")

# Persist raw metrics for the demo.
df.to_csv(EVIDENCE / "grpo_metrics.csv", index=False)
print(f"Wrote CSV: {len(df)} rows")


def _ema(s: pd.Series, span: int = 10) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _plot(df: pd.DataFrame, ycol: str, title: str, fname: str, ylabel: str) -> None:
    plt_df = df[["step", ycol]].dropna()
    if plt_df.empty:
        print(f"SKIP {ycol} — no data")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        plt_df["step"],
        plt_df[ycol],
        marker="o",
        linewidth=1.0,
        alpha=0.5,
        label=ycol,
    )
    ax.plot(
        plt_df["step"],
        _ema(plt_df[ycol]),
        linewidth=2.4,
        label="EMA(span=10)",
    )
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVIDENCE / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {fname}")


def _plot_component(df: pd.DataFrame, ycol: str) -> None:
    """Per-component plot. Missing column -> one-line warning, no abort."""
    if ycol not in df.columns:
        print(f"WARN {ycol} — column missing from log; skipping {ycol} plot")
        return
    _plot(df, ycol, f"SQLDrift GRPO — {ycol}", f"grpo_{ycol}_curve.png", ycol)


# Per-component plots come first. Order: correctness, drift, then loss, reward,
# then remaining shaping signals so the blog narrative leads with the
# correctness story rather than the noisier sum-of-shaping curve.
_plot_component(df, "r_correct")
_plot_component(df, "r_drift")

_plot(df, "loss", "SQLDrift GRPO — Loss", "grpo_loss_curve.png", "loss")
_plot(
    df,
    "reward",
    "SQLDrift GRPO — Mean Episode Return (sum of per-step shaping; see component plots for correctness signal)",
    "grpo_reward_curve.png",
    "reward (sum of per-step shaping)",
)

# Remaining components (after r_correct and r_drift handled above).
for comp in ("r_speedup", "r_step_tax", "r_gatekeepers"):
    _plot_component(df, comp)


# Combined per-component decomposition: one figure, EMA-smoothed lines for
# every available r_* column, 300 DPI 16:9 for the blog hero shot.
COMPONENT_KEYS = ("r_correct", "r_drift", "r_speedup", "r_step_tax", "r_gatekeepers")
present = [k for k in COMPONENT_KEYS if k in df.columns]
if not present:
    print("WARN no r_* component columns found; skipping combined plot")
else:
    fig, ax = plt.subplots(figsize=(16, 9))
    for key in present:
        series = df[["step", key]].dropna()
        if series.empty:
            continue
        ax.plot(series["step"], _ema(series[key]), linewidth=2.2, label=f"{key} (EMA)")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("reward component")
    ax.set_title("SQLDrift GRPO — Per-component reward decomposition")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(EVIDENCE / "grpo_components_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote grpo_components_combined.png ({len(present)} components)")
