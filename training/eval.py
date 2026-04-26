"""Evaluation harness for SQLDrift.

K-rollouts-per-scenario sweep that emits a markdown ``report.md`` and a
row-level ``per_episode.csv`` so reviewers can eyeball pass rate and
reward distribution without spinning up a notebook.

Invocation::

    python -m training.eval \
        --checkpoint base \
        --scenarios 1-10 \
        --seeds-per-scenario 5 \
        --out outputs/evals/<run_id>/

``--checkpoint base`` runs the CPU :class:`RandomAgent` as a baseline
(no LLM, no GPU). A non-"base" value is a pointer to a saved adapter
and will attempt the lazy Unsloth import path in :func:`load_agent`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from models import REWARD_COMPONENT_KEYS, SqlDriftAction, SqlDriftObservation
from server import SqlDriftEnvironment
from training.config import ALL_SCENARIOS
from training.random_agent import RandomAgent
from training.seeding import set_seed
from utilities.logger import get_module_logger

_LOG = get_module_logger(__name__)

PASS_REWARD_THRESHOLD = 0.5
"""An episode is a "pass" when its terminal reward (the step that flips
``obs.done``—the submit step or the last step before budget exhaustion)
meets this threshold. The rubric scores a correct submission at
``+1.0`` in :attr:`r_correct`, so ``0.5`` matches the
"correct-but-not-improved" knee. Per-step values accumulate in
``episode_return`` and are not used for the pass bit."""


# -----------------------------------------------------------------------------
# Agent interface
# -----------------------------------------------------------------------------


class Agent(Protocol):
    """Duck-typed policy — :meth:`RandomAgent.act` fits this shape."""

    def reset(self, seed: int | None = None, scenario_id: str | None = None) -> None: ...
    def act(self, obs: SqlDriftObservation) -> SqlDriftAction: ...


def load_agent(
    checkpoint: str,
    *,
    seed: int = 0,
    base_model: str | None = None,
    temperature: float = 0.0,
) -> Agent:
    """Resolve a checkpoint spec to a concrete agent.

    * ``base``/``random`` — CPU-only :class:`RandomAgent` baseline.
    * Any other value is treated as a filesystem path (a full
      Hugging Face checkpoint directory or a PEFT adapter directory). The
      :class:`training.llm_agent.LLMAgent` is imported lazily so
      CPU-only CI that never calls ``load_agent`` with a path never
      has to install ``transformers``/``peft``.

    ``base_model`` is forwarded to :class:`LLMAgent` when the adapter
    directory does not pin its base model; ``temperature=0`` (greedy)
    is the default for deterministic eval sweeps.
    """
    if checkpoint in ("base", "random"):
        return RandomAgent(seed=seed)

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint {checkpoint!r} is not 'base'/'random' and does not exist on disk"
        )

    # Lazy import: keeps the heavy transformers/peft tree out of
    # the default import surface for ``training.eval``.
    from training.llm_agent import LLMAgent

    return LLMAgent(
        str(path),
        base_model=base_model,
        temperature=temperature,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# Episode runner
# -----------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    scenario_id: str
    seed: int
    terminal_reward: float
    episode_return: float
    steps: int
    passed: bool
    submitted: bool
    drift_fired: bool
    wall_ms: float
    reward_components: dict[str, float] = field(default_factory=dict)
    effective_speedup: float | None = None

    def as_row(self) -> dict[str, str]:
        row: dict[str, str] = {
            "scenario_id": self.scenario_id,
            "seed": str(self.seed),
            "terminal_reward": f"{self.terminal_reward:.4f}",
            "episode_return": f"{self.episode_return:.4f}",
            "steps": str(self.steps),
            "passed": "1" if self.passed else "0",
            "submitted": "1" if self.submitted else "0",
            "drift_fired": "1" if self.drift_fired else "0",
            "wall_ms": f"{self.wall_ms:.2f}",
            "effective_speedup": _format_speedup(self.effective_speedup),
        }
        for k in REWARD_COMPONENT_KEYS:
            row[k] = f"{self.reward_components.get(k, 0.0):.4f}"
        return row


def _format_speedup(value: float | None) -> str:
    """Render an effective_speedup cell for CSV output.

    ``None`` (no submission) and ``+∞`` (baseline invalidated by drift)
    need distinct, non-numeric representations so a downstream parser
    cannot conflate "no data" with "infinite" — both read back as empty
    cells today, which would bias per-scenario means.
    """
    if value is None:
        return ""
    if math.isinf(value):
        return "inf"
    return f"{value:.3f}"


def _effective_speedup(env: SqlDriftEnvironment) -> float | None:
    """Read the current episode's effective speedup through the env surface.

    Kept as a thin shim so tests and reporting code don't need to reach
    into env internals themselves.
    """
    return env.effective_speedup()


def _run_one_episode(
    env: SqlDriftEnvironment,
    agent: Agent,
    *,
    scenario_id: str,
    seed: int,
    max_steps: int = 30,
) -> EpisodeResult:
    t0 = time.perf_counter()
    obs = env.reset(seed=seed, scenario_id=scenario_id)
    agent.reset(seed=seed, scenario_id=scenario_id)

    episode_return = 0.0
    terminal_reward = 0.0
    last_components: dict[str, float] = {}
    steps = 0
    while not obs.done and steps < max_steps:
        action = agent.act(obs)
        obs = env.step(action)
        if obs.reward is not None:
            episode_return += obs.reward
            terminal_reward = obs.reward
        if obs.reward_components:
            last_components = dict(obs.reward_components)
        steps += 1

    state = env.state
    wall_ms = (time.perf_counter() - t0) * 1000.0

    return EpisodeResult(
        scenario_id=scenario_id,
        seed=seed,
        terminal_reward=terminal_reward,
        episode_return=episode_return,
        steps=steps,
        passed=terminal_reward >= PASS_REWARD_THRESHOLD,
        submitted=state.submitted,
        drift_fired=state.drift_fired,
        wall_ms=wall_ms,
        reward_components=last_components,
        effective_speedup=_effective_speedup(env),
    )


def run_eval(
    *,
    checkpoint: str,
    scenarios: list[str],
    seeds_per_scenario: int,
    out_dir: Path,
    max_steps: int = 30,
    base_seed: int = 0,
    progress_cb: Callable[[int, int], None] | None = None,
    base_model: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Drive the full sweep. Returns the summary dict also written to JSON.

    ``base_model`` / ``temperature`` are forwarded to
    :func:`load_agent` for LLM-checkpoint runs (ignored for
    ``base``/``random``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(base_seed)
    agent = load_agent(
        checkpoint,
        seed=base_seed,
        base_model=base_model,
        temperature=temperature,
    )

    results: list[EpisodeResult] = []
    env = SqlDriftEnvironment()
    total = len(scenarios) * seeds_per_scenario
    done = 0
    try:
        for scenario_id in scenarios:
            for k in range(seeds_per_scenario):
                seed = base_seed + k
                res = _run_one_episode(
                    env,
                    agent,
                    scenario_id=scenario_id,
                    seed=seed,
                    max_steps=max_steps,
                )
                results.append(res)
                done += 1
                if progress_cb:
                    progress_cb(done, total)
    finally:
        env.close()

    _write_per_episode_csv(results, out_dir / "per_episode.csv")
    summary = _build_summary(
        results,
        checkpoint=checkpoint,
        scenarios=scenarios,
        seeds_per_scenario=seeds_per_scenario,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.md").write_text(render_report(summary, results))
    return summary


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def _write_per_episode_csv(results: list[EpisodeResult], path: Path) -> None:
    if not results:
        path.write_text("")
        return
    fieldnames = list(results[0].as_row().keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_row())


def _speedup_stats(
    speedups: list[float],
) -> tuple[float | None, int]:
    """Finite-mean + ``+∞`` count for a list of effective_speedup values.

    ``statistics.fmean`` on a list containing ``+∞`` returns ``+∞``,
    which poisons JSON output (``json.dumps`` rejects ``Infinity`` by
    default) and washes out the finite distribution. We split the two
    channels: the reported mean is over *finite* speedups only, and the
    ``infinite_count`` is surfaced separately so reviewers can see how
    often drift invalidated the baseline.
    """
    finite = [s for s in speedups if not math.isinf(s)]
    infinite_count = len(speedups) - len(finite)
    mean = statistics.fmean(finite) if finite else None
    return mean, infinite_count


def _build_summary(
    results: list[EpisodeResult],
    *,
    checkpoint: str,
    scenarios: list[str],
    seeds_per_scenario: int,
) -> dict[str, Any]:
    by_scenario: dict[str, dict[str, Any]] = {}
    for sid in scenarios:
        scoped = [r for r in results if r.scenario_id == sid]
        if not scoped:
            continue
        terminals = [r.terminal_reward for r in scoped]
        returns = [r.episode_return for r in scoped]
        speedups = [r.effective_speedup for r in scoped if r.effective_speedup is not None]
        mean_sp, inf_sp = _speedup_stats(speedups)
        by_scenario[sid] = {
            "n": len(scoped),
            "pass_rate": sum(1 for r in scoped if r.passed) / len(scoped),
            "mean_terminal_reward": statistics.fmean(terminals),
            "std_terminal_reward": statistics.pstdev(terminals) if len(terminals) > 1 else 0.0,
            "mean_episode_return": statistics.fmean(returns),
            "submit_rate": sum(1 for r in scoped if r.submitted) / len(scoped),
            "mean_effective_speedup": mean_sp,
            "infinite_speedup_count": inf_sp,
        }

    all_speedups = [r.effective_speedup for r in results if r.effective_speedup is not None]
    mean_sp_all, inf_sp_all = _speedup_stats(all_speedups)
    overall = {
        "checkpoint": checkpoint,
        "n_episodes": len(results),
        "seeds_per_scenario": seeds_per_scenario,
        "pass_rate": sum(1 for r in results if r.passed) / len(results) if results else 0.0,
        "mean_terminal_reward": (
            statistics.fmean(r.terminal_reward for r in results) if results else 0.0
        ),
        "std_terminal_reward": (
            statistics.pstdev([r.terminal_reward for r in results]) if len(results) > 1 else 0.0
        ),
        "mean_episode_return": (
            statistics.fmean(r.episode_return for r in results) if results else 0.0
        ),
        "submit_rate": sum(1 for r in results if r.submitted) / len(results) if results else 0.0,
        "mean_effective_speedup": mean_sp_all,
        "infinite_speedup_count": inf_sp_all,
    }
    return {"overall": overall, "by_scenario": by_scenario}


def render_report(
    summary: dict[str, Any],
    results: list[EpisodeResult],
) -> str:
    """Compose a reviewer-friendly ``report.md`` string."""
    overall = summary["overall"]
    lines: list[str] = []
    lines.append("# SQLDrift evaluation report")
    lines.append("")
    speedup = overall.get("mean_effective_speedup")
    speedup_cell = f"{speedup:.2f}x" if speedup is not None else "—"
    lines.append(f"- Checkpoint: `{overall['checkpoint']}`")
    lines.append(f"- Episodes: **{overall['n_episodes']}**")
    lines.append(f"- Seeds/scenario: {overall['seeds_per_scenario']}")
    lines.append(
        f"- Overall pass rate (terminal reward ≥ {PASS_REWARD_THRESHOLD}): "
        f"**{overall['pass_rate']:.1%}**"
    )
    lines.append(
        f"- Mean terminal reward: **{overall['mean_terminal_reward']:.3f}** "
        f"(σ = {overall['std_terminal_reward']:.3f})"
    )
    lines.append(f"- Mean episode return: {overall['mean_episode_return']:.3f}")
    lines.append(f"- Submit rate: {overall['submit_rate']:.1%}")
    lines.append(f"- Mean effective speedup (finite, submitted episodes): **{speedup_cell}**")
    inf_count = overall.get("infinite_speedup_count", 0)
    if inf_count:
        lines.append(f"- Infinite-speedup episodes (drift invalidated baseline): **{inf_count}**")
    lines.append("")

    lines.append("## Per-scenario")
    lines.append("")
    lines.append("| Scenario | N | Pass | Terminal μ | Return μ | Submit | Speedup |")
    lines.append("|----------|---|------|-----------|----------|--------|---------|")
    for sid, row in summary["by_scenario"].items():
        sp = row.get("mean_effective_speedup")
        sp_cell = f"{sp:.2f}x" if sp is not None else "—"
        lines.append(
            f"| `{sid}` | {row['n']} | {row['pass_rate']:.0%} | "
            f"{row['mean_terminal_reward']:.3f} | {row['mean_episode_return']:.3f} | "
            f"{row['submit_rate']:.0%} | {sp_cell} |"
        )
    lines.append("")

    lines.append("## Reward-component bars (mean across episodes)")
    lines.append("")
    lines.append(_render_component_bars(results))
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Random baseline does NOT attempt rewrites intelligently; "
        "nonzero pass rate here is a lower bound on a well-trained agent."
    )
    lines.append(
        "- Pass threshold is set at reward ≥ "
        f"{PASS_REWARD_THRESHOLD}, matching the rubric's "
        '"correct-but-not-improved" +0.5 partial credit (rubric).'
    )
    return "\n".join(lines) + "\n"


def _render_component_bars(results: list[EpisodeResult]) -> str:
    if not results:
        return "_no data_"
    sums: dict[str, float] = {k: 0.0 for k in REWARD_COMPONENT_KEYS}
    counts: dict[str, int] = {k: 0 for k in REWARD_COMPONENT_KEYS}
    for r in results:
        for k, v in r.reward_components.items():
            if k in sums:
                sums[k] += v
                counts[k] += 1
    means = {k: (sums[k] / counts[k] if counts[k] else 0.0) for k in sums}

    # ASCII bar with sign (+/-).
    max_abs = max((abs(v) for v in means.values()), default=1.0) or 1.0
    width = 30
    lines = ["```"]
    for k in REWARD_COMPONENT_KEYS:
        v = means[k]
        bar_len = int(round(abs(v) / max_abs * width))
        bar = ("█" * bar_len) if v >= 0 else ("▒" * bar_len)
        lines.append(f"{k:<14} {v:+7.3f}  {bar}")
    lines.append("```")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _expand_scenarios(spec: str) -> list[str]:
    """Accept either ``1-10``, ``1,3,5``, or a comma-list of raw ids."""
    if "-" in spec and all(part.isdigit() for part in spec.split("-")):
        lo, hi = (int(x) for x in spec.split("-"))
        want_range: set[int] = set(range(lo, hi + 1))
        return [s for s in ALL_SCENARIOS if int(s.split("_", 1)[0]) in want_range]
    if all(part.strip().isdigit() for part in spec.split(",")):
        want_set = {int(part) for part in spec.split(",")}
        return [s for s in ALL_SCENARIOS if int(s.split("_", 1)[0]) in want_set]
    return [s.strip() for s in spec.split(",")]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate an agent on SQLDrift.")
    ap.add_argument("--checkpoint", required=True, help="'base' or adapter path")
    ap.add_argument(
        "--scenarios",
        default="1-10",
        help="Scenario range/list (e.g. '1-10', '1,3,5') or raw ids",
    )
    ap.add_argument("--seeds-per-scenario", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument(
        "--base-model",
        default=None,
        help="Override the base model id when --checkpoint points at a PEFT adapter",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature for LLM checkpoints (0 = greedy / deterministic).",
    )
    ns = ap.parse_args(argv)

    scenarios = _expand_scenarios(ns.scenarios)
    if not scenarios:
        raise SystemExit(f"no scenarios matched spec {ns.scenarios!r}")

    def _prog(done: int, total: int) -> None:
        if done == total or done % max(1, total // 10) == 0:
            _LOG.info("eval: %d/%d episodes", done, total)

    summary = run_eval(
        checkpoint=ns.checkpoint,
        scenarios=scenarios,
        seeds_per_scenario=ns.seeds_per_scenario,
        out_dir=ns.out,
        max_steps=ns.max_steps,
        base_seed=ns.base_seed,
        progress_cb=_prog,
        base_model=ns.base_model,
        temperature=ns.temperature,
    )
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "Agent",
    "EpisodeResult",
    "PASS_REWARD_THRESHOLD",
    "load_agent",
    "main",
    "render_report",
    "run_eval",
]
