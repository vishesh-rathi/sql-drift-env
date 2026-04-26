---
title: SQLDrift
emoji: 🐘
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
short_description: SQL repair and optimization under live schema drift
---

# SQLDrift

> An OpenEnv gym that teaches an LLM agent to **repair and optimize SQL** when the
> database schema and business rules shift out from under it.

## Why this problem exists

Production analytics breaks in familiar ways: a deploy renames a column, splits an enum
into two fields, or changes how dates are stored. The dashboard query that worked on
Monday may error or return wrong results on Tuesday. LLMs are a natural fit—they can
read changelogs, run diagnostics, and propose rewrites—but static SQL benchmarks assume
a **fixed** schema. They cannot reward an agent for noticing that the world changed
**during** a task or for trading correctness against latency under a tool budget.

**SQLDrift** closes that gap: each episode is a small DuckDB world, a slow baseline
`SELECT`, a fixed tool budget, and (for drift scenarios) a mid-episode schema or rule
change the agent must recover from before submitting a rewrite that is both semantically
correct and materially faster than the baseline.

## Live environment (Hugging Face Space)

Run the OpenEnv server without cloning the repo:

**[SQLDrift on Hugging Face Spaces](https://huggingface.co/spaces/visheshrathi/sql-drift-env)**

For TRL / client rollouts, point `SQL_DRIFT_ENV_URL` (or your `SqlDriftEnv` /
`SqlDriftToolEnv` base URL) at the Space runtime, for example
`https://visheshrathi-sql-drift-env.hf.space` — same host the training notebook
health-checks by default. If you need many concurrent WebSocket sessions (e.g.
`num_generations` > 1 in GRPO), duplicate the Space to your own account per the
[TRL OpenEnv concurrency notes](https://huggingface.co/docs/trl/openenv#server-concurrency).

SQLDrift is a production-grade [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
environment for RL fine-tuning with TRL’s
[`GRPOTrainer`](https://huggingface.co/docs/trl/openenv) (and
[Unsloth](https://github.com/unslothai/unsloth) LoRA on consumer GPUs). Each episode
gives the agent a slow baseline query, a lightly populated DuckDB database, read-only
introspection and execution tools, and a **25-step budget**. Mid-episode, a schema or
business-rule **drift event** may fire; the agent must read the deploy changelog,
re-diagnose, and submit a rewrite that is (a) semantically correct and (b) ≥ **1.2×**
faster than the baseline. A hierarchical rubric turns that into six reward signals
(correctness, drift-adapt, speedup, step-tax, gatekeepers, consult-dba).

Highlights:

- **10 hand-crafted scenarios** — 6 classic anti-patterns (correlated subqueries,
  `SELECT *` joins, Cartesian joins, `DISTINCT` groupbys, nested subqueries,
  `HAVING`-as-`WHERE`) and 4 drift events (column rename, date format flip, enum rule
  split, field deprecation).
- **Deterministic fixtures** — 15–30-column schemas, 2–4 tables per scenario,
  regenerated in-process at `reset()` from a seed. No Parquet files, no pre-baked
  sqlite dumps.
- **Sqlglot-canonicalised baseline-verbatim gate** — agents that rename whitespace and
  resubmit the baseline cannot farm the +1.0 correct bonus.
- **Skill library** — 12 pre-seeded playbook/drift-card entries plus an on-disk,
  `fcntl.flock`-guarded JSON store that grows as the agent solves episodes, surfaced
  next `reset()` via Jaccard retrieval.
- **Feature-flagged DBA Oracle** — 3-tier escalating hints per scenario, penalised by
  the `ConsultDBA` rubric. Off by default.
- **Dockerised** — `server/Dockerfile` layers the env on `ghcr.io/meta-pytorch/openenv-base`
  and exposes `/health`, `/reset`, `/step`, `/ws` out-of-the-box.

## How the environment works

1. **`reset(scenario_id, seed, …)`** builds a fresh in-memory DuckDB instance, loads the
   scenario’s tables and rows, and returns an observation with the baseline SQL, phase
   (`DIAGNOSE` → …), step budget, and `learned_hints` from the skill library.
2. **Tooling** — The agent acts via OpenEnv actions: list/describe tables, sample rows,
   `EXPLAIN`, timed read-only `run_query`, `read_changelog`, optional `consult_dba`, and
   terminal `submit_rewrite`. Each step consumes budget and accrues shaping reward (e.g.
   step tax and gatekeeper penalties).
3. **Drift** — On drift scenarios, an event fires in a configured step window: DDL / rule
   operators run idempotently (`engine/drift.py`); the changelog updates; ground truth and
   baseline behaviour may change; the observation signals `drift_recovery` so the agent can
   adapt.
4. **Verification** — Submissions are checked with an order-independent result hash
   (`engine/verifier.py`) and median-of-3 timed execution against the baseline
   (`engine/profiler.py`). Correctness and speedup components of the rubric enforce the
   ≥ 1.2× speedup bar for the top correctness tier.
5. **Reward** — `SqlDriftRubric` composes six child rubrics; every observation exposes
   `reward_components` (e.g. `r_correct`, `r_drift`, `r_speedup`, `r_step_tax`,
   `r_gatekeepers`, `r_consult_dba`) for analysis and logging.
6. **Server** — `server/app.py` uses the OpenEnv FastAPI factory; `client.py` provides a
   WebSocket `SqlDriftEnv` for remote episodes. See `openenv.yaml` for the Space-oriented
   manifest.

## Results

### Random baseline (floor)

We ship a reproducible random-agent evaluation under
`outputs/evals/baseline_random_v1/` (50 episodes, 5 seeds × 10 scenarios). It establishes
a **floor** for any trained policy:

| Metric                   | Value                 |
| ------------------------ | --------------------- |
| Mean total reward        | **−2.048** (σ ≈ 0.70) |
| Pass rate (reward ≥ 0.5) | **0%**                |
| Submit rate              | 80%                   |

Mean reward is roughly **−1.8 to −2.3** per scenario; drift scenarios trend slightly
harder than static anti-pattern scenarios. See
[`outputs/evals/baseline_random_v1/report.md`](outputs/evals/baseline_random_v1/report.md)
for the full per-scenario table and component bars.

### Trained policies

GRPO training is driven by `training/grpo_train.py` and `sql_drift_grpo_training.ipynb`.
After a run, compare checkpoints with:

```bash
uv run python -m training.eval \
    --checkpoint path/to/adapter \
    --scenarios 1-10 \
    --seeds-per-scenario 5 \
    --out outputs/evals/my_run
```

Training logs and plots: `training/grpo_train.py` can emit step-wise JSONL;
`utilities/plot_curves.py` produces reward/loss figures under `training/evidence/` when
that log exists.

## Quick start

```bash
git clone <your-fork>/sql_drift_env.git
cd sql_drift_env
uv sync                                        # install deps
uv run pytest -q                               # 300+ tests, ~90s on CPU
uv run uvicorn server.app:app --reload         # serve the env on :8000
```

Or via Docker:

```bash
docker build -f server/Dockerfile -t sql-drift-env:latest .
docker run -p 8000:8000 sql-drift-env:latest
curl -s http://localhost:8000/health
```

## Programmatic rollout

A five-step in-process episode against scenario `01_correlated_subquery`:

```python
from client import SqlDriftEnv
from server import SqlDriftEnvironment

env = SqlDriftEnvironment()
obs = env.reset(seed=42, scenario_id="01_correlated_subquery")
print(obs.learned_hints)             # surfaced from skill library

obs = env.step(SqlDriftEnv.action_list_tables())
obs = env.step(SqlDriftEnv.action_describe_table("users"))
obs = env.step(SqlDriftEnv.action_run_query("SELECT COUNT(*) FROM users"))
obs = env.step(SqlDriftEnv.action_submit_rewrite(
    "SELECT u.*, COALESCE(c.n, 0) FROM users u "
    "LEFT JOIN (SELECT user_id, COUNT(*) AS n FROM orders GROUP BY 1) c "
    "ON c.user_id = u.user_id"
))
print(obs.reward, obs.reward_components)
env.close()
```

End-to-end over an HTTP+WS OpenEnv server, see `SqlDriftEnv` in `client.py` and the
integration test suite in `tests/integration/test_client_server.py` /
`tests/integration/test_state_no_leak.py`.

## Evaluation

```bash
uv run python -m training.eval \
    --checkpoint base \
    --scenarios 1-10 \
    --seeds-per-scenario 5 \
    --out outputs/evals/my_run
```

Emits `report.md`, `per_episode.csv`, and `summary.json`.

## Training (GPU)

`training/grpo_train.py` contains the GRPO entrypoint used by the hackathon training
notebook: it builds the curriculum dataset, loads **`Qwen/Qwen3-1.7B`** (or your
`SQL_DRIFT_MODEL_NAME`) via `transformers.AutoModelForCausalLM` + `BitsAndBytesConfig`
4-bit nf4 (QLoRA), attaches a PEFT LoRA adapter, and lets TRL's `GRPOTrainer` drive
multi-turn OpenEnv rollouts through `SqlDriftToolEnv` via `environment_factory`. The
install/runtime stack mirrors Hugging Face TRL's reference notebooks
(`grpo_trl_lora_qlora.ipynb` + `openenv_wordle_grpo.ipynb`). Open
`sql_drift_grpo_training.ipynb` on a free Colab T4, set `SQL_DRIFT_ENV_URL` to your
deployed SQLDrift Space (see [Live environment](#live-environment-hugging-face-space)),
and run all cells.

```bash
uv sync --extra train           # installs trl, transformers, accelerate, unsloth
uv sync --extra evidence        # matplotlib + pandas for utilities/plot_curves.py
```

## Repository layout

```
sql_drift_env/
├── models.py                   # Pydantic v2 action/observation/state
├── client.py                   # SqlDriftEnv EnvClient (/ws)
├── engine/
│   ├── runtime.py              # private RuntimeEpisodeState
│   ├── drift.py                # 4 DDL drift operators
│   ├── reward.py               # SqlDriftRubric (6 child rubrics)
│   ├── verifier.py
|   |-- profiler.py
├── scenarios/                  # 10 hand-crafted scenario modules + registry
├── skill_library/              # pre-seeds + JSON store + Jaccard retrieval
├── actors/                     # engineering_manager (changelog), dba_oracle
├── server/                     # FastAPI app, Dockerfile, env wrapper class
├── training/                   # config, prompt, random_agent, grpo_train, eval
├── utilities/                  # env_loader, logger, plot_curves, run_training_job (HF Jobs), …
├── tests/                      # 300+ unit + integration tests
├── outputs/evals/              # baseline eval artifacts
└── design/                     # design docs (ignored by docker)
```

## References

- **[SQLDrift — Hugging Face Space](https://huggingface.co/spaces/visheshrathi/sql-drift-env)** —
  deployed OpenEnv server (Docker SDK).
