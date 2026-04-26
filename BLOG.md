# SQL Under Drift: Why Static Benchmarks Miss the Point

Production databases do not hold still. Schemas get evolved, columns are renamed, enums split, formats change, and deploy changelogs record what humans already agreed on. Training agents only on fixed schemas rewards a skill that rarely matches the job: adapting while the workflow is still in flight.

**SQLDrift** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-style gym built for that harder setting. Each episode gives an agent a real DuckDB instance, a toolbox of read-only actions (`list_tables`, `describe_table`, `sample_rows`, `run_query`, `explain_query`, `read_changelog`, `submit_rewrite`, and optional DBA guidance), and a bounded step budget. Drift can arrive mid-episode: schema and business-rule changes are applied in the environment, not only described in prose. Rewards combine semantic correctness (canonical result checks), adaptation after drift, measured runtime improvement against a calibrated baseline, and sensible penalties for wasted steps and brittle patterns—so “correct but lazy” and “fast but wrong” both fail in predictable ways.

## How this differs from recent SQL RL lines

The last year has seen strong momentum in SQL-focused RL: executable feedback, equivalence-aware rewrites, and efficiency signals are now familiar ingredients in lines such as BIRD-CRITIC / `six-gym-sqlite`, the BIRD-Talon and BIRD-Zeno–style models, and work like E3-Rewrite that optimizes for executability, equivalence, and speed. Those settings still largely treat the world as **stable for the episode**: the puzzle is to fix or improve a query against a schema that does not reorganize underneath you.

SQLDrift’s emphasis is different. The novelty is not “SQL plus RL” by itself—it is **live schema and business-rule drift during the rollout**, with changelog-grounded adaptation, long-horizon tool use, and deterministic validation in a stateful loop. Agents must read what changed, re-verify behavior, and submit a rewrite that is both faithful to the new rules and meaningfully faster than the baseline. That combination targets the gap between leaderboard SQL and the kind of maintenance engineers actually perform.

## Why it matters

We are not claiming a world without prior art; we are pointing at a clear axis of generalization. When the environment can shift mid-episode, memorizing one layout stops being enough. SQLDrift offers a compact, reproducible setting to train and measure that skill: demanding for the policy, and closer to how deployments actually behave than a single frozen schema alone can provide.
