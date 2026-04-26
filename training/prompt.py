"""Single-source-of-truth system prompt for SQLDrift agents.

Every component that constructs an agent context (GRPO trainer,
``random_agent``, eval harness, demo notebook) goes through
:func:`render_system_prompt` so the tool surface described to the
model stays in lockstep with :mod:`models.ToolName` and the tool
payload schemas.

The rendered string is tokenizer-agnostic: no chat-template
markers, no special tokens, no role wrappers. Callers that need a
chat format wrap the returned string with their tokenizer's
``apply_chat_template``.
"""

from __future__ import annotations

from models import EpisodePhase, SqlDriftObservation, ToolName

# -----------------------------------------------------------------------------
# Tool catalog — keep in sync with ``models.ToolName`` + payload schemas.
# The env enforces argument shapes server-side, but the agent needs a
# human-readable cheat sheet to plan its turn.
# -----------------------------------------------------------------------------

TOOL_DOCS: dict[ToolName, dict[str, str]] = {
    ToolName.LIST_TABLES: {
        "signature": "list_tables()",
        "purpose": "Enumerate tables visible to the session (cheap, always safe).",
    },
    ToolName.DESCRIBE_TABLE: {
        "signature": "describe_table(table: str)",
        "purpose": "Return column names + types for one table.",
    },
    ToolName.SAMPLE_ROWS: {
        "signature": "sample_rows(table: str, limit: int ∈ [1, 5] = 5)",
        "purpose": "Peek at up to 5 rows for fast schema intuition.",
    },
    ToolName.RUN_QUERY: {
        "signature": "run_query(sql: str)",
        "purpose": (
            "Execute a read-only SELECT against the live database. "
            "Timing counts toward the step budget; repeat-failing queries are "
            "penalised."
        ),
    },
    ToolName.EXPLAIN_QUERY: {
        "signature": "explain_query(sql: str)",
        "purpose": "Return the DuckDB plan for a SELECT (no execution).",
    },
    ToolName.READ_CHANGELOG: {
        "signature": "read_changelog()",
        "purpose": (
            "Read all drift-related deploy notes published so far. Always "
            "consult this after drift is announced in an observation."
        ),
    },
    ToolName.SUBMIT_REWRITE: {
        "signature": "submit_rewrite(sql: str)",
        "purpose": (
            "Commit your final SELECT. Terminates the episode. Reward requires "
            "the result to match ground truth AND the rewrite to be ≥1.2x "
            "faster than the baseline query."
        ),
    },
    ToolName.CONSULT_DBA: {
        "signature": "consult_dba(question: str)",
        "purpose": (
            "Ask the on-call DBA for a hint. Each consultation escalates the "
            "hint tier and incurs a compounding penalty; use sparingly and "
            "only after diagnostics."
        ),
    },
}


PHASE_NUDGES: dict[EpisodePhase, str] = {
    EpisodePhase.DIAGNOSE: (
        "You are in DIAGNOSE. Explore the schema and sample data. Do NOT submit a rewrite yet."
    ),
    EpisodePhase.REWRITE: (
        "You are in REWRITE. Draft candidate queries with run_query and, "
        "once confident, call submit_rewrite."
    ),
    EpisodePhase.DRIFT_RECOVERY: (
        "Drift has fired. Read the changelog, re-describe affected tables, "
        "and adapt your rewrite before submitting."
    ),
    EpisodePhase.FINALIZE: "The episode is finalizing; no further tools will help.",
}


SYSTEM_PROMPT_HEADER = (
    "You are a senior SQL engineer operating an analytical database that is "
    "under live schema and business-rule drift. Your job is to repair and "
    "optimize a slow baseline SELECT under tight step and runtime budgets. "
    "Prefer read-only tools; never emit DDL or DML (INSERT/UPDATE/DELETE). "
    "When a changelog is published, treat it as authoritative."
)


def _render_tool_catalog(dba_enabled: bool = False) -> str:
    lines = ["Tools available (exact JSON shapes enforced by the env):"]
    for tool in ToolName:
        if tool == ToolName.CONSULT_DBA and not dba_enabled:
            continue
        doc = TOOL_DOCS[tool]
        lines.append(f"- {doc['signature']}: {doc['purpose']}")
    return "\n".join(lines)


def render_system_prompt(
    *,
    scenario_id: str,
    learned_hints: str = "",
    phase: EpisodePhase = EpisodePhase.DIAGNOSE,
    budget_steps_remaining: int | None = None,
    drift_fired: bool = False,
    dba_enabled: bool = False,
) -> str:
    """Render the per-episode system prompt.

    Args:
        scenario_id: Current scenario id (so the model sees context).
        learned_hints: Pre-rendered bullet list from the skill library
            (already capped at 800 chars by the env).
        phase: Current episode phase — drives the phase nudge line.
        budget_steps_remaining: If provided, surfaces the hard budget.
        drift_fired: If True, the drift-recovery nudge is reinforced.
    """
    parts: list[str] = [SYSTEM_PROMPT_HEADER, _render_tool_catalog(dba_enabled=dba_enabled)]
    parts.append(f"Current scenario: {scenario_id}")
    if budget_steps_remaining is not None:
        parts.append(
            f"Remaining step budget: {budget_steps_remaining}. Each tool "
            "call costs one step; plan accordingly."
        )
    parts.append(PHASE_NUDGES.get(phase, ""))
    if drift_fired:
        parts.append(
            "Drift has already fired in this episode — if you have not yet "
            "called read_changelog since, do that FIRST."
        )
    if learned_hints:
        parts.append("Learned hints (from past episodes):\n" + learned_hints)
    return "\n\n".join(p for p in parts if p).strip()


def render_prompt_from_observation(
    *,
    scenario_id: str,
    observation: SqlDriftObservation,
) -> str:
    """Convenience wrapper: pull phase / hints / budget from an observation."""
    return render_system_prompt(
        scenario_id=scenario_id,
        learned_hints=observation.learned_hints,
        phase=observation.phase,
        budget_steps_remaining=observation.budget_steps_remaining,
        drift_fired=observation.drift_fired,
    )


__all__ = [
    "PHASE_NUDGES",
    "SYSTEM_PROMPT_HEADER",
    "TOOL_DOCS",
    "render_prompt_from_observation",
    "render_system_prompt",
]
