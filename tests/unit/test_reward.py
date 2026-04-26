"""P6 — unit tests for :mod:`engine.reward`.

Every branch of every child rubric is exercised with a synthesized
``(RuntimeEpisodeState, SqlDriftAction, SqlDriftObservation)`` triple —
no DuckDB dependency required. The Rev-3 invariant
``no_op_rewrite_does_not_grant_full_credit`` (rubric doc) is codified below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from engine.reward import (
    CONSULT_ESCALATION,
    GATE_BASELINE_VERBATIM,
    GATE_CONSECUTIVE_TOOL_ERROR,
    GATE_MALFORMED_TOOL_CALL,
    GATE_REPEAT_FAILING_QUERY,
    SPEEDUP_MIN,
    STEP_REBATE_READ_CHANGELOG,
    STEP_TAX,
    SqlDriftRubric,
    canonicalize_sql,
)
from models import (
    REWARD_COMPONENT_KEYS,
    ConsultDBAPayload,
    EpisodePhase,
    RunQueryPayload,
    RunQueryResult,
    SqlDriftAction,
    SqlDriftObservation,
    SubmitRewritePayload,
    SubmitRewriteResult,
    ToolError,
    ToolErrorCode,
    ToolName,
)

# -----------------------------------------------------------------------------
# Fake RuntimeEpisodeState that mimics engine.runtime.RuntimeEpisodeState.
# Kept local to avoid instantiating a real DuckDB connection in rubric tests.
# -----------------------------------------------------------------------------


@dataclass
class _FakeInstance:
    postdrift_identifiers: frozenset[str] = field(default_factory=frozenset)
    predrift_identifiers: frozenset[str] = field(default_factory=frozenset)


@dataclass
class _FakeRuntime:
    gt_result_hash_predrift: str = "GT_PRE"
    gt_result_hash_postdrift: str | None = None
    baseline_runtime_ms: float = 100.0
    baseline_sql_canonical: str = "select 1"
    baseline_postdrift_raises: bool = False
    submitted: bool = False
    submitted_sql: str | None = None
    submitted_sql_canonical: str | None = None
    submitted_result_hash: str | None = None
    submitted_runtime_ms: float | None = None
    drift_fired_step: int | None = None
    consultations_used: int = 0
    last_step_was_repeat_failing_query: bool = False
    last_step_repeat_failing_query_count: int = 0
    last_step_productive_rebate: float = 0.0
    consecutive_tool_errors: int = 0
    dba_oracle_enabled: bool = False
    instance: _FakeInstance = field(default_factory=_FakeInstance)

    @property
    def drift_fired(self) -> bool:
        return self.drift_fired_step is not None


def _rubric(rt: _FakeRuntime) -> SqlDriftRubric:
    return SqlDriftRubric(ctx_provider=lambda: rt)


def _submit_action(sql: str = "SELECT * FROM t") -> SqlDriftAction:
    return SqlDriftAction(
        tool=ToolName.SUBMIT_REWRITE,
        payload=SubmitRewritePayload(sql=sql),
    )


def _run_query_action(sql: str = "SELECT 1") -> SqlDriftAction:
    return SqlDriftAction(
        tool=ToolName.RUN_QUERY,
        payload=RunQueryPayload(sql=sql),
    )


def _submit_obs(accepted: bool = True, runtime_ms: float = 50.0) -> SqlDriftObservation:
    return SqlDriftObservation(
        step=10,
        phase=EpisodePhase.FINALIZE,
        last_tool=ToolName.SUBMIT_REWRITE,
        tool_result=SubmitRewriteResult(
            accepted=accepted, runtime_ms=runtime_ms, matches_ground_truth=accepted
        ),
        budget_steps_remaining=0,
        done=True,
        reward=None,
    )


def _run_query_obs(result_rows: int = 1) -> SqlDriftObservation:
    return SqlDriftObservation(
        step=3,
        phase=EpisodePhase.DIAGNOSE,
        last_tool=ToolName.RUN_QUERY,
        tool_result=RunQueryResult(
            columns=["c"], rows=[[1]] * result_rows, runtime_ms=0.5, row_count=result_rows
        ),
        budget_steps_remaining=22,
        reward=None,
    )


def _tool_error_obs() -> SqlDriftObservation:
    return SqlDriftObservation(
        step=4,
        phase=EpisodePhase.DIAGNOSE,
        last_tool=ToolName.RUN_QUERY,
        tool_result=ToolError(code=ToolErrorCode.DB_ERROR, message="boom"),
        budget_steps_remaining=21,
        reward=None,
    )


# =============================================================================
# Correctness
# =============================================================================


class TestCorrectness:
    def test_not_yet_submitted_is_zero(self) -> None:
        rt = _FakeRuntime()
        r = _rubric(rt)
        assert r.correctness(_run_query_action(), _run_query_obs()) == 0.0

    def test_correct_and_fast_grants_full_credit(self) -> None:
        rt = _FakeRuntime(
            submitted=True,
            submitted_result_hash="GT_PRE",
            submitted_runtime_ms=10.0,  # 100ms baseline / 10ms → 10x speedup
            baseline_runtime_ms=100.0,
        )
        r = _rubric(rt)
        assert r.correctness(_submit_action(), _submit_obs()) == 1.0

    def test_correct_but_not_fast_grants_half_credit(self) -> None:
        rt = _FakeRuntime(
            submitted=True,
            submitted_result_hash="GT_PRE",
            submitted_runtime_ms=100.0,  # 1.0x — below 1.2 threshold
            baseline_runtime_ms=100.0,
        )
        r = _rubric(rt)
        assert r.correctness(_submit_action(), _submit_obs()) == 0.5

    def test_wrong_hash_is_negative_one(self) -> None:
        rt = _FakeRuntime(
            submitted=True,
            submitted_result_hash="WRONG",
            submitted_runtime_ms=10.0,
            baseline_runtime_ms=100.0,
        )
        r = _rubric(rt)
        assert r.correctness(_submit_action(), _submit_obs()) == -1.0

    def test_drift_fired_baseline_raises_gives_effective_infinity(self) -> None:
        """On a drift where baseline fails post-drift, any correct agent gets +1.0."""
        rt = _FakeRuntime(
            submitted=True,
            submitted_result_hash="GT_POST",
            submitted_runtime_ms=5000.0,  # doesn't matter — speedup is +∞
            baseline_runtime_ms=100.0,
            gt_result_hash_postdrift="GT_POST",
            baseline_postdrift_raises=True,
            drift_fired_step=7,
        )
        r = _rubric(rt)
        assert r.correctness(_submit_action(), _submit_obs()) == 1.0


# =============================================================================
# Rev-3 invariant — no-op rewrite never grants full credit
# =============================================================================


class TestNoOpRewriteInvariant:
    @pytest.mark.parametrize(
        "agent_sql",
        [
            "SELECT 1",  # identical
            "   SELECT    1   ",  # whitespace-only difference
            "select 1",  # case difference — sqlglot normalizes
        ],
    )
    def test_no_op_rewrite_caps_at_half_credit_and_gatekeeper_fires(self, agent_sql: str) -> None:
        baseline = canonicalize_sql("SELECT 1")
        rt = _FakeRuntime(
            submitted=True,
            submitted_sql=agent_sql,
            submitted_sql_canonical=canonicalize_sql(agent_sql),
            submitted_result_hash="GT_PRE",
            submitted_runtime_ms=100.0,  # equal — 1.0x speedup
            baseline_runtime_ms=100.0,
            baseline_sql_canonical=baseline,
        )
        r = _rubric(rt)
        r(_submit_action(agent_sql), _submit_obs())
        scores = r.component_scores()
        assert scores["r_correct"] == 0.5  # partial, NOT full
        assert scores["r_gatekeepers"] == GATE_BASELINE_VERBATIM
        # And the total submission reward is capped at 0.3.
        assert scores["r_correct"] + scores["r_gatekeepers"] == pytest.approx(0.3)

    def test_where_reorder_does_not_grant_full_credit(self) -> None:
        baseline = canonicalize_sql("SELECT a FROM t WHERE x > 1 AND y < 2")
        rewritten = "SELECT a FROM t WHERE y < 2 AND x > 1"
        rt = _FakeRuntime(
            submitted=True,
            submitted_sql=rewritten,
            submitted_sql_canonical=canonicalize_sql(rewritten),
            submitted_result_hash="GT_PRE",
            submitted_runtime_ms=100.0,  # essentially same — 1.0x
            baseline_runtime_ms=100.0,
            baseline_sql_canonical=baseline,
        )
        r = _rubric(rt)
        r(_submit_action(rewritten), _submit_obs())
        scores = r.component_scores()
        # May or may not trip the baseline-verbatim gate depending on sqlglot's
        # AST normalization; the invariant we care about is:
        assert scores["r_correct"] != 1.0


# =============================================================================
# Speedup
# =============================================================================


class TestSpeedup:
    def test_zero_when_incorrect(self) -> None:
        rt = _FakeRuntime(
            submitted=True,
            submitted_result_hash="WRONG",
            submitted_runtime_ms=10.0,
            baseline_runtime_ms=100.0,
        )
        r = _rubric(rt)
        assert r.speedup(_submit_action(), _submit_obs()) == 0.0

    def test_zero_when_not_faster(self) -> None:
        rt = _FakeRuntime(
            submitted=True,
            submitted_result_hash="GT_PRE",
            submitted_runtime_ms=200.0,  # slower than baseline
            baseline_runtime_ms=100.0,
        )
        r = _rubric(rt)
        assert r.speedup(_submit_action(), _submit_obs()) == 0.0

    def test_positive_when_correct_and_faster(self) -> None:
        rt = _FakeRuntime(
            submitted=True,
            submitted_result_hash="GT_PRE",
            submitted_runtime_ms=10.0,  # 10x speedup
            baseline_runtime_ms=100.0,
        )
        r = _rubric(rt)
        # 0.3 * tanh(log2(10)/3) ≈ 0.3 * tanh(1.107) ≈ 0.3 * 0.803 ≈ 0.241
        score = r.speedup(_submit_action(), _submit_obs())
        assert 0.15 < score < 0.30
        assert score == pytest.approx(0.3 * math.tanh(math.log2(10.0) / 3.0))

    def test_saturates_below_cap(self) -> None:
        rt = _FakeRuntime(
            submitted=True,
            submitted_result_hash="GT_PRE",
            baseline_postdrift_raises=True,
            drift_fired_step=5,
            gt_result_hash_postdrift="GT_PRE",  # match post-drift
            submitted_runtime_ms=1.0,
            baseline_runtime_ms=1.0,
        )
        r = _rubric(rt)
        # speedup is +∞ → capped at 64; still saturates near 0.3.
        score = r.speedup(_submit_action(), _submit_obs())
        assert 0.25 < score <= 0.3


# =============================================================================
# DriftAdapt
# =============================================================================


class TestDriftAdapt:
    def test_zero_on_static_scenario(self) -> None:
        rt = _FakeRuntime(submitted=True, submitted_result_hash="GT_PRE")
        r = _rubric(rt)
        assert r.drift_adapt(_submit_action(), _submit_obs()) == 0.0

    def test_bonus_when_drift_fired_and_agent_uses_new_identifiers(self) -> None:
        rt = _FakeRuntime(
            submitted=True,
            submitted_sql="SELECT account_id, COUNT(*) FROM orders GROUP BY account_id",
            submitted_result_hash="GT_POST",
            submitted_runtime_ms=10.0,
            gt_result_hash_postdrift="GT_POST",
            drift_fired_step=7,
            instance=_FakeInstance(postdrift_identifiers=frozenset({"account_id"})),
        )
        r = _rubric(rt)
        assert (
            r.drift_adapt(
                _submit_action("SELECT account_id, COUNT(*) FROM orders GROUP BY account_id"),
                _submit_obs(),
            )
            == 0.5
        )

    def test_penalty_when_drift_fired_and_agent_uses_pre_drift_identifiers(self) -> None:
        rt = _FakeRuntime(
            submitted=True,
            submitted_sql="SELECT user_id, COUNT(*) FROM orders GROUP BY user_id",
            submitted_result_hash="WRONG",  # query would fail / return wrong hash
            submitted_runtime_ms=10.0,
            gt_result_hash_postdrift="GT_POST",
            drift_fired_step=7,
            instance=_FakeInstance(
                postdrift_identifiers=frozenset({"account_id"}),
                predrift_identifiers=frozenset({"user_id"}),
            ),
        )
        r = _rubric(rt)
        assert (
            r.drift_adapt(
                _submit_action("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id"),
                _submit_obs(),
            )
            == -0.5
        )

    def test_no_penalty_when_drift_fired_and_agent_sql_lacks_pre_drift_markers(
        self,
    ) -> None:
        """Neither adapted nor obviously pre-drift: neutral DriftAdapt score."""
        rt = _FakeRuntime(
            submitted=True,
            submitted_sql="SELECT foo FROM bar",
            submitted_result_hash="WRONG",
            submitted_runtime_ms=10.0,
            gt_result_hash_postdrift="GT_POST",
            drift_fired_step=7,
            instance=_FakeInstance(
                postdrift_identifiers=frozenset({"account_id"}),
                predrift_identifiers=frozenset({"user_id"}),
            ),
        )
        r = _rubric(rt)
        assert r.drift_adapt(_submit_action("SELECT foo FROM bar"), _submit_obs()) == 0.0

    def test_bonus_for_date_format_drift_with_empty_post_identifiers(self) -> None:
        """Scenario 08 shape — postdrift_identifiers empty, predrift carries
        the ISO anchor literals. A submission without those literals that
        matches the post-drift GT hash earns the +0.5 adapt bonus even
        though no distinctive post identifier is present."""
        rt = _FakeRuntime(
            submitted=True,
            submitted_sql=(
                "SELECT kind, COUNT(*) AS n FROM events "
                "WHERE ts >= 1745193600000 AND ts < 1745280000000 "
                "GROUP BY kind ORDER BY kind"
            ),
            submitted_result_hash="GT_POST",
            submitted_runtime_ms=10.0,
            gt_result_hash_postdrift="GT_POST",
            drift_fired_step=7,
            instance=_FakeInstance(
                postdrift_identifiers=frozenset(),
                predrift_identifiers=frozenset({"2026-04-21T00:00:00Z", "2026-04-22T00:00:00Z"}),
            ),
        )
        r = _rubric(rt)
        assert r.drift_adapt(_submit_action(rt.submitted_sql), _submit_obs()) == 0.5

    def test_penalty_for_date_format_drift_with_baseline_iso_literals(self) -> None:
        """Scenario 08 shape — a baseline-style submission keeps the ISO
        anchor literals; DriftAdapt must penalise it now (previously
        this case slipped through because the shared ``ts`` identifier
        made the old ``not uses_post`` branch unreachable)."""
        rt = _FakeRuntime(
            submitted=True,
            submitted_sql=(
                "SELECT kind, COUNT(*) AS n FROM events "
                "WHERE ts >= '2026-04-21T00:00:00Z' AND ts < '2026-04-22T00:00:00Z' "
                "GROUP BY kind ORDER BY kind"
            ),
            submitted_result_hash="WRONG",
            submitted_runtime_ms=10.0,
            gt_result_hash_postdrift="GT_POST",
            drift_fired_step=7,
            instance=_FakeInstance(
                postdrift_identifiers=frozenset(),
                predrift_identifiers=frozenset({"2026-04-21T00:00:00Z", "2026-04-22T00:00:00Z"}),
            ),
        )
        r = _rubric(rt)
        assert r.drift_adapt(_submit_action(rt.submitted_sql), _submit_obs()) == -0.5

    def test_alias_label_of_dropped_column_does_not_count_as_pre_drift(self) -> None:
        """Live-rollout regression (Gemini 3.1 flash, 2026-04-25): a correct
        post-drift rewrite that retains the dropped column name only as an
        output alias (e.g. ``SELECT account_id AS user_id ...``) MUST still
        earn the +0.5 adapt bonus. The rubric previously read the alias LHS
        as a surviving pre-drift marker via a loose regex extractor and
        returned 0.0 — punishing the agent for keeping a backward-compat
        output column name.
        """
        sql = (
            "SELECT account_id AS user_id, COUNT(*) AS n_orders, "
            "ROUND(SUM(amount), 2) AS total "
            "FROM orders GROUP BY account_id ORDER BY account_id"
        )
        rt = _FakeRuntime(
            submitted=True,
            submitted_sql=sql,
            submitted_result_hash="GT_POST",
            submitted_runtime_ms=10.0,
            gt_result_hash_postdrift="GT_POST",
            drift_fired_step=7,
            instance=_FakeInstance(
                postdrift_identifiers=frozenset({"account_id"}),
                predrift_identifiers=frozenset({"user_id"}),
            ),
        )
        r = _rubric(rt)
        assert r.drift_adapt(_submit_action(sql), _submit_obs()) == 0.5


# =============================================================================
# Step tax
# =============================================================================


class TestStepTax:
    def test_fires_on_non_terminal_step(self) -> None:
        rt = _FakeRuntime()
        r = _rubric(rt)
        assert r.step_tax(_run_query_action(), _run_query_obs()) == STEP_TAX

    def test_productive_rebate_offsets_step_tax(self) -> None:
        rt = _FakeRuntime(last_step_productive_rebate=STEP_REBATE_READ_CHANGELOG)
        r = _rubric(rt)
        assert r.step_tax(_run_query_action(), _run_query_obs()) == pytest.approx(
            STEP_TAX + STEP_REBATE_READ_CHANGELOG
        )

    def test_zero_on_terminal_submission(self) -> None:
        rt = _FakeRuntime(
            submitted=True,
            submitted_result_hash="GT_PRE",
            submitted_runtime_ms=10.0,
            baseline_runtime_ms=100.0,
        )
        r = _rubric(rt)
        assert r.step_tax(_submit_action(), _submit_obs()) == 0.0

    def test_accumulates_negative_over_episode(self) -> None:
        rt = _FakeRuntime()
        r = _rubric(rt)
        total = 0.0
        for _ in range(25):
            total += r.step_tax(_run_query_action(), _run_query_obs())
        assert total == pytest.approx(STEP_TAX * 25)
        assert total <= -0.75  # inaction floor (rubric)


# =============================================================================
# Gatekeepers
# =============================================================================


class TestGatekeepers:
    def test_tool_error_penalty(self) -> None:
        rt = _FakeRuntime()
        r = _rubric(rt)
        assert r.gatekeepers(_run_query_action(), _tool_error_obs()) == GATE_MALFORMED_TOOL_CALL

    def test_repeat_failing_query_penalty(self) -> None:
        rt = _FakeRuntime(
            last_step_was_repeat_failing_query=True, last_step_repeat_failing_query_count=2
        )
        r = _rubric(rt)
        assert r.gatekeepers(_run_query_action(), _run_query_obs()) == GATE_REPEAT_FAILING_QUERY

    def test_consecutive_tool_errors_escalate_penalty(self) -> None:
        rt = _FakeRuntime(consecutive_tool_errors=3)
        r = _rubric(rt)
        assert r.gatekeepers(_run_query_action(), _tool_error_obs()) == pytest.approx(
            GATE_MALFORMED_TOOL_CALL + (2 * GATE_CONSECUTIVE_TOOL_ERROR)
        )

    def test_baseline_verbatim_on_submit(self) -> None:
        baseline = canonicalize_sql("SELECT * FROM t")
        rt = _FakeRuntime(
            submitted=True,
            submitted_sql_canonical=baseline,
            baseline_sql_canonical=baseline,
            submitted_result_hash="GT_PRE",
            submitted_runtime_ms=100.0,
        )
        r = _rubric(rt)
        score = r.gatekeepers(_submit_action("SELECT * FROM t"), _submit_obs())
        assert score == GATE_BASELINE_VERBATIM


# =============================================================================
# Consult DBA
# =============================================================================


class TestConsultDBA:
    def test_off_by_default(self) -> None:
        rt = _FakeRuntime(consultations_used=1)
        r = _rubric(rt)
        action = SqlDriftAction(
            tool=ToolName.CONSULT_DBA,
            payload=ConsultDBAPayload(question="help"),
        )
        obs = _run_query_obs()
        assert r.consult_dba(action, obs) == 0.0

    def test_escalates_when_enabled(self) -> None:
        for i, expected in enumerate(CONSULT_ESCALATION, start=1):
            rt = _FakeRuntime(consultations_used=i, dba_oracle_enabled=True)
            r = _rubric(rt)
            action = SqlDriftAction(
                tool=ToolName.CONSULT_DBA,
                payload=ConsultDBAPayload(question="help"),
            )
            obs = _run_query_obs()
            assert r.consult_dba(action, obs) == expected

    def test_non_consult_action_is_zero(self) -> None:
        rt = _FakeRuntime(consultations_used=1, dba_oracle_enabled=True)
        r = _rubric(rt)
        assert r.consult_dba(_run_query_action(), _run_query_obs()) == 0.0


# =============================================================================
# Composite behaviour
# =============================================================================


class TestComposite:
    def test_forward_is_sum_of_components(self) -> None:
        rt = _FakeRuntime(
            submitted=True,
            submitted_sql="SELECT 1",
            submitted_sql_canonical=canonicalize_sql("SELECT 1"),
            submitted_result_hash="GT_PRE",
            submitted_runtime_ms=10.0,
            baseline_runtime_ms=100.0,
        )
        r = _rubric(rt)
        total = r(_submit_action(), _submit_obs())
        scores = r.component_scores()
        assert total == pytest.approx(sum(scores.values()))
        assert set(scores.keys()) == set(REWARD_COMPONENT_KEYS)

    def test_named_rubrics_enumerates_six_children(self) -> None:
        rt = _FakeRuntime()
        r = _rubric(rt)
        names = {name for name, _ in r.named_rubrics()}
        assert names == {
            "correctness",
            "drift_adapt",
            "speedup",
            "step_tax",
            "gatekeepers",
            "consult_dba",
        }

    def test_wrong_submission_on_drift_scenario_nets_below_minus_one(self) -> None:
        """Incorrect pre-drift-only submission after drift nets below -1 total reward."""
        rt = _FakeRuntime(
            submitted=True,
            submitted_sql="SELECT user_id FROM orders",
            submitted_sql_canonical=canonicalize_sql("SELECT user_id FROM orders"),
            submitted_result_hash="WRONG",
            submitted_runtime_ms=100.0,
            baseline_runtime_ms=100.0,
            drift_fired_step=7,
            gt_result_hash_postdrift="GT_POST",
            instance=_FakeInstance(
                postdrift_identifiers=frozenset({"account_id"}),
                predrift_identifiers=frozenset({"user_id"}),
            ),
        )
        r = _rubric(rt)
        total = r(_submit_action("SELECT user_id FROM orders"), _submit_obs())
        assert total < -1.0


# =============================================================================
# Canonicalize helper
# =============================================================================


class TestCanonicalize:
    def test_case_and_whitespace_folded(self) -> None:
        assert canonicalize_sql("SELECT 1") == canonicalize_sql("  select   1 ")

    def test_invalid_sql_falls_back_to_whitespace_fold(self) -> None:
        got = canonicalize_sql("totally bogus not sql $$$")
        assert " " not in got.strip().split(" ")[0]  # non-empty first token
        assert got == canonicalize_sql("totally     bogus not sql $$$")


def test_speedup_min_matches_plan() -> None:
    assert SPEEDUP_MIN == 1.2
