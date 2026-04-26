"""P9 — unit + integration tests for the DBA Oracle actor."""

from __future__ import annotations

import pytest

from actors import dba_oracle
from models import (
    ConsultDBAPayload,
    ConsultDBAResult,
    SqlDriftAction,
    ToolError,
    ToolErrorCode,
    ToolName,
)
from scenarios import REGISTRY
from server import SqlDriftEnvironment

# =============================================================================
# Feature flag resolution
# =============================================================================


class TestFeatureFlag:
    def test_default_is_off(self, monkeypatch) -> None:
        monkeypatch.delenv("SQL_DRIFT_ENABLE_DBA_ORACLE", raising=False)
        assert dba_oracle.is_enabled() is False

    def test_explicit_true_overrides_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SQL_DRIFT_ENABLE_DBA_ORACLE", "0")
        assert dba_oracle.is_enabled(True) is True

    def test_explicit_false_overrides_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SQL_DRIFT_ENABLE_DBA_ORACLE", "1")
        assert dba_oracle.is_enabled(False) is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_env_var_truthy_values(self, monkeypatch, val: str) -> None:
        monkeypatch.setenv("SQL_DRIFT_ENABLE_DBA_ORACLE", val)
        assert dba_oracle.is_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "", "no", "nope"])
    def test_env_var_falsy_values(self, monkeypatch, val: str) -> None:
        monkeypatch.setenv("SQL_DRIFT_ENABLE_DBA_ORACLE", val)
        assert dba_oracle.is_enabled() is False


# =============================================================================
# Hint table
# =============================================================================


class TestHintTable:
    def test_every_registered_scenario_has_hints(self) -> None:
        missing = set(REGISTRY) - dba_oracle.known_scenarios()
        assert not missing, f"scenarios without DBA hints: {missing}"

    def test_three_tiers_per_scenario(self) -> None:
        for sid in dba_oracle.known_scenarios():
            t1 = dba_oracle.get_hint(sid, 1)
            t2 = dba_oracle.get_hint(sid, 2)
            t3 = dba_oracle.get_hint(sid, 3)
            # Distinct, non-empty tiers.
            assert len({t1, t2, t3}) == 3
            assert all(len(t) >= 20 for t in (t1, t2, t3))

    def test_tiers_are_structured_as_expert_guidance(self) -> None:
        for sid in dba_oracle.known_scenarios():
            tiers = [dba_oracle.get_hint(sid, tier) for tier in (1, 2, 3)]
            assert tiers[0].startswith("[DBA tier 1]")
            assert tiers[1].startswith("[DBA tier 2]")
            assert tiers[2].startswith("[DBA tier 3]")
            assert "SELECT" in tiers[2], f"tier 3 should provide a SQL skeleton for {sid}"

    def test_tier_clamping(self) -> None:
        assert dba_oracle.get_hint("01_correlated_subquery", 0) == dba_oracle.get_hint(
            "01_correlated_subquery", 1
        )
        assert dba_oracle.get_hint("01_correlated_subquery", 99) == dba_oracle.get_hint(
            "01_correlated_subquery", 3
        )

    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(KeyError):
            dba_oracle.get_hint("does_not_exist", 1)


# =============================================================================
# Environment integration
# =============================================================================


class TestEnvWiring:
    def test_oracle_off_returns_tool_error(self) -> None:
        env = SqlDriftEnvironment()
        try:
            env.reset(seed=1, scenario_id="01_correlated_subquery")
            obs = env.step(
                SqlDriftAction(tool=ToolName.CONSULT_DBA, payload=ConsultDBAPayload(question="?"))
            )
            assert isinstance(obs.tool_result, ToolError)
            assert obs.tool_result.code == ToolErrorCode.INVALID_TOOL_ARGUMENT
        finally:
            env.close()

    def test_oracle_on_returns_tier1_hint_first(self) -> None:
        env = SqlDriftEnvironment()
        try:
            env.reset(seed=1, scenario_id="01_correlated_subquery", enable_dba_oracle=True)
            obs = env.step(
                SqlDriftAction(tool=ToolName.CONSULT_DBA, payload=ConsultDBAPayload(question="?"))
            )
            assert isinstance(obs.tool_result, ConsultDBAResult)
            assert obs.tool_result.tier == 1
            assert "subquery" in obs.tool_result.hint.lower()
            assert env.state.consultations_used == 1
        finally:
            env.close()

    def test_consult_tiers_escalate(self) -> None:
        env = SqlDriftEnvironment()
        try:
            env.reset(seed=1, scenario_id="01_correlated_subquery", enable_dba_oracle=True)
            tiers = []
            for _ in range(4):  # 4 consults — 4th should clamp at tier 3
                obs = env.step(
                    SqlDriftAction(
                        tool=ToolName.CONSULT_DBA,
                        payload=ConsultDBAPayload(question="?"),
                    )
                )
                assert isinstance(obs.tool_result, ConsultDBAResult)
                tiers.append(obs.tool_result.tier)
            assert tiers == [1, 2, 3, 3]
        finally:
            env.close()

    def test_consult_penalty_escalates(self) -> None:
        env = SqlDriftEnvironment()
        try:
            env.reset(seed=1, scenario_id="01_correlated_subquery", enable_dba_oracle=True)
            penalties: list[float] = []
            for _ in range(3):
                obs = env.step(
                    SqlDriftAction(
                        tool=ToolName.CONSULT_DBA,
                        payload=ConsultDBAPayload(question="?"),
                    )
                )
                penalties.append(obs.reward_components["r_consult_dba"])
            assert penalties == [-0.1, -0.3, -0.8]
        finally:
            env.close()

    def test_env_var_enables_oracle(self, monkeypatch) -> None:
        monkeypatch.setenv("SQL_DRIFT_ENABLE_DBA_ORACLE", "1")
        env = SqlDriftEnvironment()
        try:
            env.reset(seed=1, scenario_id="01_correlated_subquery")
            obs = env.step(
                SqlDriftAction(
                    tool=ToolName.CONSULT_DBA,
                    payload=ConsultDBAPayload(question="?"),
                )
            )
            assert isinstance(obs.tool_result, ConsultDBAResult)
        finally:
            env.close()
