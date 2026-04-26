"""P12 — :mod:`training.prompt` unit tests."""

from __future__ import annotations

from models import EpisodePhase, ToolName
from training.prompt import (
    PHASE_NUDGES,
    TOOL_DOCS,
    render_prompt_from_observation,
    render_system_prompt,
)


def test_tool_catalog_covers_every_tool() -> None:
    assert set(TOOL_DOCS.keys()) == set(ToolName)
    for tool, doc in TOOL_DOCS.items():
        assert "signature" in doc and "purpose" in doc
        assert tool.value in doc["signature"]


def test_phase_nudges_cover_every_phase() -> None:
    assert set(PHASE_NUDGES.keys()) == set(EpisodePhase)


def test_render_lists_every_tool_signature_when_dba_enabled() -> None:
    """All 8 tools advertised when the DBA oracle is enabled for the episode."""
    txt = render_system_prompt(scenario_id="01_correlated_subquery", dba_enabled=True)
    for tool in ToolName:
        assert tool.value in txt, f"missing {tool.value} in prompt"


def test_render_excludes_consult_dba_when_dba_disabled() -> None:
    """Curriculum-locked invariant: with the DBA oracle off, consult_dba must
    NOT appear in the rendered tool catalog. Otherwise the agent will call
    it during training, get a ToolError, and absorb a -0.3 gatekeeper
    penalty per the live Gemini rollout (2026-04-25, scenario 07 step 7).
    """
    txt = render_system_prompt(scenario_id="01_correlated_subquery", dba_enabled=False)
    assert "consult_dba" not in txt
    # All other tools still listed.
    for tool in ToolName:
        if tool == ToolName.CONSULT_DBA:
            continue
        assert tool.value in txt, f"non-DBA tool {tool.value} missing"


def test_render_default_excludes_consult_dba() -> None:
    """Default dba_enabled=False matches the curriculum's locked default."""
    txt = render_system_prompt(scenario_id="01_correlated_subquery")
    assert "consult_dba" not in txt


def test_render_includes_scenario_id_and_phase() -> None:
    txt = render_system_prompt(
        scenario_id="07_drift_column_rename",
        phase=EpisodePhase.DRIFT_RECOVERY,
        drift_fired=True,
    )
    assert "07_drift_column_rename" in txt
    assert "Drift has fired" in txt or "Drift has already fired" in txt


def test_render_includes_learned_hints_when_present() -> None:
    hints = "- [ecommerce] rewrite correlated subqueries as GROUP BY joins"
    txt = render_system_prompt(scenario_id="01_correlated_subquery", learned_hints=hints)
    assert hints in txt


def test_render_from_observation_passes_through() -> None:
    from models import SqlDriftObservation

    obs = SqlDriftObservation(
        step=3,
        phase=EpisodePhase.REWRITE,
        learned_hints="- hint A",
        budget_steps_remaining=20,
        drift_fired=False,
    )
    txt = render_prompt_from_observation(scenario_id="03_cartesian_join", observation=obs)
    assert "03_cartesian_join" in txt
    assert "- hint A" in txt
    assert "Remaining step budget: 20" in txt
    assert "REWRITE" in txt
