"""P2 — registry auto-discovery + placeholder materialize round-trip."""

from __future__ import annotations

import pytest

import scenarios
from scenarios.base import BASELINE_MIN_MS


class TestRegistry:
    def test_registry_populated(self) -> None:
        assert len(scenarios.REGISTRY) >= 1

    def test_get_spec_known_and_unknown(self) -> None:
        spec = scenarios.get_spec("01_correlated_subquery")
        assert spec.scenario_id == "01_correlated_subquery"
        with pytest.raises(KeyError, match="unknown scenario_id"):
            scenarios.get_spec("does_not_exist")

    def test_iter_specs_sorted_by_id(self) -> None:
        ids = [s.scenario_id for s in scenarios.iter_specs()]
        assert ids == sorted(ids)

    def test_no_duplicate_scenario_ids(self) -> None:
        ids = [s.scenario_id for s in scenarios.iter_specs()]
        assert len(ids) == len(set(ids))


class TestFixtures:
    def test_seeded_rng_is_deterministic(self) -> None:
        from scenarios._fixtures import seeded_rng

        a = seeded_rng("s", 1).random()
        b = seeded_rng("s", 1).random()
        assert a == b
        assert seeded_rng("s", 2).random() != a

    def test_zipfian_choices_shape(self) -> None:
        from scenarios._fixtures import seeded_rng, zipfian_choices

        rng = seeded_rng("test", 0)
        draws = zipfian_choices(rng, list(range(100)), 1_000)
        assert len(draws) == 1_000
        assert min(draws) == 0
        assert max(draws) < 100

    def test_lognormal_amounts_positive(self) -> None:
        from scenarios._fixtures import lognormal_amounts, seeded_rng

        rng = seeded_rng("test", 0)
        xs = lognormal_amounts(rng, 500, mu=3.0, sigma=0.8)
        assert all(x > 0 for x in xs)
        assert len(xs) == 500

    def test_iso_roundtrip_sorted(self) -> None:
        from scenarios._fixtures import (
            date_range_epoch_ms,
            iso_strings_from_epoch_ms,
            seeded_rng,
        )

        rng = seeded_rng("iso", 0)
        eps = date_range_epoch_ms(rng, 10, start_epoch_ms=1_700_000_000_000, window_days=30)
        iso = iso_strings_from_epoch_ms(eps)
        # ISO-8601 strings sort lexicographically iff timestamps sort chronologically.
        assert sorted(iso) == [
            x for _, x in sorted(zip(eps, iso, strict=False), key=lambda t: t[0])
        ]


class TestBaselineFloor:
    def test_materialize_difficulty_controls_builder_scale(self) -> None:
        import duckdb

        from scenarios.base import BuilderResult, ScenarioSpec

        seen_scales: list[int] = []

        def _trivial(spec: ScenarioSpec, seed: int, scale: int) -> BuilderResult:
            del spec, seed
            seen_scales.append(scale)
            conn = duckdb.connect(":memory:")
            conn.execute("CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (1);")
            return (
                conn,
                "SELECT x FROM t",
                "SELECT x FROM t",
                None,
                "trivial",
                frozenset(),
                frozenset(),
            )

        spec = ScenarioSpec(
            scenario_id="difficulty_scale",
            family="ecommerce",
            tags=frozenset(),
            drift_config=None,
            builder=_trivial,
            base_scale=100,
        )

        easy = spec.materialize(seed=0, difficulty="easy")
        normal = spec.materialize(seed=0, difficulty="normal")
        hard = spec.materialize(seed=0, difficulty="hard")
        try:
            assert seen_scales == [50, 100, 200]
            assert easy.gt_result_hash_predrift
            assert normal.gt_result_hash_predrift
            assert hard.gt_result_hash_predrift
        finally:
            easy.conn.close()
            normal.conn.close()
            hard.conn.close()

    def test_soft_floor_warns_but_still_returns(self, caplog) -> None:
        """An under-floor baseline logs a warning and still returns an instance.

        The old design raised :class:`ScenarioMaterializationError` from a
        retry loop; that loop was removed because it coupled fixture RNG
        seeds to timing-driven retry counts, breaking determinism. Now
        ``base_scale`` is author-tuned and a shortfall is a soft signal
        for the author — not a runtime error — so episodes still proceed
        with whatever baseline the fixture produced.
        """
        import duckdb

        from scenarios.base import BuilderResult, ScenarioSpec

        def _trivial(spec: ScenarioSpec, seed: int, scale: int) -> BuilderResult:
            conn = duckdb.connect(":memory:")
            conn.execute("CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (1);")
            return (
                conn,
                "SELECT x FROM t",
                "SELECT x FROM t",
                None,
                "trivial",
                frozenset(),
                frozenset(),
            )

        spec = ScenarioSpec(
            scenario_id="under_floor",
            family="ecommerce",
            tags=frozenset(),
            drift_config=None,
            builder=_trivial,
            base_scale=1,
            baseline_min_ms=10_000.0,
        )
        with caplog.at_level(30, logger="sql_drift_env.app.scenarios.base"):  # logging.WARNING
            inst = spec.materialize(seed=0)
        try:
            assert inst.baseline_runtime_ms < 10_000.0
            assert "10000.00ms floor" in caplog.text
            assert "under_floor" in caplog.text
        finally:
            inst.conn.close()

    def test_module_level_default_floor_matches_doc(self) -> None:
        assert BASELINE_MIN_MS == 0.3


class TestDriftConfigValidation:
    def test_min_max_ordering(self) -> None:
        from scenarios.base import DriftConfig

        with pytest.raises(ValueError, match="max_step"):
            DriftConfig(kind="column_rename", payload={}, min_step=10, max_step=5)

    def test_cooldown_nonnegative(self) -> None:
        from scenarios.base import DriftConfig

        with pytest.raises(ValueError, match="cooldown"):
            DriftConfig(kind="column_rename", payload={}, cooldown_steps=-1)

    def test_min_step_positive(self) -> None:
        from scenarios.base import DriftConfig

        with pytest.raises(ValueError, match="min_step"):
            DriftConfig(kind="column_rename", payload={}, min_step=0)
