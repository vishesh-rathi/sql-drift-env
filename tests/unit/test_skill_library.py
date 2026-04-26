"""P8 — unit tests for the self-improvement library."""

from __future__ import annotations

from pathlib import Path

import pytest

import scenarios
from skill_library import (
    PRESEED_DRIFT_CARDS,
    PRESEED_PLAYBOOK,
    DriftAdaptationCard,
    PlaybookEntry,
    Store,
    jaccard,
    load_all,
    retrieve,
    top_k_drift_cards,
    top_k_playbook,
)

# --- pre-seeds -------------------------------------------------------------


class TestPreseeds:
    def test_eight_playbook_and_four_drift_cards(self) -> None:
        assert len(PRESEED_PLAYBOOK) == 8
        assert len(PRESEED_DRIFT_CARDS) == 4

    def test_every_drift_kind_has_a_card(self) -> None:
        kinds = {c.drift_kind for c in PRESEED_DRIFT_CARDS}
        assert kinds == {
            "column_rename",
            "date_format",
            "enum_rule",
            "field_deprecation",
        }

    def test_playbook_covers_each_static_scenario_tag(self) -> None:
        # At least one pre-seed must Jaccard-match each shipped static
        # scenario's live tag set above the retrieval threshold.
        static_tag_sets = [
            spec.tags for spec in scenarios.iter_specs() if spec.drift_config is None
        ]
        for tags in static_tag_sets:
            hits = top_k_playbook(tags, PRESEED_PLAYBOOK, k=3)
            assert len(hits) >= 1, f"no preseed hit for tags {tags}"


# --- Jaccard + retrieval ---------------------------------------------------


class TestRetrieval:
    def test_jaccard_identity(self) -> None:
        a = frozenset({"x", "y"})
        assert jaccard(a, a) == 1.0

    def test_jaccard_disjoint(self) -> None:
        assert jaccard(frozenset({"x"}), frozenset({"y"})) == 0.0

    def test_jaccard_partial(self) -> None:
        got = jaccard(frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"}))
        assert got == pytest.approx(2 / 4)

    def test_top_k_filters_below_threshold(self) -> None:
        # Tags that share nothing with any preseed → empty.
        got = top_k_playbook(frozenset({"totally_novel_tag"}), PRESEED_PLAYBOOK, k=3)
        assert got == ()

    def test_top_k_respects_k(self) -> None:
        got = top_k_playbook(
            frozenset({"ecommerce", "over_projection", "select_star", "join"}),
            PRESEED_PLAYBOOK,
            k=2,
        )
        assert len(got) <= 2

    def test_top_k_deterministic_tiebreak(self) -> None:
        query = frozenset({"correlated_subquery", "projection_subquery", "ecommerce"})
        a = top_k_playbook(query, PRESEED_PLAYBOOK, k=3)
        b = top_k_playbook(query, PRESEED_PLAYBOOK, k=3)
        assert a == b

    def test_top_k_drift_cards_exact_match(self) -> None:
        got = top_k_drift_cards("column_rename", PRESEED_DRIFT_CARDS, k=1)
        assert len(got) == 1
        assert got[0].drift_kind == "column_rename"

    def test_top_k_drift_cards_unknown_kind(self) -> None:
        assert top_k_drift_cards(None, PRESEED_DRIFT_CARDS, k=1) == ()
        assert top_k_drift_cards("bogus", PRESEED_DRIFT_CARDS, k=1) == ()

    def test_retrieve_combines_both(self) -> None:
        result = retrieve(
            query_tags=frozenset({"correlated_subquery", "projection_subquery", "ecommerce"}),
            drift_kind="column_rename",
            playbook=PRESEED_PLAYBOOK,
            drift_cards=PRESEED_DRIFT_CARDS,
        )
        assert len(result.playbook) >= 1
        assert len(result.drift_cards) == 1
        rendered = result.render()
        assert len(rendered) <= 800
        assert rendered.count("\n") == (len(result.playbook) + len(result.drift_cards) - 1)


# --- Store -----------------------------------------------------------------


class TestStore:
    def test_empty_store_returns_nothing(self, tmp_path: Path) -> None:
        store = Store(directory=tmp_path)
        assert store.read_playbook() == ()
        assert store.read_drift_cards() == ()

    def test_append_and_read_playbook_roundtrip(self, tmp_path: Path) -> None:
        store = Store(directory=tmp_path)
        entry = PlaybookEntry(
            tag_set=frozenset({"foo", "bar"}),
            before_snippet="SELECT *",
            after_snippet="SELECT id",
            avg_speedup=1.5,
            scenario_family="ecommerce",
            source="learned",
        )
        store.append_playbook(entry)
        got = store.read_playbook()
        assert len(got) == 1
        assert got[0].tag_set == frozenset({"foo", "bar"})
        assert got[0].source == "learned"

    def test_append_drift_card(self, tmp_path: Path) -> None:
        store = Store(directory=tmp_path)
        card = DriftAdaptationCard(
            drift_kind="column_rename",
            symptom_regex="test",
            recovery_template="rewrite",
            success_rate=0.5,
            source="learned",
        )
        store.append_drift_card(card)
        got = store.read_drift_cards()
        assert len(got) == 1
        assert got[0].drift_kind == "column_rename"

    def test_cache_invalidated_on_mtime_change(self, tmp_path: Path) -> None:
        store = Store(directory=tmp_path)
        entry1 = PlaybookEntry(
            tag_set=frozenset({"a"}),
            before_snippet="x",
            after_snippet="y",
            avg_speedup=1.0,
            scenario_family="ecommerce",
        )
        store.append_playbook(entry1)
        first = store.read_playbook()
        entry2 = PlaybookEntry(
            tag_set=frozenset({"b"}),
            before_snippet="x",
            after_snippet="y",
            avg_speedup=1.0,
            scenario_family="ecommerce",
        )
        store.append_playbook(entry2)
        second = store.read_playbook()
        assert len(second) == len(first) + 1

    def test_corrupt_file_returns_empty(self, tmp_path: Path) -> None:
        store = Store(directory=tmp_path)
        store.playbook_path().parent.mkdir(parents=True, exist_ok=True)
        store.playbook_path().write_text("{{{this is not valid json")
        assert store.read_playbook() == ()

    def test_load_all_unions_preseeds_and_store(self, tmp_path: Path) -> None:
        store = Store(directory=tmp_path)
        entry = PlaybookEntry(
            tag_set=frozenset({"novel"}),
            before_snippet="x",
            after_snippet="y",
            avg_speedup=1.0,
            scenario_family="ecommerce",
            source="learned",
        )
        store.append_playbook(entry)
        pb, dc = load_all(store)
        assert len(pb) == len(PRESEED_PLAYBOOK) + 1
        assert len(dc) == len(PRESEED_DRIFT_CARDS)
        assert pb[-1].source == "learned"

    def test_load_all_without_store_returns_preseeds_only(self) -> None:
        pb, dc = load_all(None)
        assert pb == PRESEED_PLAYBOOK
        assert dc == PRESEED_DRIFT_CARDS
