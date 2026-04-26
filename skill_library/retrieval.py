"""Pure-function Jaccard top-k retrieval over tag sets.

No embeddings, no torch, no tokenizer. Deterministic — same inputs
produce the same ranking, same top-k, same tie-break.

A conservative Jaccard threshold (0.3) limits retrieval noise when
broad pre-seeds would otherwise match every scenario.
"""

from __future__ import annotations

from collections.abc import Iterable

from skill_library.entries import (
    DriftAdaptationCard,
    PlaybookEntry,
    RetrievalResult,
)

JACCARD_MIN: float = 0.3


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Standard Jaccard on sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def top_k_playbook(
    query_tags: frozenset[str],
    entries: Iterable[PlaybookEntry],
    k: int = 3,
    *,
    min_overlap: float = JACCARD_MIN,
) -> tuple[PlaybookEntry, ...]:
    """Top-k playbook entries by Jaccard overlap with ``query_tags``.

    Ties broken by descending ``avg_speedup`` then by ``before_snippet``
    lexicographic order so the result is stable across runs.
    """
    scored = [(jaccard(query_tags, e.tag_set), e) for e in entries]
    scored = [(j, e) for j, e in scored if j >= min_overlap]
    scored.sort(key=lambda t: (-t[0], -t[1].avg_speedup, t[1].before_snippet))
    return tuple(e for _, e in scored[:k])


def top_k_drift_cards(
    drift_kind: str | None,
    cards: Iterable[DriftAdaptationCard],
    k: int = 1,
) -> tuple[DriftAdaptationCard, ...]:
    """Filter cards by exact drift_kind match, sorted by success_rate desc."""
    if drift_kind is None:
        return ()
    matches = [c for c in cards if c.drift_kind == drift_kind]
    matches.sort(key=lambda c: (-c.success_rate, c.drift_kind))
    return tuple(matches[:k])


def retrieve(
    query_tags: frozenset[str],
    drift_kind: str | None,
    playbook: Iterable[PlaybookEntry],
    drift_cards: Iterable[DriftAdaptationCard],
    *,
    playbook_k: int = 3,
    drift_k: int = 1,
) -> RetrievalResult:
    """Combined retrieval: top-k playbook + top-k drift cards."""
    return RetrievalResult(
        playbook=top_k_playbook(query_tags, playbook, k=playbook_k),
        drift_cards=top_k_drift_cards(drift_kind, drift_cards, k=drift_k),
    )


__all__ = [
    "JACCARD_MIN",
    "jaccard",
    "retrieve",
    "top_k_drift_cards",
    "top_k_playbook",
]
