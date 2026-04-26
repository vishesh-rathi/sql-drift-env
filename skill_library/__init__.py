"""Self-improvement library — pre-seeded playbook + on-disk learned store."""

from skill_library.entries import (
    DriftAdaptationCard,
    PlaybookEntry,
    RetrievalResult,
)
from skill_library.retrieval import (
    JACCARD_MIN,
    jaccard,
    retrieve,
    top_k_drift_cards,
    top_k_playbook,
)
from skill_library.seeds import PRESEED_DRIFT_CARDS, PRESEED_PLAYBOOK
from skill_library.store import DEFAULT_STORE_DIR, Store, cleanup_stale_session_dirs


def load_all(
    store: Store | None = None,
) -> tuple[
    tuple[PlaybookEntry, ...],
    tuple[DriftAdaptationCard, ...],
]:
    """Union of pre-seeds and any entries persisted on disk.

    Returns ``(playbook, drift_cards)``. Order: pre-seeds first, then
    learned entries, so deterministic retrieval tie-breaks prefer
    the hand-authored pre-seeds when tags and speedup match exactly.
    """
    learned_pb: tuple[PlaybookEntry, ...] = ()
    learned_dc: tuple[DriftAdaptationCard, ...] = ()
    if store is not None:
        learned_pb = store.read_playbook()
        learned_dc = store.read_drift_cards()
    return PRESEED_PLAYBOOK + learned_pb, PRESEED_DRIFT_CARDS + learned_dc


__all__ = [
    "DEFAULT_STORE_DIR",
    "DriftAdaptationCard",
    "JACCARD_MIN",
    "PRESEED_DRIFT_CARDS",
    "PRESEED_PLAYBOOK",
    "PlaybookEntry",
    "RetrievalResult",
    "Store",
    "cleanup_stale_session_dirs",
    "jaccard",
    "load_all",
    "retrieve",
    "top_k_drift_cards",
    "top_k_playbook",
]
