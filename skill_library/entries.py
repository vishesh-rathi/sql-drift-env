"""Data classes for the self-improvement (skill) library.

Two entry kinds — both immutable dataclasses so they can live on
frozen-dict caches and be safely shared across episodes.

- :class:`PlaybookEntry` — a "before/after" SQL rewrite nugget,
  tagged by anti-pattern + scenario family, with an empirical
  speedup number. Populated by pre-seeds and extended at
  terminal-success (``r_correct > 0 ∧ speedup > 1.2``).
- :class:`DriftAdaptationCard` — a drift-kind recovery card with a
  symptom regex and a recovery template. Pre-seeded 1-per-drift-kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DriftKind = Literal["column_rename", "date_format", "enum_rule", "field_deprecation"]


@dataclass(frozen=True)
class PlaybookEntry:
    """A reusable SQL-rewrite recipe keyed by anti-pattern tags."""

    tag_set: frozenset[str]
    before_snippet: str
    after_snippet: str
    avg_speedup: float
    scenario_family: str  # matches scenarios.base.Family
    # Optional provenance — "preseed" for hand-authored, "learned" for
    # append-on-success entries. Used by the retrieval UI only.
    source: Literal["preseed", "learned"] = "preseed"

    def render_hint(self, max_chars: int = 200) -> str:
        """Render a one-liner suitable for inclusion in ``learned_hints``."""
        body = (
            f"[{self.scenario_family}] "
            f"replace `{self.before_snippet[:60]}...` with "
            f"`{self.after_snippet[:60]}...` "
            f"(~{self.avg_speedup:.1f}x)"
        )
        return body[:max_chars]


@dataclass(frozen=True)
class DriftAdaptationCard:
    """A drift-kind recovery card."""

    drift_kind: DriftKind
    symptom_regex: str
    recovery_template: str
    success_rate: float = 0.0
    source: Literal["preseed", "learned"] = "preseed"

    def render_hint(self, max_chars: int = 200) -> str:
        body = (
            f"[drift:{self.drift_kind}] "
            f"symptom=/{self.symptom_regex}/ → "
            f"{self.recovery_template[:120]}"
        )
        return body[:max_chars]


@dataclass(frozen=True)
class RetrievalResult:
    """Top-k blend of playbook hits + drift cards for one retrieval call."""

    playbook: tuple[PlaybookEntry, ...] = field(default_factory=tuple)
    drift_cards: tuple[DriftAdaptationCard, ...] = field(default_factory=tuple)

    def render(self, max_chars: int = 800) -> str:
        """Concatenate rendered hints, truncated to ``max_chars``.

        Deterministic ordering: playbook entries first (by descending
        ``avg_speedup``, ties broken by ``before_snippet``), then drift
        cards (by descending ``success_rate``, ties by ``drift_kind``).
        """
        lines: list[str] = []
        for e in self.playbook:
            lines.append("- " + e.render_hint(max_chars=200))
        for c in self.drift_cards:
            lines.append("- " + c.render_hint(max_chars=200))
        out = "\n".join(lines)
        return out[:max_chars]


__all__ = [
    "DriftAdaptationCard",
    "DriftKind",
    "PlaybookEntry",
    "RetrievalResult",
]
