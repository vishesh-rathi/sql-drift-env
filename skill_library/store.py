"""JSON-backed playbook and drift-card store with file locking.

Each store file is a JSON array. Appends go through a single locked
read-modify-write cycle that writes to a sibling ``*.tmp`` file and
atomically ``os.replace``s it onto the target path, so a crash can only
leave either the old array or the new one — never a truncated file.

The lock is held on a dedicated ``*.lock`` file via ``fcntl.flock`` with
a caller-configurable timeout (default 5s). We never lock the data file
itself: that way an ``os.replace`` inside the critical section can't
race against a reader holding a shared lock on the old inode.

Reads are cached by mtime so hot-path episodes don't re-parse the file
on every ``reset()``. Corrupt trailers (from a pre-atomic-write era or
a partial disk write) log a warning and fall back to empty — we prefer
a running trainer over one that dies because of a bad card.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from skill_library.entries import DriftAdaptationCard, PlaybookEntry
from utilities.logger import get_module_logger

_LOG = get_module_logger(__name__)

DEFAULT_STORE_DIR = Path("outputs") / "skill_library"
PLAYBOOK_FILENAME = "playbook.json"
DRIFT_CARDS_FILENAME = "drift_cards.json"
DEFAULT_LOCK_TIMEOUT_S: float = 5.0

T = TypeVar("T")


try:
    import fcntl

    def _try_lock_exclusive(fh: Any) -> bool:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _unlock(fh: Any) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    _HAS_FCNTL = True

except ImportError:
    _HAS_FCNTL = False

    def _try_lock_exclusive(fh: Any) -> bool:
        return True

    def _unlock(fh: Any) -> None:
        return None


@contextlib.contextmanager
def _locked(path: Path, timeout_s: float) -> Iterator[None]:
    """Poll-acquire an exclusive flock on ``path`` within ``timeout_s``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    with path.open("a+") as fh:
        while not _try_lock_exclusive(fh):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not acquire {path} within {timeout_s}s")
            time.sleep(0.02)
        try:
            yield
        finally:
            if _HAS_FCNTL:
                _unlock(fh)


def _atomic_write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2)
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_json_array(path: Path) -> list[Any]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        _LOG.warning("skill-store read failed for %s: %s", path, exc)
        return []
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _LOG.warning("skill-store corrupt at %s (%s); returning empty", path, exc)
        return []
    return data if isinstance(data, list) else []


class Store:
    """Append-only JSON store for learned playbook entries + drift cards."""

    def __init__(
        self,
        directory: Path | None = None,
        lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
    ) -> None:
        self.dir = Path(directory) if directory is not None else DEFAULT_STORE_DIR
        self.lock_timeout_s = lock_timeout_s
        self._playbook_cache: tuple[tuple[PlaybookEntry, ...], float] | None = None
        self._drift_cache: tuple[tuple[DriftAdaptationCard, ...], float] | None = None

    def playbook_path(self) -> Path:
        return self.dir / PLAYBOOK_FILENAME

    def drift_cards_path(self) -> Path:
        return self.dir / DRIFT_CARDS_FILENAME

    def read_playbook(self) -> tuple[PlaybookEntry, ...]:
        return self._read_cached(
            self.playbook_path(),
            cache_attr="_playbook_cache",
            decode=_entry_from_dict,
        )

    def read_drift_cards(self) -> tuple[DriftAdaptationCard, ...]:
        return self._read_cached(
            self.drift_cards_path(),
            cache_attr="_drift_cache",
            decode=lambda d: DriftAdaptationCard(**d),
        )

    def append_playbook(self, entry: PlaybookEntry) -> None:
        self._locked_append(
            self.playbook_path(),
            encode_new=_entry_to_dict,
            new_item=entry,
        )
        self._playbook_cache = None

    def append_drift_card(self, card: DriftAdaptationCard) -> None:
        self._locked_append(
            self.drift_cards_path(),
            encode_new=asdict,
            new_item=card,
        )
        self._drift_cache = None

    def _read_cached(
        self,
        path: Path,
        *,
        cache_attr: str,
        decode: Callable[[dict[str, Any]], T],
    ) -> tuple[T, ...]:
        mtime = _safe_mtime(path)
        # ``getattr``/``setattr`` is intentional — the same implementation
        # services both the playbook and drift-card caches, whose Python
        # types differ. The cast below restores the precise
        # ``(tuple[T, ...], float) | None`` shape for mypy.
        cached = cast("tuple[tuple[T, ...], float] | None", getattr(self, cache_attr))
        if cached is not None and cached[1] == mtime:
            return cached[0]
        items: list[T] = []
        for d in _read_json_array(path):
            try:
                items.append(decode(d))
            except (TypeError, KeyError, ValueError) as exc:
                _LOG.warning("skipping malformed store entry %s: %s", d, exc)
        tup = tuple(items)
        setattr(self, cache_attr, (tup, mtime))
        return tup

    def _locked_append(
        self,
        path: Path,
        *,
        encode_new: Callable[[Any], dict[str, Any]],
        new_item: Any,
    ) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with _locked(lock_path, self.lock_timeout_s):
            existing = _read_json_array(path)
            existing.append(encode_new(new_item))
            _atomic_write_json(path, existing)


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _entry_to_dict(e: PlaybookEntry) -> dict[str, Any]:
    return {
        "tag_set": sorted(e.tag_set),
        "before_snippet": e.before_snippet,
        "after_snippet": e.after_snippet,
        "avg_speedup": e.avg_speedup,
        "scenario_family": e.scenario_family,
        "source": e.source,
    }


def _entry_from_dict(d: dict[str, Any]) -> PlaybookEntry:
    source: Literal["preseed", "learned"] = d.get("source", "learned")
    return PlaybookEntry(
        tag_set=frozenset(d.get("tag_set") or []),
        before_snippet=d["before_snippet"],
        after_snippet=d["after_snippet"],
        avg_speedup=float(d["avg_speedup"]),
        scenario_family=d["scenario_family"],
        source=source,
    )


def cleanup_stale_session_dirs(root: Path, ttl_hours: float) -> int:
    """Remove session subdirectories under *root* whose mtime is older than *ttl_hours*.

    Returns the number of directories removed.  Errors on individual
    subdirectories are logged and skipped so a single bad entry cannot abort
    the sweep.  Pass ``ttl_hours=0`` to disable (returns 0 immediately).
    """
    import shutil

    if ttl_hours <= 0 or not root.exists():
        return 0
    cutoff = time.time() - ttl_hours * 3600
    removed = 0
    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        try:
            if session_dir.stat().st_mtime < cutoff:
                shutil.rmtree(session_dir, ignore_errors=True)
                removed += 1
        except OSError as exc:
            _LOG.warning("cleanup_stale_session_dirs: skipping %s: %s", session_dir, exc)
    return removed


__all__ = ["DEFAULT_LOCK_TIMEOUT_S", "DEFAULT_STORE_DIR", "Store", "cleanup_stale_session_dirs"]
