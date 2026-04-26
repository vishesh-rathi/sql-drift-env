"""End-to-end ``/ws`` session test — proves stateful multi-step episodes.

The stateless ``/step`` HTTP handler spawns a fresh env per request and
tears it down in ``finally`` (see ``test_client_server.py``).
The whole point of ``/ws`` is the opposite: one WebSocket connection
owns one long-lived ``SqlDriftEnvironment`` instance across many step
calls. These tests verify that contract on the in-process app using
Starlette's synchronous ``TestClient`` (which is the supported way to
drive FastAPI WebSockets under pytest).

What we check, in order:
1. ``reset`` returns a step-0 observation and the chosen ``scenario_id``.
2. A sequence of diagnostic steps increments ``step`` monotonically and
   keeps ``phase == "diagnose"`` until the first non-diagnostic submit.
3. A final ``submit_rewrite`` terminates the episode (``done=True``)
   and transitions out of the diagnose phase — proving the same env
   instance saw every step, which is impossible with stateless HTTP.
4. ``close`` ends the session cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import server.app as app_mod
from scenarios import get_spec
from skill_library import Store


def _reset(seed: int, scenario_id: str) -> dict[str, Any]:
    return {"type": "reset", "data": {"seed": seed, "scenario_id": scenario_id}}


def _step(tool: str, payload_kind: str, **payload_extras: Any) -> dict[str, Any]:
    payload = {"kind": payload_kind, **payload_extras}
    return {"type": "step", "data": {"tool": tool, "payload": payload}}


def _recv_observation(ws) -> dict[str, Any]:
    """Receive the next WS frame and unwrap an ``observation`` response.

    OpenEnv's ``/ws`` handler wraps the serialised ``SqlDriftObservation``
    in ``frame["data"] = {"observation": {...}, "reward": ..., "done": ...}``.
    We flatten it here so callers can inspect fields with a single dict
    lookup (``obs["step"]`` etc.), and attach ``reward``/``done`` back
    onto the flat dict for the assertions below. Anything other than an
    observation frame (error, unknown type) is surfaced loudly so a
    protocol regression fails the test with the actual payload rather
    than a confusing ``KeyError`` downstream.
    """
    raw = ws.receive_text()
    frame = json.loads(raw)
    assert frame.get("type") == "observation", f"unexpected frame: {frame}"
    data = frame["data"]
    obs = dict(data["observation"])
    obs["reward"] = data.get("reward")
    obs["done"] = data.get("done", False)
    return obs


def _session_store_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir())


def _learned_entry_count(store_dir: Path) -> int:
    return sum(1 for e in Store(directory=store_dir).read_playbook() if e.source == "learned")


def _postdrift_gt_sql(seed: int, scenario_id: str) -> str:
    inst = get_spec(scenario_id).materialize(seed)
    try:
        assert inst.gt_sql_postdrift is not None
        return inst.gt_sql_postdrift
    finally:
        inst.conn.close()


def _solve_drift_episode(ws, *, seed: int, scenario_id: str = "07_drift_column_rename") -> None:
    gt_sql = _postdrift_gt_sql(seed, scenario_id)
    ws.send_text(json.dumps(_reset(seed=seed, scenario_id=scenario_id)))
    obs = _recv_observation(ws)
    baseline_sql = obs["baseline_sql"]

    ws.send_text(json.dumps(_step("run_query", "run_query", sql=baseline_sql)))
    obs = _recv_observation(ws)

    for _ in range(20):
        if obs["drift_fired"]:
            break
        ws.send_text(json.dumps(_step("list_tables", "list_tables")))
        obs = _recv_observation(ws)
    assert obs["drift_fired"] is True

    ws.send_text(json.dumps(_step("submit_rewrite", "submit_rewrite", sql=gt_sql)))
    obs = _recv_observation(ws)
    assert obs["done"] is True
    assert obs["tool_result"]["kind"] == "submit_rewrite_result"
    assert obs["reward_components"]["r_correct"] == 1.0


class TestWebSocketSession:
    def test_multi_step_episode_terminates_on_submit(self) -> None:
        app = app_mod.create_app()
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps(_reset(seed=7, scenario_id="01_correlated_subquery")))
            obs = _recv_observation(ws)
            assert obs["step"] == 0
            assert obs["phase"] == "diagnose"
            assert obs["baseline_sql"], "reset observation must carry baseline SQL"
            assert obs["schema_synopsis"], "reset observation must carry schema synopsis"
            assert obs["budget_steps_remaining"] > 0

            ws.send_text(json.dumps(_step("list_tables", "list_tables")))
            obs = _recv_observation(ws)
            assert obs["step"] == 1
            # First successful diagnostic transitions DIAGNOSE -> REWRITE.
            assert obs["phase"] == "rewrite"
            assert obs["tool_result"]["kind"] == "list_tables_result"
            assert obs["done"] is False

            ws.send_text(json.dumps(_step("list_tables", "list_tables")))
            obs = _recv_observation(ws)
            assert obs["step"] == 2
            assert obs["phase"] == "rewrite"
            assert obs["done"] is False

            bad_sql = "SELECT 1 AS definitely_wrong"
            ws.send_text(json.dumps(_step("submit_rewrite", "submit_rewrite", sql=bad_sql)))
            obs = _recv_observation(ws)
            assert obs["done"] is True
            assert obs["phase"] == "finalize"
            assert obs["tool_result"]["kind"] == "submit_rewrite_result"
            assert obs["reward"] is not None
            assert obs["reward"] < 0, (
                f"wrong SQL must net negative terminal reward, got {obs['reward']}"
            )

            ws.send_text(json.dumps({"type": "close"}))

    def test_submit_before_diagnose_is_rejected_over_ws(self) -> None:
        """Phase machine is enforced server-side even across WS boundary."""
        app = app_mod.create_app()
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps(_reset(seed=1, scenario_id="01_correlated_subquery")))
            _recv_observation(ws)

            ws.send_text(json.dumps(_step("submit_rewrite", "submit_rewrite", sql="SELECT 1")))
            obs = _recv_observation(ws)
            result = obs["tool_result"]
            assert result["kind"] == "tool_error"
            assert result["code"] == "submit_before_diagnose"
            assert obs["done"] is False

            ws.send_text(json.dumps({"type": "close"}))

    def test_same_ws_connection_reuses_one_store_across_resets(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(app_mod, "_SESSION_STORE_ROOT", tmp_path)
        app = app_mod.create_app()
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            _solve_drift_episode(ws, seed=7)
            dirs = _session_store_dirs(tmp_path)
            assert len(dirs) == 1
            assert _learned_entry_count(dirs[0]) == 1

            _solve_drift_episode(ws, seed=8)
            dirs = _session_store_dirs(tmp_path)
            assert len(dirs) == 1
            assert _learned_entry_count(dirs[0]) == 2

            ws.send_text(json.dumps({"type": "close"}))

    def test_different_ws_connections_get_different_store_dirs(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # With cleanup_on_close=True each session directory is removed when the
        # WebSocket closes.  Verify the isolation property — different sessions
        # use different directories — by capturing the live directory path while
        # each session is still open, rather than inspecting the filesystem after
        # both sessions have closed (at which point dirs may already be gone).
        monkeypatch.setattr(app_mod, "_SESSION_STORE_ROOT", tmp_path)
        app = app_mod.create_app()
        client = TestClient(app)

        dirs_seen: list[list[Path]] = []

        with client.websocket_connect("/ws") as ws:
            _solve_drift_episode(ws, seed=7)
            dirs_seen.append(_session_store_dirs(tmp_path))
            ws.send_text(json.dumps({"type": "close"}))

        with client.websocket_connect("/ws") as ws:
            _solve_drift_episode(ws, seed=8)
            dirs_seen.append(_session_store_dirs(tmp_path))
            ws.send_text(json.dumps({"type": "close"}))

        assert len(dirs_seen[0]) == 1, "first session must use exactly one store dir"
        assert len(dirs_seen[1]) == 1, "second session must use exactly one store dir"
        assert dirs_seen[0][0] != dirs_seen[1][0], (
            "each WS session must receive an independent skill-store directory"
        )
