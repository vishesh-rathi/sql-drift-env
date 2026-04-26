"""P10 — HTTP round-trip tests via ``httpx.ASGITransport``.

These cover the stateless HTTP shape only (``/health``, ``/schema``,
``/metadata``, ``/reset``, single ``/step``). The OpenEnv HTTP handlers
spawn a fresh env per request and tear it down in ``finally`` (stateless HTTP),
so we can't drive a multi-step episode through plain HTTP; that path
belongs to the WebSocket session or direct in-process calls (see
``test_env_smoke.py``).

``ASGITransport`` is async-only — all tests below use the async httpx
client (pytest-asyncio auto-mode picks them up).
"""

from __future__ import annotations

import httpx
import pytest_asyncio

from server.app import create_app


@pytest_asyncio.fixture()
async def client():
    app = create_app()
    # raise_app_exceptions=False so that server-side RuntimeError (e.g.
    # calling /step before /reset on the stateless path) becomes HTTP
    # 500 instead of propagating into the test and masquerading as a
    # client bug.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


class TestStatelessRoutes:
    async def test_health(self, client) -> None:
        r = await client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, dict)

    async def test_schema_reports_action_and_observation(self, client) -> None:
        r = await client.get("/schema")
        assert r.status_code == 200
        body = r.json()
        assert "action" in body
        assert "observation" in body
        action_props = body["action"].get("properties", {})
        assert "tool" in action_props
        assert "payload" in action_props

    async def test_metadata_env_name(self, client) -> None:
        r = await client.get("/metadata")
        assert r.status_code == 200
        body = r.json()
        assert body.get("name") == "SqlDriftEnvironment"

    async def test_reset_returns_step0_observation(self, client) -> None:
        r = await client.post("/reset", json={"seed": 42, "scenario_id": "01_correlated_subquery"})
        assert r.status_code == 200, r.text
        body = r.json()
        obs = body["observation"]
        assert obs["step"] == 0
        assert obs["phase"] == "diagnose"
        assert "learned_hints" in obs
        assert obs["budget_steps_remaining"] == 25

    async def test_stateless_step_without_reset_fails_loudly(self, client) -> None:
        """OpenEnv HTTP spawns a fresh env per request. SQLDrift
        is stateful (reset materialises DuckDB + scenario GT), so a
        bare ``/step`` with no matching ``/reset`` MUST error out
        rather than silently running against an empty world. Agents
        requiring multi-step dialogue MUST use ``/ws``.

        The env raises :class:`RuntimeError` in :meth:`SqlDriftEnvironment.step`
        when no reset has been run, which the ASGI transport maps to
        HTTP 500 (no FastAPI handler catches generic ``RuntimeError``).
        We lock to 5xx — and in particular 500 — so a regression where
        the env accidentally accepts the request (200) or degrades to
        a generic 4xx (masking the contract) is caught immediately.
        """
        r = await client.post(
            "/step",
            json={
                "action": {
                    "tool": "list_tables",
                    "payload": {"kind": "list_tables"},
                }
            },
        )
        assert r.status_code == 500, r.text


class TestActionValidationAtEnvelope:
    """Envelope-level ValidationError maps to HTTP 422, not ToolError."""

    async def test_missing_tool_field_is_422(self, client) -> None:
        r = await client.post("/step", json={"action": {"payload": {"kind": "list_tables"}}})
        assert r.status_code == 422

    async def test_tool_payload_kind_mismatch_is_422(self, client) -> None:
        await client.post("/reset", json={"seed": 1, "scenario_id": "01_correlated_subquery"})
        r = await client.post(
            "/step",
            json={
                "action": {
                    "tool": "run_query",
                    "payload": {"kind": "list_tables"},  # mismatch
                }
            },
        )
        assert r.status_code == 422

    async def test_sql_too_long_is_422(self, client) -> None:
        await client.post("/reset", json={"seed": 1, "scenario_id": "01_correlated_subquery"})
        big = "SELECT " + ("a," * 5000) + "1"
        r = await client.post(
            "/step",
            json={
                "action": {
                    "tool": "run_query",
                    "payload": {"kind": "run_query", "sql": big[:20_000]},
                }
            },
        )
        assert r.status_code == 422
