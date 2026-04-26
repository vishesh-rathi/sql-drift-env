"""Server settings loaded from the repo-local `.env` / process env."""

from __future__ import annotations

import importlib

import server.app as app_mod
import server.settings as settings_mod


class TestServerSettings:
    def test_env_overrides_are_applied(self, monkeypatch) -> None:
        with monkeypatch.context() as m:
            m.setenv("SQL_DRIFT_SERVER_HOST", "127.0.0.1")
            m.setenv("SQL_DRIFT_SERVER_PORT", "9001")
            m.setenv("SQL_DRIFT_MAX_CONCURRENT_ENVS", "7")
            m.setenv("SQL_DRIFT_DEFAULT_STEP_BUDGET", "31")
            m.setenv("SQL_DRIFT_MAX_RESULT_ROWS", "222")
            m.setenv("SQL_DRIFT_QUERY_TIMEOUT_S", "4.5")
            settings = importlib.reload(settings_mod)
            assert settings.SERVER_HOST == "127.0.0.1"
            assert settings.SERVER_PORT == 9001
            assert settings.MAX_CONCURRENT_ENVS == 7
            assert settings.DEFAULT_STEP_BUDGET == 31
            assert settings.MAX_RESULT_ROWS == 222
            assert settings.QUERY_TIMEOUT_S == 4.5
        importlib.reload(settings_mod)

    def test_create_app_uses_configured_concurrency_by_default(self, monkeypatch) -> None:
        with monkeypatch.context() as m:
            m.setenv("SQL_DRIFT_MAX_CONCURRENT_ENVS", "9")
            settings = importlib.reload(settings_mod)
            app = importlib.reload(app_mod)
            m.setattr(app, "_openenv_create_app", lambda **kwargs: kwargs)
            payload = app.create_app()
            assert payload["max_concurrent_envs"] == settings.MAX_CONCURRENT_ENVS
        importlib.reload(settings_mod)
        importlib.reload(app_mod)
