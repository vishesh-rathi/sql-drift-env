"""Single logging entrypoint: JSONL to ``outputs/logs/sql_drift_env.log``.

The stdlib :mod:`logging` must not be used elsewhere in this package;
use :func:`get_module_logger` for diagnostic messages
(:meth:`~logging.Logger.info` / :meth:`~logging.Logger.warning` / …) and
:func:`log_interaction` for structured agent/env events.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = _REPO_ROOT / "outputs" / "logs"
LOG_FILE = LOG_DIR / "sql_drift_env.log"
INTERACTION_LOGGER_NAME = "sql_drift_env.interactions"
APP_LOGGER_ROOT = "sql_drift_env.app"
IST = ZoneInfo("Asia/Kolkata")
TIMESTAMP_FORMAT = "%d-%b-%Y %I:%M:%S %p"

# Rotating log limits — override via environment variables.
_LOG_MAX_BYTES: int = int(os.environ.get("SQL_DRIFT_LOG_MAX_BYTES", str(50 * 1024 * 1024)))
_LOG_BACKUP_COUNT: int = int(os.environ.get("SQL_DRIFT_LOG_BACKUP_COUNT", "5"))

# When SQL_DRIFT_LOG_PROMPTS is not set (default), llm_prompt and llm_response
# are suppressed from the interaction log to prevent unbounded disk growth and
# accidental prompt/credential leakage in production.
_LOG_PROMPTS: bool = os.environ.get("SQL_DRIFT_LOG_PROMPTS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

_app_logging_configured: bool = False


def get_interaction_jsonl_logger() -> logging.Logger:
    """Return the rotating-file logger used to append JSONL lines, handler registered once."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(INTERACTION_LOGGER_NAME)
    lg.setLevel(logging.INFO)
    lg.propagate = False

    log_path = str(LOG_FILE)
    for handler in lg.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path:
            return lg

    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(handler)
    return lg


interaction_jsonl_logger = get_interaction_jsonl_logger()


def log_interaction(
    *,
    event_type: str,
    agent_id: str | None = None,
    action_taken: Any = None,
    observation_returned: Any = None,
    reward: Any = None,
    done: Any = None,
    llm_prompt: Any = None,
    llm_response: Any = None,
    error: str | None = None,
) -> None:
    """Write one structured interaction event to the central log file.

    LLM prompts and responses are omitted unless the environment variable
    ``SQL_DRIFT_LOG_PROMPTS=1`` is set, preventing unbounded disk growth and
    accidental leakage of prompt content in production deployments.
    """
    entry = {
        "timestamp": datetime.now(IST).strftime(TIMESTAMP_FORMAT),
        "event_type": event_type,
        "agent_id": agent_id,
        "action_taken": _to_jsonable(action_taken),
        "observation_returned": _to_jsonable(observation_returned),
        "reward": _to_jsonable(reward),
        "done": _to_jsonable(done),
        "llm_prompt": _to_jsonable(llm_prompt) if _LOG_PROMPTS else None,
        "llm_response": _to_jsonable(llm_response) if _LOG_PROMPTS else None,
    }
    if error is not None:
        entry["error"] = error
    interaction_jsonl_logger.info(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))


class _AppJsonlLogHandler(logging.Handler):
    """Routes :class:`logging.LogRecord` instances into :func:`log_interaction`."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_interaction(
                event_type="app",
                action_taken={
                    "level": record.levelname,
                    "source": record.name,
                    "message": record.getMessage(),
                },
            )
        except Exception:
            self.handleError(record)


def _ensure_app_logging() -> None:
    global _app_logging_configured
    if _app_logging_configured:
        return
    _app_logging_configured = True
    parent = logging.getLogger(APP_LOGGER_ROOT)
    parent.setLevel(logging.DEBUG)
    # Allow records to continue to ancestors (e.g. root, for pytest caplog).
    parent.propagate = True
    if not any(type(h) is _AppJsonlLogHandler for h in parent.handlers):
        parent.addHandler(_AppJsonlLogHandler(level=logging.DEBUG))


def get_module_logger(qualname: str) -> logging.Logger:
    """Diagnostic logger for a module; records go to the central JSONL file."""
    _ensure_app_logging()
    return logging.getLogger(f"{APP_LOGGER_ROOT}.{qualname}")


def get_logger() -> logging.Logger:
    """Same as :func:`get_interaction_jsonl_logger`."""
    return get_interaction_jsonl_logger()


logger = interaction_jsonl_logger


def log_env_reset(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for OpenEnv ``reset()`` methods."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        action = {
            "seed": args[0] if args else kwargs.get("seed"),
            "episode_id": args[1] if len(args) > 1 else kwargs.get("episode_id"),
            **kwargs,
        }
        try:
            observation = func(self, *args, **kwargs)
        except Exception as exc:
            episode_id = action.get("episode_id")
            log_interaction(
                event_type="reset",
                agent_id=str(episode_id) if episode_id is not None else None,
                action_taken=action,
                error=repr(exc),
            )
            raise
        log_interaction(
            event_type="reset",
            agent_id=_agent_id_from_env(self),
            action_taken=action,
            observation_returned=observation,
            reward=getattr(observation, "reward", None),
            done=getattr(observation, "done", None),
        )
        return observation

    return wrapper


def log_env_step(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for OpenEnv ``step()`` methods."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        action = args[0] if args else kwargs.get("action")
        try:
            observation = func(self, *args, **kwargs)
        except Exception as exc:
            log_interaction(
                event_type="step",
                agent_id=_agent_id_from_env(self),
                action_taken=action,
                error=repr(exc),
            )
            raise
        log_interaction(
            event_type="step",
            agent_id=_agent_id_from_env(self),
            action_taken=action,
            observation_returned=observation,
            reward=getattr(observation, "reward", None),
            done=getattr(observation, "done", None),
        )
        return observation

    return wrapper


def _agent_id_from_env(env: Any) -> str | None:
    runtime = getattr(env, "_runtime", None)
    episode_id = getattr(runtime, "episode_id", None)
    return str(episode_id) if episode_id is not None else None


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_jsonable(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(value)


__all__ = [
    "APP_LOGGER_ROOT",
    "LOG_FILE",
    "get_interaction_jsonl_logger",
    "get_logger",
    "get_module_logger",
    "interaction_jsonl_logger",
    "logger",
    "log_env_reset",
    "log_env_step",
    "log_interaction",
]  # _LOG_PROMPTS / _LOG_MAX_BYTES / _LOG_BACKUP_COUNT are intentionally private
