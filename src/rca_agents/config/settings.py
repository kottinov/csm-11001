from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

DEFAULT_ENV_FILE = Path(os.getenv("RCA_ENV_FILE", ".env"))


def _load_env_file(path: Path) -> None:
    """Populate os.environ with values from a dotenv-style file."""
    if not path.exists() or not path.is_file():
        return

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
    except OSError as exc:
        raise RuntimeError(f"Failed to read environment file at {path}: {exc}") from exc


def _parse_int(name: str, default: int) -> int:
    """Parse an integer environment variable with validation."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw_value}") from exc


def _parse_float(name: str, default: float) -> float:
    """Parse a float environment variable with validation."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got: {raw_value}") from exc


@dataclass(frozen=True)
class AnthropicConfig:
    api_key: str
    log_model: str
    metrics_model: str
    supervisor_model: str


@dataclass(frozen=True)
class RuntimeConfig:
    log_agent_max_depth: int
    metrics_agent_max_depth: int
    log_agent_max_iterations: int
    metrics_agent_max_iterations: int
    candidate_batch_size: int
    search_temperature: float
    exploration_weight: float
    self_consistency_weight: float


@dataclass(frozen=True)
class DatasetConfig:
    logs_root: Path
    metrics_csv: Path
    metrics_chart_dir: Path


@dataclass(frozen=True)
class Settings:
    anthropic: AnthropicConfig
    runtime: RuntimeConfig
    dataset: DatasetConfig


@lru_cache(1)
def get_settings(env_file: Optional[Path] = None) -> Settings:
    """Load application settings, optionally from a provided dotenv file."""
    path = env_file or DEFAULT_ENV_FILE
    _load_env_file(path)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Provide it via the environment or an .env file."
        )

    log_model = os.getenv("RCA_LOG_MODEL", "claude-3-5-sonnet-20240620")
    metrics_model = os.getenv("RCA_METRICS_MODEL", log_model)
    supervisor_model = os.getenv("RCA_SUPERVISOR_MODEL", log_model)

    runtime = RuntimeConfig(
        log_agent_max_depth=_parse_int("RCA_LOG_AGENT_MAX_DEPTH", 5),
        metrics_agent_max_depth=_parse_int("RCA_METRICS_AGENT_MAX_DEPTH", 3),
        log_agent_max_iterations=_parse_int("RCA_LOG_AGENT_MAX_ITERATIONS", 50),
        metrics_agent_max_iterations=_parse_int("RCA_METRICS_AGENT_MAX_ITERATIONS", 40),
        candidate_batch_size=_parse_int("RCA_CANDIDATE_BATCH_SIZE", 5),
        search_temperature=_parse_float("RCA_SEARCH_TEMPERATURE", 0.7),
        exploration_weight=_parse_float("RCA_EXPLORATION_WEIGHT", 1.0),
        self_consistency_weight=_parse_float("RCA_SELF_CONSISTENCY_WEIGHT", 0.5),
    )

    dataset = DatasetConfig(
        logs_root=Path(
            os.getenv(
                "RCA_LOGS_ROOT",
                "lo2-sample/logs/light-oauth2-data-1719592986/access_token_auth_header_error_401",
            )
        ),
        metrics_csv=Path(
            os.getenv(
                "RCA_METRICS_CSV",
                "lo2-sample/metrics/light-oauth2-data-1719592986.csv",
            )
        ),
        metrics_chart_dir=Path(os.getenv("RCA_METRICS_CHART_DIR", "charts")),
    )

    os.environ["ANTHROPIC_API_KEY"] = api_key
    os.environ.setdefault("RCA_METRICS_CHART_DIR", str(dataset.metrics_chart_dir))
    os.environ.setdefault("RCA_LOGS_ROOT", str(dataset.logs_root))
    os.environ.setdefault("RCA_METRICS_CSV", str(dataset.metrics_csv))

    anthropic = AnthropicConfig(
        api_key=api_key,
        log_model=log_model,
        metrics_model=metrics_model,
        supervisor_model=supervisor_model,
    )

    return Settings(anthropic=anthropic, runtime=runtime, dataset=dataset)


__all__ = [
    "AnthropicConfig",
    "RuntimeConfig",
    "Settings",
    "DatasetConfig",
    "get_settings",
]
