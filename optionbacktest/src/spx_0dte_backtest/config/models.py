"""Typed, immutable application configuration."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictFrozenModel(BaseModel):
    """Base model that rejects misspelled keys and mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(StrictFrozenModel):
    timezone: str = "America/New_York"
    results_directory: Path = Path("results")

    @field_validator("timezone")
    @classmethod
    def timezone_must_exist(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value


class DataConfig(StrictFrozenModel):
    timestamp_column: str = "timestamp"
    timestamp_semantics: str = "bar_start"
    source_timezone: str | None = None
    expected_frequency_minutes: int = Field(default=5, gt=0)
    strict_validation: bool = True

    @field_validator("timestamp_semantics")
    @classmethod
    def valid_timestamp_semantics(cls, value: str) -> str:
        if value not in {"bar_start", "bar_end"}:
            raise ValueError("timestamp_semantics must be 'bar_start' or 'bar_end'")
        return value

    @field_validator("source_timezone")
    @classmethod
    def source_timezone_must_exist(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                ZoneInfo(value)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value


class MarketConfig(StrictFrozenModel):
    calendar: str = "XNYS"
    regular_open: str = "09:30"
    regular_close: str = "16:00"


class LoggingConfig(StrictFrozenModel):
    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"unsupported logging level: {value}")
        return normalized


class AppConfig(StrictFrozenModel):
    project: ProjectConfig = ProjectConfig()
    data: DataConfig = DataConfig()
    market: MarketConfig = MarketConfig()
    logging: LoggingConfig = LoggingConfig()


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load configuration from YAML, applying model defaults for omitted values."""

    if path is None:
        return AppConfig()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return AppConfig.model_validate(raw)
