from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spx_0dte_backtest.config import AppConfig, load_config


def test_defaults_are_immutable_and_use_new_york() -> None:
    config = AppConfig()
    assert config.project.timezone == "America/New_York"
    assert config.data.expected_frequency_minutes == 5
    with pytest.raises(ValidationError):
        config.data.strict_validation = False  # type: ignore[misc]


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("data:\n  source_timezone: UTC\nlogging:\n  level: debug\n")
    config = load_config(path)
    assert config.data.source_timezone == "UTC"
    assert config.logging.level == "DEBUG"


@pytest.mark.parametrize(
    "contents",
    [
        "project:\n  timezone: Not/AZone\n",
        "data:\n  expected_frequency_minutes: 0\n",
        "data:\n  timestamp_semantics: midpoint\n",
        "data:\n  misspelled_key: true\n",
    ],
)
def test_invalid_configuration_is_rejected(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(contents)
    with pytest.raises(ValidationError):
        load_config(path)
