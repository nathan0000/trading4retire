from __future__ import annotations

from pathlib import Path

import pytest

from spx_0dte_backtest.config import AppConfig


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def write_csv(tmp_path: Path):
    def _write(contents: str, name: str = "bars.csv") -> Path:
        path = tmp_path / name
        path.write_text(contents, encoding="utf-8")
        return path

    return _write
