"""Versioned, transactional SQLite storage for research inputs and results."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spx_0dte_backtest.backtest import TradeRecord, daily_summary
from spx_0dte_backtest.data import OptionQuote
from spx_0dte_backtest.strategies import TradeCandidate

SCHEMA_VERSION = 1


class DuplicateRunError(ValueError):
    """A run ID already exists and cannot be overwritten."""


class SQLiteStore:
    """Persist one complete run per atomic transaction."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            current = connection.execute("PRAGMA user_version").fetchone()[0]
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema version {current} is newer than supported {SCHEMA_VERSION}"
                )
            connection.executescript(_SCHEMA)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def persist_run(
        self,
        *,
        run_id: str,
        metadata: Mapping[str, Any],
        output_directory: Path | str,
        market_bars: pd.DataFrame,
        option_quotes: pd.DataFrame,
        market_features: pd.DataFrame,
        candidates: Sequence[TradeCandidate],
        records: Sequence[TradeRecord],
        trades: pd.DataFrame,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run ID cannot be empty")
        self.initialize()
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO backtest_runs (
                        run_id, created_at, status, strategy, data_classification,
                        code_version, configuration_json, metadata_json, output_directory
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        datetime.now(UTC).isoformat(),
                        "COMPLETED",
                        metadata.get("strategy"),
                        metadata.get("data_classification"),
                        metadata.get("code_version"),
                        _json(metadata.get("configuration", {})),
                        _json(metadata),
                        str(Path(output_directory).resolve()),
                    ),
                )
                self._market_bars(connection, run_id, market_bars)
                self._option_quotes(connection, run_id, option_quotes)
                self._features(connection, run_id, market_features)
                self._candidates(connection, run_id, candidates)
                self._trades(connection, run_id, records, trades)
                self._daily(connection, run_id, trades)
        except sqlite3.IntegrityError as exc:
            if "backtest_runs.run_id" in str(exc):
                raise DuplicateRunError(f"run ID already exists: {run_id}") from exc
            raise

    def list_runs(self) -> list[dict[str, Any]]:
        self.initialize()
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT run_id, created_at, status, strategy, data_classification,
                       code_version, output_directory
                FROM backtest_runs ORDER BY created_at, run_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def load_trades(self, run_id: str) -> pd.DataFrame:
        self.initialize()
        with closing(self._connect()) as connection:
            return pd.read_sql_query(
                "SELECT * FROM trades WHERE run_id = ? ORDER BY entry_timestamp, trade_id",
                connection,
                params=(run_id,),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _market_bars(connection: sqlite3.Connection, run_id: str, frame: pd.DataFrame) -> None:
        rows = [
            (
                run_id,
                _iso(row.timestamp),
                _nullable(row.open),
                _nullable(row.high),
                _nullable(row.low),
                _nullable(row.close),
                _nullable(row.volume),
                _nullable(row.symbol),
                _nullable(row.source),
            )
            for row in frame.itertuples(index=False)
        ]
        connection.executemany(
            """
            INSERT INTO market_bars
            (run_id, timestamp, open, high, low, close, volume, symbol, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    @staticmethod
    def _option_quotes(connection: sqlite3.Connection, run_id: str, frame: pd.DataFrame) -> None:
        rows = [
            (
                run_id,
                _iso(row.timestamp),
                _iso(row.expiration),
                float(row.strike),
                str(row.option_type),
                float(row.bid),
                float(row.ask),
                _nullable(row.last),
                _nullable(row.volume),
                _nullable(row.open_interest),
                _nullable(row.delta),
                _nullable(row.gamma),
                _nullable(row.theta),
                _nullable(row.vega),
                _nullable(row.implied_volatility),
                _nullable(row.contract_symbol),
                _nullable(row.source),
            )
            for row in frame.itertuples(index=False)
        ]
        connection.executemany(
            """
            INSERT INTO option_quotes (
                run_id, timestamp, expiration, strike, option_type, bid, ask, last,
                volume, open_interest, delta, gamma, theta, vega,
                implied_volatility, contract_symbol, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    @staticmethod
    def _features(connection: sqlite3.Connection, run_id: str, frame: pd.DataFrame) -> None:
        rows = [
            (run_id, _iso(row["timestamp"]), _json(row.to_dict())) for _, row in frame.iterrows()
        ]
        connection.executemany(
            "INSERT INTO market_features (run_id, timestamp, payload_json) VALUES (?, ?, ?)",
            rows,
        )

    @staticmethod
    def _candidates(
        connection: sqlite3.Connection,
        run_id: str,
        candidates: Sequence[TradeCandidate],
    ) -> None:
        rows = [
            (
                run_id,
                index,
                _iso(candidate.timestamp),
                candidate.strategy.value,
                int(candidate.is_accepted),
                None if candidate.rejection_reason is None else candidate.rejection_reason.value,
                candidate.rejection_detail,
                candidate.conservative_executable_credit,
                candidate.theoretical_mid_credit,
                candidate.context_features.get("factor_score"),
                _json(candidate.context_features),
            )
            for index, candidate in enumerate(candidates, start=1)
        ]
        connection.executemany(
            """
            INSERT INTO trade_candidates (
                run_id, candidate_id, timestamp, strategy, accepted,
                rejection_reason, rejection_detail, executable_credit,
                mid_credit, factor_score, context_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    @staticmethod
    def _trades(
        connection: sqlite3.Connection,
        run_id: str,
        records: Sequence[TradeRecord],
        frame: pd.DataFrame,
    ) -> None:
        if len(records) != len(frame):
            raise ValueError("trade records and normalized trade frame must have equal length")
        normalized_by_entry = {_iso(row["entry_timestamp"]): row for _, row in frame.iterrows()}
        for index, record in enumerate(records, start=1):
            normalized = normalized_by_entry.get(_iso(record.entry_timestamp))
            if normalized is None:
                raise ValueError(
                    f"normalized trade is missing entry timestamp: {_iso(record.entry_timestamp)}"
                )
            trade_id = f"{run_id}_{index:06d}"
            connection.execute(
                """
                INSERT INTO trades (
                    run_id, trade_id, entry_timestamp, exit_timestamp, strategy,
                    initial_credit, exit_debit, gross_pnl, commission, fees,
                    net_pnl, exit_reason, regime, factor_score, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    trade_id,
                    _iso(record.entry_timestamp),
                    _iso(record.exit_timestamp),
                    record.candidate.strategy.value,
                    record.initial_credit,
                    record.exit_debit,
                    record.gross_pnl,
                    record.commission,
                    record.fees,
                    record.net_pnl,
                    record.exit_reason.value,
                    record.candidate.context_features.get("regime"),
                    record.candidate.context_features.get("factor_score"),
                    _json(normalized.to_dict()),
                ),
            )
            exit_by_role = {fill.role: fill.quote for fill in record.exit_execution.fills}
            for leg in record.candidate.legs:
                connection.execute(
                    """
                    INSERT INTO trade_legs (
                        run_id, trade_id, role, quantity, expiration, strike,
                        option_type, entry_quote_json, exit_quote_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        trade_id,
                        leg.role.value,
                        leg.quantity,
                        _iso(leg.quote.expiration),
                        leg.quote.strike,
                        leg.quote.option_type.value,
                        _json(_quote(leg.quote)),
                        _json(_quote(exit_by_role[leg.role])),
                    ),
                )

    @staticmethod
    def _daily(connection: sqlite3.Connection, run_id: str, trades: pd.DataFrame) -> None:
        summary = daily_summary(trades)
        rows = [
            (
                run_id,
                _iso(row.date),
                int(row.trades),
                float(row.gross_pnl),
                float(row.commission),
                float(row.fees),
                float(row.net_pnl),
            )
            for row in summary.itertuples(index=False)
        ]
        connection.executemany(
            """
            INSERT INTO daily_summary
            (run_id, date, trades, gross_pnl, commission, fees, net_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _quote(quote: OptionQuote) -> dict[str, Any]:
    return {
        "timestamp": quote.timestamp,
        "expiration": quote.expiration,
        "strike": quote.strike,
        "option_type": quote.option_type.value,
        "bid": quote.bid,
        "ask": quote.ask,
        "last": quote.last,
        "volume": quote.volume,
        "open_interest": quote.open_interest,
        "delta": quote.delta,
        "gamma": quote.gamma,
        "theta": quote.theta,
        "vega": quote.vega,
        "implied_volatility": quote.implied_volatility,
        "contract_symbol": quote.contract_symbol,
        "source": quote.source,
    }


def _iso(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _nullable(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('COMPLETED')),
    strategy TEXT NOT NULL,
    data_classification TEXT NOT NULL,
    code_version TEXT NOT NULL,
    configuration_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    output_directory TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_bars (
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    timestamp TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    symbol TEXT, source TEXT,
    PRIMARY KEY (run_id, timestamp)
);

CREATE TABLE IF NOT EXISTS option_quotes (
    quote_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    timestamp TEXT NOT NULL,
    expiration TEXT NOT NULL,
    strike REAL NOT NULL,
    option_type TEXT NOT NULL,
    bid REAL NOT NULL,
    ask REAL NOT NULL,
    last REAL,
    volume INTEGER,
    open_interest INTEGER,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    implied_volatility REAL,
    contract_symbol TEXT,
    source TEXT
);
CREATE INDEX IF NOT EXISTS idx_option_quotes_snapshot
ON option_quotes(run_id, timestamp, expiration, option_type, strike);

CREATE TABLE IF NOT EXISTS market_features (
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    timestamp TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, timestamp)
);

CREATE TABLE IF NOT EXISTS trade_candidates (
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    candidate_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    strategy TEXT NOT NULL,
    accepted INTEGER NOT NULL CHECK (accepted IN (0, 1)),
    rejection_reason TEXT,
    rejection_detail TEXT,
    executable_credit REAL,
    mid_credit REAL,
    factor_score INTEGER,
    context_json TEXT NOT NULL,
    PRIMARY KEY (run_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS trades (
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    trade_id TEXT NOT NULL,
    entry_timestamp TEXT NOT NULL,
    exit_timestamp TEXT NOT NULL,
    strategy TEXT NOT NULL,
    initial_credit REAL NOT NULL,
    exit_debit REAL NOT NULL,
    gross_pnl REAL NOT NULL,
    commission REAL NOT NULL,
    fees REAL NOT NULL,
    net_pnl REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    regime TEXT,
    factor_score INTEGER,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, trade_id)
);

CREATE TABLE IF NOT EXISTS trade_legs (
    run_id TEXT NOT NULL,
    trade_id TEXT NOT NULL,
    role TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    expiration TEXT NOT NULL,
    strike REAL NOT NULL,
    option_type TEXT NOT NULL,
    entry_quote_json TEXT NOT NULL,
    exit_quote_json TEXT NOT NULL,
    PRIMARY KEY (run_id, trade_id, role),
    FOREIGN KEY (run_id, trade_id) REFERENCES trades(run_id, trade_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS daily_summary (
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    trades INTEGER NOT NULL,
    gross_pnl REAL NOT NULL,
    commission REAL NOT NULL,
    fees REAL NOT NULL,
    net_pnl REAL NOT NULL,
    PRIMARY KEY (run_id, date)
);
"""
