from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from spx_0dte_backtest.config import AppConfig
from spx_0dte_backtest.data.loaders import OptionChainLoader

HEADER = (
    "timestamp,expiration,strike,option_type,bid,ask,last,volume,open_interest,"
    "delta,gamma,theta,vega,implied_volatility,contract_symbol,source\n"
)


def row(
    timestamp: str = "2026-08-12T10:45:00-04:00",
    expiration: str = "2026-08-12",
    strike: str = "6000",
    option_type: str = "CALL",
    bid: str = "1.00",
    ask: str = "1.20",
    delta: str = "0.10",
    gamma: str = "0.01",
) -> str:
    return (
        f"{timestamp},{expiration},{strike},{option_type},{bid},{ask},1.10,10,100,"
        f"{delta},{gamma},-0.50,0.10,0.20,SPXW_TEST,SYNTHETIC_TEST\n"
    )


def test_loads_canonical_same_day_quotes(config: AppConfig, write_csv) -> None:
    path = write_csv(HEADER + row() + row(strike="5970", option_type="P", delta="-0.10"))
    result = OptionChainLoader(config).load(path)
    assert result.validation.is_valid
    assert result.validation.accepted_rows == 2
    assert set(result.quotes["option_type"]) == {"CALL", "PUT"}
    assert str(result.quotes["timestamp"].dt.tz) == "America/New_York"


def test_rejects_crossed_zero_bid_invalid_strike_and_missing_greeks(config, write_csv) -> None:
    path = write_csv(
        HEADER
        + row(bid="1.30", ask="1.20")
        + row(strike="0")
        + row(bid="0")
        + row(gamma="")
        + row().replace(",10,100,", ",-1,100,")
    )
    report = OptionChainLoader(config).load(path).validation
    assert report.rejected_rows == 5
    assert report.counts_by_code["CROSSED_MARKET"] == 1
    assert report.counts_by_code["INVALID_STRIKE"] == 1
    assert report.counts_by_code["ZERO_BID"] == 1
    assert report.counts_by_code["MISSING_GREEKS"] == 1
    assert report.counts_by_code["INVALID_SIZE"] == 1


def test_rejects_wrong_expiration_expired_quote_and_duplicate_contract(config, write_csv) -> None:
    path = write_csv(
        HEADER
        + row(expiration="2026-08-13")
        + row(timestamp="2026-08-13T10:45:00-04:00", expiration="2026-08-12")
        + row()
        + row()
    )
    report = OptionChainLoader(config).load(path).validation
    assert report.counts_by_code["INCORRECT_EXPIRATION"] == 1
    assert report.counts_by_code["EXPIRED_CONTRACT"] == 1
    assert report.counts_by_code["DUPLICATE_OPTION_QUOTE"] == 2


def test_rejects_off_session_and_naive_timestamp(config, write_csv) -> None:
    path = write_csv(
        HEADER + row(timestamp="2026-08-12T08:00:00-04:00") + row(timestamp="2026-08-12 10:45:00")
    )
    report = OptionChainLoader(config).load(path).validation
    assert report.counts_by_code["OUTSIDE_SESSION"] == 1
    assert report.counts_by_code["NAIVE_TIMESTAMP"] == 1


def test_end_cutoff_excludes_future_snapshots(config, write_csv) -> None:
    path = write_csv(HEADER + row() + row(timestamp="2026-08-12T10:50:00-04:00"))
    cutoff = datetime(2026, 8, 12, 10, 45, tzinfo=ZoneInfo("America/New_York"))
    result = OptionChainLoader(config).load(path, end=cutoff)
    assert len(result.quotes) == 1
    assert result.quotes.iloc[0]["timestamp"] == pd.Timestamp(cutoff)


def test_parquet_has_same_contract(config, tmp_path) -> None:
    frame = pd.read_csv(pd.io.common.StringIO(HEADER + row()))
    path = tmp_path / "quotes.parquet"
    frame.to_parquet(path)
    result = OptionChainLoader(config).load(path)
    assert result.validation.is_valid
    assert len(result.quotes) == 1
