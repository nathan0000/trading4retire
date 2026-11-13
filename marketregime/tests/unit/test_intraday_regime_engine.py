"""End-to-end MarketRegimeSnapshot tests over synthetic sessions.

Covers session anchoring, the point-in-time rule, data-quality failure paths,
logging, thread safety, and the 10 ms latency budget.
"""

import logging
import statistics
import threading
import time as clock
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from regime_engine.application.intraday_regime import MarketRegimeSnapshot
from regime_engine.domain.intraday import (
    IntradayRegimeConfig,
    PrimaryRegime,
    RegimeSnapshotPayload,
    VolatilityState,
    VwapAlignment,
)

NY = ZoneInfo("America/New_York")
DAY = date(2026, 10, 30)


def _at(hh: int, mm: int, day: date = DAY, ss: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm, ss), tzinfo=NY)


def _walk(n: int, up: float, down: float, start: float = 6800.0) -> list[float]:
    """Alternating up/down steps: a trend with RSI well inside 30..70."""
    closes = [start]
    for i in range(1, n):
        closes.append(closes[-1] + (up if i % 2 else -down))
    return closes


def _table(  # noqa: PLR0913 - fixture builder, keyword-only knobs
    closes: Sequence[float],
    *,
    day: date = DAY,
    start: time = time(9, 30),
    minutes: int = 1,
    wiggle: float | Sequence[float] = 0.5,
    volume: Sequence[float] | None = None,
    proxies: dict[str, Sequence[float]] | None = None,
) -> dict[str, list[object]]:
    first = datetime.combine(day, start, tzinfo=NY)
    wiggles = [wiggle] * len(closes) if isinstance(wiggle, float) else list(wiggle)
    table: dict[str, list[object]] = {k: [] for k in ("timestamp", "open", "high", "low", "close")}
    table["volume"] = []
    previous = closes[0]
    for i, close in enumerate(closes):
        table["timestamp"].append(first + timedelta(minutes=minutes * i))
        table["open"].append(previous)
        table["high"].append(max(previous, close) + wiggles[i])
        table["low"].append(min(previous, close) - wiggles[i])
        table["close"].append(close)
        table["volume"].append(1000.0 if volume is None else volume[i])
        previous = close
    for name, values in (proxies or {}).items():
        table[name] = list(values)
    return table


def _scaled(closes: Sequence[float], factor: float) -> list[float]:
    return [c / factor for c in closes]


def _bull_day(n: int = 390) -> dict[str, list[object]]:
    closes = _walk(n, 1.0, 0.6)
    return _table(
        closes, proxies={"spy_close": _scaled(closes, 10), "qqq_close": _scaled(closes, 12)}
    )


def _engine(**overrides: object) -> MarketRegimeSnapshot:
    return MarketRegimeSnapshot(IntradayRegimeConfig(**overrides))  # type: ignore[arg-type]


def _truncate(table: dict[str, list[object]], n: int) -> dict[str, list[object]]:
    return {k: v[:n] for k, v in table.items()}


# -- regimes ---------------------------------------------------------------------------


def test_bull_session() -> None:
    payload = _engine().evaluate(_bull_day(), _at(10, 30))
    assert payload.primary_regime is PrimaryRegime.BULL_TREND
    assert payload.ema_gate_open
    assert payload.vwap_alignment is VwapAlignment.ABOVE_VWAP
    assert payload.volatility_state is VolatilityState.NORMAL
    assert payload.internal_bias_match
    assert payload.rsi_status is not None and 30 < payload.rsi_status < 70
    assert payload.bars_used == 60
    assert payload.last_bar_close_time == _at(10, 30).astimezone(UTC)
    assert payload.timestamp == _at(10, 30).astimezone(UTC)
    assert payload.config_version == "intraday-gate-0.1.0"


def test_bear_session() -> None:
    closes = _walk(390, 0.6, 1.0)
    table = _table(closes, proxies={"spy_close": _scaled(closes, 10)})
    payload = _engine().evaluate(table, _at(11, 0))
    assert payload.primary_regime is PrimaryRegime.BEAR_TREND
    assert payload.vwap_alignment is VwapAlignment.BELOW_VWAP
    assert payload.internal_bias_match


def test_extreme_volatility_after_quiet_open() -> None:
    closes = _walk(390, 1.0, 0.6)
    wiggles = [0.5] * 60 + [4.0] * 330
    payload = _engine().evaluate(_table(closes, wiggle=wiggles), _at(10, 45))
    assert payload.primary_regime is PrimaryRegime.EXTREME_VOLATILITY
    assert payload.volatility_state is VolatilityState.CRITICAL
    assert payload.volatility_ratio is not None and payload.volatility_ratio > 1.5
    assert not payload.ema_gate_open


def test_bearish_proxies_veto_bullish_spx() -> None:
    closes = _walk(390, 1.0, 0.6)
    falling = _walk(390, 0.3, 0.5, start=680.0)
    table = _table(closes, proxies={"spy_close": falling, "qqq_close": _scaled(falling, 1.2)})
    payload = _engine().evaluate(table, _at(10, 30))
    assert payload.primary_regime is PrimaryRegime.WHIPSAW_NEUTRAL
    assert not payload.internal_bias_match
    assert all(bps < -3 for _, bps in payload.proxy_bias)


def test_bullish_ema_below_vwap_logs_rejection(caplog: pytest.LogCaptureFixture) -> None:
    # Heavy volume trades at 6830 on the open, then price resets lower and grinds up on
    # light volume: the 20 EMA is above the 40 EMA, but price never reclaims VWAP.
    closes = _walk(10, 0.3, 0.3, start=6830.0) + _walk(80, 1.0, 0.6, start=6790.0)
    volume = [1e6] * 10 + [1e3] * 80
    with caplog.at_level(logging.INFO, logger="regime_engine.application.intraday_regime"):
        payload = _engine().evaluate(_table(closes, volume=volume), _at(11, 0))
    assert payload.ema_fast is not None and payload.ema_slow is not None
    assert payload.ema_fast > payload.ema_slow
    assert payload.primary_regime is PrimaryRegime.WHIPSAW_NEUTRAL
    assert (
        "[RegimeEngine] Bullish EMA detected but rejected by below-VWAP structural price "
        "action" in caplog.text
    )
    assert "Snapshot state: WHIPSAW_NEUTRAL" in caplog.text


# -- warm-up and session anchoring --------------------------------------------------------


def test_unknown_until_slow_ema_warms_up() -> None:
    engine = _engine()
    early = engine.evaluate(_bull_day(), _at(10, 9))
    assert early.primary_regime is PrimaryRegime.UNKNOWN
    assert "40 EMA" in early.reasons[0]
    assert early.spx_spot is not None  # facts still reported during warm-up
    assert engine.evaluate(_bull_day(), _at(10, 10)).primary_regime is PrimaryRegime.BULL_TREND


def test_before_first_bar_completes_is_unknown() -> None:
    payload = _engine().evaluate(_bull_day(), _at(9, 30, ss=59))
    assert payload.primary_regime is PrimaryRegime.UNKNOWN
    assert payload.spx_spot is None


def test_prior_session_and_premarket_bars_ignored() -> None:
    today = _bull_day()
    noise = _table([1.0 + i for i in range(30)], start=time(9, 0))  # premarket, wild prices
    yesterday = _table(_walk(390, 0.6, 1.0, start=9000.0), day=date(2026, 10, 29))
    combined = {k: yesterday[k] + noise[k] + today[k] for k in today if k in noise}
    assert _engine().evaluate(combined, _at(10, 30)) == _engine().evaluate(
        {k: today[k] for k in combined}, _at(10, 30)
    )


@pytest.mark.parametrize("day", [date(2026, 10, 30), date(2026, 12, 1), date(2026, 3, 9)])
def test_open_anchor_follows_dst(day: date) -> None:
    closes = _walk(390, 1.0, 0.6)
    payload = _engine().evaluate(_table(closes, day=day), _at(10, 30, day=day))
    assert payload.primary_regime is PrimaryRegime.BULL_TREND
    assert payload.bars_used == 60


def test_five_minute_bars() -> None:
    closes = _walk(78, 1.0, 0.6)
    payload = _engine(bar_minutes=5).evaluate(_table(closes, minutes=5), _at(13, 0))
    assert payload.bars_used == 42
    assert payload.primary_regime is PrimaryRegime.BULL_TREND


def test_bar_end_timestamps_match_bar_start() -> None:
    start_table = _bull_day()
    end_table = dict(start_table)
    end_table["timestamp"] = [t + timedelta(minutes=1) for t in start_table["timestamp"]]  # type: ignore[operator]
    assert _engine(timestamp_semantics="bar_end").evaluate(end_table, _at(10, 30)) == _engine(
        timestamp_semantics="bar_start"
    ).evaluate(start_table, _at(10, 30))


# -- point-in-time ------------------------------------------------------------------------


@pytest.mark.parametrize("minute", [40, 75, 200, 389])
def test_future_rows_never_change_result(minute: int) -> None:
    full = _bull_day()
    as_of = _at(9, 30) + timedelta(minutes=minute)
    tampered = {k: list(v) for k, v in full.items()}
    for i in range(minute, 390):  # the in-progress bar and everything after it
        for column in ("open", "high", "low", "close", "spy_close", "qqq_close"):
            tampered[column][i] = 1.0 if column != "high" else 99999.0
    reference = _engine().evaluate(_truncate(full, minute), as_of)
    assert _engine().evaluate(full, as_of) == reference
    assert _engine().evaluate(tampered, as_of) == reference


def test_in_progress_bar_excluded() -> None:
    payload = _engine().evaluate(_bull_day(), _at(10, 30, ss=45))
    assert payload.bars_used == 60  # the 10:30 bar closes at 10:31


# -- data quality -------------------------------------------------------------------------


def test_missing_open_bar_is_unknown() -> None:
    table = {k: v[1:] for k, v in _bull_day().items()}
    payload = _engine().evaluate(table, _at(10, 30))
    assert payload.primary_regime is PrimaryRegime.UNKNOWN
    assert "expected the open" in payload.reasons[0]


def test_gap_is_unknown() -> None:
    table = {k: v[:20] + v[21:] for k, v in _bull_day().items()}
    payload = _engine().evaluate(table, _at(10, 30))
    assert payload.primary_regime is PrimaryRegime.UNKNOWN
    assert "not contiguous after 09:49" in payload.reasons[0]


def test_non_finite_price_is_unknown() -> None:
    table = _bull_day()
    table["close"][10] = float("nan")
    assert _engine().evaluate(table, _at(10, 30)).primary_regime is PrimaryRegime.UNKNOWN


def test_naive_timestamp_is_unknown_not_exception() -> None:
    table = _bull_day()
    table["timestamp"][0] = datetime(2026, 10, 30, 9, 30)
    payload = _engine().evaluate(table, _at(10, 30))
    assert payload.primary_regime is PrimaryRegime.UNKNOWN
    assert "tz-aware" in payload.reasons[0]


def test_naive_as_of_is_caller_error() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _engine().evaluate(_bull_day(), datetime(2026, 10, 30, 10, 30))


def test_missing_column_is_unknown() -> None:
    table = _bull_day()
    del table["low"]
    payload = _engine().evaluate(table, _at(10, 30))
    assert "missing columns ['low']" in payload.reasons[0]


def test_zero_volume_index_has_no_vwap() -> None:
    table = _bull_day()
    table["volume"] = [0.0] * 390
    payload = _engine().evaluate(table, _at(10, 30))
    assert payload.primary_regime is PrimaryRegime.UNKNOWN
    assert payload.vwap_alignment is VwapAlignment.UNKNOWN
    assert payload.ema_fast is not None  # other indicators still reported


def test_explicit_volume_column() -> None:
    table = _bull_day()
    table["spy_volume"] = table.pop("volume")
    assert _engine().evaluate(table, _at(10, 30)).primary_regime is PrimaryRegime.UNKNOWN
    payload = _engine(volume_column="spy_volume").evaluate(table, _at(10, 30))
    assert payload.primary_regime is PrimaryRegime.BULL_TREND


def test_proxy_with_gap_is_dropped_and_noted() -> None:
    table = _bull_day()
    table["spy_close"][5] = float("nan")
    payload = _engine().evaluate(table, _at(10, 30))
    assert [name for name, _ in payload.proxy_bias] == ["qqq_close"]
    assert any("spy_close" in reason for reason in payload.reasons)


def test_pandas_dataframe_input() -> None:
    pd = pytest.importorskip("pandas")
    table = _bull_day()
    assert _engine().evaluate(pd.DataFrame(table), _at(10, 30)) == _engine().evaluate(
        table, _at(10, 30)
    )


# -- payload, concurrency, and latency -------------------------------------------------------


def test_payload_is_immutable() -> None:
    payload = _engine().evaluate(_bull_day(), _at(10, 30))
    with pytest.raises(FrozenInstanceError):
        payload.primary_regime = PrimaryRegime.UNKNOWN  # type: ignore[misc]
    assert replace(payload, spx_spot=1.0) != payload


def test_older_as_of_does_not_replace_latest() -> None:
    engine = _engine()
    newer = engine.evaluate(_bull_day(), _at(11, 0))
    engine.evaluate(_bull_day(), _at(10, 30))
    assert engine.latest == newer


def test_concurrent_evaluations() -> None:
    engine = _engine()
    table = _bull_day()
    results: list[RegimeSnapshotPayload] = []
    errors: list[BaseException] = []

    def worker(minute: int) -> None:
        try:
            results.append(engine.evaluate(table, _at(10, 0) + timedelta(minutes=minute)))
        except BaseException as exc:  # pragma: no cover - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(m,)) for m in range(0, 160, 5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert engine.latest is not None
    assert engine.latest.timestamp == max(r.timestamp for r in results)


def test_regime_change_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    engine = _engine()
    with caplog.at_level(logging.INFO, logger="regime_engine.application.intraday_regime"):
        engine.evaluate(_bull_day(), _at(10, 30))
        engine.evaluate(_bull_day(), _at(10, 31))
    assert sum("Snapshot state: BULL_TREND" in r.message for r in caplog.records) == 1


def test_full_session_evaluates_under_10ms() -> None:
    engine = _engine()
    table = _bull_day()
    as_of = _at(16, 0)
    timings = []
    for _ in range(25):
        began = clock.perf_counter()
        engine.evaluate(table, as_of)
        timings.append(clock.perf_counter() - began)
    assert statistics.median(timings) < 0.010
