"""MarketRegimeSnapshot: evaluate the intraday SPX regime gate at an arbitrary `as_of`.

Input is one column table, either a dict of equal-length sequences or a pandas
DataFrame, with columns:

    timestamp  tz-aware datetime, interpreted per `config.timestamp_semantics`
    open, high, low, close
    <config.volume_column>          weights VWAP (optional; no volume -> UNKNOWN)
    <each of config.proxy_columns>  proxy closes on the same timestamps (optional)

Point-in-time rules, applied before any indicator is computed:

* only bars from `as_of`'s exchange date, starting at the 09:30 ET open, are used,
  so EMAs, RSI, ATR, and VWAP all anchor at the open and never see prior sessions;
* a bar is used only once it has *completed* by `as_of`, so the bar in progress
  and anything later in the table are ignored, and appending future rows cannot
  change a result;
* a missing 09:30 bar, a gap, or a non-finite price yields UNKNOWN, never a guess.

Every evaluation recomputes from the table, so replay is deterministic. Each
indicator is a single pass over at most 390 bars, well inside 10 ms.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from regime_engine.domain.intraday import (
    IntradayRegimeConfig,
    PrimaryRegime,
    RegimeSnapshotPayload,
    VolatilityState,
    VwapAlignment,
)
from regime_engine.features.intraday_structure import session_vwap
from regime_engine.features.realized_vol import atr_last, true_range
from regime_engine.features.trend import ema_last, rsi_last
from regime_engine.inference.intraday_gate import GateInputs, classify_intraday

_LOG = logging.getLogger(__name__)
_REQUIRED = ("timestamp", "open", "high", "low", "close")
_BPS = 10_000.0


class ColumnTable(Protocol):
    """Anything indexable by column name: a dict of sequences or a pandas DataFrame."""

    def __getitem__(self, key: str, /) -> Any: ...

    def __contains__(self, key: object, /) -> bool: ...


class _InputError(Exception):
    """The table cannot be evaluated; becomes an UNKNOWN payload, never an exception."""


@dataclass(frozen=True, slots=True)
class _Session:
    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64] | None
    proxies: tuple[tuple[str, NDArray[np.float64]], ...]
    last_close_time: datetime | None
    notes: tuple[str, ...]


class MarketRegimeSnapshot:
    """Thread-safe intraday regime engine. `evaluate` may be called from any thread.

    Results are immutable `RegimeSnapshotPayload`s. `latest` returns the one with
    the greatest `as_of` seen so far, so an older replay call finishing late can't
    overwrite a newer live result.
    """

    def __init__(
        self,
        config: IntradayRegimeConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config or IntradayRegimeConfig()
        self._log = logger or _LOG
        self._tz = ZoneInfo(self._config.market_timezone)
        self._interval = timedelta(minutes=self._config.bar_minutes)
        self._lock = threading.Lock()
        self._latest: RegimeSnapshotPayload | None = None

    @property
    def config(self) -> IntradayRegimeConfig:
        return self._config

    @property
    def latest(self) -> RegimeSnapshotPayload | None:
        with self._lock:
            return self._latest

    def evaluate(self, bars: ColumnTable, as_of: datetime) -> RegimeSnapshotPayload:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        as_of = as_of.astimezone(UTC)
        try:
            session = self._extract_session(bars, as_of)
            payload = self._compute(session, as_of)
        except _InputError as exc:
            payload = self._unknown(as_of, (f"Input rejected: {exc}",))
        self._publish(payload)
        return payload

    # -- session extraction ---------------------------------------------------------

    def _extract_session(self, bars: ColumnTable, as_of: datetime) -> _Session:
        cfg = self._config
        missing = [column for column in _REQUIRED if column not in bars]
        if missing:
            raise _InputError(f"missing columns {missing}")
        stamps = list(bars["timestamp"])
        rows, starts = self._session_rows(stamps, as_of)
        notes: list[str] = []
        index = np.asarray(rows, dtype=np.intp)

        def column(name: str) -> NDArray[np.float64]:
            values = np.asarray(bars[name], dtype=np.float64)
            if values.shape != (len(stamps),):
                raise _InputError(f"column {name!r} length does not match timestamp")
            return values[index]

        open_, high, low, close = (column(name) for name in _REQUIRED[1:])
        if not np.isfinite(np.concatenate([open_, high, low, close])).all():
            raise _InputError("non-finite OHLC value")
        if (high < low).any() or (close > high).any() or (close < low).any():
            raise _InputError("OHLC inconsistent: close outside high/low")

        volume: NDArray[np.float64] | None = None
        if cfg.volume_column not in bars:
            notes.append(f"volume column {cfg.volume_column!r} absent; VWAP undefined")
        else:
            volume = column(cfg.volume_column)
            if not np.isfinite(volume).all() or (volume < 0).any():
                notes.append(f"volume column {cfg.volume_column!r} has invalid values")
                volume = None

        proxies: list[tuple[str, NDArray[np.float64]]] = []
        for name in cfg.proxy_columns:
            if name not in bars:
                continue
            values = column(name)
            if np.isfinite(values).all():
                proxies.append((name, values))
            else:
                notes.append(f"proxy {name!r} has gaps; ignored")

        return _Session(
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            proxies=tuple(proxies),
            last_close_time=starts[-1] + self._interval,
            notes=tuple(notes),
        )

    def _session_rows(
        self, stamps: list[object], as_of: datetime
    ) -> tuple[list[int], list[datetime]]:
        """Row numbers and UTC bar starts of today's session bars completed by as_of."""
        cfg = self._config
        day = as_of.astimezone(self._tz).date()
        open_at = datetime.combine(day, cfg.session_open, tzinfo=self._tz).astimezone(UTC)
        close_at = datetime.combine(day, cfg.session_close, tzinfo=self._tz).astimezone(UTC)
        shift = timedelta(0) if cfg.timestamp_semantics == "bar_start" else self._interval

        rows: list[int] = []
        starts: list[datetime] = []
        for row, stamp in enumerate(stamps):
            if not isinstance(stamp, datetime) or stamp.tzinfo is None:
                raise _InputError(f"timestamp at row {row} is not a tz-aware datetime")
            start = stamp.astimezone(UTC) - shift
            # Completed by as_of and inside today's regular session.
            if open_at <= start < close_at and start + self._interval <= as_of:
                rows.append(row)
                starts.append(start)

        if not rows:
            raise _InputError("no completed regular-session bars at as_of")
        if starts[0] != open_at:
            first = starts[0].astimezone(self._tz).strftime("%H:%M")
            raise _InputError(f"first session bar starts {first}, expected the open")
        for previous, current in pairwise(starts):
            if current - previous != self._interval:
                at = previous.astimezone(self._tz).strftime("%H:%M")
                raise _InputError(f"bars not contiguous after {at}")
        return rows, starts

    # -- indicators and classification -------------------------------------------------

    def _compute(self, session: _Session, as_of: datetime) -> RegimeSnapshotPayload:
        cfg = self._config
        close = session.close
        spot = float(close[-1])

        fast = ema_last(close, cfg.ema_fast)
        slow = ema_last(close, cfg.ema_slow)
        rsi = rsi_last(close, cfg.rsi_period)
        vwap = (
            session_vwap(session.high, session.low, close, session.volume)
            if session.volume is not None
            else None
        )

        ranges = true_range(session.high, session.low, close)
        atr = atr_last(ranges, cfg.atr_period)
        baseline_bars = cfg.baseline_minutes // cfg.bar_minutes
        baseline = float(ranges[:baseline_bars].mean()) if ranges.size >= baseline_bars else None
        ratio = atr / baseline if atr is not None and baseline else None
        if ratio is None:
            vol_state = VolatilityState.UNKNOWN
        elif ratio > cfg.critical_ratio:
            vol_state = VolatilityState.CRITICAL
        elif ratio > cfg.expanding_ratio:
            vol_state = VolatilityState.EXPANDING
        else:
            vol_state = VolatilityState.NORMAL

        proxy_bias: list[tuple[str, float]] = []
        for name, values in session.proxies:
            proxy_fast = ema_last(values, cfg.ema_fast)
            proxy_slow = ema_last(values, cfg.ema_slow)
            if proxy_fast is not None and proxy_slow:
                proxy_bias.append((name, (proxy_fast - proxy_slow) / proxy_slow * _BPS))

        decision = classify_intraday(
            GateInputs(
                close=spot,
                ema_fast=fast,
                ema_slow=slow,
                vwap=vwap,
                rsi=rsi,
                volatility_state=vol_state,
                volatility_ratio=ratio,
                proxy_bias_bps=tuple(proxy_bias),
            ),
            cfg,
        )
        if vwap is None:
            alignment = VwapAlignment.UNKNOWN
        elif spot > vwap:
            alignment = VwapAlignment.ABOVE_VWAP
        else:
            alignment = VwapAlignment.BELOW_VWAP

        return RegimeSnapshotPayload(
            timestamp=as_of,
            spx_spot=spot,
            primary_regime=decision.regime,
            ema_gate_open=decision.regime in (PrimaryRegime.BULL_TREND, PrimaryRegime.BEAR_TREND),
            vwap_alignment=alignment,
            rsi_status=rsi,
            volatility_state=vol_state,
            internal_bias_match=decision.internal_bias_match,
            ema_fast=fast,
            ema_slow=slow,
            vwap=vwap,
            atr=atr,
            atr_baseline=baseline,
            volatility_ratio=ratio,
            proxy_bias=tuple(proxy_bias),
            bars_used=int(close.size),
            last_bar_close_time=session.last_close_time,
            reasons=decision.reasons + session.notes,
            config_version=cfg.config_version,
        )

    def _unknown(self, as_of: datetime, reasons: tuple[str, ...]) -> RegimeSnapshotPayload:
        return RegimeSnapshotPayload(
            timestamp=as_of,
            spx_spot=None,
            primary_regime=PrimaryRegime.UNKNOWN,
            ema_gate_open=False,
            vwap_alignment=VwapAlignment.UNKNOWN,
            rsi_status=None,
            volatility_state=VolatilityState.UNKNOWN,
            internal_bias_match=False,
            reasons=reasons,
            config_version=self._config.config_version,
        )

    # -- publication -------------------------------------------------------------------

    def _publish(self, payload: RegimeSnapshotPayload) -> None:
        with self._lock:
            previous = self._latest
            if previous is not None and payload.timestamp < previous.timestamp:
                stale = True
            else:
                stale = False
                self._latest = payload
        changed = previous is None or previous.primary_regime is not payload.primary_regime
        level = logging.INFO if changed and not stale else logging.DEBUG
        self._log.log(
            level,
            "[RegimeEngine] %s. Snapshot state: %s",
            "; ".join(payload.reasons),
            payload.primary_regime.value,
            extra={
                "regime_as_of": payload.timestamp.isoformat(),
                "regime_config_version": payload.config_version,
            },
        )
