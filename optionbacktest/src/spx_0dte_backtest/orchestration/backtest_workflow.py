"""Causal data-to-report orchestration for the iron-condor backtest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from spx_0dte_backtest.analysis import FactorScoreEngine
from spx_0dte_backtest.backtest import BacktestEngine, records_to_frame
from spx_0dte_backtest.config import AppConfig
from spx_0dte_backtest.data import OptionQuote, OptionType
from spx_0dte_backtest.data.loaders import OptionChainLoader, SpxBarLoader
from spx_0dte_backtest.database import DuplicateRunError, SQLiteStore
from spx_0dte_backtest.features import FeatureEngine
from spx_0dte_backtest.regimes import RegimeEngine
from spx_0dte_backtest.strategies import MarketContext, StrategyEngine, StrategyType
from spx_0dte_backtest.visualization import ReportArtifacts, ReportGenerator


class WorkflowError(RuntimeError):
    """A run cannot produce a trustworthy completed result."""


class WorkflowDataQualityError(WorkflowError):
    """Strict source-data validation failed."""


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    scheduled_entries: int
    accepted_candidates: int
    rejected_candidates: int
    completed_trades: int
    rejection_counts: dict[str, int]
    output_directory: Path
    artifacts: ReportArtifacts
    database_path: Path | None = None

    def summary(self) -> dict[str, object]:
        return {
            "scheduled_entries": self.scheduled_entries,
            "accepted_candidates": self.accepted_candidates,
            "rejected_candidates": self.rejected_candidates,
            "completed_trades": self.completed_trades,
            "rejection_counts": self.rejection_counts,
            "output_directory": str(self.output_directory),
            "database_path": None if self.database_path is None else str(self.database_path),
        }


class BacktestWorkflow:
    """Connect validated historical inputs to a completed report bundle."""

    def __init__(
        self,
        config: AppConfig,
        *,
        strategy_engine: StrategyEngine | None = None,
        backtest_engine: BacktestEngine | None = None,
        report_generator: ReportGenerator | None = None,
    ) -> None:
        self.config = config
        self.strategy_engine = strategy_engine or StrategyEngine()
        self.backtest_engine = backtest_engine or BacktestEngine()
        self.report_generator = report_generator or ReportGenerator()
        self.factor_scores = FactorScoreEngine()

    def run(
        self,
        *,
        spx_source: Path | str,
        option_source: Path | str,
        output_directory: Path | str,
        run_id: str,
        vix_source: Path | str | None = None,
        database_path: Path | str | None = None,
    ) -> WorkflowResult:
        output = Path(output_directory)
        database = None if database_path is None else SQLiteStore(database_path)
        if database is not None and any(run["run_id"] == run_id for run in database.list_runs()):
            raise DuplicateRunError(f"run ID already exists: {run_id}")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"report directory is not empty: {output}")
        spx = SpxBarLoader(self.config).load(spx_source)
        options = OptionChainLoader(self.config).load(option_source)
        if self.config.data.strict_validation:
            invalid = []
            if not spx.validation.is_valid:
                invalid.append("SPX")
            if not options.validation.is_valid:
                invalid.append("options")
            if invalid:
                raise WorkflowDataQualityError(
                    f"strict validation failed for: {', '.join(invalid)}"
                )
        vix = self._load_vix(vix_source) if vix_source is not None else None
        features = FeatureEngine(self.config).compute(spx.bars, vix=vix)
        featured = RegimeEngine().classify(features)
        quote_frame = options.quotes
        candidates = []
        records = []

        for _, row in self._scheduled_rows(featured).iterrows():
            timestamp = pd.Timestamp(row["timestamp"])
            expiration = timestamp.tz_convert(self.config.project.timezone).date()
            snapshot = quote_frame.loc[quote_frame["timestamp"] == timestamp]
            context = MarketContext(
                timestamp=timestamp,
                expiration=expiration,
                spx_price=float(row["close"]),
                option_quotes=snapshot,
                features=row.to_dict(),
            )
            candidate = self.strategy_engine.generate_trade(StrategyType.IRON_CONDOR, context)
            candidates.append(candidate)
            if not candidate.is_accepted:
                continue
            candidate = self.factor_scores.enrich_candidate(candidate)
            candidates[-1] = candidate
            monitoring = quote_frame.loc[
                (quote_frame["timestamp"] > timestamp) & (quote_frame["expiration"] == expiration)
            ]
            snapshots = [
                self._quotes(group) for _, group in monitoring.groupby("timestamp", sort=True)
            ]
            try:
                records.append(self.backtest_engine.run_candidate(candidate, snapshots))
            except (RuntimeError, ValueError) as exc:
                raise WorkflowError(
                    f"forced-exit execution failed at {timestamp.isoformat()}: {exc}"
                ) from exc

        trade_frame = records_to_frame(records)
        if not trade_frame.empty:
            closes = spx.bars.set_index("timestamp")["close"]
            trade_frame["spx_exit"] = trade_frame["exit_timestamp"].map(closes)
        else:
            trade_frame["spx_exit"] = pd.Series(dtype="float64")
        metadata = self._metadata(run_id, spx, options, vix_source)
        artifacts = self.report_generator.generate(
            trade_frame,
            output,
            metadata=metadata,
        )
        candidate_frame = self._candidate_frame(candidates)
        candidate_path = output / "candidates.parquet"
        candidate_frame.to_parquet(candidate_path, index=False)
        rejection_counts = Counter(
            candidate.rejection_reason.value
            for candidate in candidates
            if candidate.rejection_reason is not None
        )
        result = WorkflowResult(
            scheduled_entries=len(candidates),
            accepted_candidates=sum(candidate.is_accepted for candidate in candidates),
            rejected_candidates=sum(not candidate.is_accepted for candidate in candidates),
            completed_trades=len(records),
            rejection_counts=dict(sorted(rejection_counts.items())),
            output_directory=output,
            artifacts=artifacts,
            database_path=None if database is None else database.path.resolve(),
        )
        summary_path = output / "workflow_summary.json"
        summary_path.write_text(
            json.dumps(result.summary(), indent=2, sort_keys=True), encoding="utf-8"
        )
        if database is not None:
            database.persist_run(
                run_id=run_id,
                metadata=metadata,
                output_directory=output,
                market_bars=spx.bars,
                option_quotes=options.quotes,
                market_features=featured,
                candidates=candidates,
                records=records,
                trades=trade_frame,
            )
        self._extend_manifest(artifacts.manifest_path, result.database_path)
        return result

    def _scheduled_rows(self, features: pd.DataFrame) -> pd.DataFrame:
        local = features["timestamp"].dt.tz_convert(self.config.project.timezone)
        minutes = local.dt.hour * 60 + local.dt.minute
        first = 10 * 60 + 45
        last = 15 * 60 + 15
        return features.loc[(minutes >= first) & (minutes <= last) & ((minutes - first) % 15 == 0)]

    @staticmethod
    def _quotes(frame: pd.DataFrame) -> tuple[OptionQuote, ...]:
        return tuple(
            OptionQuote(
                timestamp=pd.Timestamp(row.timestamp).to_pydatetime(),
                expiration=row.expiration,
                strike=float(row.strike),
                option_type=OptionType(row.option_type),
                bid=float(row.bid),
                ask=float(row.ask),
                last=None if pd.isna(row.last) else float(row.last),
                volume=None if pd.isna(row.volume) else int(row.volume),
                open_interest=None if pd.isna(row.open_interest) else int(row.open_interest),
                delta=float(row.delta),
                gamma=float(row.gamma),
                theta=float(row.theta),
                vega=float(row.vega),
                implied_volatility=float(row.implied_volatility),
                contract_symbol=None if pd.isna(row.contract_symbol) else str(row.contract_symbol),
                source=None if pd.isna(row.source) else str(row.source),
            )
            for row in frame.itertuples(index=False)
        )

    @staticmethod
    def _candidate_frame(candidates: list[Any]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": candidate.timestamp,
                    "strategy": candidate.strategy.value,
                    "accepted": candidate.is_accepted,
                    "rejection_reason": None
                    if candidate.rejection_reason is None
                    else candidate.rejection_reason.value,
                    "rejection_detail": candidate.rejection_detail,
                    "conservative_executable_credit": candidate.conservative_executable_credit,
                    "theoretical_mid_credit": candidate.theoretical_mid_credit,
                    "factor_score": candidate.context_features.get("factor_score"),
                }
                for candidate in candidates
            ]
        )

    def _load_vix(self, source: Path | str) -> pd.DataFrame:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            raise ValueError(f"unsupported VIX data format: {path.suffix or '<none>'}")
        missing = {"timestamp", "vix"}.difference(frame.columns)
        if missing:
            raise ValueError(f"VIX data missing required columns: {sorted(missing)}")
        result = frame[["timestamp", "vix"]].copy()
        parsed: list[pd.Timestamp] = []
        for value in result["timestamp"]:
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                source_timezone = self.config.data.source_timezone
                if source_timezone is None:
                    raise ValueError("naive VIX timestamp requires data.source_timezone")
                timestamp = timestamp.tz_localize(
                    source_timezone, ambiguous="raise", nonexistent="raise"
                )
            parsed.append(timestamp.tz_convert(self.config.project.timezone))
        result["timestamp"] = pd.DatetimeIndex(parsed)
        result["vix"] = pd.to_numeric(result["vix"], errors="coerce")
        if result["vix"].isna().any() or (result["vix"] < 0).any():
            raise ValueError("VIX values must be numeric and non-negative")
        if not result["timestamp"].is_monotonic_increasing or not result["timestamp"].is_unique:
            raise ValueError("VIX timestamps must be sorted and unique")
        return result

    def _metadata(self, run_id: str, spx: Any, options: Any, vix_source: Any) -> dict[str, Any]:
        if not run_id.strip():
            raise ValueError("run ID cannot be empty")
        sources = set(spx.bars["source"].dropna().astype(str)) | set(
            options.quotes["source"].dropna().astype(str)
        )
        classification = (
            "SYNTHETIC" if any("SYNTHETIC" in value.upper() for value in sources) else "HISTORICAL"
        )
        execution = self.backtest_engine.execution_engine.settings
        risk = self.backtest_engine.risk_settings
        lifecycle = self.backtest_engine.settings
        spx_path = Path(spx.source_path)
        option_path = Path(options.source_path)
        vix_path = None if vix_source is None else Path(vix_source).resolve()
        return {
            "run_id": run_id,
            "strategy": StrategyType.IRON_CONDOR.value,
            "strategy_version": "iron_condor_v1",
            "code_version": version("spx-0dte-backtest"),
            "data_classification": classification,
            "spx_source": str(spx_path),
            "option_source": str(option_path),
            "vix_source": None if vix_path is None else str(vix_path),
            "data_versions": {
                "spx_sha256": self._sha256(spx_path),
                "options_sha256": self._sha256(option_path),
                "vix_sha256": None if vix_path is None else self._sha256(vix_path),
            },
            "configuration": self.config.model_dump(mode="json"),
            "execution": {
                "mode": execution.mode.value,
                "slippage_points": execution.slippage_points,
                "contract_multiplier": execution.contract_multiplier,
                "commission_per_contract": execution.cost_model.commission_per_contract,
                "exchange_fee_per_contract": execution.cost_model.exchange_fee_per_contract,
            },
            "risk": {
                "stop_multiplier": risk.stop_multiplier,
                "profit_target_fraction": risk.profit_target_fraction,
            },
            "lifecycle": {
                "forced_exit_time": lifecycle.forced_exit_time.isoformat(),
                "timezone": lifecycle.timezone,
            },
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @staticmethod
    def _extend_manifest(path: Path, database_path: Path | None) -> None:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"]["candidates"] = "candidates.parquet"
        manifest["files"]["workflow_summary"] = "workflow_summary.json"
        if database_path is not None:
            manifest["files"]["database"] = str(database_path)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
