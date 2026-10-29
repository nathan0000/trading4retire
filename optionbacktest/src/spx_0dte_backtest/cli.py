"""Validation and end-to-end backtest command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from spx_0dte_backtest.config import load_config
from spx_0dte_backtest.data.loaders import OptionChainLoader, OptionLoadResult, SpxBarLoader
from spx_0dte_backtest.data.loaders.base import LoadResult
from spx_0dte_backtest.orchestration import (
    BacktestWorkflow,
    WorkflowDataQualityError,
    WorkflowError,
)
from spx_0dte_backtest.providers import (
    HistoricalDataRequest,
    ProviderConnectionError,
    ProviderDownloader,
    ThetaDataAdapter,
    ThetaDataSettings,
    YFinanceAdapter,
    YFinanceSettings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spx-0dte")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-spx", help="validate SPX five-minute bars")
    validate.add_argument("--input", type=Path, required=True, help="CSV or Parquet input")
    validate.add_argument(
        "--config", type=Path, default=Path("config/settings.yaml"), help="YAML settings"
    )
    validate.add_argument("--json", action="store_true", help="emit a JSON summary")
    options = subparsers.add_parser(
        "validate-options", help="validate historical SPX option snapshots"
    )
    options.add_argument("--input", type=Path, required=True, help="CSV or Parquet input")
    options.add_argument(
        "--config", type=Path, default=Path("config/settings.yaml"), help="YAML settings"
    )
    options.add_argument("--json", action="store_true", help="emit a JSON summary")
    run = subparsers.add_parser("run-backtest", help="run the complete iron-condor workflow")
    run.add_argument("--spx", type=Path, required=True, help="SPX CSV or Parquet input")
    run.add_argument("--options", type=Path, required=True, help="option CSV or Parquet input")
    run.add_argument("--vix", type=Path, help="optional VIX CSV or Parquet input")
    run.add_argument("--output", type=Path, required=True, help="new report directory")
    run.add_argument("--run-id", help="reproducible run identifier")
    run.add_argument("--database", type=Path, help="optional SQLite research database")
    run.add_argument(
        "--config", type=Path, default=Path("config/settings.yaml"), help="YAML settings"
    )
    run.add_argument("--json", action="store_true", help="emit a JSON summary")
    download = subparsers.add_parser("download-data", help="download canonical historical data")
    download.add_argument("--provider", choices=["thetadata", "yfinance"], default="thetadata")
    download.add_argument("--start", required=True, help="first date, YYYY-MM-DD")
    download.add_argument("--end", required=True, help="last date, YYYY-MM-DD")
    download.add_argument("--output", type=Path, required=True, help="new download directory")
    download.add_argument("--base-url", default="http://127.0.0.1:25510/v2")
    download.add_argument("--timeout", type=float, default=60.0)
    download.add_argument("--include-vix", action="store_true")
    download.add_argument("--interval", type=int, default=5, help="bar interval in minutes")
    download.add_argument("--json", action="store_true", help="emit a JSON summary")
    return parser


def _summary(result: LoadResult | OptionLoadResult) -> dict[str, object]:
    bars = result.bars if isinstance(result, LoadResult) else result.quotes
    report = result.validation
    return {
        "source": str(result.source_path),
        "total_rows": report.total_rows,
        "accepted_rows": report.accepted_rows,
        "rejected_rows": report.rejected_rows,
        "first_timestamp": None if bars.empty else bars["timestamp"].min().isoformat(),
        "last_timestamp": None if bars.empty else bars["timestamp"].max().isoformat(),
        "issue_counts": report.counts_by_code,
        "valid": report.is_valid,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-backtest":
        return _run_backtest(args)
    if args.command == "download-data":
        return _download_data(args)
    try:
        config = load_config(args.config)
        logging.basicConfig(level=config.logging.level)
        loader = (
            SpxBarLoader(config) if args.command == "validate-spx" else OptionChainLoader(config)
        )
        result = loader.load(args.input)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        print(f"error: {exc}")
        return 2

    summary = _summary(result)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Source: {summary['source']}")
        print(
            "Rows: "
            f"{summary['total_rows']} total, "
            f"{summary['accepted_rows']} accepted, "
            f"{summary['rejected_rows']} rejected"
        )
        print(f"Range: {summary['first_timestamp']} to {summary['last_timestamp']}")
        print(f"Issues: {summary['issue_counts']}")
        print(f"Valid: {summary['valid']}")

    if config.data.strict_validation and not result.validation.is_valid:
        return 1
    return 0


def _run_backtest(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        logging.basicConfig(level=config.logging.level)
        run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        result = BacktestWorkflow(config).run(
            spx_source=args.spx,
            option_source=args.options,
            vix_source=args.vix,
            output_directory=args.output,
            run_id=run_id,
            database_path=args.database,
        )
    except WorkflowDataQualityError as exc:
        print(f"error: {exc}")
        return 1
    except (FileNotFoundError, FileExistsError, ValueError, ValidationError, WorkflowError) as exc:
        print(f"error: {exc}")
        return 2

    summary = result.summary()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Output: {summary['output_directory']}")
        print(
            f"Candidates: {summary['accepted_candidates']} accepted, "
            f"{summary['rejected_candidates']} rejected"
        )
        print(f"Completed trades: {summary['completed_trades']}")
        print(f"Rejections: {summary['rejection_counts']}")
    return 0


def _download_data(args: argparse.Namespace) -> int:
    try:
        request = HistoricalDataRequest(
            start=args.start,
            end=args.end,
            interval_minutes=args.interval,
            include_vix=args.include_vix,
        )
        adapter = (
            ThetaDataAdapter(
                ThetaDataSettings(base_url=args.base_url, timeout_seconds=args.timeout)
            )
            if args.provider == "thetadata"
            else YFinanceAdapter(YFinanceSettings())
        )
        dataset = adapter.download(request)
        artifacts = ProviderDownloader().write(dataset, request, args.output)
    except (
        FileExistsError,
        FileNotFoundError,
        ProviderConnectionError,
        ValueError,
    ) as exc:
        print(f"error: {exc}")
        return 2

    summary = {
        "provider": dataset.provider,
        "spx_rows": len(dataset.spx_bars),
        "option_rows": len(dataset.option_quotes),
        "vix_rows": 0 if dataset.vix is None else len(dataset.vix),
        "output_directory": str(artifacts.output_directory),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Provider: {summary['provider']}")
        print(f"Output: {summary['output_directory']}")
        print(
            f"Rows: {summary['spx_rows']} SPX, "
            f"{summary['option_rows']} options, {summary['vix_rows']} VIX"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
