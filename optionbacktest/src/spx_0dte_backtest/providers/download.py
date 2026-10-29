"""Canonical file writer for provider datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from spx_0dte_backtest.providers.base import (
    DownloadArtifacts,
    HistoricalDataRequest,
    HistoricalDataset,
)


class ProviderDownloader:
    def write(
        self,
        dataset: HistoricalDataset,
        request: HistoricalDataRequest,
        output_directory: Path | str,
    ) -> DownloadArtifacts:
        output = Path(output_directory)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"download directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        spx_path = output / f"spx_{request.interval_minutes}min.parquet"
        options_path = output / "spx_0dte_options.parquet"
        vix_path = output / "vix.parquet" if dataset.vix is not None else None
        manifest_path = output / "download_manifest.json"
        dataset.spx_bars.to_parquet(spx_path, index=False)
        dataset.option_quotes.to_parquet(options_path, index=False)
        if dataset.vix is not None and vix_path is not None:
            dataset.vix.to_parquet(vix_path, index=False)
        manifest = {
            "provider": dataset.provider,
            "request": {
                "start": request.start.isoformat(),
                "end": request.end.isoformat(),
                "interval_minutes": request.interval_minutes,
                "spx_root": request.spx_root,
                "vix_root": request.vix_root,
                "include_vix": request.include_vix,
            },
            "provider_metadata": dict(dataset.metadata),
            "row_counts": {
                "spx": len(dataset.spx_bars),
                "options": len(dataset.option_quotes),
                "vix": 0 if dataset.vix is None else len(dataset.vix),
            },
            "files": {
                "spx": spx_path.name,
                "options": options_path.name,
                "vix": None if vix_path is None else vix_path.name,
            },
            "sha256": {
                "spx": sha256(spx_path),
                "options": sha256(options_path),
                "vix": None if vix_path is None else sha256(vix_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return DownloadArtifacts(output, spx_path, options_path, vix_path, manifest_path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
