# Historical-data provider adapters

The provider layer separates vendor retrieval and response parsing from the
canonical loaders, validation, backtest, and report components.

## ThetaData

`ThetaDataAdapter` is the first network provider. It talks to a locally running
Theta Terminal using the documented v2 REST API:

- SPX five-minute OHLC: `/hist/index/ohlc`
- same-day full option chain: `/bulk_hist/option/all_greeks`

Official references:

- [ThetaData index OHLC](https://http-docs.thetadata.us/operations/get-v2-hist-index-ohlc.html)
- [ThetaData bulk all Greeks](https://http-docs.thetadata.us/operations/get-bulk_hist-option-all_greeks.html)
- [ThetaData pagination](https://http-docs.thetadata.us/articles/pagination.html)

The terminal defaults to `http://127.0.0.1:25510/v2`. The adapter requests
300,000 ms intervals for the default five-minute contract, regular trading hours
only, and one option request per weekday with expiration equal to that date.

ThetaData strike integers are divided by 1,000. Milliseconds from midnight ET
are localized with `America/New_York`, including DST. Response parsing uses the
returned `header.format` rather than hard-coded array offsets, and follows the
`Next-Page` header without resending initial query parameters.

The option endpoint supplies bid, ask, delta, gamma, theta, vega, and implied
volatility. ThetaData calculates these Greeks historically; they are labeled
`THETADATA_HISTORICAL_CALCULATION`, not treated as exchange-observed Greeks.
Fields absent from this endpoint (`last`, volume, and open interest) remain null.
No current values are substituted.

The bulk all-Greeks endpoint requires the appropriate ThetaData subscription.
Provider errors, including permission errors, fail explicitly instead of being
treated as empty market days.

## Download command

Start Theta Terminal, then run:

```bash
uv run spx-0dte download-data \
  --provider thetadata \
  --start 2024-01-02 \
  --end 2024-01-31 \
  --output data/thetadata_2024_01 \
  --include-vix \
  --json
```

The output contains canonical Parquet files and `download_manifest.json` with
the request, provider metadata, row counts, and SHA-256 hashes. A non-empty
destination is never overwritten. The downloaded files pass through the same
SPX and option validation loaders used by manually supplied files.

## Cboe and licensed OPRA extension path

No Cboe or OPRA endpoint or schema is assumed in the current implementation.
Access methods and licensed file layouts can differ, so silently guessing column
mappings would be unsafe.

To add one, implement `HistoricalDataProvider.download()` and return a
`HistoricalDataset` containing the canonical columns. Register the adapter with
`ProviderRegistry`. Network adapters can reuse the injected `JsonTransport` or
provide a CSV transport; downloaded CSV files can also be normalized through the
existing `SpxBarLoader` and `OptionChainLoader` once their explicit vendor column
mapping is defined.

The required option contract remains:

```text
timestamp, expiration, strike, option_type, bid, ask, last,
volume, open_interest, delta, gamma, theta, vega,
implied_volatility, contract_symbol, source
```

An OPRA or Cboe adapter must preserve vendor provenance, timezone semantics, and
historical Greeks availability. It must not derive unavailable historical fields
without an explicit, documented model and label.
