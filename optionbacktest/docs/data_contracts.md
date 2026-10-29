# SPX five-minute bar contract

Required source columns are `timestamp`, `open`, `high`, `low`, and `close`.
Optional columns are `symbol`, `volume`, and `source`.

| Field | Canonical type | Rule |
|---|---|---|
| `timestamp` | timezone-aware datetime | Converted to America/New_York; on configured bar grid |
| `symbol` | string | Defaults to `SPX` |
| `open` | float | Finite and greater than zero |
| `high` | float | At least open, low, and close |
| `low` | float | At most open, high, and close |
| `close` | float | Finite and greater than zero |
| `volume` | nullable float | Finite and non-negative when present |
| `source` | string | Defaults to input filename |

CSV and Parquet are accepted. A timestamp such as `2026-08-12T09:30:00-04:00`
is unambiguous. A naive timestamp is accepted only when configuration declares the
source timezone. Duplicate timestamps and malformed values are reported and rejected.

The example file is synthetic and exists solely to exercise ingestion.

