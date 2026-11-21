# Data folder

Data moves through three stages. Everything here is rebuilt by `make data`
(`python -m vrp.pipeline --stage data`).

| Folder | What's in it | Committed? |
|---|---|---|
| `raw/` | Untouched downloads (`vix.csv`, `spx.csv`, `vxo.csv`, `usrec.csv`), each with a `.meta.json` sidecar: source, download time (UTC), library versions, row count, date range and SHA-256 | **No** (gitignored) |
| `interim/` | One cleaned series per source: dates parsed, sorted, de-duplicated, missing values dropped (never forward-filled) | **No** (gitignored) |
| `processed/` | Analysis-ready files: `daily_panel.csv`/`.parquet`, `monthly_sample.csv`, `events.csv` | **Yes** (small, needed by teammates) |

Why raw data is not committed: Yahoo Finance's terms do not allow redistribution, so each
person downloads it locally. Downloads are cached: `make data` will not re-download if a raw
file exists. Use `make refresh` to force fresh downloads.

To check you have the same inputs as everyone else, compare the `sha256` values in
`raw/*.meta.json` with `outputs/reports/manifest.json`.

Column definitions: `docs/data_dictionary.md`. Sources and caveats: `docs/data_sources.md`.
