# Data dictionary

Every column of every processed file. **"Known at close of t?"** is the look-ahead
flag: *yes* means the value could have been known when the market closed on date *t*;
*no (outcome)* means it uses data from after *t* and must never be used as a predictor.

Notation: *t* = row date; P = S&P 500 close; r = daily log return; W = 21; A = 252.
"Trading day" = a row of the merged daily panel (a date with both a VIX and an S&P close).

## `data/processed/daily_panel.csv` (also `.parquet`)

One row per trading day, 1990-01-02 to 2026-03-31 (9,125 rows). Index column: `date`.

| Column | Unit | Formula / meaning | Known at close of t? |
|---|---|---|---|
| `date` | date | Trading date *t* (index) | yes |
| `spx_close` | index points | S&P 500 close, Yahoo `^GSPC` | yes |
| `vix_close` | vol points (annualised %) | VIX close, FRED `VIXCLS` | yes |
| `vxo_close` | vol points (annualised %) | VXO close, FRED `VXOCLS`; NaN where missing (ends 2021-09-23) | yes |
| `ret_log` | decimal | ln(P_t / P_{t−1}); P_{t−1} = close on previous panel date (first row: last S&P close before it) | yes |
| `ret_spans_dropped_day` | bool | True if an S&P trading day between the previous panel date and t was dropped by the merge, so `ret_log` covers two trading days (3 rows) | yes |
| `iv_ann_var` | annualised variance (decimal) | (VIX_t / 100)² | yes |
| `rv_past21_ann_var` | annualised variance | (A/W) · Σ r² over t−20 … t (includes t); NaN for the first 20 rows | yes |
| `rv_fwd21_ann_var` | annualised variance | (A/W) · Σ r² over **t+1 … t+21**, non-demeaned | **no (outcome)** |
| `rv_fwd21_ann_vol_pct` | vol points | 100 · √`rv_fwd21_ann_var` | **no (outcome)** |
| `payoff_var` | annualised variance | `iv_ann_var` − `rv_fwd21_ann_var` (short-variance seller's payoff) | **no (outcome)** |
| `payoff_scaled` | fraction of IV | `payoff_var` / `iv_ann_var` = 1 − RV/IV; max +1, no lower bound | **no (outcome)** |
| `vol_spread_pts` | vol points | VIX_t − `rv_fwd21_ann_vol_pct` | **no (outcome)** |
| `spx_fwd21_ret_log` | decimal | ln(P_{t+21} / P_t), same window as the realised variance | **no (outcome)** |
| `fwd_window_start` | date | Date of trading day t+1 (first return in the window) | n/a (window label) |
| `fwd_window_end` | date | Date of trading day t+21 (last return in the window) | n/a (window label) |
| `fwd_window_complete` | bool | True if all 21 forward returns exist; False only for the last 21 rows | n/a (window label) |

Outcome columns are NaN where `fwd_window_complete` is False.

## `data/processed/monthly_sample.csv`

One row per month, January 1990 to December 2025 (432 rows): the **first trading day
of each month** from the daily panel, keeping only complete forward windows. It has
**all daily-panel columns above** plus:

| Column | Unit | Formula / meaning | Known at close of t? |
|---|---|---|---|
| `month` | YYYY-MM | Calendar month of *t* | yes |
| `is_selloff` | bool | `spx_fwd21_ret_log` < −0.05 (log return, ≈ −4.9% simple) | **no (outcome)** |
| `selloff_decile` | integer 1–10 | Decile of `spx_fwd21_ret_log` within the monthly sample; 1 = worst 10% | **no (outcome)**, and uses the full sample |
| `post_2003` | bool | *t* ≥ 2003-09-22 (current VIX methodology live; earlier VIX values are back-calculated) | yes |

## `data/processed/events.csv`

Hand-written event windows (dates only; no statistics). One row per event.

| Column | Meaning |
|---|---|
| `event_id` | Short identifier, e.g. `gfc_2008` |
| `name` | Human-readable name |
| `start_date`, `end_date` | Event window (inclusive), chosen by the team |
| `category` | geopolitical / financial crisis / bear market / market structure / macro / policy / pandemic |
| `description` | One-line summary of what happened |
| `source_note` | Where the dates come from, plus any data caveat |

## `outputs/reports/event_stats.csv`

Computed from the processed data by `events.py`. One row per event.

| Column | Unit | Meaning |
|---|---|---|
| `event_id`, `name`, `start_date`, `end_date` | — | As in `events.csv` |
| `trading_days` | count | Panel rows inside the window |
| `max_vix_close`, `max_vix_date` | vol points, date | Highest VIX close in the window and its date |
| `worst_spx_log_ret`, `worst_spx_ret_date` | decimal, date | Most negative `ret_log` in the window and its date |
| `spx_drawdown` | decimal (negative) | Largest peak-to-trough fall of `spx_close`, from the last close before the window to the window end |
| `drawdown_peak_date`, `drawdown_trough_date` | date | Peak and trough of that drawdown |
| `n_overlapping_months` | count | Monthly rows whose forward window intersects the event window |
| `overlapping_months` | text | `YYYY-MM:payoff_scaled` for each of those rows |
| `min_payoff_scaled_overlap`, `mean_payoff_scaled_overlap` | fraction | Worst and mean `payoff_scaled` across those rows |

## Interim files (`data/interim/`, not committed)

| File | Columns |
|---|---|
| `vix.csv` | `date`, `vix_close` |
| `spx.csv` | `date`, `open`, `high`, `low`, `close`, `volume` |
| `vxo.csv` | `date`, `vxo_close` |
| `usrec.csv` | `date` (first of month), `usrec` (0/1). **Not known in real time**: chart shading only |
