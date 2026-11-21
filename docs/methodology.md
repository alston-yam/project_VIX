# Methodology (v1: data layer)

This document explains, step by step, how raw downloads become the analysis-ready
files in `data/processed/`. Read `docs/primer.md` first for the concepts. Every
step below is in `src/vrp/` and every parameter is in `config.yaml`.

Numbers quoted here are from the v1 build (data downloaded 21 Nov 2026). The
pipeline regenerates the exact figures in `outputs/reports/` on every run.

## 1. Pipeline at a glance

```
download.py ─► data/raw/*.csv (+ .meta.json)          untouched downloads
clean.py    ─► data/interim/{vix,spx,vxo,usrec}.csv   one clean series per source
clean.py    ─► merged VIX + S&P panel (inner join)
features.py ─► data/processed/daily_panel.{csv,parquet}
sampling.py ─► data/processed/monthly_sample.csv
events.py   ─► data/processed/events.csv, outputs/reports/event_stats.csv, docs/events.md
validate.py ─► outputs/reports/{data_quality_report,sanity_summary}.md
pipeline.py ─► outputs/reports/manifest.json, and runs everything above
plots.py    ─► outputs/figures/*.png|svg
```

## 2. Cleaning (`clean.py`)

Applied to each source, with every action logged and counted in
`outputs/reports/data_quality_report.md`:

1. **Dates.** Parse to a timezone-naive, midnight-normalised `DatetimeIndex`, sort, and drop
   exact duplicate dates (keeping the first). No duplicates were found in v1.
2. **Date window.** Keep 1989-11-01 to 2026-03-31.
3. **Missing values.** FRED writes missing as `"."` or a blank; both become NaN and the
   row is **dropped**. Every dropped date is listed in the quality report.
4. **Never forward-fill.** A missing day is removed, never invented. Filling a missing
   VIX with yesterday's value would fake a quiet day.
5. **Range checks (flag, don't remove).** Values must be positive; VIX should lie in
   [5, 100]; daily |log return| above 10% is flagged with its date. v1 flagged three days
   (13 Oct 2008, 28 Oct 2008, 16 Mar 2020). These are real crash days and they are the point of
   the study, so they stay.
6. **Calendar gaps.** Gaps of more than 4 calendar days between consecutive dates are listed:
   9/11 (10 → 17 Sep 2001), the Ford day of mourning (29 Dec 2006 → 3 Jan 2007) and Hurricane
   Sandy (26 → 31 Oct 2012).
7. **Merge.** Inner join of VIX and S&P on dates where **both** exist (9,125 rows,
   1990-01-02 to 2026-03-31).
   - 3 S&P trading days have no VIX value (1991-03-01, 1997-01-31, 1997-11-26) and are dropped.
   - 30 VIX dates have no S&P close (holidays/closures where FRED still lists a VIX value,
     mostly after 2022) and are dropped.

## 3. Returns and variances (`features.py`)

All formulas use **log returns** and **annualised variance** (decimal, so 0.04 = 20% vol²).

| Quantity | Formula |
|---|---|
| Daily log return | r_t = ln(P_t / P_{t−1}) |
| Implied variance | IV_t = (VIX_t / 100)² |
| Forward realised variance | RV_t = (252 / 21) · Σ_{k=1..21} r²_{t+k} |
| Seller's payoff | payoff_var_t = IV_t − RV_t |
| Scaled payoff | payoff_scaled_t = payoff_var_t / IV_t = 1 − RV_t / IV_t |

- **Return definition.** P_{t−1} is the S&P close on the previous **panel** date. Because
  the panel is an inner join, if an S&P day was dropped (3 cases) the return spans two
  trading days; this keeps every price move in the data instead of losing one. Those rows
  are flagged by `ret_spans_dropped_day`. For the very first panel date (1990-01-02),
  P_{t−1} is the last S&P close before it (1989-12-29), so the first return exists.
- **Non-demeaned.** Realised variance sums raw squared returns (no mean subtracted), the
  standard variance-swap convention. A demeaned version is a v2 robustness check.
- **Annualisation.** ×252/21 turns a 21-day sum into an annual rate, matching VIX², which
  is also annualised.
- **Sign convention.** Implied − realised (Bollerslev–Tauchen–Zhou style). Positive means
  the variance **seller** made money. Carr & Wu use the opposite sign.
- **Horizon mismatch.** The VIX covers 30 *calendar* days; we use 21 *trading* days. Both are
  annualised, so they are comparable. H = 22 is a v2 robustness check.

## 4. No look-ahead: the argument in plain English

On each trading day *t*:

1. We **read the VIX at the close of day t**. That is the market's forecast, and it was public
   at 4:15 pm that day.
2. We measure what actually happened using **returns from day t+1 to day t+21**. The first of
   those returns is the move from t's close to t+1's close, which happens **entirely after**
   the VIX was observed. Day t's own return (from t−1 to t) is **not** in the window.
3. Anything that uses the future is an **outcome**: it is named with `fwd` (or is listed as an
   outcome in `docs/data_dictionary.md`: `payoff_*`, `vol_spread_pts`). Outcomes are things
   we *explain*; they are never used as predictors. Columns with `past` or no prefix
   (`vix_close`, `ret_log`, `iv_ann_var`, `rv_past21_ann_var`) are known at the close of t.

The tests in `tests/test_alignment.py` enforce this mechanically:

- every complete row has `fwd_window_start` > t and `fwd_window_end` exactly 21 panel rows after t;
- `rv_fwd21_ann_var` equals a hand computation for 20 random rows (tolerance 1e-12), on
  synthetic and real data;
- `rv_past21_ann_var` equals a hand computation using only dates ≤ t;
- **mutation tests:** changing every price and VIX value *after* t+21 leaves row t
  unchanged; changing the price at t+1 changes `rv_fwd21_ann_var` but not `iv_ann_var` or
  `rv_past21_ann_var`; changing the price at t−1 changes `ret_log` at t but no outcome at t;
- no monthly row's window extends past the last available date.

As a check that the tests have teeth, we deliberately shifted the forward window by one day
(so it included day t). Three tests failed immediately.

## 5. Monthly sample (`sampling.py`)

- Take the **first trading day of each month** in the daily panel, from January 1990 to
  December 2025, and keep rows with `fwd_window_complete == True`.
- Result: **432 rows** (36 years × 12), none dropped for incomplete windows.
- Extra columns: `month` (YYYY-MM), `is_selloff` (`spx_fwd21_ret_log` < −0.05),
  `selloff_decile` (decile of `spx_fwd21_ret_log` across the 432 months; 1 = worst 10%;
  ties broken by date order), and `post_2003` (date ≥ 2003-09-22, the live-VIX era: 267 rows).

### Window overlap between consecutive months

Months have 19–23 trading days but our window is always 21, so consecutive windows do not tile
perfectly. We measured it (`overlap_days` = trading days shared by month m's window and month
m+1's window; negative = days covered by neither):

| overlap_days | pairs of consecutive months |
|---|---|
| −2 | 43 |
| −1 | 105 |
| 0 | 141 |
| +1 | 89 |
| +2 | 51 |
| +3 | 1 (Nov → Dec 1997: dropped VIX day 26 Nov plus Thanksgiving) |
| +6 | 1 (Sep → Oct 2001: four days of 9/11 closure push September's window to 9 Oct) |

Across 431 pairs: 200 shared trading days in total, 191 uncovered days, mean overlap +0.02
days. The overlap is small, but it is non-zero, so v2 will report **Newey–West** standard
errors (a few lags) alongside plain t-stats.

**Attribution caveat:** with first-of-month sampling a crash is credited to whichever
month's window contains it. For example, the window opened on 2 Sep 2008 runs from 3 Sep to
1 Oct 2008 and includes the Lehman failure and the 29 Sep 2008 crash, so September 2008 (not October) carries the biggest 2008
loss in the monthly file. See `docs/events.md`.

## 6. Events (`events.py`)

We define event **windows** by hand (dates only, `data/processed/events.csv`). Every number
about an event (max VIX close, worst S&P day, drawdown, overlapping monthly payoffs) is
**computed from our data** and written to `outputs/reports/event_stats.csv` and
`docs/events.md`. The drawdown starts from the last close before the window, so a first-day
crash (e.g. 17 Sep 2001) counts.

## 7. Validation (`validate.py`)

Hard checks stop the pipeline (exit code 1) if they fail: unique sorted index, no NaN in core
columns, forward windows complete everywhere except the last 21 rows, non-negative realised
variance, `payoff_scaled` ≤ 1, forward window strictly after t, one monthly row per month and
no window past the data. Soft checks warn loudly but do not stop the run: highest VIX close
= 82.69 on 2020-03-16, monthly row count = 432, VIX within [5, 100]. All checks passed in v1.

`outputs/reports/sanity_summary.md` compares our numbers with published benchmarks, without
adjusting anything to match.

## 8. Decisions & assumptions

Where the brief was ambiguous we chose the most transparent, auditable option:

1. **Repository root.** The brief's tree shows a `vrp-sp500/` folder; this repository *is*
   that folder (its root holds `README.md`, `config.yaml`, …).
2. **Returns across dropped days.** Returns are computed between consecutive panel dates, so
   3 returns span two S&P trading days. The alternative (S&P-calendar returns, then join)
   would silently omit one day's move from some windows. Both choices affect at most 3 of
   9,125 rows; the affected rows are flagged.
3. **First return.** Seeded from the last S&P close before the first VIX date, so the panel
   has no leading NaN return.
4. **"Trading day" means panel row.** The forward window is the next 21 rows of the merged
   panel (days on which both VIX and S&P exist).
5. **`fwd_window_complete`** is True only if all 21 forward returns exist; only the final
   21 panel rows (March 2026) are incomplete, which is why the download runs to 31 Mar 2026.
6. **`selloff_decile`** is computed within the 432-row monthly sample (not the daily panel),
   with ties broken by date (`rank(method="first")`).
7. **`is_selloff` threshold** of −5% applies to the **log** forward return (≈ −4.9% simple).
8. **VIX range check** [5, 100] only flags; it never removes data.
9. **Holiday VIX values** after 2022 are dropped by the inner join rather than treated as
   errors.
10. **USREC** is used only for shading charts, because it was not known in real time.
11. **VXO** is kept as a nullable `vxo_close` column in the panel for robustness work; it is
    not used in any v1 calculation.
12. **Benchmark tolerance.** ±1 vol point for "≈ 19.6" and "≈ 15.5", because the CFA figures
    may use different sampling or windows. Our means (19.54 and 15.34) are well inside.
13. **Processed data is committed** (small, needed by teammates); raw and interim data are not
    (Yahoo terms; rebuilt by `make data`).
14. **Extra file.** `pyproject.toml` holds the black/ruff/pytest settings (not in the brief's
    tree); `tests/conftest.py` holds shared test fixtures.
15. **Parquet.** The daily panel is written as both CSV (human-readable) and Parquet (exact
    dtypes); the tests read the Parquet file.
