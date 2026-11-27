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

## `data/processed/monthly_variants.csv` (v2, robustness inputs)

Written by `--stage stats` (`stats.build_monthly_variants`). Same 432 monthly dates as
`monthly_sample.csv`. New columns only: the v1 definitions are unchanged. Index column `date`.

| Column | Unit | Formula / meaning | Known at close of t? |
|---|---|---|---|
| `month`, `post_2003`, `iv_ann_var`, `rv_fwd21_ann_var`, `payoff_var`, `payoff_scaled` | — | Copied from `monthly_sample.csv` | as there |
| `rv_fwd22_ann_var` | annualised variance | (A/22) · Σ r² over **t+1 … t+22** (H = 22 ≈ 30 calendar days) | **no (outcome)** |
| `rv_fwd21_dm_ann_var` | annualised variance | A · sample variance (ddof = 1) of r over **t+1 … t+21** (demeaned) | **no (outcome)** |
| `iv_vxo_ann_var` | annualised variance | (VXO_t / 100)²; NaN where VXO is missing (ends 2021-09) | yes |
| `payoff_var_h22`, `payoff_scaled_h22` | as v1 | `iv_ann_var` − `rv_fwd22_ann_var`; ÷ `iv_ann_var` | **no (outcome)** |
| `payoff_var_dm`, `payoff_scaled_dm` | as v1 | `iv_ann_var` − `rv_fwd21_dm_ann_var`; ÷ `iv_ann_var` | **no (outcome)** |
| `payoff_var_vxo`, `payoff_scaled_vxo` | as v1 | `iv_vxo_ann_var` − `rv_fwd21_ann_var`; ÷ `iv_vxo_ann_var` (S&P 100 options vs S&P 500 RV) | **no (outcome)** |

## `outputs/reports/stats_tables.csv` (v2 results)

One row per statistic. Sign convention: implied − realised.

| Column | Meaning |
|---|---|
| `test` | `1_mean`, `2_shape`, `3_crash`, `4_forecast`, `5_robustness` |
| `measure` | Column tested (`payoff_var`, `payoff_scaled`, `vol_spread_pts`; Test 4: `variance`, `vol_pct`, `log_variance`) |
| `sample` | Sample or robustness variant |
| `statistic` | What is estimated (e.g. `mean`, `win_rate`, `skewness`, `slope_b`) |
| `estimate` | Point estimate, in the measure's unit (fractions for rates and shares) |
| `ci_low`, `ci_high`, `ci_method` | 95% confidence interval and how it was computed (blank for descriptive rows) |
| `null_value` | Value under H0 |
| `test_name`, `p_value` | Hypothesis test and its two-sided p-value |
| `n` | Observations used |
| `note` | Extra detail (alternative p-values, caveats) |
| `p_holm_within_measure` | Test 5 only: Holm-adjusted p across the robustness variants of one measure |

`outputs/reports/stats_deciles.csv`: mean payoff per `selloff_decile` (`n`, `mean`, `ci_low`,
`ci_high` (i.i.d. t), `win_rate`, `spx_fwd21_ret_mean`, `measure`). Descriptive; conditions on
an outcome.

`outputs/reports/pitch_claims.csv`: one row per candidate pitch claim (`id`, `test`, `claim`,
`p_max` (largest p behind the claim), `ci_ok`, `p_holm`, `supported`, `reasons`, `evidence`,
`caveat`, `fallback`). Rendered as `pitch_claims.md`.

## `data/processed/rv_calendar_month.csv` (research, realised-vol dynamics)

One row per **calendar month** of the sample (432 rows, 1990-01 to 2025-12), indexed by the
first day of the month. Non-overlapping, unlike the 21-day forward windows of
`monthly_sample.csv`. Describes the past: it is used only to study how realised volatility
behaves over time, never as a predictor of a same-month outcome.

| Column | Unit | Meaning | Known at t? |
|---|---|---|---|
| `date` | date | First calendar day of the month (index) | n/a (label) |
| `n_days` | count | Panel trading days in the month with a return | yes, at month end |
| `rv_cal_ann_var` | annualised variance | (252 / `n_days`) · Σ r² over the month's trading days | yes, at the month's last close |
| `rv_cal_ann_vol_pct` | vol points | 100 · √`rv_cal_ann_var` | yes, at the month's last close |

## Research outputs (`outputs/reports/`)

`research_tables.csv`: same columns as `stats_tables.csv`. `test` is `R_dynamics`
(calendar-month log realised variance: AR(1), ADF, trend, change skewness), `R_conditional`
(seller payoff by **known-at-t** buckets of `vix_close` and of the gap VIX − 100·√`rv_past21_ann_var`)
or `R_time` (linear trend in the payoff), and for step 5 `R_breakeven` (break-even crash
frequency), `R_jensen` (convexity part of VIX − realised vol) and `R_case` (case-study
numbers; `sample` is the month). Conditioning variables are known at the close of t; the
payoffs being described are outcomes. The Jensen forecast of log forward RV uses only
`iv_ann_var` and `rv_past21_ann_var` (known at t) as predictors.

`research_rv_by_year.csv`: one row per calendar year: `n_days`, `rv_ann_vol_pct` (all daily
returns in the year), `vol_of_vol_log` (std of the 12 monthly log realised variances),
`max_month_vol_pct`, `min_month_vol_pct`, `rank_irregular` (1 = highest vol-of-vol). Descriptive.

`research_rolling_premium.csv`: rolling mean of `payoff_scaled` over `window_months` months,
indexed by the window's last month, with a Newey–West 95% band (`ci_low`, `ci_high`). Windows
overlap; chart only. Uses outcomes (payoffs), so it describes the past.

`research_breakeven.csv` (step 5): one row per (`measure`, `threshold`). A "crash" is a month
with payoff ≤ `threshold`; `is_worst` marks the headline scenario (the single worst month,
`worst_month`). All columns describe outcomes (the past); none is known at t.

| Column | Unit | Meaning |
|---|---|---|
| `n_crash`, `n` | count | crash months; all months |
| `crash_mean` | payoff units | L: mean payoff of the crash months (the scenario size, held fixed) |
| `ordinary_mean` (`_lo`, `_hi`) | payoff units | g: mean payoff of the other months, block-bootstrap 95% CI |
| `breakeven_rate` (`_lo`, `_hi`) | crashes per year | 12 · g / (g − L): crash frequency at which the mean payoff is zero |
| `observed_rate` (`_lo`, `_hi`) | crashes per year | `n_crash` / `years`, exact Poisson (Garwood) 95% CI |
| `years` | years | sample length, `n` / 12 |
| `p_conservative` | probability | P(N ≤ `n_crash`) for Poisson crashes at `breakeven_rate_lo` |
| `ratio_breakeven_to_observed` | ratio | `breakeven_rate` / `observed_rate` |

`research_case_paths.csv` (step 5): daily rows around each case-study month (`case`), indexed
by date. `day` = trading days from t (0 = t, the month's first trading day); `spx_close`,
`vix_close`, `ret_log` as in the panel; `spx_rel` = S&P close ÷ close at t;
`accrued_rv_ann_var` = (252 / 21) · Σ r² over days 1…`day` (0 at t, blank before; equals
`rv_fwd21_ann_var` on day 21); `accrued_over_implied` = that ÷ `iv_ann_var` at t (above 1 = the
seller's premium is gone). Rows after t are **outcomes**: a description, never a predictor.

`research_claims.csv`: as `pitch_claims.csv`, for the eleven pre-registered research claims
R-1 … R-11 (own Holm family). Rendered as `research_claims.md`; full tables in
`research_summary.md`.

## Interim files (`data/interim/`, not committed)

| File | Columns |
|---|---|
| `vix.csv` | `date`, `vix_close` |
| `spx.csv` | `date`, `open`, `high`, `low`, `close`, `volume` |
| `vxo.csv` | `date`, `vxo_close` |
| `usrec.csv` | `date` (first of month), `usrec` (0/1). **Not known in real time**: chart shading only |
