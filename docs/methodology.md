# Methodology (v1: data layer; v2: statistical tests; research section)

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

## 9. v2: statistical tests (`stats.py`, `stats_report.py`)

Run with `make stats` (`python -m vrp.pipeline --stage stats`); `make all` includes it.
Every parameter is under `stats:` in `config.yaml`. Results: `outputs/reports/stats_summary.md`
(all tables), `stats_tables.csv`, `stats_deciles.csv`, `pitch_claims.md` / `.csv`, and
figures `07`–`11`. Sign convention throughout: **implied − realised**.

### 9.1 Inference toolkit

- **Confidence intervals.** Every estimate that is a test statistic has a 95% CI. Means use
  three methods side by side: i.i.d. t, **Newey–West** (Bartlett kernel, 6 lags monthly, 30 lags
  for the overlapping daily sample; statsmodels' n/(n−1) correction) and a **circular block
  bootstrap** (blocks of 6 consecutive months, 10,000 resamples, fixed seed). Statistics with no
  closed-form SE (median, skewness, kurtosis, ratios, shares, trimmed means) use the bootstrap.
- **Why a block bootstrap.** Monthly payoffs have lag-1 autocorrelation of about 0.14–0.21
  (volatility clusters), so resampling single months would understate uncertainty. Blocks keep
  that dependence.
- **p-values** are two-sided. Bootstrap p = P*(|θ* − θ̂| ≥ |θ̂ − θ0|) with a +1 correction, so
  the smallest reportable value is 1/10,001 ≈ 0.0001 ("≤ 0.0001" in the reports).
- **Multiple testing.** Robustness variants are Holm-adjusted per measure; the pitch claims are
  Holm-adjusted across all candidate claims.

### 9.2 The five tests

1. **Mean (H0: mean = 0)** for `payoff_scaled`, `payoff_var` and `vol_spread_pts`; plus the
   median (sign test, bootstrap CI) and the mean excluding the worst 5 months (bootstrapped: the
   worst months are re-identified in each resample).
2. **Shape.** Win rate vs 50% (exact binomial, Wilson and bootstrap CIs); skewness and excess
   kurtosis (bias-corrected; bootstrap CIs; D'Agostino / Anscombe–Glynn p-values); Jarque–Bera;
   mean − median (a robust lopsidedness measure); average gain vs average loss; months more than
   3 sd below the mean vs the normal expectation (exact binomial); worst month and loss
   concentration (descriptive).
3. **Crash clustering.** Correlation of the payoff with the S&P 500 log return **over the same
   window** (Fisher-z and bootstrap CIs; Spearman), sell-off vs other months (Welch t,
   Mann–Whitney, bootstrap; Fisher exact on loss rates), worst vs best S&P decile (both contain
   large moves, so this isolates direction), and the share of all losses in the worst decile vs
   a 10% no-clustering null (label-permutation test, 10,000 shuffles).
4. **Forecast bias (Mincer–Zarnowitz).** RV_{t→t+21} = a + b·IV_t + e in variance, vol and log
   variance; Newey–West SEs; Wald test of a = 0 and b = 1 jointly.
5. **Robustness.** Mean payoff (NW CI, plus bootstrap CI in the note) for: post-2003 only,
   pre-2003 only, H = 22, demeaned RV, VXO instead of VIX (and the VIX on the same months),
   excluding the worst 5 months, and all overlapping daily windows (NW 30 lags). Also the
   post- minus pre-2003 difference (Welch t, bootstrap).

### 9.3 No look-ahead in v2

- The only **predictor** anywhere is implied variance / the VIX at the close of t (Test 4).
  `stats.require_known_at_t` raises if any outcome column (or any name containing `fwd`) is used
  as a regressor; the tests check this.
- Tests 1, 2 and 5 describe outcomes. Test 3 **conditions on** outcomes (`spx_fwd21_ret_log`,
  `is_selloff`, `selloff_decile`) to describe *when* losses happen; every table, chart and claim
  says it is not a forecast available at t.
- The H = 22 and demeaned variants use returns t+1 … t+H only. `tests/test_stats.py` hand-checks
  them and runs mutation tests: changing prices after t+22 or the VIX after t leaves row t
  unchanged; changing the price before t (day t's own return) leaves them unchanged; changing the
  price at t+1 changes them.
- `selloff_decile` uses the full-sample distribution of forward returns. That is fine for
  description, and it is never used as a signal.

### 9.4 Pitch-claim rule

`pitch_claims.md` lists candidate statements. A claim is **supported** only if, for every piece of
evidence, the 95% CI excludes the null *in the claimed direction*, and the largest p-value behind
it, Holm-adjusted across all candidate claims, is below 0.05. Unsupported claims are listed
with the reason and an honest alternative wording. All numbers in the file are generated.

### 9.5 Decisions & assumptions (v2)

16. **Five tests.** The brief's three questions (positive mean, lopsidedness, sell-off clustering)
    plus robustness became Tests 1, 2, 3 and 5. We added Test 4 (Mincer–Zarnowitz) because it
    asks the project title's question directly: is the VIX a biased forecast?
17. **Block length 6 months** for the bootstrap, matching the 6 Newey–West lags; the
    autocorrelation of the payoffs dies out after 1–2 lags, so 6 is conservative.
18. **Two-sided tests** throughout, even where the hypothesis has a direction (more
    conservative; the claim rule then also checks the direction).
19. **Skewness/kurtosis CIs.** Percentile bootstrap CIs of higher moments are wide and unstable
    when a single month dominates (e.g. March 2020 in raw variance). We report them anyway and
    added mean − median as a robust check; the raw-variance skewness claim fails the rule for
    this reason.
20. **Sell-off = worst decile.** In this sample the −5% `is_selloff` threshold selects exactly the
    44 decile-1 months, so Test 3 reports them once and notes the overlap.
21. **Worst vs best decile bootstrap** resamples blocks within the 88 months of the two deciles
    (not consecutive in time); the Welch t-test is the primary test.
22. **Sharpe ratio** is reported for the scaled payoff (no risk-free rate subtracted: a variance
    swap costs nothing upfront), only to show that it flatters short volatility.
23. **Demeaned RV** uses ddof = 1 and ×252 (not ×252/(H−1)), fixing the primer-noted bug.
24. **VXO** comparisons use S&P 100 implied variance against S&P 500 realised variance, so they
    test the sign of the premium, not its exact size.

## 10. Research section (`research.py`, `research_report.py`)

`make research` (`--stage research`) answers the research questions in
[`research_plan.md`](research_plan.md), where the tests (R-1 … R-9, then the step 5 tests
R-10 and R-11), their nulls and their claimed directions were fixed **before** the results
were read.

### 10.1 What is computed

- **A. Realised-vol dynamics** on a new series, **calendar-month realised variance**
  (`rv_calendar_month.csv`), in logs: AR(1) coefficient with Newey–West SE and half-life (R-1),
  augmented Dickey–Fuller test for mean reversion (R-2), linear time trend with Newey–West SE
  (R-3), skewness of month-to-month changes with the block bootstrap (R-4). Per-year realised
  vol and vol-of-vol ("which years were most irregular?") are descriptive.
- **B. Premium by conditions known at t**: mean `payoff_var` and `payoff_scaled` in fixed
  buckets of the VIX level and of the gap VIX − past 21-day realised vol; high − low
  differences by block bootstrap (R-5 … R-8).
- **C. Time variation**: linear trend in `payoff_scaled` with Newey–West SE (R-9), and a
  60-month rolling mean (chart only).
- **D. Is it *too* pessimistic? (step 5)**
  - *Break-even crash frequency* (R-10 scaled, R-11 raw variance). Take a crash the size of the
    worst month, L, and let the other months earn g on average. If a share f of months were
    such crashes the mean payoff would be (1 − f)·g + f·L, which is zero at f\* = g / (g − L).
    We report f\* as "once every N\* years" next to the observed frequency (k crashes in 36
    years, exact Poisson CI). The CI on f\* comes from block-bootstrapping g with L held fixed.
    The claim's p-value is P(N ≤ k) for Poisson crashes arriving at the **lower** CI bound of
    the break-even rate, a deliberately conservative number. A curve over fixed crash
    thresholds (`config.yaml`) is chart-only (figure 16).
  - *Jensen gap* (descriptive). Unconditional √E[RV] − E[√RV] as in the plan, which is an upper
    bound because it also counts vol moving between months. Conditional version: regress log
    forward RV on log VIX² and log past RV; with residuals e, E_t[√RV] / √E_t[RV] =
    mean(e^{e/2}) / √mean(e^e) (Duan smearing, no normality assumed). If VIX² were a fair
    forecast, mean(VIX)·(1 − ratio) of the spread VIX − realised vol would be convexity; we
    also report the share of months such a fair VIX would still exceed realised vol,
    P(e < log mean e^e).
  - *Case studies* (descriptive): 2020-03 and 2025-04 day by day (figure 17).

Outputs: `research_tables.csv`, `research_summary.md`, `research_claims.md`/`.csv`,
`research_rv_by_year.csv`, `research_rolling_premium.csv`, `research_breakeven.csv`,
`research_case_paths.csv`; figures 12–17.

### 10.2 No look-ahead in the research section

Section B conditions on `vix_close` and `rv_past21_ann_var` only, both known at the close of t
(enforced by `require_known_at_t`), with thresholds fixed in `config.yaml` rather than sample
quantiles, which would use the whole sample. Results are **in-sample descriptions**, not an
out-of-sample trading rule. Sections A and C describe the past and make no forecast. In
section D the Jensen forecast uses only `iv_ann_var` and `rv_past21_ann_var` (guarded by
`require_known_at_t`); the break-even test and the case studies describe outcomes.

### 10.3 Decisions & assumptions (research)

25. **Calendar-month RV for dynamics.** The 21-day forward windows of the monthly sample overlap
    or leave gaps between consecutive months, which would distort autocorrelations. Calendar
    months do not overlap; their length varies (15–23 trading days), so each month's sum of
    squared returns is annualised by its own day count.
26. **Separate Holm family.** The research claims are corrected among themselves (11 claims
    since step 5), not added to the 23 v2 claims, so the v2 verdicts are unchanged. A reader
    who wants one family of all 34 claims can apply Bonferroni (raw p × 34), which is stricter
    than Holm; the three supported research claims (R-1, R-2, R-10) still pass at that level
    (raw p ≤ 0.0001).
27. **Trend tests are linear.** A linear trend through spiky, persistent data is a blunt tool;
    R-3 uses 12 Newey–West lags because log RV is persistent, and the ADF result (R-2) shows
    the series is mean-reverting, so a trend regression is not spurious.
28. **Bucket means use Newey–West within the bucket**, although the bucket's months are not
    consecutive; treat those CIs as approximate. The pre-registered tests are the high − low
    differences, which use the block bootstrap on the full series.
29. **Exploratory look.** Bucket means by VIX tercile were looked at once before the
    thresholds were fixed; the directions in R-5 and R-6 are the ones expected from
    economics (see `research_plan.md`, "Changes").
30. **Structural-break test dropped** (26 Nov 2026) for time; the pre/post-2003 comparison
    (v2) and the linear trend (R-9) remain.
31. **Break-even holds the crash size fixed.** L is the scenario ("a March-2020-sized loss"),
    not an estimate, so only the ordinary-month mean g is resampled. Resampling L as well
    would give resamples without the crash month, where the scenario is undefined.
32. **Break-even is a reframing of the mean, plus one new question.** Because the mean payoff
    equals (1 − f)·g + f·L, the break-even frequency exceeds the observed one exactly when the
    mean is positive. What the test adds is the Poisson question: with only k crashes seen,
    how likely is it that they really come as often as break-even requires? The p-value takes
    the low end of the break-even rate's CI, so it is conservative.
33. **Both payoff measures are claims** (R-10 scaled, R-11 raw variance), following the rule of
    reporting both. Adding them to the research Holm family can only make R-1 … R-9 harder to
    pass (R-4's Holm p rose from 0.056 to 0.064; no verdict changed).
34. **Jensen: the conditional estimate is an upper estimate of convexity.** A forecast with only
    two inputs leaves more residual variance than the market's true conditional variance, and
    more variance means more convexity. It also assumes the residual distribution is the same
    every month. The unconditional gap is reported because the plan named it, labelled as an
    upper bound.
35. **Case-study VIX percentile** ("higher than 95% of months") ranks the VIX at t against the
    whole 1990–2025 sample, so it is a description, not something known at t.

