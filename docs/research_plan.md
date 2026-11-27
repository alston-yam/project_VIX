# Research section: storyline and pre-registered tests

Owner: research workstream (branch `research`). Pitch: **Wednesday 2 December 2026**.
This file fixes, *before* the results are read, what the research section will show and
which tests it will run, so nothing on a slide looks cherry-picked. Change it only by adding
a dated note under "Changes" at the bottom.

**Sign convention: implied − realised** (positive = the VIX over-forecast; the variance
seller gains). Variances are annualised decimals; vol is in percentage points.

## 1. The story in one line

> The VIX is pessimistic most of the time and the seller is paid for it, but the payment is
> compensation for rare, violent crashes. Whether it is *too* pessimistic depends on whether
> the premium is larger than those crashes justify (the break-even test).

## 2. Slide map

Slide titles stay as **questions** until the numbers come out of our own code (CLAUDE.md
rule 7). Each slide has one chart and about three bullets; every number comes from
`outputs/reports/pitch_claims.md` or `outputs/reports/research_claims.md`.

| # | Slide (working title) | Main chart | Evidence |
|---|---|---|---|
| I1 | What is the VIX, and what is the variance risk premium? | schematic + `02_vix_vs_realised_vol` | primer; Jensen note (step 5) |
| I2 | Data and validation | sanity table | `sanity_summary.md` |
| R1 | Does the VIX over-forecast on average? | `07_test1_mean_ci` | C1, C4, C5 |
| R2 | How bad are the losing months, and when do they happen? | `08_test2_distribution` / `09_test3_selloff` | C6, C8, C15 |
| R3 | Case study: March 2020 vs April 2025 | case-study chart (step 5) | step 5 |
| R4 | How does realised volatility behave? | `12_rv_dynamics`, `13_rv_by_year` | R-tests 1–4 |
| R5 | When is the VIX most pessimistic? (known at t) | `14_conditional_premium` | R-tests 5–8 |
| R6 | Is it changing, and is it *too* pessimistic? | `15_rolling_premium` + break-even (step 5) | R-test 9, C19–C21, break-even |

## 3. Pre-registered research tests (`--stage research`)

All tests use the 432-month sample (1990-01 to 2025-12) unless stated. CIs are 95%.
Monthly inference uses Newey–West or the 6-month circular block bootstrap from `stats:`.
The research claims form **their own Holm family**, separate from the 23 v2 claims, so adding
them does not change any v2 result.

### A. Realised-volatility dynamics (slide R4)

Series: **calendar-month realised variance**, `rv_cal_ann_var` = (252 / n_days) · Σ r² over
the trading days of each calendar month (a new, non-overlapping series; see
`docs/data_dictionary.md`). Analysed in logs.

| ID | Question | Test | H0 | Claim direction |
|---|---|---|---|---|
| R-1 | Does vol cluster (high months follow high months)? | AR(1) of log RV, HAC SE | ρ = 0 | ρ > 0; report half-life ln 0.5 / ln ρ |
| R-2 | Do vol shocks fade (mean reversion)? | Augmented Dickey–Fuller on log RV (AIC lags, constant) | unit root | reject |
| R-3 | Has realised vol risen gradually over time? | OLS of log RV on time (years), HAC (12 lags) | slope = 0 | slope > 0 (your hypothesis); fallback "no evidence of a trend" |
| R-4 | Does vol jump up faster than it falls? | skewness of monthly Δ log RV, block bootstrap | skew = 0 | skew > 0 |

Descriptive (no test): realised vol and **vol-of-vol** (std of the 12 monthly log RVs) for each
calendar year, ranked, answering "which years were most irregular?"

### B. Premium by conditions known at t (slide R5)

Conditioning variables are **known at the close of t** (checked with `require_known_at_t`):
the VIX level, and the gap between the VIX and the past 21-day realised vol
(`vix_close − 100·√rv_past21_ann_var`, vol points). Buckets use **fixed thresholds from
`config.yaml`**, not sample quantiles, so the breakpoints are not themselves look-ahead:
VIX ≤ 15 / 15–25 / > 25; gap ≤ 0 / 0–5 / > 5 vol points.

| ID | Question | Statistic | H0 | Claim direction |
|---|---|---|---|---|
| R-5 | Is the premium larger in raw variance when the VIX starts high? | mean `payoff_var`, high − low VIX bucket | diff = 0 | > 0 |
| R-6 | Is it smaller as a share of implied variance when the VIX starts high? | mean `payoff_scaled`, high − low VIX bucket | diff = 0 | < 0 |
| R-7 | Does the seller still win on average when the VIX starts high? | mean `payoff_scaled` in the high-VIX bucket | 0 | > 0 |
| R-8 | Is the premium larger when the VIX sits well above recent realised vol? | mean `payoff_scaled`, wide − inverted gap bucket | diff = 0 | > 0 |

Differences use the block bootstrap (CI and p); bucket means use Newey–West.
These are **in-sample descriptions**, not a trading rule; the forecasting model and the
backtest (other workstreams) test whether any of it works out of sample.

### C. Is the premium changing? (slide R6)

| ID | Question | Test | H0 | Claim direction |
|---|---|---|---|---|
| R-9 | Has the scaled premium shrunk over time? | OLS of `payoff_scaled` on time (years), HAC (6 lags) | slope = 0 | slope < 0; fallback "no evidence it is shrinking" |

Descriptive: 60-month rolling mean of `payoff_scaled` with a Newey–West band (chart only;
overlapping windows, so the band is not a test). The structural-break test was **dropped**
(26 Nov 2026) to save time; the pre/post-2003 comparison (v2 C21) remains.

### D. Step 5 (done 27 Nov): break-even, Jensen, case studies

Exact definitions: "Changes", 27 Nov. Claims R-10 (scaled) and R-11 (raw variance); the Jensen
gap and the case studies are descriptive. Outputs: `research_breakeven.csv`,
`research_case_paths.csv`, figures 16–17, section D of `research_summary.md`.

Defined before running: (i) **break-even crash frequency**, i.e. how often a loss the size of the
worst month would have to occur for the mean scaled payoff to be zero, against how often it
actually occurred; (ii) the Jensen gap, √E[RV] vs E[√RV], to show how much of "VIX >
realised vol" is convexity rather than premium; (iii) case studies of 2020-03 and 2025-04 (VIX
and past RV at t, the daily path through the window, payoff).

## 4. Order of work

| Day | Task | Who |
|---|---|---|
| Thu 26 | 1. This plan. 2. Teammate brief (`docs/team_brief.md`). 3–4. Sections A–C above | Claude + you |
| Fri 27 | Review A–C results; fix anything; start step 5 | Claude + you |
| Sat 28 | 5. Break-even, Jensen, case studies. 6. Claims and docs. 7. **You decide the verdict wording** | Claude, then you |
| Sun 29 | 8. Deck charts (`--stage deck`). 9. Slide text and speaker notes | Claude drafts, you edit |
| Mon 30 | 10. You build our slides in the OAF template. 11. Teammates' cutoff; Claude checks their numbers | you + Claude |
| Tue 1 | 12. Full deck; fallback conclusion and appendix. 13. Q&A sheet and rehearsal | you + Claude |

## Changes

- 26 Nov 2026: plan written; structural-break test dropped from section C.
- 26 Nov 2026: a quick exploratory look at bucket means (VIX terciles) was taken before the
  thresholds above were fixed. The directions in R-5 and R-6 are the economically expected
  ones (high VIX ⇒ more variance to sell, but a crash is more likely), stated here for honesty.
- 27 Nov 2026 (written before the step 5 code was run): exact definitions for section D.
  **Break-even (R-10 scaled, R-11 raw variance).** Scenario: a loss the size of the worst
  month, L. Ordinary months (all others) earn g on average. If a share f of months were such
  crashes, the mean payoff is (1 − f)·g + f·L, so it is zero at f\* = g / (g − L). We report
  f\* as "a crash this size once every N\* years" and compare it with the observed frequency
  (k crashes in T = 36 years, exact Poisson CI). Uncertainty in g: 6-month block bootstrap,
  with L held fixed (it is the scenario). Claim: N\* is below the observed years between
  crashes and its 95% CI excludes it; p-value = exact Poisson P(N ≤ k) at the
  *conservative* end of the break-even rate (the lower bound of its bootstrap CI), i.e.
  "even if the premium is at the low end of its CI and crashes really are that frequent, how
  likely is seeing only k?". Both measures are claims (rule: report raw and scaled), and they
  join the research Holm family (now 11 claims), which can only make R-1 … R-9 harder to pass.
  A chart-only curve repeats this for fixed crash thresholds from `config.yaml`. Caveat
  stated up front: by construction f\* exceeds the observed share whenever the mean payoff is
  positive; what the test adds is the Poisson question "how unlucky would we need to have
  been?", not new evidence on the mean.
  **Jensen gap (descriptive, no claim).** (a) Unconditional √E[RV] − E[√RV] in vol points
  (plan wording; an upper bound, because it also counts vol moving between months, which the
  VIX tracks). (b) Conditional: regress log forward RV on log VIX² and log past RV (both
  known at t); with the residuals e, E_t[√RV] / √E_t[RV] = mean(e^{e/2}) / √mean(e^e)
  (smearing, no normality assumed). If VIX² were a fair forecast, the convexity part of
  "VIX − realised vol" is mean(VIX)·(1 − that ratio); we report it as a share of the mean
  spread, and the share of months a fair VIX would still exceed realised vol
  (P(e < log mean e^e)) against the actual 83.8%. Block bootstrap CIs.
  **Case studies (descriptive).** 2020-03 and 2025-04: VIX and past RV at t, realised vol,
  payoffs and rank, days with |r| above 3× the VIX-implied daily s.d., the first day the
  realised variance accrued in the window exceeded the implied variance sold, the window's
  S&P drawdown, months of average premium lost, and months until the cumulative scaled
  payoff from the crash month was back above zero.
