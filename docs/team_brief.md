# Team brief: model and strategy workstreams

**Pitch: Wednesday 2 December. Content cutoff: Monday 30 November, 22:00.** Send your
slides' charts, numbers and two-line takeaways to the deck owner by then. After the cutoff
only fixes go in.

## 1. The shared story

The research section shows that the VIX over-forecasts next month's variance in about 84% of
months, but the seller's losses come in rare crashes. Your slides answer the follow-up:
**can we predict when the premium is thin, and does a strategy that steps aside in those
months beat always selling?** Your slides come after the research section, so assume the
audience already knows what the VIX and the variance risk premium are.

## 2. Rules every number must follow

1. **No look-ahead.** A decision for month t may use only data up to the close of t. Columns
   whose names contain `fwd`, plus `payoff_*`, `vol_spread_pts`, `is_selloff` and
   `selloff_decile`, are **outcomes**: never inputs. Fit models on an expanding window and
   report out-of-sample results (suggested start: January 2004).
2. **Sign convention: implied − realised.** Positive means the seller wins. State it on every
   chart.
3. **Units.** Variance as an annualised decimal (0.04 = 20% vol); vol in percentage points.
   Label which one a chart shows.
4. **Report costs.** Show results with and without transaction costs, and say what you
   assumed.
5. **Keep a baseline.** Always show the plain "sell every month" strategy next to yours,
   with Sharpe, max drawdown, worst month and return skewness.
6. **No numbers from memory.** Every figure on a slide comes from your code or a cited source.

## 3. Data

Clone the repo and use the committed processed files; no download is needed:

```python
from vrp.config import load_config
from vrp.pipeline import read_processed
cfg = load_config()
monthly = read_processed(cfg, "monthly_sample")   # 432 months, 1990-01 to 2025-12
panel = read_processed(cfg, "daily_panel")        # 9,125 trading days
```

Useful known-at-t columns: `vix_close`, `iv_ann_var`, `rv_past21_ann_var`, `spx_close`,
`post_2003`. The outcome you are forecasting or trading is `rv_fwd21_ann_var` (realised
variance over the next 21 trading days) and the seller's payoff `payoff_scaled` / `payoff_var`.
Column meanings: `docs/data_dictionary.md`. Concepts and pitfalls: `docs/primer.md`.

## 4. Handoff between the model and strategy pairs

Agree this on day one so the strategy pair is never blocked:

- **File:** `data/processed/rv_forecast.csv`, one row per monthly-sample date (`date` index).
- **Columns:** `rv_hat_ann_var` (forecast of next-21-day realised variance, made at the close
  of t, annualised decimal) and `vrp_hat_ann_var` = `iv_ann_var − rv_hat_ann_var`. These are
  **known at t**, so they deliberately do not contain `fwd`.
- **Out-of-sample only:** each row's forecast must come from a model fitted on data before t.
- **Placeholder until the model is ready:** `rv_hat_ann_var = rv_past21_ann_var` (last
  month's realised variance). The strategy pair builds the backtester on this and swaps in
  the real forecast later.

## 5. What we need from you (by the cutoff)

| Workstream | Slides | Must include |
|---|---|---|
| Forecast model | 2–3 | model and inputs; out-of-sample accuracy against the two naive benchmarks ("RV = VIX²" and "RV = last month's RV"); one chart of forecast vs realised |
| Strategy and backtest | 3–4 | the instrument and why; the rules; baseline vs enhanced table (return, Sharpe, max drawdown, worst month, skew), with and without costs; equity curve with crash months shaded |

Charts: PNG, 16:9, readable at slide size; title, axis labels with units, source footnote.
Code goes on a branch in the repo with a PR, so the numbers can be checked before Wednesday.
