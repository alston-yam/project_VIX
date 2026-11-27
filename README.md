# S&P 500 Variance Risk Premium: Is the VIX Too Pessimistic?

Oxford Alpha Fund Quant Bootcamp team project. **v1** builds clean, documented,
look-ahead-safe datasets and supporting charts. **v2** runs five statistical tests on them,
with 95% confidence intervals for every statistic, draws one figure per test (07–11) and
writes a log of the claims the data does and does not support for the pitch
(`outputs/reports/pitch_claims.md`).

## Project question

Each month from 1990 to 2025 we compare the variance the **VIX forecast** with the variance
the **S&P 500 then delivered**. v2 tests:

1. whether the gap is reliably positive (i.i.d. t, Newey–West, block bootstrap);
2. how lopsided the payoff is (win rate, skew, kurtosis, worst loss vs average gain);
3. whether losses cluster in market sell-offs (crash-risk compensation);
4. whether the VIX is an unbiased forecast of realised variance (Mincer–Zarnowitz);
5. whether the result survives robustness checks (post-2003, H = 22, demeaned RV, VXO,
   overlapping daily windows, dropping the worst months).

Background reading: [`docs/primer.md`](docs/primer.md).

**Contributing (people or AI assistants):** read [`CLAUDE.md`](CLAUDE.md) first. It covers
the architecture, the no-look-ahead and naming rules, conventions, and exactly how to add v2
statistics or v3 deck charts so they fit the framework. Claude Code loads it automatically.

## Quickstart

Requires Python 3.11 and `make`.

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && make all
```

`make all` downloads the data (cached after the first run), cleans it, builds every dataset
and report, draws the figures and runs the tests. Other targets:

| Command | Does |
|---|---|
| `make data` | download (cached) → clean → features → monthly sample → events → validation reports |
| `make refresh` | same as `make data`, but re-downloads everything |
| `make figures` | redraw the figures from processed data |
| `make stats` | v2: run the five tests → reports, pitch claims, figures 07–11 |
| `make test` | run the test suite |
| `make lint` / `make format` | check / apply black + ruff |
| `make clean` | delete raw/interim data, logs and caches (keeps committed outputs) |

## Repo structure

```
├── README.md                 this file
├── CLAUDE.md                 contributor / AI-assistant guide (read before changing code)
├── config.yaml               every parameter: dates, tickers, URLs, window (21), annualisation (252), paths
├── Makefile                  make data / figures / test / all / clean
├── requirements.txt          pinned versions
├── pyproject.toml            black / ruff / pytest settings
├── docs/
│   ├── primer.md             concepts, formulas, pitfalls (read first)
│   ├── data_sources.md       every source: URL, ID, citation, licence, coverage, quirks
│   ├── methodology.md        cleaning, returns, alignment, no-look-ahead, sampling, decisions
│   ├── data_dictionary.md    every column of every processed file
│   └── events.md             key market events and what our data shows (auto-generated)
├── data/
│   ├── raw/                  untouched downloads + .meta.json (gitignored)
│   ├── interim/              cleaned single-source series (gitignored)
│   └── processed/            daily_panel, monthly_sample, events (committed)
├── src/vrp/                  all logic
│   ├── config.py             loads config.yaml
│   ├── logging_setup.py      console + logs/pipeline.log
│   ├── download.py           fetch + cache + metadata (+ retry, Stooq fallback)
│   ├── clean.py              per-source cleaning and the VIX/S&P merge
│   ├── features.py           returns, implied/realised variance, payoffs, windows
│   ├── sampling.py           monthly sample, window-overlap measurement
│   ├── events.py             event table + event-window stats
│   ├── validate.py           data-quality checks, reports, sanity summary
│   ├── plots.py              figures (v1: 01–06; v2 tests: 07–11)
│   ├── stats.py              v2: the five statistical tests (CIs, p-values, robustness variants)
│   ├── stats_report.py       v2: stats_summary.md and the pitch-claims log
│   └── pipeline.py           orchestrates everything: python -m vrp.pipeline
├── tests/                    look-ahead, formula and cleaning tests
├── outputs/
│   ├── figures/              PNG (150 dpi) + SVG
│   └── reports/              data_quality_report.md, sanity_summary.md, event_stats.csv, manifest.json
├── logs/                     pipeline.log (gitignored)
└── notebooks/01_data_exploration.ipynb   loads processed data only
```

## Data sources

| Data | Source | ID |
|---|---|---|
| VIX daily close | Cboe via FRED (no API key) | `VIXCLS` |
| S&P 500 daily close | Yahoo Finance via `yfinance` (fallback: Stooq) | `^GSPC` |
| VXO daily close (robustness) | Cboe via FRED | `VXOCLS` |
| NBER recessions (chart shading) | NBER via FRED | `USREC` |

Window: 1989-11-01 to 2026-03-31. Citations, licences and quirks: [`docs/data_sources.md`](docs/data_sources.md).
Raw data is **not** committed (Yahoo's terms); `make data` rebuilds it locally.

## Method summary

On the first trading day *t* of each month:

| | Formula |
|---|---|
| Implied variance | IV_t = (VIX_t / 100)² |
| Realised variance | RV_t = (252/21) · Σ r²_{t+k}, k = 1…21, with r = ln(P_t / P_{t−1}) |
| Seller's payoff (scaled) | payoff_scaled = (IV_t − RV_t) / IV_t = 1 − RV/IV |

Positive payoff = the VIX over-forecast and a variance seller made money. Our sign is
**implied − realised**. Details and every judgement call: [`docs/methodology.md`](docs/methodology.md).

## No-look-ahead statement

The VIX is read at the **close of day t**. Realised variance uses returns from **t+1 to t+21**
only, all strictly after the VIX observation; day t's own return is excluded. Every column
that uses future data is marked as an outcome (`fwd`, `payoff_*`, `vol_spread_pts`) and is
never used as a predictor. `tests/test_alignment.py` enforces this, including mutation tests
that change future prices and check that past values do not move.

## Outputs

| File | What |
|---|---|
| `data/processed/daily_panel.csv` / `.parquet` | 9,125 trading days with all variables |
| `data/processed/monthly_sample.csv` | 432 months, Jan 1990 – Dec 2025 |
| `data/processed/events.csv` | 16 key market event windows |
| `outputs/reports/event_stats.csv`, `docs/events.md` | per-event VIX peak, worst day, drawdown, payoffs |
| `outputs/reports/data_quality_report.md` | dropped dates, gaps, merge mismatches, flags, assertions |
| `outputs/reports/sanity_summary.md` | our numbers vs published benchmarks |
| `outputs/reports/manifest.json` | each processed file's rows, columns, dates, SHA-256 and git commit |
| `outputs/figures/01…06` | price, VIX vs realised vol, scatter, payoff histogram, payoff timeline, coverage |
| `data/processed/monthly_variants.csv` | v2 robustness payoffs: H = 22, demeaned RV, VXO |
| `outputs/reports/stats_summary.md`, `stats_tables.csv`, `stats_deciles.csv` | v2: every test statistic with 95% CI, test and p-value |
| `outputs/reports/pitch_claims.md` / `.csv` | v2: claims the data supports (and does not), with evidence |
| `outputs/figures/07…11` | v2: one figure per test (means, distribution, sell-offs, Mincer–Zarnowitz, robustness) |
| `data/processed/rv_calendar_month.csv` | research: realised variance of each calendar month |
| `outputs/reports/research_summary.md`, `research_tables.csv`, `research_rv_by_year.csv`, `research_rolling_premium.csv` | research: vol dynamics, premium by known-at-t conditions, time variation |
| `outputs/reports/research_claims.md` / `.csv` | research: the nine pre-registered claims (R-1 … R-9) and verdicts |
| `outputs/figures/12…15` | research: vol dynamics, vol by year, premium by conditions at t, rolling premium |
| `deck/VIX_pitch.pptx` | pitch deck built from the OAF template by `make deck`; every number read from the tables above |

v1 sanity check (1990–2024, monthly sample): mean VIX **19.54** (benchmark ≈ 19.6), mean
realised vol **15.34** (≈ 15.5), **432** monthly rows, highest VIX close **82.69 on 2020-03-16**.

## Next steps

- **v2: statistical tests (done).** See `outputs/reports/stats_summary.md`,
  `outputs/reports/pitch_claims.md` and `docs/methodology.md` §9. v2 needs `scipy` and
  `statsmodels`, so re-run `pip install -r requirements.txt` after pulling.
- **Research section (in progress, branch `research`).** `make research`; plan and
  pre-registered tests in `docs/research_plan.md`; teammates' brief in `docs/team_brief.md`.
- **v3: deck charts.** Slide-ready versions of the figures with the v2 results.
