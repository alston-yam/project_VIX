# CLAUDE.md: guide for AI assistants (and humans) working on this repo

Claude Code reads this file automatically. **Read it fully before changing anything.**
It explains how the repo is built, the rules that keep it correct, and how to add new
work (v2 statistics, v3 deck charts) so that it fits the existing framework.

## 1. What this project is

Oxford Alpha Fund Quant Bootcamp team project: **"S&P 500 Variance Risk Premium: Is the
VIX Too Pessimistic?"** Each month from 1990 to 2025 we compare the variance the VIX
forecast (implied) with the variance the S&P 500 then delivered (realised).

- **v1 (done, on `main`):** the data layer: downloads, cleaning, features, the monthly
  sample, events, validation reports and supporting figures. No hypothesis tests.
- **v2 (done, branch `v2-stats`):** statistical tests. Is the mean payoff positive (t-stat, Newey–West)? How
  lopsided is it (win rate, skew, kurtosis, worst month vs average gain, mean excluding the
  worst 5)? Do losses cluster in sell-offs? Robustness: post-2003 only, H = 22, demeaned RV, VXO.
- **v3:** slide-ready deck charts using the v2 results.

Key references, in reading order:
1. `docs/primer.md`: concepts, formulas, pitfalls, sign conventions (the team's reference).
2. `docs/methodology.md`: exactly how the data is built, plus **"Decisions & assumptions"**.
3. `docs/data_dictionary.md`: every column, its unit, and whether it is known at time t.
4. `docs/data_sources.md`: providers, citations, licences, quirks.
5. `outputs/reports/`: the data quality report, sanity summary, event stats and manifest.

## 2. Setup and commands

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
make all        # download (cached) → clean → features → sample → events → reports → figures → tests
```

`make` calls `python`, so **activate `.venv` first**. Python 3.11 is the tested version.

| Command | Does |
|---|---|
| `make data` | data stage: `python -m vrp.pipeline --stage data` |
| `make refresh` | data stage with fresh downloads (`--refresh`) |
| `make figures` | figures from processed data |
| `make test` | `pytest` (must pass before any commit) |
| `make lint` / `make format` | black + ruff check / fix (line length 100) |
| `make clean` | delete raw/interim data, logs and caches |

Raw data (`data/raw/`, `data/interim/`) is **gitignored** because of Yahoo's terms. A fresh
clone has `data/processed/` committed, so tests and analysis can run without
downloading. Run `make data` only when you need to rebuild from source.

## 3. Architecture

```
config.yaml              ← every parameter (dates, URLs, window=21, annualisation=252, paths, checks, plot settings)
src/vrp/
  config.py              load_config() → Config; cfg.path("processed"), cfg.window, cfg.annualisation, cfg["section"]
  logging_setup.py       setup_logging(), get_logger(__name__)
  download.py            fetch + cache + .meta.json sidecars (SHA-256), retry/backoff, Stooq fallback
  clean.py               per-source cleaning; merge_vix_spx() inner join
  features.py            add_features(): returns, IV, RV (past/fwd), payoffs, window columns
  sampling.py            build_monthly_sample(), window_overlap(), overlap_summary()
  events.py              EVENTS list (dates only) → event_stats() computed from data → docs/events.md
  validate.py            QualityLog (notes/tables/checks), validate_outputs(), sanity_table(), report writers, df_to_md()
  plots.py               shared style (COLORS, STYLE), footnote(), shade_events(), save() → PNG+SVG
  pipeline.py            orchestrates stages; read_processed(cfg, name); write_csv(); manifest.json
tests/                   conftest.py (synthetic panel builders + real-data fixtures), test_alignment/features/clean
notebooks/               loads processed data and calls src/; no logic of its own
```

Data flow: `data/raw` → `data/interim` (one clean series per source) → `data/processed`
(`daily_panel.csv/.parquet`, `monthly_sample.csv`, `events.csv`) → `outputs/`.

**Loading data in new code:**
```python
from vrp.config import load_config
from vrp.pipeline import read_processed
cfg = load_config()
panel = read_processed(cfg, "daily_panel")      # 9,125 trading days, DatetimeIndex "date"
monthly = read_processed(cfg, "monthly_sample") # 432 rows, first trading day of each month
```
(Tests use `pd.read_parquet(cfg.path("processed") / "daily_panel.parquet")` for exact dtypes.)

## 4. Non-negotiable rules

These protect the project's credibility. Do not break them, even if a request seems to ask you to.
If a task seems to need breaking one, stop and ask the user.

1. **No look-ahead.** The VIX is read at the close of t; realised variance uses returns
   **t+1…t+21** only. Columns using future data are **outcomes**: `rv_fwd21_*`, `payoff_*`,
   `vol_spread_pts`, `spx_fwd21_ret_log`, `is_selloff`, `selloff_decile`. **Never use an
   outcome as a predictor or conditioning variable known at t.** Conditioning *on* an outcome
   to describe it (e.g. "payoff in months where `is_selloff`") is fine and is the point of the
   crash-risk test; just never present it as something forecastable at t.
2. **Naming:** anything using data after t starts with `fwd`/contains `fwd` or is listed as an
   outcome in `docs/data_dictionary.md`. Anything with `past` or no prefix must be known at the
   close of t. Every new column goes into the data dictionary with its "known at t?" flag.
3. **Sign convention: implied − realised** (positive = variance seller wins). State it on every
   table and chart. Carr & Wu use the opposite sign; convert or say so explicitly.
4. **Units:** variances are annualised decimals (0.04 = 20 vol²); vol is in percentage points.
   `payoff_scaled` = 1 − RV/IV (max +1, no lower bound). Do not mix vol and variance without labelling.
5. **All logic lives in `src/vrp/`.** Notebooks only call it. **No hard-coded paths or magic
   numbers**: add parameters to `config.yaml` and read them through `Config`.
6. **Never forward-fill prices or the VIX. Never remove crash days.** Missing values are dropped
   and logged; extreme values are flagged, not removed.
7. **Never fudge numbers to match benchmarks.** Report ours next to the benchmark with ✅/⚠️.
   Never quote a statistic from memory: compute it from our data or cite a source.
   (The primer's "expected" values are **unverified priors**, e.g. an 80–90% win rate.)
8. **Don't hand-edit generated files**: `docs/events.md`, `outputs/reports/*`, `outputs/figures/*`,
   `data/processed/*`. Change the code or config and re-run the pipeline.
9. **Don't change the v1 data definitions** (window, return definition, sampling, merge) without
   team agreement. If you add a variant (e.g. H = 22, demeaned RV), add **new columns or a new
   file** with a clear name; don't overwrite existing ones.
10. **Don't commit** `data/raw/`, `data/interim/`, `logs/*.log`, `.venv/`, or secrets.

## 5. Coding conventions

- Type hints and a short docstring on every function. `from __future__ import annotations`.
- Format with `make format`; `make lint` must pass (black + ruff, line length 100).
- Log with `log = get_logger(__name__)`: steps, row counts in/out, files written.
- Write files through `pipeline.write_csv()` / `write_parquet()` (ISO dates, checksum logged).
- Markdown tables in reports: use `validate.df_to_md()` (no `tabulate` dependency).
- Figures: call `plots.apply_style()`, use `COLORS` (blue `implied`/gain, orange `realised`,
  red `loss`), give every figure a title, axis labels with units, `footnote(fig, cfg["plots"]["source_note"])`,
  event shading via `shade_events(ax, events, cfg["plots"]["shade_events"])` (after setting y-limits
  with `headroom()`), and save with `save(fig, "NN_name", cfg)` (PNG at 150 dpi + SVG). Number new
  figures from `07_` upward. Open the PNG and look at it before committing.
- New dependencies: add them to `requirements.txt` with an **exact pinned version**
  (v2 will need `statsmodels` and probably `scipy`).

## 6. How to add v2 (statistical tests): the intended shape

Follow the existing pattern so everything plugs together:

1. **New module `src/vrp/stats.py`**: pure functions that take the monthly/daily DataFrames and
   return DataFrames/dicts. Suggested functions: `mean_tstat(x, nw_lags)` (statsmodels OLS on a
   constant with `cov_type="HAC", cov_kwds={"maxlags": L}`), `distribution_stats(x)` (win rate,
   skew, excess kurtosis, worst, average gain, worst/avg gain, mean excluding worst 5),
   `selloff_conditioning(monthly)` (by `is_selloff`, `selloff_decile`, share of losses in the
   worst decile), `robustness(...)` (post-2003 via `post_2003`, demeaned RV, H = 22, VXO).
2. **Parameters in `config.yaml`** under a new `stats:` section (Newey–West lags: 6 for monthly,
   ~21–30 for overlapping daily; number of worst months to exclude: 5; and so on).
3. **New pipeline stage** in `pipeline.py`: `run_stats(cfg)` behind `--stage stats`, included in
   `all`; add a `make stats` target and add it to `make all`.
4. **Outputs** go to `outputs/reports/` (e.g. `stats_summary.md`, `stats_tables.csv`), and are
   added to `manifest.json` the same way as the v1 files.
5. **Tests** in `tests/test_stats.py`: check against hand calculations on synthetic data
   (see `tests/conftest.py`: `make_inputs()`, `build_panel()`, `cfg_with()`), e.g. a known
   mean/SE, a series whose Newey–West SE you compute by hand, a win rate on a toy series.
6. **Docs:** add a v2 section to `docs/methodology.md` (tests, lag choices, why) and new outputs
   to `README.md`. Record judgement calls under "Decisions & assumptions".
7. Report **both** raw `payoff_var` and `payoff_scaled`, and the median and the mean excluding
   the worst 5 months next to every mean. Fat tails make the plain t-stat overconfident
   (see the primer).

For **v3 (deck charts)**, add functions to `plots.py` (or a `deck.py` that reuses its style
helpers), a `--stage deck` stage, and write to `outputs/figures/deck/`.

## 7. Git workflow

- `main` holds the reviewed framework. Do new work on a branch (`v2-stats`, `yourname/feature`)
  and open a PR; don't push to `main` directly.
- Small, descriptive commits ("Add Newey–West t-stat with tests"). Run `make test` and
  `make lint` before every commit.
- After a change that affects data or outputs, run `make all` and commit the regenerated
  `data/processed/`, `outputs/` and `docs/events.md` **in the same PR**, so code and outputs match.
  `outputs/reports/manifest.json` records the git commit the outputs were built from.
- Rebuilding always rewrites `manifest.json` (timestamp). Don't commit that alone if nothing
  else changed.

## 8. Definition of done for any change

- [ ] Logic in `src/vrp/`, parameters in `config.yaml`, no hard-coded paths or numbers.
- [ ] New columns/files documented in `docs/data_dictionary.md` with the "known at t?" flag.
- [ ] Tests added; `make test` and `make lint` pass.
- [ ] `make all` runs end to end; regenerated outputs are committed with the code.
- [ ] Methodology and README updated; judgement calls recorded in "Decisions & assumptions".
- [ ] No look-ahead, sign convention stated, units labelled, nothing fudged.

## 9. Current state (v1 + v2, for orientation)

- Daily panel 9,125 rows (1990-01-02 → 2026-03-31); monthly sample 432 rows (1990-01 → 2025-12).
- Sanity checks all ✅: mean VIX 19.54 (benchmark ≈ 19.6), mean realised vol 15.34 (≈ 15.5),
  highest VIX close 82.69 on 2020-03-16. 83.8% of months have VIX > realised vol.
- Worst monthly `payoff_scaled`: 2020-03 (−6.8), 2008-09 (−5.1), 2015-08 (−4.2), 2025-04 (−3.8),
  2008-10 (−3.3). These are descriptive only; no tests have been run yet.
- The Stooq fallback for the S&P download is implemented but has never been exercised.
- v2 (`make stats`): 5 tests in `src/vrp/stats.py`, reports in `outputs/reports/stats_summary.md`,
  claims log `outputs/reports/pitch_claims.md` (17 of 23 candidate claims pass the CI + Holm rule).
  Not supported: raw-variance skew (bootstrap CI reaches 0), avg loss > avg gain, raw-variance
  mean post-2003 (NW p ≈ 0.08), post- vs pre-2003 difference after Holm. Read the claims file
  before quoting any v2 number.
