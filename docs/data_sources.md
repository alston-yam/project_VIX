# Data sources

Every input to the project, where it comes from, and what to watch out for.
All downloads are made by `src/vrp/download.py`. Each raw file in `data/raw/`
gets a `<name>.meta.json` sidecar recording the URL or ticker, provider,
download time (UTC), library versions, row count, first/last date and the
file's SHA-256, so any teammate can check they are working from the same bytes.

**Requested window:** 1989-11-01 to 2026-03-31. We start two months before 1990 so
the first 1990 return exists, and finish three months after 2025 so the
December 2025 forward window (21 trading days) can complete.

## Summary

| Name | Provider | ID / ticker | Frequency | Coverage in our files | Used for |
|---|---|---|---|---|---|
| VIX close | Cboe, via FRED | `VIXCLS` | daily | 1990-01-02 → 2026-03-31 | implied variance (the forecast) |
| S&P 500 close | Yahoo Finance, via `yfinance` | `^GSPC` | daily | 1989-11-01 → 2026-03-31 | returns, realised variance |
| VXO close (optional) | Cboe, via FRED | `VXOCLS` | daily | 1989-11-01 → 2021-09-23 | robustness: old-method VIX |
| NBER recession indicator (optional) | NBER, via FRED | `USREC` | monthly | 1989-11 → 2026-03 | grey shading on charts only |

Coverage is after cleaning; see `outputs/reports/data_quality_report.md` for exact row counts.

---

## 1. VIX: Cboe Volatility Index (FRED `VIXCLS`)

- **URL:** <https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS> (no API key)
- **Series page:** <https://fred.stlouisfed.org/series/VIXCLS>
- **What it is:** the market's expected 30-calendar-day volatility of the S&P 500,
  built from S&P 500 index option prices, quoted in annualised percentage points.
  Daily **close** only (no intraday highs).
- **Citation (FRED's requested format):**
  Chicago Board Options Exchange, CBOE Volatility Index: VIX [VIXCLS], retrieved from
  FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/VIXCLS,
  <download date from `data/raw/vix.meta.json`>.
- **Licence:** the VIX is Cboe's data. FRED redistributes it; Cboe's copyright applies.
  Fine for academic/educational use with citation. We commit only derived, processed data.
- **Known quirks:**
  - **Missing values.** FRED lists every weekday. Days with no value appear as `"."`
    (older downloads) or a blank (newer ones). We read both as missing and drop them;
    nearly all are US market holidays.
  - **Holiday values since 2022.** From 2022 FRED shows VIX values on some US market
    holidays (e.g. Memorial Day, Juneteenth), presumably from Cboe's extended
    global trading hours. The S&P has no close on those days, so the inner join drops them.
  - **Three trading days with no VIX value:** 1991-03-01, 1997-01-31 and 1997-11-26. The
    S&P traded on these days, so they are dropped from the panel and the next return covers
    two days (flagged in `ret_spans_dropped_day`).
  - **Back-calculation.** Today's VIX methodology (SPX options, model-free) went live on
    **22 Sep 2003**. Earlier values were back-calculated by Cboe with the new method; they are
    not the numbers traders saw at the time. The monthly file has a `post_2003` flag for this.
  - **Record closes.** The highest close in FRED is 82.69 on 2020-03-16. Sources disagree
    about the 2008 record (80.86 on 20 Nov vs 80.74 on 21 Nov); FRED shows 80.86 on 20 Nov.
    The famous ~65 on 5 Aug 2024 was intraday; FRED's close that day is 38.57.

## 2. S&P 500 index (Yahoo Finance `^GSPC` via `yfinance`)

- **Call:**
  `yf.download("^GSPC", start="1989-11-01", end="2026-04-01", auto_adjust=True, multi_level_index=False, progress=False)`
  (`end` is exclusive in yfinance, so we add a day to make 2026-03-31 inclusive).
- **What it is:** the S&P 500 **price** index (no dividends). We keep Open, High, Low,
  Close and Volume in the raw and interim files; only Close is used.
- **`auto_adjust` is set explicitly** because its default has changed across yfinance
  versions. For an index it makes no difference (there are no splits or dividends to
  adjust), but being explicit protects us from silent changes.
- **Price vs total return:** a price index ignores dividends. Daily dividend yield is about
  0.005–0.01% a day, tiny next to typical daily moves of ~1%, and it enters squared
  returns only at second order. So it is **immaterial for daily realised variance**. It
  would matter for long-horizon return comparisons, which v1 does not make.
- **Terms of use:** Yahoo Finance data is for personal, non-commercial use and may not be
  redistributed. That is why `data/raw/` and `data/interim/` are **gitignored**: each
  teammate rebuilds them locally with `make data`. The processed files in
  `data/processed/` contain derived values for our research and are committed so the team can
  work without hitting Yahoo's rate limits; do not republish them.
- **Rate limits:** Yahoo throttles heavy users (`YFRateLimitError`), sometimes for hours. We
  cache the raw file and retry up to 3 times with exponential backoff (5 s, 10 s).
- **Fallback:** if Yahoo fails, we try Stooq ticker `^SPX` through `pandas_datareader`.
  The source actually used is logged and written to `spx.meta.json` (`source_used`).
  The committed v1 outputs used **yfinance**.
- **Known quirks:** market closures appear as gaps: 9/11 (closed 11–14 Sep 2001), the
  Ford national day of mourning (2 Jan 2007) and Hurricane Sandy (29–30 Oct 2012).
  Three daily moves exceed 10% in absolute log terms: 2008-10-13 (+10.96%), 2008-10-28
  (+10.25%) and 2020-03-16 (−12.77%). They are flagged and kept.

## 3. VXO (FRED `VXOCLS`), optional

- **URL:** <https://fred.stlouisfed.org/graph/fredgraph.csv?id=VXOCLS>
- **Citation:** Chicago Board Options Exchange, CBOE S&P 100 Volatility Index: VXO [VXOCLS],
  retrieved from FRED, Federal Reserve Bank of St. Louis;
  https://fred.stlouisfed.org/series/VXOCLS, <download date>.
- **What it is:** the original 1993 VIX: implied vol from **at-the-money S&P 100** options
  (Black–Scholes based). Cboe discontinued it in **September 2021** (our last value:
  2021-09-23). It is in the panel as `vxo_close` (left-joined, NaN where missing) for
  robustness checks only.

## 4. NBER recession indicator (FRED `USREC`), optional

- **URL:** <https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC>
- **Citation:** Federal Reserve Bank of St. Louis, NBER based Recession Indicators for the
  United States from the Period following the Peak through the Trough [USREC], retrieved
  from FRED; https://fred.stlouisfed.org/series/USREC, <download date>.
- **What it is:** monthly 0/1 flag, 1 in NBER recession months.
- **Caution:** NBER announces turning points months or years after the fact, so this
  indicator was **not known in real time**. It is used **only for chart shading** and is
  deliberately not in the daily panel or monthly sample.

## Re-downloading

```bash
python -m vrp.pipeline --stage data --refresh   # or: make refresh
```

New downloads will change the SHA-256 in the sidecars and in `outputs/reports/manifest.json`
whenever the provider revises history. Compare checksums before comparing results.
