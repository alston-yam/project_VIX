"""Per-source cleaning and the VIX/S&P merge.

Principles: never forward-fill, never invent a value. Missing days are dropped
and listed. Suspicious values (extreme returns, VIX outside [5, 100]) are
flagged in the quality report but kept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vrp.config import Config
from vrp.logging_setup import get_logger
from vrp.validate import QualityLog

log = get_logger(__name__)


# --------------------------------------------------------------------------- generic helpers
def to_date_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Parse to a tz-naive, midnight-normalised ``DatetimeIndex``."""
    dt = pd.DatetimeIndex(pd.to_datetime(idx))
    if dt.tz is not None:
        dt = dt.tz_localize(None)
    return dt.normalize().astype("datetime64[ns]")


def standardise_index(
    obj: pd.Series | pd.DataFrame, name: str, q: QualityLog
) -> pd.Series | pd.DataFrame:
    """Normalise dates, sort, and drop duplicate dates (keeping the first), logging any."""
    out = obj.copy()
    out.index = to_date_index(out.index)
    out.index.name = "date"
    out = out.sort_index()
    dup = out.index.duplicated(keep="first")
    n_dup = int(dup.sum())
    if n_dup:
        dates = ", ".join(str(d.date()) for d in out.index[dup])
        log.warning("%s: dropping %d duplicate dates: %s", name, n_dup, dates)
        q.note(f"{name}: duplicates", f"dropped {n_dup} duplicate dates (kept first): {dates}")
    else:
        q.note(f"{name}: duplicates", "no duplicate dates found")
    return out[~dup]


def restrict_dates(
    obj: pd.Series | pd.DataFrame, name: str, cfg: Config, q: QualityLog
) -> pd.Series | pd.DataFrame:
    """Keep rows inside the configured download window (inclusive)."""
    start = pd.Timestamp(cfg["dates"]["download_start"])
    end = pd.Timestamp(cfg["dates"]["download_end"])
    mask = (obj.index >= start) & (obj.index <= end)
    n_before, n_out = int((obj.index < start).sum()), int((obj.index > end).sum())
    log.info(
        "%s: keeping %d of %d rows inside %s..%s (%d before, %d after window)",
        name,
        int(mask.sum()),
        len(obj),
        start.date(),
        end.date(),
        n_before,
        n_out,
    )
    q.note(
        f"{name}: date window",
        f"kept {int(mask.sum())} of {len(obj)} rows inside "
        f"{start.date()}..{end.date()} ({n_before} earlier, {n_out} later rows removed)",
    )
    return obj[mask]


def drop_missing(s: pd.Series, name: str, q: QualityLog) -> pd.Series:
    """Drop NaN values and list every dropped date in the report."""
    miss = s.isna()
    n = int(miss.sum())
    if n:
        dates = [str(d.date()) for d in s.index[miss]]
        log.info("%s: dropping %d missing values (FRED '.'/blank or NaN)", name, n)
        q.note(
            f"{name}: missing values",
            f"dropped {n} dates with no value. FRED lists every "
            "weekday, so most of these are US market holidays; the ones that matter (S&P "
            "traded but this series has no value) are listed under 'Merge'",
        )
        q.table(
            f"{name}: missing values::Dropped dates ({name})",
            pd.DataFrame({"date": dates, "weekday": [pd.Timestamp(d).day_name() for d in dates]}),
        )
    else:
        q.note(f"{name}: missing values", "no missing values")
    return s[~miss]


def drop_non_positive(s: pd.Series, name: str, q: QualityLog) -> pd.Series:
    """Drop values <= 0 (cannot take logs). Expected count is zero."""
    bad = s <= 0
    n = int(bad.sum())
    q.check(
        f"{name}_positive",
        n == 0,
        f"{name}: {n} non-positive values"
        + ("" if n == 0 else " (dropped: " + ", ".join(str(d.date()) for d in s.index[bad]) + ")"),
        severity="warn",
    )
    return s[~bad]


def find_gaps(index: pd.DatetimeIndex, max_days: int) -> pd.DataFrame:
    """List consecutive dates more than ``max_days`` calendar days apart."""
    if len(index) < 2:
        return pd.DataFrame(columns=["prev_date", "next_date", "calendar_days"])
    prev, nxt = index[:-1], index[1:]
    gap = (nxt - prev).days
    m = gap > max_days
    return pd.DataFrame(
        {"prev_date": prev[m], "next_date": nxt[m], "calendar_days": np.asarray(gap)[m]}
    )


def record_gaps(index: pd.DatetimeIndex, name: str, cfg: Config, q: QualityLog) -> pd.DataFrame:
    """Find gaps > N calendar days and add them to the report."""
    max_days = int(cfg["checks"]["max_calendar_gap_days"])
    gaps = find_gaps(index, max_days)
    log.info("%s: %d gaps of more than %d calendar days", name, len(gaps), max_days)
    q.note(
        f"{name}: calendar gaps",
        f"{len(gaps)} gaps of more than {max_days} calendar days " "between consecutive dates",
    )
    q.table(f"{name}: calendar gaps::Gaps > {max_days} days ({name})", gaps)
    return gaps


# --------------------------------------------------------------------------- readers
def read_fred_raw(path: str | Path, value_name: str) -> pd.Series:
    """Read a FRED ``fredgraph.csv`` file. ``'.'`` and blanks become NaN."""
    df = pd.read_csv(path, na_values=["."], keep_default_na=True)
    df.columns = ["date", value_name]
    s = pd.to_numeric(df[value_name], errors="coerce")
    s.index = pd.Index(df["date"])
    s.name = value_name
    return s


def read_spx_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw S&P CSV written by ``download.fetch_spx``."""
    df = pd.read_csv(path, index_col=0)
    df.columns = [c.lower() for c in df.columns]
    return df


# --------------------------------------------------------------------------- cleaners
def clean_vix_like(
    raw: pd.Series, name: str, cfg: Config, q: QualityLog, check_range: bool = True
) -> pd.Series:
    """Clean a FRED daily vol index (VIX or VXO)."""
    n0 = len(raw)
    s = standardise_index(raw, name, q)
    s = restrict_dates(s, name, cfg, q)
    s = drop_missing(s, name, q)
    s = drop_non_positive(s, name, q)
    if check_range:
        lo, hi = cfg["checks"]["vix_min"], cfg["checks"]["vix_max"]
        out = s[(s < lo) | (s > hi)]
        q.check(
            f"{name}_range",
            out.empty,
            f"{name}: {len(out)} closes outside [{lo}, {hi}] (flagged, not removed)"
            + ("" if out.empty else ": " + ", ".join(f"{d.date()}={v}" for d, v in out.items())),
            severity="warn",
        )
    record_gaps(s.index, name, cfg, q)
    q.counts[f"{name}: raw rows"] = n0
    q.counts[f"{name}: clean rows"] = len(s)
    log.info("%s: %d raw rows -> %d clean rows", name, n0, len(s))
    return s.astype(float)


def clean_usrec(raw: pd.Series, cfg: Config, q: QualityLog) -> pd.Series:
    """Clean the monthly NBER recession indicator (0/1, first of month)."""
    n0 = len(raw)
    s = standardise_index(raw, "usrec", q)
    s = restrict_dates(s, "usrec", cfg, q)
    s = drop_missing(s, "usrec", q)
    bad = ~s.isin([0, 1])
    q.check(
        "usrec_binary",
        not bad.any(),
        f"usrec: {int(bad.sum())} values not in {{0, 1}}",
        severity="warn",
    )
    s = s[~bad].astype(int)
    q.counts["usrec: raw rows"], q.counts["usrec: clean rows"] = n0, len(s)
    return s


def clean_spx(raw: pd.DataFrame, cfg: Config, q: QualityLog) -> pd.DataFrame:
    """Clean S&P 500 OHLCV. Only ``close`` drives the analysis."""
    name = "spx"
    n0 = len(raw)
    df = standardise_index(raw, name, q)
    df = restrict_dates(df, name, cfg, q)
    df = df.apply(pd.to_numeric, errors="coerce")
    close = drop_missing(df["close"], name, q)
    close = drop_non_positive(close, name, q)
    df = df.loc[close.index]
    # OHLC consistency (flag only).
    inconsistent = df[(df["low"] > df["close"] + 1e-9) | (df["high"] < df["close"] - 1e-9)]
    q.note(
        f"{name}: OHLC consistency",
        f"{len(inconsistent)} rows where close lies outside [low, high] (flagged, not removed)",
    )
    zero_vol = int((df["volume"].fillna(0) <= 0).sum())
    q.note(
        f"{name}: OHLC consistency",
        f"{zero_vol} rows with zero or missing volume " "(volume is not used in the analysis)",
    )
    # Extreme daily moves on the S&P's own calendar (flag, never remove).
    thr = float(cfg["checks"]["extreme_abs_log_return"])
    r = np.log(df["close"]).diff()
    ext = r[r.abs() > thr]
    tbl = pd.DataFrame(
        {"date": ext.index, "log_return": ext.values, "simple_return": np.expm1(ext.values)}
    )
    q.note(
        f"{name}: extreme returns",
        f"{len(ext)} days with |log return| > {thr:.0%} "
        "(kept: real crash days are the point of the study)",
    )
    q.table(f"{name}: extreme returns::Flagged extreme days", tbl)
    for d, v in ext.items():
        log.warning("spx: extreme daily log return %.4f on %s (flagged, kept)", v, d.date())
    record_gaps(df.index, name, cfg, q)
    q.counts["spx: raw rows"], q.counts["spx: clean rows"] = n0, len(df)
    log.info("spx: %d raw rows -> %d clean rows", n0, len(df))
    return df


# --------------------------------------------------------------------------- merge
def merge_vix_spx(
    vix: pd.Series, spx: pd.DataFrame, q: QualityLog, vxo: pd.Series | None = None
) -> pd.DataFrame:
    """Inner-join VIX and S&P closes on dates where both exist.

    Also adds:
    * ``spx_prev_close``: the S&P close used as P_{t-1} for ``ret_log``. For the
      first panel date this is the last S&P close before it (so the first
      return exists); otherwise it is the S&P close on the previous panel date.
    * ``ret_spans_dropped_day``: True if an S&P trading day between the
      previous panel date and t was dropped by the merge, i.e. ``ret_log``
      covers more than one S&P trading day.
    * ``vxo_close`` (optional): left-joined, NaN where VXO is unavailable.
    """
    close = spx["close"]
    only_spx = close.index.difference(vix.index)
    only_vix = vix.index.difference(close.index)
    first_vix = vix.index.min()
    only_spx_in = only_spx[only_spx >= first_vix]
    pre_sample = only_spx[only_spx < first_vix]
    log.info("merge: VIX %d dates, S&P %d dates", len(vix), len(close))
    log.info(
        "merge: %d S&P dates before first VIX date %s (used only for the first return)",
        len(pre_sample),
        first_vix.date(),
    )
    log.info("merge: %d S&P dates with no VIX value (dropped)", len(only_spx_in))
    log.info("merge: %d VIX dates with no S&P close (dropped)", len(only_vix))
    q.note("Merge (inner join)", f"VIX has {len(vix)} dates; S&P has {len(close)} dates")
    q.note(
        "Merge (inner join)",
        f"{len(pre_sample)} S&P dates fall before the first VIX date "
        f"({first_vix.date()}); they only seed the first return",
    )
    q.note(
        "Merge (inner join)",
        f"{len(only_spx_in)} S&P trading dates have no VIX value and are " "dropped from the panel",
    )
    q.note(
        "Merge (inner join)",
        f"{len(only_vix)} VIX dates have no S&P close (exchange holidays "
        "or closures on which Cboe still published a value) and are dropped",
    )
    q.table(
        "Merge (inner join)::S&P dates with no VIX",
        pd.DataFrame({"date": only_spx_in, "weekday": only_spx_in.day_name()}),
    )
    q.table(
        "Merge (inner join)::VIX dates with no S&P close",
        pd.DataFrame(
            {
                "date": only_vix,
                "weekday": only_vix.day_name(),
                "vix_close": vix.loc[only_vix].values,
            }
        ),
    )

    both = close.index.intersection(vix.index)
    panel = pd.DataFrame({"spx_close": close.loc[both], "vix_close": vix.loc[both]})
    panel.index.name = "date"

    # P_{t-1}: previous panel close; for the first row, last S&P close before it.
    prev = panel["spx_close"].shift(1)
    before_first = close[close.index < both[0]]
    if not before_first.empty:
        prev.iloc[0] = before_first.iloc[-1]
    panel["spx_prev_close"] = prev

    # Did the merge drop an S&P trading day between consecutive panel dates?
    pos = close.index.get_indexer(both)
    spans = np.r_[False, np.diff(pos) > 1]
    panel["ret_spans_dropped_day"] = spans
    q.note(
        "Merge (inner join)",
        f"{int(spans.sum())} panel returns span a dropped S&P day "
        "(the return then covers two trading days of price change; listed below)",
    )
    q.table(
        "Merge (inner join)::Returns spanning a dropped S&P day",
        pd.DataFrame({"date": panel.index[spans]}),
    )

    if vxo is not None:
        panel["vxo_close"] = vxo.reindex(panel.index)
    q.counts["merged panel rows"] = len(panel)
    log.info(
        "merge: panel has %d rows (%s..%s)",
        len(panel),
        panel.index.min().date(),
        panel.index.max().date(),
    )
    return panel
