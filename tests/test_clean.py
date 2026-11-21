"""Cleaning rules: parsing, duplicates, missing values, flags, gaps and the merge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vrp import clean
from vrp.validate import QualityLog


def test_fred_dot_and_blank_become_missing(tmp_path: Path, cfg) -> None:
    f = tmp_path / "vix.csv"
    f.write_text(
        "observation_date,VIXCLS\n2000-01-03,24.21\n2000-01-04,.\n" "2000-01-05,\n2000-01-06,22.0\n"
    )
    raw = clean.read_fred_raw(f, "vix_close")
    assert raw.isna().sum() == 2
    q = QualityLog()
    s = clean.clean_vix_like(raw, "vix", cfg, q)
    assert list(s.index) == [pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-06")]
    assert "vix: missing values::Dropped dates (vix)" in q.tables
    assert len(q.tables["vix: missing values::Dropped dates (vix)"]) == 2


def test_no_forward_fill(cfg) -> None:
    raw = pd.Series([20.0, np.nan, 22.0], index=["2000-01-03", "2000-01-04", "2000-01-05"])
    s = clean.clean_vix_like(raw, "vix", cfg, QualityLog())
    assert pd.Timestamp("2000-01-04") not in s.index
    assert s.tolist() == [20.0, 22.0]


def test_duplicates_dropped_sorted_and_tz_normalised() -> None:
    idx = pd.DatetimeIndex(
        ["2000-01-05 16:00", "2000-01-03 16:00", "2000-01-05 16:00"], tz="America/New_York"
    )
    s = pd.Series([3.0, 1.0, 99.0], index=idx)
    q = QualityLog()
    out = clean.standardise_index(s, "x", q)
    assert out.index.tz is None
    assert list(out.index) == [pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-05")]
    assert out.iloc[-1] == 3.0  # first occurrence kept
    assert "dropped 1 duplicate" in q.sections["x: duplicates"][0]


def test_out_of_range_vix_flagged_not_removed(cfg) -> None:
    raw = pd.Series([4.0, 20.0, 120.0], index=["2000-01-03", "2000-01-04", "2000-01-05"])
    q = QualityLog()
    s = clean.clean_vix_like(raw, "vix", cfg, q)
    assert len(s) == 3
    chk = [c for c in q.checks if c.name == "vix_range"][0]
    assert not chk.passed and "2 closes" in chk.detail


def test_non_positive_dropped(cfg) -> None:
    raw = pd.Series([20.0, 0.0, 21.0], index=["2000-01-03", "2000-01-04", "2000-01-05"])
    s = clean.clean_vix_like(raw, "vix", cfg, QualityLog())
    assert len(s) == 2


def test_extreme_return_flagged_not_removed(cfg) -> None:
    dates = pd.bdate_range("2008-10-01", periods=5)
    close = [100.0, 101.0, 89.0, 90.0, 91.0]  # ln(89/101) ≈ -12.6%
    raw = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1}, index=dates
    )
    q = QualityLog()
    df = clean.clean_spx(raw, cfg, q)
    assert len(df) == 5
    flagged = q.tables["spx: extreme returns::Flagged extreme days"]
    assert list(flagged["date"]) == [dates[2]]


def test_find_gaps_threshold() -> None:
    idx = pd.DatetimeIndex(["2001-09-07", "2001-09-10", "2001-09-17", "2001-09-21", "2001-09-24"])
    g = clean.find_gaps(idx, 4)
    assert len(g) == 1  # 7-day gap only; the 4-day gap (Fri->Mon is 3, 17->21 is 4) is not
    assert g.iloc[0]["prev_date"] == pd.Timestamp("2001-09-10")
    assert g.iloc[0]["calendar_days"] == 7


def test_merge_inner_join_and_return_seed() -> None:
    spx_dates = pd.DatetimeIndex(
        ["1999-12-31", "2000-01-03", "2000-01-04", "2000-01-05", "2000-01-06"]
    )
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=spx_dates)
    spx = pd.DataFrame({"close": close})
    # VIX starts 2000-01-03, is missing 2000-01-05, and has an extra (holiday) date.
    vix = pd.Series(
        [20.0, 21.0, 22.0, 23.0],
        index=pd.DatetimeIndex(["2000-01-03", "2000-01-04", "2000-01-06", "2000-01-07"]),
    )
    q = QualityLog()
    p = clean.merge_vix_spx(vix, spx, q)
    assert list(p.index) == list(pd.DatetimeIndex(["2000-01-03", "2000-01-04", "2000-01-06"]))
    # First return seeded from the 1999-12-31 S&P close.
    assert p["spx_prev_close"].iloc[0] == 100.0
    # 2000-01-06's previous panel close is 2000-01-04 (2000-01-05 dropped) -> flagged.
    assert p.loc["2000-01-06", "spx_prev_close"] == 102.0
    assert p["ret_spans_dropped_day"].tolist() == [False, False, True]
    assert len(q.tables["Merge (inner join)::S&P dates with no VIX"]) == 1
    assert len(q.tables["Merge (inner join)::VIX dates with no S&P close"]) == 1
