"""Tests for the research section (``vrp.research``, ``vrp.research_report``)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from conftest import build_panel, make_inputs

from vrp import research
from vrp.research_report import evaluate_research_claims
from vrp.stats import Bootstrapper, StatsSettings, mean_nw


# --------------------------------------------------------------------------- A. dynamics
def test_calendar_month_rv_constant_returns(cfg) -> None:
    """|r| = c every day ⇒ calendar-month RV = 252·c² exactly, whatever the month length."""
    c = 0.01
    r = np.where(np.arange(120) % 2 == 0, c, -c)
    vix, spx = make_inputs(r, 20.0, start="1990-01-02")
    panel = build_panel(vix, spx)
    rvm = research.calendar_month_rv(panel, cfg)
    assert len(rvm) >= 4
    np.testing.assert_allclose(rvm["rv_cal_ann_var"], 252 * c**2, rtol=1e-12)
    np.testing.assert_allclose(rvm["rv_cal_ann_vol_pct"], 100 * np.sqrt(252) * c, rtol=1e-12)
    # Months are calendar months: day counts are business days in each month.
    assert rvm.index[0] == pd.Timestamp("1990-01-01")
    assert rvm["n_days"].iloc[1] == len(pd.bdate_range("1990-02-01", "1990-02-28"))


def test_ar1_recovers_coefficient() -> None:
    """A long simulated AR(1) with ρ = 0.6: the estimate's CI covers 0.6."""
    rng = np.random.default_rng(1)
    x = np.zeros(20_000)
    for i in range(1, len(x)):
        x[i] = 0.6 * x[i - 1] + rng.normal()
    a = research.ar1(x, lags=6, alpha=0.05)
    assert a["lo"] < 0.6 < a["hi"]
    assert abs(a["rho"] - 0.6) < 0.02


def test_half_life() -> None:
    """ρ = 0.5 halves in one period; ρ outside (0, 1) has no half-life."""
    assert research.half_life(0.5) == pytest.approx(1.0)
    assert research.half_life(0.25) == pytest.approx(0.5)
    assert np.isnan(research.half_life(1.0))
    assert np.isnan(research.half_life(-0.2))


def test_time_trend_exact_line() -> None:
    """y = 2 + 0.3·t exactly ⇒ slope 0.3, intercept 2."""
    t = np.arange(100) / 12.0
    tr = research.time_trend(2 + 0.3 * t, t, lags=6, alpha=0.05)
    assert tr["slope"] == pytest.approx(0.3)
    assert tr["intercept"] == pytest.approx(2.0)


def test_years_since_start() -> None:
    """First date is 0; one year later is about 1."""
    idx = pd.DatetimeIndex(["2000-01-01", "2001-01-01"])
    np.testing.assert_allclose(research.years_since_start(idx), [0.0, 366 / 365.25])


# --------------------------------------------------------------------------- B. conditional
def test_bucket_edges() -> None:
    """Thresholds are inclusive on the right: x ≤ t1, t1 < x ≤ t2, x > t2."""
    labels = research.bucket_labels("VIX", (15, 25), "")
    b = research.bucket(pd.Series([10.0, 15.0, 15.01, 25.0, 25.01, np.nan]), (15, 25), labels)
    assert list(b.iloc[:5].astype(str)) == [labels[0], labels[0], labels[1], labels[1], labels[2]]
    assert pd.isna(b.iloc[5])
    assert labels == ["VIX ≤ 15", "15 < VIX ≤ 25", "VIX > 25"]


def test_gap_is_known_at_t() -> None:
    """The gap uses only the VIX at t and past realised variance."""
    m = pd.DataFrame({"vix_close": [20.0], "rv_past21_ann_var": [0.0225]})
    assert research.gap_pts(m).iloc[0] == pytest.approx(5.0)
    assert not any("fwd" in c for c in research.CONDITIONING)


def test_s_diff_two_by_hand() -> None:
    """mean(x | a) − mean(x | b) on a toy row."""
    x = np.array([[1.0, 2.0, 3.0, 10.0]])
    a = np.array([[True, True, False, False]])
    b = np.array([[False, False, False, True]])
    assert research.s_diff_two(x, a, b)[0] == pytest.approx(1.5 - 10.0)


# --------------------------------------------------------------------------- C. time
def test_rolling_premium_matches_pandas() -> None:
    """The rolling mean equals pandas' rolling mean; the band comes from mean_nw."""
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    rng = np.random.default_rng(3)
    m = pd.DataFrame({"payoff_scaled": rng.normal(0.3, 0.5, 30)}, index=idx)
    roll = research.rolling_premium(m, "payoff_scaled", window=12, lags=3, alpha=0.05)
    ref = m["payoff_scaled"].rolling(12).mean().dropna()
    np.testing.assert_allclose(roll["mean"], ref.to_numpy())
    assert roll.index[0] == idx[11]
    nw = mean_nw(m["payoff_scaled"].iloc[:12], 3, 0.05)
    assert roll["ci_low"].iloc[0] == pytest.approx(nw["ci_low"])


# --------------------------------------------------------------------------- real data
@pytest.fixture(scope="module")
def real_research(cfg, real_panel, real_monthly):
    """The research results on the processed data (fewer resamples for speed)."""
    raw = copy.deepcopy(cfg.raw)
    raw["stats"]["bootstrap"]["n_resamples"] = 500
    cfg2 = type(cfg)(raw=raw, root=cfg.root)
    return research.run_research(real_panel, real_monthly, cfg2)


def test_real_calendar_months_cover_sample(real_research, cfg) -> None:
    """One row per sample month; plausible day counts."""
    rvm = real_research.rv_monthly
    assert len(rvm) == cfg["checks"]["expected_monthly_rows"]
    assert rvm["n_days"].between(15, 23).all()


def test_real_buckets_partition_sample(real_research, real_monthly) -> None:
    """VIX buckets cover every month; gap buckets every month with past RV."""
    b = real_research.buckets
    assert b["vix_bucket"].notna().sum() == len(real_monthly)
    assert b["gap_bucket"].notna().sum() == real_monthly["rv_past21_ann_var"].notna().sum()


def test_real_claims_evaluate(real_research) -> None:
    """All nine pre-registered claims resolve to exactly one evidence row each."""
    claims = evaluate_research_claims(real_research)
    assert list(claims["id"]) == [f"R-{i}" for i in range(1, 10)]
    assert claims["p_holm"].between(0, 1).all()


def test_settings_read(cfg) -> None:
    """Config sections parse."""
    rs = research.ResearchSettings.from_config(cfg)
    s = StatsSettings.from_config(cfg)
    assert rs.vix_thresholds[0] < rs.vix_thresholds[1]
    assert isinstance(Bootstrapper(s.block_length, 10, 1), Bootstrapper)
