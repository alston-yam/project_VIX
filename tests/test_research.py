"""Tests for the research section (``vrp.research``, ``vrp.research_report``)."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd
import pytest
from conftest import build_panel, make_inputs
from scipy import stats as sps

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
    """All eleven pre-registered claims resolve to exactly one evidence row each."""
    claims = evaluate_research_claims(real_research)
    assert list(claims["id"]) == [f"R-{i}" for i in range(1, 12)]
    assert claims["p_holm"].between(0, 1).all()
    assert not claims["claim"].str.contains(r"\{|\}", regex=True).any()
    assert not claims["fallback"].str.contains(r"\{|\}", regex=True).any()


def test_settings_read(cfg) -> None:
    """Config sections parse."""
    rs = research.ResearchSettings.from_config(cfg)
    s = StatsSettings.from_config(cfg)
    assert rs.vix_thresholds[0] < rs.vix_thresholds[1]
    assert isinstance(Bootstrapper(s.block_length, 10, 1), Bootstrapper)


# --------------------------------------------------------------------------- D. break-even
def _settings(cfg, n_boot: int = 200):
    """Stats settings with fewer resamples, plus a matching bootstrapper."""
    s = dataclasses.replace(StatsSettings.from_config(cfg), n_boot=n_boot)
    return s, Bootstrapper(s.block_length, s.n_boot, s.boot_seed)


@pytest.mark.parametrize("k", [0, 1, 4])
def test_poisson_rate_ci_is_exact(k: int) -> None:
    """Garwood bounds: P(N ≥ k | lo) = α/2 and P(N ≤ k | hi) = α/2 (rates × exposure)."""
    T, a = 36.0, 0.05
    lo, hi = research.poisson_rate_ci(k, T, a)
    assert sps.poisson.cdf(k, hi * T) == pytest.approx(a / 2)
    if k == 0:
        assert lo == 0.0
    else:
        assert sps.poisson.sf(k - 1, lo * T) == pytest.approx(a / 2)


def test_breakeven_share_by_hand() -> None:
    """g = 1, L = −9 ⇒ f* = 1/10, and the mixture mean at f* is exactly zero."""
    f = research.breakeven_share(1.0, -9.0)
    assert f == pytest.approx(0.1)
    assert (1 - f) * 1.0 + f * -9.0 == pytest.approx(0.0)
    with pytest.raises(ValueError):
        research.breakeven_share(1.0, 0.5)


@pytest.mark.parametrize("seed", range(5))
def test_breakeven_vs_observed_matches_sign_of_mean(seed: int) -> None:
    """Identity: mean = (1 − f)·g + f·L, so f* > observed share ⇔ mean > 0."""
    rng = np.random.default_rng(seed)
    x = rng.standard_t(3, 300) * 0.5 + rng.choice([-0.3, 0.3])
    th = np.sort(x)[3]
    crash = x <= th
    f_obs, g, L = crash.mean(), x[~crash].mean(), x[crash].mean()
    assert (1 - f_obs) * g + f_obs * L == pytest.approx(x.mean())
    assert (research.breakeven_share(g, L) > f_obs) == (x.mean() > 0)


def test_s_breakeven_rate_toy() -> None:
    """x = [1, 1, 1, 1, −9], crash = −9 ⇒ g = 1 ⇒ f* = 0.1 a month = 1.2 a year; L fixed."""
    f = research.make_s_breakeven_rate(-9.0, -9.0, 12)
    x = np.array([[1.0, 1.0, 1.0, 1.0, -9.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    np.testing.assert_allclose(f(x), [1.2, 1.2])  # a resample without the crash keeps L


def _toy_monthly(n: int = 120) -> pd.DataFrame:
    """Constant ordinary payoff 0.5 (scaled) / 0.01 (raw) and one crash in the last month."""
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    xs = np.full(n, 0.5)
    xs[-1] = -10.0
    xv = np.full(n, 0.01)
    xv[-1] = -0.5
    return pd.DataFrame({"payoff_scaled": xs, "payoff_var": xv}, index=idx)


def test_breakeven_curve_toy(cfg) -> None:
    """Hand values on a series with one crash; constant ordinary months ⇒ a degenerate CI."""
    s, boot = _settings(cfg)
    rs = dataclasses.replace(
        research.ResearchSettings.from_config(cfg),
        breakeven_thresholds={"payoff_scaled": (-20.0, -1.0), "payoff_var": (-0.1,)},
    )
    m = _toy_monthly()
    c = research.breakeven_curve(m, s, rs, boot)
    # The threshold below the worst month (−20) is dropped; the worst month is always added.
    sc = c[c["measure"] == "payoff_scaled"]
    assert list(sc["threshold"]) == [-10.0, -1.0]
    assert sc["is_worst"].tolist() == [True, False]
    w = sc[sc["is_worst"]].iloc[0]
    rate = 12 * 0.5 / (0.5 + 10.0)
    assert w["n_crash"] == 1 and w["years"] == pytest.approx(10.0)
    assert w["breakeven_rate"] == pytest.approx(rate)
    assert w["breakeven_rate_lo"] == pytest.approx(rate)  # every resample has g = 0.5
    assert w["observed_rate"] == pytest.approx(0.1)
    assert w["p_conservative"] == pytest.approx(sps.poisson.cdf(1, rate * 10))
    assert w["worst_month"] == "2009-12"
    # −1 and −10 select the same single crash month here, so the rows agree.
    assert sc.iloc[1]["breakeven_rate"] == pytest.approx(rate)

    t = research.test_breakeven(m, c, s)
    yb = t[(t["measure"] == "payoff_scaled") & (t["statistic"] == "breakeven_years_between")]
    assert yb["estimate"].iloc[0] == pytest.approx(1 / rate)
    assert yb["null_value"].iloc[0] == pytest.approx(10.0)
    mp = t[(t["measure"] == "payoff_scaled") & (t["statistic"] == "months_of_premium_per_crash")]
    assert mp["estimate"].iloc[0] == pytest.approx(20.0)


# --------------------------------------------------------------------------- D. Jensen
def test_batched_ols_matches_lstsq() -> None:
    """Batched residuals equal numpy's least squares, row by row."""
    rng = np.random.default_rng(7)
    X = np.concatenate([np.ones((4, 50, 1)), rng.normal(size=(4, 50, 2))], axis=2)
    y = rng.normal(size=(4, 50))
    e = research.batched_ols_resid(y, X)
    for b in range(4):
        beta = np.linalg.lstsq(X[b], y[b], rcond=None)[0]
        np.testing.assert_allclose(e[b], y[b] - X[b] @ beta, atol=1e-10)


def test_smearing_ratio_lognormal() -> None:
    """Normal log residuals with variance σ² ⇒ E[√RV]/√E[RV] → exp(−σ²/8)."""
    rng = np.random.default_rng(11)
    sig = 0.6
    e = rng.normal(0, sig, (1, 400_000))
    assert research.smearing_ratio(e)[0] == pytest.approx(np.exp(-(sig**2) / 8), rel=2e-3)
    assert research.smearing_ratio(np.zeros((1, 10)))[0] == pytest.approx(1.0)


def test_uncond_gap() -> None:
    """Constant RV ⇒ no Jensen gap; two values by hand."""
    assert research.s_uncond_gap_pts(np.full((1, 5), 0.04))[0] == pytest.approx(0.0)
    rv = np.array([[0.01, 0.09]])  # vols 10 and 30; √mean = √0.05
    assert research.s_uncond_gap_pts(rv)[0] == pytest.approx(100 * (np.sqrt(0.05) - 0.2))


def test_jensen_recovers_fair_vix() -> None:
    """Simulate a VIX with zero premium (VIX² = E_t[RV]): all of the spread is convexity.

    log RV = m_t + e, e ~ N(0, σ²); VIX² = E_t[RV] = exp(m_t + σ²/2). The estimator should give
    a convexity share ≈ 1 and a 'fair VIX above realised' share equal to the actual share.
    """
    rng = np.random.default_rng(5)
    n, sig = 60_000, 0.65
    m = np.log(0.03) + 0.5 * rng.normal(size=n)
    iv = np.exp(m + sig**2 / 2)
    rv = np.exp(m + rng.normal(0, sig, n))
    rv_past = np.exp(m + rng.normal(0, 0.8, n))  # a noisy extra predictor
    vix, rvol = 100 * np.sqrt(iv), 100 * np.sqrt(rv)
    arrs = [a[None, :] for a in (np.log(rv), np.log(iv), np.log(rv_past), vix, rvol)]
    assert research.s_convexity_share(*arrs)[0] == pytest.approx(1.0, abs=0.04)
    fair = research.s_fair_vix_exceeds(*arrs)[0]
    assert fair == pytest.approx(research.s_vix_exceeds(vix[None], rvol[None])[0], abs=0.01)
    # A VIX that over-prices variance by 20% shows a share well below one.
    vix_hi = 100 * np.sqrt(1.2 * iv)
    arrs_hi = [a[None, :] for a in (np.log(rv), np.log(1.2 * iv), np.log(rv_past), vix_hi, rvol)]
    assert research.s_convexity_share(*arrs_hi)[0] < 0.6


def test_jensen_predictors_known_at_t() -> None:
    """The forecast that sizes the convexity uses only information at the close of t."""
    from vrp.stats import require_known_at_t

    require_known_at_t(research.JENSEN_PREDICTORS)
    assert not any("fwd" in c for c in research.JENSEN_PREDICTORS)


# --------------------------------------------------------------------------- D. case studies
def test_case_path_by_hand(cfg) -> None:
    """|r| = c daily, VIX 20: accrued RV on day k is (252/21)·k·c², reaching the implied 0.04
    on day ⌈0.04 / (12 c²)⌉ = 9 for c = 0.02, and day 21 equals rv_fwd21_ann_var."""
    c = 0.02
    r = np.where(np.arange(120) % 2 == 0, c, -c)
    panel = build_panel(*make_inputs(r, 20.0))
    t = panel.index[40]
    p = research.case_path(panel, t, cfg, pre_days=10)
    assert p["day"].iloc[0] == -10 and p["day"].iloc[-1] == 21
    win = p[p["day"] > 0]
    np.testing.assert_allclose(win["accrued_rv_ann_var"], 12 * np.arange(1, 22) * c**2)
    assert win["accrued_rv_ann_var"].iloc[-1] == pytest.approx(panel.loc[t, "rv_fwd21_ann_var"])
    assert p.loc[p["day"] == 0, "accrued_rv_ann_var"].iloc[0] == 0.0
    assert p.loc[p["day"] < 0, "accrued_rv_ann_var"].isna().all()
    assert win.loc[win["accrued_over_implied"] >= 1, "day"].iloc[0] == 9
    assert p.loc[p["day"] == 0, "spx_rel"].iloc[0] == pytest.approx(1.0)


def test_months_to_recover() -> None:
    """Cumulative payoff from the crash month: −3, −2, −1, 0 ⇒ recovered 3 months later."""
    idx = pd.date_range("2000-01-01", periods=6, freq="MS")
    x = pd.Series([1.0, -3.0, 1.0, 1.0, 1.0, 1.0], index=idx)
    assert research.months_to_recover(x, idx[1]) == 3
    assert np.isnan(research.months_to_recover(x.iloc[:4], idx[1]))


# --------------------------------------------------------------------------- D. real data
def test_real_breakeven_consistent(real_research, real_monthly) -> None:
    """Break-even rows agree with each other and with the sample mean (the identity above)."""
    c = real_research.breakeven
    for col, g in c.groupby("measure"):
        x = real_monthly[col].dropna()
        assert g["is_worst"].sum() == 1
        w = g[g["is_worst"]].iloc[0]
        assert w["threshold"] == pytest.approx(x.min())
        assert w["worst_month"] == f"{x.idxmin():%Y-%m}"
        assert w["years"] == pytest.approx(len(x) / 12)
        mix = (1 - g["n_crash"] / g["n"]) * g["ordinary_mean"] + g["n_crash"] / g["n"] * g[
            "crash_mean"
        ]
        np.testing.assert_allclose(mix, x.mean())
        assert ((g["breakeven_rate"] > g["observed_rate"]) == (x.mean() > 0)).all()
        assert (g["breakeven_rate_lo"] <= g["breakeven_rate"]).all()
        assert (g["breakeven_rate"] <= g["breakeven_rate_hi"]).all()
    t = real_research.tables
    yb = t[(t["test"] == "R_breakeven") & (t["statistic"] == "breakeven_years_between")]
    br = t[(t["test"] == "R_breakeven") & (t["statistic"] == "breakeven_rate_per_year")]
    np.testing.assert_allclose(yb["estimate"].to_numpy() * br["estimate"].to_numpy(), 1.0)


def test_real_jensen_sane(real_research) -> None:
    """Convexity is a positive part of the spread, below the unconditional upper bound."""
    t = real_research.tables.set_index(["test", "measure", "statistic"])["estimate"]
    share = t[("R_jensen", "rv_fwd21", "cond_convexity_share_of_spread")]
    assert 0 < share < 1
    assert (
        0
        < t[("R_jensen", "rv_fwd21", "cond_convexity_pts")]
        <= t[("R_jensen", "rv_fwd21", "uncond_jensen_gap_pts")]
    )
    fair = t[("R_jensen", "rv_fwd21", "fair_vix_share_above_realised")]
    actual = t[("R_jensen", "vol_spread_pts", "share_vix_above_realised")]
    assert 0.5 < fair < actual


def test_real_case_studies(real_research, real_monthly, cfg) -> None:
    """Each case month reproduces its sample row; 2020-03 is the worst month."""
    t = real_research.tables
    case = t[t["test"] == "R_case"].set_index(["sample", "statistic"])["estimate"]
    paths = real_research.case_paths
    for m in real_research.research.case_studies:
        mr = real_monthly[real_monthly.index.to_period("M") == pd.Period(m, "M")].iloc[0]
        assert case[(m, "payoff_scaled")] == pytest.approx(mr["payoff_scaled"])
        p = paths[paths["case"] == m]
        win = p[p["day"] > 0]
        assert len(win) == cfg.window
        assert win["accrued_rv_ann_var"].iloc[-1] == pytest.approx(mr["rv_fwd21_ann_var"])
        gone = win.loc[win["accrued_over_implied"] >= 1, "day"].iloc[0]
        assert case[(m, "premium_gone_by_day")] == gone
    assert case[("2020-03", "payoff_scaled_rank")] == 1


# --------------------------------------------------------------------------- headline sample
def test_headline_view_suffixes_everything(cfg) -> None:
    """The headline config starts at the first flagged month and suffixes every output."""
    from vrp.pipeline import headline_view

    idx = pd.date_range("2002-01-01", periods=30, freq="MS")
    m = pd.DataFrame({"post_2003": idx >= "2003-10-01"}, index=idx)
    hcfg, sub = headline_view(cfg, m)
    sfx = cfg["headline"]["suffix"]
    assert sub.index[0] == pd.Timestamp("2003-10-01") and sub["post_2003"].all()
    assert hcfg["dates"]["sample_start_month"] == "2003-10"
    assert hcfg["plots"]["name_suffix"] == sfx
    for sec in ("stats", "research"):
        for k, v in hcfg[sec]["outputs"].items():
            assert sfx in v and v != cfg[sec]["outputs"][k]
    # The full-sample config is untouched.
    assert cfg["dates"]["sample_start_month"] == "1990-01"
    assert "name_suffix" not in cfg["plots"]
    with pytest.raises(ValueError):
        headline_view(cfg, m.assign(post_2003=False))


def test_relabel_sample(cfg) -> None:
    """Sample labels and 'since' phrases move to the headline years; numbers and NaN do not."""
    from vrp.pipeline import headline_view, relabel_sample

    idx = pd.date_range("2003-10-01", periods=3, freq="MS")
    hcfg, _ = headline_view(cfg, pd.DataFrame({"post_2003": [True] * 3}, index=idx))
    df = pd.DataFrame(
        {
            "sample": ["monthly 1990–2025", "VIX > 25", None],
            "claim": ["risen since 1990", "x", "y"],
            "estimate": [1.0, 2.0, 3.0],
        }
    )
    out = relabel_sample(df, cfg, hcfg)
    assert out["sample"].iloc[0] == "monthly 2003–2025"
    assert out["sample"].iloc[1] == "VIX > 25"
    assert pd.isna(out["sample"].iloc[2])
    assert out["claim"].iloc[0] == "risen since 2003"
    assert out["estimate"].tolist() == [1.0, 2.0, 3.0]


def test_real_core_tests_headline(cfg, real_monthly) -> None:
    """Tests 1–4 on the live-VIX months: no robustness rows; claims C1–C18 only."""
    from vrp import stats
    from vrp.stats_report import evaluate_claims

    raw = copy.deepcopy(cfg.raw)
    raw["stats"]["bootstrap"]["n_resamples"] = 200
    sub = real_monthly[real_monthly["post_2003"].astype(bool)]
    res = stats.run_core_tests(sub, type(cfg)(raw=raw, root=cfg.root))
    assert not (res.tables["test"] == "5_robustness").any()
    assert res.tables.loc[res.tables["test"] == "1_mean", "n"].max() == len(sub)
    claims = evaluate_claims(res, [str(t) for t in cfg["headline"]["stats_tests"]])
    assert list(claims["id"]) == [f"C{i}" for i in range(1, 19)]
