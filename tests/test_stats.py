"""v2 statistics: hand calculations on toy data, look-ahead guards, real-data consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import ANN, WINDOW, build_panel, cfg_with, make_inputs
from scipy import stats as sps

from vrp import features, sampling, stats, stats_report

# --------------------------------------------------------------------------- Test 1 maths


def test_mean_iid_matches_hand_formula() -> None:
    x = np.array([0.5, 0.2, -0.1, 0.4, 0.3, 0.1])
    r = stats.mean_iid(x)
    se = x.std(ddof=1) / np.sqrt(len(x))
    assert r["mean"] == pytest.approx(x.mean())
    assert r["se"] == pytest.approx(se)
    assert r["t"] == pytest.approx(x.mean() / se)
    assert r["p"] == pytest.approx(sps.ttest_1samp(x, 0).pvalue)
    q = sps.t.ppf(0.975, len(x) - 1)
    assert r["ci_low"] == pytest.approx(x.mean() - q * se)


def test_newey_west_zero_lags_equals_iid_se() -> None:
    x = np.random.default_rng(1).normal(0.1, 1, 200)
    assert stats.mean_nw(x, 0)["se"] == pytest.approx(stats.mean_iid(x)["se"], rel=1e-12)


@pytest.mark.parametrize("lags", [1, 3, 6])
def test_newey_west_matches_hand_bartlett(lags: int) -> None:
    rng = np.random.default_rng(7)
    e = rng.normal(size=300)
    x = np.empty(300)
    x[0] = e[0]
    for i in range(1, 300):  # AR(1): positive autocorrelation widens the NW SE
        x[i] = 0.5 * x[i - 1] + e[i]
    nw = stats.mean_nw(x, lags)
    assert nw["se"] == pytest.approx(stats.newey_west_se_by_hand(x, lags), rel=1e-10)
    assert nw["se"] > stats.mean_iid(x)["se"]


def test_mean_excl_worst_and_median_toy() -> None:
    x = np.array([[1.0, 2.0, 3.0, -10.0, 4.0, -20.0]])
    assert stats.make_s_mean_excl_worst(2)(x)[0] == pytest.approx(2.5)
    assert stats.s_median(x)[0] == pytest.approx(1.5)
    assert stats.s_mean_minus_median(x)[0] == pytest.approx(x.mean() - 1.5)


# --------------------------------------------------------------------------- bootstrap


def test_block_bootstrap_indices_are_circular_blocks() -> None:
    idx = stats.block_bootstrap_indices(10, 3, 50, np.random.default_rng(0))
    assert idx.shape == (50, 10)
    assert idx.min() >= 0 and idx.max() <= 9
    for r in idx:  # within each block of 3, consecutive (mod n)
        for b in range(0, 9, 3):
            assert (r[b + 1] - r[b]) % 10 == 1 and (r[b + 2] - r[b + 1]) % 10 == 1


def test_bootstrapper_is_reproducible_and_sensible() -> None:
    rng = np.random.default_rng(3)
    pos = rng.normal(1.0, 1.0, 300)
    zero = rng.normal(0.0, 1.0, 300)
    b1 = stats.Bootstrapper(6, 2000, 11)
    b2 = stats.Bootstrapper(6, 2000, 11)
    r1, r2 = b1.run(stats.s_mean, pos), b2.run(stats.s_mean, pos)
    assert r1 == r2
    assert r1["ci_low"] < pos.mean() < r1["ci_high"]
    assert r1["ci_low"] > 0 and r1["p"] < 0.01
    assert b1.run(stats.s_mean, zero - zero.mean())["p"] > 0.5


def test_bootstrap_ci_width_close_to_analytic_for_iid_normal() -> None:
    x = np.random.default_rng(5).normal(0, 1, 400)
    b = stats.Bootstrapper(1, 4000, 2).run(stats.s_mean, x)
    a = stats.mean_iid(x)
    assert (b["ci_high"] - b["ci_low"]) == pytest.approx(a["ci_high"] - a["ci_low"], rel=0.1)


# --------------------------------------------------------------------------- Test 2 maths


def test_vectorised_moments_match_scipy() -> None:
    x = np.random.default_rng(9).standard_t(4, size=(3, 200))
    np.testing.assert_allclose(stats.s_skew(x), [sps.skew(r, bias=False) for r in x])
    np.testing.assert_allclose(stats.s_exkurt(x), [sps.kurtosis(r, bias=False) for r in x])


def test_win_rate_gain_loss_toy() -> None:
    x = np.array([[0.5, 0.5, 0.5, -3.0]])
    assert stats.s_win_rate(x)[0] == pytest.approx(0.75)
    assert stats.s_avg_gain(x)[0] == pytest.approx(0.5)
    assert stats.s_avg_loss(x)[0] == pytest.approx(-3.0)
    assert stats.s_loss_gain_ratio(x)[0] == pytest.approx(6.0)
    assert stats.s_sharpe_annual(x)[0] == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(12))


# --------------------------------------------------------------------------- Test 3 maths


def test_corr_and_spearman_match_scipy() -> None:
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=100), rng.normal(size=100)
    b = b + 0.5 * a
    assert stats.s_corr(a[None], b[None])[0] == pytest.approx(sps.pearsonr(a, b).statistic)
    assert stats.s_spearman(a[None], b[None])[0] == pytest.approx(sps.spearmanr(a, b).statistic)


def test_loss_share_and_diff_means_toy() -> None:
    x = np.array([[1.0, 1.0, -2.0, -6.0, 1.0]])
    g = np.array([[False, False, False, True, False]])
    assert stats.s_loss_share(x, g)[0] == pytest.approx(0.75)  # 6 of 8 total loss
    assert stats.s_diff_means(x, g)[0] == pytest.approx(-6.0 - (1 + 1 - 2 + 1) / 4)


def test_permutation_detects_clustering_and_not_noise() -> None:
    rng = np.random.default_rng(12)
    n = 200
    g = np.zeros(n, dtype=bool)
    g[:20] = True
    x = rng.normal(0.5, 0.2, n)
    x[:20] = -2.0  # every loss is in the flagged 10%
    clustered = stats.permutation_p(stats.s_loss_share, x, g, 0.10, 2000, 1)
    assert clustered["estimate"] == pytest.approx(1.0, abs=1e-9)
    assert clustered["p"] < 0.01
    y = rng.normal(0.0, 1.0, n)
    noise = stats.permutation_p(stats.s_loss_share, y, g, 0.10, 2000, 1)
    assert noise["perm_mean"] == pytest.approx(0.10, abs=0.02)
    assert noise["p"] > 0.05


# --------------------------------------------------------------------------- Test 4 maths


def test_mincer_zarnowitz_unbiased_and_biased() -> None:
    rng = np.random.default_rng(8)
    iv = pd.Series(rng.uniform(0.01, 0.09, 500), name="iv_ann_var")
    unbiased = iv + rng.normal(0, 0.005, 500)
    r = stats.mincer_zarnowitz(unbiased, iv, 6, 0.05)
    assert r["b"] == pytest.approx(1.0, abs=0.05)
    assert r["p_wald"] > 0.01
    biased = 0.6 * iv + rng.normal(0, 0.005, 500)
    r2 = stats.mincer_zarnowitz(biased, iv, 6, 0.05)
    assert r2["b"] == pytest.approx(0.6, abs=0.05)
    assert r2["p_wald"] < 1e-6


@pytest.mark.parametrize(
    "col", ["rv_fwd21_ann_var", "payoff_scaled", "spx_fwd21_ret_log", "is_selloff", "my_fwd_x"]
)
def test_outcome_columns_cannot_be_predictors(col: str) -> None:
    with pytest.raises(ValueError, match="look-ahead"):
        stats.require_known_at_t([col])
    y = pd.Series(np.arange(10.0))
    with pytest.raises(ValueError, match="look-ahead"):
        stats.mincer_zarnowitz(y, pd.Series(np.arange(10.0), name=col), 1, 0.05)


def test_known_at_t_predictors_pass() -> None:
    stats.require_known_at_t(["iv_ann_var", "vix_close", "rv_past21_ann_var"])


# --------------------------------------------------------------------------- multiple testing


def test_holm_hand_values() -> None:
    np.testing.assert_allclose(stats.holm([0.01, 0.04, 0.03], 0.05), [0.03, 0.06, 0.06])


# --------------------------------------------------------------------------- Test 5 variants


def test_demeaned_forward_rv_hand_calc(random_panel: pd.DataFrame) -> None:
    r = random_panel["ret_log"]
    dm = features.forward_realised_variance_demeaned(r, WINDOW, ANN)
    for i in [0, 50, 200]:
        w = r.iloc[i + 1 : i + 1 + WINDOW].to_numpy()
        assert dm.iloc[i] == pytest.approx(ANN * w.var(ddof=1), rel=1e-12)
    assert dm.iloc[len(r) - WINDOW :].isna().all()


def test_h22_forward_rv_hand_calc(random_panel: pd.DataFrame) -> None:
    r = random_panel["ret_log"]
    h22 = features.forward_realised_variance(r, 22, ANN)
    for i in [0, 77, 250]:
        w = r.iloc[i + 1 : i + 23].to_numpy()
        assert h22.iloc[i] == pytest.approx(ANN / 22 * (w**2).sum(), rel=1e-12)
    comp = features.forward_window_complete(r, 22)
    assert comp.iloc[: len(r) - 22].all() and not comp.iloc[len(r) - 22 :].any()


def _variants(vix: pd.Series, spx: pd.DataFrame, cfg) -> pd.DataFrame:  # noqa: ANN001
    panel = build_panel(vix, spx)
    c = cfg_with(cfg, sample_start_month="2000-01", sample_end_month="2000-10")
    m = sampling.build_monthly_sample(panel, c)
    return stats.build_monthly_variants(panel, m, c)


def test_variants_have_no_look_ahead(cfg, random_inputs) -> None:  # noqa: ANN001
    """Changing anything after t+22 (or day t's own return) leaves row t unchanged."""
    vix, spx = random_inputs
    base = _variants(vix, spx, cfg)
    t = base.index[2]
    pos = spx.index.get_loc(t)
    cols = ["payoff_var_h22", "payoff_var_dm", "payoff_scaled_h22", "payoff_scaled_dm"]

    later = spx.copy()
    later.iloc[pos + 23 :, :4] *= 1.5  # every price after t+22
    v2 = vix.copy()
    v2[v2.index > t] = 80.0  # every VIX after t
    after = _variants(v2, later, cfg)
    pd.testing.assert_series_equal(base.loc[t, cols], after.loc[t, cols])

    # Scaling every price before t changes day t's own return, but no forward return.
    past = spx.copy()
    past.iloc[:pos, :4] *= 1.3
    before = _variants(vix, past, cfg)
    for c in ["rv_fwd22_ann_var", "rv_fwd21_dm_ann_var"]:
        assert before.loc[t, c] == pytest.approx(base.loc[t, c], rel=1e-12)

    # ...but moving the price at t+1 must change them.
    nxt = spx.copy()
    nxt.iloc[pos + 1, :4] *= 1.05
    moved = _variants(vix, nxt, cfg)
    assert moved.loc[t, "rv_fwd22_ann_var"] != pytest.approx(base.loc[t, "rv_fwd22_ann_var"])


def test_constant_move_variants(cfg) -> None:  # noqa: ANN001
    """Constant 1% daily move: H22 RV = 0.0252, demeaned RV = 0 (no dispersion)."""
    v = _variants(*make_inputs(np.full(260, 0.01)), cfg)
    np.testing.assert_allclose(v["rv_fwd22_ann_var"], 0.0252, rtol=1e-12)
    np.testing.assert_allclose(v["rv_fwd21_dm_ann_var"], 0.0, atol=1e-12)


# --------------------------------------------------------------------------- real data


def test_real_variants_consistent(cfg, real_panel, real_monthly) -> None:  # noqa: ANN001
    v = stats.build_monthly_variants(real_panel, real_monthly, cfg)
    assert len(v) == len(real_monthly)
    assert v["rv_fwd22_ann_var"].notna().all() and v["rv_fwd21_dm_ann_var"].notna().all()
    # H = 21 recomputed through the same helper equals the v1 column.
    rv21 = features.forward_realised_variance(real_panel["ret_log"], 21, ANN)
    np.testing.assert_allclose(rv21.loc[v.index], v["rv_fwd21_ann_var"], rtol=1e-10)
    # Demeaned RV ≤ non-demeaned × 21/20 (subtracting the mean can only lower Σ(r−r̄)²).
    assert (v["rv_fwd21_dm_ann_var"] <= v["rv_fwd21_ann_var"] * 21 / 20 + 1e-15).all()
    # VXO available only up to its discontinuation.
    assert v.loc[v["iv_vxo_ann_var"].notna()].index.max() < pd.Timestamp("2021-10-01")


def test_real_daily_sample_window(cfg, real_panel) -> None:  # noqa: ANN001
    d = stats.daily_sample(real_panel, cfg)
    assert d.index.min() >= pd.Timestamp("1990-01-01")
    assert d.index.max() <= pd.Timestamp("2025-12-31")
    assert d["fwd_window_complete"].all()


@pytest.fixture(scope="module")
def real_results(cfg, real_panel, real_monthly):  # noqa: ANN001, ANN201
    """Full test run on the real data with fewer resamples (fast)."""
    import copy

    from vrp.config import Config

    raw = copy.deepcopy(cfg.raw)
    raw["stats"]["bootstrap"]["n_resamples"] = 1000
    raw["stats"]["permutation"]["n_resamples"] = 1000
    c = Config(raw=raw, root=cfg.root)
    return stats.run_all_tests(real_panel, real_monthly, c)


def test_real_results_shape_and_ci_order(real_results) -> None:  # noqa: ANN001
    t = real_results.tables
    assert set(t["test"]) == {"1_mean", "2_shape", "3_crash", "4_forecast", "5_robustness"}
    ci = t.dropna(subset=["ci_low", "ci_high"])
    assert (ci["ci_low"] <= ci["ci_high"]).all()
    p = t["p_value"].dropna()
    assert ((p >= 0) & (p <= 1)).all()


def test_real_point_estimates_match_direct_computation(
    real_results, real_monthly
) -> None:  # noqa: ANN001
    t = real_results.tables

    def est(test: str, measure: str, stat: str) -> float:
        r = t[(t["test"] == test) & (t["measure"] == measure) & (t["statistic"] == stat)]
        return float(r["estimate"].iloc[0])

    m = real_monthly
    assert est("1_mean", "payoff_scaled", "mean") == pytest.approx(m["payoff_scaled"].mean())
    assert est("1_mean", "payoff_var", "median") == pytest.approx(m["payoff_var"].median())
    assert est("2_shape", "payoff_scaled", "win_rate") == pytest.approx(
        (m["payoff_scaled"] > 0).mean()
    )
    assert est("3_crash", "payoff_scaled", "corr_with_spx_fwd21_ret") == pytest.approx(
        m["payoff_scaled"].corr(m["spx_fwd21_ret_log"])
    )
    worst = m["payoff_scaled"].nsmallest(5).index
    assert est("1_mean", "payoff_scaled", "mean_excl_worst_5") == pytest.approx(
        m["payoff_scaled"].drop(worst).mean()
    )


def test_claims_rule_is_applied(real_results) -> None:  # noqa: ANN001
    claims = stats_report.evaluate_claims(real_results)
    a = real_results.settings.alpha
    sup = claims[claims["supported"]]
    assert (sup["ci_ok"] & (sup["p_holm"] < a)).all()
    assert (claims["p_holm"] >= claims["p_max"] - 1e-15).all()
    uns = claims[~claims["supported"]]
    assert ((~uns["ci_ok"]) | (uns["p_holm"] >= a)).all()
