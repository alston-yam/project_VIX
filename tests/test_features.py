"""Feature formulas versus hand calculations and the primer's worked examples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import ANN, WINDOW, build_panel, cfg_with, make_inputs

from vrp import features, sampling


def _complete(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["fwd_window_complete"]]


def test_constant_one_percent_move_gives_0_0252() -> None:
    vix, spx = make_inputs(np.full(80, 0.01))
    p = build_panel(vix, spx)
    np.testing.assert_allclose(p["ret_log"], 0.01, atol=1e-15)
    np.testing.assert_allclose(_complete(p)["rv_fwd21_ann_var"], 252 * 0.0001, rtol=1e-12)
    np.testing.assert_allclose(p["rv_past21_ann_var"].dropna(), 0.0252, rtol=1e-12)


def test_alternating_sign_same_variance() -> None:
    """Non-demeaned: direction does not matter, only size."""
    r = np.tile([0.01, -0.01], 40)
    p = build_panel(*make_inputs(r))
    np.testing.assert_allclose(_complete(p)["rv_fwd21_ann_var"], 0.0252, rtol=1e-12)


def _panel_with_rv_vol(rv_vol: float, vix: float = 20.0) -> pd.DataFrame:
    d = rv_vol / np.sqrt(ANN)  # daily move so that 252 * d^2 = rv_vol^2
    r = np.tile([d, -d], 40)
    return build_panel(*make_inputs(r, vix))


def test_primer_calm_example() -> None:
    """VIX 20, realised vol 15: payoff 0.0175, scaled +0.4375."""
    row = _complete(_panel_with_rv_vol(0.15)).iloc[0]
    assert row["iv_ann_var"] == pytest.approx(0.04, abs=1e-15)
    assert row["rv_fwd21_ann_var"] == pytest.approx(0.0225, rel=1e-12)
    assert row["payoff_var"] == pytest.approx(0.0175, rel=1e-10)
    assert row["payoff_scaled"] == pytest.approx(0.4375, rel=1e-10)
    assert row["rv_fwd21_ann_vol_pct"] == pytest.approx(15.0, rel=1e-12)
    assert row["vol_spread_pts"] == pytest.approx(5.0, rel=1e-10)


def test_primer_crash_example() -> None:
    """VIX 20, realised vol 40: payoff -0.12, scaled -3.0."""
    row = _complete(_panel_with_rv_vol(0.40)).iloc[0]
    assert row["payoff_var"] == pytest.approx(-0.12, rel=1e-10)
    assert row["payoff_scaled"] == pytest.approx(-3.0, rel=1e-10)
    assert row["vol_spread_pts"] == pytest.approx(-20.0, rel=1e-10)


def test_scaled_payoff_identity(random_panel: pd.DataFrame) -> None:
    c = _complete(random_panel)
    np.testing.assert_allclose(
        c["payoff_scaled"], 1 - c["rv_fwd21_ann_var"] / c["iv_ann_var"], rtol=1e-12
    )
    assert (c["payoff_scaled"] <= 1).all()


def test_implied_variance_units() -> None:
    s = pd.Series([10.0, 20.0, 82.69])
    np.testing.assert_allclose(features.implied_variance(s), [0.01, 0.04, 0.8269**2])


def test_fwd_log_return_sums_window_returns(random_panel: pd.DataFrame) -> None:
    r = random_panel["ret_log"].to_numpy()
    for i in [0, 50, 200]:
        assert random_panel["spx_fwd21_ret_log"].iloc[i] == pytest.approx(
            r[i + 1 : i + 1 + WINDOW].sum(), abs=1e-12
        )


def test_monthly_flags(random_panel: pd.DataFrame, cfg) -> None:
    c = cfg_with(
        cfg,
        sample_start_month=random_panel.index[0].strftime("%Y-%m"),
        sample_end_month=random_panel.index[-1].strftime("%Y-%m"),
        vix_new_methodology_date="2000-06-01",
    )
    m = sampling.build_monthly_sample(random_panel, c)
    assert (m["is_selloff"] == (m["spx_fwd21_ret_log"] < -0.05)).all()
    assert m["selloff_decile"].between(1, 10).all()
    # decile 1 holds the worst forward return.
    assert m.loc[m["spx_fwd21_ret_log"].idxmin(), "selloff_decile"] == 1
    assert m.loc[m["spx_fwd21_ret_log"].idxmax(), "selloff_decile"] == 10
    assert (m["post_2003"] == (m.index >= "2000-06-01")).all()
    assert (m["month"] == m.index.strftime("%Y-%m")).all()


def test_overlap_measure_tiles() -> None:
    """Windows of 21 days started every 21 rows tile exactly (overlap 0)."""
    p = build_panel(*make_inputs(np.full(200, 0.001)))
    m = p.iloc[::WINDOW]
    m = m[m["fwd_window_complete"]].assign(month=lambda d: d.index.strftime("%Y-%m-%d"))
    ov = sampling.window_overlap(m, p)
    assert (ov["overlap_days"] == 0).all()
