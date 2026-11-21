"""Look-ahead and window-alignment tests.

These are the guarantees in docs/methodology.md, "No look-ahead". Each runs on
a synthetic panel (always) and, where useful, on the real processed panel
(skipped if `make data` has not been run).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import ANN, WINDOW, build_panel, cfg_with

from vrp import sampling

KNOWN_AT_T = ["spx_close", "vix_close", "ret_log", "iv_ann_var", "rv_past21_ann_var"]
OUTCOMES = [
    "rv_fwd21_ann_var",
    "rv_fwd21_ann_vol_pct",
    "payoff_var",
    "payoff_scaled",
    "vol_spread_pts",
    "spx_fwd21_ret_log",
]


# --------------------------------------------------------------------------- window dates
def _check_window_dates(panel: pd.DataFrame) -> None:
    complete = panel[panel["fwd_window_complete"]]
    assert len(complete) > 0
    assert (complete["fwd_window_start"] > complete.index).all()
    pos = pd.Series(np.arange(len(panel)), index=panel.index)
    t_pos = pos.loc[complete.index].to_numpy()
    assert (pos.loc[complete["fwd_window_start"]].to_numpy() == t_pos + 1).all()
    assert (pos.loc[complete["fwd_window_end"]].to_numpy() == t_pos + WINDOW).all()


def test_window_dates_synthetic(random_panel: pd.DataFrame) -> None:
    _check_window_dates(random_panel)


def test_window_dates_real(real_panel: pd.DataFrame) -> None:
    _check_window_dates(real_panel)


def test_incomplete_windows_only_at_end(random_panel: pd.DataFrame) -> None:
    inc = random_panel[~random_panel["fwd_window_complete"]]
    assert len(inc) == WINDOW
    assert inc.index.equals(random_panel.index[-WINDOW:])
    assert inc[OUTCOMES].isna().all().all()


# --------------------------------------------------------------------------- hand computations
def _hand_rv_fwd(panel: pd.DataFrame, i: int) -> float:
    r = panel["ret_log"].to_numpy()
    return ANN / WINDOW * float(np.sum(r[i + 1 : i + 1 + WINDOW] ** 2))


def _hand_rv_past(panel: pd.DataFrame, i: int) -> float:
    r = panel["ret_log"].to_numpy()
    return ANN / WINDOW * float(np.sum(r[i - WINDOW + 1 : i + 1] ** 2))


@pytest.mark.parametrize("which", ["synthetic", "real"])
def test_rv_fwd_matches_hand_computation(which: str, request: pytest.FixtureRequest) -> None:
    panel = request.getfixturevalue("random_panel" if which == "synthetic" else "real_panel")
    rng = np.random.default_rng(7)
    idx = rng.choice(np.flatnonzero(panel["fwd_window_complete"].to_numpy()), 20, replace=False)
    for i in idx:
        assert abs(panel["rv_fwd21_ann_var"].iloc[i] - _hand_rv_fwd(panel, i)) < 1e-12


@pytest.mark.parametrize("which", ["synthetic", "real"])
def test_rv_past_uses_only_dates_up_to_t(which: str, request: pytest.FixtureRequest) -> None:
    panel = request.getfixturevalue("random_panel" if which == "synthetic" else "real_panel")
    assert panel["rv_past21_ann_var"].iloc[: WINDOW - 1].isna().all()
    rng = np.random.default_rng(11)
    idx = rng.choice(np.arange(WINDOW - 1, len(panel)), 20, replace=False)
    for i in idx:
        assert abs(panel["rv_past21_ann_var"].iloc[i] - _hand_rv_past(panel, i)) < 1e-12


def test_spx_fwd_ret_matches_window(random_panel: pd.DataFrame) -> None:
    c = random_panel[random_panel["fwd_window_complete"]]
    expected = np.log(
        random_panel.loc[c["fwd_window_end"], "spx_close"].to_numpy() / c["spx_close"].to_numpy()
    )
    np.testing.assert_allclose(c["spx_fwd21_ret_log"].to_numpy(), expected, atol=1e-12)


# --------------------------------------------------------------------------- mutation tests
def test_mutating_after_window_does_not_change_row_t(
    random_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    vix, spx = random_inputs
    base = build_panel(vix, spx)
    t_pos = 100
    t = base.index[t_pos]
    # Mutate every S&P close and VIX value strictly after t+21.
    cutoff = base.index[t_pos + WINDOW]
    spx2, vix2 = spx.copy(), vix.copy()
    spx2.loc[spx2.index > cutoff, "close"] *= 1.5
    vix2.loc[vix2.index > cutoff] += 30
    mutated = build_panel(vix2, spx2)
    cols = KNOWN_AT_T + OUTCOMES
    pd.testing.assert_series_equal(base.loc[t, cols], mutated.loc[t, cols])
    # Sanity: the mutation did change something later on.
    assert not np.isclose(
        base.loc[cutoff:, "spx_close"].iloc[-1], mutated.loc[cutoff:, "spx_close"].iloc[-1]
    )


def test_mutating_t_plus_1_changes_only_outcomes(
    random_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    vix, spx = random_inputs
    base = build_panel(vix, spx)
    t_pos = 100
    t, t1 = base.index[t_pos], base.index[t_pos + 1]
    spx2, vix2 = spx.copy(), vix.copy()
    spx2.loc[t1, "close"] *= 1.05
    vix2.loc[t1] += 10
    mutated = build_panel(vix2, spx2)
    assert mutated.loc[t, "rv_fwd21_ann_var"] != pytest.approx(
        base.loc[t, "rv_fwd21_ann_var"], abs=1e-15
    )
    for col in ["iv_ann_var", "rv_past21_ann_var", "vix_close", "spx_close", "ret_log"]:
        assert mutated.loc[t, col] == base.loc[t, col], col


def test_return_on_day_t_not_in_forward_window(
    random_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    """Changing P_{t-1} changes ret_log at t (and rv_past21) but never rv_fwd21 at t."""
    vix, spx = random_inputs
    base = build_panel(vix, spx)
    t_pos = 100
    t, t_prev = base.index[t_pos], base.index[t_pos - 1]
    spx2 = spx.copy()
    spx2.loc[t_prev, "close"] *= 0.9
    mutated = build_panel(vix, spx2)
    assert mutated.loc[t, "ret_log"] != base.loc[t, "ret_log"]
    assert mutated.loc[t, "rv_past21_ann_var"] != base.loc[t, "rv_past21_ann_var"]
    for col in OUTCOMES:
        assert mutated.loc[t, col] == base.loc[t, col], col


def test_mutating_t_changes_iv_not_rv_fwd_prices(
    random_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    """VIX at t drives iv_ann_var; it must not enter the realised window."""
    vix, spx = random_inputs
    base = build_panel(vix, spx)
    t = base.index[100]
    vix2 = vix.copy()
    vix2.loc[t] += 5
    mutated = build_panel(vix2, spx)
    assert mutated.loc[t, "iv_ann_var"] != base.loc[t, "iv_ann_var"]
    assert mutated.loc[t, "rv_fwd21_ann_var"] == base.loc[t, "rv_fwd21_ann_var"]


# --------------------------------------------------------------------------- monthly sample
def test_monthly_windows_within_available_data_real(
    real_panel: pd.DataFrame, real_monthly: pd.DataFrame
) -> None:
    assert (real_monthly["fwd_window_end"] <= real_panel.index.max()).all()
    assert real_monthly["fwd_window_complete"].all()
    assert (real_monthly["fwd_window_start"] > real_monthly.index).all()


def test_monthly_drops_month_with_incomplete_window(random_panel: pd.DataFrame, cfg) -> None:
    first, last = random_panel.index[0], random_panel.index[-1]
    c = cfg_with(
        cfg, sample_start_month=first.strftime("%Y-%m"), sample_end_month=last.strftime("%Y-%m")
    )
    m = sampling.build_monthly_sample(random_panel, c)
    assert (m["fwd_window_end"] <= last).all()
    # The final month's first trading day lies within 21 days of the end -> dropped.
    last_first_day = sampling.first_trading_days(random_panel).index[-1]
    assert (random_panel.index > last_first_day).sum() < WINDOW
    assert last_first_day not in m.index


def test_monthly_rows_are_first_trading_days(
    real_panel: pd.DataFrame, real_monthly: pd.DataFrame
) -> None:
    firsts = sampling.first_trading_days(real_panel).index
    assert real_monthly.index.isin(firsts).all()
    assert real_monthly["month"].is_unique
