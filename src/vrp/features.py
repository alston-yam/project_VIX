"""Returns, implied/realised variance, payoffs and forward/trailing windows.

Every formula here matches ``docs/data_dictionary.md``. Columns starting with
``fwd_``/``rv_fwd``/``payoff``/``vol_spread`` use data **after** day t and are
outcomes; nothing else looks forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vrp.logging_setup import get_logger

log = get_logger(__name__)

PANEL_COLUMNS: list[str] = [
    "spx_close", "vix_close", "vxo_close", "ret_log", "ret_spans_dropped_day",
    "iv_ann_var", "rv_past21_ann_var",
    "rv_fwd21_ann_var", "rv_fwd21_ann_vol_pct", "payoff_var", "payoff_scaled",
    "vol_spread_pts", "spx_fwd21_ret_log",
    "fwd_window_start", "fwd_window_end", "fwd_window_complete",
]


def log_returns(close: pd.Series, prev_close: pd.Series) -> pd.Series:
    """ret_log_t = ln(P_t / P_{t-1})."""
    return np.log(close / prev_close)


def implied_variance(vix: pd.Series) -> pd.Series:
    """Annualised implied variance (VIX_t / 100)^2."""
    return (vix / 100.0) ** 2


def trailing_realised_variance(ret: pd.Series, window: int, ann: int) -> pd.Series:
    """(ann / window) * sum of r^2 over t-window+1 .. t (inclusive of t)."""
    return (ann / window) * ret.pow(2).rolling(window, min_periods=window).sum()


def forward_realised_variance(ret: pd.Series, window: int, ann: int) -> pd.Series:
    """(ann / window) * sum of r^2 over t+1 .. t+window (strictly after t).

    The trailing sum at row t+window covers t+1..t+window, so shifting it back
    by ``window`` rows places it on row t. Rows without ``window`` future
    returns are NaN.
    """
    return trailing_realised_variance(ret, window, ann).shift(-window)


def add_features(panel: pd.DataFrame, window: int, ann: int) -> pd.DataFrame:
    """Add every daily-panel feature column to a merged VIX/S&P panel."""
    df = panel.copy()
    df["ret_log"] = log_returns(df["spx_close"], df["spx_prev_close"])
    df["iv_ann_var"] = implied_variance(df["vix_close"])
    df["rv_past21_ann_var"] = trailing_realised_variance(df["ret_log"], window, ann)

    # ---- outcomes: everything below uses data strictly after t -------------
    df["rv_fwd21_ann_var"] = forward_realised_variance(df["ret_log"], window, ann)
    df["rv_fwd21_ann_vol_pct"] = 100.0 * np.sqrt(df["rv_fwd21_ann_var"])
    df["payoff_var"] = df["iv_ann_var"] - df["rv_fwd21_ann_var"]
    df["payoff_scaled"] = df["payoff_var"] / df["iv_ann_var"]
    df["vol_spread_pts"] = df["vix_close"] - df["rv_fwd21_ann_vol_pct"]
    df["spx_fwd21_ret_log"] = np.log(df["spx_close"].shift(-window) / df["spx_close"])

    dates = pd.Series(df.index, index=df.index)
    df["fwd_window_start"] = dates.shift(-1)
    df["fwd_window_end"] = dates.shift(-window)
    n_future = df["ret_log"].notna().astype(int)[::-1].rolling(window, min_periods=1).sum()[::-1]
    n_future = n_future.shift(-1)  # returns available in t+1..t+window
    df["fwd_window_complete"] = (n_future == window) & df["fwd_window_end"].notna()

    if "vxo_close" not in df:
        df["vxo_close"] = np.nan
    out = df[PANEL_COLUMNS]
    log.info("features: %d rows, %d with complete forward window, %d with trailing window",
             len(out), int(out["fwd_window_complete"].sum()),
             int(out["rv_past21_ann_var"].notna().sum()))
    return out
