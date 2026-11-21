"""Monthly sample: the first trading day of each month from the daily panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vrp.config import Config
from vrp.logging_setup import get_logger

log = get_logger(__name__)


def first_trading_days(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the first panel row of each calendar month."""
    return panel.groupby(panel.index.to_period("M"), sort=True).head(1)


def build_monthly_sample(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per month (first trading day) with a complete forward window."""
    d = cfg["dates"]
    f = cfg["features"]
    first = first_trading_days(panel)
    months = first.index.to_period("M")
    in_range = (months >= pd.Period(d["sample_start_month"], "M")) & (
        months <= pd.Period(d["sample_end_month"], "M"))
    m = first[in_range]
    n_in = len(m)
    dropped = m.index[~m["fwd_window_complete"]]
    m = m[m["fwd_window_complete"]].copy()
    log.info("sampling: %d first-trading-days in %s..%s; %d dropped (incomplete window); "
             "%d monthly rows", n_in, d["sample_start_month"], d["sample_end_month"],
             len(dropped), len(m))
    if len(dropped):
        log.warning("sampling: incomplete forward windows on %s",
                    ", ".join(str(x.date()) for x in dropped))

    m.insert(0, "month", m.index.strftime("%Y-%m"))
    m["is_selloff"] = m["spx_fwd21_ret_log"] < float(f["selloff_threshold_log"])
    n_dec = int(f["n_deciles"])
    m["selloff_decile"] = (pd.qcut(m["spx_fwd21_ret_log"].rank(method="first"), n_dec,
                                   labels=False) + 1).astype(int)
    m["post_2003"] = m.index >= pd.Timestamp(d["vix_new_methodology_date"])
    return m


def window_overlap(monthly: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Trading-day overlap between consecutive monthly forward windows.

    ``overlap_days`` > 0: the next window starts before this one ends (shared
    returns). < 0: that many trading days fall in neither window. 0: windows
    tile exactly.
    """
    pos = pd.Series(np.arange(len(panel)), index=panel.index)
    end_pos = pos.loc[monthly["fwd_window_end"]].to_numpy()
    start_pos = pos.loc[monthly["fwd_window_start"]].to_numpy()
    overlap = end_pos[:-1] - start_pos[1:] + 1
    return pd.DataFrame({"month": monthly["month"].iloc[:-1].to_numpy(),
                         "next_month": monthly["month"].iloc[1:].to_numpy(),
                         "overlap_days": overlap})


def overlap_summary(ov: pd.DataFrame) -> dict[str, float | int]:
    """Summary statistics of consecutive-window overlap (in trading days)."""
    o = ov["overlap_days"]
    return {
        "pairs": int(len(o)),
        "overlapping_pairs": int((o > 0).sum()),
        "exact_tiling_pairs": int((o == 0).sum()),
        "gap_pairs": int((o < 0).sum()),
        "max_overlap_days": int(o.max()),
        "max_gap_days": int(-o.min()) if o.min() < 0 else 0,
        "mean_overlap_days": float(o.mean()),
        "total_shared_days": int(o.clip(lower=0).sum()),
        "total_uncovered_days": int((-o).clip(lower=0).sum()),
    }
