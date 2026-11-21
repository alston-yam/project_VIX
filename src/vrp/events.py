"""Key market events: the event table and statistics computed from our data.

Only the event *windows* (dates) are written by hand below. Every number in
``event_stats.csv`` and ``docs/events.md`` is computed from the processed data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vrp.logging_setup import get_logger
from vrp.validate import df_to_md

log = get_logger(__name__)

_GENERIC = ("Window chosen by the team from widely reported timelines; "
            "statistics computed from our data.")

# event_id, name, start, end, category, description, source_note
EVENTS: list[tuple[str, str, str, str, str, str, str]] = [
    ("gulf_war_1990", "Gulf War / 1990 recession", "1990-08-01", "1990-10-31", "geopolitical",
     "Iraq invades Kuwait (2 Aug 1990); oil spike during the 1990–91 NBER recession.", _GENERIC),
    ("asian_crisis_1997", "Asian financial crisis", "1997-10-01", "1997-10-31", "financial crisis",
     "Contagion from Asian currency crises; 27 Oct 1997 US sell-off and trading halts.", _GENERIC),
    ("ltcm_1998", "Russia default / LTCM", "1998-08-01", "1998-10-31", "financial crisis",
     "Russia defaults (17 Aug 1998); hedge fund LTCM rescued (23 Sep 1998).", _GENERIC),
    ("dotcom_bust", "Dot-com bust", "2000-03-24", "2002-10-09", "bear market",
     "S&P 500 falls from its Mar 2000 peak to its Oct 2002 low.", _GENERIC),
    ("sep11_2001", "9/11 attacks", "2001-09-11", "2001-09-28", "geopolitical",
     "Terrorist attacks; NYSE closed 11–14 Sep 2001 and reopened 17 Sep.",
     "NYSE closure dates per exchange records; " + _GENERIC),
    ("worldcom_2002", "WorldCom / accounting scandals", "2002-06-25", "2002-07-31",
     "financial crisis",
     "WorldCom reveals accounting fraud (25 Jun 2002), files for bankruptcy (21 Jul 2002).",
     _GENERIC),
    ("gfc_2008", "Global Financial Crisis", "2008-09-15", "2009-03-09", "financial crisis",
     "Lehman bankruptcy (15 Sep 2008); peak stress Oct–Nov 2008; S&P low on 9 Mar 2009.",
     "Sources conflict on the 2008 record close (80.86 on 20 Nov vs 80.74 on 21 Nov); "
     "we report FRED. " + _GENERIC),
    ("flash_crash_2010", "Flash Crash", "2010-05-06", "2010-05-28", "market structure",
     "Intraday collapse and rebound on 6 May 2010; euro-area worries in the following weeks.",
     _GENERIC),
    ("us_downgrade_2011", "US downgrade / euro crisis", "2011-08-01", "2011-08-31",
     "macro / policy",
     "S&P downgrades US debt (5 Aug 2011); euro sovereign debt stress.", _GENERIC),
    ("china_deval_2015", "China devaluation", "2015-08-11", "2015-08-31", "macro / policy",
     "PBoC devalues the yuan (11 Aug 2015); 24 Aug 2015 opening crash.", _GENERIC),
    ("volmageddon_2018", "Volmageddon (XIV collapse)", "2018-02-01", "2018-02-09",
     "market structure",
     "Short-vol products blow up on 5 Feb 2018; XIV loses ~96% and is terminated.", _GENERIC),
    ("q4_2018", "Q4 2018 sell-off", "2018-10-01", "2018-12-31", "macro / policy",
     "Fed tightening and trade-war fears; S&P low on 24 Dec 2018.", _GENERIC),
    ("covid_2020", "COVID crash", "2020-02-19", "2020-03-31", "pandemic",
     "Pandemic sell-off from the 19 Feb 2020 peak; record VIX close on 16 Mar 2020.", _GENERIC),
    ("bear_2022", "2022 bear market", "2022-01-03", "2022-10-12", "bear market",
     "Inflation and rapid Fed hikes; S&P low on 12 Oct 2022.", _GENERIC),
    ("yen_carry_2024", "Yen carry-trade unwind", "2024-08-01", "2024-08-09", "market structure",
     "BoJ hike and yen carry unwind; VIX spikes on 5 Aug 2024.",
     "The famous ~65 VIX print on 5 Aug 2024 was intraday; FRED has closes only. "
     + _GENERIC),
    ("tariff_2025", "2025 tariff shock", "2025-04-02", "2025-04-30", "macro / policy",
     "US 'Liberation Day' tariff announcement (2 Apr 2025) and sell-off.", _GENERIC),
]


def event_table() -> pd.DataFrame:
    """Return the event definitions as a DataFrame."""
    cols = ["event_id", "name", "start_date", "end_date", "category", "description",
            "source_note"]
    df = pd.DataFrame(EVENTS, columns=cols)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def max_drawdown(close: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Largest peak-to-trough fall (as a negative fraction) with peak and trough dates."""
    if close.empty:
        return float("nan"), None, None
    dd = close / close.cummax() - 1.0
    trough = dd.idxmin()
    peak = close.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def event_stats(events: pd.DataFrame, panel: pd.DataFrame,
                monthly: pd.DataFrame) -> pd.DataFrame:
    """Compute VIX / S&P / payoff statistics for each event window.

    Drawdown uses closes from the last trading day *before* the window through
    the window end, so a crash on the first day of the window is counted.
    Monthly rows overlap an event when their forward window [t+1, t+21]
    intersects [start_date, end_date].
    """
    rows = []
    for ev in events.itertuples(index=False):
        s, e = ev.start_date, ev.end_date
        win = panel.loc[s:e]
        prior = panel.loc[:s - pd.Timedelta(days=1)].tail(1)
        dd_close = pd.concat([prior["spx_close"], win["spx_close"]])
        dd, peak, trough = max_drawdown(dd_close)
        ov = monthly[(monthly["fwd_window_start"] <= e) & (monthly["fwd_window_end"] >= s)]
        rows.append({
            "event_id": ev.event_id,
            "name": ev.name,
            "start_date": s.date(),
            "end_date": e.date(),
            "trading_days": len(win),
            "max_vix_close": float(win["vix_close"].max()) if len(win) else np.nan,
            "max_vix_date": win["vix_close"].idxmax().date() if len(win) else None,
            "worst_spx_log_ret": float(win["ret_log"].min()) if len(win) else np.nan,
            "worst_spx_ret_date": win["ret_log"].idxmin().date() if len(win) else None,
            "spx_drawdown": dd,
            "drawdown_peak_date": peak.date() if peak is not None else None,
            "drawdown_trough_date": trough.date() if trough is not None else None,
            "n_overlapping_months": len(ov),
            "overlapping_months": "; ".join(
                f"{m}:{p:+.3f}" for m, p in zip(ov["month"], ov["payoff_scaled"], strict=True)),
            "min_payoff_scaled_overlap": float(ov["payoff_scaled"].min()) if len(ov) else np.nan,
            "mean_payoff_scaled_overlap": float(ov["payoff_scaled"].mean()) if len(ov) else np.nan,
        })
    out = pd.DataFrame(rows)
    log.info("events: computed stats for %d events", len(out))
    return out


def write_events_md(events: pd.DataFrame, stats: pd.DataFrame, panel: pd.DataFrame,
                    path: Path) -> Path:
    """Write ``docs/events.md`` (auto-generated; numbers come from the data)."""
    tbl = stats[["name", "start_date", "end_date", "max_vix_close", "max_vix_date",
                 "worst_spx_log_ret", "worst_spx_ret_date", "spx_drawdown"]].copy()
    tbl["max_vix_close"] = tbl["max_vix_close"].map(lambda v: f"{v:.2f}")
    tbl["worst_spx_log_ret"] = tbl["worst_spx_log_ret"].map(lambda v: f"{v:+.2%}")
    tbl["spx_drawdown"] = tbl["spx_drawdown"].map(lambda v: f"{v:.1%}")
    tbl.columns = ["Event", "Start", "End", "Max VIX close", "Date", "Worst S&P day (log)",
                   "Date ", "S&P drawdown"]

    def vix_on(d: str) -> str:
        ts = pd.Timestamp(d)
        return f"{panel.loc[ts, 'vix_close']:.2f}" if ts in panel.index else "not in panel"

    vmax_d = panel["vix_close"].idxmax()
    lines = [
        "# Key market events",
        "",
        "> **Auto-generated** by `python -m vrp.pipeline` (`src/vrp/events.py`). Do not edit by "
        "hand: change the event list in `events.py` and re-run `make data`.",
        "",
        "Only the event windows below were chosen by hand. Every statistic is computed from our "
        "processed data (VIX closes from FRED, S&P 500 closes from Yahoo Finance).",
        "",
        "How to read the columns:",
        "",
        "- **Max VIX close**: highest daily *closing* VIX inside the window.",
        "- **Worst S&P day**: most negative daily log return inside the window.",
        "- **S&P drawdown**: largest peak-to-trough fall of the S&P close, starting from the "
        "last close *before* the window (so a first-day crash counts).",
        "",
        "## Event table",
        "",
        df_to_md(events.assign(start_date=events["start_date"].dt.date,
                               end_date=events["end_date"].dt.date)
                 [["event_id", "name", "start_date", "end_date", "category", "description"]]),
        "",
        "## What the data shows",
        "",
        df_to_md(tbl),
        "",
        "## Monthly-sample rows that overlap each event",
        "",
        "A monthly row overlaps an event if its forward window (t+1 … t+21) intersects the "
        "event window. `payoff_scaled` = 1 − RV/IV (positive = variance seller gained).",
        "",
        df_to_md(stats[["name", "n_overlapping_months", "overlapping_months",
                        "min_payoff_scaled_overlap"]].rename(columns={
                            "name": "Event", "n_overlapping_months": "Rows",
                            "overlapping_months": "month:payoff_scaled",
                            "min_payoff_scaled_overlap": "Worst payoff_scaled"}), floatfmt=".3f"),
        "",
        "## Verification notes",
        "",
        f"- **2008 record close.** Published sources conflict (80.86 on 20 Nov 2008 vs 80.74 on "
        f"21 Nov 2008). FRED shows **{vix_on('2008-11-20')}** on 2008-11-20 and "
        f"**{vix_on('2008-11-21')}** on 2008-11-21. We use FRED.",
        f"- **Highest close in the sample.** {panel['vix_close'].max():.2f} on "
        f"{vmax_d.date()} (expected 82.69 on 2020-03-16; checked in `validate.py`).",
        f"- **5 Aug 2024.** The widely quoted VIX of about 65 was an *intraday* print before the "
        f"US open. FRED records closes only: **{vix_on('2024-08-05')}** on 2024-08-05.",
        "- **9/11.** US equity markets were closed 11–14 Sep 2001, so the window's first "
        "trading day is 17 Sep 2001 and the drawdown starts from the 10 Sep close.",
        "- **Pre-2003 VIX values** (1990, 1997, 1998, 2001, 2002 events) are back-calculated by "
        "Cboe with today's method; they are not what traders saw in real time.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", path)
    return path
