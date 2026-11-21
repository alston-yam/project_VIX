"""Supporting figures for v1 (no statistics or conclusions on them yet)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

from vrp.config import Config  # noqa: E402
from vrp.logging_setup import get_logger  # noqa: E402

log = get_logger(__name__)

# ---------------------------------------------------------------------- style
COLORS = {
    "implied": "#2a78d6",  # blue: VIX / implied / seller gain
    "realised": "#eb6834",  # orange: realised
    "loss": "#e34948",  # red: seller loss
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e4e3df",
    "event": "#eda100",
    "recession": "#b9b8b2",
}
STYLE = {
    "figure.figsize": (11, 5.6),
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": COLORS["ink2"],
    "axes.labelcolor": COLORS["ink"],
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": COLORS["ink2"],
    "ytick.color": COLORS["ink2"],
    "legend.frameon": False,
    "font.size": 10,
    "lines.linewidth": 1.6,
    "svg.fonttype": "none",
}


def apply_style() -> None:
    """Apply the project matplotlib style."""
    plt.rcParams.update(STYLE)


def footnote(fig: plt.Figure, text: str) -> None:
    """Add the data-source footnote at the bottom-left of the figure."""
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8, color=COLORS["ink2"])


def save(fig: plt.Figure, name: str, cfg: Config) -> list[Path]:
    """Save a figure as PNG (configured dpi) and SVG under ``outputs/figures``."""
    out_dir = cfg.path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in cfg["plots"]["formats"]:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(
            p,
            dpi=cfg["plots"]["dpi"],
            bbox_inches="tight",
            metadata={"Date": None} if ext == "svg" else None,
        )
        paths.append(p)
    plt.close(fig)
    log.info("plots: wrote %s", ", ".join(p.name for p in paths))
    return paths


def shade_events(
    ax: plt.Axes, events: pd.DataFrame, labels: dict[str, str], label_y: float = 0.985
) -> None:
    """Shade selected event windows and write their short labels near the top.

    ``labels`` maps event_id -> short label (from ``config.yaml``). Call after
    setting y-limits with headroom so labels sit above the data.
    """
    trans = ax.get_xaxis_transform()
    for ev in events[events["event_id"].isin(list(labels))].itertuples(index=False):
        s, e = ev.start_date, ev.end_date
        if (e - s).days < 20:  # make very short windows visible
            mid = s + (e - s) / 2
            s, e = mid - pd.Timedelta(days=12), mid + pd.Timedelta(days=12)
        ax.axvspan(s, e, color=COLORS["event"], alpha=0.22, lw=0, zorder=0)
        ax.text(
            s,
            label_y,
            labels[ev.event_id] + " ",
            transform=trans,
            rotation=90,
            ha="right",
            va="top",
            fontsize=7.5,
            color=COLORS["ink2"],
        )


def headroom(ax: plt.Axes, top: float, factor: float, log_scale: bool = False) -> None:
    """Extend the y-axis above ``top`` so event labels do not overlap the data."""
    lo = ax.get_ylim()[0]
    if log_scale:
        ax.set_ylim(lo, top * factor**2.5)
    else:
        ax.set_ylim(lo, top * factor)


def shade_recessions(ax: plt.Axes, usrec: pd.Series | None) -> bool:
    """Shade NBER recession months in light grey. Returns True if anything was drawn."""
    if usrec is None or usrec.empty or usrec.sum() == 0:
        return False
    r = usrec.astype(int)
    starts = r.index[(r == 1) & (r.shift(1, fill_value=0) == 0)]
    ends = r.index[(r == 1) & (r.shift(-1, fill_value=0) == 0)]
    for s, e in zip(starts, ends, strict=True):
        ax.axvspan(
            s, e + pd.offsets.MonthEnd(0), color=COLORS["recession"], alpha=0.25, lw=0, zorder=0
        )
    return True


def _date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# ---------------------------------------------------------------------- figures
def fig_spx_price(
    panel: pd.DataFrame, events: pd.DataFrame, usrec: pd.Series | None, cfg: Config
) -> list[Path]:
    """01: S&P 500 close on a log scale with events and recessions shaded."""
    fig, ax = plt.subplots()
    rec = shade_recessions(ax, usrec)
    ax.plot(panel.index, panel["spx_close"], color=COLORS["implied"], lw=1.2)
    ax.set_yscale("log")
    headroom(ax, float(panel["spx_close"].max()), cfg["plots"]["headroom"], log_scale=True)
    shade_events(ax, events, cfg["plots"]["shade_events"])
    ax.yaxis.set_major_locator(FixedLocator([250, 500, 1000, 2000, 4000, 8000]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("S&P 500 close (index points, log scale)")
    ax.set_xlabel("Date")
    ax.set_title("S&P 500 index level, 1990–2026 (log scale)")
    _date_axis(ax)
    note = "Shaded: key events (orange)" + ("; NBER recessions (grey, FRED USREC)" if rec else "")
    footnote(fig, cfg["plots"]["source_note"] + ". " + note + ".")
    return save(fig, "01_spx_price_log", cfg)


def fig_vix_vs_rv(monthly: pd.DataFrame, events: pd.DataFrame, cfg: Config) -> list[Path]:
    """02: VIX vs forward 21-day realised vol (vol %, monthly sample)."""
    fig, ax = plt.subplots()
    ax.plot(
        monthly.index,
        monthly["vix_close"],
        color=COLORS["implied"],
        label="VIX close on first trading day of month (implied vol)",
    )
    ax.plot(
        monthly.index,
        monthly["rv_fwd21_ann_vol_pct"],
        color=COLORS["realised"],
        label="Realised vol over the next 21 trading days",
    )
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_xlabel("Month (first trading day)")
    ax.set_title("Implied (VIX) vs subsequently realised S&P 500 volatility, monthly", pad=28)
    ax.set_ylim(bottom=0)
    headroom(
        ax,
        float(monthly[["vix_close", "rv_fwd21_ann_vol_pct"]].max().max()),
        cfg["plots"]["headroom"],
    )
    shade_events(ax, events, cfg["plots"]["shade_events"])
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=9)
    _date_axis(ax)
    footnote(
        fig,
        cfg["plots"]["source_note"] + ". Realised vol = 100·√[(252/21)·Σ r²], "
        "r = daily log return, t+1…t+21.",
    )
    return save(fig, "02_vix_vs_realised_vol", cfg)


def fig_scatter(monthly: pd.DataFrame, cfg: Config) -> list[Path]:
    """03: VIX (x) vs realised vol (y) with the 45° line; seller-loss months in red."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    x, y = monthly["vix_close"], monthly["rv_fwd21_ann_vol_pct"]
    loss = y > x
    ax.scatter(
        x[~loss],
        y[~loss],
        s=22,
        color=COLORS["implied"],
        alpha=0.75,
        edgecolor="white",
        lw=0.5,
        label=f"Seller gain: realised < VIX ({int((~loss).sum())} months)",
    )
    ax.scatter(
        x[loss],
        y[loss],
        s=22,
        color=COLORS["loss"],
        alpha=0.85,
        edgecolor="white",
        lw=0.5,
        label=f"Seller loss: realised > VIX ({int(loss.sum())} months)",
    )
    lim = float(np.ceil(max(x.max(), y.max()) / 10) * 10)
    ax.plot([0, lim], [0, lim], color=COLORS["ink2"], lw=1, ls="--", label="45° line (RV = VIX)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    worst = monthly.nsmallest(cfg["plots"]["n_label_worst"], "payoff_scaled")
    for _, row in worst.iterrows():
        ax.annotate(
            f"{row['month']}\n(payoff {row['payoff_scaled']:+.1f})",
            (row["vix_close"], row["rv_fwd21_ann_vol_pct"]),
            xytext=(8, -4),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["ink"],
        )
    ax.set_xlabel("VIX close on first trading day of month (annualised vol, %)")
    ax.set_ylabel("Realised vol over next 21 trading days (annualised, %)")
    ax.set_title("VIX vs realised volatility, monthly 1990–2025")
    ax.legend(loc="lower right")
    footnote(fig, cfg["plots"]["source_note"] + ". Labels: 5 worst months by payoff_scaled.")
    return save(fig, "03_scatter_iv_vs_rv", cfg)


def fig_payoff_hist(monthly: pd.DataFrame, cfg: Config) -> list[Path]:
    """04: Histogram of payoff_scaled with mean and median lines."""
    fig, ax = plt.subplots()
    p = monthly["payoff_scaled"]
    bins = np.arange(np.floor(p.min() * 4) / 4, 1.0 + 0.125, 0.125)
    _, edges, patches = ax.hist(p, bins=bins, color=COLORS["implied"], edgecolor="white", lw=1)
    for patch, left in zip(patches, edges[:-1], strict=True):
        if left < 0:
            patch.set_facecolor(COLORS["loss"])
    ax.axvline(p.mean(), color=COLORS["ink"], lw=1.4, ls="-", label=f"Mean = {p.mean():+.3f}")
    ax.axvline(p.median(), color=COLORS["ink"], lw=1.4, ls=":", label=f"Median = {p.median():+.3f}")
    ax.axvline(1.0, color=COLORS["ink2"], lw=0.8, ls="--", label="Upper bound = +1 (RV = 0)")
    ax.set_xlabel("Scaled seller payoff, payoff_scaled = 1 − RV/IV (fraction of implied variance)")
    ax.set_ylabel("Number of months")
    ax.set_title("Distribution of the monthly scaled variance-seller payoff, 1990–2025")
    ax.legend(loc="upper left")
    footnote(fig, cfg["plots"]["source_note"] + f". n = {len(p)} months; red bars = losses.")
    return save(fig, "04_payoff_scaled_hist", cfg)


def fig_payoff_timeline(monthly: pd.DataFrame, events: pd.DataFrame, cfg: Config) -> list[Path]:
    """05: Monthly payoff_scaled as bars over time, losses in red, events annotated."""
    fig, ax = plt.subplots()
    p = monthly["payoff_scaled"]
    colors = np.where(p < 0, COLORS["loss"], COLORS["implied"])
    ax.bar(p.index, p.values, width=25, color=colors, lw=0)
    ax.axhline(0, color=COLORS["ink2"], lw=0.8)
    ax.set_ylim(top=1.0 + (1.0 - p.min()) * 0.25)
    shade_events(ax, events, cfg["plots"]["shade_events"])
    ax.set_ylabel("payoff_scaled = 1 − RV/IV")
    ax.set_xlabel("Month (first trading day)")
    ax.set_title("Monthly scaled variance-seller payoff over time (red = seller loss)")
    _date_axis(ax)
    footnote(
        fig,
        cfg["plots"]["source_note"] + ". Each bar: VIX at month start vs realised "
        "variance over the next 21 trading days.",
    )
    return save(fig, "05_payoff_timeline", cfg)


def fig_coverage(series: dict[str, pd.DatetimeIndex], cfg: Config) -> list[Path]:
    """06: Daily observation counts per calendar year per source."""
    fig, ax = plt.subplots()
    palette = [COLORS["implied"], COLORS["realised"], COLORS["aqua"], COLORS["ink2"]]
    markers = ["o", "s", "^", "D"]
    for (name, idx), c, mk in zip(series.items(), palette, markers, strict=False):
        counts = pd.Series(1, index=idx).groupby(idx.year).sum()
        ax.plot(counts.index, counts.values, marker=mk, ms=4, color=c, label=name)
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Daily observations per year")
    ax.set_title("Data coverage: daily observations per year by source")
    ax.legend(loc="lower left")
    ax.set_ylim(bottom=0)
    footnote(
        fig,
        cfg["plots"]["source_note"] + "; VXO via FRED (VXOCLS).\n1989 and 2026 are "
        "partial (window 1 Nov 1989 – 31 Mar 2026); VXO discontinued Sep 2021; since 2022 FRED "
        "also lists VIX values on some US market holidays.",
    )
    return save(fig, "06_data_coverage", cfg)


def make_all_figures(
    panel: pd.DataFrame,
    monthly: pd.DataFrame,
    events: pd.DataFrame,
    usrec: pd.Series | None,
    coverage: dict[str, pd.DatetimeIndex],
    cfg: Config,
) -> list[Path]:
    """Draw every figure and return the written paths."""
    apply_style()
    paths: list[Path] = []
    paths += fig_spx_price(panel, events, usrec, cfg)
    paths += fig_vix_vs_rv(monthly, events, cfg)
    paths += fig_scatter(monthly, cfg)
    paths += fig_payoff_hist(monthly, cfg)
    paths += fig_payoff_timeline(monthly, events, cfg)
    paths += fig_coverage(coverage, cfg)
    return paths
