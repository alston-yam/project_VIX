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
    "svg.hashsalt": "vrp",  # deterministic SVG element ids, so rebuilds do not churn git diffs
}


def apply_style() -> None:
    """Apply the project matplotlib style."""
    plt.rcParams.update(STYLE)


def footnote(fig: plt.Figure, text: str) -> None:
    """Add the data-source footnote just below the bottom-left of the figure."""
    fig.text(0.01, -0.01, text, ha="left", va="top", fontsize=8, color=COLORS["ink2"])


def save(fig: plt.Figure, name: str, cfg: Config) -> list[Path]:
    """Save a figure as PNG (configured dpi) and SVG under ``outputs/figures``."""
    out_dir = cfg.path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    name += cfg["plots"].get("name_suffix", "")  # headline sub-sample figures get a suffix
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


def span_label(cfg: Config) -> str:
    """Years of the sample being plotted, e.g. '1990–2025' or '2003–2025' (headline)."""
    d = cfg["dates"]
    return cfg["plots"].get("sample_label") or (
        f"{d['sample_start_month'][:4]}–{d['sample_end_month'][:4]}"
    )


def mark_live_vix(ax: plt.Axes, cfg: Config) -> None:
    """Mark the start of the live VIX (earlier values are back-calculated), if on the axis."""
    d = pd.Timestamp(cfg["dates"]["vix_new_methodology_date"])
    lo, hi = (mdates.num2date(v).replace(tzinfo=None) for v in ax.get_xlim())
    first = mdates.num2date(ax.dataLim.x0).replace(tzinfo=None)
    if not first < d < hi:  # no plotted data before the live VIX (e.g. post-2003 figures)
        return
    ax.axvspan(lo, d, color=COLORS["recession"], alpha=0.12, lw=0, zorder=0)
    ax.axvline(d, color=COLORS["ink2"], lw=1, ls="--", zorder=1)
    ax.annotate(
        "← VIX back-calculated | live VIX →",
        (d, 0.03),
        xycoords=("data", "axes fraction"),
        ha="center",
        fontsize=8,
        color=COLORS["ink2"],
        bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5},
    )


def shade_events(
    ax: plt.Axes, events: pd.DataFrame, labels: dict[str, str], label_y: float = 0.985
) -> None:
    """Shade selected event windows and write their short labels near the top.

    ``labels`` maps event_id -> short label (from ``config.yaml``). Call after
    setting y-limits with headroom so labels sit above the data.
    """
    trans = ax.get_xaxis_transform()
    xlim = ax.get_xlim()
    lo, hi = (mdates.num2date(v).replace(tzinfo=None) for v in xlim)
    for ev in events[events["event_id"].isin(list(labels))].itertuples(index=False):
        s, e = ev.start_date, ev.end_date
        if e < lo or s > hi:  # outside the plotted sample (e.g. the post-2003 figures)
            continue
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
    ax.set_xlim(xlim)


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
    mark_live_vix(ax, cfg)
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


# ====================================================================== v2: test figures
SIGN_FOOT = "Sign: implied − realised (positive = variance seller gains)."


def _sig_color(lo: float, hi: float, null: float) -> str:
    """Blue if the CI is entirely above the null, red if entirely below, grey otherwise."""
    if lo > null:
        return COLORS["implied"]
    if hi < null:
        return COLORS["loss"]
    return COLORS["ink2"]


def _forest(ax: plt.Axes, labels: list[str], est, lo, hi, null: float = 0.0) -> None:
    """Horizontal point-and-interval plot, top row first."""
    y = np.arange(len(labels))[::-1]
    for yi, e, a, b in zip(y, est, lo, hi, strict=True):
        c = _sig_color(a, b, null)
        ax.plot([a, b], [yi, yi], color=c, lw=2.2, solid_capstyle="butt")
        ax.plot(e, yi, "o", color=c, ms=6, mec="white", mew=0.8)
    ax.axvline(null, color=COLORS["ink2"], lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)


def fig_test1_mean(tables: pd.DataFrame, cfg: Config) -> list[Path]:
    """07: Test 1, mean (three SE methods), median and trimmed mean, with 95% CIs."""
    t = tables[tables["test"] == "1_mean"].dropna(subset=["ci_low"])
    n_months = int(t["n"].max())
    measures = [
        ("payoff_scaled", "Scaled payoff (fraction of IV)"),
        ("payoff_var", "Raw payoff (annualised variance)"),
        ("vol_spread_pts", "Vol spread (VIX − realised vol, pts)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (m, title) in zip(axes, measures, strict=True):
        d = t[t["measure"] == m]
        labels = [
            f"{r.statistic.replace('_', ' ')} · "
            + ("i.i.d. t" if r.ci_method.startswith("t ") else r.ci_method.split(" (")[0])
            for r in d.itertuples()
        ]
        _forest(ax, labels, d["estimate"], d["ci_low"], d["ci_high"])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Estimate with 95% CI")
        lo = min(0.0, float(d["ci_low"].min()))
        ax.set_xlim(lo - 0.08 * (d["ci_high"].max() - lo), None)
    fig.suptitle(
        "Test 1: the average seller payoff is positive under every standard-error method",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout()
    s = cfg["stats"]
    footnote(
        fig,
        cfg["plots"]["source_note"] + f". {SIGN_FOOT} Monthly {span_label(cfg)} (n = {n_months}). "
        f"Newey–West {s['nw_lags_monthly']} lags; circular block bootstrap "
        f"({s['bootstrap']['block_length']}-month blocks, {s['bootstrap']['n_resamples']:,} "
        "resamples). Blue = 95% CI excludes 0.",
    )
    return save(fig, "07_test1_mean_ci", cfg)


def fig_test2_distribution(monthly: pd.DataFrame, tables: pd.DataFrame, cfg: Config) -> list[Path]:
    """08: Test 2, histogram vs a normal with the same mean/sd (log counts) and a QQ plot."""
    from scipy import stats as sps

    x = monthly["payoff_scaled"].to_numpy(dtype=float)
    t = tables[(tables["test"] == "2_shape") & (tables["measure"] == "payoff_scaled")]

    def g(stat: str, method: str = "") -> pd.Series:
        d = t[t["statistic"] == stat]
        if method:
            d = d[d["ci_method"].str.contains(method, regex=False)]
        return d.iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
    bins = np.arange(np.floor(x.min() * 4) / 4, 1.0 + 0.125, 0.125)
    counts, edges, patches = ax1.hist(x, bins=bins, color=COLORS["implied"], edgecolor="white")
    for patch, left in zip(patches, edges[:-1], strict=True):
        if left < 0:
            patch.set_facecolor(COLORS["loss"])
    mu, sd = x.mean(), x.std(ddof=1)
    grid = np.linspace(edges[0], 1.0, 400)
    ax1.plot(
        grid,
        sps.norm.pdf(grid, mu, sd) * len(x) * (edges[1] - edges[0]),
        color=COLORS["ink"],
        lw=1.3,
        ls="--",
        label="Normal with the same mean and sd",
    )
    ax1.set_yscale("log")
    ax1.set_ylim(0.5, counts.max() * 3)
    ax1.set_xlabel("payoff_scaled = 1 − RV/IV (fraction of implied variance)")
    ax1.set_ylabel("Number of months (log scale)")
    ax1.set_title("Distribution vs a normal curve", fontsize=11)
    ax1.legend(loc="upper left")
    wr, sk, ku = g("win_rate", "Wilson"), g("skewness"), g("excess_kurtosis")
    tail = g("months_below_mean_minus_3sd")
    txt = (
        f"Win rate {100 * wr.estimate:.1f}% [{100 * wr.ci_low:.1f}, {100 * wr.ci_high:.1f}]\n"
        f"Skewness {sk.estimate:.2f} [{sk.ci_low:.2f}, {sk.ci_high:.2f}]\n"
        f"Excess kurtosis {ku.estimate:.1f} [{ku.ci_low:.1f}, {ku.ci_high:.1f}]\n"
        f"Months < mean − 3sd: {int(tail.estimate)} (normal: {tail.null_value:.1f})\n"
        "95% CIs: Wilson (win rate), block bootstrap (moments)"
    )
    ax1.text(
        0.02,
        0.80,
        txt,
        transform=ax1.transAxes,
        va="top",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": COLORS["grid"]},
    )

    (osm, osr), (slope, intercept, _) = sps.probplot(x, dist="norm")
    ax2.scatter(osm, osr, s=12, color=np.where(osr < 0, COLORS["loss"], COLORS["implied"]))
    ax2.plot(
        osm,
        intercept + slope * osm,
        color=COLORS["ink2"],
        lw=1,
        ls="--",
        label="Normal reference line",
    )
    ax2.set_xlabel("Theoretical normal quantile")
    ax2.set_ylabel("Observed payoff_scaled quantile")
    ax2.set_title("Normal QQ plot: the left tail is far too long", fontsize=11)
    ax2.legend(loc="upper left")
    fig.suptitle(
        "Test 2: the payoff is lopsided (frequent small gains, rare huge losses)",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout()
    footnote(
        fig,
        cfg["plots"]["source_note"]
        + f". {SIGN_FOOT} Monthly {span_label(cfg)}, n = {len(monthly)}.",
    )
    return save(fig, "08_test2_distribution", cfg)


def fig_test3_selloff(
    monthly: pd.DataFrame, deciles: pd.DataFrame, tables: pd.DataFrame, cfg: Config
) -> list[Path]:
    """09: Test 3, mean payoff by S&P decile with CIs, and payoff vs S&P return scatter."""
    s = cfg["stats"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
    d = deciles
    colors = [_sig_color(lo, hi, 0.0) for lo, hi in zip(d["ci_low"], d["ci_high"], strict=True)]
    ax1.bar(d["selloff_decile"], d["mean"], color=colors, width=0.7)
    ax1.errorbar(
        d["selloff_decile"],
        d["mean"],
        yerr=[d["mean"] - d["ci_low"], d["ci_high"] - d["mean"]],
        fmt="none",
        ecolor=COLORS["ink"],
        capsize=3,
        lw=1,
    )
    ax1.axhline(0, color=COLORS["ink2"], lw=0.8)
    ax1.set_xticks(d["selloff_decile"])
    ax1.set_xticklabels(
        [
            f"{i}\n{100 * r:+.1f}%"
            for i, r in zip(d["selloff_decile"], d["spx_fwd21_ret_mean"], strict=True)
        ],
        fontsize=8.5,
    )
    ax1.set_xlabel("S&P 500 decile of the same 21-day window (1 = worst), mean log return")
    ax1.set_ylabel("Mean payoff_scaled (95% CI, i.i.d. t)")
    ax1.set_title("Mean scaled payoff by S&P decile", fontsize=11)
    t = tables[(tables["test"] == "3_crash") & (tables["measure"] == "payoff_var")]
    share = t[t["statistic"] == f"loss_share_in_decile_{s['worst_decile']}"].iloc[0]
    ax1.text(
        0.98,
        0.04,
        f"Worst decile carries {100 * share.estimate:.0f}% of all raw-variance losses\n"
        f"(95% CI {100 * share.ci_low:.0f}–{100 * share.ci_high:.0f}%; no clustering ≈ "
        f"{100 * share.null_value:.0f}%; permutation p ≤ {max(share.p_value, 1e-4):.1g})",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": COLORS["grid"]},
    )

    x = 100 * monthly["spx_fwd21_ret_log"]
    y = monthly["payoff_scaled"]
    ax2.scatter(
        x, y, s=14, alpha=0.8, color=np.where(y < 0, COLORS["loss"], COLORS["implied"]), lw=0
    )
    ax2.axvline(
        100 * cfg["features"]["selloff_threshold_log"],
        color=COLORS["yellow"],
        lw=1.2,
        ls="--",
        label="Sell-off threshold (−5% log)",
    )
    ax2.axhline(0, color=COLORS["ink2"], lw=0.8)
    c = tables[
        (tables["test"] == "3_crash")
        & (tables["measure"] == "payoff_scaled")
        & (tables["statistic"] == "corr_with_spx_fwd21_ret")
        & tables["ci_method"].str.contains("bootstrap")
    ].iloc[0]
    ax2.set_xlabel("S&P 500 log return over the same 21 trading days (%)")
    ax2.set_ylabel("payoff_scaled = 1 − RV/IV")
    ax2.set_title(
        f"Payoff vs S&P move: correlation {c.estimate:.2f} "
        f"(95% CI {c.ci_low:.2f}–{c.ci_high:.2f})",
        fontsize=11,
    )
    ax2.legend(loc="lower right")
    fig.suptitle(
        "Test 3: seller losses cluster in the months when the S&P 500 falls",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout()
    footnote(
        fig,
        cfg["plots"]["source_note"] + f". {SIGN_FOOT} The S&P return is over the same window "
        "(an outcome): this shows when losses happen, not a forecast at t.",
    )
    return save(fig, "09_test3_selloff", cfg)


def fig_test4_mz(monthly: pd.DataFrame, mz: dict, tables: pd.DataFrame, cfg: Config) -> list[Path]:
    """10: Test 4, Mincer–Zarnowitz fit in vol and log-variance space vs the 45° line."""
    t = tables[tables["test"] == "4_forecast"]

    def g(measure: str, stat: str) -> pd.Series:
        return t[(t["measure"] == measure) & (t["statistic"] == stat)].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x, y = monthly["vix_close"], monthly["rv_fwd21_ann_vol_pct"]
    ax1.scatter(x, y, s=12, alpha=0.6, color=COLORS["implied"], lw=0)
    lim = float(np.ceil(max(x.max(), y.max()) / 10) * 10)
    ax1.plot([0, lim], [0, lim], color=COLORS["ink2"], ls="--", lw=1, label="Unbiased: RV = VIX")
    ax1.plot(mz["grid"], mz["fit"], color=COLORS["realised"], lw=2, label="Fitted RV = a + b·VIX")
    ax1.fill_between(
        mz["grid"],
        mz["lo"],
        mz["hi"],
        color=COLORS["realised"],
        alpha=0.25,
        lw=0,
        label="95% CI of fit (Newey–West)",
    )
    a, b, w = g("vol_pct", "intercept_a"), g("vol_pct", "slope_b"), g("vol_pct", "joint_a0_b1")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xlabel("VIX at close of t (annualised vol, %)")
    ax1.set_ylabel("Realised vol t+1…t+21 (annualised, %)")
    ax1.set_title(
        f"Vol: a = {a.estimate:.2f} [{a.ci_low:.2f}, {a.ci_high:.2f}], "
        f"b = {b.estimate:.2f} [{b.ci_low:.2f}, {b.ci_high:.2f}]\n"
        f"Wald test of a = 0, b = 1: p = {w.p_value:.1e}",
        fontsize=10.5,
    )
    ax1.legend(loc="upper left")

    lx, ly = np.log(monthly["iv_ann_var"]), np.log(monthly["rv_fwd21_ann_var"])
    ax2.scatter(lx, ly, s=12, alpha=0.6, color=COLORS["implied"], lw=0)
    a, b, w = (
        g("log_variance", "intercept_a"),
        g("log_variance", "slope_b"),
        g("log_variance", "joint_a0_b1"),
    )
    lo_, hi_ = float(min(lx.min(), ly.min())), float(max(lx.max(), ly.max()))
    gx = np.linspace(lo_, hi_, 50)
    ax2.plot(gx, gx, color=COLORS["ink2"], ls="--", lw=1, label="Unbiased: ln RV = ln IV")
    ax2.plot(
        gx,
        a.estimate + b.estimate * gx,
        color=COLORS["realised"],
        lw=2,
        label="Fitted ln RV = a + b·ln IV",
    )
    ax2.set_xlabel("ln implied variance at t, ln[(VIX/100)²]")
    ax2.set_ylabel("ln realised variance t+1…t+21")
    ax2.set_title(
        f"Log variance: a = {a.estimate:.2f} [{a.ci_low:.2f}, {a.ci_high:.2f}], "
        f"b = {b.estimate:.2f} [{b.ci_low:.2f}, {b.ci_high:.2f}]\n"
        f"Wald test of a = 0, b = 1: p = {w.p_value:.1e}",
        fontsize=10.5,
    )
    ax2.legend(loc="upper left")
    fig.suptitle(
        "Test 4: the VIX is a biased forecast: realised volatility sits below the 45° line",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout()
    footnote(
        fig,
        cfg["plots"]["source_note"] + f". Mincer–Zarnowitz regressions, monthly {span_label(cfg)} "
        f"(n = {len(monthly)}); Newey–West {cfg['stats']['nw_lags_monthly']} lags; "
        "brackets = 95% CIs. "
        "The only regressor is the VIX at the close of t.",
    )
    return save(fig, "10_test4_mincer_zarnowitz", cfg)


def fig_test5_robustness(tables: pd.DataFrame, cfg: Config) -> list[Path]:
    """11: Test 5, mean payoff with Newey–West 95% CIs across robustness variants."""
    t = tables[(tables["test"] == "5_robustness") & (tables["statistic"] == "mean")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for ax, (m, title) in zip(
        axes,
        [
            ("payoff_scaled", "Scaled payoff (fraction of IV)"),
            ("payoff_var", "Raw payoff (annualised variance)"),
        ],
        strict=True,
    ):
        d = t[t["measure"] == m]
        labels = [f"{r.sample}  (n = {int(r.n):,})" for r in d.itertuples()]
        _forest(ax, labels, d["estimate"], d["ci_low"], d["ci_high"])
        for yi, r in zip(np.arange(len(d))[::-1], d.itertuples(), strict=True):
            ax.text(
                ax.get_xlim()[1],
                yi,
                f" p_Holm={r.p_holm_within_measure:.1g}",
                va="center",
                fontsize=7.8,
                color=COLORS["ink2"],
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Mean payoff with Newey–West 95% CI")
    fig.suptitle(
        "Test 5: robustness of the mean payoff across samples and definitions",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout()
    s = cfg["stats"]
    footnote(
        fig,
        cfg["plots"]["source_note"] + "; VXO via FRED (VXOCLS). "
        f"{SIGN_FOOT} Newey–West {s['nw_lags_monthly']} lags (monthly), "
        f"{s['nw_lags_daily']} lags (daily overlapping). Blue = CI excludes 0; grey = does not. "
        "p_Holm: Holm-adjusted across the variants.",
    )
    return save(fig, "11_test5_robustness", cfg)


def make_stats_figures(monthly: pd.DataFrame, res: object, cfg: Config) -> list[Path]:
    """Draw figures 07–11 from a ``vrp.stats.StatsResults``."""
    apply_style()
    t = res.tables  # type: ignore[attr-defined]
    paths: list[Path] = []
    paths += fig_test1_mean(t, cfg)
    paths += fig_test2_distribution(monthly, t, cfg)
    paths += fig_test3_selloff(monthly, res.deciles["payoff_scaled"], t, cfg)  # type: ignore
    paths += fig_test4_mz(monthly, res.mz_line, t, cfg)  # type: ignore[attr-defined]
    if (t["test"] == "5_robustness").any():  # not run on the headline sub-sample
        paths += fig_test5_robustness(t, cfg)
    return paths


# ====================================================================== research figures
def _foot_wrapped(fig: plt.Figure, text: str) -> None:
    """Footnote wrapped to the figure width, so long notes do not widen the saved image."""
    import textwrap

    width = int(fig.get_figwidth() * 15)
    footnote(fig, "\n".join(textwrap.wrap(text, width)))


def _get(tables: pd.DataFrame, test: str, measure: str, stat: str, sample: str = "") -> pd.Series:
    m = (tables["test"] == test) & (tables["measure"] == measure) & (tables["statistic"] == stat)
    if sample:
        m &= tables["sample"] == sample
    return tables[m].iloc[0]


def fig_rv_dynamics(res: object, events: pd.DataFrame, cfg: Config) -> list[Path]:
    """12: calendar-month realised vol with its fitted trend; autocorrelation of log RV."""
    rvm = res.rv_monthly  # type: ignore[attr-defined]
    t = res.tables  # type: ignore[attr-defined]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
    vol = rvm["rv_cal_ann_vol_pct"]
    ax.plot(vol.index, vol, color=COLORS["realised"], lw=1.1, label="Realised vol, calendar month")
    tr = _get(t, "R_dynamics", "log_rv_cal", "trend_slope_per_year")
    yrs = np.asarray((vol.index - vol.index[0]).days / 365.25)
    lv = np.log(rvm["rv_cal_ann_var"].to_numpy(float))
    icpt = float(lv.mean() - tr["estimate"] * yrs.mean())
    ax.plot(
        vol.index,
        100 * np.sqrt(np.exp(icpt + tr["estimate"] * yrs)),
        color=COLORS["ink"],
        ls="--",
        lw=1.2,
        label=f"Fitted log-linear trend (p = {tr['p_value']:.2f})",
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 40, 80]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    headroom(ax, float(vol.max()), cfg["plots"]["headroom"], log_scale=True)
    shade_events(ax, events, cfg["plots"]["shade_events"])
    ax.set_ylabel("Annualised realised volatility (%, log scale)")
    ax.set_xlabel("Calendar month")
    ax.set_title("Realised volatility spikes and fades; no long-run trend in the fit", fontsize=11)
    ax.legend(loc="lower left", fontsize=8.5)
    _date_axis(ax)

    n = len(lv)
    lags = np.arange(1, 25)
    acf = [float(pd.Series(lv).autocorr(k)) for k in lags]
    ax2.set_axisbelow(True)
    ax2.bar(lags, acf, color=COLORS["implied"], width=0.7)
    band = 1.96 / np.sqrt(n)
    ax2.axhspan(-band, band, color=COLORS["grid"], alpha=0.8, zorder=0)
    ar = _get(t, "R_dynamics", "log_rv_cal", "ar1_rho")
    hl = _get(t, "R_dynamics", "log_rv_cal", "half_life_months")
    ax2.text(
        0.97,
        0.95,
        f"AR(1) ρ = {ar['estimate']:.2f} [{ar['ci_low']:.2f}, {ar['ci_high']:.2f}]\n"
        f"half-life ≈ {hl['estimate']:.1f} months",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    ax2.set_xlabel("Lag (months)")
    ax2.set_ylabel("Autocorrelation of log realised variance")
    ax2.set_title("Vol clusters, but the memory is short", fontsize=11)
    ax2.set_ylim(-0.1, 1)
    fig.suptitle(f"Realised-volatility dynamics, {span_label(cfg)}", fontweight="bold", fontsize=13)
    fig.tight_layout()
    _foot_wrapped(
        fig,
        cfg["plots"]["source_note"] + ". Calendar-month realised variance = (252/n)·Σ r² over "
        "the month's trading days. Trend: OLS of log RV on time, Newey–West "
        f"{cfg['research']['trend_nw_lags']} lags. Grey band: ±1.96/√n.",
    )
    return save(fig, "12_rv_dynamics", cfg)


def fig_rv_by_year(res: object, cfg: Config, n_highlight: int = 5) -> list[Path]:
    """13: realised vol and vol-of-vol by calendar year; most irregular years highlighted."""
    yr = res.rv_year  # type: ignore[attr-defined]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True)
    top = yr["rank_irregular"] <= n_highlight
    ax.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax.bar(yr.index, yr["rv_ann_vol_pct"], color=COLORS["realised"], width=0.75)
    ax.set_ylabel("Realised vol (%)")
    ax.set_title("Annual realised volatility of the S&P 500", fontsize=11)
    ax2.bar(
        yr.index,
        yr["vol_of_vol_log"],
        color=np.where(top, COLORS["loss"], COLORS["ink2"]),
        width=0.75,
    )
    for y, v in yr.loc[top, "vol_of_vol_log"].items():
        ax2.text(y, v, f"{y}", ha="center", va="bottom", fontsize=8, color=COLORS["loss"])
    ax2.set_ylabel("Vol-of-vol (std of monthly log RV)")
    ax2.set_xlabel("Calendar year")
    ax2.set_title(
        f"How irregular each year was (red: the {n_highlight} most irregular)", fontsize=11
    )
    ax2.set_ylim(0, float(yr["vol_of_vol_log"].max()) * 1.15)
    fig.suptitle("Which years were most irregular?", fontweight="bold", fontsize=13)
    fig.tight_layout()
    _foot_wrapped(
        fig,
        cfg["plots"]["source_note"] + ". Annual vol from all daily log returns in the year; "
        "vol-of-vol = standard deviation of the 12 calendar-month log realised variances.",
    )
    return save(fig, "13_rv_by_year", cfg)


def fig_conditional_premium(res: object, cfg: Config) -> list[Path]:
    """14: mean scaled payoff (Newey–West 95% CI) by VIX level and by VIX − past RV at t."""
    t = res.tables  # type: ignore[attr-defined]
    c = t[(t["test"] == "R_conditional") & (t["measure"] == "payoff_scaled")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    panels = [
        ("VIX", "VIX level at the close of t"),
        ("gap", "VIX − past 21-day realised vol at t (vol pts)"),
    ]
    for ax, (key, xlabel) in zip(axes, panels, strict=True):
        means = c[(c["statistic"] == "mean") & c["sample"].str.contains(key, regex=False)]
        means = means[~means["sample"].str.contains("minus")]
        wins = c[c["statistic"] == "win_rate"].set_index("sample")["estimate"]
        x = np.arange(len(means))
        cols = [
            _sig_color(lo, hi, 0.0)
            for lo, hi in zip(means["ci_low"], means["ci_high"], strict=True)
        ]
        ax.set_axisbelow(True)
        ax.bar(x, means["estimate"], color=cols, width=0.6, alpha=0.85)
        ax.errorbar(
            x,
            means["estimate"],
            yerr=[means["estimate"] - means["ci_low"], means["ci_high"] - means["estimate"]],
            fmt="none",
            ecolor=COLORS["ink"],
            capsize=5,
            lw=1.2,
        )
        for xi, r in zip(x, means.itertuples(), strict=True):
            ax.text(
                xi,
                -0.06,
                f"n = {int(r.n)}\nwins {100 * wins[r.sample]:.0f}%",
                ha="center",
                va="top",
                fontsize=8.5,
                color=COLORS["ink2"],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(means["sample"])
        ax.axhline(0, color=COLORS["ink2"], lw=0.9)
        d = c[(c["statistic"] == "diff_mean_high_minus_low") & c["sample"].str.contains(key)]
        d = d.iloc[0]
        ax.set_title(
            f"High − low: {100 * d['estimate']:+.1f}% [{100 * d['ci_low']:+.1f}, "
            f"{100 * d['ci_high']:+.1f}], bootstrap p = {d['p_value']:.2f}",
            fontsize=10,
        )
        ax.set_xlabel(xlabel)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    axes[0].set_ylabel("Mean scaled seller payoff (% of implied variance)")
    axes[0].set_ylim(-0.25, None)
    fig.suptitle(
        "When is the VIX most pessimistic? Seller payoff by conditions known at t",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout()
    _foot_wrapped(
        fig,
        cfg["plots"]["source_note"] + f". {SIGN_FOOT} Monthly {span_label(cfg)}. Bars: mean with "
        f"Newey–West 95% CI (blue = excludes 0). Fixed thresholds from config. In-sample "
        "description, not a trading rule.",
    )
    return save(fig, "14_conditional_premium", cfg)


def fig_rolling_premium(
    res: object, monthly: pd.DataFrame, events: pd.DataFrame, claims: pd.DataFrame, cfg: Config
) -> list[Path]:
    """15: rolling mean of the scaled payoff with a Newey–West band and the linear trend."""
    roll = res.rolling  # type: ignore[attr-defined]
    t = res.tables  # type: ignore[attr-defined]
    w = int(roll["window_months"].iloc[0])
    live = pd.Timestamp(cfg["dates"]["vix_new_methodology_date"])
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.fill_between(
        roll.index, roll["ci_low"], roll["ci_high"], color=COLORS["implied"], alpha=0.18, lw=0
    )
    ax.plot(roll.index, roll["mean"], color=COLORS["implied"], label=f"Rolling {w}-month mean")
    full = float(monthly["payoff_scaled"].mean())
    ax.axhline(full, color=COLORS["ink2"], ls=":", lw=1.2, label=f"Sample mean {full:.0%}")
    tr = _get(t, "R_time", "payoff_scaled", "trend_slope_per_year")
    yrs = np.asarray((monthly.index - monthly.index[0]).days / 365.25)
    icpt = float(monthly["payoff_scaled"].mean() - tr["estimate"] * yrs.mean())
    p_holm = float(claims.set_index("id").loc["R-9", "p_holm"])
    ax.plot(
        monthly.index,
        icpt + tr["estimate"] * yrs,
        color=COLORS["ink"],
        ls="--",
        lw=1.1,
        label=f"Linear trend, all months ({100 * tr['estimate']:+.2f}%/yr; "
        f"p = {tr['p_value']:.3f}, Holm-adjusted p = {p_holm:.3f})",
    )
    ax.axhline(0, color=COLORS["ink2"], lw=0.8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlim(roll.index[0], roll.index[-1])
    headroom(ax, float(roll["ci_high"].max()), cfg["plots"]["headroom"])
    shade_events(ax, events, cfg["plots"]["shade_events"])
    mark_live_vix(ax, cfg)
    ax.set_ylabel("Scaled seller payoff (% of implied variance)")
    ax.set_xlabel("Month (end of rolling window)")
    ax.set_title("Is the variance risk premium changing over time?", pad=42)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8.5)
    _date_axis(ax)
    _foot_wrapped(
        fig,
        cfg["plots"]["source_note"] + f". {SIGN_FOOT} Band: Newey–West "
        f"{cfg['stats']['nw_lags_monthly']}-lag 95% CI of each window's mean (windows overlap, so "
        "the band is not a test). The trend test is in research_claims.md (R-9)."
        + (
            f" Windows ending before {live + pd.DateOffset(months=w):%b %Y} include "
            f"back-calculated (pre-{live:%b %Y}) VIX months."
            if roll.index[0] < live
            else ""
        ),
    )
    return save(fig, "15_rolling_premium", cfg)


def _years(rate: np.ndarray) -> np.ndarray:
    """Rate per year → years between events (inf for a zero rate)."""
    rate = np.asarray(rate, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(rate > 0, 1 / rate, np.inf)


def fig_breakeven(res: object, claims: pd.DataFrame, cfg: Config) -> list[Path]:
    """16: years between crashes needed to break even vs years between crashes observed."""
    curve = res.breakeven  # type: ignore[attr-defined]
    cl = claims.set_index("id")
    panels = [
        ("payoff_scaled", "Scaled payoff (share of implied variance)", "R-10"),
        ("payoff_var", "Raw variance payoff (annualised variance)", "R-11"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    top = 1.0
    for ax, (col, label, cid) in zip(axes, panels, strict=True):
        c = curve[curve["measure"] == col].sort_values("threshold", ascending=False)
        x = np.arange(len(c))
        be, be_lo, be_hi = (
            _years(c[k]) for k in ("breakeven_rate", "breakeven_rate_hi", "breakeven_rate_lo")
        )
        ob, ob_lo, ob_hi = (
            _years(c[k]) for k in ("observed_rate", "observed_rate_hi", "observed_rate_lo")
        )
        ax.errorbar(
            x - 0.12,
            be,
            yerr=[be - be_lo, be_hi - be],
            fmt="o",
            ms=7,
            capsize=4,
            color=COLORS["implied"],
            label="Break-even: crash needed once every … years",
        )
        ax.errorbar(
            x + 0.12,
            ob,
            yerr=[ob - ob_lo, ob_hi - ob],
            fmt="s",
            ms=7,
            capsize=4,
            color=COLORS["realised"],
            label="Observed: one crash every … years",
        )
        top = max(top, float(np.nanmax(ob_hi[np.isfinite(ob_hi)])))
        ax.set_yscale("log")
        ticks = [
            f"≤ {t:.3g}".replace("-", "\u2212") + (f"\n(worst: {m})" if w else "")
            for t, w, m in zip(c["threshold"], c["is_worst"], c["worst_month"], strict=True)
        ]
        ax.set_xticks(x, ticks, fontsize=9)
        ax.set_xlabel(f"Crash = month with payoff at or below … ({label.split(' (')[0].lower()})")
        r = cl.loc[cid]
        verdict = "supported" if r["supported"] else "not supported"
        ax.set_title(f"{label}\n{cid} {verdict} (Holm-adjusted p = {r['p_holm']:.2g})", fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Years between crashes (log scale)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_ylim(0.3, top * 2)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "How often would crashes have to happen to wipe out the premium?",
        fontweight="bold",
        fontsize=13,
        y=1.02,
    )
    _foot_wrapped(
        fig,
        cfg["plots"]["source_note"] + f". {SIGN_FOOT} Break-even: a share g / (g − L) of "
        "months would have to be crashes of the mean size L for the mean payoff to be zero "
        "(g = mean of the other months; 95% CI from the block bootstrap). Observed: crashes "
        "seen in the sample, exact Poisson 95% CI. A blue point below the whole orange bar = "
        "the premium covers crashes of that size even if they are as frequent as the data allow.",
    )
    return save(fig, "16_breakeven", cfg)


def fig_case_studies(res: object, cfg: Config) -> list[Path]:
    """17: the two case-study months day by day: S&P, VIX, accrued realised vs implied variance."""
    paths = res.case_paths  # type: ignore[attr-defined]
    t = res.tables  # type: ignore[attr-defined]
    cases = list(paths["case"].drop_duplicates())
    h = cfg.window
    fig, axes = plt.subplots(
        3, len(cases), figsize=(12.5, 8.6), sharex=True, sharey="row", squeeze=False
    )
    for j, case in enumerate(cases):
        p = paths[paths["case"] == case]
        d = p["day"].to_numpy()
        a0, a1, a2 = axes[0, j], axes[1, j], axes[2, j]
        a0.plot(d, 100 * (p["spx_rel"] - 1), color=COLORS["ink"])
        a0.axhline(0, color=COLORS["ink2"], lw=0.8)
        a1.plot(d, p["vix_close"], color=COLORS["implied"], label="VIX")
        a1.axhline(
            float(p.loc[p["day"] == 0, "vix_close"].iloc[0]),
            color=COLORS["implied"],
            ls=":",
            lw=1.1,
            label="VIX at t (the implied vol sold)",
        )
        acc = p["accrued_over_implied"]
        a2.fill_between(d, 0, acc, where=d >= 0, color=COLORS["realised"], alpha=0.25, lw=0)
        a2.plot(d, acc, color=COLORS["realised"], label="Realised variance accrued so far")
        a2.axhline(
            1,
            color=COLORS["implied"],
            ls="--",
            lw=1.2,
            label="Implied variance sold at t",
        )

        def get(stat: str, case: str = case) -> float:
            m = (t["test"] == "R_case") & (t["sample"] == case) & (t["statistic"] == stat)
            return float(t.loc[m, "estimate"].iloc[0])

        gone = get("premium_gone_by_day")
        if np.isfinite(gone):
            a2.axvline(gone, color=COLORS["loss"], lw=1, ls=":")
            a2.annotate(
                f"premium gone\nby day {gone:.0f}",
                (gone, 1),
                xytext=(6, 18),
                textcoords="offset points",
                fontsize=8.5,
                color=COLORS["loss"],
                bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5},
            )
        a2.annotate(
            f"day {h}: {acc.iloc[-1]:.1f}× implied\nscaled payoff "
            f"{get('payoff_scaled'):.1f}".replace("-", "\u2212"),
            (d[-1], acc.iloc[-1]),
            xytext=(-14, -16),
            textcoords="offset points",
            bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5},
            ha="right",
            va="top",
            fontsize=8.5,
        )
        for ax in (a0, a1, a2):
            ax.axvline(0, color=COLORS["ink2"], lw=0.9)
            ax.axvspan(0.5, h + 0.5, color=COLORS["grid"], alpha=0.35, lw=0)
        label = pd.Period(case, "M").strftime("%B %Y")
        a0.set_title(
            f"{label}: VIX {get('vix_at_t'):.1f} at t, realised vol "
            f"{get('realised_vol_fwd'):.0f} over the next {h} days",
            fontsize=11,
        )
        a2.set_xlabel("Trading days from t")
    axes[0, 0].set_ylabel("S&P 500 vs close at t (%)")
    axes[1, 0].set_ylabel("VIX (vol points)")
    axes[2, 0].set_ylabel("Realised ÷ implied variance (×)")
    axes[1, 0].legend(loc="upper left", fontsize=8.5)
    axes[2, 0].legend(loc="upper left", fontsize=8.5)
    for a in axes[2]:
        a.set_ylim(0, None)
    fig.suptitle(
        "Case studies: how the seller's premium disappeared", fontweight="bold", fontsize=13, y=1.0
    )
    fig.tight_layout()
    _foot_wrapped(
        fig,
        cfg["plots"]["source_note"] + f". {SIGN_FOOT} t = first trading day of the month; "
        f"shaded = the {h}-day window. Realised variance accrued by day k = "
        f"({cfg.annualisation} / {h})·Σ r² over days 1…k, so on day {h} it is the month's "
        "realised variance; dividing by the implied variance at t gives 1 − scaled payoff. "
        "Outcomes after t: a description of what happened, not a signal.",
    )
    return save(fig, "17_case_studies", cfg)


def make_research_figures(
    res: object, monthly: pd.DataFrame, events: pd.DataFrame, claims: pd.DataFrame, cfg: Config
) -> list[Path]:
    """Draw research figures 12–17 from a ``vrp.research.ResearchResults``."""
    apply_style()
    paths: list[Path] = []
    paths += fig_rv_dynamics(res, events, cfg)
    paths += fig_rv_by_year(res, cfg)
    paths += fig_conditional_premium(res, cfg)
    paths += fig_rolling_premium(res, monthly, events, claims, cfg)
    paths += fig_breakeven(res, claims, cfg)
    paths += fig_case_studies(res, cfg)
    return paths
