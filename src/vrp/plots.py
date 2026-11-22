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
    """Add the data-source footnote just below the bottom-left of the figure."""
    fig.text(0.01, -0.01, text, ha="left", va="top", fontsize=8, color=COLORS["ink2"])


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
        cfg["plots"]["source_note"] + f". {SIGN_FOOT} Monthly 1990–2025 (n = 432). "
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
    footnote(fig, cfg["plots"]["source_note"] + f". {SIGN_FOOT} Monthly 1990–2025, n = 432.")
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
        cfg["plots"]["source_note"] + ". Mincer–Zarnowitz regressions, monthly 1990–2025 "
        f"(n = 432); Newey–West {cfg['stats']['nw_lags_monthly']} lags; brackets = 95% CIs. "
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
    paths += fig_test5_robustness(t, cfg)
    return paths
