"""Markdown outputs for the research section: ``research_summary.md`` and ``research_claims.md``.

The claims below are the tests pre-registered in ``docs/research_plan.md`` (R-1 … R-9, and the
step 5 break-even tests R-10 / R-11). They form their own Holm family, separate from the 23 v2
claims, and use the same rule: every CI excludes the null in the claimed direction and the
Holm-adjusted p is below alpha.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vrp.logging_setup import get_logger
from vrp.research import CAL_SAMPLE, MONTHLY_SAMPLE, ResearchResults, bucket_labels
from vrp.stats_report import (
    SIGN_NOTE,
    ClaimSpec,
    Evidence,
    evaluate_specs,
    fmt,
    fmt_p,
    pct,
    tidy,
)
from vrp.validate import df_to_md

log = get_logger(__name__)


def _observed_text(res: ResearchResults, measure: str) -> tuple[str, str]:
    """('March 2020', 'once in 36 years') for the worst-month break-even scenario of ``measure``."""
    c = res.breakeven[(res.breakeven["measure"] == measure) & res.breakeven["is_worst"]].iloc[0]
    k = int(c["n_crash"])
    times = "once" if k == 1 else f"{k} times"
    label = pd.Period(str(c["worst_month"]), "M").strftime("%B %Y")
    return label, f"{times} in {c['years']:.0f} years"


def research_specs(res: ResearchResults) -> list[ClaimSpec]:
    """The pre-registered research claims, each tied to rows of ``res.tables``."""
    E = Evidence
    rs = res.research
    worst_s, obs_s = _observed_text(res, "payoff_scaled")
    worst_v, obs_v = _observed_text(res, "payoff_var")
    t = res.tables
    be_v = float(
        t.loc[
            (t["test"] == "R_breakeven")
            & (t["measure"] == "payoff_var")
            & (t["statistic"] == "breakeven_years_between"),
            "estimate",
        ].iloc[0]
    )
    be_caveat = (
        "In-sample. By construction the break-even frequency exceeds the observed one whenever "
        "the mean payoff is positive; the test asks how unlucky we would have to have been to "
        "see so few crashes. 1987 is outside the sample, and a crash bigger than any seen "
        "since 1990 would change the answer."
    )
    vix = bucket_labels("VIX", rs.vix_thresholds, "")
    gap = bucket_labels("gap", rs.gap_thresholds, " pts")
    dyn = {"sample": CAL_SAMPLE}
    return [
        ClaimSpec(
            "R-1",
            "A",
            "Volatility clusters: a high-vol month tends to be followed by another (AR(1) "
            "coefficient of monthly log realised variance {est}, 95% CI {lo} to {hi}).",
            ">",
            (E("R_dynamics", "log_rv_cal", "ar1_rho", **dyn),),
            caveat="Persistence fades fast: see the half-life and autocorrelations in the summary.",
        ),
        ClaimSpec(
            "R-2",
            "A",
            "Volatility shocks fade: realised variance is mean-reverting (augmented "
            "Dickey–Fuller statistic {est} rejects a unit root).",
            "reject",
            (E("R_dynamics", "log_rv_cal", "adf_stat", **dyn),),
        ),
        ClaimSpec(
            "R-3",
            "A",
            "Realised volatility has risen gradually since 1990 (trend in log realised "
            "variance {est} per year, 95% CI {lo} to {hi}).",
            ">",
            (E("R_dynamics", "log_rv_cal", "trend_slope_per_year", **dyn),),
            fallback="Say: there is no evidence of a long-run trend in realised volatility since "
            "1990; it moves in spikes that fade, around a stable level.",
        ),
        ClaimSpec(
            "R-4",
            "A",
            "Volatility jumps up faster than it falls: month-to-month changes in log realised "
            "variance are positively skewed (skewness {est}, 95% CI {lo} to {hi}).",
            ">",
            (E("R_dynamics", "dlog_rv_cal", "skewness", **dyn),),
            fallback="Say: month-to-month changes in volatility lean towards sharp rises (positive "
            "sample skewness), but the effect does not survive the multiple-testing correction; "
            "show it on the chart rather than claim it.",
        ),
        ClaimSpec(
            "R-5",
            "B",
            f"In raw variance the seller earns more when the VIX starts high: {vix[2]} months "
            f"pay {{est}} more than {vix[0]} months (95% CI {{lo}} to {{hi}}).",
            ">",
            (
                E(
                    "R_conditional",
                    "payoff_var",
                    "diff_mean_high_minus_low",
                    f"{vix[2]} minus {vix[0]}",
                ),
            ),
            caveat="Known-at-t buckets, in-sample: a description, not an out-of-sample rule.",
            fallback=f"Say: the raw premium is larger on average in {vix[2]} months, but the "
            "difference is not significant: high-VIX months also contain the biggest losses.",
        ),
        ClaimSpec(
            "R-6",
            "B",
            f"As a share of implied variance the premium is smaller when the VIX starts high: "
            f"{vix[2]} minus {vix[0]} = {{est}} (95% CI {{lo}} to {{hi}}).",
            "<",
            (
                E(
                    "R_conditional",
                    "payoff_scaled",
                    "diff_mean_high_minus_low",
                    f"{vix[2]} minus {vix[0]}",
                ),
            ),
            "pct",
            caveat="Known-at-t buckets, in-sample: a description, not an out-of-sample rule.",
            fallback="Say: the point estimate is lower in high-VIX months, but we cannot "
            "distinguish it from the low-VIX premium.",
        ),
        ClaimSpec(
            "R-7",
            "B",
            f"Even when the VIX starts high ({vix[2]}) the seller wins on average: mean scaled "
            "payoff {est} (95% CI {lo} to {hi}).",
            ">",
            (E("R_conditional", "payoff_scaled", "mean", vix[2]),),
            "pct",
            fallback=f"Say: in {vix[2]} months the seller still wins most months (see the win "
            "rate), but the average is not significantly above zero because those months hold "
            "the largest crashes.",
        ),
        ClaimSpec(
            "R-8",
            "B",
            f"The premium is larger when the VIX sits well above recent realised vol: {gap[2]} "
            f"months pay {{est}} of implied variance more than {gap[0]} months (95% CI {{lo}} "
            "to {hi}).",
            ">",
            (
                E(
                    "R_conditional",
                    "payoff_scaled",
                    "diff_mean_high_minus_low",
                    f"{gap[2]} minus {gap[0]}",
                ),
            ),
            "pct",
            caveat="Known-at-t buckets, in-sample: a description, not an out-of-sample rule.",
            fallback="Say: the point estimate is larger when the VIX is well above recent "
            "realised vol, but with only a few dozen inverted months the gap is not significant.",
        ),
        ClaimSpec(
            "R-9",
            "C",
            "The scaled premium has shrunk over time: linear trend {est} of implied variance "
            "per year (95% CI {lo} to {hi}).",
            "<",
            (E("R_time", "payoff_scaled", "trend_slope_per_year", MONTHLY_SAMPLE),),
            "pct",
            caveat="A linear trend through 35 years of spiky data; pre-2003 VIX values are "
            "back-calculated. Show the rolling chart next to it.",
            fallback="Say: there is no significant evidence that the premium is shrinking once "
            "we correct for multiple testing.",
        ),
        ClaimSpec(
            "R-10",
            "D",
            f"The premium more than pays for crashes: for the seller to break even, a loss the "
            f"size of {worst_s} would have to strike once every {{est}} years (95% CI {{lo}} to "
            f"{{hi}}), measured as a share of implied variance; it struck {obs_s}.",
            "<",
            (E("R_breakeven", "payoff_scaled", "breakeven_years_between", MONTHLY_SAMPLE),),
            caveat=be_caveat,
        ),
        ClaimSpec(
            "R-11",
            "D",
            f"In raw variance too, break-even needs a loss the size of {worst_v} once every "
            f"{{est}} years (95% CI {{lo}} to {{hi}}); it struck {obs_v}.",
            "<",
            (E("R_breakeven", "payoff_var", "breakeven_years_between", MONTHLY_SAMPLE),),
            caveat=be_caveat,
            fallback=f"Say: in raw variance the margin is thinner: break-even needs a "
            f"{worst_v}-sized loss about every {be_v:.0f} years against {obs_v}; with so few "
            "crashes observed we cannot rule out that they are that frequent.",
        ),
    ]


def _curve_table(c: pd.DataFrame) -> pd.DataFrame:
    """Break-even curve → display table (years between crashes)."""

    def yrs(rate: float) -> str:
        return "∞" if rate <= 0 else f"{1 / rate:.1f}"

    return pd.DataFrame(
        {
            "measure": c["measure"],
            "crash = payoff ≤": [
                f"{v:.4g}" + (" (worst month)" if w else "")
                for v, w in zip(c["threshold"], c["is_worst"], strict=True)
            ],
            "crashes seen": c["n_crash"],
            "mean crash L": c["crash_mean"].map(fmt),
            "ordinary mean g": c["ordinary_mean"].map(fmt),
            "break-even: once every … years [95% CI]": [
                f"{yrs(r)} [{yrs(h)}, {yrs(lo)}]"
                for r, lo, h in zip(
                    c["breakeven_rate"], c["breakeven_rate_lo"], c["breakeven_rate_hi"], strict=True
                )
            ],
            "observed: once every … years [95% CI]": [
                f"{yrs(r)} [{yrs(h)}, {yrs(lo)}]"
                for r, lo, h in zip(
                    c["observed_rate"], c["observed_rate_lo"], c["observed_rate_hi"], strict=True
                )
            ],
            "conservative p": c["p_conservative"].map(lambda p: fmt_p(p)),
        }
    )


def _case_table(t: pd.DataFrame) -> pd.DataFrame:
    """Case-study rows → one column per month; month-specific notes go into the cell."""
    out = []
    for stat, g in t.groupby("statistic", sort=False):
        notes = g["note"].fillna("")
        specific = notes.nunique() > 1
        rec = {"statistic": stat, "note": "" if specific else notes.iloc[0]}
        for r in g.itertuples():
            rec[r.sample] = fmt(r.estimate) + (f" ({r.note})" if specific and r.note else "")
        out.append(rec)
    df = pd.DataFrame(out)
    return df[["statistic"] + list(t["sample"].drop_duplicates()) + ["note"]]


def evaluate_research_claims(res: ResearchResults) -> pd.DataFrame:
    """Check the research claims (own Holm family)."""
    s = res.settings
    specs = research_specs(res)
    return evaluate_specs(specs, res.tables, [c.cid for c in specs], s.alpha, 1 / (s.n_boot + 1))


def _rule_text(res: ResearchResults, n_claims: int) -> str:
    s = res.settings
    return (
        "**Rule for a claim to be usable:** every 95% CI behind it excludes the null in the "
        f"claimed direction, and its largest p-value, after a Holm–Bonferroni correction across "
        f"the {n_claims} pre-registered research claims (a family separate from the v2 claims), "
        f"is below {s.alpha}. Pre-registration: `docs/research_plan.md`."
    )


def write_research_claims(res: ResearchResults, claims: pd.DataFrame, path: Path) -> Path:
    """Write ``research_claims.md``."""
    floor = 1 / (res.settings.n_boot + 1)
    sup, uns = claims[claims["supported"]], claims[~claims["supported"]]
    L = [
        "# Research claims log",
        "",
        "_Generated by `python -m vrp.pipeline --stage research` from `research_tables.csv`. "
        "Do not edit by hand; change the code or config and re-run._",
        "",
        SIGN_NOTE,
        "",
        _rule_text(res, len(claims)),
        "",
        f"## ✅ Supported claims ({len(sup)})",
        "",
    ]
    for r in sup.itertuples():
        L += [
            f"### {r.id} (section {r.test}). {r.claim}",
            "",
            f"- Holm-adjusted p = {fmt_p(r.p_holm, floor)} (raw p = {fmt_p(r.p_max, floor)})",
            f"- Evidence: {r.evidence}",
        ]
        if r.caveat:
            L.append(f"- ⚠️ Caveat: {r.caveat}")
        L.append("")
    L += [f"## ❌ Not supported: do not claim ({len(uns)})", ""]
    for r in uns.itertuples():
        L += [
            f"### {r.id} (section {r.test}). ~~{r.claim}~~",
            "",
            f"- Why not: {r.reasons}",
            f"- **Say instead:** {r.fallback}",
            f"- Evidence: {r.evidence}",
        ]
        if r.caveat:
            L.append(f"- Caveat: {r.caveat}")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s (%d/%d supported)", path.name, len(sup), len(claims))
    return path


def write_research_summary(res: ResearchResults, claims: pd.DataFrame, path: Path) -> Path:
    """Write ``research_summary.md``: every result table plus the claims verdicts."""
    s, rs = res.settings, res.research
    floor = 1 / (s.n_boot + 1)
    t = res.tables
    yr = res.rv_year.reset_index()
    top = yr.sort_values("rank_irregular").head(10)
    show_year = lambda d: d.assign(  # noqa: E731
        rv_ann_vol_pct=d["rv_ann_vol_pct"].round(1),
        vol_of_vol_log=d["vol_of_vol_log"].round(2),
        max_month_vol_pct=d["max_month_vol_pct"].round(1),
        min_month_vol_pct=d["min_month_vol_pct"].round(1),
    )
    last = res.rolling.iloc[-1]
    hi_roll = res.rolling["mean"].idxmax()
    lo_roll = res.rolling["mean"].idxmin()
    L = [
        "# Research section: summary of results",
        "",
        "_Generated by `python -m vrp.pipeline --stage research`. Do not edit by hand._",
        "",
        SIGN_NOTE,
        "",
        "Pre-registered tests: `docs/research_plan.md`. Claims verdicts: "
        "`research_claims.md`. Variances are annualised decimals; vol is in percentage points.",
        "",
        "## Verdicts",
        "",
        df_to_md(
            claims.assign(
                verdict=claims["supported"].map({True: "✅ supported", False: "❌ not supported"}),
                p_holm=claims["p_holm"].map(lambda p: fmt_p(p, floor)),
            )[["id", "verdict", "p_holm", "claim"]]
        ),
        "",
        "## A. Realised-volatility dynamics (calendar-month realised variance, logs)",
        "",
        "Series: realised variance of each calendar month, (252 / n_days) · Σ r², "
        f"{len(res.rv_monthly)} months. Non-overlapping, unlike the 21-day forward windows.",
        "",
        df_to_md(tidy(t[t["test"] == "R_dynamics"], floor)),
        "",
        "### Most irregular years (vol-of-vol = std of the 12 monthly log realised variances)",
        "",
        df_to_md(show_year(top)),
        "",
        "### Every year",
        "",
        df_to_md(show_year(yr)),
        "",
        "## B. Seller payoff by conditions known at the close of t",
        "",
        f"Buckets use fixed thresholds from `config.yaml`: VIX {rs.vix_thresholds}; gap = VIX "
        f"− past 21-day realised vol, {rs.gap_thresholds} vol points. Bucket means use "
        "Newey–West within the bucket (months are not contiguous, so treat it as approximate); "
        "high − low differences use the block bootstrap on the full series.",
        "",
        df_to_md(tidy(t[(t["test"] == "R_conditional") & (t["statistic"] != "win_rate")], floor)),
        "",
        "## C. Is the premium changing?",
        "",
        df_to_md(tidy(t[t["test"] == "R_time"], floor)),
        "",
        f"Rolling {rs.rolling_window}-month mean of the scaled payoff (chart only; windows "
        f"overlap): latest {pct(last['mean'])} (window ending {last.name:%Y-%m}); highest "
        f"{pct(res.rolling['mean'].max())} (ending {hi_roll:%Y-%m}); lowest "
        f"{pct(res.rolling['mean'].min())} (ending {lo_roll:%Y-%m}).",
        "",
        "## D. Break-even crash frequency, Jensen gap, case studies",
        "",
        "### Break-even: how often would a crash this size have to happen?",
        "",
        "Scenario: a loss the size of the worst month, L (held fixed). Ordinary months earn g. "
        "The mean payoff is zero when a share f* = g / (g − L) of months are such crashes; "
        "rates are per year. The CI on the break-even rate resamples the ordinary months "
        "(block bootstrap). The p-value is the exact Poisson probability of seeing no more "
        "crashes than we did if they arrived at the *low* end of that CI (conservative).",
        "",
        df_to_md(tidy(t[t["test"] == "R_breakeven"], floor)),
        "",
        "Curve for the chart (crash = month at or below the threshold; L is then the mean of "
        "those months):",
        "",
        df_to_md(_curve_table(res.breakeven)),
        "",
        "### Jensen gap: how much of 'VIX above realised vol' is convexity?",
        "",
        "The VIX prices variance, so even a VIX with zero premium sits above the realised vol "
        "that follows on average (√ is concave). Conditional version: regress log forward RV on "
        "log VIX² and log past RV (both known at t); the residuals give E_t[√RV] / √E_t[RV] "
        "without assuming normality (smearing). A larger residual variance means more "
        "convexity, so a forecast that uses only these two inputs overstates it: the share is "
        "an upper estimate. Descriptive; block bootstrap CIs.",
        "",
        df_to_md(tidy(t[t["test"] == "R_jensen"], floor)),
        "",
        "### Case studies",
        "",
        df_to_md(_case_table(t[t["test"] == "R_case"])),
        "",
        "Daily paths: `research_case_paths.csv` (day 0 = t; accrued realised variance on the "
        "swap's scale, so day 21 equals the month's forward realised variance).",
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s", path.name)
    return path
