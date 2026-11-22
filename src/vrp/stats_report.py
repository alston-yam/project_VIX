"""Markdown outputs for v2: ``stats_summary.md`` and the pitch-claims log ``pitch_claims.md``.

Every number in both files is read from the result tables built by ``vrp.stats``; nothing
is typed in by hand. A claim is marked **supported** only if, for every piece of evidence
behind it, the confidence interval excludes the null *in the claimed direction*, and its
p-value, after a Holm–Bonferroni correction across all claims, is below alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vrp.logging_setup import get_logger
from vrp.stats import UNITS, StatsResults, holm
from vrp.validate import df_to_md

log = get_logger(__name__)

SIGN_NOTE = (
    "**Sign convention: implied − realised.** Positive = the VIX over-forecast and a "
    "variance seller made money. (Carr & Wu use the opposite sign.)"
)


# --------------------------------------------------------------------------- formatting
def fmt(v: float, digits: int = 4) -> str:
    """Compact number: 4 significant digits, no scientific notation for normal sizes."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    v = float(v)
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.{digits - 1}e}"
    return f"{v:.{digits}g}"


def fmt_p(p: float, floor: float | None = None) -> str:
    """p-value text. With ``floor`` (resampling tests), values at the floor show as '≤ floor'."""
    if p is None or np.isnan(p):
        return ""
    if floor is not None and p <= floor * 1.0001:
        return f"≤ {floor:.1g}"
    if p == 0:
        return "< 1e-300"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3g}" if p < 0.1 else f"{p:.2f}"


def p_text(p: float, test_name: str, floor: float) -> str:
    """Use the resampling floor only for bootstrap / permutation p-values."""
    resampled = any(w in str(test_name) for w in ("bootstrap", "permutation"))
    return fmt_p(p, floor if resampled else None)


def pct(v: float) -> str:
    """Fraction as a percentage with one decimal."""
    return f"{100 * v:.1f}%"


def tidy(df: pd.DataFrame, floor: float) -> pd.DataFrame:
    """Result rows → display table."""
    out = pd.DataFrame(
        {
            "measure": df["measure"],
            "sample": df["sample"],
            "statistic": df["statistic"],
            "estimate": df["estimate"].map(fmt),
            "95% CI": [
                "" if np.isnan(lo) else f"[{fmt(lo)}, {fmt(hi)}]"
                for lo, hi in zip(df["ci_low"], df["ci_high"], strict=True)
            ],
            "CI method": df["ci_method"],
            "H0": df["null_value"].map(fmt),
            "test": df["test_name"],
            "p (two-sided)": [
                p_text(p, tn, floor) for p, tn in zip(df["p_value"], df["test_name"], strict=True)
            ],
            "n": df["n"].map(lambda n: "" if np.isnan(n) else str(int(n))),
            "note": df["note"],
        }
    )
    if "p_holm_within_measure" in df and df["p_holm_within_measure"].notna().any():
        out.insert(9, "p Holm", df["p_holm_within_measure"].map(lambda p: fmt_p(p, floor)))
    return out


# --------------------------------------------------------------------------- claims
@dataclass(frozen=True)
class Evidence:
    """Selects one result row: (test, measure, statistic, sample, ci_method substring)."""

    test: str
    measure: str
    statistic: str
    sample: str = "monthly 1990–2025"
    method: str = ""


@dataclass(frozen=True)
class ClaimSpec:
    """A statement we might make in the pitch, and the evidence it needs."""

    cid: str
    test: str
    text: str  # may use {est}, {lo}, {hi} from the first evidence row (formatted by ``style``)
    direction: str  # ">" estimate above null, "<" below, "reject" p-value only
    evidence: tuple[Evidence, ...]
    style: str = "num"  # "num" | "pct" | "abspct" (percentage of the absolute value)
    caveat: str = ""
    fallback: str = ""  # what to say instead if the claim is not supported


def claim_specs(k: int, h: int, worst: int, best: int) -> list[ClaimSpec]:
    """The candidate pitch claims, each tied to rows of the result tables."""
    E = Evidence
    rob = [
        "post-2003 only (live VIX)",
        f"H = {h} trading days",
        "demeaned realised variance",
        "VXO instead of VIX (1990–2021)",
        "daily overlapping windows",
        f"excluding worst {k} months",
    ]
    return [
        ClaimSpec(
            "C1",
            "1",
            "On average the VIX over-forecasts the next month's variance: the mean scaled "
            "seller payoff is {est} of implied variance (95% CI {lo} to {hi}).",
            ">",
            (
                E("1_mean", "payoff_scaled", "mean", method="Newey"),
                E("1_mean", "payoff_scaled", "mean", method="bootstrap"),
            ),
            "pct",
        ),
        ClaimSpec(
            "C2",
            "1",
            "The raw variance payoff is positive on average: {est} in annualised variance "
            "(95% CI {lo} to {hi}).",
            ">",
            (
                E("1_mean", "payoff_var", "mean", method="Newey"),
                E("1_mean", "payoff_var", "mean", method="bootstrap"),
            ),
            caveat="Raw variance is dominated by a few crash months; quote it next to the "
            "scaled payoff.",
        ),
        ClaimSpec(
            "C3",
            "1",
            "The VIX has exceeded the volatility that followed by {est} vol points on average "
            "(95% CI {lo} to {hi}).",
            ">",
            (
                E("1_mean", "vol_spread_pts", "mean", method="Newey"),
                E("1_mean", "vol_spread_pts", "mean", method="bootstrap"),
            ),
        ),
        ClaimSpec(
            "C4",
            "1",
            "The typical (median) month pays the seller {est} of implied variance (95% CI "
            "{lo} to {hi}).",
            ">",
            (E("1_mean", "payoff_scaled", "median"),),
            "pct",
        ),
        ClaimSpec(
            "C6",
            "2",
            "The variance seller wins in {est} of months (95% CI {lo} to {hi}), far more "
            "often than a coin flip.",
            ">",
            (
                E("2_shape", "payoff_scaled", "win_rate", method="Wilson"),
                E("2_shape", "payoff_scaled", "win_rate", method="bootstrap"),
            ),
            "pct",
        ),
        ClaimSpec(
            "C7",
            "2",
            "The scaled payoff is strongly negatively skewed (skewness {est}, 95% CI {lo} to "
            "{hi}): many small gains, a few huge losses.",
            "<",
            (E("2_shape", "payoff_scaled", "skewness"),),
        ),
        ClaimSpec(
            "C8",
            "2",
            "The raw variance payoff is negatively skewed (skewness {est}, 95% CI {lo} to "
            "{hi}).",
            "<",
            (E("2_shape", "payoff_var", "skewness"),),
            fallback="Say: the raw variance payoff has a large negative sample skewness, but it "
            "is driven by a few months (the block-bootstrap CI reaches zero), so lead with the "
            "scaled payoff's skewness instead.",
        ),
        ClaimSpec(
            "C9",
            "2",
            "The payoff has fat tails: excess kurtosis {est} (95% CI {lo} to {hi}; normal = 0).",
            ">",
            (E("2_shape", "payoff_scaled", "excess_kurtosis"),),
        ),
        ClaimSpec(
            "C10",
            "2",
            "The mean sits well below the median (mean − median = {est}, 95% CI {lo} to {hi}): "
            "a long left tail of bad months pulls the average down.",
            "<",
            (E("2_shape", "payoff_scaled", "mean_minus_median"),),
            "pct",
        ),
        ClaimSpec(
            "C11",
            "2",
            "Extreme losing months are far more common than a bell curve predicts: {est} "
            "months fall more than 3 standard deviations below the mean (95% CI {lo} to {hi}).",
            ">",
            (E("2_shape", "payoff_scaled", "months_below_mean_minus_3sd"),),
        ),
        ClaimSpec(
            "C12",
            "2",
            "The average losing month is larger than the average winning month (|avg loss| / "
            "avg gain = {est}, 95% CI {lo} to {hi}).",
            ">",
            (E("2_shape", "payoff_scaled", "loss_to_gain_ratio"),),
            fallback="Say: the *worst* months are many times the average gain (descriptive "
            "facts below), but the *average* loss is not significantly bigger than the average "
            "gain. The lopsidedness is in the extreme tail, not the typical loss.",
        ),
        ClaimSpec(
            "C13",
            "3",
            "Seller payoffs move with the S&P 500 over the same 21 days (correlation {est}, "
            "95% CI {lo} to {hi}): the seller loses when stocks fall.",
            ">",
            (
                E("3_crash", "payoff_scaled", "corr_with_spx_fwd21_ret", method="Fisher"),
                E("3_crash", "payoff_scaled", "corr_with_spx_fwd21_ret", method="bootstrap"),
            ),
            caveat="Same-window outcomes: this describes when losses happen, it is not a "
            "forecast made at t.",
        ),
        ClaimSpec(
            "C14",
            "3",
            "In sell-off months (S&P down more than 5% over the window) the seller loses on "
            "average: mean scaled payoff {est} (95% CI {lo} to {hi}).",
            "<",
            (E("3_crash", "payoff_scaled", "mean_in_selloff"),),
            "pct",
            caveat="Conditioning on an outcome: descriptive, not forecastable at t.",
        ),
        ClaimSpec(
            "C15",
            "3",
            "Sell-off months pay the seller {est} of implied variance less than other months "
            "(95% CI of the gap {hi} to {lo}).",
            "<",
            (E("3_crash", "payoff_scaled", "diff_mean_selloff_minus_other"),),
            "abspct",
        ),
        ClaimSpec(
            "C16",
            "3",
            f"The worst 10% of S&P months (decile {worst}) carry {{est}} of all variance-"
            "seller losses (95% CI {lo} to {hi}); with no clustering they would carry about 10%.",
            ">",
            (E("3_crash", "payoff_var", f"loss_share_in_decile_{worst}"),),
            "pct",
        ),
        ClaimSpec(
            "C17",
            "3",
            f"It is the direction of the market, not just big moves: the worst S&P decile pays "
            f"the seller {{est}} of implied variance less than the best decile (decile {best}), "
            "although both contain large moves (95% CI of the gap {hi} to {lo}).",
            "<",
            (E("3_crash", "payoff_scaled", f"diff_mean_decile_{worst}_minus_decile_{best}"),),
            "abspct",
        ),
        ClaimSpec(
            "C18",
            "4",
            "The VIX is a biased forecast of realised variance: an unbiased forecast "
            "(RV = 0 + 1·IV) is rejected (HAC Wald χ² = {est}).",
            "reject",
            (
                E("4_forecast", "variance", "joint_a0_b1"),
                E("4_forecast", "vol_pct", "joint_a0_b1"),
                E("4_forecast", "log_variance", "joint_a0_b1"),
            ),
        ),
        ClaimSpec(
            "C19",
            "4",
            "In log terms the VIX's bias is a constant over-statement: intercept {est} (95% CI "
            "{lo} to {hi}) with a slope not significantly different from 1.",
            "<",
            (E("4_forecast", "log_variance", "intercept_a"),),
            caveat="'Slope not different from 1' is absence of evidence, not proof that b = 1.",
            fallback="Say: in logs the point estimate suggests a constant proportional "
            "over-statement, but it does not survive the multiple-testing correction. Use the "
            "joint unbiasedness rejection instead.",
        ),
        ClaimSpec(
            "C20",
            "5",
            "The scaled premium is positive in the live-VIX era alone (post-Sept 2003): mean "
            "{est} (95% CI {lo} to {hi}).",
            ">",
            (E("5_robustness", "payoff_scaled", "mean", "post-2003 only (live VIX)"),),
            "pct",
        ),
        ClaimSpec(
            "C21",
            "5",
            "The raw variance premium is positive in the live-VIX era alone (post-Sept 2003): "
            "mean {est} (95% CI {lo} to {hi}).",
            ">",
            (E("5_robustness", "payoff_var", "mean", "post-2003 only (live VIX)"),),
            fallback="Say: since 2003 the raw variance premium is positive but not "
            "statistically significant: raw variance is dominated by a few crash months (see the "
            "worst-months table). Quote the scaled premium for the post-2003 era instead.",
        ),
        ClaimSpec(
            "C22",
            "5",
            "The scaled premium has been smaller since 2003 than before: difference {est} "
            "(95% CI {lo} to {hi}).",
            "<",
            (E("5_robustness", "payoff_scaled", "diff_mean", "post-2003 minus pre-2003"),),
            "pct",
            caveat="Pre-2003 VIX values are back-calculated, so part of the gap may be "
            "measurement, not economics.",
            fallback="Say: the point estimate is lower after 2003, but the difference is not "
            "significant once we correct for multiple testing.",
        ),
        ClaimSpec(
            "C23",
            "5",
            "The positive scaled premium survives every robustness check (post-2003, H = 22, "
            "demeaned RV, VXO, daily overlapping windows with Newey–West, dropping the worst "
            "months).",
            ">",
            tuple(E("5_robustness", "payoff_scaled", "mean", smp) for smp in rob),
        ),
        ClaimSpec(
            "C24",
            "5",
            "The positive raw variance premium survives every robustness check.",
            ">",
            tuple(E("5_robustness", "payoff_var", "mean", smp) for smp in rob),
            fallback="Say: the raw variance premium is significant in most checks, but not in: "
            "{failed}.",
        ),
    ]


def _find(tables: pd.DataFrame, e: Evidence) -> pd.Series:
    m = (
        (tables["test"] == e.test)
        & (tables["measure"] == e.measure)
        & (tables["statistic"] == e.statistic)
        & (tables["sample"] == e.sample)
    )
    if e.method:
        m &= tables["ci_method"].str.contains(e.method, regex=False)
    hit = tables[m]
    if len(hit) != 1:
        raise KeyError(f"evidence {e} matched {len(hit)} rows")
    return hit.iloc[0]


def evaluate_claims(res: StatsResults) -> pd.DataFrame:
    """Check every candidate claim against the evidence; Holm-adjust across claims."""
    s = res.settings
    specs = claim_specs(s.n_exclude_worst, s.robustness_window, s.worst_decile, s.best_decile)
    out = []
    for i, c in enumerate(specs, start=1):
        rows = [_find(res.tables, e) for e in c.evidence]
        ok_dir, reasons, failed = True, [], []
        for r in rows:
            est, lo, hi, null = r["estimate"], r["ci_low"], r["ci_high"], r["null_value"]
            if c.direction == "reject":
                continue
            if c.direction == ">":
                good = est > null and (np.isnan(lo) or lo > null)
            else:
                good = est < null and (np.isnan(hi) or hi < null)
            if not good:
                ok_dir = False
                failed.append(str(r["sample"]))
                reasons.append(
                    f"{r['sample'] if r['sample'] != 'monthly 1990–2025' else r['measure']}: "
                    f"CI [{fmt(lo)}, {fmt(hi)}] does not exclude {fmt(null)}"
                )
        p = float(max(r["p_value"] for r in rows))
        first = rows[0]
        f = {"pct": pct, "abspct": lambda v: pct(abs(v))}.get(c.style, fmt)
        text = c.text.format(
            est=f(first["estimate"]), lo=f(first["ci_low"]), hi=f(first["ci_high"])
        )
        out.append(
            {
                "id": f"C{i}",
                "test": c.test,
                "claim": text,
                "p_max": p,
                "ci_ok": ok_dir,
                "reasons": "; ".join(reasons),
                "evidence": "; ".join(
                    f"{r['measure']} / {r['sample']} / {r['statistic']}: {fmt(r['estimate'])}"
                    + (
                        f" [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] ({r['ci_method']})"
                        if not np.isnan(r["ci_low"])
                        else ""
                    )
                    + f", {r['test_name']}, p = "
                    + p_text(r["p_value"], r["test_name"], 1 / (s.n_boot + 1))
                    for r in rows
                ),
                "caveat": c.caveat,
                "fallback": (c.fallback or "Do not make this claim.").replace(
                    "{failed}", ", ".join(failed) or "none"
                ),
            }
        )
    df = pd.DataFrame(out)
    df["p_holm"] = holm(df["p_max"].tolist(), s.alpha)
    df["supported"] = df["ci_ok"] & (df["p_holm"] < s.alpha)
    for i, r in df.iterrows():
        if not r["supported"] and r["ci_ok"]:
            df.at[i, "reasons"] = f"Holm-adjusted p = {r['p_holm']:.3g} ≥ {s.alpha}"
    return df


# --------------------------------------------------------------------------- writers
def write_claims(res: StatsResults, claims: pd.DataFrame, path: Path) -> Path:
    """Write ``pitch_claims.md``: what we can and cannot say, with the evidence."""
    s = res.settings
    floor = 1 / (s.n_boot + 1)
    sup, uns = claims[claims["supported"]], claims[~claims["supported"]]
    t = res.tables

    def get(test: str, measure: str, stat: str, method: str = "") -> pd.Series:
        return _find(t, Evidence(test, measure, stat, method=method))

    worst_rows = res.worst_months
    L = [
        "# Pitch claims log",
        "",
        "_Generated by `python -m vrp.pipeline --stage stats` from `stats_tables.csv`. "
        "Do not edit by hand; change the code or config and re-run._",
        "",
        SIGN_NOTE,
        "",
        "**Rule for a claim to be usable:** for every piece of evidence the 95% confidence "
        "interval excludes the null in the claimed direction, and the largest p-value behind "
        f"the claim, after a Holm–Bonferroni correction across all {len(claims)} candidate "
        f"claims, is below {s.alpha}. Confidence intervals use Newey–West ({s.nw_lags_monthly} "
        f"lags) or a circular block bootstrap ({s.block_length}-month blocks, {s.n_boot:,} "
        "resamples), so they allow for autocorrelation between months. "
        f"p-values shown as '≤ {floor:.1g}' are at the resampling floor.",
        "",
        "Sample: 432 months, January 1990 to December 2025 (VIX at the close of the first "
        "trading day of each month vs S&P 500 realised variance over the next 21 trading "
        "days). The payoff is an idealised one-month variance swap, held to expiry, with no "
        "costs.",
        "",
        f"## ✅ Supported claims ({len(sup)})",
        "",
    ]
    for r in sup.itertuples():
        L += [
            f"### {r.id} (Test {r.test}). {r.claim}",
            "",
            f"- Holm-adjusted p = {fmt_p(r.p_holm, floor)} (largest raw p = "
            f"{fmt_p(r.p_max, floor)})",
            f"- Evidence: {r.evidence}",
        ]
        if r.caveat:
            L.append(f"- ⚠️ Caveat: {r.caveat}")
        L.append("")
    L += [
        f"## ❌ Not supported: do not claim ({len(uns)})",
        "",
        "These statements fail the rule above. Use the honest wording under each one instead.",
        "",
    ]
    for r in uns.itertuples():
        L += [
            f"### {r.id} (Test {r.test}). ~~{r.claim}~~",
            "",
            f"- Why not: {r.reasons}",
            f"- **Say instead:** {r.fallback}",
            f"- Evidence: {r.evidence}",
        ]
        if r.caveat:
            L.append(f"- Caveat: {r.caveat}")
        L.append("")

    k = s.n_exclude_worst
    ex = get("2_shape", "payoff_var", f"worst_{k}_as_share_of_total_gains")
    sh = get("2_shape", "payoff_scaled", "sharpe_annualised")
    wg = get("2_shape", "payoff_scaled", "worst_over_avg_gain")
    ls = get("2_shape", "payoff_var", f"share_of_losses_in_worst_{k}")
    m_all = get("1_mean", "payoff_scaled", "mean", "Newey")
    m_ex = get("1_mean", "payoff_scaled", f"mean_excl_worst_{k}")
    L += [
        "## Descriptive facts (no hypothesis test; safe to quote as facts about our sample)",
        "",
        f"- Dropping just the worst {k} of {int(m_all['n'])} months raises the mean scaled "
        f"payoff from {pct(m_all['estimate'])} to {pct(m_ex['estimate'])} (block-bootstrap "
        f"95% CI of the trimmed mean [{pct(m_ex['ci_low'])}, {pct(m_ex['ci_high'])}]).",
        f"- The worst {k} months (raw variance) wiped out {pct(ex['estimate'])} of the total "
        f"gains earned in all winning months, and account for {pct(ls['estimate'])} of all "
        "losses.",
        f"- The worst month for the scaled payoff was "
        f"{worst_rows.iloc[0]['month']} at {fmt(worst_rows.iloc[0]['payoff_scaled'])} "
        f"(i.e. realised variance {fmt(1 - worst_rows.iloc[0]['payoff_scaled'])}× implied), "
        f"{fmt(abs(wg['estimate']))} times the average winning month.",
        f"- Annualised Sharpe ratio of the scaled payoff: {fmt(sh['estimate'])} (95% CI "
        f"[{fmt(sh['ci_low'])}, {fmt(sh['ci_high'])}]). **Use only to make the point that "
        "Sharpe flatters short volatility**: it ignores the skew and crash risk above.",
        "",
        "Ten worst months (scaled payoff). With first-of-month sampling a crash is credited "
        "to the month whose 21-day window contains it:",
        "",
        df_to_md(
            worst_rows.assign(
                window=lambda d: d["fwd_window_start"].dt.strftime("%Y-%m-%d")
                + " → "
                + d["fwd_window_end"].dt.strftime("%Y-%m-%d"),
                vix_close=lambda d: d["vix_close"].round(2),
                rv_fwd21_ann_vol_pct=lambda d: d["rv_fwd21_ann_vol_pct"].round(1),
                payoff_scaled=lambda d: d["payoff_scaled"].round(2),
                spx_fwd21_ret_log=lambda d: (100 * d["spx_fwd21_ret_log"]).round(1),
            )[
                [
                    "month",
                    "window",
                    "vix_close",
                    "rv_fwd21_ann_vol_pct",
                    "payoff_scaled",
                    "spx_fwd21_ret_log",
                ]
            ].rename(
                columns={
                    "vix_close": "VIX",
                    "rv_fwd21_ann_vol_pct": "realised vol %",
                    "spx_fwd21_ret_log": "S&P log return %",
                }
            ),
            floatfmt=".2f",
        ),
        "",
        "## Standing cautions for every slide",
        "",
        "1. State the sign (implied − realised) and the unit (scaled = fraction of implied "
        "variance; raw = annualised variance; spread = vol points) on every number.",
        "2. Sell-off results (Test 3) condition on what the S&P did **during** the window. "
        "They show *when* the seller loses; they are not a signal available at t.",
        "3. The payoff is an idealised variance swap. The VIX itself is not tradable; VIX "
        "futures and ETPs earn a different premium. No transaction costs, margin or "
        "position-sizing are modelled.",
        "4. VIX values before 22 Sept 2003 are back-calculated by Cboe. Quote the post-2003 "
        "result (Test 5) next to the full-sample one.",
        "5. A non-significant result is not proof of no effect (e.g. 'slope not different "
        "from 1' or a raw-variance mean that is not significant post-2003).",
        "6. Literature numbers (Carr & Wu, BTZ) come from other samples and other sign or "
        "unit conventions; see `docs/primer.md` before quoting them next to ours.",
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s (%d supported, %d not supported)", path.name, len(sup), len(uns))
    return path


def write_summary(res: StatsResults, claims: pd.DataFrame, path: Path) -> Path:
    """Write ``stats_summary.md``: method, every result table, and the figure list."""
    s = res.settings
    floor = 1 / (s.n_boot + 1)
    t = res.tables
    sec = {
        "1_mean": (
            "Test 1. Is the seller's payoff positive on average?",
            "H0: mean = 0. Three ways to get the standard error: the plain i.i.d. t-test, "
            f"Newey–West with {s.nw_lags_monthly} lags (allows autocorrelation and the small "
            "window overlap between months), and a circular block bootstrap. The median uses a "
            "sign test; the mean excluding the worst months is bootstrapped (the worst months "
            "are re-identified in every resample).",
            "07_test1_mean_ci",
        ),
        "2_shape": (
            "Test 2. How lopsided is the payoff?",
            "Win rate vs 50% (exact binomial; Wilson and bootstrap CIs). Skewness and excess "
            "kurtosis (bias-corrected sample moments) with bootstrap CIs; the D'Agostino and "
            "Anscombe–Glynn p-values assume independent months, so the bootstrap CI is the "
            "robustness check. Mean − median is a robust lopsidedness measure that does not "
            "depend on higher moments. The tail count compares months below mean − "
            f"{s.tail_sigma:g}·sd with what a normal distribution would give.",
            "08_test2_distribution",
        ),
        "3_crash": (
            "Test 3. Do losses cluster in S&P sell-offs?",
            "Conditions on the S&P 500's log return over the **same** 21-day window (an "
            "outcome). This describes when the seller loses; it is not a forecast made at t. "
            "Pearson correlation (Fisher-z and bootstrap CIs), sell-off vs other months (Welch "
            "t, Mann–Whitney, bootstrap), worst vs best S&P decile (both have large moves, so "
            "this isolates direction), and the share of all losses in the worst decile vs a "
            f"no-clustering null of {pct(s.loss_share_null)} (label-permutation test, "
            f"{s.n_perm:,} shuffles).",
            "09_test3_selloff",
        ),
        "4_forecast": (
            "Test 4. Is the VIX an unbiased forecast? (Mincer–Zarnowitz)",
            "Regress realised on implied: RV_{t→t+21} = a + b·IV_t + e. The only regressor is "
            "the VIX at the close of t (checked in code by `require_known_at_t`). An unbiased "
            "forecast has a = 0 and b = 1; we test that jointly with a Newey–West Wald test, in "
            "variance, in vol and in log variance (logs dampen the crash months).",
            "10_test4_mincer_zarnowitz",
        ),
        "5_robustness": (
            "Test 5. Does the result survive robustness checks?",
            "Mean payoff with Newey–West CIs across alternative samples and definitions. "
            "'p Holm' adjusts across the variants for each measure. Daily overlapping windows "
            f"share 20 of 21 returns, so they use {s.nw_lags_daily} Newey–West lags and no "
            "bootstrap.",
            "11_test5_robustness",
        ),
    }
    L = [
        "# v2 statistical tests: summary",
        "",
        "_Generated by `python -m vrp.pipeline --stage stats`. Do not edit by hand._",
        "",
        SIGN_NOTE,
        "",
        "Units: " + "; ".join(f"`{k}` = {v}" for k, v in UNITS.items()) + ".",
        "",
        f"Sample: {int(t.loc[t['test'] == '1_mean', 'n'].max())} months (1990-01 to "
        "2025-12), VIX at the close of the first trading day of each month vs realised "
        "variance from returns t+1 … t+21. All p-values are two-sided; all CIs are "
        f"{100 * (1 - s.alpha):.0f}%. Bootstrap: circular blocks of {s.block_length} months, "
        f"{s.n_boot:,} resamples, seed {s.boot_seed}. p-values shown as '≤ {floor:.1g}' are "
        "at the resampling floor.",
        "",
        f"**Pitch claims:** {int(claims['supported'].sum())} of {len(claims)} candidate claims "
        "are supported after a Holm correction; see `pitch_claims.md`.",
        "",
    ]
    for key, (title, method, fig) in sec.items():
        L += [
            f"## {title}",
            "",
            method,
            "",
            df_to_md(tidy(t[t["test"] == key], floor)),
            "",
            f"Figure: `outputs/figures/{fig}.png`",
            "",
        ]
        if key == "3_crash":
            for col, d in res.deciles.items():
                L += [
                    f"Mean `{col}` by S&P forward-return decile (1 = worst 10% of months; "
                    "i.i.d. t CIs, descriptive):",
                    "",
                    df_to_md(
                        d.assign(spx_fwd21_ret_mean=d["spx_fwd21_ret_mean"] * 100).rename(
                            columns={"spx_fwd21_ret_mean": "mean S&P log return %"}
                        )
                    ),
                    "",
                ]
    L += [
        "## Claims checked",
        "",
        df_to_md(
            claims.assign(
                p_holm=claims["p_holm"].map(lambda p: fmt_p(p, floor)),
                supported=claims["supported"].map({True: "✅", False: "❌"}),
            )[["id", "test", "supported", "p_holm", "claim"]]
        ),
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s", path.name)
    return path
