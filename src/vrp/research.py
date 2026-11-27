"""Research section: realised-vol dynamics, the premium by known-at-t conditions, time variation.

Tests are pre-registered in ``docs/research_plan.md`` (IDs R-1 … R-11). Every function returns
tidy rows in the same format as ``vrp.stats`` (``ROW_COLUMNS``), so the claims machinery in
``vrp.stats_report`` can check them.

A. Dynamics (``test_rv_dynamics``): calendar-month realised variance, a new non-overlapping
   series. AR(1) persistence, ADF mean reversion, linear trend, asymmetry of monthly changes.
B. Conditional premium (``test_conditional``): seller payoff by the VIX level and by the gap
   between the VIX and past realised vol, both known at the close of t, in fixed buckets.
C. Time variation (``test_time``, ``rolling_premium``): linear trend in the scaled premium and
   a rolling mean for the chart.
D. Step 5 (``test_breakeven``, ``test_jensen``, ``case_studies``): how often a crash the size
   of the worst month would have to occur to wipe out the premium, how much of "VIX above
   realised vol" is convexity rather than premium, and the two worst recent months day by day.

Sign convention: **implied − realised** (positive = variance seller gains).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from statsmodels.tsa.stattools import adfuller

from vrp.config import Config
from vrp.logging_setup import get_logger
from vrp.stats import (
    ROW_COLUMNS,
    Bootstrapper,
    StatsSettings,
    _masked_mean,
    mean_nw,
    require_known_at_t,
    row,
    s_skew,
)

log = get_logger(__name__)

CAL_SAMPLE = "calendar months 1990–2025"
MONTHLY_SAMPLE = "monthly 1990–2025"
PERIODS_PER_YEAR = 12  # monthly sample


# =========================================================================== settings
@dataclass(frozen=True)
class ResearchSettings:
    """Parameters from ``config.yaml`` → ``research:``."""

    ar_nw_lags: int
    trend_nw_lags: int
    adf_autolag: str
    vix_thresholds: tuple[float, float]
    gap_thresholds: tuple[float, float]
    rolling_window: int
    breakeven_thresholds: dict[str, tuple[float, ...]]
    case_studies: tuple[str, ...]
    case_pre_days: int
    case_big_move_sigma: float

    @classmethod
    def from_config(cls, cfg: Config) -> ResearchSettings:
        """Read the ``research:`` section."""
        r = cfg["research"]
        return cls(
            ar_nw_lags=int(r["ar_nw_lags"]),
            trend_nw_lags=int(r["trend_nw_lags"]),
            adf_autolag=str(r["adf_autolag"]),
            vix_thresholds=tuple(float(v) for v in r["vix_thresholds"]),  # type: ignore[arg-type]
            gap_thresholds=tuple(float(v) for v in r["gap_thresholds"]),  # type: ignore[arg-type]
            rolling_window=int(r["rolling_window_months"]),
            breakeven_thresholds={
                k: tuple(sorted(float(v) for v in vals))
                for k, vals in r["breakeven_thresholds"].items()
            },
            case_studies=tuple(str(m) for m in r["case_studies"]),
            case_pre_days=int(r["case_pre_days"]),
            case_big_move_sigma=float(r["case_big_move_sigma"]),
        )


# =========================================================================== A. dynamics
def calendar_month_rv(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Realised variance of each calendar month: (A / n_days) · Σ r² over its trading days.

    Non-overlapping, unlike the 21-day forward windows of the monthly sample. Indexed by the
    first day of the month; restricted to the configured sample months. Columns:
    ``n_days``, ``rv_cal_ann_var`` (annualised decimal), ``rv_cal_ann_vol_pct`` (vol points).
    """
    ann = cfg.annualisation
    r = panel["ret_log"].dropna()
    g = r.groupby(r.index.to_period("M"))
    out = pd.DataFrame({"n_days": g.size(), "sum_r2": g.apply(lambda x: float((x**2).sum()))})
    out["rv_cal_ann_var"] = ann / out["n_days"] * out["sum_r2"]
    out["rv_cal_ann_vol_pct"] = 100 * np.sqrt(out["rv_cal_ann_var"])
    d = cfg["dates"]
    lo, hi = pd.Period(d["sample_start_month"], "M"), pd.Period(d["sample_end_month"], "M")
    out = out[(out.index >= lo) & (out.index <= hi)]
    out.index = out.index.to_timestamp()
    out.index.name = "date"
    log.info("research: %d calendar months of realised variance", len(out))
    return out.drop(columns="sum_r2")


def rv_by_year(panel: pd.DataFrame, rvm: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per calendar year: realised vol (all daily returns) and vol-of-vol (std of monthly log RV).

    Vol-of-vol answers "which years were most irregular?". Ranked, 1 = most irregular.
    """
    ann = cfg.annualisation
    r = panel["ret_log"].dropna()
    years = rvm.index.year.unique()
    r = r[r.index.year.isin(years)]
    g = r.groupby(r.index.year)
    out = pd.DataFrame(
        {
            "n_days": g.size(),
            "rv_ann_vol_pct": 100 * np.sqrt(ann * g.apply(lambda x: float((x**2).mean()))),
        }
    )
    lv = np.log(rvm["rv_cal_ann_var"])
    out["vol_of_vol_log"] = lv.groupby(lv.index.year).std(ddof=1)
    vol_m = rvm["rv_cal_ann_vol_pct"]
    out["max_month_vol_pct"] = vol_m.groupby(vol_m.index.year).max()
    out["min_month_vol_pct"] = vol_m.groupby(vol_m.index.year).min()
    out["rank_irregular"] = out["vol_of_vol_log"].rank(ascending=False).astype(int)
    out.index.name = "year"
    return out


def ar1(x: np.ndarray, lags: int, alpha: float) -> dict[str, float]:
    """AR(1) coefficient of x (OLS of x_{m+1} on 1, x_m) with a Newey–West SE."""
    y, xl = x[1:], x[:-1]
    res = sm.OLS(y, sm.add_constant(xl)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    b, se = float(res.params[1]), float(res.bse[1])
    z = sps.norm.ppf(1 - alpha / 2)
    return {"rho": b, "se": se, "lo": b - z * se, "hi": b + z * se, "p": float(res.pvalues[1])}


def half_life(rho: float) -> float:
    """Months for a shock to halve under an AR(1) with coefficient rho (nan if not 0<rho<1)."""
    return float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else float("nan")


def time_trend(y: np.ndarray, t_years: np.ndarray, lags: int, alpha: float) -> dict[str, float]:
    """OLS slope of y on time (years) with a Newey–West SE."""
    res = sm.OLS(y, sm.add_constant(t_years)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    b, se = float(res.params[1]), float(res.bse[1])
    z = sps.norm.ppf(1 - alpha / 2)
    return {
        "slope": b,
        "se": se,
        "lo": b - z * se,
        "hi": b + z * se,
        "p": float(res.pvalues[1]),
        "intercept": float(res.params[0]),
    }


def years_since_start(idx: pd.DatetimeIndex) -> np.ndarray:
    """Time in years since the first date (365.25-day years)."""
    return np.asarray((idx - idx[0]).days / 365.25, dtype=float)


def test_rv_dynamics(
    rvm: pd.DataFrame, s: StatsSettings, rs: ResearchSettings, boot: Bootstrapper
) -> pd.DataFrame:
    """R-1 … R-4 on calendar-month log realised variance."""
    lv = np.log(rvm["rv_cal_ann_var"].to_numpy(float))
    n = len(lv)
    rows: list[dict[str, object]] = []
    common = {"test": "R_dynamics", "measure": "log_rv_cal", "sample": CAL_SAMPLE}

    a = ar1(lv, rs.ar_nw_lags, s.alpha)
    rows.append(
        row(
            **common,
            statistic="ar1_rho",
            estimate=a["rho"],
            ci_low=a["lo"],
            ci_high=a["hi"],
            ci_method=f"Newey–West ({rs.ar_nw_lags} lags)",
            null_value=0.0,
            test_name=f"HAC z, z = {a['rho'] / a['se']:.2f}",
            p_value=a["p"],
            n=n - 1,
            note=f"half-life {half_life(a['rho']):.2f} months",
        )
    )
    rows.append(row(**common, statistic="half_life_months", estimate=half_life(a["rho"]), n=n - 1))
    for k in (1, 3, 6, 12):
        rows.append(
            row(
                **common,
                statistic=f"autocorr_lag_{k}",
                estimate=float(pd.Series(lv).autocorr(k)),
                n=n - k,
            )
        )

    stat, p, used, nobs, crit, _ = adfuller(
        lv, regression="c", autolag=rs.adf_autolag, result_object=False
    )
    rows.append(
        row(
            **common,
            statistic="adf_stat",
            estimate=float(stat),
            test_name=f"ADF (constant, {used} lags by {rs.adf_autolag}), H0: unit root",
            p_value=float(p),
            n=int(nobs),
            note="critical values " + ", ".join(f"{k}: {v:.2f}" for k, v in crit.items()),
        )
    )

    t_years = years_since_start(rvm.index)
    tr = time_trend(lv, t_years, rs.trend_nw_lags, s.alpha)
    rows.append(
        row(
            **common,
            statistic="trend_slope_per_year",
            estimate=tr["slope"],
            ci_low=tr["lo"],
            ci_high=tr["hi"],
            ci_method=f"Newey–West ({rs.trend_nw_lags} lags)",
            null_value=0.0,
            test_name=f"HAC z, z = {tr['slope'] / tr['se']:.2f}",
            p_value=tr["p"],
            n=n,
            note=f"≈ {100 * (np.exp(10 * tr['slope'] / 2) - 1):+.1f}% realised vol per decade "
            "(half the log-variance slope)",
        )
    )

    d = np.diff(lv)
    bt = boot.run(s_skew, d, alpha=s.alpha)
    rows.append(
        row(
            test="R_dynamics",
            measure="dlog_rv_cal",
            sample=CAL_SAMPLE,
            statistic="skewness",
            estimate=bt["estimate"],
            ci_low=bt["ci_low"],
            ci_high=bt["ci_high"],
            ci_method=f"block bootstrap ({boot.block}-month blocks)",
            null_value=0.0,
            test_name="bootstrap",
            p_value=bt["p"],
            n=len(d),
            note=f"D'Agostino (i.i.d.) p = {sps.skewtest(d).pvalue:.3g}",
        )
    )
    vol = rvm["rv_cal_ann_vol_pct"]
    for stat_name, val in [
        ("sd", float(d.std(ddof=1))),
        ("largest_rise", float(d.max())),
        ("largest_fall", float(d.min())),
    ]:
        rows.append(
            row(
                test="R_dynamics",
                measure="dlog_rv_cal",
                sample=CAL_SAMPLE,
                statistic=stat_name,
                estimate=val,
                n=len(d),
                note=(
                    f"into {vol.index[1:][np.argmax(d)]:%Y-%m}"
                    if stat_name == "largest_rise"
                    else (
                        f"into {vol.index[1:][np.argmin(d)]:%Y-%m}"
                        if stat_name == "largest_fall"
                        else "std of month-to-month change in log realised variance"
                    )
                ),
            )
        )
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


# =========================================================================== B. conditional
CONDITIONING = ["vix_close", "rv_past21_ann_var"]


def gap_pts(monthly: pd.DataFrame) -> pd.Series:
    """VIX minus past 21-day realised vol at t (vol points). Known at the close of t."""
    require_known_at_t(CONDITIONING)
    return monthly["vix_close"] - 100 * np.sqrt(monthly["rv_past21_ann_var"])


def bucket(x: pd.Series, thresholds: tuple[float, float], labels: list[str]) -> pd.Series:
    """Three buckets: x ≤ t1, t1 < x ≤ t2, x > t2 (NaN stays NaN)."""
    t1, t2 = thresholds
    return pd.cut(x, [-np.inf, t1, t2, np.inf], labels=labels, right=True)


def bucket_labels(name: str, thresholds: tuple[float, float], unit: str) -> list[str]:
    """Readable labels such as 'VIX ≤ 15', '15 < VIX ≤ 25', 'VIX > 25'."""
    t1, t2 = (f"{t:g}" for t in thresholds)
    return [f"{name} ≤ {t1}{unit}", f"{t1} < {name} ≤ {t2}{unit}", f"{name} > {t2}{unit}"]


def conditioning_buckets(monthly: pd.DataFrame, rs: ResearchSettings) -> pd.DataFrame:
    """The two known-at-t bucket columns: ``vix_bucket`` and ``gap_bucket``."""
    require_known_at_t(CONDITIONING)
    out = pd.DataFrame(index=monthly.index)
    out["gap_pts"] = gap_pts(monthly)
    out["vix_bucket"] = bucket(
        monthly["vix_close"], rs.vix_thresholds, bucket_labels("VIX", rs.vix_thresholds, "")
    )
    out["gap_bucket"] = bucket(
        out["gap_pts"], rs.gap_thresholds, bucket_labels("gap", rs.gap_thresholds, " pts")
    )
    return out


def s_diff_two(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """mean(x | a) − mean(x | b) per row; a and b are boolean masks."""
    return _masked_mean(x, a.astype(bool)) - _masked_mean(x, b.astype(bool))


def test_conditional(
    monthly: pd.DataFrame, s: StatsSettings, rs: ResearchSettings, boot: Bootstrapper
) -> pd.DataFrame:
    """R-5 … R-8: seller payoff by VIX level and by VIX − past RV gap (both known at t)."""
    b = conditioning_buckets(monthly, rs)
    rows: list[dict[str, object]] = []
    for var in ("vix_bucket", "gap_bucket"):
        keep = b[var].notna()
        labels = list(b[var].cat.categories)
        for col in s.shape_measures:
            x_all = monthly.loc[keep, col].to_numpy(float)
            lab = b.loc[keep, var].astype(str).to_numpy()
            for lb in labels:
                x = x_all[lab == lb]
                nw = mean_nw(x, s.nw_lags_monthly, s.alpha)
                rows.append(
                    row(
                        test="R_conditional",
                        measure=col,
                        sample=lb,
                        statistic="mean",
                        estimate=nw["mean"],
                        ci_low=nw["ci_low"],
                        ci_high=nw["ci_high"],
                        ci_method=f"Newey–West ({s.nw_lags_monthly} lags, within bucket)",
                        null_value=0.0,
                        test_name=f"HAC t, t = {nw['t']:.2f}",
                        p_value=nw["p"],
                        n=nw["n"],
                        note=f"median {np.median(x):.4g}; win rate {(x > 0).mean():.3f}",
                    )
                )
                rows.append(
                    row(
                        test="R_conditional",
                        measure=col,
                        sample=lb,
                        statistic="win_rate",
                        estimate=float((x > 0).mean()),
                        n=len(x),
                    )
                )
            hi_m, lo_m = lab == labels[2], lab == labels[0]
            bt = boot.run(s_diff_two, x_all, hi_m, lo_m, alpha=s.alpha)
            welch = sps.ttest_ind(x_all[hi_m], x_all[lo_m], equal_var=False)
            rows.append(
                row(
                    test="R_conditional",
                    measure=col,
                    sample=f"{labels[2]} minus {labels[0]}",
                    statistic="diff_mean_high_minus_low",
                    estimate=bt["estimate"],
                    ci_low=bt["ci_low"],
                    ci_high=bt["ci_high"],
                    ci_method=f"block bootstrap ({boot.block}-month blocks)",
                    null_value=0.0,
                    test_name="bootstrap",
                    p_value=bt["p"],
                    n=int(hi_m.sum() + lo_m.sum()),
                    note=f"Welch t = {welch.statistic:.2f}, p = {welch.pvalue:.3g}",
                )
            )
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


# =========================================================================== C. time
def test_time(monthly: pd.DataFrame, s: StatsSettings) -> pd.DataFrame:
    """R-9: linear time trend in the payoff (scaled is the pre-registered test)."""
    t_years = years_since_start(pd.DatetimeIndex(monthly.index))
    rows = []
    for col in s.shape_measures:
        tr = time_trend(monthly[col].to_numpy(float), t_years, s.nw_lags_monthly, s.alpha)
        rows.append(
            row(
                test="R_time",
                measure=col,
                sample=MONTHLY_SAMPLE,
                statistic="trend_slope_per_year",
                estimate=tr["slope"],
                ci_low=tr["lo"],
                ci_high=tr["hi"],
                ci_method=f"Newey–West ({s.nw_lags_monthly} lags)",
                null_value=0.0,
                test_name=f"HAC z, z = {tr['slope'] / tr['se']:.2f}",
                p_value=tr["p"],
                n=len(monthly),
                note=f"fitted value 1990: {tr['intercept']:.4g}; "
                f"2025: {tr['intercept'] + tr['slope'] * t_years[-1]:.4g}",
            )
        )
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def rolling_premium(
    monthly: pd.DataFrame, col: str, window: int, lags: int, alpha: float
) -> pd.DataFrame:
    """Rolling mean of ``col`` over ``window`` months with a Newey–West band (chart only).

    Indexed by the window's last month. Overlapping windows: the band is not a test.
    """
    x = monthly[col].to_numpy(float)
    out = []
    for i in range(window - 1, len(x)):
        nw = mean_nw(x[i - window + 1 : i + 1], lags, alpha)
        out.append((monthly.index[i], nw["mean"], nw["ci_low"], nw["ci_high"]))
    df = pd.DataFrame(out, columns=["date", "mean", "ci_low", "ci_high"]).set_index("date")
    df["measure"] = col
    df["window_months"] = window
    return df


# =========================================================================== D. break-even
def poisson_rate_ci(k: int, exposure: float, alpha: float) -> tuple[float, float]:
    """Exact (Garwood) CI for a Poisson rate: ``k`` events over ``exposure`` units of time."""
    lo = 0.0 if k == 0 else float(sps.chi2.ppf(alpha / 2, 2 * k)) / 2
    hi = float(sps.chi2.ppf(1 - alpha / 2, 2 * k + 2)) / 2
    return lo / exposure, hi / exposure


def breakeven_share(ordinary_mean: np.ndarray | float, crash_size: float) -> np.ndarray | float:
    """Share f of months that must be crashes for (1 − f)·g + f·L = 0, i.e. f = g / (g − L).

    ``ordinary_mean`` g is the mean payoff of non-crash months, ``crash_size`` L < 0 the payoff
    of a crash month. f ≤ 0 means ordinary months already lose on average.
    """
    if crash_size >= 0:
        raise ValueError(f"crash size must be negative (a loss), got {crash_size}")
    return ordinary_mean / (ordinary_mean - crash_size)


def make_s_breakeven_rate(
    threshold: float, crash_size: float, periods_per_year: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Vectorised break-even crash rate (per year) for the block bootstrap.

    Ordinary months are those above ``threshold``; their mean g is re-estimated on each
    resample while the crash size L is held fixed (it is the scenario, not an estimate).
    """

    def f(x: np.ndarray) -> np.ndarray:
        g = _masked_mean(x, x > threshold)
        return periods_per_year * breakeven_share(g, crash_size)

    return f


def _breakeven_one(
    x: np.ndarray, threshold: float, s: StatsSettings, boot: Bootstrapper, periods: int
) -> dict[str, float]:
    """Break-even vs observed crash frequency for one threshold (see ``test_breakeven``)."""
    crash = x <= threshold
    k = int(crash.sum())
    L = float(x[crash].mean())
    g = float(x[~crash].mean())
    years = len(x) / periods
    bt = boot.run(make_s_breakeven_rate(threshold, L, periods), x, alpha=s.alpha)
    gb = boot.run(lambda a: _masked_mean(a, a > threshold), x, alpha=s.alpha)
    obs = k / years
    obs_lo, obs_hi = poisson_rate_ci(k, years, s.alpha)
    # Conservative p: P(N ≤ k) if crashes arrived at the LOW end of the break-even rate's CI.
    p = float(sps.poisson.cdf(k, max(bt["ci_low"], 0.0) * years))
    return {
        "threshold": threshold,
        "n_crash": k,
        "crash_mean": L,
        "ordinary_mean": g,
        "ordinary_mean_lo": gb["ci_low"],
        "ordinary_mean_hi": gb["ci_high"],
        "breakeven_rate": bt["estimate"],
        "breakeven_rate_lo": bt["ci_low"],
        "breakeven_rate_hi": bt["ci_high"],
        "observed_rate": obs,
        "observed_rate_lo": obs_lo,
        "observed_rate_hi": obs_hi,
        "years": years,
        "p_conservative": p,
        "n": len(x),
    }


def breakeven_curve(
    monthly: pd.DataFrame, s: StatsSettings, rs: ResearchSettings, boot: Bootstrapper
) -> pd.DataFrame:
    """Break-even vs observed crash frequency for each configured threshold and the worst month.

    One row per (measure, threshold); ``is_worst`` marks the headline scenario (the single worst
    month). Rates are per year. Chart and table only; the claims use the worst-month rows.
    """
    rows = []
    for col in s.shape_measures:
        xs = monthly[col].dropna()
        x = xs.to_numpy(float)
        worst = float(x.min())
        ths = [t for t in rs.breakeven_thresholds.get(col, ()) if t > worst] + [worst]
        for th in sorted(set(ths)):
            r = _breakeven_one(x, th, s, boot, PERIODS_PER_YEAR)
            r.update(measure=col, is_worst=th == worst, worst_month=f"{xs.idxmin():%Y-%m}")
            rows.append(r)
    out = pd.DataFrame(rows)
    out["ratio_breakeven_to_observed"] = out["breakeven_rate"] / out["observed_rate"]
    lead = ["measure", "threshold", "is_worst"]
    return out[lead + [c for c in out.columns if c not in lead]]


def test_breakeven(monthly: pd.DataFrame, curve: pd.DataFrame, s: StatsSettings) -> pd.DataFrame:
    """R-10 / R-11: years between worst-month-sized crashes needed to break even vs observed."""
    rows: list[dict[str, object]] = []
    for col in s.shape_measures:
        c = curve[(curve["measure"] == col) & curve["is_worst"]].iloc[0]
        worst_month = str(c["worst_month"])
        k, years = int(c["n_crash"]), float(c["years"])
        common = {"test": "R_breakeven", "measure": col, "sample": MONTHLY_SAMPLE}
        rows.append(
            row(
                **common,
                statistic="crash_size",
                estimate=c["crash_mean"],
                n=k,
                note=f"worst month {worst_month} (the scenario; held fixed)",
            )
        )
        rows.append(
            row(
                **common,
                statistic="ordinary_mean",
                estimate=c["ordinary_mean"],
                ci_low=c["ordinary_mean_lo"],
                ci_high=c["ordinary_mean_hi"],
                ci_method=f"block bootstrap ({s.block_length}-month blocks)",
                n=int(c["n"]) - k,
                note="mean payoff of every other month",
            )
        )
        rows.append(
            row(
                **common,
                statistic="breakeven_rate_per_year",
                estimate=c["breakeven_rate"],
                ci_low=c["breakeven_rate_lo"],
                ci_high=c["breakeven_rate_hi"],
                ci_method=f"block bootstrap ({s.block_length}-month blocks)",
                n=int(c["n"]),
            )
        )
        rows.append(
            row(
                **common,
                statistic="observed_rate_per_year",
                estimate=c["observed_rate"],
                ci_low=c["observed_rate_lo"],
                ci_high=c["observed_rate_hi"],
                ci_method="exact Poisson (Garwood)",
                n=int(c["n"]),
                note=f"{k} crash(es) of this size in {years:.0f} years",
            )
        )
        lo_rate, hi_rate = c["breakeven_rate_lo"], c["breakeven_rate_hi"]
        rows.append(
            row(
                **common,
                statistic="breakeven_years_between",
                estimate=1 / c["breakeven_rate"],
                ci_low=1 / hi_rate if hi_rate > 0 else np.nan,
                ci_high=1 / lo_rate if lo_rate > 0 else np.inf,
                ci_method=f"block bootstrap ({s.block_length}-month blocks)",
                null_value=years / k,
                test_name="one-sided exact Poisson P(N ≤ observed) at the lower CI bound "
                "of the break-even rate (conservative)",
                p_value=c["p_conservative"],
                n=int(c["n"]),
                note=f"observed: {k} in {years:.0f} years (once every {years / k:.0f} years; "
                f"95% CI once every {1 / c['observed_rate_hi']:.1f} to "
                f"{1 / c['observed_rate_lo']:.0f} years)",
            )
        )
        rows.append(
            row(
                **common,
                statistic="breakeven_crashes_in_sample",
                estimate=c["breakeven_rate"] * years,
                n=int(c["n"]),
                note=f"crashes of this size the sample would have needed (observed {k})",
            )
        )
        rows.append(
            row(
                **common,
                statistic="months_of_premium_per_crash",
                estimate=-c["crash_mean"] / c["ordinary_mean"],
                n=int(c["n"]),
                note="one crash wipes out this many ordinary months' average payoff",
            )
        )
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


# =========================================================================== D. Jensen gap
JENSEN_PREDICTORS = ["iv_ann_var", "rv_past21_ann_var"]


def batched_ols_resid(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals for each row: ``y`` (B, n), ``X`` (B, n, k) → residuals (B, n)."""
    xtx = np.einsum("bnk,bnj->bkj", X, X)
    xty = np.einsum("bnk,bn->bk", X, y)
    beta = np.linalg.solve(xtx, xty[..., None])[..., 0]
    return y - np.einsum("bnk,bk->bn", X, beta)


def _jensen_resid(ly: np.ndarray, *lx: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """Residuals of log forward RV on a constant and the log predictors, row by row."""
    out = np.empty_like(ly, dtype=float)
    for i in range(0, len(ly), chunk):
        sl = slice(i, i + chunk)
        X = np.stack([np.ones_like(ly[sl])] + [x[sl] for x in lx], axis=-1)
        out[sl] = batched_ols_resid(ly[sl], X)
    return out


def smearing_ratio(e: np.ndarray) -> np.ndarray:
    """E[√RV] / √E[RV] implied by log-scale residuals e (Duan smearing): rows of a 2-D array."""
    return np.mean(np.exp(e / 2), axis=1) / np.sqrt(np.mean(np.exp(e), axis=1))


def s_convexity_pts(ly, l_iv, l_rvp, vix, rvol) -> np.ndarray:  # noqa: ANN001
    """Convexity part of mean(VIX − realised vol), vol points, if VIX² were a fair forecast."""
    rho = smearing_ratio(_jensen_resid(ly, l_iv, l_rvp))
    return vix.mean(axis=1) * (1 - rho)


def s_convexity_share(ly, l_iv, l_rvp, vix, rvol) -> np.ndarray:  # noqa: ANN001
    """Convexity part as a share of the mean spread VIX − realised vol."""
    return s_convexity_pts(ly, l_iv, l_rvp, vix, rvol) / (vix - rvol).mean(axis=1)


def s_fair_vix_exceeds(ly, l_iv, l_rvp, vix, rvol) -> np.ndarray:  # noqa: ANN001
    """Share of months a fair VIX (VIX² = E_t[RV]) would still exceed realised vol."""
    e = _jensen_resid(ly, l_iv, l_rvp)
    return np.mean(e < np.log(np.mean(np.exp(e), axis=1, keepdims=True)), axis=1)


def s_uncond_gap_pts(rv: np.ndarray) -> np.ndarray:
    """√E[RV] − E[√RV] in vol points (rows of annualised variances)."""
    return 100 * (np.sqrt(rv.mean(axis=1)) - np.sqrt(rv).mean(axis=1))


def s_vix_exceeds(vix: np.ndarray, rvol: np.ndarray) -> np.ndarray:
    """Share of months the VIX exceeded realised vol."""
    return np.mean(vix > rvol, axis=1)


def s_mean_spread(vix: np.ndarray, rvol: np.ndarray) -> np.ndarray:
    """Mean of VIX − realised vol (vol points)."""
    return (vix - rvol).mean(axis=1)


def jensen_inputs(monthly: pd.DataFrame) -> dict[str, np.ndarray]:
    """Arrays for the Jensen statistics: months with forward RV and both predictors."""
    require_known_at_t(JENSEN_PREDICTORS)
    cols = ["rv_fwd21_ann_var", "rv_fwd21_ann_vol_pct", "vix_close"] + JENSEN_PREDICTORS
    d = monthly[cols].dropna()
    return {
        "ly": np.log(d["rv_fwd21_ann_var"].to_numpy(float)),
        "l_iv": np.log(d["iv_ann_var"].to_numpy(float)),
        "l_rvp": np.log(d["rv_past21_ann_var"].to_numpy(float)),
        "vix": d["vix_close"].to_numpy(float),
        "rvol": d["rv_fwd21_ann_vol_pct"].to_numpy(float),
        "rv": d["rv_fwd21_ann_var"].to_numpy(float),
    }


def test_jensen(monthly: pd.DataFrame, s: StatsSettings, boot: Bootstrapper) -> pd.DataFrame:
    """Jensen gap: how much of "VIX above realised vol" is convexity, not premium (descriptive)."""
    a = jensen_inputs(monthly)
    reg = (a["ly"], a["l_iv"], a["l_rvp"], a["vix"], a["rvol"])
    e = _jensen_resid(*[v[None, :] for v in reg[:3]])[0]
    sig2 = float(e.var(ddof=3))
    r2 = 1 - float(e.var()) / float(a["ly"].var())
    meth = f"block bootstrap ({boot.block}-month blocks)"
    common = {"test": "R_jensen", "sample": MONTHLY_SAMPLE, "ci_method": meth}
    rows: list[dict[str, object]] = []

    def add(measure: str, statistic: str, stat, arrays, note: str = "") -> None:  # noqa: ANN001
        bt = boot.run(stat, *arrays, alpha=s.alpha)
        rows.append(
            row(
                **common,
                measure=measure,
                statistic=statistic,
                estimate=bt["estimate"],
                ci_low=bt["ci_low"],
                ci_high=bt["ci_high"],
                n=bt["n"],
                note=note,
            )
        )

    vr = (a["vix"], a["rvol"])
    add("vol_spread_pts", "mean", s_mean_spread, vr, "mean VIX − realised vol (vol points)")
    add("vol_spread_pts", "share_vix_above_realised", s_vix_exceeds, vr)
    add(
        "rv_fwd21",
        "uncond_jensen_gap_pts",
        s_uncond_gap_pts,
        (a["rv"],),
        "√E[RV] − E[√RV]; an upper bound (also counts vol moving between months)",
    )
    add(
        "rv_fwd21",
        "cond_convexity_pts",
        s_convexity_pts,
        reg,
        f"mean(VIX)·(1 − E_t√RV/√E_tRV); log-RV forecast R² {r2:.2f}, residual var {sig2:.3f}",
    )
    add("rv_fwd21", "cond_convexity_share_of_spread", s_convexity_share, reg)
    add(
        "rv_fwd21",
        "fair_vix_share_above_realised",
        s_fair_vix_exceeds,
        reg,
        "share of months a VIX with zero premium (VIX² = E_t[RV]) would still beat realised vol",
    )
    rho = float(smearing_ratio(e[None, :])[0])
    rows.append(
        row(
            **{**common, "ci_method": ""},
            measure="rv_fwd21",
            statistic="cond_vol_ratio",
            estimate=rho,
            n=len(e),
            note=f"E_t[√RV] / √E_t[RV] by smearing; lognormal check exp(−σ²/8) = "
            f"{np.exp(-sig2 / 8):.4f}",
        )
    )
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


# =========================================================================== D. case studies
CASE_PATH_COLUMNS = [
    "case",
    "day",
    "spx_close",
    "spx_rel",
    "vix_close",
    "ret_log",
    "accrued_rv_ann_var",
    "accrued_over_implied",
]


def case_path(panel: pd.DataFrame, t: pd.Timestamp, cfg: Config, pre_days: int) -> pd.DataFrame:
    """Daily path around month start t: ``pre_days`` before t through the forward window.

    ``day`` is trading days from t (0 = t). For days 1…H, ``accrued_rv_ann_var`` is the realised
    variance accrued so far on the swap's scale, (A / H)·Σ r², so on day H it equals
    ``rv_fwd21_ann_var``; ``accrued_over_implied`` divides it by the implied variance sold at t.
    Outcome data (after t): a description of what happened, never a predictor.
    """
    h, ann = cfg.window, cfg.annualisation
    i = panel.index.get_loc(t)
    w = panel.iloc[max(i - pre_days, 0) : i + h + 1]
    day = np.arange(len(w)) - min(pre_days, i)
    r = w["ret_log"].where(day > 0, 0.0)
    acc = (ann / h) * (r**2).cumsum()
    iv = float(panel["iv_ann_var"].iloc[i])
    out = pd.DataFrame(
        {
            "day": day,
            "spx_close": w["spx_close"].to_numpy(),
            "spx_rel": (w["spx_close"] / panel["spx_close"].iloc[i]).to_numpy(),
            "vix_close": w["vix_close"].to_numpy(),
            "ret_log": w["ret_log"].to_numpy(),
            "accrued_rv_ann_var": np.where(day >= 0, acc, np.nan),
            "accrued_over_implied": np.where(day >= 0, acc / iv, np.nan),
        },
        index=w.index,
    )
    return out


def months_to_recover(x: pd.Series, start: pd.Timestamp) -> float:
    """Months after ``start`` until the cumulative payoff since (and including) it is ≥ 0.

    0 = never below zero (not a loss); nan = not recovered by the end of the sample.
    """
    cum = x.loc[start:].cumsum().to_numpy()
    hit = np.flatnonzero(cum >= 0)
    return float(hit[0]) if len(hit) else float("nan")


def case_studies(
    panel: pd.DataFrame, monthly: pd.DataFrame, rs: ResearchSettings, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive rows (test ``R_case``) and daily paths for each configured month."""
    rows: list[dict[str, object]] = []
    paths = []
    xs = monthly["payoff_scaled"]
    for m in rs.case_studies:
        hit = monthly[monthly.index.to_period("M") == pd.Period(m, "M")]
        if len(hit) != 1:
            raise KeyError(f"case study {m}: {len(hit)} sample months")
        t = hit.index[0]
        mr = hit.iloc[0]
        p = case_path(panel, t, cfg, rs.case_pre_days)
        p.insert(0, "case", m)
        paths.append(p)
        win = p[p["day"] > 0]
        if len(win) != cfg.window or not np.isclose(
            win["accrued_rv_ann_var"].iloc[-1], mr["rv_fwd21_ann_var"], rtol=1e-9
        ):
            raise ValueError(f"case study {m}: path does not reproduce rv_fwd21_ann_var")
        sd_day = np.sqrt(mr["iv_ann_var"] / cfg.annualisation)
        big = int((win["ret_log"].abs() > rs.case_big_move_sigma * sd_day).sum())
        breach = win.loc[win["accrued_over_implied"] >= 1, "day"]
        imax = win["ret_log"].abs().idxmax()
        vmax = win["vix_close"].idxmax()
        others = xs.drop(t)
        vals: list[tuple[str, float, str]] = [
            ("vix_at_t", mr["vix_close"], f"t = {t:%Y-%m-%d}"),
            (
                "vix_at_t_percentile",
                float((monthly["vix_close"] < mr["vix_close"]).mean()),
                "share of sample months with a lower VIX at t (descriptive, full sample)",
            ),
            ("past_rv_vol_at_t", 100 * np.sqrt(mr["rv_past21_ann_var"]), "vol points"),
            ("realised_vol_fwd", mr["rv_fwd21_ann_vol_pct"], "vol points"),
            ("payoff_var", mr["payoff_var"], ""),
            ("payoff_scaled", mr["payoff_scaled"], ""),
            (
                "payoff_scaled_rank",
                float((xs < mr["payoff_scaled"]).sum() + 1),
                f"1 = worst of {len(xs)} months",
            ),
            ("spx_return_window", float(np.expm1(mr["spx_fwd21_ret_log"])), "simple return"),
            ("spx_max_drawdown_window", float(win["spx_rel"].min() - 1), "vs the close at t"),
            ("vix_max_window", float(win["vix_close"].max()), f"on {vmax:%Y-%m-%d}"),
            (
                "largest_daily_move",
                float(win.loc[imax, "ret_log"]),
                f"log return on {imax:%Y-%m-%d}",
            ),
            (
                "big_move_days",
                float(big),
                f"days with an absolute return above {rs.case_big_move_sigma:g}× the "
                "VIX-implied daily s.d. "
                f"({100 * sd_day:.2f}%), of {cfg.window}",
            ),
            (
                "premium_gone_by_day",
                float(breach.iloc[0]) if len(breach) else float("nan"),
                "first trading day the accrued realised variance exceeded the implied variance",
            ),
            (
                "months_of_premium_lost",
                float(-mr["payoff_scaled"] / others.mean()),
                "loss ÷ mean scaled payoff of all other months",
            ),
            (
                "months_to_recover",
                months_to_recover(xs, t),
                "until cumulative scaled payoff from this month is ≥ 0 (nan = not by sample end)",
            ),
        ]
        for stat_name, val, note in vals:
            rows.append(
                row(
                    test="R_case",
                    measure="case_study",
                    sample=m,
                    statistic=stat_name,
                    estimate=float(val),
                    n=1,
                    note=note,
                )
            )
    return pd.DataFrame(rows, columns=ROW_COLUMNS), pd.concat(paths)[CASE_PATH_COLUMNS]


# =========================================================================== runner
@dataclass
class ResearchResults:
    """Everything the research report, claims and figures need."""

    tables: pd.DataFrame
    rv_monthly: pd.DataFrame
    rv_year: pd.DataFrame
    rolling: pd.DataFrame
    buckets: pd.DataFrame
    breakeven: pd.DataFrame
    case_paths: pd.DataFrame
    settings: StatsSettings
    research: ResearchSettings


def run_research(panel: pd.DataFrame, monthly: pd.DataFrame, cfg: Config) -> ResearchResults:
    """Run sections A–D of the research plan."""
    s = StatsSettings.from_config(cfg)
    rs = ResearchSettings.from_config(cfg)
    boot = Bootstrapper(s.block_length, s.n_boot, s.boot_seed)
    rvm = calendar_month_rv(panel, cfg)
    yr = rv_by_year(panel, rvm, cfg)
    ta = test_rv_dynamics(rvm, s, rs, boot)
    log.info("research: A (dynamics) %d rows", len(ta))
    tb = test_conditional(monthly, s, rs, boot)
    log.info("research: B (conditional) %d rows", len(tb))
    tc = test_time(monthly, s)
    log.info("research: C (time) %d rows", len(tc))
    roll = rolling_premium(monthly, "payoff_scaled", rs.rolling_window, s.nw_lags_monthly, s.alpha)
    curve = breakeven_curve(monthly, s, rs, boot)
    td = [test_breakeven(monthly, curve, s), test_jensen(monthly, s, boot)]
    tcase, paths = case_studies(panel, monthly, rs, cfg)
    log.info("research: D (break-even, Jensen, cases) %d rows", sum(map(len, td)) + len(tcase))
    tables = pd.concat([ta, tb, tc, *td, tcase], ignore_index=True)
    return ResearchResults(
        tables, rvm, yr, roll, conditioning_buckets(monthly, rs), curve, paths, s, rs
    )
