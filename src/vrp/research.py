"""Research section: realised-vol dynamics, the premium by known-at-t conditions, time variation.

Tests are pre-registered in ``docs/research_plan.md`` (IDs R-1 … R-9). Every function returns
tidy rows in the same format as ``vrp.stats`` (``ROW_COLUMNS``), so the claims machinery in
``vrp.stats_report`` can check them.

A. Dynamics (``test_rv_dynamics``): calendar-month realised variance, a new non-overlapping
   series. AR(1) persistence, ADF mean reversion, linear trend, asymmetry of monthly changes.
B. Conditional premium (``test_conditional``): seller payoff by the VIX level and by the gap
   between the VIX and past realised vol, both known at the close of t, in fixed buckets.
C. Time variation (``test_time``, ``rolling_premium``): linear trend in the scaled premium and
   a rolling mean for the chart.

Sign convention: **implied − realised** (positive = variance seller gains).
"""

from __future__ import annotations

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


# =========================================================================== runner
@dataclass
class ResearchResults:
    """Everything the research report, claims and figures need."""

    tables: pd.DataFrame
    rv_monthly: pd.DataFrame
    rv_year: pd.DataFrame
    rolling: pd.DataFrame
    buckets: pd.DataFrame
    settings: StatsSettings
    research: ResearchSettings


def run_research(panel: pd.DataFrame, monthly: pd.DataFrame, cfg: Config) -> ResearchResults:
    """Run sections A–C of the research plan."""
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
    tables = pd.concat([ta, tb, tc], ignore_index=True)
    return ResearchResults(tables, rvm, yr, roll, conditioning_buckets(monthly, rs), s, rs)
