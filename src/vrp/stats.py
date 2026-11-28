"""v2 statistical tests of the variance risk premium.

Five tests, each returning tidy rows (one row per statistic) with a point estimate,
a (1 − alpha) confidence interval, the test used and its two-sided p-value:

1. ``test_mean``         Is the seller's payoff positive on average? (i.i.d. t, Newey–West,
                         block bootstrap; also the median and the mean without the worst N.)
2. ``test_shape``        How lopsided is it? (win rate, skewness, excess kurtosis, normality,
                         average gain vs average loss, tail counts.)
3. ``test_crash``        Do losses cluster in S&P sell-offs? (correlation with the forward S&P
                         return, sell-off vs other months, worst vs best decile, loss share.)
4. ``test_forecast``     Is the VIX a biased forecast of realised variance? (Mincer–Zarnowitz
                         regression RV = a + b·IV; H0: a = 0, b = 1.)
5. ``test_robustness``   Does the result survive post-2003, H = 22, demeaned RV, VXO, daily
                         overlapping windows and dropping the worst months?

Sign convention everywhere: **implied − realised** (positive = variance seller gains).

No look-ahead: the only *predictor* used anywhere is implied variance at the close of t
(Test 4, checked by ``require_known_at_t``). Tests 1–3 and 5 describe outcomes; Test 3
conditions **on** outcomes (``is_selloff``, ``selloff_decile``, ``spx_fwd21_ret_log``) to
describe when losses happen, never as something forecastable at t.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from vrp import features
from vrp.config import Config
from vrp.logging_setup import get_logger

log = get_logger(__name__)

# Columns that use data after t (see docs/data_dictionary.md). Never predictors.
OUTCOME_COLUMNS: frozenset[str] = frozenset(
    {
        "rv_fwd21_ann_var",
        "rv_fwd21_ann_vol_pct",
        "payoff_var",
        "payoff_scaled",
        "vol_spread_pts",
        "spx_fwd21_ret_log",
        "is_selloff",
        "selloff_decile",
        "rv_fwd22_ann_var",
        "rv_fwd21_dm_ann_var",
        "payoff_var_h22",
        "payoff_scaled_h22",
        "payoff_var_dm",
        "payoff_scaled_dm",
        "payoff_var_vxo",
        "payoff_scaled_vxo",
    }
)

UNITS: dict[str, str] = {
    "payoff_var": "annualised variance (decimal)",
    "payoff_scaled": "fraction of implied variance (1 − RV/IV)",
    "vol_spread_pts": "vol points (VIX − realised vol)",
}

ROW_COLUMNS: list[str] = [
    "test",
    "measure",
    "sample",
    "statistic",
    "estimate",
    "ci_low",
    "ci_high",
    "ci_method",
    "null_value",
    "test_name",
    "p_value",
    "n",
    "note",
]


def require_known_at_t(columns: list[str]) -> None:
    """Raise if any column is an outcome (uses data after t). Guards every predictor."""
    bad = [c for c in columns if c in OUTCOME_COLUMNS or "fwd" in c]
    if bad:
        raise ValueError(f"look-ahead: outcome column(s) used as predictor: {bad}")


# =========================================================================== settings
@dataclass(frozen=True)
class StatsSettings:
    """Parameters from ``config.yaml`` → ``stats:``."""

    alpha: float
    nw_lags_monthly: int
    nw_lags_daily: int
    n_exclude_worst: int
    tail_sigma: float
    win_rate_null: float
    loss_share_null: float
    worst_decile: int
    best_decile: int
    measures: list[str]
    shape_measures: list[str]
    robustness_window: int
    block_length: int
    n_boot: int
    boot_seed: int
    n_perm: int
    perm_seed: int

    @classmethod
    def from_config(cls, cfg: Config) -> StatsSettings:
        """Read the ``stats:`` section."""
        s = cfg["stats"]
        return cls(
            alpha=float(s["alpha"]),
            nw_lags_monthly=int(s["nw_lags_monthly"]),
            nw_lags_daily=int(s["nw_lags_daily"]),
            n_exclude_worst=int(s["n_exclude_worst"]),
            tail_sigma=float(s["tail_sigma"]),
            win_rate_null=float(s["win_rate_null"]),
            loss_share_null=float(s["loss_share_null_decile"]),
            worst_decile=int(s["worst_decile"]),
            best_decile=int(s["best_decile"]),
            measures=list(s["measures"]),
            shape_measures=list(s["shape_measures"]),
            robustness_window=int(s["robustness_window"]),
            block_length=int(s["bootstrap"]["block_length"]),
            n_boot=int(s["bootstrap"]["n_resamples"]),
            boot_seed=int(s["bootstrap"]["seed"]),
            n_perm=int(s["permutation"]["n_resamples"]),
            perm_seed=int(s["permutation"]["seed"]),
        )


# =========================================================================== primitives
def row(**kw: object) -> dict[str, object]:
    """One result row with every standard column present."""
    out = {c: np.nan for c in ROW_COLUMNS}
    out.update({"ci_method": "", "test_name": "", "note": ""})
    out.update(kw)
    return out


def _clean(x: pd.Series | np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


def mean_iid(x: pd.Series | np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Plain one-sample t-test of mean = 0 (assumes independent observations)."""
    a = _clean(x)
    n = len(a)
    m = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n))
    t = m / se
    q = sps.t.ppf(1 - alpha / 2, n - 1)
    return {
        "mean": m,
        "se": se,
        "t": t,
        "p": float(2 * sps.t.sf(abs(t), n - 1)),
        "ci_low": m - q * se,
        "ci_high": m + q * se,
        "n": n,
    }


def mean_nw(x: pd.Series | np.ndarray, lags: int, alpha: float = 0.05) -> dict[str, float]:
    """Mean with a Newey–West (HAC, Bartlett kernel) standard error.

    OLS of x on a constant with ``cov_type="HAC"``; statsmodels' default small-sample
    correction n/(n−1) is applied. Inference uses the normal distribution.
    """
    a = _clean(x)
    res = sm.OLS(a, np.ones(len(a))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    m, se = float(res.params[0]), float(res.bse[0])
    z = sps.norm.ppf(1 - alpha / 2)
    return {
        "mean": m,
        "se": se,
        "t": m / se,
        "p": float(2 * sps.norm.sf(abs(m / se))),
        "ci_low": m - z * se,
        "ci_high": m + z * se,
        "n": len(a),
        "lags": lags,
    }


def newey_west_se_by_hand(x: np.ndarray, lags: int) -> float:
    """Reference Newey–West SE of the mean (Bartlett weights, n/(n−1) correction).

    Used by the tests to check ``mean_nw`` against an independent calculation.
    """
    a = np.asarray(x, dtype=float)
    n = len(a)
    u = a - a.mean()
    s = u @ u / n
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        s += 2 * w * (u[j:] @ u[:-j]) / n
    return float(np.sqrt(s / n * n / (n - 1)))


def block_bootstrap_indices(
    n: int, block: int, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Circular block bootstrap: (n_resamples, n) row indices made of contiguous blocks.

    Each resample glues together random blocks of ``block`` consecutive months
    (wrapping round the end), which keeps short-run autocorrelation intact.
    """
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_resamples, -1)[:, :n]


@dataclass
class Bootstrapper:
    """Caches one set of block-bootstrap indices per sample size (common random numbers)."""

    block: int
    n_resamples: int
    seed: int
    _cache: dict[int, np.ndarray] = field(default_factory=dict)

    def indices(self, n: int) -> np.ndarray:
        """Indices for a sample of size ``n`` (same seed every time → reproducible)."""
        if n not in self._cache:
            rng = np.random.default_rng(self.seed + n)
            self._cache[n] = block_bootstrap_indices(n, self.block, self.n_resamples, rng)
        return self._cache[n]

    def run(
        self,
        stat: Callable[..., np.ndarray],
        *arrays: np.ndarray,
        null: float = 0.0,
        alpha: float = 0.05,
    ) -> dict[str, float]:
        """Percentile CI and a two-sided bootstrap p-value for ``stat``.

        ``stat`` takes 2-D arrays (resamples × n) and returns one value per resample.
        p-value: share of centred resamples at least as far from the estimate as the
        estimate is from ``null``, i.e. P*(|θ* − θ̂| ≥ |θ̂ − θ0|), with a +1 correction.
        """
        arrs = [np.asarray(a) for a in arrays]
        n = len(arrs[0])
        est = float(stat(*[a[None, :] for a in arrs])[0])
        idx = self.indices(n)
        b = np.asarray(stat(*[a[idx] for a in arrs]), dtype=float)
        b = b[~np.isnan(b)]
        lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
        p = (1 + np.sum(np.abs(b - est) >= abs(est - null))) / (len(b) + 1)
        return {
            "estimate": est,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "p": float(min(p, 1.0)),
            "n": n,
            "se": float(b.std(ddof=1)),
            "n_valid": int(len(b)),
        }


# ---- vectorised statistics (operate along axis 1 of a (B, n) array) ----------------
def s_mean(x: np.ndarray) -> np.ndarray:
    """Row means."""
    return x.mean(axis=1)


def s_median(x: np.ndarray) -> np.ndarray:
    """Row medians."""
    return np.median(x, axis=1)


def s_mean_minus_median(x: np.ndarray) -> np.ndarray:
    """Mean − median per row: negative when a long left tail drags the mean down."""
    return x.mean(axis=1) - np.median(x, axis=1)


def s_win_rate(x: np.ndarray) -> np.ndarray:
    """Share of strictly positive values per row."""
    return (x > 0).mean(axis=1)


def s_skew(x: np.ndarray) -> np.ndarray:
    """Adjusted Fisher–Pearson sample skewness (``bias=False``)."""
    return sps.skew(x, axis=1, bias=False)


def s_exkurt(x: np.ndarray) -> np.ndarray:
    """Sample excess kurtosis (``bias=False``; normal = 0)."""
    return sps.kurtosis(x, axis=1, fisher=True, bias=False)


def s_sharpe_annual(x: np.ndarray) -> np.ndarray:
    """Monthly mean / sd × √12 (no risk-free rate: a variance swap costs nothing upfront)."""
    return x.mean(axis=1) / x.std(axis=1, ddof=1) * np.sqrt(12)


def make_s_mean_excl_worst(k: int) -> Callable[[np.ndarray], np.ndarray]:
    """Mean after dropping the k smallest values in each row."""

    def f(x: np.ndarray) -> np.ndarray:
        return np.sort(x, axis=1)[:, k:].mean(axis=1)

    return f


def _masked_mean(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    cnt = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (x * m).sum(axis=1) / cnt, np.nan)


def s_avg_gain(x: np.ndarray) -> np.ndarray:
    """Mean of the positive values in each row."""
    return _masked_mean(x, x > 0)


def s_avg_loss(x: np.ndarray) -> np.ndarray:
    """Mean of the negative values in each row (a negative number)."""
    return _masked_mean(x, x < 0)


def s_loss_gain_ratio(x: np.ndarray) -> np.ndarray:
    """|average loss| / average gain."""
    return -s_avg_loss(x) / s_avg_gain(x)


def s_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation."""
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    return (xc * yc).sum(axis=1) / np.sqrt((xc**2).sum(axis=1) * (yc**2).sum(axis=1))


def s_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Spearman correlation (Pearson on ranks)."""
    rx = sps.rankdata(x, axis=1)
    ry = sps.rankdata(y, axis=1)
    return s_corr(rx, ry)


def s_diff_means(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """mean(x | g) − mean(x | not g) per row; g is boolean."""
    g = g.astype(bool)
    return _masked_mean(x, g) - _masked_mean(x, ~g)


def s_loss_share(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Share of the total loss (sum of negative x) that falls in rows where g is True."""
    loss = np.minimum(x, 0.0)
    tot = loss.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot < 0, (loss * g.astype(bool)).sum(axis=1) / tot, np.nan)


def permutation_p(
    stat: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    g: np.ndarray,
    null: float,
    n_resamples: int,
    seed: int,
) -> dict[str, float]:
    """Two-sided permutation test: shuffle the labels ``g`` against ``x``.

    Returns the observed statistic, the permutation mean (≈ the null) and
    p = P(|θ_perm − null| ≥ |θ_obs − null|) with a +1 correction.
    """
    rng = np.random.default_rng(seed)
    obs = float(stat(x[None, :], g[None, :])[0])
    perm = np.argsort(rng.random((n_resamples, len(g))), axis=1)
    sims = stat(np.broadcast_to(x, perm.shape), g[perm])
    sims = sims[~np.isnan(sims)]
    p = (1 + np.sum(np.abs(sims - null) >= abs(obs - null))) / (len(sims) + 1)
    return {"estimate": obs, "perm_mean": float(sims.mean()), "p": float(p)}


def holm(pvalues: list[float], alpha: float) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (controls the family-wise error rate)."""
    return multipletests(pvalues, alpha=alpha, method="holm")[1]


# =========================================================================== variants
def build_monthly_variants(panel: pd.DataFrame, monthly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Robustness payoffs on the monthly dates: H = 22, demeaned RV, VXO as implied.

    New columns only; the v1 definitions are untouched. Every realised measure uses
    returns strictly after t (t+1 … t+H); VXO is the close of t.
    """
    s = StatsSettings.from_config(cfg)
    ann, w, h = cfg.annualisation, cfg.window, s.robustness_window
    r = panel["ret_log"]
    daily = pd.DataFrame(index=panel.index)
    daily[f"rv_fwd{h}_ann_var"] = features.forward_realised_variance(r, h, ann).where(
        features.forward_window_complete(r, h)
    )
    daily[f"rv_fwd{w}_dm_ann_var"] = features.forward_realised_variance_demeaned(r, w, ann).where(
        features.forward_window_complete(r, w)
    )
    daily["iv_vxo_ann_var"] = features.implied_variance(panel["vxo_close"])

    keep = ["month", "post_2003", "iv_ann_var", "rv_fwd21_ann_var", "payoff_var", "payoff_scaled"]
    out = monthly[keep].join(daily, how="left")
    iv = out["iv_ann_var"]
    out[f"payoff_var_h{h}"] = iv - out[f"rv_fwd{h}_ann_var"]
    out[f"payoff_scaled_h{h}"] = out[f"payoff_var_h{h}"] / iv
    out["payoff_var_dm"] = iv - out[f"rv_fwd{w}_dm_ann_var"]
    out["payoff_scaled_dm"] = out["payoff_var_dm"] / iv
    out["payoff_var_vxo"] = out["iv_vxo_ann_var"] - out["rv_fwd21_ann_var"]
    out["payoff_scaled_vxo"] = out["payoff_var_vxo"] / out["iv_vxo_ann_var"]
    log.info(
        "variants: %d months; H=%d complete %d; demeaned complete %d; VXO available %d",
        len(out),
        h,
        int(out[f"rv_fwd{h}_ann_var"].notna().sum()),
        int(out[f"rv_fwd{w}_dm_ann_var"].notna().sum()),
        int(out["iv_vxo_ann_var"].notna().sum()),
    )
    return out


def daily_sample(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Every daily row in the sample window with a complete forward window (overlapping)."""
    d = cfg["dates"]
    start = pd.Period(d["sample_start_month"], "M").start_time
    end = pd.Period(d["sample_end_month"], "M").end_time
    sel = panel[(panel.index >= start) & (panel.index <= end) & panel["fwd_window_complete"]]
    return sel


# =========================================================================== Test 1
def test_mean(monthly: pd.DataFrame, s: StatsSettings, boot: Bootstrapper) -> pd.DataFrame:
    """Test 1: is the mean seller payoff different from zero? (H0: mean = 0)."""
    rows: list[dict[str, object]] = []
    k = s.n_exclude_worst
    for col in s.measures:
        x = _clean(monthly[col])
        base = {"test": "1_mean", "measure": col, "sample": "monthly 1990–2025", "null_value": 0.0}
        a = mean_iid(x, s.alpha)
        rows.append(
            row(
                **base,
                statistic="mean",
                estimate=a["mean"],
                ci_low=a["ci_low"],
                ci_high=a["ci_high"],
                ci_method="t (i.i.d.)",
                test_name=f"one-sample t, t = {a['t']:.2f}",
                p_value=a["p"],
                n=a["n"],
            )
        )
        b = mean_nw(x, s.nw_lags_monthly, s.alpha)
        rows.append(
            row(
                **base,
                statistic="mean",
                estimate=b["mean"],
                ci_low=b["ci_low"],
                ci_high=b["ci_high"],
                ci_method=f"Newey–West ({s.nw_lags_monthly} lags)",
                test_name=f"HAC t, t = {b['t']:.2f}",
                p_value=b["p"],
                n=b["n"],
            )
        )
        c = boot.run(s_mean, x, alpha=s.alpha)
        rows.append(
            row(
                **base,
                statistic="mean",
                estimate=c["estimate"],
                ci_low=c["ci_low"],
                ci_high=c["ci_high"],
                ci_method=f"block bootstrap ({boot.block}-month blocks)",
                test_name="bootstrap",
                p_value=c["p"],
                n=c["n"],
            )
        )
        med = boot.run(s_median, x, alpha=s.alpha)
        n_pos, n_nz = int((x > 0).sum()), int((x != 0).sum())
        rows.append(
            row(
                **base,
                statistic="median",
                estimate=med["estimate"],
                ci_low=med["ci_low"],
                ci_high=med["ci_high"],
                ci_method=f"block bootstrap ({boot.block}-month blocks)",
                test_name=f"sign test ({n_pos}/{n_nz} positive)",
                p_value=float(sps.binomtest(n_pos, n_nz, 0.5).pvalue),
                n=len(x),
                note="sign test assumes independent months",
            )
        )
        ex = boot.run(make_s_mean_excl_worst(k), x, alpha=s.alpha)
        rows.append(
            row(
                **base,
                statistic=f"mean_excl_worst_{k}",
                estimate=ex["estimate"],
                ci_low=ex["ci_low"],
                ci_high=ex["ci_high"],
                ci_method=f"block bootstrap ({boot.block}-month blocks)",
                test_name="bootstrap",
                p_value=ex["p"],
                n=len(x) - k,
                note=f"drops the {k} lowest months in each resample",
            )
        )
        rows.append(
            row(
                **{**base, "null_value": np.nan},
                statistic=f"mean_change_from_dropping_worst_{k}_pct",
                estimate=100 * (ex["estimate"] / a["mean"] - 1),
                n=len(x),
                note="descriptive: % change in the mean when the worst months are removed",
            )
        )
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


# =========================================================================== Test 2
def test_shape(monthly: pd.DataFrame, s: StatsSettings, boot: Bootstrapper) -> pd.DataFrame:
    """Test 2: how lopsided is the payoff? (win rate, skew, kurtosis, gains vs losses)."""
    rows: list[dict[str, object]] = []
    bm = f"block bootstrap ({boot.block}-month blocks)"
    for col in s.shape_measures:
        ser = monthly[col].dropna()
        x = ser.to_numpy(dtype=float)
        n = len(x)
        base = {"test": "2_shape", "measure": col, "sample": "monthly 1990–2025"}

        wins = int((x > 0).sum())
        lo, hi = proportion_confint(wins, n, alpha=s.alpha, method="wilson")
        bt = sps.binomtest(wins, n, s.win_rate_null)
        rows.append(
            row(
                **base,
                statistic="win_rate",
                estimate=wins / n,
                ci_low=lo,
                ci_high=hi,
                ci_method="Wilson (i.i.d.)",
                null_value=s.win_rate_null,
                test_name=f"exact binomial ({wins}/{n})",
                p_value=bt.pvalue,
                n=n,
            )
        )
        wb = boot.run(s_win_rate, x, null=s.win_rate_null, alpha=s.alpha)
        rows.append(
            row(
                **base,
                statistic="win_rate",
                estimate=wb["estimate"],
                ci_low=wb["ci_low"],
                ci_high=wb["ci_high"],
                ci_method=bm,
                null_value=s.win_rate_null,
                test_name="bootstrap",
                p_value=wb["p"],
                n=n,
            )
        )

        sk = boot.run(s_skew, x, alpha=s.alpha)
        rows.append(
            row(
                **base,
                statistic="skewness",
                estimate=sk["estimate"],
                ci_low=sk["ci_low"],
                ci_high=sk["ci_high"],
                ci_method=bm,
                null_value=0.0,
                test_name="D'Agostino skewness test (i.i.d.)",
                p_value=float(sps.skewtest(x).pvalue),
                n=n,
                note=f"bootstrap p = {sk['p']:.4g}",
            )
        )
        ku = boot.run(s_exkurt, x, alpha=s.alpha)
        rows.append(
            row(
                **base,
                statistic="excess_kurtosis",
                estimate=ku["estimate"],
                ci_low=ku["ci_low"],
                ci_high=ku["ci_high"],
                ci_method=bm,
                null_value=0.0,
                test_name="Anscombe–Glynn kurtosis test (i.i.d.)",
                p_value=float(sps.kurtosistest(x).pvalue),
                n=n,
                note=f"bootstrap p = {ku['p']:.4g}",
            )
        )
        jb = sps.jarque_bera(x)
        rows.append(
            row(
                **base,
                statistic="jarque_bera",
                estimate=float(jb.statistic),
                null_value=np.nan,
                test_name="Jarque–Bera normality (H0: normal)",
                p_value=float(jb.pvalue),
                n=n,
            )
        )

        for name, fn in [
            ("avg_gain", s_avg_gain),
            ("avg_loss", s_avg_loss),
        ]:
            r_ = boot.run(fn, x, alpha=s.alpha)
            rows.append(
                row(
                    **base,
                    statistic=name,
                    estimate=r_["estimate"],
                    ci_low=r_["ci_low"],
                    ci_high=r_["ci_high"],
                    ci_method=bm,
                    n=int((x > 0).sum() if name == "avg_gain" else (x < 0).sum()),
                )
            )
        lg = boot.run(s_loss_gain_ratio, x, null=1.0, alpha=s.alpha)
        rows.append(
            row(
                **base,
                statistic="loss_to_gain_ratio",
                estimate=lg["estimate"],
                ci_low=lg["ci_low"],
                ci_high=lg["ci_high"],
                ci_method=bm,
                null_value=1.0,
                test_name="bootstrap (H0: |avg loss| = avg gain)",
                p_value=lg["p"],
                n=n,
            )
        )
        mm = boot.run(s_mean_minus_median, x, alpha=s.alpha)
        rows.append(
            row(
                **base,
                statistic="mean_minus_median",
                estimate=mm["estimate"],
                ci_low=mm["ci_low"],
                ci_high=mm["ci_high"],
                ci_method=bm,
                null_value=0.0,
                test_name="bootstrap (H0: mean = median)",
                p_value=mm["p"],
                n=n,
                note="robust lopsidedness check: < 0 means a left tail pulls the mean down",
            )
        )
        worst_i = int(np.argmin(x))
        avg_gain = float(x[x > 0].mean())
        rows.append(
            row(
                **base,
                statistic="worst_month",
                estimate=float(x[worst_i]),
                n=n,
                note=f"{ser.index[worst_i].strftime('%Y-%m')}; descriptive (no CI: an extreme "
                "order statistic cannot be bootstrapped)",
            )
        )
        rows.append(
            row(
                **base,
                statistic="worst_over_avg_gain",
                estimate=float(x[worst_i] / avg_gain),
                n=n,
                note="descriptive: worst month in units of the average winning month",
            )
        )
        k = s.n_exclude_worst
        loss_tot = np.minimum(x, 0).sum()
        rows.append(
            row(
                **base,
                statistic=f"share_of_losses_in_worst_{k}",
                estimate=float(np.sort(x)[:k].sum() / loss_tot),
                n=int((x < 0).sum()),
                note=f"descriptive: worst {k} of {int((x < 0).sum())} losing months",
            )
        )
        rows.append(
            row(
                **base,
                statistic=f"worst_{k}_as_share_of_total_gains",
                estimate=float(-np.sort(x)[:k].sum() / x[x > 0].sum()),
                n=n,
                note=f"descriptive: the worst {k} months' losses ÷ the sum of all winning months",
            )
        )
        # Tail count vs a normal distribution with the same mean and sd.
        mu, sd = x.mean(), x.std(ddof=1)
        thr = mu - s.tail_sigma * sd
        k_obs = int((x < thr).sum())
        p_norm = float(sps.norm.cdf(-s.tail_sigma))
        lo, hi = proportion_confint(k_obs, n, alpha=s.alpha, method="wilson")
        rows.append(
            row(
                **base,
                statistic=f"months_below_mean_minus_{s.tail_sigma:g}sd",
                estimate=k_obs,
                ci_low=lo * n,
                ci_high=hi * n,
                ci_method="Wilson (i.i.d.), in months",
                null_value=p_norm * n,
                test_name="exact binomial vs normal tail probability",
                p_value=float(sps.binomtest(k_obs, n, p_norm).pvalue),
                n=n,
                note=f"normal expects {p_norm * n:.2f} months",
            )
        )
        if col == "payoff_scaled":
            sh = boot.run(s_sharpe_annual, x, alpha=s.alpha)
            rows.append(
                row(
                    **base,
                    statistic="sharpe_annualised",
                    estimate=sh["estimate"],
                    ci_low=sh["ci_low"],
                    ci_high=sh["ci_high"],
                    ci_method=bm,
                    null_value=0.0,
                    test_name="bootstrap",
                    p_value=sh["p"],
                    n=n,
                    note="ignores skew and crash risk (primer); idealised variance swap, "
                    "no costs",
                )
            )
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def decile_table(monthly: pd.DataFrame, col: str, alpha: float) -> pd.DataFrame:
    """Mean payoff per S&P forward-return decile with i.i.d. t CIs (descriptive)."""
    out = []
    for d, g in monthly.groupby("selloff_decile"):
        a = mean_iid(g[col], alpha)
        out.append(
            {
                "selloff_decile": int(d),
                "n": a["n"],
                "mean": a["mean"],
                "ci_low": a["ci_low"],
                "ci_high": a["ci_high"],
                "win_rate": float((g[col] > 0).mean()),
                "spx_fwd21_ret_mean": float(g["spx_fwd21_ret_log"].mean()),
            }
        )
    return pd.DataFrame(out)


# =========================================================================== Test 3
def test_crash(
    monthly: pd.DataFrame, s: StatsSettings, boot: Bootstrapper
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Test 3: do seller losses cluster in S&P sell-offs?

    Conditions on outcomes (the S&P return over the same 21 days) to *describe* when
    losses occur. Nothing here is a forecast made at t.
    """
    rows: list[dict[str, object]] = []
    deciles: dict[str, pd.DataFrame] = {}
    bm = f"block bootstrap ({boot.block}-month blocks)"
    spx = monthly["spx_fwd21_ret_log"].to_numpy(dtype=float)
    sell = monthly["is_selloff"].to_numpy(dtype=bool)
    dec = monthly["selloff_decile"].to_numpy(dtype=int)
    worst = dec == s.worst_decile
    best = dec == s.best_decile
    for col in s.shape_measures:
        x = monthly[col].to_numpy(dtype=float)
        n = len(x)
        base = {"test": "3_crash", "measure": col, "sample": "monthly 1990–2025"}

        pr = sps.pearsonr(x, spx)
        ci = pr.confidence_interval(confidence_level=1 - s.alpha)
        rows.append(
            row(
                **base,
                statistic="corr_with_spx_fwd21_ret",
                estimate=float(pr.statistic),
                ci_low=float(ci.low),
                ci_high=float(ci.high),
                ci_method="Fisher z (i.i.d.)",
                null_value=0.0,
                test_name="Pearson",
                p_value=float(pr.pvalue),
                n=n,
            )
        )
        cb = boot.run(s_corr, x, spx, alpha=s.alpha)
        rows.append(
            row(
                **base,
                statistic="corr_with_spx_fwd21_ret",
                estimate=cb["estimate"],
                ci_low=cb["ci_low"],
                ci_high=cb["ci_high"],
                ci_method=bm,
                null_value=0.0,
                test_name="bootstrap",
                p_value=cb["p"],
                n=n,
            )
        )
        sp = sps.spearmanr(x, spx)
        sb = boot.run(s_spearman, x, spx, alpha=s.alpha)
        rows.append(
            row(
                **base,
                statistic="spearman_with_spx_fwd21_ret",
                estimate=float(sp.statistic),
                ci_low=sb["ci_low"],
                ci_high=sb["ci_high"],
                ci_method=bm,
                null_value=0.0,
                test_name="Spearman rank",
                p_value=float(sp.pvalue),
                n=n,
                note=f"bootstrap p = {sb['p']:.4g}",
            )
        )

        groups = [("selloff", sell)]
        same = bool(np.array_equal(sell, worst))
        if not same:
            groups.append((f"decile_{s.worst_decile}", worst))
        same_note = (
            f"is_selloff months are identical to decile {s.worst_decile} in this sample"
            if same
            else ""
        )
        for label, g in groups:
            m_in, m_out = float(x[g].mean()), float(x[~g].mean())
            rows.append(
                row(
                    **base,
                    statistic=f"mean_in_{label}",
                    estimate=m_in,
                    n=int(g.sum()),
                    note=same_note,
                    **_t_ci(x[g], s.alpha),
                )
            )
            rows.append(
                row(
                    **base,
                    statistic=f"mean_outside_{label}",
                    estimate=m_out,
                    n=int((~g).sum()),
                    **_t_ci(x[~g], s.alpha),
                )
            )
            welch = sps.ttest_ind(x[g], x[~g], equal_var=False)
            db = boot.run(s_diff_means, x, g, alpha=s.alpha)
            rows.append(
                row(
                    **base,
                    statistic=f"diff_mean_{label}_minus_other",
                    estimate=m_in - m_out,
                    ci_low=db["ci_low"],
                    ci_high=db["ci_high"],
                    ci_method=bm,
                    null_value=0.0,
                    test_name=f"Welch t, t = {welch.statistic:.2f}",
                    p_value=float(welch.pvalue),
                    n=n,
                    note=f"bootstrap p = {db['p']:.4g}; Mann–Whitney p = "
                    f"{sps.mannwhitneyu(x[g], x[~g]).pvalue:.3g}",
                )
            )
            loss_in, loss_out = int((x[g] < 0).sum()), int((x[~g] < 0).sum())
            fe = sps.fisher_exact(
                [[loss_in, int(g.sum()) - loss_in], [loss_out, int((~g).sum()) - loss_out]]
            )
            rows.append(
                row(
                    **base,
                    statistic=f"loss_rate_in_{label}",
                    estimate=loss_in / g.sum(),
                    ci_low=proportion_confint(loss_in, g.sum(), s.alpha, "wilson")[0],
                    ci_high=proportion_confint(loss_in, g.sum(), s.alpha, "wilson")[1],
                    ci_method="Wilson (i.i.d.)",
                    null_value=loss_out / (~g).sum(),
                    test_name=f"Fisher exact vs other months ({loss_out}/{int((~g).sum())} "
                    "lose outside)",
                    p_value=float(fe.pvalue),
                    n=int(g.sum()),
                    note=f"{loss_in}/{int(g.sum())} months lose",
                )
            )

        # Worst vs best decile: both have big |moves|, so this isolates the direction.
        sub = worst | best
        welch = sps.ttest_ind(x[worst], x[best], equal_var=False)
        diff = float(x[worst].mean() - x[best].mean())
        db = boot.run(s_diff_means, x[sub], worst[sub], alpha=s.alpha)
        rows.append(
            row(
                **base,
                statistic=f"diff_mean_decile_{s.worst_decile}_minus_decile_{s.best_decile}",
                estimate=diff,
                ci_low=db["ci_low"],
                ci_high=db["ci_high"],
                ci_method="bootstrap over the 2 deciles' months",
                null_value=0.0,
                test_name=f"Welch t, t = {welch.statistic:.2f}",
                p_value=float(welch.pvalue),
                n=int(sub.sum()),
                note=f"bootstrap p = {db['p']:.4g}; mean decile {s.best_decile} = "
                f"{x[best].mean():.4g}",
            )
        )

        share_b = boot.run(s_loss_share, x, worst, null=s.loss_share_null, alpha=s.alpha)
        perm = permutation_p(s_loss_share, x, worst, s.loss_share_null, s.n_perm, s.perm_seed)
        rows.append(
            row(
                **base,
                statistic=f"loss_share_in_decile_{s.worst_decile}",
                estimate=share_b["estimate"],
                ci_low=share_b["ci_low"],
                ci_high=share_b["ci_high"],
                ci_method=bm,
                null_value=s.loss_share_null,
                test_name=f"permutation ({s.n_perm:,} shuffles; null mean "
                f"{perm['perm_mean']:.3f})",
                p_value=perm["p"],
                n=n,
                note="share of the total of all negative payoffs; "
                f"bootstrap p = {share_b['p']:.4g}",
            )
        )
        deciles[col] = decile_table(monthly, col, s.alpha)
    return pd.DataFrame(rows, columns=ROW_COLUMNS), deciles


def _t_ci(x: np.ndarray, alpha: float) -> dict[str, object]:
    """i.i.d. t CI and two-sided p-value (H0: mean = 0) for a subset of months."""
    a = mean_iid(x, alpha)
    return {
        "ci_low": a["ci_low"],
        "ci_high": a["ci_high"],
        "ci_method": "t (i.i.d.)",
        "null_value": 0.0,
        "test_name": f"one-sample t, t = {a['t']:.2f}",
        "p_value": a["p"],
    }


# =========================================================================== Test 4
def mincer_zarnowitz(
    y: pd.Series, x: pd.Series, lags: int, alpha: float
) -> dict[str, float | object]:
    """OLS y = a + b·x with Newey–West SEs; Wald test of a = 0 and b = 1 jointly.

    ``x`` must be known at the close of t (checked by name).
    """
    require_known_at_t([str(x.name)])
    d = pd.concat([y, x], axis=1).dropna()
    X = sm.add_constant(d.iloc[:, 1].to_numpy(dtype=float))
    res = sm.OLS(d.iloc[:, 0].to_numpy(dtype=float), X).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    ci = res.conf_int(alpha=alpha)
    a, b = res.params
    se_a, se_b = res.bse
    w = res.wald_test((np.eye(2), np.array([0.0, 1.0])), scalar=True, use_f=False)
    return {
        "a": float(a),
        "b": float(b),
        "a_ci": (float(ci[0, 0]), float(ci[0, 1])),
        "b_ci": (float(ci[1, 0]), float(ci[1, 1])),
        "p_a0": float(2 * sps.norm.sf(abs(a / se_a))),
        "p_b1": float(2 * sps.norm.sf(abs((b - 1) / se_b))),
        "wald": float(w.statistic),
        "p_wald": float(w.pvalue),
        "r2": float(res.rsquared),
        "n": int(res.nobs),
        "res": res,
    }


def test_forecast(monthly: pd.DataFrame, s: StatsSettings) -> pd.DataFrame:
    """Test 4: Mincer–Zarnowitz forecast-bias regressions (predictor = VIX at close of t)."""
    specs = [
        ("variance", monthly["rv_fwd21_ann_var"], monthly["iv_ann_var"]),
        ("vol_pct", monthly["rv_fwd21_ann_vol_pct"], monthly["vix_close"]),
        (
            "log_variance",
            np.log(monthly["rv_fwd21_ann_var"]).rename("log_rv_fwd21"),
            np.log(monthly["iv_ann_var"]).rename("log_iv_ann_var"),
        ),
    ]
    rows: list[dict[str, object]] = []
    ci_m = f"Newey–West ({s.nw_lags_monthly} lags)"
    for name, y, x in specs:
        r = mincer_zarnowitz(y, x, s.nw_lags_monthly, s.alpha)
        base = {"test": "4_forecast", "measure": name, "sample": "monthly 1990–2025", "n": r["n"]}
        rows.append(
            row(
                **base,
                statistic="intercept_a",
                estimate=r["a"],
                ci_low=r["a_ci"][0],
                ci_high=r["a_ci"][1],
                ci_method=ci_m,
                null_value=0.0,
                test_name="HAC z (H0: a = 0)",
                p_value=r["p_a0"],
            )
        )
        rows.append(
            row(
                **base,
                statistic="slope_b",
                estimate=r["b"],
                ci_low=r["b_ci"][0],
                ci_high=r["b_ci"][1],
                ci_method=ci_m,
                null_value=1.0,
                test_name="HAC z (H0: b = 1)",
                p_value=r["p_b1"],
            )
        )
        rows.append(
            row(
                **base,
                statistic="joint_a0_b1",
                estimate=r["wald"],
                null_value=np.nan,
                test_name="HAC Wald χ²(2) (H0: unbiased forecast, a = 0 and b = 1)",
                p_value=r["p_wald"],
                note=f"R² = {r['r2']:.3f}",
            )
        )
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def mz_fit_line(monthly: pd.DataFrame, s: StatsSettings) -> dict[str, object]:
    """Vol-space MZ fit plus its NW confidence band on a grid (for the figure)."""
    r = mincer_zarnowitz(
        monthly["rv_fwd21_ann_vol_pct"], monthly["vix_close"], s.nw_lags_monthly, s.alpha
    )
    grid = np.linspace(monthly["vix_close"].min(), monthly["vix_close"].max(), 100)
    pred = r["res"].get_prediction(sm.add_constant(grid)).summary_frame(alpha=s.alpha)
    return {
        "grid": grid,
        "fit": pred["mean"].to_numpy(),
        "lo": pred["mean_ci_lower"].to_numpy(),
        "hi": pred["mean_ci_upper"].to_numpy(),
        "a": r["a"],
        "b": r["b"],
        "p_wald": r["p_wald"],
        "r2": r["r2"],
    }


# =========================================================================== Test 5
def test_robustness(
    monthly: pd.DataFrame,
    variants: pd.DataFrame,
    daily: pd.DataFrame,
    s: StatsSettings,
    boot: Bootstrapper,
) -> pd.DataFrame:
    """Test 5: mean payoff (H0: mean = 0) across alternative samples and definitions."""
    h = s.robustness_window
    k = s.n_exclude_worst
    vxo_rows = variants["iv_vxo_ann_var"].notna()
    post = monthly["post_2003"].astype(bool)
    rows: list[dict[str, object]] = []
    for col in s.shape_measures:
        suffix = col.replace("payoff_", "")  # "var" or "scaled"
        worst_idx = monthly[col].nsmallest(k).index
        cases: list[tuple[str, pd.Series, int]] = [
            ("baseline (H = 21, monthly 1990–2025)", monthly[col], s.nw_lags_monthly),
            ("post-2003 only (live VIX)", monthly.loc[post, col], s.nw_lags_monthly),
            ("pre-2003 only (back-calculated VIX)", monthly.loc[~post, col], s.nw_lags_monthly),
            (f"H = {h} trading days", variants[f"payoff_{suffix}_h{h}"], s.nw_lags_monthly),
            ("demeaned realised variance", variants[f"payoff_{suffix}_dm"], s.nw_lags_monthly),
            ("VXO instead of VIX (1990–2021)", variants[f"payoff_{suffix}_vxo"], s.nw_lags_monthly),
            ("VIX, same months as VXO", monthly.loc[vxo_rows, col], s.nw_lags_monthly),
            (f"excluding worst {k} months", monthly[col].drop(worst_idx), s.nw_lags_monthly),
            ("daily overlapping windows", daily[col], s.nw_lags_daily),
        ]
        for label, ser, lags in cases:
            x = _clean(ser)
            nw = mean_nw(x, lags, s.alpha)
            daily_case = label.startswith("daily")
            note = f"win rate {float((x > 0).mean()):.3f}"
            if not daily_case:
                bt = boot.run(s_mean, x, alpha=s.alpha)
                note += (
                    f"; bootstrap CI [{bt['ci_low']:.4g}, {bt['ci_high']:.4g}], "
                    f"p = {bt['p']:.4g}"
                )
            else:
                note += "; no bootstrap (daily windows overlap by 20 days; NW only)"
            rows.append(
                row(
                    test="5_robustness",
                    measure=col,
                    sample=label,
                    statistic="mean",
                    estimate=nw["mean"],
                    ci_low=nw["ci_low"],
                    ci_high=nw["ci_high"],
                    ci_method=f"Newey–West ({lags} lags)",
                    null_value=0.0,
                    test_name=f"HAC t, t = {nw['t']:.2f}",
                    p_value=nw["p"],
                    n=nw["n"],
                    note=note,
                )
            )
        # Pre vs post 2003: is the premium different in the live-VIX era?
        x_post, x_pre = _clean(monthly.loc[post, col]), _clean(monthly.loc[~post, col])
        welch = sps.ttest_ind(x_post, x_pre, equal_var=False)
        diff = boot.run(s_diff_means, monthly[col].to_numpy(float), post.to_numpy(), alpha=s.alpha)
        rows.append(
            row(
                test="5_robustness",
                measure=col,
                sample="post-2003 minus pre-2003",
                statistic="diff_mean",
                estimate=float(x_post.mean() - x_pre.mean()),
                ci_low=diff["ci_low"],
                ci_high=diff["ci_high"],
                ci_method=f"block bootstrap ({boot.block}-month blocks)",
                null_value=0.0,
                test_name=f"Welch t, t = {welch.statistic:.2f}",
                p_value=float(welch.pvalue),
                n=len(x_post) + len(x_pre),
                note=f"bootstrap p = {diff['p']:.4g}",
            )
        )
    out = pd.DataFrame(rows, columns=ROW_COLUMNS)
    # Holm across the robustness variants (per measure) so "significant in every variant"
    # survives multiple testing.
    out["p_holm_within_measure"] = np.nan
    for col in s.shape_measures:
        m = (out["measure"] == col) & (out["statistic"] == "mean")
        out.loc[m, "p_holm_within_measure"] = holm(out.loc[m, "p_value"].tolist(), s.alpha)
    return out


# =========================================================================== runner
@dataclass
class StatsResults:
    """Everything the report, claims and figures need."""

    tables: pd.DataFrame
    deciles: dict[str, pd.DataFrame]
    mz_line: dict[str, object]
    variants: pd.DataFrame
    worst_months: pd.DataFrame
    settings: StatsSettings


def run_core_tests(monthly: pd.DataFrame, cfg: Config) -> StatsResults:
    """Run Tests 1–4 only (no robustness variants), e.g. on the headline sub-sample."""
    s = StatsSettings.from_config(cfg)
    boot = Bootstrapper(s.block_length, s.n_boot, s.boot_seed)
    t3, deciles = test_crash(monthly, s, boot)
    tables = pd.concat(
        [test_mean(monthly, s, boot), test_shape(monthly, s, boot), t3, test_forecast(monthly, s)],
        ignore_index=True,
    )
    log.info("stats (core tests 1–4): %d months, %d result rows", len(monthly), len(tables))
    worst = monthly.nsmallest(10, "payoff_scaled")
    return StatsResults(tables, deciles, mz_fit_line(monthly, s), pd.DataFrame(), worst, s)


def run_all_tests(panel: pd.DataFrame, monthly: pd.DataFrame, cfg: Config) -> StatsResults:
    """Run Tests 1–5 and return the combined results."""
    s = StatsSettings.from_config(cfg)
    boot = Bootstrapper(s.block_length, s.n_boot, s.boot_seed)
    log.info(
        "stats: %d months, NW lags %d (monthly) / %d (daily), %d bootstrap resamples "
        "(%d-month blocks)",
        len(monthly),
        s.nw_lags_monthly,
        s.nw_lags_daily,
        s.n_boot,
        s.block_length,
    )
    variants = build_monthly_variants(panel, monthly, cfg)
    daily = daily_sample(panel, cfg)
    log.info("stats: daily overlapping sample %d rows", len(daily))

    t1 = test_mean(monthly, s, boot)
    log.info("stats: test 1 (mean) %d rows", len(t1))
    t2 = test_shape(monthly, s, boot)
    log.info("stats: test 2 (shape) %d rows", len(t2))
    t3, deciles = test_crash(monthly, s, boot)
    log.info("stats: test 3 (crash) %d rows", len(t3))
    t4 = test_forecast(monthly, s)
    log.info("stats: test 4 (forecast) %d rows", len(t4))
    t5 = test_robustness(monthly, variants, daily, s, boot)
    log.info("stats: test 5 (robustness) %d rows", len(t5))

    tables = pd.concat([t1, t2, t3, t4, t5], ignore_index=True)
    worst = monthly.nsmallest(10, "payoff_scaled")[
        [
            "month",
            "vix_close",
            "rv_fwd21_ann_vol_pct",
            "payoff_var",
            "payoff_scaled",
            "spx_fwd21_ret_log",
            "fwd_window_start",
            "fwd_window_end",
        ]
    ]
    return StatsResults(tables, deciles, mz_fit_line(monthly, s), variants, worst, s)
