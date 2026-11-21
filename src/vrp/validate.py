"""Data-quality bookkeeping, assertions and the two markdown reports.

``QualityLog`` collects findings while the pipeline runs (dropped dates, gaps,
merge mismatches, extreme returns, assertion results). At the end,
``write_quality_report`` and ``write_sanity_summary`` turn them into markdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vrp.config import Config
from vrp.logging_setup import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- recorder
@dataclass
class Check:
    """One named assertion and whether it passed."""

    name: str
    passed: bool
    detail: str
    severity: str = "error"  # "error" fails the run; "warn" only warns


@dataclass
class QualityLog:
    """Accumulates everything that goes into ``data_quality_report.md``."""

    sections: dict[str, list[str]] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)

    def note(self, section: str, line: str) -> None:
        """Add a bullet line under ``section``."""
        self.sections.setdefault(section, []).append(line)

    def table(self, key: str, df: pd.DataFrame) -> None:
        """Attach a table (rendered as markdown) under ``key``."""
        self.tables[key] = df

    def check(self, name: str, passed: bool, detail: str, severity: str = "error") -> bool:
        """Record an assertion; log loudly when it fails."""
        self.checks.append(Check(name, bool(passed), detail, severity))
        if passed:
            log.info("CHECK PASS  %s: %s", name, detail)
        else:
            level = logging.ERROR if severity == "error" else logging.WARNING
            banner = "!" * 70
            log.log(level, "%s\nCHECK FAIL  %s: %s\n%s", banner, name, detail, banner)
        return bool(passed)

    @property
    def hard_failures(self) -> list[Check]:
        """Checks with severity 'error' that did not pass."""
        return [c for c in self.checks if not c.passed and c.severity == "error"]


def df_to_md(df: pd.DataFrame, index: bool = False, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a GitHub markdown table without extra dependencies."""
    if df.empty:
        return "_(none)_"
    d = df.reset_index() if index else df.copy()

    def fmt(v: Any) -> str:
        if isinstance(v, float | np.floating):
            return "" if np.isnan(v) else format(v, floatfmt)
        if isinstance(v, pd.Timestamp):
            return str(v.date())
        return str(v)

    header = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


# --------------------------------------------------------------------------- validations
def validate_outputs(panel: pd.DataFrame, monthly: pd.DataFrame, cfg: Config,
                     q: QualityLog) -> None:
    """Run the end-of-pipeline assertions on the processed datasets."""
    chk = cfg["checks"]
    w = cfg.window

    # Highest VIX close in the sample: 82.69 on 2020-03-16 (loud warning if not).
    vmax_date = panel["vix_close"].idxmax()
    vmax = float(panel["vix_close"].max())
    q.check("max_vix_close",
            np.isclose(vmax, chk["expected_max_vix_close"])
            and vmax_date == pd.Timestamp(chk["expected_max_vix_date"]),
            f"max VIX close = {vmax:.2f} on {vmax_date.date()} "
            f"(expected {chk['expected_max_vix_close']} on {chk['expected_max_vix_date']})",
            severity="warn")

    # Panel integrity.
    q.check("panel_index_unique_sorted",
            panel.index.is_unique and panel.index.is_monotonic_increasing,
            "daily panel index is unique and sorted")
    q.check("panel_no_nan_core",
            not panel[["spx_close", "vix_close", "iv_ann_var"]].isna().any().any(),
            "no NaN in spx_close / vix_close / iv_ann_var")
    n_nan_ret = int(panel["ret_log"].isna().sum())
    q.check("panel_ret_log_complete", n_nan_ret == 0,
            f"ret_log NaN count = {n_nan_ret} (first panel date uses prior S&P close)")
    complete = panel["fwd_window_complete"]
    q.check("fwd_complete_iff_values",
            panel.loc[complete, "rv_fwd21_ann_var"].notna().all()
            and panel.loc[~complete, "rv_fwd21_ann_var"].isna().all(),
            "rv_fwd21_ann_var is defined exactly when fwd_window_complete is True")
    n_incomplete = int((~complete).sum())
    q.check("fwd_incomplete_only_at_end", n_incomplete == w,
            f"{n_incomplete} rows lack a complete forward window (expected {w}: the last {w} days)")
    q.check("rv_non_negative", (panel["rv_fwd21_ann_var"].dropna() >= 0).all()
            and (panel["rv_past21_ann_var"].dropna() >= 0).all(),
            "realised variances are non-negative")
    q.check("payoff_scaled_le_1", (panel["payoff_scaled"].dropna() <= 1 + 1e-12).all(),
            "payoff_scaled <= 1 everywhere (RV cannot be negative)")
    q.check("fwd_start_after_t",
            (panel.loc[complete, "fwd_window_start"] > panel.index[complete]).all(),
            "fwd_window_start > t for every complete row")

    # Monthly sample.
    n = len(monthly)
    q.check("monthly_row_count", n == chk["expected_monthly_rows"],
            f"monthly rows = {n} (expected about {chk['expected_monthly_rows']})", severity="warn")
    q.check("monthly_one_per_month", monthly["month"].is_unique, "one row per month")
    q.check("monthly_windows_within_data",
            (monthly["fwd_window_end"] <= panel.index.max()).all(),
            "no monthly forward window extends past the last available date")
    q.check("monthly_all_complete", monthly["fwd_window_complete"].all(),
            "every monthly row has a complete forward window")


# --------------------------------------------------------------------------- sanity summary
def sanity_table(panel: pd.DataFrame, monthly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute our numbers next to the published benchmarks (no fudging)."""
    chk = cfg["checks"]
    d = cfg["dates"]
    tol = chk["benchmark_tolerance_vol_pts"]
    b = monthly.loc[(monthly.index >= d["benchmark_start"]) & (monthly.index <= d["benchmark_end"])]
    bd = panel.loc[d["benchmark_start"]:d["benchmark_end"]]
    rows: list[dict[str, Any]] = []

    def add(item: str, ours: str, bench: str, ok: bool | None, note: str = "") -> None:
        mark = "—" if ok is None else ("✅" if ok else "⚠️")
        rows.append({"Item": item, "Ours": ours, "Benchmark": bench, "Status": mark, "Note": note})

    m_vix = float(b["vix_close"].mean())
    add("Mean VIX, 1990–2024 (monthly sample)", f"{m_vix:.2f}", f"≈ {chk['benchmark_mean_vix']}",
        abs(m_vix - chk["benchmark_mean_vix"]) <= tol,
        f"first-trading-day sample, n={len(b)}; daily-panel mean = {bd['vix_close'].mean():.2f}")
    m_rv = float(b["rv_fwd21_ann_vol_pct"].mean())
    add("Mean realised vol, 1990–2024 (fwd 21d, monthly sample)", f"{m_rv:.2f}",
        f"≈ {chk['benchmark_mean_rv']}", abs(m_rv - chk["benchmark_mean_rv"]) <= tol,
        f"mean of √RV in vol %; daily-panel mean = {bd['rv_fwd21_ann_vol_pct'].mean():.2f}")
    n = len(monthly)
    add("Monthly row count, 1990-01 to 2025-12", str(n), f"≈ {chk['expected_monthly_rows']}",
        n == chk["expected_monthly_rows"])
    share = float((monthly["vix_close"] > monthly["rv_fwd21_ann_vol_pct"]).mean())
    add("Share of months with VIX > realised vol", f"{share:.1%}", "no published benchmark",
        None, "full monthly sample 1990–2025")
    vmax_date = panel["vix_close"].idxmax()
    vmax = float(panel["vix_close"].max())
    add("Highest VIX close and date", f"{vmax:.2f} on {vmax_date.date()}",
        f"{chk['expected_max_vix_close']} on {chk['expected_max_vix_date']}",
        bool(np.isclose(vmax, chk["expected_max_vix_close"])
             and vmax_date == pd.Timestamp(chk["expected_max_vix_date"])))
    return pd.DataFrame(rows)


def write_sanity_summary(table: pd.DataFrame, path: Path) -> Path:
    """Write ``sanity_summary.md``."""
    warn = table[table["Status"] == "⚠️"]
    lines = [
        "# Sanity summary",
        "",
        "Our computed values next to published benchmarks. Nothing has been adjusted to match.",
        "✅ = within tolerance (±1 vol point for the means; exact for counts and the record close),",
        "⚠️ = outside tolerance, — = no benchmark.",
        "",
        df_to_md(table),
        "",
        "Benchmarks: mean VIX ≈ 19.6 and mean realised vol ≈ 15.5 for 1990–2024 "
        "(CFA Institute Enterprising Investor blog, 31 Jul 2024, "
        "\"How Well Does the Market Predict Volatility?\"). The CFA figures may use a different "
        "realised-vol window and sampling, so a gap of a few tenths is expected.",
        "",
        f"**⚠️ items: {len(warn)}**" + ("" if warn.empty else " — " + "; ".join(warn["Item"])),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def write_quality_report(q: QualityLog, path: Path) -> Path:
    """Write ``data_quality_report.md`` from the collected findings."""
    lines = ["# Data quality report", "",
             "Generated by `python -m vrp.pipeline`. Every cleaning action is listed with counts.",
             "Flagged values are **reported, not removed** (real crash days are the point).", ""]
    if q.counts:
        lines += ["## Row counts", "", df_to_md(pd.DataFrame(
            [{"Step": k, "Rows": v} for k, v in q.counts.items()])), ""]
    for section, items in q.sections.items():
        lines += [f"## {section}", ""] + [f"- {s}" for s in items] + [""]
        for key, tbl in q.tables.items():
            if key.startswith(section + "::"):
                lines += [f"**{key.split('::', 1)[1]}**", "", df_to_md(tbl), ""]
    lines += ["## Assertion results", ""]
    chk_df = pd.DataFrame([{"Check": c.name, "Result": "✅ pass" if c.passed else
                            ("❌ FAIL" if c.severity == "error" else "⚠️ warn"),
                            "Detail": c.detail} for c in q.checks])
    lines += [df_to_md(chk_df), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", path)
    return path
