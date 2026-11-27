"""Orchestrate the v1 data layer.

Usage::

    python -m vrp.pipeline                 # everything (data + figures)
    python -m vrp.pipeline --stage data    # download (cached) → clean → features → sample → events
    python -m vrp.pipeline --stage figures # figures from processed data
    python -m vrp.pipeline --stage stats   # v2 statistical tests, reports, claims, figures 07–11
    python -m vrp.pipeline --stage research  # research tests R-1…R-9, reports, figures 12–15
    python -m vrp.pipeline --refresh       # ignore the raw-data cache
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from vrp import (
    clean,
    events,
    features,
    plots,
    research,
    research_report,
    sampling,
    stats,
    stats_report,
    validate,
)
from vrp.config import Config, load_config
from vrp.download import download_all, read_meta, sha256_file
from vrp.logging_setup import get_logger, setup_logging

log = get_logger("vrp.pipeline")
DATE_COLS = ["fwd_window_start", "fwd_window_end"]


# --------------------------------------------------------------------------- io helpers
def git_info(root: Path) -> dict[str, str | bool]:
    """Current git commit hash and whether the working tree is dirty."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "src", "config.yaml"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": head, "code_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "code_dirty": True}


def write_csv(df: pd.DataFrame, path: Path, index_label: str | None = "date") -> Path:
    """Write a CSV with ISO dates and log its checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index_label is not None, index_label=index_label, date_format="%Y-%m-%d")
    log.info("wrote %s (%d rows, sha256=%s…)", path.name, len(df), sha256_file(path)[:12])
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write a parquet file and log its checksum."""
    df.to_parquet(path, index=True)
    log.info("wrote %s (%d rows, sha256=%s…)", path.name, len(df), sha256_file(path)[:12])
    return path


def read_processed(cfg: Config, name: str) -> pd.DataFrame:
    """Read a processed CSV with date columns parsed."""
    df = pd.read_csv(cfg.path("processed") / f"{name}.csv", index_col=0, parse_dates=[0])
    for c in DATE_COLS:
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def file_entry(path: Path, df: pd.DataFrame, cfg: Config) -> dict[str, object]:
    """Manifest entry for one processed file."""
    if isinstance(df.index, pd.DatetimeIndex) and len(df):
        first, last = str(df.index.min().date()), str(df.index.max().date())
    elif "start_date" in df and len(df):
        first, last = str(pd.to_datetime(df["start_date"]).min().date()), str(
            pd.to_datetime(df["end_date"]).max().date()
        )
    else:
        first = last = None
    return {
        "path": str(path.relative_to(cfg.root)),
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "first_date": first,
        "last_date": last,
        "sha256": sha256_file(path),
    }


# --------------------------------------------------------------------------- stages
def run_data(cfg: Config, refresh: bool) -> None:
    """Download → clean → merge → features → monthly sample → events → validate → reports."""
    q = validate.QualityLog()
    raw = download_all(cfg, refresh=refresh)
    for name, p in raw.items():
        if p is not None:
            m = read_meta(p)
            q.note(
                "Sources used",
                f"{name}: {m.get('provider')} ({m.get('series_id') or m.get('ticker')}), "
                f"source_used={m.get('source_used')}, downloaded {m.get('downloaded_utc')}, "
                f"{m.get('row_count')} raw rows {m.get('first_date')}..{m.get('last_date')}, "
                f"sha256 {str(m.get('sha256'))[:12]}…",
            )
        else:
            q.note("Sources used", f"{name}: NOT AVAILABLE (optional source failed)")

    interim = cfg.path("interim")
    vix = clean.clean_vix_like(clean.read_fred_raw(raw["vix"], "vix_close"), "vix", cfg, q)
    write_csv(vix.to_frame("vix_close"), interim / "vix.csv")
    spx = clean.clean_spx(clean.read_spx_raw(raw["spx"]), cfg, q)
    write_csv(spx, interim / "spx.csv")
    vxo = None
    if raw.get("vxo") is not None:
        vxo = clean.clean_vix_like(
            clean.read_fred_raw(raw["vxo"], "vxo_close"), "vxo", cfg, q, check_range=False
        )
        write_csv(vxo.to_frame("vxo_close"), interim / "vxo.csv")
    if raw.get("usrec") is not None:
        usrec = clean.clean_usrec(clean.read_fred_raw(raw["usrec"], "usrec"), cfg, q)
        write_csv(usrec.to_frame("usrec"), interim / "usrec.csv")

    merged = clean.merge_vix_spx(vix, spx, q, vxo=vxo)
    panel = features.add_features(merged, cfg.window, cfg.annualisation)
    monthly = sampling.build_monthly_sample(panel, cfg)
    q.counts["daily panel rows"] = len(panel)
    q.counts["daily panel rows with complete forward window"] = int(
        panel["fwd_window_complete"].sum()
    )
    q.counts["monthly sample rows"] = len(monthly)

    ov = sampling.window_overlap(monthly, panel)
    summ = sampling.overlap_summary(ov)
    q.note(
        "Monthly sampling",
        f"{len(monthly)} monthly rows "
        f"({monthly['month'].iloc[0]} to {monthly['month'].iloc[-1]})",
    )
    q.note(
        "Monthly sampling",
        "consecutive forward-window overlap (trading days): "
        + ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in summ.items()
        ),
    )
    q.table(
        "Monthly sampling::Overlap distribution (overlap_days > 0 shared, < 0 uncovered)",
        ov["overlap_days"]
        .value_counts()
        .sort_index()
        .rename_axis("overlap_days")
        .reset_index(name="pairs"),
    )
    log.info("sampling: window overlap summary %s", summ)

    ev = events.event_table()
    stats = events.event_stats(ev, panel, monthly)

    validate.validate_outputs(panel, monthly, cfg, q)

    proc = cfg.path("processed")
    written = {
        "daily_panel.csv": (write_csv(panel, proc / "daily_panel.csv"), panel),
        "daily_panel.parquet": (write_parquet(panel, proc / "daily_panel.parquet"), panel),
        "monthly_sample.csv": (write_csv(monthly, proc / "monthly_sample.csv"), monthly),
        "events.csv": (write_csv(ev, proc / "events.csv", index_label=None), ev),
    }
    reports = cfg.path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    write_csv(stats, reports / "event_stats.csv", index_label=None)
    events.write_events_md(ev, stats, panel, cfg.path("events_docs"))
    sanity = validate.sanity_table(panel, monthly, cfg)
    validate.write_sanity_summary(sanity, reports / "sanity_summary.md")
    validate.write_quality_report(q, reports / "data_quality_report.md")

    manifest = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(cfg.root),
        "config": {"window": cfg.window, "annualisation": cfg.annualisation, **cfg["dates"]},
        "raw_sources": {
            n: {
                k: read_meta(p).get(k)
                for k in ("provider", "source_used", "sha256", "row_count", "downloaded_utc")
            }
            for n, p in raw.items()
            if p is not None
        },
        "files": [file_entry(p, df, cfg) for p, df in written.values()]
        + [file_entry(reports / "event_stats.csv", stats.set_index("event_id"), cfg)],
    }
    (reports / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote manifest.json (%d files)", len(manifest["files"]))

    print("\n" + validate.df_to_md(sanity) + "\n")
    if q.hard_failures:
        for c in q.hard_failures:
            log.error("HARD CHECK FAILED: %s — %s", c.name, c.detail)
        raise SystemExit(1)
    warns = [c for c in q.checks if not c.passed]
    log.info(
        "data stage finished: %d checks, %d warnings, 0 hard failures", len(q.checks), len(warns)
    )


def run_figures(cfg: Config) -> None:
    """Draw every figure from the processed (and interim) data."""
    panel = read_processed(cfg, "daily_panel")
    monthly = read_processed(cfg, "monthly_sample")
    ev = pd.read_csv(cfg.path("processed") / "events.csv", parse_dates=["start_date", "end_date"])
    interim = cfg.path("interim")
    usrec = None
    if (interim / "usrec.csv").exists():
        usrec = pd.read_csv(interim / "usrec.csv", index_col=0, parse_dates=[0])["usrec"]
    coverage: dict[str, pd.DatetimeIndex] = {}
    for label, fname in [
        ("VIX (FRED VIXCLS)", "vix.csv"),
        ("S&P 500 (Yahoo ^GSPC)", "spx.csv"),
        ("Merged panel (both)", None),
        ("VXO (FRED VXOCLS)", "vxo.csv"),
    ]:
        if fname is None:
            coverage[label] = pd.DatetimeIndex(panel.index)
        elif (interim / fname).exists():
            coverage[label] = pd.DatetimeIndex(
                pd.read_csv(interim / fname, index_col=0, parse_dates=[0]).index
            )
    paths = plots.make_all_figures(panel, monthly, ev, usrec, coverage, cfg)
    log.info("figures stage finished: %d files", len(paths))


def run_stats(cfg: Config) -> None:
    """v2: run Tests 1–5, write tables, summary, pitch claims, variants and figures 07–11."""
    panel = read_processed(cfg, "daily_panel")
    monthly = read_processed(cfg, "monthly_sample")
    res = stats.run_all_tests(panel, monthly, cfg)
    out = cfg["stats"]["outputs"]
    reports = cfg.path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    variants_path = write_csv(res.variants, cfg.path("processed") / out["variants"])
    tables_path = write_csv(res.tables, reports / out["tables"], index_label=None)
    dec = pd.concat([d.assign(measure=m) for m, d in res.deciles.items()], ignore_index=True)
    deciles_path = write_csv(dec, reports / "stats_deciles.csv", index_label=None)
    claims = stats_report.evaluate_claims(res)
    claims_csv = write_csv(claims, reports / "pitch_claims.csv", index_label=None)
    stats_report.write_summary(res, claims, reports / out["summary"])
    stats_report.write_claims(res, claims, reports / out["claims"])
    figs = plots.make_stats_figures(monthly, res, cfg)

    # Add the v2 files to the manifest (replacing older entries for the same paths).
    man_path = reports / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    entries = [
        file_entry(variants_path, res.variants, cfg),
        file_entry(tables_path, res.tables, cfg),
        file_entry(deciles_path, dec, cfg),
        file_entry(claims_csv, claims, cfg),
    ]
    new_paths = {e["path"] for e in entries}
    manifest["files"] = [
        f for f in manifest.get("files", []) if f["path"] not in new_paths
    ] + entries
    manifest["stats"] = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(cfg.root),
        "settings": {k: v for k, v in cfg["stats"].items() if k != "outputs"},
        "claims_supported": int(claims["supported"].sum()),
        "claims_total": int(len(claims)),
    }
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(
        "stats stage finished: %d result rows, %d/%d claims supported, %d figure files",
        len(res.tables),
        int(claims["supported"].sum()),
        len(claims),
        len(figs),
    )


def _update_manifest(cfg: Config, entries: list[dict[str, object]], key: str, info: dict) -> None:
    """Add file entries (replacing older ones for the same paths) and a stage section."""
    man_path = cfg.path("reports") / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    new_paths = {e["path"] for e in entries}
    manifest["files"] = [
        f for f in manifest.get("files", []) if f["path"] not in new_paths
    ] + entries
    manifest[key] = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(cfg.root),
        **info,
    }
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_research(cfg: Config) -> None:
    """Research section: tests R-1…R-9, tables, summary, claims and figures 12–15."""
    panel = read_processed(cfg, "daily_panel")
    monthly = read_processed(cfg, "monthly_sample")
    ev = pd.read_csv(cfg.path("processed") / "events.csv", parse_dates=["start_date", "end_date"])
    res = research.run_research(panel, monthly, cfg)
    out = cfg["research"]["outputs"]
    reports = cfg.path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    rv_path = write_csv(res.rv_monthly, cfg.path("processed") / out["rv_monthly"])
    yr_path = write_csv(res.rv_year, reports / out["rv_by_year"], index_label="year")
    tables_path = write_csv(res.tables, reports / out["tables"], index_label=None)
    roll_path = write_csv(res.rolling, reports / out["rolling"])
    claims = research_report.evaluate_research_claims(res)
    claims_csv = write_csv(claims, reports / out["claims"].replace(".md", ".csv"), index_label=None)
    research_report.write_research_summary(res, claims, reports / out["summary"])
    research_report.write_research_claims(res, claims, reports / out["claims"])
    figs = plots.make_research_figures(res, monthly, ev, claims, cfg)

    _update_manifest(
        cfg,
        [
            file_entry(rv_path, res.rv_monthly, cfg),
            file_entry(yr_path, res.rv_year, cfg),
            file_entry(tables_path, res.tables, cfg),
            file_entry(roll_path, res.rolling, cfg),
            file_entry(claims_csv, claims, cfg),
        ],
        "research",
        {
            "settings": {k: v for k, v in cfg["research"].items() if k != "outputs"},
            "claims_supported": int(claims["supported"].sum()),
            "claims_total": int(len(claims)),
        },
    )
    log.info(
        "research stage finished: %d result rows, %d/%d claims supported, %d figure files",
        len(res.tables),
        int(claims["supported"].sum()),
        len(claims),
        len(figs),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--stage", choices=["all", "data", "figures", "stats", "research"], default="all"
    )
    ap.add_argument("--refresh", action="store_true", help="re-download raw data")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cfg.ensure_dirs()
    log_path = setup_logging(cfg.path("logs"))
    log.info("=" * 70)
    log.info("pipeline start: stage=%s refresh=%s log=%s", args.stage, args.refresh, log_path)
    if args.stage in ("all", "data"):
        run_data(cfg, args.refresh)
    if args.stage in ("all", "figures"):
        run_figures(cfg)
    if args.stage in ("all", "stats"):
        run_stats(cfg)
    if args.stage in ("all", "research"):
        run_research(cfg)
    log.info("pipeline done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
