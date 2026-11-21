"""Fetch raw data, cache it under ``data/raw/`` and write a ``.meta.json`` sidecar.

Raw files are written exactly as received (FRED CSV bytes) or, for the S&P 500,
as the DataFrame returned by the provider with no transformation other than
serialising to CSV. Nothing is cleaned here.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import requests

from vrp.config import Config
from vrp.logging_setup import get_logger

log = get_logger(__name__)
T = TypeVar("T")

LIBRARIES = ["pandas", "numpy", "requests", "yfinance", "pandas-datareader", "pyarrow"]


# --------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def library_versions() -> dict[str, str]:
    """Return installed versions of the libraries that touch the data."""
    out = {"python": platform.python_version()}
    for lib in LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = "not installed"
    return out


def with_retry(fn: Callable[[], T], what: str, max_tries: int, backoff: float) -> T:
    """Call ``fn`` up to ``max_tries`` times with exponential backoff (backoff * 2**k)."""
    last: Exception | None = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise after the last try
            last = exc
            log.warning("%s: attempt %d/%d failed: %s: %s", what, attempt, max_tries,
                        type(exc).__name__, exc)
            if attempt < max_tries:
                wait = backoff * (2 ** (attempt - 1))
                log.info("%s: waiting %.0fs before retry", what, wait)
                time.sleep(wait)
    assert last is not None
    raise last


def _first_last_rows(csv_path: Path, date_col: int = 0) -> tuple[int, str | None, str | None]:
    """Row count and first/last date string of a CSV whose first column is a date."""
    df = pd.read_csv(csv_path, usecols=[date_col], dtype=str)
    col = df.iloc[:, 0].dropna()
    if col.empty:
        return 0, None, None
    dates = pd.to_datetime(col, errors="coerce").dropna()
    return len(df), str(dates.min().date()), str(dates.max().date())


def write_meta(raw_path: Path, info: dict[str, Any]) -> Path:
    """Write ``<raw_path>.meta.json`` (e.g. ``vix.meta.json``) next to the raw file."""
    n, first, last = _first_last_rows(raw_path)
    meta = {
        "file": raw_path.name,
        **info,
        "downloaded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "library_versions": library_versions(),
        "row_count": n,
        "first_date": first,
        "last_date": last,
        "sha256": sha256_file(raw_path),
    }
    meta_path = raw_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s (rows=%d, %s..%s, sha256=%s…)", meta_path.name, n, first, last,
             meta["sha256"][:12])
    return meta_path


# --------------------------------------------------------------------------- FRED
def fetch_fred(name: str, cfg: Config, refresh: bool = False) -> Path:
    """Download one FRED series CSV (no API key) to ``data/raw/<name>.csv``."""
    src = cfg["sources"][name]
    raw_path = cfg.path("raw") / f"{name}.csv"
    if raw_path.exists() and not refresh:
        log.info("%s: cached raw file found (%s); skipping download", name, raw_path.name)
        return raw_path
    dl = cfg["download"]

    def _get() -> bytes:
        resp = requests.get(src["url"], timeout=dl["http_timeout_seconds"])
        resp.raise_for_status()
        if not resp.content.strip():
            raise ValueError("empty response")
        return resp.content

    content = with_retry(_get, f"FRED {src['series_id']}", dl["max_tries"],
                         dl["backoff_base_seconds"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)
    log.info("%s: downloaded %d bytes from %s", name, len(content), src["url"])
    write_meta(raw_path, {
        "name": name,
        "provider": src["provider"],
        "series_id": src["series_id"],
        "source_url": src["url"],
        "source_used": "FRED",
    })
    return raw_path


# --------------------------------------------------------------------------- S&P 500
def _yf_download(ticker: str, start: str, end_inclusive: str, auto_adjust: bool) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo via yfinance (``end`` made inclusive)."""
    import yfinance as yf

    end_excl = (pd.Timestamp(end_inclusive) + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end_excl, auto_adjust=auto_adjust,
                     multi_level_index=False, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):  # defensive: older yfinance versions
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _stooq_download(ticker: str, start: str, end_inclusive: str) -> pd.DataFrame:
    """Fallback: download daily OHLCV from Stooq via pandas_datareader."""
    from pandas_datareader import data as pdr

    df = pdr.DataReader(ticker, "stooq", start=start, end=end_inclusive)
    if df is None or df.empty:
        raise RuntimeError(f"Stooq returned no data for {ticker}")
    df = df.sort_index()
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_spx(cfg: Config, refresh: bool = False) -> Path:
    """Download the S&P 500 index (Yahoo ``^GSPC``; Stooq ``^SPX`` as fallback)."""
    src = cfg["sources"]["spx"]
    dates = cfg["dates"]
    dl = cfg["download"]
    raw_path = cfg.path("raw") / "spx.csv"
    if raw_path.exists() and not refresh:
        log.info("spx: cached raw file found (%s); skipping download", raw_path.name)
        return raw_path

    start, end = dates["download_start"], dates["download_end"]
    info: dict[str, Any] = {"name": "spx", "requested_start": start, "requested_end": end}
    try:
        df = with_retry(lambda: _yf_download(src["ticker"], start, end, src["auto_adjust"]),
                        f"yfinance {src['ticker']}", dl["max_tries"], dl["backoff_base_seconds"])
        info.update({
            "provider": src["provider"],
            "ticker": src["ticker"],
            "source_used": "yfinance",
            "call": (f"yf.download('{src['ticker']}', start='{start}', end='{end}'+1d, "
                     f"auto_adjust={src['auto_adjust']}, multi_level_index=False)"),
            "auto_adjust": src["auto_adjust"],
        })
        log.info("spx: SOURCE USED = yfinance %s (%d rows)", src["ticker"], len(df))
    except Exception as exc:  # noqa: BLE001
        log.error("spx: yfinance failed after retries (%s). Trying fallback %s %s",
                  exc, src["fallback_provider"], src["fallback_ticker"])
        df = with_retry(lambda: _stooq_download(src["fallback_ticker"], start, end),
                        f"stooq {src['fallback_ticker']}", dl["max_tries"],
                        dl["backoff_base_seconds"])
        info.update({
            "provider": src["fallback_provider"],
            "ticker": src["fallback_ticker"],
            "source_used": "stooq (fallback)",
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        })
        log.warning("spx: SOURCE USED = Stooq fallback %s (%d rows)", src["fallback_ticker"],
                    len(df))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    df.to_csv(buf, date_format="%Y-%m-%d")
    raw_path.write_text(buf.getvalue(), encoding="utf-8")
    write_meta(raw_path, info)
    return raw_path


# --------------------------------------------------------------------------- orchestration
def download_all(cfg: Config, refresh: bool = False) -> dict[str, Path | None]:
    """Fetch every configured source. Optional sources may fail without stopping the run."""
    out: dict[str, Path | None] = {}
    for name, src in cfg["sources"].items():
        try:
            out[name] = fetch_spx(cfg, refresh) if name == "spx" else fetch_fred(name, cfg, refresh)
        except Exception as exc:  # noqa: BLE001
            if src.get("required", False):
                log.error("%s: REQUIRED source failed: %s", name, exc)
                raise
            log.warning("%s: optional source failed and will be skipped: %s", name, exc)
            out[name] = None
    return out


def read_meta(raw_path: Path) -> dict[str, Any]:
    """Load the sidecar metadata for a raw file (empty dict if missing)."""
    p = raw_path.with_suffix(".meta.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
