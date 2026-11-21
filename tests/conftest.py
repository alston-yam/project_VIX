"""Shared fixtures: a synthetic panel builder and (optionally) the real processed data."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from vrp import clean, features
from vrp.config import Config, load_config
from vrp.validate import QualityLog

WINDOW = 21
ANN = 252


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The real project config."""
    return load_config()


def make_inputs(log_returns: np.ndarray, vix: float | np.ndarray = 20.0,
                start: str = "2000-01-03", p0: float = 100.0) -> tuple[pd.Series, pd.DataFrame]:
    """Build (vix series, spx OHLCV frame) on business days from given daily log returns.

    The first S&P date precedes the first VIX date so the first panel return exists.
    """
    n = len(log_returns) + 1
    dates = pd.bdate_range(start, periods=n)
    close = p0 * np.exp(np.r_[0.0, np.cumsum(log_returns)])
    spx = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                        "volume": 1.0}, index=dates)
    v = np.broadcast_to(np.asarray(vix, dtype=float), (n - 1,))
    vix_s = pd.Series(v, index=dates[1:], name="vix_close")
    return vix_s, spx


def build_panel(vix: pd.Series, spx: pd.DataFrame) -> pd.DataFrame:
    """Run the real merge + feature code on synthetic inputs."""
    merged = clean.merge_vix_spx(vix, spx, QualityLog())
    return features.add_features(merged, WINDOW, ANN)


@pytest.fixture
def random_inputs() -> tuple[pd.Series, pd.DataFrame]:
    """300 days of random returns and random VIX levels (seeded)."""
    rng = np.random.default_rng(42)
    r = rng.normal(0, 0.012, 300)
    v = rng.uniform(10, 40, 300)
    return make_inputs(r, v)


@pytest.fixture
def random_panel(random_inputs: tuple[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """Feature panel built from ``random_inputs``."""
    return build_panel(*random_inputs)


def cfg_with(cfg: Config, **dates: str) -> Config:
    """Copy of the config with some ``dates:`` entries overridden."""
    raw = copy.deepcopy(cfg.raw)
    raw["dates"].update(dates)
    return Config(raw=raw, root=cfg.root)


@pytest.fixture(scope="session")
def real_panel(cfg: Config) -> pd.DataFrame:
    """The processed daily panel (skips if `make data` has not been run)."""
    p = cfg.path("processed") / "daily_panel.parquet"
    if not p.exists():
        pytest.skip("processed data not built; run `make data`")
    return pd.read_parquet(p)


@pytest.fixture(scope="session")
def real_monthly(cfg: Config) -> pd.DataFrame:
    """The processed monthly sample (skips if `make data` has not been run)."""
    p = cfg.path("processed") / "monthly_sample.csv"
    if not p.exists():
        pytest.skip("processed data not built; run `make data`")
    df = pd.read_csv(p, index_col=0, parse_dates=[0])
    for c in ("fwd_window_start", "fwd_window_end"):
        df[c] = pd.to_datetime(df[c])
    return df
