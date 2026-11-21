"""Load ``config.yaml`` and resolve paths relative to the repository root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Thin wrapper around the parsed YAML with convenience accessors."""

    raw: dict[str, Any]
    root: Path = REPO_ROOT

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, key: str) -> Path:
        """Return the absolute path for an entry under ``paths:``."""
        return self.root / self.raw["paths"][key]

    @property
    def window(self) -> int:
        """Realised-variance window length in trading days."""
        return int(self.raw["features"]["window"])

    @property
    def annualisation(self) -> int:
        """Trading days per year used to annualise variance."""
        return int(self.raw["features"]["annualisation"])

    def ensure_dirs(self) -> None:
        """Create every directory listed under ``paths:`` that is a folder."""
        for rel in self.raw["paths"].values():
            p = self.root / rel
            target = p.parent if p.suffix else p  # file path -> create its parent
            target.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | str | None = None) -> Config:
    """Read the YAML config file (default: ``<repo>/config.yaml``)."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=p.resolve().parent)
