"""Tests for the pitch deck builder (``vrp.deck``)."""

from __future__ import annotations

import copy

import pytest
from pptx import Presentation

from vrp import deck
from vrp.config import Config


def test_number_formats() -> None:
    """Percentages and numbers use a true minus sign."""
    assert deck.pct(0.301) == "30.1%"
    assert deck.pct(-0.78, 0) == "\u221278%"
    assert deck.num(-5.016) == "\u22125.0"
    assert deck.ci_pct(0.219, 0.383) == "[21.9%, 38.3%]"


def test_build_deck(cfg, tmp_path) -> None:
    """The deck builds from the template, quotes our numbers, and keeps no template text."""
    if not (cfg.root / cfg["deck"]["template"]).exists():
        pytest.skip("template not present")
    if not (cfg.path("reports") / cfg["research"]["outputs"]["tables"]).exists():
        pytest.skip("research outputs not built; run `make research`")
    raw = copy.deepcopy(cfg.raw)
    raw["deck"]["output"] = str(tmp_path / "deck.pptx")
    out = deck.build_deck(Config(raw=raw, root=cfg.root))
    prs = Presentation(str(out))
    text = "\n".join(
        sh.text_frame.text for sl in prs.slides for sh in sl.shapes if sh.has_text_frame
    )
    assert len(prs.slides) == 17
    assert "Is the VIX Too Pessimistic?" in text
    for leftover in ("Insert Stock Name", "GARCH", "5/5/2025", "N. Surname"):
        assert leftover not in text
    n = deck.load_numbers(cfg)
    win = n.row("2_shape", "payoff_scaled", "win_rate", method="Wilson")["estimate"]
    assert deck.pct(win) in text
    # Step 5 is filled in: no placeholder boxes left on the research slides.
    names = [sh.name for sl in prs.slides for sh in sl.shapes]
    assert not any(nm.startswith("pending: In progress") for nm in names)
    assert not any("Break-even test (step 5)" in nm for nm in names)
    be = n.row("R_breakeven", "payoff_scaled", "breakeven_years_between")["estimate"]
    assert deck.num(be) in text
    assert deck.month_label("2020-03") == "Mar\u00a02020"
    assert deck.p_short(0.0004) == "p\u00a0<\u00a00.001"
