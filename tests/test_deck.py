"""Tests for the pitch deck builder (``vrp.deck``)."""

from __future__ import annotations

import copy

import pandas as pd
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
    assert len(prs.slides) == 18
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
    # Verdict: provisional subtitle on the slide; the spoken line lives in the speaker notes.
    assert "Provisional:" in text and "Yes, on average." in text
    notes = prs.slides[12].notes_slide.notes_text_frame.text
    assert "is the VIX too pessimistic?" in notes
    assert "So, is the VIX" not in text
    # The pitch leads with the live-VIX era; the full sample is attached for comparison.
    assert n.sfx == cfg["headline"]["suffix"] and n.full is not None and n.full.sfx == ""
    assert len(n.monthly) < len(n.full.monthly)
    assert n.monthly.index[0] >= pd.Timestamp(cfg["dates"]["vix_new_methodology_date"])
    assert f"since {n.since}" in text.lower()
    assert "1990–2025, we compare" not in text
    # The 1990s stay out of the story slides (3–13); they live on the backup slide.
    story = "\n".join(
        sh.text_frame.text for sl in list(prs.slides)[2:13] for sh in sl.shapes if sh.has_text_frame
    )
    assert "1990s" not in story and "back-history" not in story
    assert "Backup: The Longer History" in text


def test_out_of_six() -> None:
    """Fractions are only spoken when they are close to the measured share."""
    assert deck.out_of_six(0.838) == "five months out of six"
    assert deck.out_of_six(0.75) is None
