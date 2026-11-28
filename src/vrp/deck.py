"""Pitch deck: build the research and intro slides into the OAF template (``make deck``).

Every number on a slide is read from the generated result tables (``stats_tables.csv``,
``research_tables.csv``, the claims CSVs, the sanity table), never typed in, so the deck
rebuilds when the data or the tests change. Charts are the PNG figures in
``outputs/figures``. Teammates' sections and step-5 results are clearly marked placeholders.

Sign convention on every slide: **implied − realised** (positive = variance seller gains).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from vrp import research, validate
from vrp.config import Config
from vrp.logging_setup import get_logger

log = get_logger(__name__)

# Template palette (theme accents of the OAF template).
NAVY = RGBColor(0x1F, 0x38, 0x64)
BLUE = RGBColor(0x2F, 0x54, 0x96)
RED = RGBColor(0xC0, 0x00, 0x00)
GREEN = RGBColor(0x2D, 0x77, 0x56)
INK = RGBColor(0x26, 0x26, 0x26)
MUTED = RGBColor(0x59, 0x59, 0x59)
CARD = RGBColor(0xF1, 0xF4, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

# Template layout indices (OAF Slide Template.pptx).
L_TITLE, L_TOC, L_CONTENT, L_DISCLAIMER = 0, 1, 2, 8

# Content area of the content layout, in inches.
X0, X1, Y0, Y1 = 0.58, 12.78, 1.6, 6.8

MINUS = "\u2212"
NBSP = "\u00a0"


# =========================================================================== numbers
def pct(x: float, d: int = 1) -> str:
    """0.301 → '30.1%' with a true minus sign."""
    s = f"{100 * x:.{d}f}%"
    return s.replace("-", MINUS)


def num(x: float, d: int = 1) -> str:
    """Number with a true minus sign."""
    return f"{x:.{d}f}".replace("-", MINUS)


def lab(label: str, text: str) -> list[tuple[str, bool, RGBColor]]:
    """A bold navy lead-in followed by body text, as runs for ``_write``."""
    return [(label, True, NAVY), (text, False, INK)]


def ci_txt(r: pd.Series, as_pct: bool = True) -> str:
    """'95% CI [lo, hi]' for a result row."""
    if as_pct:
        return f"95% CI {ci_pct(r['ci_low'], r['ci_high'])}"
    return f"95% CI [{num(r['ci_low'])}, {num(r['ci_high'])}]"


def ci_pct(lo: float, hi: float, d: int = 1) -> str:
    """'[21.9%, 38.3%]'."""
    return f"[{pct(lo, d)}, {pct(hi, d)}]"


@dataclass
class Numbers:
    """Lookups into the generated result tables."""

    stats: pd.DataFrame
    research: pd.DataFrame
    rclaims: pd.DataFrame
    pclaims: pd.DataFrame
    years: pd.DataFrame
    rolling: pd.DataFrame
    breakeven: pd.DataFrame
    monthly: pd.DataFrame
    sanity: pd.DataFrame
    rs: research.ResearchSettings
    sfx: str = ""  # figure/table suffix of this sample ("" = full 1990–2025)
    full: Numbers | None = None  # the full-sample numbers, when this is the headline sample

    def png(self, name: str) -> str:
        """File name of a figure for this sample, e.g. '07_test1_mean_ci_post2003.png'."""
        return f"{name}{self.sfx}.png"

    @property
    def since(self) -> int:
        """First year of this sample."""
        return int(self.monthly.index[0].year)

    @property
    def years_span(self) -> float:
        """Sample length in years (months / 12)."""
        return len(self.monthly) / 12

    def row(
        self,
        test: str,
        measure: str,
        statistic: str,
        sample: str | None = None,
        method: str | None = None,
    ) -> pd.Series:
        """Exactly one result row (stats or research tables)."""
        t = self.research if test.startswith("R_") else self.stats
        m = (t["test"] == test) & (t["measure"] == measure) & (t["statistic"] == statistic)
        if sample is not None:
            m &= t["sample"] == sample
        if method is not None:
            m &= t["ci_method"].fillna("").str.contains(method, regex=False)
        hit = t[m]
        if len(hit) != 1:
            raise KeyError(f"{test}/{measure}/{statistic}/{sample}/{method}: {len(hit)} rows")
        return hit.iloc[0]

    def rclaim(self, cid: str) -> pd.Series:
        """One research claim row by id (e.g. 'R-9')."""
        return self.rclaims.set_index("id").loc[cid]

    def pclaim(self, cid: str) -> pd.Series:
        """One v2 claim row by id (e.g. 'C2')."""
        return self.pclaims.set_index("id").loc[cid]

    def case(self, month: str, statistic: str) -> float:
        """One case-study number (e.g. '2020-03', 'premium_gone_by_day')."""
        return float(self.row("R_case", "case_study", statistic, sample=month)["estimate"])


def load_numbers(cfg: Config, headline: bool = True) -> Numbers:
    """Read every generated table the deck quotes.

    ``headline=True`` (the pitch) reads the headline-sample tables (``headline:`` in
    config.yaml, the live-VIX era) and attaches the full-sample numbers as ``.full``.
    """
    from vrp.pipeline import read_processed  # local import: pipeline imports this module

    sfx = cfg["headline"]["suffix"] if headline else ""

    def name(f: str) -> str:
        p = Path(f)
        return f"{p.stem}{sfx}{p.suffix}"

    rep = cfg.path("reports")
    ro = cfg["research"]["outputs"]
    panel = read_processed(cfg, "daily_panel")
    monthly = read_processed(cfg, "monthly_sample")
    if headline:
        monthly = monthly[monthly[cfg["headline"]["flag"]].astype(bool)]
    return Numbers(
        stats=pd.read_csv(rep / name(cfg["stats"]["outputs"]["tables"])),
        research=pd.read_csv(rep / name(ro["tables"])),
        rclaims=pd.read_csv(rep / name(ro["claims"].replace(".md", ".csv"))),
        pclaims=pd.read_csv(rep / name("pitch_claims.csv")),
        years=pd.read_csv(rep / name(ro["rv_by_year"]), index_col=0),
        rolling=pd.read_csv(rep / name(ro["rolling"]), index_col=0, parse_dates=[0]),
        breakeven=pd.read_csv(rep / name(ro["breakeven"]), dtype={"worst_month": str}),
        monthly=monthly,
        sanity=validate.sanity_table(panel, read_processed(cfg, "monthly_sample"), cfg),
        rs=research.ResearchSettings.from_config(cfg),
        sfx=sfx,
        full=load_numbers(cfg, headline=False) if headline else None,
    )


def p_short(p: float) -> str:
    """p-value for a slide: 'p < 0.001' or 'p = 0.034'."""
    return f"p{NBSP}<{NBSP}0.001" if p < 0.001 else f"p{NBSP}={NBSP}{p:.3f}"


def month_label(m: str) -> str:
    """'2020-03' → 'Mar 2020' (non-breaking space)."""
    return pd.Period(m, "M").strftime("%b") + NBSP + m[:4]


# =========================================================================== pptx helpers
def _next_id(slide) -> int:
    ids = [int(e.get("id")) for e in slide.shapes._spTree.iter() if e.tag == qn("p:cNvPr")]
    return max(ids + [1]) + 1


def _clear_slides(prs: Presentation) -> None:
    """Remove the template's example slides (layouts, theme and master stay)."""
    lst = prs.slides._sldIdLst
    for sld in list(lst):
        prs.part.drop_rel(sld.rId)
        lst.remove(sld)


def _add_footer(slide, date_text: str, footer_text: str) -> None:
    """Copy the layout's date / footer / slide-number placeholders onto the slide."""
    kinds = {PP_PLACEHOLDER.DATE, PP_PLACEHOLDER.FOOTER, PP_PLACEHOLDER.SLIDE_NUMBER}
    for ph in slide.slide_layout.placeholders:
        kind = ph.placeholder_format.type
        if kind not in kinds:
            continue
        el = copy.deepcopy(ph._element)
        el.find(qn("p:nvSpPr")).find(qn("p:cNvPr")).set("id", str(_next_id(slide)))
        slide.shapes._spTree.append(el)
        if kind != PP_PLACEHOLDER.SLIDE_NUMBER:
            shape = slide.placeholders[ph.placeholder_format.idx]
            runs = [r for p in shape.text_frame.paragraphs for r in p.runs]
            if runs:
                runs[0].text = date_text if kind == PP_PLACEHOLDER.DATE else footer_text
                for r in runs[1:]:
                    r.text = ""
            else:
                shape.text_frame.text = date_text if kind == PP_PLACEHOLDER.DATE else footer_text


def _prune(slide, keep: set[int]) -> None:
    """Delete placeholders the slide does not use (so no empty prompts remain)."""
    for ph in list(slide.placeholders):
        f = ph.placeholder_format
        if f.idx not in keep and f.type not in (
            PP_PLACEHOLDER.DATE,
            PP_PLACEHOLDER.FOOTER,
            PP_PLACEHOLDER.SLIDE_NUMBER,
        ):
            ph._element.getparent().remove(ph._element)


def _style_run(run, size: float, bold: bool = False, color: RGBColor = INK) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Arial"


def _set_bullet(p, level: int = 0, bullet: bool = True, space_before: float = 6) -> None:
    """Paragraph indent + a real bullet character (never a literal '•' in the text)."""
    pPr = p._p.get_or_add_pPr()
    for tag in ("a:lnSpc", "a:spcBef", "a:spcAft", "a:buClr", "a:buChar", "a:buNone"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    marl = 0.22 + 0.25 * level
    pPr.set("marL", str(Inches(marl) if bullet else 0))
    pPr.set("indent", str(-Inches(0.2) if bullet else 0))
    pPr.set("lvl", str(level))
    spc = pPr.makeelement(qn("a:spcBef"), {})
    pts = spc.makeelement(qn("a:spcPts"), {"val": str(int(space_before * 100))})
    spc.append(pts)
    children = [spc]
    if bullet:
        clr = pPr.makeelement(qn("a:buClr"), {})
        clr.append(clr.makeelement(qn("a:srgbClr"), {"val": str(BLUE)}))
        children += [clr, pPr.makeelement(qn("a:buChar"), {"char": "•"})]
    else:
        children.append(pPr.makeelement(qn("a:buNone"), {}))
    for i, c in enumerate(children):
        pPr.insert(i, c)


Run = tuple[str, bool, RGBColor]  # (text, bold, colour)


def _write(
    tf,
    items: list[str | list[Run]],
    size: float = 14,
    bullet: bool = True,
    space_before: float = 7,
    first_level: int = 0,
) -> None:
    """Write paragraphs; each item is a string or a list of styled runs."""
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        runs = [(item, False, INK)] if isinstance(item, str) else item
        for text, bold, color in runs:
            _style_run(p.add_run(), size, bold, color)
            p.runs[-1].text = text
        _set_bullet(p, first_level, bullet, space_before if i else 0)


def _textbox(slide, x: float, y: float, w: float, h: float, name: str):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    return tb


def _title(slide, text: str, bar: str | None = None, bar_idx: int = 3) -> None:
    """Fill the title and the template's blue key-message bar."""
    slide.shapes.title.text_frame.text = text
    if bar is not None:
        tf = slide.placeholders[bar_idx].text_frame
        tf.text = ""
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = bar
        _style_run(r, 16, False, BLUE)


SOURCE_NOTE = ""  # set by build_deck: the data source and sign convention under each chart
CAPTION_H = 0.24  # inches reserved under a chart for that line
LABEL_H = 0.3  # inches for the "Exhibit N: …" label above a chart
FRAME = RGBColor(0xD0, 0xD0, 0xD0)
# Exhibit labels (reference-deck style), keyed by the figure number at the start of its name.
EXHIBITS = {
    "02": "The VIX at the start of each month vs the volatility that followed",
    "06": "Daily observations per year, by source",
    "07": "Average seller payoff with 95% confidence intervals, three ways",
    "09": "Seller payoff by S&P 500 decile, and against the S&P move",
    "10": "Does the VIX forecast realised variance?",
    "11": "Average payoff across robustness checks",
    "12": "Monthly realised volatility and its autocorrelation",
    "13": "Realised volatility and vol-of-vol by year",
    "14": "Seller payoff by conditions known at the start of the month",
    "15": "Five-year rolling premium with 95% band",
    "16": "Years between crashes: needed to break even vs observed",
    "17": "March 2020 and April 2025, day by day",
}
_EXHIBIT_N = [0]  # running exhibit number, reset by build_deck


def _picture(slide, path: Path, x: float, y: float, w: float, h: float, alt: str) -> tuple:
    """Insert a PNG scaled to fit the box (aspect kept), centred horizontally, top-aligned.

    The slide versions of the charts carry no footnote, so the source line goes underneath.
    """
    iw, ih = Image.open(path).size
    label = EXHIBITS.get(path.stem[:2])
    top = LABEL_H if label else 0.0
    h_img = h - top - 0.06 - (CAPTION_H if SOURCE_NOTE else 0)
    scale = min(w / iw, h_img / ih)
    pw, ph = iw * scale, ih * scale
    px = x + (w - pw) / 2
    if label:  # thin frame around label + chart, then the label itself
        _EXHIBIT_N[0] += 1
        fr = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(px - 0.05),
            Inches(y),
            Inches(pw + 0.1),
            Inches(top + ph + 0.06),
        )
        fr.name = f"exhibit frame {_EXHIBIT_N[0]}"
        fr.fill.background()
        fr.line.color.rgb = FRAME
        fr.line.width = Pt(0.75)
        _plain(fr)
        lb = _textbox(slide, px, y + 0.02, pw, LABEL_H, f"exhibit {_EXHIBIT_N[0]}")
        lb.text_frame.margin_top = lb.text_frame.margin_bottom = 0
        r = lb.text_frame.paragraphs[0].add_run()
        r.text = f"Exhibit {_EXHIBIT_N[0]}: {label}"
        fits = pw * 72 / (0.56 * len(r.text))  # largest size that keeps the label on one line
        _style_run(r, max(8.0, min(11.0, fits)), True, INK)
    pic = slide.shapes.add_picture(str(path), Inches(px), Inches(y + top), Inches(pw), Inches(ph))
    pic.name = path.stem
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    if SOURCE_NOTE:
        tb = _textbox(slide, px - 0.05, y + top + ph + 0.08, pw + 0.1, CAPTION_H, "source")
        r = tb.text_frame.paragraphs[0].add_run()
        r.text = SOURCE_NOTE
        _style_run(r, 8.5, False, MUTED)
    return px, y + top, pw, ph


def _est_height(items: list, w: float, size: float, space: float = 5) -> float:
    """Rough height (inches) of bullet paragraphs in a box of width ``w`` (Arial)."""
    cpl = max(10.0, (w - 0.45) * 72 / (0.5 * size))  # characters per line
    h = 0.0
    for it in items:
        text = it if isinstance(it, str) else "".join(t for t, _, _ in it)
        h += math.ceil(len(text) / cpl) * size * 1.22 / 72 + space / 72
    return h


def _block(slide, x: float, y: float, w: float, h: float, head: str, items: list, size: float):
    """Reference-deck text block: short navy bar + header, bullets underneath."""
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.05), Inches(0.08), Inches(0.28)
    )
    bar.name = f"bar {head}"
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    _plain(bar)
    hb = _textbox(slide, x + 0.14, y, w - 0.14, 0.38, f"head {head}")
    r = hb.text_frame.paragraphs[0].add_run()
    r.text = head
    _style_run(r, size + 1.5, True, INK)
    tb = _textbox(slide, x + 0.1, y + 0.4, w - 0.1, max(h - 0.4, 0.3), f"text {head}")
    _write(tb.text_frame, items, size=size, space_before=5)


def _blocks(
    slide,
    x: float,
    y: float,
    w: float,
    groups: list[tuple[str, list]],
    size: float = 13,
    gap: float = 0.2,
) -> float:
    """Stack text blocks top-down from ``y``; returns the y below the last one."""
    for head, items in groups:
        h = 0.42 + _est_height(items, w - 0.1, size)
        _block(slide, x, y, w, h + 0.1, head, items, size)
        y += h + gap
    return y


def _plain(shape) -> None:
    """Drop the default shape style (its shadow and effects) from an autoshape."""
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def _card(slide, x: float, y: float, w: float, h: float, name: str):
    s = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    s.name = name
    s.adjustments[0] = 0.06
    s.fill.solid()
    s.fill.fore_color.rgb = CARD
    s.line.fill.background()
    _plain(s)
    return s


def _stat(slide, x: float, y: float, w: float, big: str, label: str, sub: str = "") -> None:
    """Big-number callout on a tinted card."""
    _card(slide, x, y, w, 1.45, f"stat card {big}")
    tb = _textbox(slide, x + 0.15, y + 0.08, w - 0.3, 1.3, f"stat {big}")
    tf = tb.text_frame
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = big
    _style_run(r, 30, True, NAVY)
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = label
    _style_run(r2, 12, False, INK)
    if sub:
        p3 = tf.add_paragraph()
        r3 = p3.add_run()
        r3.text = sub
        _style_run(r3, 10, False, MUTED)


def _todo_box(slide, x: float, y: float, w: float, h: float, head: str, lines: list[str]) -> None:
    """Dashed box marking content that is not built yet."""
    s = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    s.name = f"pending: {head}"
    s.adjustments[0] = 0.04
    s.fill.background()
    s.line.color.rgb = BLUE
    s.line.width = Pt(1.25)
    s.line.dash_style = MSO_LINE.DASH
    _plain(s)
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.3)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = head
    _style_run(r, 16, True, NAVY)
    for line in lines:
        q = tf.add_paragraph()
        q.alignment = PP_ALIGN.LEFT
        rr = q.add_run()
        rr.text = line
        _style_run(rr, 13, False, INK)


def _caption(slide, x: float, y: float, w: float, text: str) -> None:
    tb = _textbox(slide, x, y, w, 0.3, "caption")
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = text
    _style_run(r, 10, False, MUTED)


def _restyle_titles(prs, cfg: Config) -> None:
    """Swap the template's display font for ``deck.style.title_font`` and shrink the titles.

    Done on the master and layouts (not slide by slide), so slides added later in PowerPoint or
    Google Slides match.
    """
    st = cfg["deck"]["style"]
    old, new = st["replace_font"], st["title_font"]
    for part in [prs.slide_master, *prs.slide_layouts]:
        el = part.element
        for tag in ("a:latin", "a:ea", "a:cs", "a:sym", "a:buFont"):
            for e in el.iter(qn(tag)):
                if e.get("typeface") == old:
                    e.set("typeface", new)
        size = st["cover_title_size"] if part is prs.slide_layouts[L_TITLE] else st["title_size"]
        for sp in el.iter(qn("p:sp")):
            ph = sp.find(".//" + qn("p:ph"))
            if ph is not None and ph.get("type") in ("title", "ctrTitle"):
                for r in sp.iter(qn("a:defRPr")):
                    if r.get("sz"):
                        r.set("sz", str(int(size * 100)))


def _content_slide(prs, cfg: Config, keep: set[int] | None = None):
    s = prs.slides.add_slide(prs.slide_layouts[L_CONTENT])
    _prune(s, keep if keep is not None else {0, 3})
    _add_footer(s, cfg["deck"]["date_text"], cfg["deck"]["footer_text"])
    return s


# =========================================================================== slides
def slide_title(prs, cfg: Config) -> None:
    """Title slide."""
    d = cfg["deck"]
    s = prs.slides.add_slide(prs.slide_layouts[L_TITLE])
    _prune(s, {0, 1})
    s.shapes.title.text_frame.text = d["title"]
    _write(s.placeholders[1].text_frame, d["subtitle_lines"], size=16, bullet=False)


def slide_agenda(prs, cfg: Config) -> None:
    """Table of contents, on the content layout so the title is a real, editable placeholder."""
    s = _content_slide(prs, cfg, keep={0})
    s.shapes.title.text_frame.text = "Table of Contents"
    items = [
        [(f"{i}.   ", False, BLUE), (t, False, INK)]
        for i, t in enumerate(cfg["deck"]["agenda"], start=1)
    ]
    tb = _textbox(s, X0 + 0.1, Y0 - 0.15, 9.0, 4.8, "agenda")
    _write(tb.text_frame, items, size=22, bullet=False, space_before=14)


def slide_i1(prs, cfg: Config, n: Numbers, fig: Path) -> None:
    """I1: what the VIX measures."""
    s = _content_slide(prs, cfg)
    _title(
        s,
        "What Does the VIX Measure?",
        "The VIX prices next month's S&P 500 variance; selling it pays when markets stay calm",
    )
    _blocks(
        s,
        X0,
        Y0 + 0.1,
        4.1,
        [
            (
                "What the VIX is",
                [
                    "The market's 30-day forecast of S&P 500 volatility, read from S&P "
                    "options prices.",
                    "VIX² is the price of next month's variance. Selling it is a bet that the "
                    "market will be calmer than options imply.",
                ],
            ),
            (
                "How we use it",
                [
                    f"Each month since {n.monthly.index[0]:%b %Y} (the live VIX), we compare "
                    "what the "
                    "VIX implied with the variance the "
                    "S&P actually delivered over the next 21 trading days.",
                    [
                        ("Sign: implied − realised. ", True, NAVY),
                        ("Positive means the VIX over-forecast and the seller won.", False, INK),
                    ],
                ],
            ),
        ],
        size=13,
    )
    _picture(
        s,
        fig / n.png("02_vix_vs_realised_vol"),
        4.85,
        Y0 + 0.1,
        7.9,
        4.9,
        "VIX vs subsequently realised S&P 500 volatility, monthly 1990–2025",
    )
    jm = n.row("R_jensen", "vol_spread_pts", "mean")
    ja = n.row("R_jensen", "vol_spread_pts", "share_vix_above_realised")
    jf = n.row("R_jensen", "rv_fwd21", "fair_vix_share_above_realised")
    jc = n.row("R_jensen", "rv_fwd21", "cond_convexity_share_of_spread")
    s.notes_slide.notes_text_frame.text = (
        "Jensen's inequality: the VIX prices variance, so even a VIX with no premium would sit "
        f"above realised vol in about {pct(jf['estimate'], 0)} of months; it actually does in "
        f"{pct(ja['estimate'], 0)}. Convexity explains about {pct(jc['estimate'], 0)} "
        f"{ci_pct(jc['ci_low'], jc['ci_high'], 0)} of the average {num(jm['estimate'])}-point "
        "gap; the rest is premium. That is why we test in variance terms."
    )


def slide_i2(prs, cfg: Config, n: Numbers, fig: Path) -> None:
    """I2: data and validation (sanity table)."""
    s = _content_slide(prs, cfg)
    _title(s, "Data and Validation", "Our data reproduce published benchmarks without adjustment")
    rows = n.sanity[n.sanity["Status"] == "✅"]
    labels = {
        0: "Mean VIX, 1990–2024",
        1: "Mean realised vol, 1990–2024",
        2: "Monthly observations, 1990–2025",
        4: "Highest VIX close",
    }
    data = [("Check", "Ours", "Benchmark", "")] + [
        (
            labels.get(i, r["Item"]),
            str(r["Ours"]).replace(" on ", ", "),
            str(r["Benchmark"]).replace(" on ", ", "),
            "Match",
        )
        for i, r in rows.iterrows()
    ]
    x, y, w, rh = X0 + 0.1, Y0 + 0.1, 6.9, 0.48
    gt = s.shapes.add_table(len(data), 4, Inches(x), Inches(y), Inches(w), Inches(rh * len(data)))
    gt.name = "sanity table"
    tbl = gt.table
    for j, cw in enumerate([2.65, 1.75, 1.75, 0.75]):
        tbl.columns[j].width = Inches(cw)
    for i in range(len(data)):
        tbl.rows[i].height = Inches(rh)
    for i, rowv in enumerate(data):
        for j, v in enumerate(rowv):
            c = tbl.cell(i, j)
            c.text_frame.text = ""
            c.text_frame.paragraphs[0].add_run().text = v
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (CARD if i % 2 == 0 else WHITE)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            para = c.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            _style_run(para.runs[0], 13, i == 0, WHITE if i == 0 else (GREEN if j == 3 else INK))
    _caption(
        s,
        x,
        y + rh * len(data) + 0.08,
        w,
        "Benchmarks: CFA Institute Enterprising Investor, 31 Jul 2024. Nothing adjusted to "
        "match. Checks use the full 1990–2025 dataset.",
    )
    _blocks(
        s,
        7.75,
        Y0 + 0.1,
        4.95,
        [
            (
                "Data",
                [
                    lab("Sources: ", "VIX from Cboe via FRED; S&P 500 from Yahoo Finance."),
                    lab(
                        "Sample: ",
                        f"{len(n.monthly)} months, {n.monthly.index[0]:%b %Y} to "
                        f"{n.monthly.index[-1]:%b %Y} (the live VIX), read on the first "
                        "trading day "
                        "of each month.",
                    ),
                ],
            ),
            (
                "Rules",
                [
                    lab(
                        "No look-ahead: ",
                        "the VIX is read at the close of day t; realised variance uses only days "
                        "t+1 to t+21.",
                    ),
                    lab(
                        "Why 2003: ",
                        "the VIX before Sept 2003 was back-calculated by Cboe, not what "
                        "traders saw, so our evidence starts with the live VIX. The longer "
                        "history is in the backup.",
                    ),
                ],
            ),
        ],
        size=13,
    )


def slide_r1(prs, cfg: Config, n: Numbers, fig: Path) -> None:
    """R1: the VIX over-forecasts on average."""
    win = n.row("2_shape", "payoff_scaled", "win_rate", method="Wilson")
    mean = n.row("1_mean", "payoff_scaled", "mean", method="Newey")
    med = n.row("1_mean", "payoff_scaled", "median")
    spr = n.row("1_mean", "vol_spread_pts", "mean", method="Newey")
    c2 = n.pclaim("C2")
    s = _content_slide(prs, cfg)
    _title(
        s,
        "Does the VIX Over-Forecast Volatility?",
        f"Yes: since {n.since} the variance seller wins in {pct(win['estimate'], 0)} of months, "
        f"about {round(6 * win['estimate'])} in 6",
    )
    w, gap = 2.9, 0.2
    stats = [
        (pct(win["estimate"]), "of months the seller wins", ci_txt(win)),
        (pct(mean["estimate"]), "average payoff (% of implied var.)", ci_txt(mean)),
        (pct(med["estimate"]), "payoff in the median month", ci_txt(med)),
        (f"{num(spr['estimate'])} pts", "VIX above the vol that followed", ci_txt(spr, False)),
    ]
    for i, (big, label, sub) in enumerate(stats):
        _stat(s, X0 + i * (w + gap) + 0.05, Y0 + 0.05, w, big, label, sub)
    _picture(
        s,
        fig / n.png("07_test1_mean_ci"),
        X0,
        Y0 + 1.7,
        X1 - X0,
        3.45,
        "Mean seller payoff with 95% confidence intervals under three standard-error methods",
    )
    s.notes_slide.notes_text_frame.text = (
        f"Live-VIX era ({n.since}–{n.monthly.index[-1].year}). In raw variance the mean is "
        + ("significant" if c2["supported"] else "positive but not significant")
        + "; the scaled payoff is the headline measure. CIs: Newey–West 6 lags / 6-month "
        "block bootstrap. Longer back-calculated history: backup slide."
    )


def slide_r2(prs, cfg: Config, n: Numbers, fig: Path) -> None:
    """R2: losses are rare, large, and come in sell-offs."""
    skew = n.row("2_shape", "payoff_scaled", "skewness")
    share = n.row("3_crash", "payoff_var", "loss_share_in_decile_1")
    sell = n.row("3_crash", "payoff_scaled", "mean_in_selloff")
    worst = n.monthly.nsmallest(5, "payoff_scaled")
    worst_txt = ", ".join(
        f"{pd.Timestamp(r.month):%b}{NBSP}{pd.Timestamp(r.month):%Y} ({num(r.payoff_scaled)}×)"
        for r in worst.itertuples()
    )
    s = _content_slide(prs, cfg)
    _title(
        s,
        "How Bad Are the Losing Months?",
        "Losses are rare but violent, and they arrive in stock-market sell-offs",
    )
    _blocks(
        s,
        X0,
        Y0 + 0.1,
        4.05,
        [
            (
                "Rare but violent",
                [
                    lab(
                        "Lopsided: ",
                        f"skewness {num(skew['estimate'])} [{num(skew['ci_low'])}, "
                        f"{num(skew['ci_high'])}]. Many small gains, a few huge losses.",
                    ),
                    lab("Worst months ", f"(loss as a multiple of implied variance): {worst_txt}."),
                ],
            ),
            (
                "Clustered in sell-offs",
                [
                    lab(
                        "Clustered: ",
                        f"the worst 10% of S&P months carry {pct(share['estimate'])} of all seller "
                        f"losses {ci_pct(share['ci_low'], share['ci_high'])}; "
                        "with no clustering, about 10%.",
                    ),
                    lab(
                        "In sell-offs ",
                        f"(S&P down > 5%) the seller loses {pct(-sell['estimate'], 0)} of implied "
                        "variance on average.",
                    ),
                ],
            ),
        ],
        size=12.5,
    )
    _picture(
        s,
        fig / n.png("09_test3_selloff"),
        4.75,
        Y0 + 0.6,
        8.05,
        4.4,
        "Seller payoff by S&P 500 return decile; losses concentrate in the worst decile",
    )
    s.notes_slide.notes_text_frame.text = (
        "Conditioning on sell-offs describes when losses happen; it is not a forecast made at t."
    )


def slide_r3(prs, cfg: Config, n: Numbers, fig: Path) -> None:
    """R3: the two worst recent months, day by day."""
    cases = list(n.rs.case_studies)
    gone = max(n.case(m, "premium_gone_by_day") for m in cases)
    s = _content_slide(prs, cfg)
    _title(
        s,
        "Case Study: March 2020 vs April 2025",
        f"In both months the seller's whole premium was gone within {gone:.0f} trading days",
    )
    _picture(
        s,
        fig / n.png("17_case_studies"),
        X0,
        Y0 + 0.05,
        7.1,
        4.9,
        "Daily S&P 500, VIX and realised variance accrued against the implied variance sold, "
        "for March 2020 and April 2025",
    )
    groups: list = []
    for m in cases:
        vix, rv = n.case(m, "vix_at_t"), n.case(m, "realised_vol_fwd")
        rec = n.case(m, "months_to_recover")
        rec_txt = f"won back in {rec:.0f} months" if np.isfinite(rec) else "not yet won back"
        groups.append(
            (
                month_label(m).replace(NBSP, " "),
                [
                    f"VIX {vix:.0f} at the start (higher than "
                    f"{pct(n.case(m, 'vix_at_t_percentile'), 0)} of months), realised vol "
                    f"{rv:.0f}, {rv / vix:.1f}× the VIX.",
                    f"Premium gone by day {n.case(m, 'premium_gone_by_day'):.0f}; the loss "
                    f"equalled {n.case(m, 'months_of_premium_lost'):.0f} months of average "
                    f"premium, {rec_txt}.",
                ],
            )
        )
    last = cases[-1]
    groups.append(
        (
            "The path matters, not the end point",
            [
                f"In {month_label(last)} the S&P fell "
                f"{pct(-n.case(last, 'spx_max_drawdown_window'), 0)} inside the window yet "
                f"ended {pct(n.case(last, 'spx_return_window'), 1)}; the seller still lost, "
                "because variance accrues every day."
            ],
        )
    )
    _blocks(s, 7.85, Y0 + 0.05, 4.9, groups, size=12, gap=0.15)
    s.notes_slide.notes_text_frame.text = (
        "Descriptive: these are outcomes after t, not signals. A high VIX at the start was no "
        "protection in 2020. Numbers: research_summary.md, section D (case studies)."
    )


def slide_r4(prs, cfg: Config, n: Numbers, fig: Path) -> None:
    """R4: realised-vol dynamics."""
    ar = n.row("R_dynamics", "log_rv_cal", "ar1_rho")
    hl = n.row("R_dynamics", "log_rv_cal", "half_life_months")
    tr = n.row("R_dynamics", "log_rv_cal", "trend_slope_per_year")
    top = n.years.sort_values("rank_irregular").head(5)
    calm = top["rv_ann_vol_pct"].idxmin()  # most irregular year among the calmest on average
    s = _content_slide(prs, cfg)
    _title(
        s,
        "How Does Realised Volatility Behave?",
        "It clusters, then fades within a couple of months; there is no long-run upward trend",
    )
    _picture(
        s,
        fig / n.png("12_rv_dynamics"),
        X0,
        Y0 + 0.05,
        X1 - X0,
        3.4,
        f"Calendar-month realised volatility since {n.since} with fitted trend, and its "
        "autocorrelation",
    )
    col = (X1 - X0 - 0.4) / 3
    blocks = [
        (
            "Vol clusters",
            "A high-vol month tends to be followed by another: AR(1) "
            f"{num(ar['estimate'], 2)} [{num(ar['ci_low'], 2)}, {num(ar['ci_high'], 2)}].",
        ),
        (
            "But shocks fade fast",
            f"Half-life about {num(hl['estimate'])} months; vol always returns to its normal "
            f"level (unit root rejected). No trend since {n.since} (p{NBSP}={NBSP}"
            f"{tr['p_value']:.2f}).",
        ),
        (
            "Most irregular years",
            ", ".join(str(y) for y in top.index)
            + f". {calm} swung hard despite only "
            + f"{num(top.loc[calm, 'rv_ann_vol_pct'], 0)}% average vol.",
        ),
    ]
    for i, (head, body) in enumerate(blocks):
        _block(s, X0 + i * (col + 0.2), Y0 + 3.65, col, 1.5, head, [body], size=12.5)


def slide_r5(prs, cfg: Config, n: Numbers, fig: Path) -> None:
    """R5: premium by conditions known at t."""
    gl = research.bucket_labels("gap", n.rs.gap_thresholds, " pts")
    vl = research.bucket_labels("VIX", n.rs.vix_thresholds, "")
    g_hi = n.row("R_conditional", "payoff_scaled", "mean", sample=gl[2])
    g_lo = n.row("R_conditional", "payoff_scaled", "mean", sample=gl[0])
    g_d = n.row(
        "R_conditional",
        "payoff_scaled",
        "diff_mean_high_minus_low",
        sample=f"{gl[2]} minus {gl[0]}",
    )
    v = [n.row("R_conditional", "payoff_scaled", "mean", sample=lb) for lb in vl]
    wins = [
        n.row("R_conditional", "payoff_scaled", "win_rate", sample=lb)["estimate"] for lb in vl + gl
    ]
    s = _content_slide(prs, cfg)
    _title(
        s,
        "When Is the Premium Largest?",
        "Hard to tell in advance: simple signals point the expected way but are not significant",
    )
    _picture(
        s,
        fig / n.png("14_conditional_premium"),
        X0,
        Y0 + 0.05,
        X1 - X0,
        3.6,
        "Mean seller payoff by VIX level and by VIX minus past realised vol, with 95% CIs",
    )
    _block(
        s,
        X0,
        Y0 + 3.75,
        X1 - X0,
        1.5,
        "What we found",
        [
            lab(
                "By VIX level: ",
                f"{pct(v[0]['estimate'], 0)}, {pct(v[1]['estimate'], 0)} and "
                f"{pct(v[2]['estimate'], 0)} of implied variance (low, mid, high VIX). "
                "High-VIX months hold the biggest crashes.",
            ),
            lab(
                "VIX far above recent vol: ",
                f"{pct(g_hi['estimate'], 0)} vs {pct(g_lo['estimate'], 0)} when below; "
                f"difference +{pct(g_d['estimate'])} {ci_pct(g_d['ci_low'], g_d['ci_high'])}, "
                "not significant.",
            ),
            lab(
                "Seller wins ",
                f"{pct(min(wins), 0)}–{pct(max(wins), 0)} of months in every bucket; what differs "
                "is the size of the losses. Whether a model can do better is the next section.",
            ),
        ],
        12.5,
    )


def slide_r6(prs, cfg: Config, n: Numbers, fig: Path) -> None:
    """R6: is the premium shrinking, and the break-even test (live-VIX era only)."""
    alpha = float(cfg["stats"]["alpha"])
    tr = n.row("R_time", "payoff_scaled", "trend_slope_per_year")
    c9 = n.rclaim("R-9")
    roll = n.rolling["mean"]
    s = _content_slide(prs, cfg)
    _title(
        s,
        "Is the Premium Shrinking?",
        (
            f"No: since {n.since} there is no trend, just deep dips after crashes"
            if not c9["supported"] and tr["p_value"] >= alpha
            else f"The premium has fallen since {n.since}"
        ),
    )
    _picture(
        s,
        fig / n.png("15_rolling_premium"),
        X0,
        Y0 + 0.05,
        6.9,
        4.9,
        f"Rolling {int(n.rolling['window_months'].iloc[0])}-month mean of the scaled seller "
        "payoff, with a 95% band and linear trend",
    )
    _block(
        s,
        7.7,
        Y0 + 0.05,
        5.0,
        2.2,
        f"Since {n.since}",
        [
            lab(
                "No trend: ",
                f"{pct(tr['estimate'], 2)} per year (p{NBSP}={NBSP}{tr['p_value']:.2f}).",
            ),
            lab(
                "5-year premium: ",
                f"{pct(roll.iloc[0], 0)} in the window ending "
                f"{roll.index[0]:%b}{NBSP}{roll.index[0]:%Y}, {pct(roll.min(), 0)} at its low "
                f"(window ending {roll.idxmin():%b}{NBSP}{roll.idxmin():%Y}), "
                f"{pct(roll.iloc[-1], 0)} now.",
            ),
        ],
        13,
    )
    be = n.row("R_breakeven", "payoff_scaled", "breakeven_years_between")
    be_v = n.row("R_breakeven", "payoff_var", "breakeven_years_between")
    c10, c11 = n.rclaim("R-10"), n.rclaim("R-11")
    sig = (
        f"Significant ({p_short(c10['p_holm'])} after correction)."
        if c10["supported"]
        else f"Suggestive, not proven: one crash in {be['null_value']:.0f} years is too few "
        f"({p_short(c10['p_holm'])} after correction)."
    )
    _card(s, 7.7, Y0 + 2.45, 5.0, 2.5, "card break-even")
    tb2 = _textbox(s, 7.85, Y0 + 2.5, 4.7, 2.4, "text break-even")
    _write(
        tb2.text_frame,
        [
            [("Too pessimistic? The break-even test", True, NAVY)],
            [
                (f"A {_worst_label(n)}-sized loss would have to hit every ", False, INK),
                (f"{num(be['estimate'])} years", True, NAVY),
                (
                    f" [{num(be['ci_low'])}, {num(be['ci_high'])}] to wipe out the premium; "
                    f"there has been one in {be['null_value']:.0f} years.",
                    False,
                    INK,
                ),
            ],
            [(sig, False, INK)],
            [
                (
                    f"In raw variance: every {num(be_v['estimate'])} years; "
                    + ("significant." if c11["supported"] else "not significant.")
                    + " See appendix.",
                    False,
                    MUTED,
                )
            ],
        ],
        size=12,
        bullet=False,
        space_before=4,
    )


def _worst_month(n: Numbers) -> str:
    """'2020-03': the worst scaled-payoff month (the break-even scenario)."""
    c = n.breakeven
    return str(c.loc[(c["measure"] == "payoff_scaled") & c["is_worst"], "worst_month"].iloc[0])


def _worst_label(n: Numbers) -> str:
    """Month label of the worst scaled-payoff month (the break-even scenario)."""
    c = n.breakeven
    return month_label(
        str(c.loc[(c["measure"] == "payoff_scaled") & c["is_worst"], "worst_month"].iloc[0])
    )


def slide_team(prs, cfg: Config, title: str, bar: str, who: str, lines: list[str]) -> None:
    """Placeholder for a teammates' section."""
    s = _content_slide(prs, cfg)
    _title(s, title, bar)
    _todo_box(s, X0 + 0.1, Y0 + 0.3, X1 - X0 - 0.2, 4.6, who, lines)


SMALL_NUMBERS = ["zero", "one", "two", "three", "four", "five", "six"]


def out_of_six(share: float, tol: float = 0.03) -> str | None:
    """'five months out of six' when ``share`` is within ``tol`` of k/6, else None."""
    k = round(6 * share)
    return f"{SMALL_NUMBERS[k]} months out of six" if abs(k / 6 - share) <= tol else None


def slide_verdict(prs, cfg: Config, n: Numbers) -> None:
    """Provisional verdict: headline, research column filled, teammates' columns pending."""
    win = n.row("2_shape", "payoff_scaled", "win_rate", method="Wilson")
    mean = n.row("1_mean", "payoff_scaled", "mean", method="Newey")
    share = n.row("3_crash", "payoff_var", "loss_share_in_decile_1")
    be = n.row("R_breakeven", "payoff_scaled", "breakeven_years_between")
    proven = bool(n.rclaim("R-10")["supported"])
    raw_mean_ok, raw_be_ok = bool(n.pclaim("C2")["supported"]), bool(n.rclaim("R-11")["supported"])
    years = f"{be['null_value']:.0f}"
    s = _content_slide(prs, cfg)
    _title(
        s,
        "Verdict",
        f"Provisional: since {n.since} the VIX over-prices risk, and so far the premium has "
        "outrun crashes",
    )
    head = _textbox(s, X0, Y0 + 0.05, X1 - X0, 1.0, "headline")
    _write(
        head.text_frame,
        [
            [
                ("Yes, on average. ", True, NAVY),
                (
                    f"Since {n.since} the VIX has over-priced next month's risk, and so far the "
                    "premium has outrun the crashes"
                    + (
                        ". "
                        if proven
                        else f", though with one {_worst_label(n).replace(NBSP, ' ')} in "
                        f"{years} years we can't yet prove it pays for them. "
                    )
                    + "When the losses come, they are brutal.",
                    False,
                    INK,
                ),
            ]
        ],
        size=16,
        bullet=False,
    )
    if raw_mean_ok or raw_be_ok:
        raw_txt = "in raw variance terms the margin is thinner."
    else:
        raw_txt = "in raw variance terms neither the premium nor the margin is significant."
    research = [
        [
            ("Pessimistic on average: ", True, NAVY),
            (
                f"the variance seller wins {pct(win['estimate'], 0)} of months and keeps about "
                f"{pct(mean['estimate'], 0)} of implied variance (95% CI "
                f"{pct(mean['ci_low'], 0)}{NBSP}to{NBSP}{pct(mean['ci_high'], 0)}).",
                False,
                INK,
            ),
        ],
        [
            ("The premium has outrun the crashes: ", True, NAVY),
            (
                f"a {_worst_label(n)}-sized loss would have to hit every "
                f"{num(be['estimate'])} years to wipe it out; there has been one in {years}"
                + ("." if proven else ", too few to prove it."),
                False,
                INK,
            ),
        ],
        [
            ("Brutal when it goes wrong: ", True, NAVY),
            (
                f"{pct(share['estimate'], 0)} of losses come in the worst 10% of months; "
                + raw_txt,
                False,
                INK,
            ),
        ],
        [
            ("No early warning: ", True, NAVY),
            (
                "nothing known at the start of the month reliably predicts the bad months.",
                False,
                INK,
            ),
        ],
    ]
    cols = [
        ("Research", research, 5.4),
        (
            "Fair-price model",
            ["Pending (teammates): is the VIX still too high after paying for crash risk?"],
            3.3,
        ),
        ("Strategy", ["Pending (teammates): does stepping aside cut drawdowns?"], 3.3),
    ]
    gap = (X1 - X0 - sum(w for _, _, w in cols)) / (len(cols) - 1)
    x, top, h = X0, Y0 + 1.2, 3.75
    for head_txt, lines, w in cols:
        _card(s, x, top, w, h, f"card {head_txt}")
        tb = _textbox(s, x + 0.18, top + 0.12, w - 0.36, h - 0.2, f"text {head_txt}")
        _write(tb.text_frame, [[(head_txt, True, NAVY)]] + lines, size=12.5, bullet=False)
        for p in tb.text_frame.paragraphs[1:]:
            _set_bullet(p, 0, True, 7)
        x += w + gap

    often = out_of_six(win["estimate"]) or f"{pct(win['estimate'], 0)} of months"
    proof = (
        ""
        if proven
        else f" With only one crash that size since {n.since}, that's suggestive rather than "
        "proven."
    )
    s.notes_slide.notes_text_frame.text = (
        "So, is the VIX too pessimistic? On average, yes. Since "
        f"{n.since} it has over-forecast risk in {often}, and even after paying for "
        f"{pd.Period(_worst_month(n), 'M'):%B %Y} the premium comes out ahead: a crash that size "
        f"would need to happen every {be['estimate']:.0f} years to wipe it out, and there's been "
        f"one in {years}.{proof} And when the losses come, they come all at once, with no "
        "warning sign beforehand.\n\n"
        "PROVISIONAL until the fair-price model and backtest are in (Tue). If the model says "
        "the VIX is still too high after pricing crash risk: drop 'Provisional'. If it says the "
        "VIX is about fair once crash risk is priced: soften to 'Pessimistic, but mostly fairly "
        "paid for crash risk'."
    )


def slide_appendix(prs, cfg: Config, n: Numbers, fig: Path) -> None:
    """Appendix: Mincer–Zarnowitz (headline sample) and robustness (compares the eras)."""
    s = _content_slide(prs, cfg)
    _title(s, "Appendix: Forecast Bias and Robustness", "Supporting tests for Q&A")
    w2 = (X1 - X0 - 0.3) / 2
    _picture(
        s,
        fig / n.png("10_test4_mincer_zarnowitz"),
        X0,
        Y0 + 0.1,
        w2,
        Y1 - Y0 - 0.1,
        "Mincer–Zarnowitz regression of realised on implied variance",
    )
    _picture(
        s,
        fig / "11_test5_robustness.png",
        X0 + w2 + 0.3,
        Y0 + 0.1,
        w2,
        Y1 - Y0 - 0.1,
        "Mean payoff across robustness variants with 95% CIs",
    )
    c17, c17f = n.pclaim("C17"), n.full.pclaim("C17")
    s.notes_slide.notes_text_frame.text = (
        f"Top: forecast-bias regressions since {n.since}. The VIX as a biased forecast (C17) is "
        + ("supported" if c17["supported"] else "not supported")
        + f" since {n.since}"
        + (
            "; with the 1990s back-history it is."
            if c17f["supported"] and not c17["supported"]
            else "."
        )
        + " Bottom: robustness variants over 1990–2025, including post- vs pre-2003."
    )


def slide_appendix_years(prs, cfg: Config, n: Numbers, fig: Path) -> None:
    """Appendix: vol by year (headline sample)."""
    s = _content_slide(prs, cfg)
    _title(s, "Appendix: Volatility by Year", f"Which years since {n.since} were most irregular?")
    _picture(
        s,
        fig / n.png("13_rv_by_year"),
        X0,
        Y0,
        X1 - X0,
        Y1 - Y0 - 0.3,
        "Annual realised vol and vol-of-vol by year",
    )
    first = n.monthly.index[0]
    if first.month != 1:
        _caption(
            s,
            X0,
            Y1 - 0.28,
            X1 - X0,
            f"{first.year}: vol-of-vol uses {first:%b}–Dec only (the sample starts "
            f"{first:%b %Y}); annual vol uses the whole year.",
        )


def slide_appendix_breakeven(prs, cfg: Config, n: Numbers, fig: Path) -> None:
    """Appendix: the break-even curve for both payoff measures."""
    s = _content_slide(prs, cfg)
    _title(
        s,
        "Appendix: Break-Even Crash Frequency",
        f"How often would crashes of each size need to happen to wipe out the premium? Since "
        f"{n.since}",
    )
    _picture(
        s,
        fig / n.png("16_breakeven"),
        X0,
        Y0,
        X1 - X0,
        Y1 - Y0,
        "Years between crashes needed to break even vs observed, by crash size, scaled and raw",
    )


def _history_rows(x: Numbers) -> list[str]:
    """One column of the backup table: the headline numbers for one sample."""
    win = x.row("2_shape", "payoff_scaled", "win_rate", method="Wilson")
    mean = x.row("1_mean", "payoff_scaled", "mean", method="Newey")
    be = x.row("R_breakeven", "payoff_scaled", "breakeven_years_between")
    tr = x.row("R_time", "payoff_scaled", "trend_slope_per_year")
    tick = " (significant)" if x.rclaim("R-10")["supported"] else " (not significant)"
    return [
        pct(win["estimate"], 0),
        f"{pct(mean['estimate'], 0)} {ci_pct(mean['ci_low'], mean['ci_high'], 0)}",
        "significant" if x.pclaim("C2")["supported"] else "not significant",
        f"{num(be['estimate'])} yrs vs once in {be['null_value']:.0f}{tick}",
        f"{pct(tr['estimate'], 2)} (p{NBSP}={NBSP}{tr['p_value']:.2f}; "
        f"{x.rclaim('R-9')['p_holm']:.2f} after correction)",
        f"{int(x.rclaims['supported'].sum())} of {len(x.rclaims)}",
    ]


def slide_backup_history(prs, cfg: Config, n: Numbers) -> None:
    """Backup (Q&A only): live-VIX era vs the longer history with back-calculated VIX."""
    f = n.full
    labels = [
        "Seller wins",
        "Average scaled payoff [95% CI]",
        "Average raw-variance payoff",
        f"Break-even: {_worst_label(n)}-sized loss needed every …",
        "Trend in the premium, per year",
        "Research claims passing",
    ]
    head = (
        "",
        f"Live VIX, {n.monthly.index[0]:%b %Y}–{n.monthly.index[-1]:%Y}",
        f"Incl. back-calculated VIX, {f.monthly.index[0]:%b %Y}–{f.monthly.index[-1]:%Y}",
    )
    data = [head] + list(zip(labels, _history_rows(n), _history_rows(f), strict=True))
    s = _content_slide(prs, cfg)
    _title(
        s,
        "Backup: The Longer History",
        "Adding the back-calculated 1990s points the same way, and makes the break-even "
        "significant",
    )
    x, y, rh = X0 + 0.1, Y0 + 0.15, 0.52
    widths = [4.5, 3.7, 3.8]
    gt = s.shapes.add_table(
        len(data), 3, Inches(x), Inches(y), Inches(sum(widths)), Inches(rh * len(data))
    )
    gt.name = "history table"
    tbl = gt.table
    for j, cw in enumerate(widths):
        tbl.columns[j].width = Inches(cw)
    for i, rowv in enumerate(data):
        tbl.rows[i].height = Inches(rh)
        for j, v in enumerate(rowv):
            c = tbl.cell(i, j)
            c.text_frame.text = ""
            c.text_frame.paragraphs[0].add_run().text = v
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (CARD if i % 2 == 0 else WHITE)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            para = c.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            _style_run(para.runs[0], 13, i == 0 or j == 0, WHITE if i == 0 else INK)
    pre = f.row(
        "5_robustness", "payoff_scaled", "mean", sample="pre-2003 only (back-calculated VIX)"
    )
    live = pd.Timestamp(cfg["dates"]["vix_new_methodology_date"])
    _caption(
        s,
        x,
        y + rh * len(data) + 0.15,
        sum(widths),
        f"Before {live:%b %Y} the VIX was back-calculated by Cboe, not what traders saw. The "
        f"1990s premium ran higher ({pct(pre['estimate'], 0)} of implied variance), which makes "
        "the full history look like a decline; the gap between the eras is "
        + ("significant." if f.pclaim("C21")["supported"] else "not significant after correction.")
        + " Significant = passes our claim rule (every CI and the Holm-adjusted p).",
    )
    s.notes_slide.notes_text_frame.text = (
        "Backup for Q&A, not presented. If asked 'what about before 2003?': same direction, and "
        "the break-even test becomes significant, but we lead with the live VIX because the "
        "earlier values were reconstructed later. Full results: research_claims.md and "
        "pitch_claims.md (1990–2025) and their _post2003 versions."
    )


def slide_disclaimer(prs) -> None:
    """The template's disclaimer slide (text lives on the layout)."""
    prs.slides.add_slide(prs.slide_layouts[L_DISCLAIMER])


# =========================================================================== build
def _slide_by_title(prs, title: str):
    """The first slide whose title placeholder reads ``title``."""
    for sl in prs.slides:
        t = sl.shapes.title
        if t is not None and t.text_frame.text == title:
            return sl
    raise KeyError(f"no slide titled {title!r}")


def speaker_notes(n: Numbers) -> dict[str, str]:
    """Draft speaker notes (step 9) for slides 3–10, keyed by slide title.

    Each note is what to say (about a minute), then a 'Detail' line for Q&A. Every number is
    read from the result tables, so the notes follow the data if it changes.
    """
    c = n.case
    m20, m25 = n.rs.case_studies[0], n.rs.case_studies[-1]
    win = n.row("2_shape", "payoff_scaled", "win_rate", method="Wilson")["estimate"]
    mean = n.row("1_mean", "payoff_scaled", "mean", method="Newey")
    med = n.row("1_mean", "payoff_scaled", "median")["estimate"]
    skew = n.row("2_shape", "payoff_scaled", "skewness")["estimate"]
    share = n.row("3_crash", "payoff_var", "loss_share_in_decile_1")["estimate"]
    sell = n.row("3_crash", "payoff_scaled", "mean_in_selloff")["estimate"]
    ar = n.row("R_dynamics", "log_rv_cal", "ar1_rho")["estimate"]
    hl = n.row("R_dynamics", "log_rv_cal", "half_life_months")["estimate"]
    trv = n.row("R_dynamics", "log_rv_cal", "trend_slope_per_year")["p_value"]
    gl = research.bucket_labels("gap", n.rs.gap_thresholds, " pts")
    g_hi = n.row("R_conditional", "payoff_scaled", "mean", sample=gl[2])["estimate"]
    g_lo = n.row("R_conditional", "payoff_scaled", "mean", sample=gl[0])["estimate"]
    tr = n.row("R_time", "payoff_scaled", "trend_slope_per_year")
    roll = n.rolling["mean"]
    be = n.row("R_breakeven", "payoff_scaled", "breakeven_years_between")
    c10 = n.rclaim("R-10")
    jf = n.row("R_jensen", "rv_fwd21", "fair_vix_share_above_realised")["estimate"]
    ja = n.row("R_jensen", "vol_spread_pts", "share_vix_above_realised")["estimate"]
    jc = n.row("R_jensen", "rv_fwd21", "cond_convexity_share_of_spread")
    worst = n.monthly.nsmallest(5, "payoff_scaled")
    worst_txt = ", ".join(f"{pd.Timestamp(r.month):%B %Y}" for r in worst.itertuples())
    top = ", ".join(str(y) for y in n.years.sort_values("rank_irregular").head(5).index)
    first = f"{n.monthly.index[0]:%B %Y}"
    w = round(6 * win)
    return {
        "What Does the VIX Measure?": (
            "The VIX is the market's price for the next month of S&P 500 volatility, read from "
            "option prices. Squared, it is the price of next month's variance. If you sell that "
            "variance you collect the VIX and pay out whatever volatility actually happens, so "
            "you win when markets are calmer than options implied. Throughout, positive means "
            "the VIX over-forecast and the seller won.\n\n"
            "One subtlety: the VIX prices variance, so even a perfectly fair VIX would sit above "
            f"realised volatility in about {pct(jf, 0)} of months. It actually does in "
            f"{pct(ja, 0)}. Convexity explains about {pct(jc['estimate'], 0)} of the average gap "
            "and the rest is premium, which is why we test in variance, not volatility.\n\n"
            f"Detail: convexity share 95% CI {ci_pct(jc['ci_low'], jc['ci_high'], 0)}; an "
            "upper estimate (research_summary, section D)."
        ),
        "Data and Validation": (
            "Our data come from public sources, Cboe via FRED for the VIX and Yahoo Finance for "
            "the S&P 500, and they reproduce the published benchmarks without any adjustment. "
            f"Our evidence starts in {first}: before then the VIX was back-calculated by Cboe "
            f"after the fact, not something anyone could trade. That gives us {len(n.monthly)} "
            "months. Each month we read the VIX at the close of its first trading day and compare "
            "it with the variance of the next 21 trading days, so nothing from the future leaks "
            "into the starting point.\n\n"
            "Detail: the 1990–2003 back-history is on the backup slide if asked."
        ),
        "Does the VIX Over-Forecast Volatility?": (
            f"First question: does the VIX over-forecast? Since {n.since} the variance seller "
            f"wins in {pct(win, 0)} of months, about {SMALL_NUMBERS[w]} in six. On average the "
            f"seller keeps {pct(mean['estimate'], 0)} of the implied variance, and the "
            f"confidence interval runs from {pct(mean['ci_low'], 0)} to "
            f"{pct(mean['ci_high'], 0)}, so it is clearly above zero, whichever way we compute "
            f"the standard errors. The typical month is better still, at {pct(med, 0)}. The gap "
            "between the typical month and the average is the first hint of the next slide: a "
            "few very bad months drag the average down.\n\n"
            "Detail: Newey–West and 6-month block-bootstrap CIs allow for autocorrelation."
        ),
        "How Bad Are the Losing Months?": (
            "So why isn't selling volatility free money? Because the losses are rare but violent. "
            f"The payoff is heavily lopsided, with a skew of {num(skew)}. The five worst months "
            f"are {worst_txt}, and they all come when stocks fall: the worst tenth of S&P months "
            f"carries {pct(share, 0)} of all the seller's losses. In a sell-off month the seller "
            f"loses about {pct(-sell, 0)} of the implied variance. In other words, selling the "
            "VIX is selling crash insurance.\n\n"
            "Detail: sorting by the S&P move describes when losses happen; it is not a forecast "
            "made at the start of the month."
        ),
        "Case Study: March 2020 vs April 2025": (
            f"Two examples. In {pd.Period(m20, 'M'):%B %Y} the VIX started at "
            f"{c(m20, 'vix_at_t'):.0f}, already higher than "
            f"{pct(c(m20, 'vix_at_t_percentile'), 0)} of months, so the market was worried. "
            "But volatility came in at "
            f"{c(m20, 'realised_vol_fwd'):.0f}. The seller's whole premium was gone by day "
            f"{c(m20, 'premium_gone_by_day'):.0f}, the loss equalled "
            f"{c(m20, 'months_of_premium_lost'):.0f} months of average premium, and it took "
            f"{c(m20, 'months_to_recover'):.0f} months to win back.\n\n"
            f"In {pd.Period(m25, 'M'):%B %Y} the VIX started at "
            f"{c(m25, 'vix_at_t'):.0f}; the premium was gone by day "
            f"{c(m25, 'premium_gone_by_day'):.0f}. The S&P ended the month almost flat, but the "
            "seller still lost, because variance builds up every day: the path matters, not "
            "where the market ends.\n\n"
            "Detail: these are outcomes after the start of the month, not signals."
        ),
        "How Does Realised Volatility Behave?": (
            "To understand the risk we looked at how volatility itself behaves. It clusters: a "
            f"high-volatility month tends to be followed by another (AR(1) of {num(ar, 2)}). But "
            f"shocks fade quickly, with a half-life of about {num(hl)} months, and there is no "
            f"long-run trend since {n.since}. Volatility comes in bursts, which is exactly what "
            f"makes the premium lumpy. The most irregular years were {top}.\n\n"
            f"Detail: unit root rejected (ADF); trend p = {trv:.2f}."
        ),
        "When Is the Premium Largest?": (
            "Can we tell in advance which months are dangerous, using only what is known at the "
            "start of the month? We split months by the VIX level and by how far the VIX sits "
            "above recent realised volatility. The numbers point the expected way: the premium "
            f"is larger when the VIX is well above recent volatility, {pct(g_hi, 0)} against "
            f"{pct(g_lo, 0)}. But none of these differences is statistically significant. The "
            "seller wins most months in every group; what differs is the size of the losses. "
            "That is the question the fair-price model takes on next.\n\n"
            "Detail: thresholds fixed in advance (VIX 15 / 25; gap 0 / 5 points); in-sample "
            "description, not a trading rule."
        ),
        "Is the Premium Shrinking?": (
            f"Is the premium disappearing? No. Since {n.since} there is no trend, "
            f"{pct(tr['estimate'], 2)} a year with a p-value of {tr['p_value']:.2f}. It dips hard "
            f"after crashes, down to {pct(roll.min(), 0)} after COVID, and recovers; it is "
            f"{pct(roll.iloc[-1], 0)} now.\n\n"
            "Then the key question: is the VIX too pessimistic? For the seller to break even, a "
            f"March-2020-sized loss would have to happen every {num(be['estimate'])} years. We "
            f"have seen one in {be['null_value']:.0f} years. That is a wide margin, but with a "
            "single crash in the data we can't call it proven."
            + (" " if not c10["supported"] else " It passes our test. ")
            + "In raw variance terms the margin is thinner.\n\n"
            f"Detail: break-even CI {num(be['ci_low'])}–{num(be['ci_high'])} years; R-10 "
            f"Holm-adjusted p = {c10['p_holm']:.2f}."
        ),
    }


def add_speaker_notes(prs, n: Numbers) -> None:
    """Write the drafted speaker notes onto their slides (replacing any shorter note)."""
    for title, text in speaker_notes(n).items():
        _slide_by_title(prs, title).notes_slide.notes_text_frame.text = text


def build_deck(cfg: Config) -> Path:
    """Build the pitch deck from the OAF template; returns the output path."""
    d = cfg["deck"]
    tpl = cfg.root / d["template"]
    out = cfg.root / d["output"]
    global SOURCE_NOTE
    fig = cfg.path("figures") / cfg["plots"]["deck"]["subdir"]  # slide versions (step 8)
    SOURCE_NOTE = (
        cfg["plots"]["source_note"] + ". Sign: implied − realised (positive = seller gains)."
    )
    n = load_numbers(cfg)
    _EXHIBIT_N[0] = 0
    prs = Presentation(str(tpl))
    _clear_slides(prs)
    _restyle_titles(prs, cfg)

    slide_title(prs, cfg)
    slide_agenda(prs, cfg)
    slide_i1(prs, cfg, n, fig)
    slide_i2(prs, cfg, n, fig)
    slide_r1(prs, cfg, n, fig)
    slide_r2(prs, cfg, n, fig)
    slide_r3(prs, cfg, n, fig)
    slide_r4(prs, cfg, n, fig)
    slide_r5(prs, cfg, n, fig)
    slide_r6(prs, cfg, n, fig)
    slide_team(
        prs,
        cfg,
        "Fair-Price Model",
        "Is the VIX still too high after paying the seller for crash risk?",
        "Model team's slides go here",
        [
            "Pricing approaches tried (including the utility-based model that failed in "
            "crash months), the final model and its out-of-sample evidence."
        ],
    )
    slide_team(
        prs,
        cfg,
        "Strategy and Backtest",
        "Does stepping aside in thin-premium months cut the drawdowns?",
        "Strategy team's slides go here",
        [
            "Instrument, rules, and baseline vs model-gated results (Sharpe, max drawdown, "
            "worst month), with and without costs."
        ],
    )
    slide_verdict(prs, cfg, n)
    slide_appendix(prs, cfg, n, fig)
    slide_appendix_breakeven(prs, cfg, n, fig)
    slide_appendix_years(prs, cfg, n, fig)
    slide_backup_history(prs, cfg, n)
    slide_disclaimer(prs)
    add_speaker_notes(prs, n)

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    log.info("deck: wrote %s (%d slides)", out, len(prs.slides))
    return out
