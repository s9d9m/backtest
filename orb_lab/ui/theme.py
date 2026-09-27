"""Visual system: palette, number formatting, CSS and small HTML components (cards, badges, banners).

Everything that decides how numbers and statuses LOOK lives here, so every page formats them the same way.
"""

from __future__ import annotations

import html
import math

import streamlit as st

# ---------------------------------------------------------------------------------------------- palette
BLUE = "#2a78d6"      # series 1 (strategy / net)
ORANGE = "#eb6834"    # series 2 (comparison)
AQUA = "#1baf7a"      # series 3
GOOD = "#0ca30c"      # status only
WARN = "#fab219"      # status only
SERIOUS = "#ec835a"   # status only
BAD = "#d03b3b"       # status only
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
NEUTRAL = "#b9b8b0"   # de-emphasised marks (e.g. in-sample bars)
DIVERGING = [[0.0, "#b3261e"], [0.25, "#e8837f"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1.0, "#184f95"]]

TONE_COLOR = {"good": GOOD, "warn": WARN, "bad": BAD, "info": BLUE, "none": BASELINE, "serious": SERIOUS}
TONE_ICON = {"good": "✓", "warn": "!", "bad": "✕", "info": "i", "none": "○", "serious": "!"}

# ---------------------------------------------------------------------------------------------- formatting
MINUS = "−"


def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and (math.isnan(x)))


def usd(x, signed: bool = False, dp: int = 0) -> str:
    if _bad(x):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    s = f"${abs(x):,.{dp}f}"
    if x < 0 and round(abs(x), dp) > 0:
        return MINUS + s
    return ("+" + s) if signed and x > 0 else s


def pct(x, dp: int = 1, signed: bool = False) -> str:
    if _bad(x):
        return "—"
    v = x * 100
    s = f"{abs(v):.{dp}f}%"
    if v < 0 and round(abs(v), dp) > 0:
        return MINUS + s
    return ("+" + s) if signed and v > 0 else s


def rmult(x, dp: int = 2) -> str:
    if _bad(x):
        return "—"
    if round(x, dp) == 0:
        return f"{0:.{dp}f}R"
    return (MINUS if x < 0 else "+") + f"{abs(x):.{dp}f}R"


def ratio(x, dp: int = 2) -> str:
    if _bad(x):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return (MINUS if x < 0 else "") + f"{abs(x):.{dp}f}"


def price_decimals(tick_size: float) -> int:
    """Decimals needed to show every tick: 2 for $0.01/0.25, 1 for gold's 0.10, 5 for 6E's 0.00005."""
    d = 0
    while d < 8 and abs(round(tick_size, d) - tick_size) > 1e-12:
        d += 1
    return max(d, 2)


def price(x, tick_size: float) -> str:
    return "—" if _bad(x) else f"{x:,.{price_decimals(tick_size)}f}"


def count(x) -> str:
    return "—" if _bad(x) else f"{int(x):,}"


# ---------------------------------------------------------------------------------------------- glossary
TIPS = {
    "Expectancy": "Average net result per trade in R (R = the amount risked from entry to stop). +0.13R means on average you "
                  "make 13 % of what you risk, after costs.",
    "Profit factor": "Gross profits divided by gross losses, after costs. Above 1.0 = net profitable.",
    "Win rate": "Share of trades that made money. Meaningless alone: it must be read together with the size of wins and losses.",
    "Max drawdown": "The largest fall from a previous equity peak, in % of that peak.",
    "Sharpe": "Average daily return divided by its volatility, annualised. Unreliable on less than a year of data.",
    "Sortino": "Like Sharpe, but only downside volatility counts. Same short-sample caveat.",
    "Calmar": "Annual growth divided by the maximum drawdown. Unreliable on less than a year of data.",
    "t-stat": "How many standard errors the average trade is away from zero. Below about 2 the result is not distinguishable "
              "from luck, and after testing many settings even 2-3 can be luck.",
    "Top-5 share": "Share of total profit that came from the 5 best trades. Above ~50 % means a few outliers carry the result; "
                   "above 100 % means all the other trades together lost money.",
    "Trades": "Number of executed trades. Fewer than ~30 trades says almost nothing; hundreds are needed for confidence.",
    "Net return": "Net profit divided by the starting balance.",
    "OOS": "Out-of-sample: data that played no part in choosing the strategy. Only OOS results count as evidence.",
    "WFO": "Walk-forward optimisation: repeatedly choose settings on past data, freeze them, test them on the next unseen period, "
           "roll forward, and stitch the unseen pieces together.",
    "Monte Carlo": "Re-orders or re-samples the historical trades thousands of times to show how different the path (drawdowns, "
                   "losing streaks) could have been. It cannot prove an edge.",
    "In-sample": "Data that was used to choose or tune the strategy. Results on it are optimistic by construction.",
    "Validation": "A period after the training data used to compare candidates. Choosing on it is itself a selection, so it is "
                  "not final evidence.",
    "Blind holdout": "The last part of the data, hidden from every other page until a frozen strategy is tested on it once.",
    "Plateau": "Neighbouring parameter values also work: the result does not hinge on one exact setting.",
    "Spike": "Neighbouring parameter values collapse: the chosen setting is probably a lucky fit.",
    "Cumulative R": "The running sum of every trade's result in R. Independent of position size.",
    "Cost per trade": "Commission + exchange fees + modelled friction + slippage for one trade at the chosen size.",
}


def tip(term: str, text: str | None = None) -> str:
    """HTML help icon with a hover tooltip."""
    body = html.escape(text or TIPS.get(term, ""), quote=True)
    return f'<span class="orb-tip" title="{body}">?</span>' if body else ""


# ---------------------------------------------------------------------------------------------- CSS
CSS = f"""
<style>
:root {{ --orb-blue:{BLUE}; --orb-ink:{INK}; --orb-ink2:{INK_2}; --orb-muted:{MUTED}; --orb-grid:{GRID}; }}
.block-container {{ max-width: 1360px; padding-top: 1.4rem; padding-bottom: 3rem; }}
h1 {{ font-size: 1.75rem !important; font-weight: 700 !important; letter-spacing: -0.01em; margin-bottom: 0 !important; }}
h2 {{ font-size: 1.3rem !important; font-weight: 650 !important; margin-top: 0.6rem !important; }}
h3 {{ font-size: 1.08rem !important; font-weight: 650 !important; }}
p, li, label, .stMarkdown {{ font-size: 0.97rem; }}
[data-testid="stCaptionContainer"] {{ font-size: 0.86rem; color: {INK_2}; }}
/* sidebar navigation as a menu */
section[data-testid="stSidebar"] {{ background: #f4f5f7; }}
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 2px; width: 100%; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > div {{ width: 100%; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] {{ padding: 8px 12px; border-radius: 8px; width: 100% !important;
    display: flex !important; margin: 0; cursor: pointer; box-sizing: border-box; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background: #e6e9ef; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {{ display: none; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ font-size: 0.98rem; font-weight: 550; color: {INK_2}; }}
section[data-testid="stSidebar"] div[data-selected="true"] > [data-testid="stRadioOption"] {{ background: #dce8f8; }}
section[data-testid="stSidebar"] div[data-selected="true"] > [data-testid="stRadioOption"] p {{ color: #1c5cab; font-weight: 700; }}
/* page header */
.orb-eyebrow {{ font-size: 0.78rem; font-weight: 650; letter-spacing: 0.08em; color: {MUTED}; text-transform: uppercase; margin-bottom: 2px; }}
.orb-sub {{ color: {INK_2}; font-size: 0.98rem; margin: 4px 0 14px 0; }}
/* KPI cards */
.orb-grid {{ display: grid; grid-template-columns: repeat(var(--cols, 4), minmax(0, 1fr)); gap: 12px; margin: 6px 0 14px 0; }}
@media (max-width: 1100px) {{ .orb-grid {{ grid-template-columns: repeat(min(var(--cols, 4), 3), minmax(0, 1fr)); }} }}
@media (max-width: 760px) {{ .orb-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.orb-kpi {{ background: #fff; border: 1px solid rgba(11,11,11,0.09); border-radius: 12px; padding: 12px 14px 11px 14px;
            border-top: 3px solid var(--tone, {BASELINE}); box-shadow: 0 1px 2px rgba(0,0,0,0.03); min-height: 92px; }}
.orb-kpi .lbl {{ font-size: 0.8rem; color: {INK_2}; font-weight: 600; display: flex; align-items: center; gap: 6px; }}
.orb-kpi .val {{ font-size: 1.55rem; overflow: hidden; text-overflow: ellipsis; font-weight: 700; color: {INK}; line-height: 1.25; margin-top: 4px; white-space: nowrap; }}
.orb-kpi .sub {{ font-size: 0.8rem; color: {MUTED}; margin-top: 2px; }}
.orb-kpi.big .val {{ font-size: 2.0rem; }}
.orb-tip {{ display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px; border-radius: 50%;
            border: 1px solid {BASELINE}; color: {MUTED}; font-size: 0.66rem; font-weight: 700; cursor: help; }}
/* general cards */
.orb-card {{ background: #fff; border: 1px solid rgba(11,11,11,0.09); border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; }}
.orb-card .ttl {{ font-size: 0.78rem; font-weight: 650; letter-spacing: 0.06em; color: {MUTED}; text-transform: uppercase; }}
.orb-card .body {{ font-size: 1.02rem; color: {INK}; margin-top: 6px; line-height: 1.5; }}
.orb-chip {{ display: inline-block; background: #eef2f8; color: #1d3f73; border-radius: 999px; padding: 2px 10px; margin: 2px 4px 2px 0;
             font-size: 0.84rem; font-weight: 550; white-space: nowrap; }}
.orb-chip.market {{ background: {BLUE}; color: #fff; font-weight: 700; }}
.orb-chip.grey {{ background: #f1f1ef; color: {INK_2}; }}
/* status badges */
.orb-stages {{ display: grid; grid-template-columns: repeat(var(--cols, 4), minmax(0, 1fr)); gap: 10px; margin: 6px 0 14px 0; }}
@media (max-width: 1200px) {{ .orb-stages {{ grid-template-columns: repeat(min(var(--cols, 4), 4), minmax(0, 1fr)); }} }}
@media (max-width: 760px) {{ .orb-stages {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.orb-stage {{ border-radius: 10px; padding: 10px 12px; background: #fff; border: 1px solid rgba(11,11,11,0.09); border-left: 5px solid var(--tone); }}
.orb-stage .nm {{ font-size: 0.74rem; letter-spacing: 0.07em; font-weight: 700; color: {INK_2}; text-transform: uppercase; }}
.orb-stage .st {{ font-size: 0.98rem; font-weight: 650; color: {INK}; margin-top: 3px; display: flex; gap: 6px; align-items: center; }}
.orb-stage .dt {{ font-size: 0.8rem; color: {MUTED}; margin-top: 2px; }}
.orb-ico {{ display: inline-flex; width: 18px; height: 18px; border-radius: 50%; align-items: center; justify-content: center;
            color: #fff; font-size: 0.72rem; font-weight: 800; background: var(--tone); flex: none; }}
/* verdict banner */
.orb-verdict {{ border-radius: 14px; padding: 16px 20px; margin: 4px 0 16px 0; border: 1px solid rgba(11,11,11,0.08);
                background: linear-gradient(90deg, var(--bg) 0%, #ffffff 100%); border-left: 8px solid var(--tone); }}
.orb-verdict .k {{ font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; color: {INK_2}; text-transform: uppercase; }}
.orb-verdict .v {{ font-size: 1.55rem; font-weight: 800; color: {INK}; margin: 2px 0 6px 0; }}
.orb-verdict ul {{ margin: 0 0 0 1.1rem; padding: 0; color: {INK_2}; font-size: 0.93rem; }}
/* flow diagram */
.orb-flow {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 6px 0 16px 0; }}
.orb-flow .step {{ padding: 8px 14px; border-radius: 8px; font-weight: 700; font-size: 0.86rem; letter-spacing: 0.03em; }}
.orb-flow .arrow {{ color: {MUTED}; font-size: 1.1rem; }}
.orb-flow .train {{ background: #eceae4; color: {INK_2}; }}
.orb-flow .val {{ background: #fdf0cc; color: #6b4e00; }}
.orb-flow .freeze {{ background: #e5e5f7; color: #3a2f86; }}
.orb-flow .oos {{ background: {BLUE}; color: #fff; }}
.orb-flow .roll {{ background: #f1f1ef; color: {INK_2}; }}
/* notes */
.orb-note {{ border-radius: 10px; padding: 10px 14px; margin: 6px 0 12px 0; font-size: 0.93rem; color: {INK}; background: #f6f8fb;
             border: 1px solid #e3e8f0; }}
.orb-note.warn {{ background: #fff7e6; border-color: #f6d58b; }}
.orb-note.bad {{ background: #fdeeee; border-color: #f0b4b4; }}
.orb-note.good {{ background: #eef8ee; border-color: #b9e0b9; }}
.orb-section {{ font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; color: {MUTED}; text-transform: uppercase;
               margin: 18px 0 6px 0; border-bottom: 1px solid {GRID}; padding-bottom: 4px; }}
div[data-testid="stExpander"] details summary p {{ font-weight: 600; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------- components
def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    sub = f'<div class="orb-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="orb-eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if sub:
        st.markdown(sub, unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="orb-section">{html.escape(title)}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = "none", help_term: str | None = None, big: bool = False,
        help_text: str | None = None) -> str:
    t = tip(help_term or label, help_text) if (help_term or help_text or label in TIPS) else ""
    color = TONE_COLOR.get(tone, BASELINE)
    return (f'<div class="orb-kpi{" big" if big else ""}" style="--tone:{color}"><div class="lbl">{html.escape(label)} {t}</div>'
            f'<div class="val">{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>')


def kpi_grid(cards: list[str], min_width: int = 165, cols: int | None = None) -> None:
    """Cards in balanced rows: 4 per row by default, 3 for 3/6/9 cards, 5 for 5/10 (fewer on narrow screens)."""
    n = len(cards)
    if cols is None:
        cols = n if n <= 5 else (4 if n % 4 == 0 else 3 if n % 3 == 0 else 5 if n % 5 == 0 else 4)
    st.markdown(f'<div class="orb-grid" style="--cols:{cols}">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="orb-card"><div class="ttl">{html.escape(title)}</div><div class="body">{body_html}</div></div>',
                unsafe_allow_html=True)


def chips(items: list[str], first_is_market: bool = False) -> str:
    out = []
    for i, it in enumerate(items):
        cls = "orb-chip market" if (first_is_market and i == 0) else "orb-chip"
        out.append(f'<span class="{cls}">{html.escape(str(it))}</span>')
    return "".join(out)


def stage_badges(stages: list[dict], cols: int | None = None) -> None:
    """stages: [{name, status, detail, tone}]"""
    cols = cols or (len(stages) if len(stages) <= 7 else 4)
    parts = []
    for s in stages:
        tone = s.get("tone", "none")
        color = TONE_COLOR.get(tone, BASELINE)
        parts.append(f'<div class="orb-stage" style="--tone:{color}"><div class="nm">{html.escape(s["name"])}</div>'
                     f'<div class="st"><span class="orb-ico" style="--tone:{color}">{TONE_ICON.get(tone, "○")}</span>'
                     f'{html.escape(s["status"])}</div><div class="dt">{html.escape(s.get("detail", ""))}</div></div>')
    st.markdown(f'<div class="orb-stages" style="--cols:{cols}">' + "".join(parts) + "</div>", unsafe_allow_html=True)


VERDICT_TONE = {"PROMISING": "good", "MIXED": "warn", "NO PRELIMINARY EVIDENCE": "bad", "INSUFFICIENT EVIDENCE": "none"}
TONE_BG = {"good": "#eef8ee", "warn": "#fff7e6", "bad": "#fdeeee", "none": "#f3f3f1", "info": "#eef4fc", "serious": "#fdf1ea"}


def verdict_banner(kicker: str, verdict: str, reasons: list[str], tone: str | None = None) -> None:
    tone = tone or VERDICT_TONE.get(verdict, "none")
    items = "".join(f"<li>{html.escape(r)}</li>" for r in reasons)
    st.markdown(f'<div class="orb-verdict" style="--tone:{TONE_COLOR[tone]};--bg:{TONE_BG[tone]}"><div class="k">{html.escape(kicker)}</div>'
                f'<div class="v">{html.escape(verdict)}</div><ul>{items}</ul></div>', unsafe_allow_html=True)


def note(text: str, tone: str = "info") -> None:
    """Short inline note (markdown-free HTML). Use st.warning/st.error for things that must never be missed."""
    cls = {"info": "", "warn": " warn", "bad": " bad", "good": " good"}.get(tone, "")
    st.markdown(f'<div class="orb-note{cls}">{text}</div>', unsafe_allow_html=True)


def flow(steps: list[tuple[str, str]]) -> None:
    parts = []
    for i, (label, cls) in enumerate(steps):
        if i:
            parts.append('<span class="arrow">→</span>')
        parts.append(f'<span class="step {cls}">{html.escape(label)}</span>')
    st.markdown('<div class="orb-flow">' + "".join(parts) + "</div>", unsafe_allow_html=True)
