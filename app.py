"""ExperimentGuard — clearance console.

A designed Streamlit front end for the A/B test decision engine. The verdict is
framed as a go/no-go *clearance ruling*: the engine's decision gates are shown as
a pass/block trace, metrics are rendered as instrument tiles, and the primary
effect's confidence interval is drawn on a zero-anchored axis.

All statistics live in the ``experimentguard`` package; this file is presentation
only. Run: ``streamlit run app.py``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from experimentguard.pipeline import analyze
from experimentguard.report import build_report
from experimentguard.data import (
    generate_synthetic_experiment,
    validate_experiment_frame,
    DataValidationError,
)
from experimentguard.decision import Decision

st.set_page_config(page_title="ExperimentGuard", page_icon="🛰️", layout="wide")

# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --ink:      #0C1220;
  --panel:    #141D2E;
  --panel-2:  #1B2740;
  --line:     #29344E;
  --line-soft:#212C44;
  --paper:    #EAEEF6;
  --dim:      #8592A8;
  --dim-2:    #5C6A82;
  --signal:   #5CC8FF;
  --ship:     #34D399;
  --block:    #FB5A6A;
  --hold:     #FBBF3C;
}

/* --- base + Streamlit chrome resets --- */
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
#MainMenu, header[data-testid="stHeader"], footer { display: none; }
.stApp { background:
  radial-gradient(1200px 600px at 85% -10%, rgba(92,200,255,0.06), transparent 60%),
  radial-gradient(900px 500px at 0% 0%, rgba(52,211,153,0.05), transparent 55%),
  var(--ink); }
.block-container { padding: 1.4rem 2rem 4rem; max-width: 1180px; }
section[data-testid="stSidebar"] { background: #0A0F1B; border-right: 1px solid var(--line-soft); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* --- masthead --- */
.eg-mast { display:flex; align-items:baseline; justify-content:space-between;
  border-bottom:1px solid var(--line); padding-bottom:14px; margin-bottom:22px; gap:16px; }
.eg-mark { font-family:'Archivo'; font-weight:800; font-size:20px; letter-spacing:2.5px;
  color:var(--paper); text-transform:uppercase; }
.eg-mark span { color:var(--signal); }
.eg-tagline { font-family:'IBM Plex Mono'; font-size:11px; letter-spacing:1px;
  color:var(--dim-2); text-transform:uppercase; }
.eg-idchip { font-family:'IBM Plex Mono'; font-size:12px; color:var(--dim);
  border:1px solid var(--line); border-radius:999px; padding:5px 12px; white-space:nowrap; }
.eg-idchip b { color:var(--paper); font-weight:600; }

/* --- eyebrow section headers --- */
.eg-eyebrow { font-family:'Archivo'; font-weight:700; font-size:11px; letter-spacing:3px;
  text-transform:uppercase; color:var(--dim); display:flex; align-items:center; gap:9px; margin:26px 0 12px; }
.eg-eyebrow::before { content:''; width:6px; height:6px; background:var(--signal);
  box-shadow:0 0 10px var(--signal); border-radius:1px; }

/* --- verdict clearance panel --- */
.eg-verdict { position:relative; border:1px solid var(--line); border-radius:16px;
  background:linear-gradient(180deg, var(--panel), #101828); overflow:hidden;
  display:grid; grid-template-columns:1.1fr 1.4fr; gap:0; }
.eg-verdict::before { content:''; position:absolute; left:0; top:0; bottom:0; width:5px; }
.eg-verdict.ship::before   { background:var(--ship);  box-shadow:0 0 30px rgba(52,211,153,.6); }
.eg-verdict.block::before  { background:var(--block); box-shadow:0 0 30px rgba(251,90,106,.6); }
.eg-verdict.hold::before   { background:var(--hold);  box-shadow:0 0 30px rgba(251,191,60,.6); }
.eg-v-left { padding:30px 34px; border-right:1px solid var(--line-soft); }
.eg-v-kicker { font-family:'IBM Plex Mono'; font-size:11px; letter-spacing:2px;
  text-transform:uppercase; color:var(--dim); margin-bottom:14px; }
.eg-v-word { font-family:'Archivo'; font-weight:800; font-size:64px; line-height:.92;
  letter-spacing:-1.5px; text-transform:uppercase; }
.eg-verdict.ship  .eg-v-word { color:var(--ship); }
.eg-verdict.block .eg-v-word { color:var(--block); }
.eg-verdict.hold  .eg-v-word { color:var(--hold); }
.eg-v-status { display:inline-flex; align-items:center; gap:8px; margin-top:16px;
  font-family:'IBM Plex Mono'; font-size:12px; color:var(--dim); }
.eg-v-dot { width:9px; height:9px; border-radius:50%; }
.eg-verdict.ship  .eg-v-dot { background:var(--ship);  box-shadow:0 0 12px var(--ship); }
.eg-verdict.block .eg-v-dot { background:var(--block); box-shadow:0 0 12px var(--block); }
.eg-verdict.hold  .eg-v-dot { background:var(--hold);  box-shadow:0 0 12px var(--hold); }
.eg-v-right { padding:28px 30px; display:flex; flex-direction:column; justify-content:center; }
.eg-v-right .lbl { font-family:'IBM Plex Mono'; font-size:11px; letter-spacing:2px;
  text-transform:uppercase; color:var(--dim-2); margin-bottom:12px; }
.eg-ruling { list-style:none; margin:0; padding:0; }
.eg-ruling li { position:relative; padding:9px 0 9px 22px; font-size:14px; color:var(--paper);
  line-height:1.5; border-bottom:1px solid var(--line-soft); }
.eg-ruling li:last-child { border-bottom:none; }
.eg-ruling li::before { content:'›'; position:absolute; left:2px; color:var(--signal); font-weight:700; }

/* --- gate trace (signature) --- */
.eg-trace { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-top:14px; }
.eg-gate { border:1px solid var(--line-soft); border-radius:10px; background:var(--panel);
  padding:13px 14px; }
.eg-gate .g-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:9px; }
.eg-gate .g-idx { font-family:'IBM Plex Mono'; font-size:10px; color:var(--dim-2); letter-spacing:1px; }
.eg-gate .g-icon { width:18px; height:18px; border-radius:5px; display:grid; place-items:center;
  font-size:11px; font-weight:800; font-family:'Archivo'; color:var(--ink); }
.eg-gate .g-name { font-family:'Archivo'; font-weight:600; font-size:13px; color:var(--paper); }
.eg-gate .g-state { font-family:'IBM Plex Mono'; font-size:10px; letter-spacing:1px;
  text-transform:uppercase; margin-top:3px; }
.g-pass   .g-icon { background:var(--ship); }   .g-pass   .g-state { color:var(--ship); }
.g-block  .g-icon { background:var(--block); }  .g-block  .g-state { color:var(--block); }
.g-hold   .g-icon { background:var(--hold); }   .g-hold   .g-state { color:var(--hold); }
.g-idle   .g-icon { background:var(--dim-2); }  .g-idle   .g-state { color:var(--dim); }
.g-block  { border-color:rgba(251,90,106,.4); } .g-hold { border-color:rgba(251,191,60,.35); }

/* --- instrument tiles --- */
.eg-tiles { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.eg-tile { border:1px solid var(--line-soft); border-radius:12px; background:var(--panel);
  padding:16px 18px; position:relative; }
.eg-tile .t-lbl { font-family:'IBM Plex Mono'; font-size:10.5px; letter-spacing:1.5px;
  text-transform:uppercase; color:var(--dim); }
.eg-tile .t-val { font-family:'IBM Plex Mono'; font-weight:600; font-size:32px; color:var(--paper);
  margin-top:10px; letter-spacing:-1px; }
.eg-tile .t-val .u { font-size:15px; color:var(--dim); margin-left:3px; }
.eg-tile .t-meta { margin-top:9px; font-family:'IBM Plex Mono'; font-size:12px; color:var(--dim);
  display:flex; align-items:center; gap:9px; }
.eg-chip { font-family:'IBM Plex Mono'; font-size:10px; font-weight:600; letter-spacing:.5px;
  text-transform:uppercase; padding:3px 8px; border-radius:5px; }
.eg-chip.pos { background:rgba(52,211,153,.14); color:var(--ship); }
.eg-chip.neg { background:rgba(251,90,106,.14); color:var(--block); }
.eg-chip.warn{ background:rgba(251,191,60,.14); color:var(--hold); }
.eg-chip.mut { background:rgba(133,146,168,.14); color:var(--dim); }

/* --- panels (metric / guardrail / segment) --- */
.eg-panel { border:1px solid var(--line-soft); border-radius:12px; background:var(--panel);
  padding:20px 22px; height:100%; }
.eg-panel h4 { font-family:'Archivo'; font-weight:700; font-size:13px; letter-spacing:1px;
  text-transform:uppercase; color:var(--paper); margin:0 0 4px; }
.eg-panel .sub { font-size:12.5px; color:var(--dim); margin-bottom:18px; }

/* CI axis */
.eg-ci { position:relative; height:64px; margin-top:6px; }
.eg-ci .axis { position:absolute; top:34px; left:0; right:0; height:1px; background:var(--line); }
.eg-ci .zero { position:absolute; top:14px; bottom:14px; width:1px; background:var(--dim-2); }
.eg-ci .zlbl { position:absolute; top:44px; font-family:'IBM Plex Mono'; font-size:10px;
  color:var(--dim-2); transform:translateX(-50%); }
.eg-ci .bar { position:absolute; top:30px; height:8px; border-radius:4px; }
.eg-ci .bar.pos { background:linear-gradient(90deg,rgba(52,211,153,.4),var(--ship)); }
.eg-ci .bar.neg { background:linear-gradient(90deg,var(--block),rgba(251,90,106,.4)); }
.eg-ci .bar.mix { background:linear-gradient(90deg,var(--block),var(--hold),var(--ship)); }
.eg-ci .pt { position:absolute; top:26px; width:3px; height:16px; background:var(--paper); border-radius:2px; }
.eg-ci .cap { position:absolute; top:2px; font-family:'IBM Plex Mono'; font-size:11px;
  color:var(--paper); transform:translateX(-50%); }
.eg-ci-foot { font-family:'IBM Plex Mono'; font-size:11px; color:var(--dim); margin-top:14px;
  display:flex; justify-content:space-between; }

/* guardrail rows */
.eg-grow { display:flex; align-items:center; justify-content:space-between;
  padding:14px 0; border-bottom:1px solid var(--line-soft); }
.eg-grow:last-child { border-bottom:none; }
.eg-grow .gl { display:flex; flex-direction:column; gap:3px; }
.eg-grow .gname { font-family:'Archivo'; font-weight:600; font-size:14px; color:var(--paper); }
.eg-grow .gnum { font-family:'IBM Plex Mono'; font-size:12px; color:var(--dim); }
.eg-pill { font-family:'IBM Plex Mono'; font-size:10px; font-weight:600; letter-spacing:1px;
  text-transform:uppercase; padding:5px 11px; border-radius:6px; white-space:nowrap; }
.eg-pill.ok  { background:rgba(52,211,153,.13); color:var(--ship); border:1px solid rgba(52,211,153,.3); }
.eg-pill.bad { background:rgba(251,90,106,.13); color:var(--block); border:1px solid rgba(251,90,106,.3); }

/* segment chips */
.eg-segwrap { display:flex; flex-direction:column; gap:16px; }
.eg-segdim .dname { font-family:'IBM Plex Mono'; font-size:11px; letter-spacing:1.5px;
  text-transform:uppercase; color:var(--dim); margin-bottom:9px; }
.eg-segs { display:flex; flex-wrap:wrap; gap:8px; }
.eg-seg { border:1px solid var(--line-soft); border-radius:8px; padding:8px 12px; background:var(--ink);
  font-family:'IBM Plex Mono'; font-size:12px; color:var(--dim); }
.eg-seg.sig { border-color:rgba(52,211,153,.35); color:var(--paper); }
.eg-seg b { color:var(--ship); font-weight:600; }
.eg-seg.dn b { color:var(--block); }

/* sidebar labels */
.eg-side-h { font-family:'Archivo'; font-weight:700; font-size:11px; letter-spacing:2.5px;
  text-transform:uppercase; color:var(--dim); margin:4px 0 2px; }
section[data-testid="stSidebar"] .stSlider label, section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label { font-family:'IBM Plex Mono'!important;
  font-size:12px!important; color:var(--paper)!important; }

@media (max-width:900px) {
  .eg-verdict { grid-template-columns:1fr; }
  .eg-v-left { border-right:none; border-bottom:1px solid var(--line-soft); }
  .eg-trace { grid-template-columns:repeat(2,1fr); }
  .eg-tiles { grid-template-columns:repeat(2,1fr); }
  .eg-v-word { font-size:48px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def h(html_str: str) -> None:
    """Render component HTML, stripping per-line indentation.

    Streamlit's Markdown treats lines indented 4+ spaces as a code block, which
    would print our nested component HTML verbatim. Collapsing leading whitespace
    keeps the HTML but avoids the code-block trigger.
    """
    st.markdown("".join(line.strip() for line in html_str.splitlines()),
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def _k(n: float) -> str:
    return f"{n/1000:.1f}k" if n >= 1000 else f"{n:.0f}"


def _pval(p: float) -> str:
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


# --------------------------------------------------------------------------- #
# Component renderers (return HTML strings)
# --------------------------------------------------------------------------- #

_VCLASS = {Decision.SHIP: "ship", Decision.DO_NOT_SHIP: "block", Decision.CONTINUE: "hold"}
_VSTATUS = {
    Decision.SHIP: "Variant cleared for rollout",
    Decision.DO_NOT_SHIP: "Rollout blocked",
    Decision.CONTINUE: "Held — keep collecting data",
}


def verdict_panel(result) -> str:
    d = result.decision
    cls = _VCLASS[d.decision]
    reasons = "".join(f"<li>{r}</li>" for r in d.reasons)
    return f"""
    <div class="eg-verdict {cls}">
      <div class="eg-v-left">
        <div class="eg-v-kicker">Clearance ruling</div>
        <div class="eg-v-word">{d.decision.value}</div>
        <div class="eg-v-status"><span class="eg-v-dot"></span>{_VSTATUS[d.decision]}</div>
      </div>
      <div class="eg-v-right">
        <div class="lbl">Why the engine ruled this way</div>
        <ul class="eg-ruling">{reasons}</ul>
      </div>
    </div>"""


def gate_trace(result) -> str:
    conv = result.conversion
    obs = min(conv.control_n, conv.variant_n)
    req = result.power.per_arm_sample_size

    def effect_state():
        if conv.significant and conv.absolute_uplift > 0:
            return "pass", "Lift"
        if conv.significant and conv.absolute_uplift < 0:
            return "block", "Drop"
        return "idle", "Flat"

    def value_state():
        b = result.bootstrap
        if b.significant and b.mean_difference > 0:
            return "pass", "Lift"
        if b.significant and b.mean_difference < 0:
            return "block", "Drop"
        return "idle", "Flat"

    gates = [
        ("Randomization", ("pass", "Valid") if not result.srm.srm_detected else ("block", "Mismatch")),
        ("Guardrails", ("pass", "Hold") if result.guardrails.passed else ("block", "Breach")),
        ("Primary effect", effect_state()),
        ("Power", ("pass", "Adequate") if obs >= req else ("hold", "Under")),
        ("Value", value_state()),
    ]
    icons = {"pass": "✓", "block": "✕", "hold": "!", "idle": "–"}
    cells = ""
    for i, (name, (state, word)) in enumerate(gates, 1):
        cells += f"""
        <div class="eg-gate g-{state}">
          <div class="g-top"><span class="g-idx">G{i}</span>
            <span class="g-icon">{icons[state]}</span></div>
          <div class="g-name">{name}</div>
          <div class="g-state">{word}</div>
        </div>"""
    return f'<div class="eg-trace">{cells}</div>'


def tiles(result) -> str:
    conv, boot, srm, power = result.conversion, result.bootstrap, result.srm, result.power
    obs = min(conv.control_n, conv.variant_n)
    req = power.per_arm_sample_size

    conv_chip = ("pos", "Significant") if conv.significant and conv.absolute_uplift > 0 else \
                ("neg", "Significant") if conv.significant else ("mut", "Not sig")
    val_chip = ("pos", "Significant") if boot.significant and boot.mean_difference > 0 else \
               ("neg", "Significant") if boot.significant else ("mut", "Not sig")
    srm_chip = ("mut", "Balanced") if not srm.srm_detected else ("neg", "Mismatch")
    pow_chip = ("pos", "Adequate") if obs >= req else ("warn", "Underpowered")

    def tile(lbl, val, unit, chip, meta):
        return f"""<div class="eg-tile"><div class="t-lbl">{lbl}</div>
          <div class="t-val">{val}<span class="u">{unit}</span></div>
          <div class="t-meta"><span class="eg-chip {chip[0]}">{chip[1]}</span>{meta}</div></div>"""

    return f"""<div class="eg-tiles">
      {tile("Conversion uplift", f"{conv.absolute_uplift*100:+.2f}", "pp", conv_chip,
            f"{conv.relative_uplift*100:+.1f}% rel")}
      {tile("Value uplift / conv.", f"{boot.mean_difference:+.2f}", "", val_chip,
            f"p50 {boot.variant_mean:.1f}")}
      {tile("SRM p-value", _pval(srm.p_value), "", srm_chip,
            f"{srm.observed_control_ratio*100:.1f}% ctrl")}
      {tile("Sample / arm", _k(obs), "", pow_chip, f"need {_k(req)}")}
    </div>"""


def ci_axis(result) -> str:
    c = result.conversion
    lo, hi, pt = c.ci_low * 100, c.ci_high * 100, c.absolute_uplift * 100
    span = max(abs(lo), abs(hi), abs(pt), 0.1) * 1.35

    def pct(v):  # map value to 0..100 across [-span, span]
        return (v + span) / (2 * span) * 100

    bar_cls = "pos" if lo > 0 else "neg" if hi < 0 else "mix"
    lo_p, hi_p, pt_p, z_p = pct(lo), pct(hi), pct(pt), pct(0)
    return f"""
    <div class="eg-ci">
      <div class="axis"></div>
      <div class="zero" style="left:{z_p:.2f}%"></div>
      <div class="zlbl" style="left:{z_p:.2f}%">0</div>
      <div class="bar {bar_cls}" style="left:{lo_p:.2f}%; width:{hi_p-lo_p:.2f}%"></div>
      <div class="pt" style="left:{pt_p:.2f}%"></div>
      <div class="cap" style="left:{lo_p:.2f}%">{lo:+.2f}</div>
      <div class="cap" style="left:{hi_p:.2f}%">{hi:+.2f}</div>
    </div>
    <div class="eg-ci-foot"><span>95% CI on absolute uplift (pp)</span>
      <span>p = {_pval(c.p_value)}</span></div>"""


def primary_panel(result) -> str:
    c = result.conversion
    return f"""<div class="eg-panel">
      <h4>Primary metric — conversion</h4>
      <div class="sub">Control {c.control_rate*100:.2f}% (n={c.control_n:,}) →
        Variant {c.variant_rate*100:.2f}% (n={c.variant_n:,})</div>
      {ci_axis(result)}
    </div>"""


def guardrail_panel(result) -> str:
    g = result.guardrails
    ps = g.details.get("payment_success", {})
    la = g.details.get("latency", {})
    succ_bad = any("payment-success" in b for b in g.breaches)
    lat_bad = any("latency" in b for b in g.breaches)

    def pill(bad):
        return '<span class="eg-pill bad">Breach</span>' if bad else '<span class="eg-pill ok">Pass</span>'

    rows = f"""
      <div class="eg-grow"><div class="gl">
        <span class="gname">Payment success</span>
        <span class="gnum">{ps.get('control_rate',0)*100:.2f}% → {ps.get('variant_rate',0)*100:.2f}%
          &nbsp;(Δ {ps.get('absolute_drop',0)*100:+.2f}pp)</span>
      </div>{pill(succ_bad)}</div>
      <div class="eg-grow"><div class="gl">
        <span class="gname">Latency (mean)</span>
        <span class="gnum">{la.get('control_mean_ms',0):.0f}ms → {la.get('variant_mean_ms',0):.0f}ms
          &nbsp;({la.get('relative_increase',0)*100:+.1f}%, p95 {la.get('variant_p95_ms',0):.0f}ms)</span>
      </div>{pill(lat_bad)}</div>"""
    return f"""<div class="eg-panel">
      <h4>Guardrails</h4>
      <div class="sub">Reliability metrics that must not regress</div>{rows}
    </div>"""


def segment_panel(result) -> str:
    blocks = ""
    for seg in result.segments:
        chips = ""
        for row in sorted(seg.rows, key=lambda r: r.p_value_corrected):
            if row.significant:
                dn = " dn" if row.absolute_uplift < 0 else ""
                chips += (f'<div class="eg-seg sig{dn}">{row.value} '
                          f'<b>{row.absolute_uplift*100:+.2f}pp</b> · q={_pval(row.p_value_corrected)}</div>')
            else:
                chips += f'<div class="eg-seg">{row.value} · ns</div>'
        if not chips:
            chips = '<div class="eg-seg">no segments met minimum size</div>'
        blocks += f'<div class="eg-segdim"><div class="dname">{seg.dimension}</div><div class="eg-segs">{chips}</div></div>'
    return f"""<div class="eg-panel">
      <h4>Segments — BH corrected</h4>
      <div class="sub">Per-slice uplift, false-discovery controlled</div>
      <div class="eg-segwrap">{blocks}</div>
    </div>"""


# --------------------------------------------------------------------------- #
# Sidebar — controls
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown('<div class="eg-side-h">Source</div>', unsafe_allow_html=True)
    source = st.radio("data source", ["Sample scenario", "Upload CSV"],
                      label_visibility="collapsed")

    df = None
    if source == "Sample scenario":
        scenario = st.selectbox(
            "scenario",
            ["Winner", "Sample-ratio mismatch", "Guardrail failure (latency)"],
            label_visibility="collapsed",
        )
        if scenario == "Winner":
            df = generate_synthetic_experiment(seed=1)
        elif scenario == "Sample-ratio mismatch":
            df = generate_synthetic_experiment(control_ratio=0.55, seed=11)
        else:
            df = generate_synthetic_experiment(variant_latency_ms=310.0, seed=13)
    else:
        up = st.file_uploader("csv", type="csv", label_visibility="collapsed")
        if up is not None:
            df = pd.read_csv(up, parse_dates=["event_date"])

    st.markdown('<div class="eg-side-h" style="margin-top:22px">Test design</div>',
                unsafe_allow_html=True)
    mde = st.slider("Min. detectable effect (pp)", 0.1, 5.0, 1.0, 0.1) / 100
    alpha = st.slider("Significance α", 0.01, 0.20, 0.05, 0.01)
    power = st.slider("Target power", 0.50, 0.99, 0.80, 0.05)
    resamples = st.select_slider("Bootstrap resamples", [1000, 5000, 10000], 5000)


# --------------------------------------------------------------------------- #
# Masthead
# --------------------------------------------------------------------------- #

exp_id = str(df["experiment_id"].iloc[0]) if df is not None and "experiment_id" in df else "—"
st.markdown(f"""
<div class="eg-mast">
  <div>
    <div class="eg-mark">Experiment<span>Guard</span></div>
    <div class="eg-tagline">A/B clearance console · payment-flow experiments</div>
  </div>
  <div class="eg-idchip">experiment&nbsp;·&nbsp;<b>{exp_id}</b></div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if df is None:
    st.markdown(
        '<div class="eg-panel" style="text-align:center; padding:52px">'
        '<h4>No experiment loaded</h4>'
        '<div class="sub" style="margin:0">Pick a sample scenario or upload a CSV '
        'in the sidebar to run a clearance ruling.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    validate_experiment_frame(df)
except DataValidationError as exc:
    st.markdown(
        f'<div class="eg-panel" style="border-color:var(--block)">'
        f'<h4 style="color:var(--block)">Data validation failed</h4>'
        f'<div class="sub" style="margin:0">{exc}</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

result = analyze(df, planned_mde=mde, alpha=alpha, power=power,
                 bootstrap_resamples=int(resamples))

h(verdict_panel(result))
h(gate_trace(result))

h('<div class="eg-eyebrow">Instrument readout</div>')
h(tiles(result))

h('<div class="eg-eyebrow">Evidence</div>')
left, right = st.columns([1, 1], gap="medium")
with left:
    h(primary_panel(result))
with right:
    h(guardrail_panel(result))

st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
h(segment_panel(result))

h('<div class="eg-eyebrow">Full report</div>')
with st.expander("Open the one-page Markdown report"):
    st.markdown(build_report(result))
