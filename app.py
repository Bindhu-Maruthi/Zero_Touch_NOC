"""
app.py  —  Zero-Touch Network Operations Dashboard
---------------------------------------------------
Architecture:
  IsolationForest      — anomaly detection
  Q-Learning agent     — learns optimal corrective actions in real-time
  Integrated Gradients — exact signed XAI attribution
  HITL governance      — human approves/denies/escalates CRITICAL events
  Incident Response    — full IR lifecycle with SLA tracking
  Audit log            — CSV traceability for every decision

FIX LOG
-------
FIX-A  Color consistency
  Old: Two different inline color dicts for the same features across charts.
       e.g. latency = #a78bfa in telemetry chart but #f87171 in IG trend.
  New: Single FEAT_COLOURS constant used everywhere.

FIX-B  Throughput / packet_loss invisible in telemetry chart
  Old: ax.set_ylim(0, 280) — packet_loss (0–30) was a flat line at the bottom.
       Latency spikes > 280ms were clipped off the top.
  New: All features normalised to 0–100% of their expected range before plotting.
       Every feature is now visible and comparable. Y-label says "% of scale".

FIX-C  RL agent recreated every tick
  Old: rl_agent = QLearningAgent() — reloads q_table.npy from disk every 2s.
  New: Cached in st.session_state.rl_agent; disk load happens once per session.

FIX-D  UI freezing during HITL
  Old: generate_network_data() (blocking ping, 2–8s) was called on EVERY rerun,
       including when the HITL form was waiting for user input. Submitting
       the form forced a 4s wait before the decision was processed.
  New: Data fetch is skipped when hitl_active=True.
       HITL form responds instantly on submit.

FIX-E  AUTO-HEALING logged every tick (audit log spam)
  Old: log_event() called on every 2s rerun while in AUTO-HEALING state.
       → 30 minutes of degradation = 900 identical rows.
  New: Only logged when decision *transitions into* AUTO-HEALING.
       Tracked via st.session_state.last_logged_decision.

FIX-F  Memory leak — matplotlib figures never closed
  Old: plt.close() missing on 6 of 7 figures. RAM grows without bound.
  New: plt.close(fig) added after every st.pyplot() call.

FIX-G  Radar chart reflected only latest tick
  Old: abs_vals = list(ig_df["Abs_Attribution"]) — single tick snapshot.
  New: Uses mean |IG| over full ig_history for each feature.
       Shows aggregated historical impact, not a noisy single-point snapshot.

FIX-H  stress_proxy inconsistent with RL agent formula
  Old: stress_proxy = latency/250 + packet_loss/10  (ad-hoc 2-feature formula)
  New: stress_proxy = normalise(latest) — same 5-feature weighted formula
       the RL agent uses internally. Chart now matches agent's actual view.

FIX-I  Driver cards showed duplicate/blank data during warm-up
  Old: top2 = ig_df.iloc[0] during warm-up; all three cards identical.
  New: Driver grid hidden and replaced with a warm-up progress indicator
       until ig_history has enough data.

FIX-J  HITL root cause showed "—" (no percentage or confidence)
  Old: build_root_cause_string returned Direction="—" during warm-up.
  New: Fixed in explainability.py — now shows percentage impact + method tag.
       app.py HITL box displays the enriched string directly.

FIX-K  Pie chart divide-by-zero guard
  Old: No guard — if df had 0 rows both wedge values are 0, matplotlib throws.
  New: Guard added; shows placeholder if no data yet.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import os
import numpy as np

from realtime_data    import generate_network_data, FEATURE_COLS
from model            import load_model
from actions          import decide_action, rl_corrective_action, corrective_action
from logger           import log_event
from explainability   import get_ig_attribution, build_root_cause_string
from rl_agent         import QLearningAgent, ACTION_NAMES, normalise   # FIX-H: import normalise
from incident_manager import (
    open_incident, resolve_incident, escalate_incident,
    update_status, get_open_incidents, get_all_incidents, get_sla_status,
)

_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
AUDIT_FILE = os.path.join(_BASE_DIR, "audit_log.csv")

# ─────────────────────────────────────────────────────────────
# FIX-A  SINGLE COLOUR PALETTE — used across every chart
# ─────────────────────────────────────────────────────────────
FEAT_COLOURS = {
    "latency":     "#f87171",   # red    — higher = worse
    "packet_loss": "#fbbf24",   # amber  — higher = worse
    "throughput":  "#4ade80",   # green  — higher = better
    "jitter":      "#a78bfa",   # purple — higher = worse
    "bandwidth":   "#60a5fa",   # blue   — higher = better
}

# FIX-B  Per-feature display ranges for normalisation
FEAT_DISPLAY_RANGES = {
    "latency":     (0, 500),
    "packet_loss": (0, 30),
    "throughput":  (0, 200),
    "jitter":      (0, 150),
    "bandwidth":   (0, 120),
}

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZeroTouch NOC",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "savefig.facecolor":"#0f1117",
    "axes.edgecolor":   "#3d3d5c",
    "axes.labelcolor":  "#a0a0c0",
    "xtick.color":      "#6060a0",
    "ytick.color":      "#6060a0",
    "grid.color":       "#1e1e3a",
    "text.color":       "#c0c0e0",
    "legend.facecolor": "#0f1117",
    "legend.edgecolor": "#3d3d5c",
})

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS  (unchanged)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ════════════════════════════════════════════════
   ZEROTOUCH NOC — GLOBAL DARK THEME
   Works on Streamlit 1.28+ regardless of class names
   ════════════════════════════════════════════════ */

/* 1. Page background */
.stApp, .main, section.main, [data-testid="stAppViewContainer"] {
    background-color: #0a0a14 !important;
}
[data-testid="stHeader"] { background: transparent !important; display:none; }
#MainMenu, footer { display: none !important; }

/* 2. ALL text */
*, *::before, *::after { box-sizing: border-box; }
p, div, label, li, a, td, th, caption,
h1, h2, h3, h4, h5, h6,
.stMarkdown, .stText {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
[data-testid="stMarkdownContainer"] span,
[data-testid="stText"] span,
.stButton span,
label span {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] a,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] code {
    color: #d8d0f8 !important;
}

/* 3. Headings */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #ede9fe !important;
}

/* 4. Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f20 !important;
    border-right: 1px solid #2a2a4a !important;
}

/* 5. Input fields */
input, textarea, select,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #16163a !important;
    border: 1px solid #3a3a6a !important;
    color: #e8e0ff !important;
    border-radius: 8px !important;
    caret-color: #a78bfa !important;
}
input::placeholder, textarea::placeholder { color: #5050a0 !important; }
input:focus, textarea:focus {
    border-color: #7c3aed !important;
    outline: none !important;
    box-shadow: 0 0 0 2px #7c3aed33 !important;
}

/* 6. Labels */
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label,
.stRadio label, .stCheckbox label,
label { color: #b0a0e0 !important; font-size: 0.82rem !important; }

/* 7. Radio */
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: #d0c8f8 !important; }

/* 8. Buttons */
button[kind="primary"], button[kind="secondary"],
.stButton button, [data-testid="baseButton-primary"],
[data-testid="baseButton-secondary"],
[data-testid="baseButton-secondaryFormSubmit"] {
    background: #2d1060 !important;
    border: 1px solid #7c3aed !important;
    color: #e9d8ff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background 0.2s !important;
}
button[kind="primary"]:hover, button[kind="secondary"]:hover,
.stButton button:hover {
    background: #3d1a80 !important;
    border-color: #a78bfa !important;
    color: #f5f0ff !important;
}

/* 9. Dataframes */
[data-testid="stDataFrame"] iframe { border-radius: 10px !important; }
.dvn-scroller { background: #13132a !important; }
.col_heading, .blank { background: #1e1e3e !important; color: #a78bfa !important; font-weight: 700 !important; }
.data { color: #d0c8f0 !important; background: #13132a !important; }
.row_heading { background: #1e1e3e !important; color: #a78bfa !important; }

/* 10. Expanders */
[data-testid="stExpander"] {
    background: #13132a !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary:hover {
    background: #1a1a3a !important;
}
[data-testid="stExpander"] p {
    color: #c0b0e0 !important;
    font-weight: 600 !important;
}

/* 11. Alert banners */
.stAlert, div[role="alert"] {
    border-radius: 10px !important;
}
[data-testid="stNotification"][kind="success"],
div[data-baseweb="notification"][kind="positive"] {
    background: #052e16 !important;
    border: 1px solid #16a34a88 !important;
    color: #86efac !important;
}
[data-testid="stNotification"][kind="info"] {
    background: #0c1a3a !important;
    border: 1px solid #2563eb88 !important;
    color: #93c5fd !important;
}
[data-testid="stNotification"][kind="warning"] {
    background: #2a1500 !important;
    border: 1px solid #d9770688 !important;
    color: #fcd34d !important;
}
[data-testid="stNotification"][kind="error"] {
    background: #1c0000 !important;
    border: 1px solid #dc262688 !important;
    color: #fca5a5 !important;
}
[data-testid="stNotification"] p,
[data-testid="stNotification"] span,
[data-testid="stNotification"] div,
div[role="alert"] p, div[role="alert"] span { color: inherit !important; }

/* 12. Caption */
[data-testid="stCaptionContainer"] p,
.stCaption { color: #8080b0 !important; }

/* 13. Scrollbar */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#0f1117; }
::-webkit-scrollbar-thumb { background:#7c3aed; border-radius:3px; }

/* ─── CUSTOM COMPONENT CLASSES ────────────────── */

.noc-header {
    background: linear-gradient(90deg,#180535 0%,#0d1b4b 55%,#0a1828 100%);
    border: 1px solid #7c3aed33;
    border-radius: 14px;
    padding: 1.1rem 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
}
.noc-title   { font-size:1.4rem; font-weight:700; color:#ede9fe !important; letter-spacing:.05em; }
.noc-subtitle{ font-size:.7rem;  color:#9080c0 !important; margin-top:3px; letter-spacing:.08em; }
.noc-badge   { background:#7c3aed22; border:1px solid #7c3aed66; border-radius:20px;
               padding:4px 14px; font-size:.72rem; color:#d8b4fe !important; letter-spacing:.06em; }

.section-heading {
    font-size:.72rem; font-weight:700; letter-spacing:.15em;
    color:#a78bfa !important; text-transform:uppercase;
    margin:1.8rem 0 .9rem; padding-bottom:8px;
    border-bottom:1px solid #7c3aed44;
    display:flex; align-items:center; gap:.5rem;
}

.kpi-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:.75rem; margin-bottom:1rem; }
.kpi-card {
    background:#13132a; border:1px solid #2a2a4a; border-radius:12px;
    padding:1rem 1.1rem; position:relative; overflow:hidden;
    transition:border-color .2s,transform .15s;
}
.kpi-card:hover { border-color:#7c3aed99; transform:translateY(-2px); }
.kpi-card::before { content:''; position:absolute; top:0;left:0;right:0; height:3px; }
.kpi-card.purple::before { background:linear-gradient(90deg,#7c3aed,#a78bfa); }
.kpi-card.blue::before   { background:linear-gradient(90deg,#1d4ed8,#60a5fa); }
.kpi-card.green::before  { background:linear-gradient(90deg,#15803d,#4ade80); }
.kpi-card.red::before    { background:linear-gradient(90deg,#b91c1c,#f87171); }
.kpi-card.amber::before  { background:linear-gradient(90deg,#b45309,#fbbf24); }
.kpi-card.teal::before   { background:linear-gradient(90deg,#0f766e,#2dd4bf); }
.kpi-label { font-size:.62rem; font-weight:700; letter-spacing:.12em;
             color:#9090c0 !important; text-transform:uppercase; margin-bottom:8px; }
.kpi-value { font-size:1.6rem; font-weight:700; color:#f0e8ff !important;
             line-height:1; font-variant-numeric:tabular-nums; }
.kpi-unit  { font-size:.68rem; color:#7070a0 !important; margin-top:5px; }
.kpi-status-critical { color:#fca5a5 !important; font-size:1rem; font-weight:700; }
.kpi-status-healing  { color:#6ee7b7 !important; font-size:1rem; font-weight:700; }
.kpi-status-normal   { color:#93c5fd !important; font-size:1rem; font-weight:700; }

.driver-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; margin-bottom:1rem; }
.driver-card { background:#13132a; border:1px solid #2a2a4a; border-radius:12px;
               padding:1rem 1.1rem; border-left:4px solid; }
.driver-card.anomaly { border-left-color:#ef4444; background:#1a0d0d; }
.driver-card.normal  { border-left-color:#22c55e; background:#0d1a0d; }
.driver-rank  { font-size:.62rem; color:#9090b0 !important; letter-spacing:.1em;
                text-transform:uppercase; margin-bottom:5px; font-weight:600; }
.driver-feat  { font-size:1.05rem; font-weight:800; color:#f0e8ff !important;
                margin-bottom:5px; letter-spacing:.05em; }
.driver-dir   { font-size:.75rem; margin-bottom:6px; font-weight:500; }
.driver-score { font-size:.7rem; color:#9090b0 !important; font-family:monospace; }
.driver-bar-bg   { background:#1e1e3a; border-radius:4px; height:5px; margin-top:8px; }
.driver-bar-fill { height:5px; border-radius:4px; transition:width .4s ease; }

.rl-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; margin-bottom:1rem; }
.rl-card  { background:#13132a; border:1px solid #2a2a4a; border-radius:12px; padding:1rem 1.1rem; }
.rl-label { font-size:.62rem; color:#9090b0 !important; letter-spacing:.12em;
            text-transform:uppercase; margin-bottom:8px; font-weight:700; }
.rl-value { font-size:1.3rem; font-weight:700; color:#d8b4fe !important; }
.rl-sub   { font-size:.68rem; color:#7070a0 !important; margin-top:4px; }

.inc-card { background:#180d0d; border:1px solid #ef444422; border-left:4px solid #ef4444;
            border-radius:12px; padding:1.1rem 1.2rem; margin-bottom:.8rem; }
.inc-id   { font-size:.78rem; font-weight:700; color:#fca5a5 !important;
            font-family:monospace; letter-spacing:.05em; }
.inc-meta { font-size:.7rem; color:#9090b0 !important; margin:5px 0; line-height:1.6; }
.inc-cause{ font-size:.78rem; color:#c8b8e8 !important; margin-top:6px; line-height:1.5; }

.hitl-box { background:#1c0808; border:1px solid #ef444455;
            border-radius:14px; padding:1.3rem 1.5rem; margin-bottom:1rem; }
.hitl-title { font-size:1rem; font-weight:700; color:#fca5a5 !important; margin-bottom:10px; }
.hitl-cause { font-size:.8rem; color:#c8b8d8 !important; margin-bottom:6px; line-height:1.5; }
.hitl-rl    { font-size:.8rem; color:#d8b4fe !important; font-weight:500; }

.sla-ok       { background:#14532d33; color:#86efac !important; border:1px solid #22c55e66;
                border-radius:20px; padding:3px 12px; font-size:.7rem; font-weight:700; display:inline-block; }
.sla-warning  { background:#78350f33; color:#fcd34d !important; border:1px solid #f59e0b66;
                border-radius:20px; padding:3px 12px; font-size:.7rem; font-weight:700; display:inline-block; }
.sla-breached { background:#7f1d1d33; color:#fca5a5 !important; border:1px solid #ef444466;
                border-radius:20px; padding:3px 12px; font-size:.7rem; font-weight:700; display:inline-block; }

.login-wrap  { max-width:400px; margin:5rem auto; background:#13132a;
               border:1px solid #2a2a4a; border-radius:18px; padding:2.5rem;
               box-shadow:0 0 40px #7c3aed22; }
.login-title { font-size:1.3rem; font-weight:700; color:#f0e8ff !important; margin-bottom:.3rem; }
.login-sub   { font-size:.75rem; color:#9080b0 !important; margin-bottom:1.5rem; letter-spacing:.05em; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────────
USERS = {"admin": "admin123", "operator": "netops"}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None

if not st.session_state.authenticated:
    st.markdown("""
    <div class="login-wrap">
        <div class="login-title">🛰️ ZeroTouch NOC</div>
        <div class="login-sub">Network Operations Centre · Secure Access</div>
    </div>
    """, unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        u = st.text_input("Username", placeholder="admin or operator")
        p = st.text_input("Password", type="password", placeholder="••••••••")
        if st.button("Sign In →", use_container_width=True):
            if u in USERS and USERS[u] == p:
                st.session_state.authenticated = True
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
defaults = {
    "data":                 pd.DataFrame(),
    "handled_event_id":     None,
    "hitl_active":          False,
    "open_incident_id":     None,
    "pending_incident_id":  None,
    "prev_row":             None,
    "rl_history":           [],
    "ig_history":           {f: [] for f in FEATURE_COLS},
    "last_logged_decision": None,   # FIX-E: track last logged state to avoid spam
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
if "iso_model" not in st.session_state:
    st.session_state.iso_model = load_model()
iso_model = st.session_state.iso_model

# FIX-C: Cache RL agent in session_state — was recreated (disk load) every tick
if "rl_agent" not in st.session_state:
    st.session_state.rl_agent = QLearningAgent()
rl_agent = st.session_state.rl_agent

# ─────────────────────────────────────────────────────────────
# DATA + DECISIONS
# ─────────────────────────────────────────────────────────────

# FIX-D: Skip blocking ping when HITL form is waiting for user input.
#         This makes the HITL submit button respond instantly instead
#         of stalling for 2–8s while a new ping measurement completes.
if not st.session_state.get("hitl_active", False):
    try:
        new_row = generate_network_data()
        st.session_state.data = pd.concat(
            [st.session_state.data, new_row], ignore_index=True
        )
    except Exception as _fetch_err:
        # Network completely unavailable — continue with last known data
        if st.session_state.data.empty:
            st.warning(f"⚠️ Could not reach network: {_fetch_err}. Retrying…")
            time.sleep(3)
            st.rerun()

# Guard: ensure we have at least one usable row before proceeding
if st.session_state.data.empty:
    st.info("⏳ Collecting initial network data — please wait…")
    time.sleep(2)
    st.rerun()

df       = st.session_state.data.tail(200).copy()
iso_model.fit(df[FEATURE_COLS])
df["anomaly"] = iso_model.predict(df[FEATURE_COLS])
latest   = df.iloc[-1]
event_id = latest.name
prev_row = st.session_state.get("prev_row", None)

decision = decide_action(latest)
rl_action_id, rl_action_labels = rl_corrective_action(rl_agent, latest, prev_row)

# FIX-H: Use rl_agent's own normalise() for stress — was an inconsistent 2-feature formula
stress_proxy = normalise(latest)
st.session_state.rl_history.append((rl_agent.steps_trained, rl_action_id, round(stress_proxy, 3)))
st.session_state.rl_history = st.session_state.rl_history[-100:]
st.session_state.prev_row   = latest.copy()

ig_df      = get_ig_attribution(iso_model, df[FEATURE_COLS], FEATURE_COLS)
root_cause = build_root_cause_string(ig_df)

# ─────────────────────────────────────────────────────────────
# HEADER BANNER
# ─────────────────────────────────────────────────────────────
status_colour = {"CRITICAL": "#f87171", "AUTO-HEALING": "#34d399", "NORMAL": "#60a5fa"}.get(decision, "#60a5fa")
st.markdown(f"""
<div class="noc-header">
  <div>
    <div class="noc-title">🛰️ ZeroTouch Network Operations Centre</div>
    <div class="noc-subtitle">Q-LEARNING RL · INTEGRATED GRADIENTS XAI · HITL GOVERNANCE · INCIDENT RESPONSE</div>
  </div>
  <div style="display:flex;gap:0.75rem;align-items:center;">
    <div class="noc-badge">👤 {st.session_state.user.upper()}</div>
    <div class="noc-badge" style="color:{status_colour};border-color:{status_colour}44;">⬤ {decision}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SECTION 1 — KPI STRIP
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Live Telemetry</div>', unsafe_allow_html=True)

status_class = {"CRITICAL": "kpi-status-critical", "AUTO-HEALING": "kpi-status-healing", "NORMAL": "kpi-status-normal"}.get(decision, "kpi-status-normal")
anomaly_pct  = int((df["anomaly"] == -1).sum() / max(len(df), 1) * 100)

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card purple">
    <div class="kpi-label">Latency</div>
    <div class="kpi-value">{float(latest['latency']):.1f}</div>
    <div class="kpi-unit">milliseconds</div>
  </div>
  <div class="kpi-card red">
    <div class="kpi-label">Packet Loss</div>
    <div class="kpi-value">{float(latest['packet_loss']):.2f}</div>
    <div class="kpi-unit">percent</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">Throughput</div>
    <div class="kpi-value">{float(latest['throughput']):.1f}</div>
    <div class="kpi-unit">Mbps</div>
  </div>
  <div class="kpi-card teal">
    <div class="kpi-label">Jitter</div>
    <div class="kpi-value">{float(latest['jitter']):.1f}</div>
    <div class="kpi-unit">milliseconds</div>
  </div>
  <div class="kpi-card blue">
    <div class="kpi-label">Bandwidth</div>
    <div class="kpi-value">{float(latest['bandwidth']):.1f}</div>
    <div class="kpi-unit">Mbps</div>
  </div>
  <div class="kpi-card amber">
    <div class="kpi-label">Network Status</div>
    <div class="kpi-value {status_class}">{decision}</div>
    <div class="kpi-unit">Anomaly rate: {anomaly_pct}% of last 200</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Telemetry chart ───────────────────────────────────────────
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    # FIX-A + FIX-B: Normalise each feature to 0-100% of its expected range
    # so all 5 lines are visible. Old code used raw values on a fixed 0-280 axis
    # which made packet_loss (max 30) invisible and clipped latency spikes > 280ms.
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for feat in FEATURE_COLS:
        colour   = FEAT_COLOURS[feat]
        lo, hi   = FEAT_DISPLAY_RANGES[feat]
        vals     = df[feat].values[-80:]
        norm_vals = np.clip((vals - lo) / max(hi - lo, 1e-9) * 100, 0, 110)
        ax.plot(norm_vals, label=feat, color=colour, linewidth=1.3, alpha=0.9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("% of scale", fontsize=7)
    ax.legend(fontsize=7, loc="upper left", ncol=5)
    ax.set_xlabel("Recent ticks", fontsize=7)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close(fig)   # FIX-F

with col_t2:
    # FIX-K: Guard against divide-by-zero when df is empty/all-same
    normal_count  = int((df["anomaly"] == 1).sum())
    anomaly_count = int((df["anomaly"] == -1).sum())
    fig_d, ax_d = plt.subplots(figsize=(2.4, 2.4))
    if normal_count + anomaly_count > 0:
        ax_d.pie(
            [normal_count, anomaly_count],
            colors=["#16a34a", "#dc2626"],
            startangle=90,
            wedgeprops=dict(width=0.45, edgecolor="#0f1117", linewidth=2),
        )
        ax_d.text(0, 0, f"{anomaly_pct}%\nanomaly", ha="center", va="center",
                  fontsize=9, color="#f87171", fontweight="bold")
    else:
        ax_d.text(0, 0, "No data", ha="center", va="center", fontsize=9, color="#6060a0")
    ax_d.set_title("Last 200 ticks", fontsize=7, color="#6060a0", pad=4)
    plt.tight_layout(pad=0.2)
    st.pyplot(fig_d)
    plt.close(fig_d)   # FIX-F

# ─────────────────────────────────────────────────────────────
# SECTION 2 — RL AGENT
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🤖 Q-Learning Agent</div>', unsafe_allow_html=True)

action_name   = ACTION_NAMES[rl_action_id]
explore_rate  = rl_agent.exploration_rate
steps_trained = rl_agent.steps_trained
action_colour = {"REDUCE": "#f87171", "MAINTAIN": "#34d399", "RESET": "#fbbf24"}.get(action_name, "#c084fc")

st.markdown(f"""
<div class="rl-grid">
  <div class="rl-card">
    <div class="rl-label">Current RL Action</div>
    <div class="rl-value" style="color:{action_colour};">{action_name}</div>
    <div style="font-size:0.7rem;color:#6060a0;margin-top:4px;">{rl_action_labels[0]}</div>
  </div>
  <div class="rl-card">
    <div class="rl-label">Steps Trained</div>
    <div class="rl-value">{steps_trained:,}</div>
    <div class="rl-sub">Q-table updates so far</div>
  </div>
  <div class="rl-card">
    <div class="rl-label">Exploration Rate ε</div>
    <div class="rl-value">{explore_rate:.1f}%</div>
    <div class="rl-sub">Decays toward 5% floor</div>
  </div>
</div>
""", unsafe_allow_html=True)

if len(st.session_state.rl_history) > 5:
    rl_hist = st.session_state.rl_history
    steps   = [h[0] for h in rl_hist]
    actions = [h[1] for h in rl_hist]
    stress  = [h[2] for h in rl_hist]

    col_rl1, col_rl2 = st.columns(2)
    with col_rl1:
        fig_rl, (ax_a, ax_s) = plt.subplots(2, 1, figsize=(5.5, 2.8), sharex=True)
        c_map = {0: "#f87171", 1: "#34d399", 2: "#fbbf24"}
        for aid, colour in c_map.items():
            xs = [steps[i] for i, a in enumerate(actions) if a == aid]
            ys = [aid] * len(xs)
            ax_a.scatter(xs, ys, color=colour, s=22, label=ACTION_NAMES[aid], zorder=3)
        ax_a.set_yticks([0, 1, 2])
        ax_a.set_yticklabels(["REDUCE", "MAINTAIN", "RESET"], fontsize=7)
        ax_a.legend(fontsize=6, loc="upper left")
        ax_a.grid(True, alpha=0.2, linestyle="--")
        ax_a.spines[["top","right"]].set_visible(False)
        ax_a.set_title("Action timeline", fontsize=8, color="#a0a0c0")

        ax_s.fill_between(steps, stress, alpha=0.25, color="#7c3aed")
        ax_s.plot(steps, stress, color="#a78bfa", linewidth=1.2)
        ax_s.set_ylabel("Stress", fontsize=7)
        ax_s.set_xlabel("Steps", fontsize=7)
        ax_s.grid(True, alpha=0.2, linestyle="--")
        ax_s.spines[["top","right"]].set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig_rl)
        plt.close(fig_rl)   # FIX-F

    with col_rl2:
        policy_df = rl_agent.get_policy_summary()
        fig_qt, ax_qt = plt.subplots(figsize=(4.5, 2.2))
        q_matrix = policy_df[["Q(REDUCE)", "Q(MAINTAIN)", "Q(RESET)"]].values.T
        im = ax_qt.imshow(q_matrix, aspect="auto", cmap="RdYlGn", interpolation="nearest")
        ax_qt.set_yticks([0, 1, 2])
        ax_qt.set_yticklabels(["REDUCE", "MAINTAIN", "RESET"], fontsize=8)
        ax_qt.set_xlabel("State bucket (0=calm → 49=critical)", fontsize=7)
        ax_qt.set_title("Learned Q-values", fontsize=8, color="#a0a0c0")
        plt.colorbar(im, ax=ax_qt, label="Q-value")
        fig_qt.tight_layout(pad=0.5)
        st.pyplot(fig_qt)
        plt.close(fig_qt)   # FIX-F (was already closed — kept for consistency)

# ─────────────────────────────────────────────────────────────
# SECTION 3 — HITL GOVERNANCE
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading"> 🚩 Anomaly Governance</div>', unsafe_allow_html=True)

if decision == "CRITICAL" and st.session_state.handled_event_id != event_id:
    st.session_state.hitl_active = True
    # FIX-J: root_cause now includes percentage impact from explainability.py fix
    st.markdown(f"""
    <div class="hitl-box">
      <div class="hitl-title"> 👾 Critical Anomaly Detected — Human Decision Required</div>
      <div class="hitl-cause"><b>Root Cause (IG):</b> {root_cause}</div>
      <div class="hitl-rl"><b>RL Recommends:</b> {action_name} — {rl_action_labels[0]}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("hitl_form"):
        choice = st.radio(
            "Select your decision:",
            ["Approve RL recommendation", "Override: Apply rule-based action",
             "Deny (Skip this event)", "Escalate to Senior Ops"],
            horizontal=True,
        )
        submitted = st.form_submit_button("Submit Decision", use_container_width=True)

    if submitted:
        if "Approve RL" in choice:
            inc_id = open_incident("CRITICAL", root_cause, st.session_state.user)
            log_event(latest, "APPROVED-RL", rl_action_labels)
            st.session_state.open_incident_id    = inc_id
            st.session_state.pending_incident_id = inc_id
        elif "Override" in choice:
            fallback = corrective_action(latest)
            inc_id   = open_incident("CRITICAL", root_cause, st.session_state.user)
            log_event(latest, "APPROVED-OVERRIDE", fallback)
            st.session_state.open_incident_id    = inc_id
            st.session_state.pending_incident_id = inc_id
        elif "Escalate" in choice:
            inc_id = open_incident("CRITICAL", root_cause, st.session_state.user)
            escalate_incident(inc_id, escalate_to="Senior Ops")
            log_event(latest, "ESCALATED", [f"Escalated by {st.session_state.user}"])
            st.session_state.open_incident_id    = inc_id
            st.session_state.pending_incident_id = inc_id
        else:
            log_event(latest, "DENIED", [f"Denied by {st.session_state.user}"])

        st.session_state.handled_event_id      = event_id
        st.session_state.hitl_active           = False
        st.session_state.last_logged_decision  = "CRITICAL"
        st.rerun()

elif decision == "AUTO-HEALING":
    # FIX-E: Only log when decision transitions into AUTO-HEALING, not every tick
    if st.session_state.last_logged_decision != "AUTO-HEALING":
        log_event(latest, "AUTO-HEALING", rl_action_labels)
        st.session_state.last_logged_decision = "AUTO-HEALING"
    st.info(f" Auto-healing active — {rl_action_labels[0]}")
else:
    if st.session_state.last_logged_decision not in (None, "NORMAL"):
        st.session_state.last_logged_decision = "NORMAL"
    st.success(" Network operating normally — no intervention required.")

# ─────────────────────────────────────────────────────────────
# SECTION 4 — EXPLAINABILITY DASHBOARD
# ─────────────────────────────────────────────────────────────
method_used   = ig_df["Method"].iloc[0] if "Method" in ig_df.columns else "IG"
method_colour = "#a78bfa" if method_used == "IG" else "#fbbf24"
method_tip    = "Exact path-based attribution" if method_used == "IG" else "Warming up — IG activates after ~20 ticks"
st.markdown(
    f'<div class="section-heading">🔍 Explainability &nbsp;'
    f'<span style="font-size:0.65rem;background:{method_colour}22;color:{method_colour};'
    f'border:1px solid {method_colour}55;border-radius:10px;padding:2px 10px;">{method_used}</span>'
    f'<span style="font-size:0.6rem;color:#9080c0;margin-left:8px;">{method_tip}</span></div>',
    unsafe_allow_html=True
)

total_abs = float(ig_df["Abs_Attribution"].sum())

# FIX-I: Show warm-up progress bar until IG has real data; hide driver cards
_rows = len(ig_df)

if total_abs < 1e-6:
    # Warm-up state — IG not yet active
    ticks_so_far = len(df)
    warmup_needed = 20
    progress_val  = min(ticks_so_far / warmup_needed, 1.0)
    st.progress(progress_val, text=f"⏳ IG attribution warms up in {max(0, warmup_needed - ticks_so_far)} more tick(s)…")
else:
    top1 = ig_df.iloc[0]
    top2 = ig_df.iloc[1] if _rows > 1 else ig_df.iloc[0]
    top3 = ig_df.iloc[2] if _rows > 2 else top2

    max_abs = float(ig_df["Abs_Attribution"].max()) if total_abs > 1e-6 else 1.0

    def _pct(row):
        return round(float(row["Abs_Attribution"]) / total_abs * 100, 1) if total_abs > 1e-6 else 0.0

    def _driver_card_html(row, rank_label):
        ig_val  = float(row["IG_Attribution"])
        abs_val = float(row["Abs_Attribution"])
        cls     = "anomaly" if ig_val < 0 else "normal"
        d_col   = "#f87171" if ig_val < 0 else "#4ade80"
        bar_w   = int(abs_val / max_abs * 100) if max_abs > 1e-6 else 0
        bar_c   = "#dc2626" if ig_val < 0 else "#16a34a"
        return (
            '<div class="driver-card ' + cls + '">'
            '<div class="driver-rank">' + rank_label + '</div>'
            '<div class="driver-feat">' + str(row["Feature"]).upper() + '</div>'
            '<div class="driver-dir" style="color:' + d_col + ';">' + str(row["Direction"]) + ' &nbsp;·&nbsp; ' + str(_pct(row)) + '% impact</div>'
            '<div class="driver-score">IG score: ' + f"{ig_val:+.4f}" + '</div>'
            '<div class="driver-bar-bg"><div class="driver-bar-fill" style="width:' + str(bar_w) + '%;background:' + bar_c + ';"></div></div>'
            '</div>'
        )

    st.markdown(
        '<div class="driver-grid">'
        + _driver_card_html(top1, "① Primary Driver")
        + _driver_card_html(top2, "② Secondary Driver")
        + _driver_card_html(top3, "③ Tertiary Driver")
        + '</div>',
        unsafe_allow_html=True,
    )

# Charts row
col_x1, col_x2, col_x3 = st.columns(3)

with col_x1:
    # FIX-A: Use FEAT_COLOURS for consistent bar colours
    bar_colours = [FEAT_COLOURS.get(f, "#888") for f in ig_df["Feature"]]
    # Tint red/green by direction
    bar_colours = ["#dc2626" if v < 0 else "#16a34a" for v in ig_df["IG_Attribution"]]
    fig_b, ax_b = plt.subplots(figsize=(3.8, 2.8))
    bars = ax_b.barh(ig_df["Feature"][::-1], ig_df["IG_Attribution"][::-1],
                     color=bar_colours[::-1], edgecolor="#0f1117", linewidth=0.5, height=0.55)
    ax_b.axvline(0, color="#404060", linewidth=1.0, linestyle="--")
    for bar, val in zip(bars, ig_df["IG_Attribution"][::-1]):
        x = bar.get_width()
        ax_b.text(x + (0.0003 if x >= 0 else -0.0003), bar.get_y() + bar.get_height()/2,
                  f"{val:+.4f}", va="center", ha="left" if x >= 0 else "right", fontsize=6.5, color="#a0a0c0")
    ax_b.set_title("Feature attribution", fontsize=8, color="#a0a0c0")
    ax_b.spines[["top","right"]].set_visible(False)
    ax_b.tick_params(labelsize=7)
    ax_b.grid(True, alpha=0.15, axis="x", linestyle="--")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig_b)
    plt.close(fig_b)   # FIX-F

with col_x2:
    sorted_ig = ig_df.sort_values("IG_Attribution").reset_index(drop=True)
    running, bottoms, widths, wcolours = 0.0, [], [], []
    for _, row in sorted_ig.iterrows():
        v = float(row["IG_Attribution"])
        bottoms.append(running); widths.append(v)
        wcolours.append("#dc2626" if v < 0 else "#16a34a")
        running += v
    fig_w, ax_w = plt.subplots(figsize=(3.8, 2.8))
    ax_w.barh(sorted_ig["Feature"], widths, left=bottoms, color=wcolours,
              edgecolor="#0f1117", linewidth=0.5, height=0.55)
    ax_w.axvline(0, color="#404060", linewidth=1.0, linestyle="--")
    ax_w.set_title("Waterfall accumulation", fontsize=8, color="#a0a0c0")
    ax_w.spines[["top","right"]].set_visible(False)
    ax_w.tick_params(labelsize=7)
    ax_w.grid(True, alpha=0.15, axis="x", linestyle="--")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig_w)
    plt.close(fig_w)   # FIX-F

with col_x3:
    # FIX-G: Use mean |IG| over full ig_history — was showing only the latest tick snapshot.
    # The radar now reflects aggregated historical impact, not a single noisy data point.
    feats    = list(ig_df["Feature"])
    abs_vals = []
    for feat in feats:
        hist = st.session_state.ig_history.get(feat, [])
        if hist:
            abs_vals.append(float(np.mean(np.abs(hist))))
        else:
            row_val = ig_df.loc[ig_df["Feature"] == feat, "Abs_Attribution"]
            abs_vals.append(float(row_val.iloc[0]) if not row_val.empty else 0.0)

    N             = len(feats)
    angles        = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    abs_vals_plot = abs_vals + [abs_vals[0]]
    fig_r, ax_r = plt.subplots(figsize=(3.8, 2.8), subplot_kw=dict(polar=True))
    ax_r.plot(angles, abs_vals_plot, color="#a78bfa", linewidth=1.5)
    ax_r.fill(angles, abs_vals_plot, color="#7c3aed", alpha=0.20)
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(feats, fontsize=7, color="#a0a0c0")
    ax_r.set_yticklabels([])
    ax_r.set_title("Impact radar (historical avg)", fontsize=8, color="#a0a0c0", pad=12)
    ax_r.grid(True, alpha=0.3, color="#3d3d5c")
    fig_r.patch.set_facecolor("#0f1117")
    ax_r.set_facecolor("#0f1117")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig_r)
    plt.close(fig_r)   # FIX-F

# IG trend over time — update history
for feat in FEATURE_COLS:
    row_val = ig_df.loc[ig_df["Feature"] == feat, "IG_Attribution"]
    val = float(row_val.iloc[0]) if not row_val.empty else 0.0
    st.session_state.ig_history[feat].append(val)
    st.session_state.ig_history[feat] = st.session_state.ig_history[feat][-60:]

if any(len(v) > 3 for v in st.session_state.ig_history.values()):
    fig_trend, ax_t = plt.subplots(figsize=(10, 2.0))
    # FIX-A: Use FEAT_COLOURS (was a separate inline dict with different colours)
    for feat in FEATURE_COLS:
        vals = st.session_state.ig_history[feat]
        if vals:
            ax_t.plot(vals, label=feat, color=FEAT_COLOURS[feat],
                      linewidth=1.2, alpha=0.9)
    ax_t.axhline(0, color="#404060", linewidth=0.8, linestyle="--")
    ax_t.set_title("IG Attribution over time — last 60 ticks", fontsize=8, color="#a0a0c0")
    ax_t.set_ylabel("IG score", fontsize=7)
    ax_t.legend(fontsize=7, loc="upper right", ncol=5)
    ax_t.grid(True, alpha=0.15, linestyle="--")
    ax_t.spines[["top","right"]].set_visible(False)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig_trend)
    plt.close(fig_trend)   # FIX-F

with st.expander(" Full Attribution Table", expanded=False):
    display_df = ig_df.copy()
    display_df["Live value"] = [round(float(latest[f]), 3) for f in display_df["Feature"]]
    if total_abs > 1e-6:
        display_df["Contribution %"] = (
            (display_df["Abs_Attribution"] / total_abs * 100).round(1).astype(str) + "%"
        )
    else:
        display_df["Contribution %"] = ["—"] * len(display_df)
    st.table(
        display_df[["Feature", "Live value", "IG_Attribution", "Abs_Attribution", "Direction", "Contribution %"]]
        .rename(columns={"IG_Attribution": "IG score", "Abs_Attribution": "Magnitude", "Direction": "Effect"})
        .reset_index(drop=True)
    )

# ─────────────────────────────────────────────────────────────
# SECTION 5 — INCIDENT RESPONSE
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading"> Incident Response</div>', unsafe_allow_html=True)

if st.session_state.pending_incident_id:
    st.success(f" Incident **{st.session_state.pending_incident_id}** created and tracked below.")
    st.session_state.pending_incident_id = None

open_incs = get_open_incidents()

if open_incs:
    st.markdown(f'<div style="font-size:0.78rem;font-weight:600;color:#fca5a5;margin-bottom:0.75rem;">⬤ {len(open_incs)} open incident(s) — SLA tracking active</div>', unsafe_allow_html=True)
    for inc in open_incs:
        elapsed, sla_label = get_sla_status(inc)
        sla_html = f'<span class="sla-{sla_label.lower()}">{sla_label} — {elapsed} min elapsed</span>'
        st.markdown(f"""
        <div class="inc-card">
          <div style="display:flex;justify-content:space-between;align-items:start;">
            <div>
              <div class="inc-id">{inc['incident_id']}</div>
              <div class="inc-meta">Opened: {inc['opened_at']} &nbsp;·&nbsp; Assigned: {inc['assigned_to']} &nbsp;·&nbsp; Status: {inc['status']}</div>
              <div class="inc-cause">{inc['root_cause']}</div>
            </div>
            <div>{sla_html}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("▶ In Progress", key=f"prog_{inc['incident_id']}"):
                update_status(inc["incident_id"], "IN_PROGRESS", st.session_state.user)
                st.rerun()
        with col_b:
            note = st.text_input("Resolution note", key=f"note_{inc['incident_id']}", label_visibility="collapsed", placeholder="Resolution note...")
            if st.button("🔹 Resolve", key=f"res_{inc['incident_id']}"):
                resolve_incident(inc["incident_id"], note, st.session_state.user)
                log_event(latest, "RESOLVED", [f"{inc['incident_id']} resolved: {note}"])
                st.rerun()
        with col_c:
            esc = st.text_input("Escalate to", key=f"esc_tgt_{inc['incident_id']}", label_visibility="collapsed", placeholder="Escalate to...")
            if st.button("🔺 Escalate", key=f"esc_{inc['incident_id']}"):
                escalate_incident(inc["incident_id"], esc or "Senior Ops")
                st.rerun()
else:
    st.success(" No open incidents.")

with st.expander(" Incident History (last 10)", expanded=True):
    all_incs = get_all_incidents()[:10]
    if all_incs:
        _inc_raw = pd.DataFrame(all_incs)
        _wanted  = ["incident_id","opened_at","severity","status",
                    "assigned_to","resolved_at","resolution_note"]
        _cols = [c for c in _wanted if c in _inc_raw.columns]
        st.table(_inc_raw[_cols].reset_index(drop=True))
    else:
        st.info("No incidents recorded yet. Incidents appear here after HITL approval.")

# ─────────────────────────────────────────────────────────────
# SECTION 6 — AUDIT LOG
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading"> Audit & Traceability</div>', unsafe_allow_html=True)

if os.path.exists(AUDIT_FILE):
    try:
        log_df = pd.read_csv(AUDIT_FILE, header=None)
        _audit_cols = ["Time","Latency","PacketLoss","Throughput","Decision","Details"]
        log_df.columns = _audit_cols[:len(log_df.columns)]
        st.table(log_df.tail(10).reset_index(drop=True))
    except Exception as _e:
        st.warning(f"Audit log could not be parsed: {_e}")
else:
    st.info("Audit log will appear here once decisions are recorded.")

# ─────────────────────────────────────────────────────────────
# AUTO REFRESH — only when not waiting for HITL input
# ─────────────────────────────────────────────────────────────
if not st.session_state.hitl_active:
    time.sleep(2)
    st.rerun()
