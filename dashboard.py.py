# ============================================
# DASHBOARD.PY — ULTRA PREMIUM FRAUD DETECTION
# ============================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Credit Card Fraud Detector By Naef NazarI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — DARK GLASSMORPHISM THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

  /* ── Root reset ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #060818 !important;
    color: #e2e8f0 !important;
  }

  /* ── Main background ── */
  .stApp {
    background: radial-gradient(ellipse at 20% 10%, #1a0533 0%, #060818 50%, #000d1a 100%) !important;
    background-attachment: fixed !important;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f1e 0%, #0a0c1a 100%) !important;
    border-right: 1px solid rgba(139,92,246,0.2) !important;
  }
  section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1rem !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; max-width: 1400px; }

  /* ── KPI Cards ── */
  .kpi-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    text-align: center;
  }
  .kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 48px rgba(139,92,246,0.2), inset 0 1px 0 rgba(255,255,255,0.12);
  }
  .kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.5rem;
  }
  .kpi-value {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
  }
  .kpi-value.danger {
    background: linear-gradient(135deg, #f87171, #fb923c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .kpi-value.success {
    background: linear-gradient(135deg, #34d399, #6ee7b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .kpi-value.warning {
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .kpi-delta {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.3rem;
  }

  /* ── Section headers ── */
  .section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(139,92,246,0.2);
  }
  .section-header h3 {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #c4b5fd;
    margin: 0;
  }
  .section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, #a78bfa, #818cf8);
    box-shadow: 0 0 8px rgba(167,139,250,0.8);
    flex-shrink: 0;
  }

  /* ── Confusion matrix ── */
  .cm-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 1rem;
  }
  .cm-cell {
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(12px);
  }
  .cm-cell.tn { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.3); }
  .cm-cell.fp { background: rgba(251,191,36,0.12);  border: 1px solid rgba(251,191,36,0.3); }
  .cm-cell.fn { background: rgba(248,113,113,0.15); border: 1px solid rgba(248,113,113,0.4); }
  .cm-cell.tp { background: rgba(99,102,241,0.15);  border: 1px solid rgba(99,102,241,0.4); }
  .cm-cell-label { font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; color: #94a3b8; font-weight: 600; }
  .cm-cell-value { font-size: 2.8rem; font-weight: 800; line-height: 1.1; margin: 0.3rem 0; }
  .cm-cell.tn .cm-cell-value { color: #34d399; }
  .cm-cell.fp .cm-cell-value { color: #fbbf24; }
  .cm-cell.fn .cm-cell-value { color: #f87171; }
  .cm-cell.tp .cm-cell-value { color: #818cf8; }
  .cm-cell-desc { font-size: 0.7rem; color: #64748b; }

  /* ── Risk badge ── */
  .badge-high   { background: rgba(239,68,68,0.2);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.4);  border-radius: 999px; padding: 2px 10px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; }
  .badge-med    { background: rgba(251,191,36,0.2); color: #fde68a; border: 1px solid rgba(251,191,36,0.4); border-radius: 999px; padding: 2px 10px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; }
  .badge-low    { background: rgba(52,211,153,0.2); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.4); border-radius: 999px; padding: 2px 10px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; }

  /* ── Glassmorphism panel ── */
  .glass-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.5rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
  }

  /* ── Sidebar controls ── */
  .sidebar-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 0.3rem;
  }

  /* ── Alert box ── */
  .alert-high {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border: 1px solid rgba(239,68,68,0.5);
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #fca5a5;
    font-weight: 600;
  }
  .alert-med {
    background: linear-gradient(135deg, rgba(251,191,36,0.15), rgba(245,158,11,0.08));
    border: 1px solid rgba(251,191,36,0.5);
    border-left: 4px solid #fbbf24;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #fde68a;
    font-weight: 600;
  }
  .alert-low {
    background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(16,185,129,0.08));
    border: 1px solid rgba(52,211,153,0.5);
    border-left: 4px solid #34d399;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #6ee7b7;
    font-weight: 600;
  }

  /* ── Streamlit widget overrides ── */
  .stSlider > div > div > div > div { background: #7c3aed !important; }
  .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
  }
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.6) !important;
  }
  div[data-testid="stNumberInput"] input,
  div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
  }
  .stDataFrame { border-radius: 12px; overflow: hidden; }

  /* ── Logo / Header ── */
  .hero-header {
    background: linear-gradient(135deg, rgba(124,58,237,0.15) 0%, rgba(99,102,241,0.08) 100%);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    backdrop-filter: blur(20px);
  }
  .hero-title {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #818cf8 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .hero-subtitle {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.2rem;
    font-weight: 500;
    letter-spacing: 0.05em;
  }
  .live-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #34d399;
    letter-spacing: 0.08em;
  }
  .live-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #34d399;
    animation: pulse-dot 1.5s infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #060818; }
  ::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:2rem;">
      <div style="font-size:2.5rem; margin-bottom:0.3rem;">🛡️</div>
      <div style="font-size:1.1rem; font-weight:800; color:#a78bfa; letter-spacing:0.05em;">CREDIT CARD FRAUD DETECTOR BY NAEF NAZAR</div>
      <div style="font-size:0.7rem; color:#475569; margin-top:0.2rem;">Real-Time Detection Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">⚖️ Decision Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("", 0.0, 1.0, 0.35, key="thresh", label_visibility="collapsed")
    st.markdown(f'<div style="text-align:center; font-size:1.6rem; font-weight:800; color:#a78bfa; margin:-0.5rem 0 1rem;">{threshold:.2f}</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-label">💸 Cost Parameters</div>', unsafe_allow_html=True)
    FN_cost = st.number_input("Cost of Missed Fraud (FN)", value=10000, step=500)
    FP_cost = st.number_input("Cost of False Alarm (FP)", value=100, step=10)

    st.divider()

    st.markdown('<div class="sidebar-label">🔬 Sample Size</div>', unsafe_allow_html=True)
    sample_size = st.slider("", 10, 500, 50, key="sample", label_visibility="collapsed")
    st.markdown(f'<div style="text-align:center; font-size:1.4rem; font-weight:700; color:#818cf8; margin:-0.5rem 0 1rem;">{sample_size} txns</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="font-size:0.65rem; color:#334155; text-align:center;">Model endpoint: localhost:5000</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    df = pd.read_csv("creditcard.csv")
    data_loaded = True
except FileNotFoundError:
    st.error("⚠️  `creditcard.csv` not found. Place it in the same directory.")
    st.stop()


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero-header">
  <div>
    <div class="hero-title">🛡️ Fraud Detection Decision Center</div>
    <div class="hero-subtitle">REAL-TIME TRANSACTION RISK INTELLIGENCE</div>
  </div>
  <div class="live-pill">
    <div class="live-dot"></div>
    LIVE · {sample_size} SAMPLES
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SAMPLE & PREDICT
# ─────────────────────────────────────────────
sample_df = df.sample(sample_size, random_state=None)
X_sample  = sample_df.drop("Class", axis=1)
y_true    = sample_df["Class"].values

probs = []
api_error = False

for _, row in X_sample.iterrows():
    try:
        res = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"features": row.tolist()},
            timeout=2
        )
        probs.append(res.json()["fraud_probability"])
    except Exception:
        api_error = True
        probs.append(float(np.random.beta(0.5, 4)))  # fallback demo distribution

if api_error:
    st.markdown('<div style="background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.3);border-radius:10px;padding:0.6rem 1rem;font-size:0.8rem;color:#fde68a;margin-bottom:1rem;">⚠️  API unreachable — displaying simulated probabilities for demo.</div>', unsafe_allow_html=True)

probs  = np.array(probs)
y_pred = (probs > threshold).astype(int)

TP = int(np.sum((y_true == 1) & (y_pred == 1)))
TN = int(np.sum((y_true == 0) & (y_pred == 0)))
FP = int(np.sum((y_true == 0) & (y_pred == 1)))
FN = int(np.sum((y_true == 1) & (y_pred == 0)))

precision    = TP / (TP + FP) if (TP + FP) > 0 else 0
recall       = TP / (TP + FN) if (TP + FN) > 0 else 0
f1           = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
total_cost   = FN_cost * FN + FP_cost * FP
fraud_rate   = float(np.sum(y_pred)) / sample_size * 100


# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, cls="", delta=""):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value {cls}">{value}</div>
      <div class="kpi-delta">{delta}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "⚡ Precision",    f"{precision:.1%}", "success" if precision > 0.7 else "danger",  "Positive Predictive Value")
kpi(k2, "🎯 Recall",       f"{recall:.1%}",    "success" if recall    > 0.7 else "danger",  "True Positive Rate")
kpi(k3, "📊 F1 Score",     f"{f1:.3f}",        "success" if f1        > 0.7 else "warning", "Harmonic Mean")
kpi(k4, "🚨 Fraud Rate",   f"{fraud_rate:.1f}%","danger" if fraud_rate > 5   else "success","Of sampled transactions")
kpi(k5, "💰 Total Cost",   f"${total_cost:,.0f}","danger" if total_cost > 50000 else "warning", f"FN×{FN} + FP×{FP}")


# ─────────────────────────────────────────────
# ROW 2 — Confusion Matrix + Probability Dist
# ─────────────────────────────────────────────
st.markdown('<div class="section-header"><div class="section-dot"></div><h3>Model Performance</h3></div>', unsafe_allow_html=True)

left, right = st.columns([1, 1.6])

with left:
    st.markdown(f"""
    <div class="cm-grid">
      <div class="cm-cell tn">
        <div class="cm-cell-label">True Negative</div>
        <div class="cm-cell-value">{TN}</div>
        <div class="cm-cell-desc">Correctly rejected</div>
      </div>
      <div class="cm-cell fp">
        <div class="cm-cell-label">False Positive</div>
        <div class="cm-cell-value">{FP}</div>
        <div class="cm-cell-desc">False alarms</div>
      </div>
      <div class="cm-cell fn">
        <div class="cm-cell-label">False Negative</div>
        <div class="cm-cell-value">{FN}</div>
        <div class="cm-cell-desc">Missed fraud ⚠️</div>
      </div>
      <div class="cm-cell tp">
        <div class="cm-cell-label">True Positive</div>
        <div class="cm-cell-value">{TP}</div>
        <div class="cm-cell-desc">Caught fraud ✓</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    fraud_mask  = y_true == 1
    legit_mask  = y_true == 0

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=probs[legit_mask], name="Legitimate",
        marker_color="rgba(99,102,241,0.7)",
        xbins=dict(size=0.05),
        hovertemplate="P=%{x:.2f}<br>Count=%{y}<extra>Legit</extra>"
    ))
    fig_dist.add_trace(go.Histogram(
        x=probs[fraud_mask], name="Fraud",
        marker_color="rgba(239,68,68,0.8)",
        xbins=dict(size=0.05),
        hovertemplate="P=%{x:.2f}<br>Count=%{y}<extra>Fraud</extra>"
    ))
    fig_dist.add_vline(
        x=threshold,
        line_dash="dash", line_color="#fbbf24", line_width=2,
        annotation_text=f" Threshold {threshold:.2f}",
        annotation_font_color="#fbbf24",
        annotation_position="top right"
    )
    fig_dist.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Fraud Probability Distribution", font=dict(color="#c4b5fd", size=13), x=0),
        barmode="overlay",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        height=280,
    )
    fig_dist.update_traces(opacity=0.8)
    st.plotly_chart(fig_dist, use_container_width=True)


# ─────────────────────────────────────────────
# ROW 3 — Gauge + Cost Breakdown
# ─────────────────────────────────────────────
g1, g2 = st.columns(2)

with g1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(recall * 100, 1),
        number=dict(suffix="%", font=dict(color="#a78bfa", size=40)),
        delta=dict(reference=70, valueformat=".1f", suffix="%"),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#475569", tickfont=dict(color="#64748b")),
            bar=dict(color="rgba(167,139,250,0.9)", thickness=0.3),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 40],  color="rgba(239,68,68,0.15)"),
                dict(range=[40, 70], color="rgba(251,191,36,0.12)"),
                dict(range=[70, 100],color="rgba(52,211,153,0.12)"),
            ],
            threshold=dict(line=dict(color="#fbbf24", width=3), thickness=0.75, value=70)
        ),
        title=dict(text="Recall Score", font=dict(color="#c4b5fd", size=13))
    ))
    fig_gauge.update_layout(**PLOTLY_LAYOUT, height=250)
    st.plotly_chart(fig_gauge, use_container_width=True)

with g2:
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Bar(
        x=["False Negatives (Missed Fraud)", "False Positives (False Alarms)"],
        y=[FN_cost * FN, FP_cost * FP],
        marker=dict(
            color=["rgba(239,68,68,0.8)", "rgba(251,191,36,0.7)"],
            line=dict(color=["#ef4444", "#fbbf24"], width=1.5)
        ),
        text=[f"${FN_cost*FN:,.0f}", f"${FP_cost*FP:,.0f}"],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13, family="Inter"),
        hovertemplate="%{x}<br><b>$%{y:,.0f}</b><extra></extra>"
    ))
    fig_cost.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Cost Breakdown", font=dict(color="#c4b5fd", size=13), x=0),
        height=250,
        showlegend=False,
    )
    st.plotly_chart(fig_cost, use_container_width=True)


# ─────────────────────────────────────────────
# TOP RISK TRANSACTIONS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header"><div class="section-dot"></div><h3>Top Risk Transactions</h3></div>', unsafe_allow_html=True)

result_df = sample_df.copy()
result_df["fraud_probability"] = probs
result_df = result_df.sort_values(by="fraud_probability", ascending=False).head(10).reset_index(drop=True)

def risk_badge(p):
    if p >= threshold * 1.5: return "🔴 HIGH"
    elif p >= threshold:     return "🟡 MEDIUM"
    else:                    return "🟢 LOW"

display_df = pd.DataFrame({
    "Rank":              [f"#{i+1}" for i in range(len(result_df))],
    "Fraud Probability": result_df["fraud_probability"].apply(lambda x: f"{x:.4f}"),
    "Risk Level":        result_df["fraud_probability"].apply(risk_badge),
    "Amount (V1)":       result_df["V1"].apply(lambda x: f"{x:.3f}") if "V1" in result_df else ["—"]*len(result_df),
    "True Label":        result_df["Class"].apply(lambda x: "🚨 FRAUD" if x == 1 else "✅ LEGIT"),
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank":              st.column_config.TextColumn("Rank", width=60),
        "Fraud Probability": st.column_config.TextColumn("Fraud Probability", width=140),
        "Risk Level":        st.column_config.TextColumn("Risk Level", width=120),
        "True Label":        st.column_config.TextColumn("True Label", width=120),
    }
)


# ─────────────────────────────────────────────
# MANUAL TEST
# ─────────────────────────────────────────────
st.markdown('<div class="section-header"><div class="section-dot"></div><h3>Manual Transaction Test</h3></div>', unsafe_allow_html=True)

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

manual_input = st.text_area(
    "Enter 30 feature values (comma-separated)",
    placeholder="-1.359807134,-0.072781173,2.536346738,...",
    height=90
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_manual = st.button("🔍  Analyze Transaction", use_container_width=True)
with col_info:
    st.markdown('<div style="font-size:0.75rem; color:#475569; padding-top:0.7rem;">Paste exactly 30 PCA-transformed features matching your model input schema.</div>', unsafe_allow_html=True)

if run_manual:
    try:
        features = list(map(float, manual_input.strip().split(",")))
        if len(features) != 30:
            st.error(f"Expected 30 features — got {len(features)}.")
        else:
            try:
                res = requests.post("http://127.0.0.1:5000/predict", json={"features": features}, timeout=2)
                result = res.json()
                prob   = result.get("fraud_probability", 0)
                risk   = result.get("risk", "LOW")
            except Exception:
                prob = float(np.random.beta(2, 5))
                risk = "HIGH" if prob > 0.6 else ("MEDIUM" if prob > 0.35 else "LOW")
                result = {"fraud_probability": prob, "risk": risk, "note": "API demo mode"}

            # Probability gauge
            bar_pct = int(prob * 100)
            bar_color = "#ef4444" if risk == "HIGH" else ("#fbbf24" if risk == "MEDIUM" else "#34d399")

            st.markdown(f"""
            <div style="margin:1.2rem 0 0.5rem;">
              <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#94a3b8; margin-bottom:0.4rem;">
                <span>Fraud Probability</span><span style="color:{bar_color}; font-weight:700;">{prob:.4f}</span>
              </div>
              <div style="background:rgba(255,255,255,0.06); border-radius:999px; height:10px; overflow:hidden;">
                <div style="width:{bar_pct}%; height:100%; background:linear-gradient(90deg, {bar_color}99, {bar_color}); border-radius:999px; transition:width 0.5s ease;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if risk == "HIGH":
                st.markdown(f'<div class="alert-high">🚨 HIGH RISK — Probability: {prob:.4f} · Recommend immediate review</div>', unsafe_allow_html=True)
            elif risk == "MEDIUM":
                st.markdown(f'<div class="alert-med">⚠️ MEDIUM RISK — Probability: {prob:.4f} · Flag for secondary screening</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-low">✅ LOW RISK — Probability: {prob:.4f} · Transaction appears legitimate</div>', unsafe_allow_html=True)

            with st.expander("📦 Raw API Response"):
                st.json(result)

    except Exception as e:
        st.error(f"Invalid input: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding:1rem; border-top:1px solid rgba(255,255,255,0.05); font-size:0.7rem; color:#334155; letter-spacing:0.05em;">
 · CREDIT CARD FRAUD DETECTOR BY NAEF NAZAR · DECISION DASHBOARD · POWERED BY REAL-TIME ML INFERENCE
</div>
""", unsafe_allow_html=True)