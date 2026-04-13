import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
 
.top-banner {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 14px; padding: 22px 30px; margin-bottom: 24px;
    display: flex; align-items: center; gap: 16px;
}
.top-banner h1 { color: #e8f4f8; font-size: 24px; font-weight: 600; margin: 0; letter-spacing: -0.3px; }
.top-banner p  { color: #94b8c4; font-size: 12px; margin: 3px 0 0 0; }
.badge {
    background: rgba(255,255,255,0.08); color: #7ecef5; font-size: 10px;
    font-family: 'DM Mono', monospace; padding: 2px 8px; border-radius: 20px;
    border: 1px solid rgba(126,206,245,0.25); display: inline-block; margin-top: 5px;
}
.section-header {
    font-size: 11px; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase;
    color: #94a3b8; margin: 22px 0 10px 0; padding-bottom: 5px;
    border-bottom: 1px solid #e8edf2;
}
.score-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 8px;
}
.score-card .label { font-size: 10px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 5px; }
.score-card .value { font-size: 26px; font-weight: 600; color: #1e293b; font-family: 'DM Mono', monospace; line-height: 1; }
.score-card .bar-outer { background: #e2e8f0; border-radius: 4px; height: 5px; margin-top: 7px; overflow: hidden; }
.score-card .bar-inner { height: 5px; border-radius: 4px; }
 
.risk-medium { background:#fffbeb; border:1.5px solid #fcd34d; border-radius:10px; padding:14px 18px; }
.risk-low    { background:#f0fdf4; border:1.5px solid #86efac; border-radius:10px; padding:14px 18px; }
.risk-label  { font-size: 10px; font-weight:600; letter-spacing:0.07em; text-transform:uppercase; opacity:0.6; }
.risk-medium .risk-value { font-size:20px; font-weight:600; color:#d97706; margin-top:3px; }
.risk-low    .risk-value { font-size:20px; font-weight:600; color:#16a34a; margin-top:3px; }
 
@keyframes pulse-border {
    0%   { border-color: #fca5a5; box-shadow: 0 0 0 0 rgba(239,68,68,0.25); }
    50%  { border-color: #ef4444; box-shadow: 0 0 0 6px rgba(239,68,68,0); }
    100% { border-color: #fca5a5; box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
.risk-high-banner {
    background: #fef2f2; border: 2px solid #fca5a5; border-radius: 12px; padding: 18px 20px;
    animation: pulse-border 2s ease-in-out infinite;
}
.risk-high-banner .rh-label { font-size:10px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#b91c1c; opacity:0.8; }
.risk-high-banner .rh-title { font-size:22px; font-weight:700; color:#dc2626; margin: 4px 0; }
.risk-high-banner .rh-sub   { font-size:12px; color:#7f1d1d; line-height:1.5; margin-top:6px; }
.risk-high-banner .rh-checklist { margin-top: 12px; border-top: 1px solid #fecaca; padding-top: 10px; }
.risk-high-banner .rh-check { font-size: 12px; color: #991b1b; margin: 4px 0; display: flex; gap: 8px; align-items: flex-start; }
 
.conf-gauge-wrap { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; }
.conf-gauge-title { font-size: 10px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 12px; }
.conf-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.conf-row-label { font-size: 12px; font-weight: 500; width: 62px; }
.conf-bar-outer { flex: 1; background: #e2e8f0; border-radius: 3px; height: 8px; overflow: hidden; }
.conf-bar-high   { height: 8px; border-radius: 3px; background: #ef4444; }
.conf-bar-medium { height: 8px; border-radius: 3px; background: #f59e0b; }
.conf-bar-low    { height: 8px; border-radius: 3px; background: #22c55e; }
.conf-pct { font-size: 12px; font-family: 'DM Mono', monospace; color: #64748b; width: 38px; text-align: right; }
.conf-active-label { font-size: 11px; color: #64748b; margin-top: 8px; padding-top: 8px; border-top: 1px solid #e2e8f0; }
 
.explain-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 14px; margin-bottom: 6px;
    background: #f8fafc; border-radius: 8px;
    border-left: 3px solid #3b82f6;
    font-size: 13px; color: #334155; line-height: 1.55;
}
.explain-item.warn   { border-left-color: #f59e0b; background: #fffdf5; }
.explain-item.good   { border-left-color: #22c55e; background: #f9fef9; }
.explain-item.alert  { border-left-color: #ef4444; background: #fff8f8; }
.explain-item.model  { border-left-color: #8b5cf6; background: #faf8ff; }
.explain-item.info   { border-left-color: #3b82f6; }
 
.cmp-row {
    display:flex; align-items:center; justify-content:space-between;
    padding: 10px 14px; border-radius: 8px; margin-bottom: 6px; font-size: 13px;
}
.cmp-match   { background:#f0fdf4; border: 1px solid #bbf7d0; }
.cmp-differ  { background:#fff7ed; border: 1px solid #fed7aa; }
.cmp-label   { color: #64748b; font-size:10px; font-weight:600; text-transform:uppercase; }
.cmp-val-ml  { font-weight:600; color:#1e293b; }
.cmp-val-rb  { color:#64748b; font-family:'DM Mono',monospace; font-size:12px; }
 
.model-differs { background:#fff7ed; border:1.5px solid #fdba74; border-radius:10px; padding:12px 16px; margin:14px 0; font-size:13px; color:#7c2d12; }
.model-agrees  { background:#f0fdf4; border:1.5px solid #86efac; border-radius:10px; padding:12px 16px; margin:14px 0; font-size:13px; color:#14532d; }
 
.scenario-label { font-size: 11px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.07em; margin: 14px 0 6px 0; }
.disclaimer { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; font-size: 11px; color: #94a3b8; line-height: 1.6; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# RENDERERS
# ──────────────────────────────────────────────────────────────────────────────
def render_score_card(label, score, color):
    pct = min(int(score), 100)
    st.markdown(f"""
    <div class="score-card">
        <div class="label">{label}</div>
        <div class="value">{score:.0f}<span style="font-size:13px;color:#94a3b8;font-weight:400">/100</span></div>
        <div class="bar-outer"><div class="bar-inner" style="width:{pct}%;background:{color}"></div></div>
    </div>""", unsafe_allow_html=True)
 
 
def render_risk_badge(risk, proba=None):
    if risk == "high":
        high_pct = proba.get("high", 0) if proba else 0
        st.markdown(f"""
        <div class="risk-high-banner">
            <div class="rh-label">Early Non-Response Risk</div>
            <div class="rh-title">🔴 HIGH RISK</div>
            <div class="rh-sub">
                Model confidence: <strong>{high_pct:.0%}</strong> — this patient profile
                requires close monitoring and proactive escalation planning.
            </div>
            <div class="rh-checklist">
                <div style="font-size:10px;font-weight:700;color:#b91c1c;
                            text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">
                    Clinical Escalation Checklist
                </div>
                <div class="rh-check"><span>☐</span><span>Follow-up within 2 weeks of treatment initiation</span></div>
                <div class="rh-check"><span>☐</span><span>Consider combined care (pharmacotherapy + psychotherapy)</span></div>
                <div class="rh-check"><span>☐</span><span>Address sleep disturbance directly (consider CBT-I)</span></div>
                <div class="rh-check"><span>☐</span><span>If prior treatment failed, consult for switch or augmentation</span></div>
                <div class="rh-check"><span>☐</span><span>Assess safety and suicidality at each contact</span></div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        css   = "risk-medium" if risk == "medium" else "risk-low"
        icon  = "🟡" if risk == "medium" else "🟢"
        label = "Medium Risk" if risk == "medium" else "Low Risk"
        st.markdown(f"""
        <div class="{css}">
            <div class="risk-label">Early Non-Response Risk</div>
            <div class="risk-value">{icon} {label}</div>
        </div>""", unsafe_allow_html=True)
 
 
def render_confidence_gauge(proba):
    """3-bar horizontal gauge for all risk classes with active class highlight."""
    if not proba: return
    high_p = proba.get("high",   0)
    med_p  = proba.get("medium", 0)
    low_p  = proba.get("low",    0)
    dominant   = max(proba, key=proba.get)
    dom_labels = {"high":"HIGH","medium":"MEDIUM","low":"LOW"}
    dom_colors = {"high":"#dc2626","medium":"#d97706","low":"#16a34a"}
    dom_label  = dom_labels[dominant]
    dom_color  = dom_colors[dominant]
 
    st.markdown(f"""
    <div class="conf-gauge-wrap">
        <div class="conf-gauge-title">Model Confidence — Risk Classification</div>
        <div class="conf-row">
            <div class="conf-row-label" style="color:#dc2626">🔴 High</div>
            <div class="conf-bar-outer"><div class="conf-bar-high" style="width:{high_p*100:.0f}%"></div></div>
            <div class="conf-pct">{high_p:.0%}</div>
        </div>
        <div class="conf-row">
            <div class="conf-row-label" style="color:#d97706">🟡 Medium</div>
            <div class="conf-bar-outer"><div class="conf-bar-medium" style="width:{med_p*100:.0f}%"></div></div>
            <div class="conf-pct">{med_p:.0%}</div>
        </div>
        <div class="conf-row">
            <div class="conf-row-label" style="color:#16a34a">🟢 Low</div>
            <div class="conf-bar-outer"><div class="conf-bar-low" style="width:{low_p*100:.0f}%"></div></div>
            <div class="conf-pct">{low_p:.0%}</div>
        </div>
        <div class="conf-active-label">
            Predicted: <strong style="color:{dom_color}">{dom_label}</strong>
            &nbsp;·&nbsp; Confidence: <strong style="color:{dom_color}">{proba.get(dominant,0):.0%}</strong>
        </div>
    </div>""", unsafe_allow_html=True)
 
 
def render_comparison_row(field, ml_val, rb_val):
    match = (ml_val.strip() == rb_val.strip())
    css   = "cmp-match" if match else "cmp-differ"
    icon  = "✓" if match else "≠"
    st.markdown(f"""
    <div class="cmp-row {css}">
        <div>
            <div class="cmp-label">{field}</div>
            <div class="cmp-val-ml">🤖 {ml_val}</div>
        </div>
        <div style="text-align:right">
            <div class="cmp-label">Rule-Based</div>
            <div class="cmp-val-rb">{icon} {rb_val}</div>
        </div>
    </div>""", unsafe_allow_html=True)
 
 
def render_explanation(item):
    tone_icons = {"alert":"⚠️","warn":"📌","good":"✅","info":"ℹ️"}
    icon       = "🤖" if item.get("source") == "model" else tone_icons.get(item["tone"],"ℹ️")
    css_class  = "model" if item.get("source") == "model" else item["tone"]
    st.markdown(f"""
    <div class="explain-item {css_class}">
        <span style="flex-shrink:0">{icon}</span>
        <span>{item['text']}</span>
    </div>""", unsafe_allow_html=True)
