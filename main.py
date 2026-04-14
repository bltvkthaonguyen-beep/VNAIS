import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app.core.ML import ml_predict
from app.core.ML import load_models
from app.core.rules import rule_based_predict
from app.core.EXPLANATION import build_explanation
from app.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)
from app.configs import DEMO_SCENARIOS
from app.UI import *
# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gene2Care-AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — DEMO SCENARIOS + MANUAL INPUTS
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🧠 Gene2Care-AI")
st.sidebar.markdown('<div class="scenario-label">⚡ Demo Scenarios</div>',
                    unsafe_allow_html=True)
 
for scenario_name, scenario_data in DEMO_SCENARIOS.items():
    if st.sidebar.button(scenario_name, use_container_width=True,
                          help=scenario_data["desc"]):
        for key, val in scenario_data.items():
            if key not in ("desc",):
                st.session_state[f"inp_{key}"] = val
        st.session_state["run_demo"] = True
        st.rerun()
 
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Manual Input")
st.sidebar.caption("Adjust sliders or select a demo above.")
 
def ss(key, default):
    """Read from session state (set by demo buttons) or use default."""
    return st.session_state.get(f"inp_{key}", default)
 
age    = st.sidebar.slider("Age", 16, 40, ss("age", 22))
gender = st.sidebar.selectbox("Gender", ["Female","Male","Other"],
             index=["Female","Male","Other"].index(ss("gender","Female")))
phq9   = st.sidebar.slider("PHQ-9 (Depression)", 10, 27, ss("phq9", 16),
             help="10–27 = moderate to severe")
gad7   = st.sidebar.slider("GAD-7 (Anxiety)", 0, 21, ss("gad7", 10))
sleep_quality    = st.sidebar.slider("Sleep Quality", 1, 5, ss("sleep_quality", 3),
                       help="1 = Very good  ·  5 = Very poor")
symptom_duration = st.sidebar.slider("Symptom Duration (months)", 2, 72,
                       ss("symptom_duration_months", 6))
recurrence = st.sidebar.slider("Recurrence Count", 0, 5, ss("recurrence_count", 0))
 
ptx_opts = ["none","medication_only","psychotherapy_only","both","failed_or_discontinued"]
prior_tx = st.sidebar.selectbox("Prior Treatment", ptx_opts,
               index=ptx_opts.index(ss("prior_treatment_status","none")))
fi_opts  = ["mild","moderate","severe"]
func_impairment = st.sidebar.selectbox("Functional Impairment", fi_opts,
                      index=fi_opts.index(ss("functional_impairment","mild")))
ta_opts  = ["low","medium","high"]
therapy_access = st.sidebar.selectbox("Therapy Access", ta_opts,
                    index=ta_opts.index(ss("therapy_access","high")))
 
st.sidebar.markdown("---")
run_btn = st.sidebar.button("⚡ Run Assessment", use_container_width=True, type="primary")
run_btn = run_btn or st.session_state.pop("run_demo", False)
 
# ──────────────────────────────────────────────────────────────────────────────
# COLLECT INPUTS
# ──────────────────────────────────────────────────────────────────────────────
inputs = {
    "age": age, "gender": gender, "phq9": phq9, "gad7": gad7,
    "sleep_quality": sleep_quality,
    "symptom_duration_months": symptom_duration,
    "recurrence_count": recurrence,
    "prior_treatment_status": prior_tx,
    "functional_impairment": func_impairment,
    "therapy_access": therapy_access,
}
 
# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <span style="font-size:34px">🧠</span>
  <div>
    <h1>Gene2Care-AI</h1>
    <p>Clinical Decision Support — Initial Treatment-Response Stratification</p>
    <span class="badge">Synthetic prototype · Research stage only · Not for clinical use</span>
  </div>
</div>
""", unsafe_allow_html=True)
 
# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────────
if not run_btn:
    cl, cr = st.columns([2, 1])
    with cl:
        st.markdown("""
        **This prototype demonstrates:**
        - Multi-output ML model predicting 3 suitability scores + non-response risk
        - DSM-inspired rule-based baseline for side-by-side comparison
        - Explanations grounded in **both model score patterns and clinical logic**
        - **Confidence / probability gauge** for all 3 risk classes
        - **HIGH RISK pulsing alert** with clinical escalation checklist
        """)
        st.info("👈  Click a **demo scenario** in the sidebar or adjust sliders, "
                "then click **Run Assessment**.")
    with cr:
        st.markdown("**Available demo scenarios:**")
        for name, data in DEMO_SCENARIOS.items():
            st.markdown(f"**{name}** — _{data['desc']}_")
else:
    rb_result = rule_based_predict(inputs)
    ml_result = ml_predict(inputs, models)
    using_fallback = (ml_result is None)
    if using_fallback:
        ml_result = {**rb_result,
                     "scores": {t: 50.0 for t in TARGET_REG},
                     "risk_proba": None}
 
    tab_ml, tab_compare, tab_explain = st.tabs([
        "🤖 ML Prediction", "⚖️ Baseline Comparison", "🔍 Explanation"
    ])
 
    # ════════════════════════════════════════════════════════
    # TAB 1 — ML PREDICTION
    # ════════════════════════════════════════════════════════
    model_type = st.selectbox(
    "🔧 Model Selection",
    ["Random Forest", "Gradient Boosting"],
    help="Compare predictions between models"
    )
    with tab_ml:
        if using_fallback:
            st.error("**gene2care_models.joblib not found** — "
                     "place it in the same folder as app.py.")
 
        st.markdown('<div class="section-header">Suitability Scores</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            render_score_card("Medication First",
                ml_result["scores"]["medication_first_suitability"],
                SCORE_COLORS["medication_first_suitability"])
        with c2:
            render_score_card("Psychotherapy First",
                ml_result["scores"]["psychotherapy_first_suitability"],
                SCORE_COLORS["psychotherapy_first_suitability"])
        with c3:
            render_score_card("Combined Care",
                ml_result["scores"]["combined_care_suitability"],
                SCORE_COLORS["combined_care_suitability"])
 
        st.markdown('<div class="section-header">Risk Assessment &amp; Confidence</div>',
                    unsafe_allow_html=True)
        col_risk, col_conf = st.columns([1, 1])
        with col_risk:
            render_risk_badge(ml_result["risk"], ml_result.get("risk_proba"))
        with col_conf:
            if not using_fallback and ml_result.get("risk_proba"):
                render_confidence_gauge(ml_result["risk_proba"])
            else:
                st.caption("Confidence gauge requires ML model.")
 
        st.markdown('<div class="section-header">Treatment Recommendations</div>',
                    unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"**💊 Medication**\n\n{ml_result['medication']}")
        with r2:
            st.markdown(f"**🗣️ Therapy**\n\n{ml_result['therapy']}")
        with r3:
            st.markdown(f"**🤝 Combined Care**\n\n{ml_result['combined_care']}")
 
    # ════════════════════════════════════════════════════════
    # TAB 2 — BASELINE COMPARISON
    # ════════════════════════════════════════════════════════
    with tab_compare:
        if using_fallback:
            st.warning("ML model unavailable — comparison not possible.")
        else:
            fields = {
                "Medication":    (ml_result["medication"],    rb_result["medication"]),
                "Therapy":       (ml_result["therapy"],       rb_result["therapy"]),
                "Combined Care": (ml_result["combined_care"], rb_result["combined_care"]),
                "Risk Level":    (ml_result["risk"],          rb_result["risk"]),
            }
            n_differ = sum(1 for mv, rv in fields.values()
                           if mv.strip() != rv.strip())
 
            if n_differ == 0:
                st.markdown("""<div class="model-agrees">
                    ✅ <strong>Model agrees with rule-based baseline</strong> on all 4 outputs.
                </div>""", unsafe_allow_html=True)
            else:
                plural = "output" if n_differ == 1 else "outputs"
                st.markdown(f"""<div class="model-differs">
                    ⚠️ <strong>Model differs from rule-based baseline</strong> on {n_differ}
                    {plural} — the ML model may be capturing interaction effects the rule
                    system misses. Review carefully.
                </div>""", unsafe_allow_html=True)
 
            st.markdown('<div class="section-header">Output Comparison</div>',
                        unsafe_allow_html=True)
            for field, (mv, rv) in fields.items():
                render_comparison_row(field, mv, rv)
 
            st.markdown('<div class="section-header">Suitability Score Table</div>',
                        unsafe_allow_html=True)
            scores_df = pd.DataFrame({
                "Treatment Path":  [SCORE_LABELS[t] for t in TARGET_REG],
                "ML Score (/100)": [f"{ml_result['scores'][t]:.0f}" for t in TARGET_REG],
            })
            st.dataframe(scores_df.set_index("Treatment Path"), use_container_width=True)
            st.caption("Rule-based baseline does not output numeric suitability scores.")
 
    # ════════════════════════════════════════════════════════
    # TAB 3 — EXPLANATION
    # ════════════════════════════════════════════════════════
    with tab_explain:
        explanation_ml = ml_result if not using_fallback else None
        all_items = build_explanation(inputs, explanation_ml)
 
        clinical_items = [i for i in all_items if i.get("source") == "clinical"]
        model_items    = [i for i in all_items if i.get("source") == "model"]
 
        # Clinical section
        st.markdown('<div class="section-header">Clinical Logic (DSM-Inspired)</div>',
                    unsafe_allow_html=True)
        st.caption("Grounded in validated clinical criteria and established treatment guidelines.")
        for item in clinical_items:
            render_explanation(item)
 
        # Model pattern section
        if model_items and not using_fallback:
            st.markdown('<div class="section-header">Model Pattern Explanations 🤖</div>',
                        unsafe_allow_html=True)
            st.caption("Explains WHY the model produced this specific score/risk pattern — "
                       "referencing learned relationships from training data. "
                       "Purple border = model-derived insight.")
            for item in model_items:
                render_explanation(item)
 
        # Input summary
        st.markdown('<div class="section-header">Input Summary</div>',
                    unsafe_allow_html=True)
        summary = pd.DataFrame([{
            "PHQ-9": phq9, "GAD-7": gad7, "Sleep": f"{sleep_quality}/5",
            "Duration": f"{symptom_duration}mo", "Recurrence": recurrence,
            "Prior Tx": prior_tx, "Impairment": func_impairment, "Access": therapy_access,
        }]).T.rename(columns={0: "Value"})
        st.dataframe(summary, use_container_width=True)
 
# ──────────────────────────────────────────────────────────────────────────────
# DISCLAIMER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Research Prototype Only.</strong> This application uses a synthetic dataset
    and has not been clinically validated. All outputs are illustrative and must not be used
    to inform real clinical decisions. Gene2Care-AI is a demonstration of ML methodology only.
</div>
""", unsafe_allow_html=True)
